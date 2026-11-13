"""Shared, opt-in best-checkpoint ONNX export (no torch import in the parent)."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".state-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class BestOnnxExporter:
    """One active conversion and one replaceable pending snapshot per run.

    Call notify only after committing the aggregate best checkpoint. close()
    drains exports; close(cancel=True) kills conversion and discards pending work.
    Export errors never propagate into training. Instances are owned by one run.
    """

    def __init__(
        self,
        repository,
        config=None,
        *,
        inference=None,
        resume=False,
        log=None,
        backend="torch",
    ):
        self.root = Path(repository).resolve() / "onnx"
        self.config = config if isinstance(config, dict) else {}
        self.requested = self.config.get("enabled", False) is True
        self.enabled = self.requested
        self.inference = dict(inference or {})
        self.backend = backend
        self.log = log or (lambda message: warnings.warn(message, RuntimeWarning))
        self.condition = threading.Condition()
        self.pending = self.process = self.thread = None
        self.closing = self.cancelled = False
        self.state = {
            "status": "ready",
            "best_iteration": None,
            "exported_iteration": None,
            "stale": False,
        }
        if not self.requested:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            statefile = self.root / "state.json"
            if resume and statefile.exists():
                self.state = json.loads(statefile.read_text())
                if self.state["status"] == "disabled":
                    self._cleanup_orphans()
                    self.enabled = False
                    return
                if self.state["status"] == "running":
                    raise RuntimeError("previous export was interrupted")
            self._cleanup_orphans()
            timeout = self.config.get("timeout_seconds", 120)
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError(
                    "onnx_export.timeout_seconds must be positive and finite"
                )
            opset = self.config.get("opset_version", 17)
            if isinstance(opset, bool) or not isinstance(opset, int) or opset <= 0:
                raise ValueError("onnx_export.opset_version must be a positive integer")
            self.state["status"] = "ready"
            self.state.pop("active_iteration", None)
            self._persist()
            if resume:
                self._resume_best()
        except Exception as error:
            self._disable(str(error))

    def _cleanup_orphans(self):
        for path in self.root.iterdir():
            if path.name.startswith(".candidate-") or (
                path.name.startswith("export-")
                and path.name != self.state.get("bundle")
            ):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)

    def _resume_best(self):
        marker = self.root.parent / "best_model.txt"
        if not marker.exists():
            return
        fields = dict(line.split(":", 1) for line in marker.read_text().splitlines())
        iteration = int(fields["iteration"])
        if iteration == self.state.get("exported_iteration"):
            return
        extensions = (".pt",) if self.backend == "pytorch" else (".pt", ".ptw", ".npt")
        paths = [
            self.root.parent / f"checkpoint-{iteration}{ext}" for ext in extensions
        ]
        self.notify(iteration, [path for path in paths if path.exists()])

    def snapshot(self):
        with self.condition:
            return dict(self.state)

    def _persist(self):
        self.state["stale"] = self.state.get("exported_iteration") != self.state.get(
            "best_iteration"
        )
        atomic_json(self.root / "state.json", self.state)

    def _disable(self, reason):
        already_disabled = self.state.get("status") == "disabled"
        self.enabled = False
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
        self.state.update(status="disabled", error=reason)
        if self.pending:
            shutil.rmtree(self.pending[1], ignore_errors=True)
            self.pending = None
        try:
            self._persist()
        except Exception:
            reason += " (could not persist failure latch)"
        if not already_disabled:
            try:
                self.log(f"ONNX export disabled for this run: {reason}")
            except Exception:
                pass  # Logging must not fail training either.
        with self.condition:
            self.condition.notify_all()

    def notify(self, iteration, checkpoints):
        with self.condition:
            if not self.requested or self.closing:
                return
            # Keep provenance current even after the latch has tripped.
            self.state["best_iteration"] = iteration
            if not self.enabled:
                try:
                    self._persist()
                except Exception:
                    pass
                return
            if iteration in (
                self.state.get("active_iteration"),
                self.state.get("exported_iteration"),
            ) or (self.pending and self.pending[0] == iteration):
                return
            stage = None
            try:
                if not checkpoints:
                    raise ValueError("best checkpoint has no model components")
                stage = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.root))
                copies, identities = [], []
                for checkpoint in checkpoints:
                    source = Path(checkpoint)
                    destination = stage / source.name
                    if destination.exists():
                        raise ValueError("duplicate checkpoint component name")
                    shutil.copy2(source, destination)
                    with destination.open("rb") as stream:
                        hasher = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(block)
                        digest = hasher.hexdigest()
                    copies.append(str(destination))
                    identities.append({"name": source.name, "sha256": digest})
                atomic_json(
                    stage / "request.json",
                    {
                        "checkpoints": copies,
                        "checkpoint_identity": identities,
                        "inference": self.inference,
                        "config": self.config,
                        "backend": self.backend,
                        "iteration": iteration,
                    },
                )
                if self.pending:
                    shutil.rmtree(self.pending[1], ignore_errors=True)
                self.pending = (iteration, stage)
                self._persist()
                if self.thread is None:
                    self.thread = threading.Thread(target=self._run, daemon=True)
                    self.thread.start()
                self.condition.notify_all()
            except Exception as error:
                if stage:
                    shutil.rmtree(stage, ignore_errors=True)
                self._disable(str(error))

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.pending or self.closing or not self.enabled
                )
                if (
                    not self.enabled
                    or self.cancelled
                    or (self.closing and not self.pending)
                ):
                    return
                iteration, stage = self.pending
                self.pending = None
                self.state["active_iteration"] = iteration
            published = None
            try:
                with (stage / "stderr.log").open("w+b") as stderr:
                    with self.condition:
                        self.state["status"] = "running"
                        self._persist()
                        if self.cancelled or not self.enabled:
                            return
                        self.process = subprocess.Popen(
                            [
                                self.config.get("python") or sys.executable,
                                "-m",
                                "deepdetect.onnx_export",
                                str(stage / "request.json"),
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=stderr,
                            env={**os.environ, "DD_ONNX_PARENT_PID": str(os.getpid())},
                        )
                    self.process.wait(timeout=self.config.get("timeout_seconds", 120))
                    with self.condition:
                        if self.cancelled or not self.enabled:
                            return
                    if self.process.returncode:
                        stderr.seek(max(0, stderr.seek(0, 2) - 4000))
                        raise RuntimeError(
                            stderr.read().decode(errors="replace")
                            or f"export child exited {self.process.returncode}"
                        )
                with self.condition:
                    if self.cancelled or not self.enabled:
                        return
                    bundle = stage / "bundle"
                    if (
                        not (bundle / "model.onnx").is_file()
                        or not (bundle / "metadata.json").is_file()
                    ):
                        raise RuntimeError(
                            "export child did not produce a complete bundle"
                        )
                    published = self.root / (
                        "export-" + stage.name.removeprefix(".candidate-")
                    )
                    os.replace(bundle, published)
                    old = self.state.get("bundle")
                    previous = dict(self.state)
                    self.state.update(
                        status="ready",
                        exported_iteration=iteration,
                        bundle=published.name,
                    )
                    self.state.pop("active_iteration", None)
                    try:
                        self._persist()
                    except Exception:
                        self.state = previous
                        raise
                    published = None  # Manifest now owns this directory.
                    if old and old.startswith("export-") and Path(old).name == old:
                        shutil.rmtree(self.root / old, ignore_errors=True)
            except Exception as error:
                if self.process and self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                with self.condition:
                    if not self.cancelled:
                        self._disable(str(error))
                return
            finally:
                with self.condition:
                    self.process = None
                shutil.rmtree(stage, ignore_errors=True)
                if published:
                    shutil.rmtree(published, ignore_errors=True)

    def close(self, *, cancel=False, cancellation=None):
        with self.condition:
            self.closing = True
            self.cancelled = self.cancelled or cancel
            if self.cancelled:
                if self.pending:
                    shutil.rmtree(self.pending[1], ignore_errors=True)
                    self.pending = None
                if self.process and self.process.poll() is None:
                    self.process.kill()
                if self.enabled:
                    self.state["status"] = "cancelled"
                    try:
                        self._persist()
                    except Exception as error:
                        self._disable(str(error))
            self.condition.notify_all()
        if self.thread:
            while self.thread.is_alive():
                self.thread.join(0.05)
                if cancellation is not None and cancellation():
                    self.close(cancel=True)
                    break


def convert(request):
    import torch
    import onnx

    config = request["config"]
    provider = config.get("provider")
    if provider:
        module, function = provider.split(":", 1)
        model, inputs, names = getattr(importlib.import_module(module), function)(
            request["checkpoints"], request["inference"]
        )
    else:
        if request.get("backend") == "pytorch":
            raise ValueError("worker ONNX export requires an explicit provider")
        paths = request["checkpoints"]
        if len(paths) != 1 or not paths[0].endswith(".pt"):
            raise ValueError(
                "requires a self-contained TorchScript checkpoint or an explicit provider"
            )
        model = torch.jit.load(paths[0], map_location="cpu")
        settings = request["inference"]
        inputs = (
            torch.zeros(
                1, settings.get("channels", 3), settings["height"], settings["width"]
            ),
        )
        names = {"input_names": ["input"]}
    if not isinstance(inputs, tuple) or not all(
        isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in inputs
    ):
        raise ValueError("provider must return a tuple of CPU tensor inputs")
    if not isinstance(names, dict) or set(names) - {"input_names", "output_names"}:
        raise ValueError("provider names may contain only input_names and output_names")
    model = model.cpu().eval()
    bundle = Path(request["checkpoints"][0]).parent / "bundle"
    bundle.mkdir()
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(bundle / "model.onnx"),
            opset_version=config.get("opset_version", 17),
            dynamo=False,
            **names,
        )
    onnx.checker.check_model(str(bundle / "model.onnx"))
    graph = onnx.load_model(str(bundle / "model.onnx"), load_external_data=False).graph
    atomic_json(
        bundle / "metadata.json",
        {
            "iteration": request["iteration"],
            "checkpoints": request["checkpoint_identity"],
            "opset_version": config.get("opset_version", 17),
            "names": {
                "input_names": [value.name for value in graph.input],
                "output_names": [value.name for value in graph.output],
            },
            "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs],
            "inference": request["inference"],
        },
    )


def serve():
    """Private JSONL bridge. Acknowledgement means checkpoint copies are safe."""
    import select

    exporter = None
    draining = None
    completed = False
    try:
        while True:
            if draining and not draining.is_alive():
                print(json.dumps(exporter.snapshot()), flush=True)
                completed = True
                return
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                return
            message = json.loads(line)
            command = message.pop("command")
            if command == "start":
                exporter = BestOnnxExporter(**message)
            elif command == "notify":
                exporter.notify(**message)
            elif command == "close":
                draining = threading.Thread(target=exporter.close, kwargs=message)
                draining.start()
                continue
            else:
                raise ValueError("unknown export command")
            print(json.dumps(exporter.snapshot()), flush=True)
    finally:
        if exporter and not completed:
            exporter.close(cancel=True)
        if draining:
            draining.join()


if __name__ == "__main__":
    # Linux conversion children die if their supervisor is killed, including
    # forced native training cancellation. Check for death before prctl too.
    if "DD_ONNX_PARENT_PID" in os.environ and sys.platform == "linux":
        import ctypes
        import signal

        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != int(os.environ["DD_ONNX_PARENT_PID"]):
            sys.exit(1)
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        convert(json.loads(Path(sys.argv[1]).read_text()))
