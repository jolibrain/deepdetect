import json
import sys
from deepdetect.onnx_export import BestOnnxExporter


def test_disabled_has_no_side_effects(tmp_path):
    exporter = BestOnnxExporter(tmp_path)
    exporter.notify(1, [tmp_path / "missing.pt"])
    exporter.close()
    assert not (tmp_path / "onnx").exists()


def test_first_failure_latches_across_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint-1.pt"
    checkpoint.write_bytes(b"checkpoint")
    warnings = []
    config = {"enabled": True, "python": "/nonexistent/python"}
    exporter = BestOnnxExporter(tmp_path, config, log=warnings.append)
    exporter.notify(1, [checkpoint])
    exporter.close()
    assert len(warnings) == 1
    assert exporter.state["status"] == "disabled"
    resumed = BestOnnxExporter(tmp_path, config, resume=True, log=lambda _: None)
    resumed.notify(2, [checkpoint])
    assert not resumed.enabled
    assert resumed.thread is None
    assert (
        json.loads((tmp_path / "onnx/state.json").read_text())["status"] == "disabled"
    )


def test_new_run_resets_latch(tmp_path):
    exporter = BestOnnxExporter(tmp_path, {"enabled": True}, log=lambda _: None)
    exporter.notify(1, [tmp_path / "missing"])
    assert not exporter.enabled
    fresh = BestOnnxExporter(tmp_path, {"enabled": True})
    assert fresh.enabled
    fresh.close()


import os
from pathlib import Path
import subprocess
import threading
import time
import pytest
from deepdetect import onnx_export


def checkpoint(root, iteration):
    path = root / f"checkpoint-{iteration}.pt"
    path.write_bytes(str(iteration).encode())
    return path


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


class FakeProcess:
    def __init__(self, args, gate, started, **kwargs):
        self.request = json.loads(Path(args[-1]).read_text())
        self.gate = gate
        self.returncode = None
        self.killed = False
        started.append(self)

    def wait(self, timeout=None):
        if not self.gate.wait(timeout):
            raise subprocess.TimeoutExpired("fake exporter", timeout)
        if not self.killed:
            stage = Path(self.request["checkpoints"][0]).parent
            # Snapshot must still exist even if the original was deleted.
            assert Path(self.request["checkpoints"][0]).read_bytes()
            bundle = stage / "bundle"
            bundle.mkdir(exist_ok=True)
            (bundle / "model.onnx").write_bytes(b"fake")
            (bundle / "metadata.json").write_text("{}")
            self.returncode = 0
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.gate.set()


def fake_processes(monkeypatch):
    gate, started = threading.Event(), []
    monkeypatch.setattr(
        onnx_export.subprocess,
        "Popen",
        lambda args, **kwargs: FakeProcess(args, gate, started, **kwargs),
    )
    return gate, started


def test_latest_pending_and_snapshot_survive_retention(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    exporter = BestOnnxExporter(tmp_path, {"enabled": True})
    first = checkpoint(tmp_path, 1)
    exporter.notify(1, [first])
    wait_for(lambda: len(started) == 1)
    first.unlink()
    exporter.notify(1, [first])  # Duplicate does not try to copy deleted original.
    exporter.notify(2, [checkpoint(tmp_path, 2)])
    exporter.notify(3, [checkpoint(tmp_path, 3)])
    gate.set()
    exporter.close()
    assert [p.request["iteration"] for p in started] == [1, 3]
    assert exporter.state["exported_iteration"] == 3
    assert not exporter.state["stale"]
    assert len(list((tmp_path / "onnx").glob("export-*"))) == 1
    assert not list((tmp_path / "onnx").glob(".candidate-*"))


def test_later_failure_preserves_success_and_tracks_best(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    gate.set()
    messages = []
    exporter = BestOnnxExporter(tmp_path, {"enabled": True}, log=messages.append)
    exporter.notify(1, [checkpoint(tmp_path, 1)])
    wait_for(lambda: exporter.state["exported_iteration"] == 1)
    bundle = exporter.root / exporter.state["bundle"]
    exporter.notify(2, [tmp_path / "missing"])
    exporter.notify(3, [tmp_path / "missing"])
    exporter.close()
    assert len(messages) == 1
    assert len(started) == 1
    assert bundle.exists()
    assert exporter.state["exported_iteration"] == 1
    assert exporter.state["best_iteration"] == 3
    assert exporter.state["stale"]


def test_timeout_disables_and_discards_pending(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    exporter = BestOnnxExporter(
        tmp_path, {"enabled": True, "timeout_seconds": 0.1}, log=lambda _: None
    )
    exporter.notify(1, [checkpoint(tmp_path, 1)])
    wait_for(lambda: started)
    exporter.notify(2, [checkpoint(tmp_path, 2)])
    exporter.close()
    assert exporter.state["status"] == "disabled"
    assert len(started) == 1
    assert started[0].killed
    assert not list(exporter.root.glob(".candidate-*"))


def test_cancel_kills_without_latching_and_resume_requeues(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    exporter = BestOnnxExporter(tmp_path, {"enabled": True})
    exporter.notify(1, [checkpoint(tmp_path, 1)])
    wait_for(lambda: started)
    exporter.close(cancel=True)
    assert started[0].killed
    assert exporter.state["status"] == "cancelled"
    (tmp_path / "best_model.txt").write_text("iteration:1\nmap:0.5\n")
    resumed = BestOnnxExporter(tmp_path, {"enabled": True}, resume=True)
    resumed.close()
    assert resumed.state["exported_iteration"] == 1


def test_interrupted_attempt_latches(tmp_path):
    exporter = BestOnnxExporter(tmp_path, {"enabled": True})
    exporter.state["status"] = "running"
    exporter._persist()
    resumed = BestOnnxExporter(
        tmp_path, {"enabled": True}, resume=True, log=lambda _: None
    )
    assert not resumed.enabled
    assert resumed.state["status"] == "disabled"


@pytest.mark.parametrize(
    "config",
    [{"timeout_seconds": -1}, {"timeout_seconds": float("inf")}, {"opset_version": 0}],
)
def test_invalid_settings_nonfatal(tmp_path, config):
    exporter = BestOnnxExporter(
        tmp_path, {"enabled": True, **config}, log=lambda _: None
    )
    assert not exporter.enabled


def test_real_torchscript_export(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    model = torch.nn.Conv2d(3, 2, 1).eval()
    example = torch.randn(1, 3, 4, 4)
    path = tmp_path / "checkpoint-1.pt"
    torch.jit.trace(model, example).save(str(path))
    exporter = BestOnnxExporter(
        tmp_path, {"enabled": True}, inference={"width": 4, "height": 4}
    )
    exporter.notify(1, [path])
    exporter.close()
    assert exporter.state["status"] == "ready", exporter.state
    bundle = exporter.root / exporter.state["bundle"]
    onnx.checker.check_model(str(bundle / "model.onnx"))
    output = ReferenceEvaluator(str(bundle / "model.onnx")).run(
        None, {"input": example.numpy()}
    )[0]
    torch.testing.assert_close(torch.from_numpy(output), model(example))
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["iteration"] == 1
    assert metadata["inputs"][0]["shape"] == [1, 3, 4, 4]
    assert len(metadata["checkpoints"][0]["sha256"]) == 64


def test_real_worker_provider(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    provider = tmp_path / "sample_provider.py"
    provider.write_text("""import torch

def build(paths, config):
    model = torch.nn.Linear(2, 1)
    model.load_state_dict(torch.load(paths[0], weights_only=True))
    return model, (torch.zeros(1, 2),), {'input_names': ['input'], 'output_names': ['score']}
""")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    path = tmp_path / "checkpoint-1.pt"
    torch.save(torch.nn.Linear(2, 1).state_dict(), path)
    exporter = BestOnnxExporter(
        tmp_path,
        {"enabled": True, "provider": "sample_provider:build"},
        backend="pytorch",
    )
    exporter.notify(1, [path])
    exporter.close()
    assert exporter.state["status"] == "ready", exporter.state
    assert exporter.state["exported_iteration"] == 1


@pytest.fixture(scope="module")
def native_bridge(tmp_path_factory):
    import shutil

    compiler = shutil.which("g++")
    if not compiler or not Path("/usr/include/rapidjson/document.h").exists():
        pytest.skip("native bridge test requires a C++ compiler and RapidJSON headers")
    # Wheel CI copies this module outside the checkout, but keeps native
    # test sources available through the same override as other wheel tests.
    source_root = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
    root = (
        Path(source_root).resolve()
        if source_root
        else Path(__file__).resolve().parents[3]
    )
    build = tmp_path_factory.mktemp("native-onnx")
    source = build / "main.cc"
    source.write_text(r"""
#include "utils/onnx_export.h"
#include <fstream>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
int main(int argc, char **argv) {
  std::ifstream input(std::string(argv[1]) + "/start.json");
  std::string start((std::istreambuf_iterator<char>(input)), {});
  std::atomic<bool> cancelled{false};
  dd::OnnxExport exporter(argv[1], argv[2], start, 30,
      [&]() { return cancelled.load(); },
      [](const std::string &s) { std::cerr << s << std::endl; });
  exporter.notify(1, {std::string(argv[1]) + "/checkpoint-1.pt"});
  if (argc > 3) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cancelled.store(true);
  }
  exporter.finish();
}
""")
    executable = build / "bridge"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-I" + str(root / "src"),
            str(source),
            str(root / "src/utils/onnx_export.cc"),
            "-o",
            str(executable),
        ],
        check=True,
        env={**os.environ, "CCACHE_DISABLE": "1"},
    )
    return executable


def test_native_bridge_real_export(tmp_path, native_bridge):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    torch.jit.trace(torch.nn.Identity(), torch.zeros(1, 3, 4, 4)).save(
        str(tmp_path / "checkpoint-1.pt")
    )
    (tmp_path / "start.json").write_text(
        json.dumps(
            {
                "command": "start",
                "repository": str(tmp_path),
                "config": {"enabled": True},
                "inference": {"width": 4, "height": 4},
            }
        )
    )
    subprocess.run(
        [str(native_bridge), str(tmp_path), sys.executable], check=True, timeout=40
    )
    state = json.loads((tmp_path / "onnx/state.json").read_text())
    assert state["exported_iteration"] == 1, state
    assert state["status"] == "ready"


def test_native_bridge_launch_failure_nonfatal(tmp_path, native_bridge):
    (tmp_path / "start.json").write_text("{}")
    subprocess.run([str(native_bridge), str(tmp_path), "/missing/python"], check=True)
    assert (
        json.loads((tmp_path / "onnx/state.json").read_text())["status"] == "disabled"
    )


def test_native_bridge_cancel_while_draining(tmp_path, native_bridge, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    checkpoint(tmp_path, 1)
    (tmp_path / "slow_provider.py").write_text(
        "import time\ndef build(*args):\n time.sleep(60)\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    (tmp_path / "start.json").write_text(
        json.dumps(
            {
                "command": "start",
                "repository": str(tmp_path),
                "config": {"enabled": True, "provider": "slow_provider:build"},
            }
        )
    )
    subprocess.run(
        [str(native_bridge), str(tmp_path), sys.executable, "cancel"],
        check=True,
        timeout=10,
    )
    state = json.loads((tmp_path / "onnx/state.json").read_text())
    assert state["status"] == "cancelled", state
    assert not list((tmp_path / "onnx").glob(".candidate-*"))


def test_publication_failure_preserves_old_manifest(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    gate.set()
    exporter = BestOnnxExporter(tmp_path, {"enabled": True}, log=lambda _: None)
    exporter.notify(1, [checkpoint(tmp_path, 1)])
    wait_for(lambda: exporter.state["exported_iteration"] == 1)
    old = exporter.root / exporter.state["bundle"]
    replace = os.replace

    def fail_bundle(source, destination):
        if Path(source).name == "bundle":
            raise OSError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_bundle)
    exporter.notify(2, [checkpoint(tmp_path, 2)])
    exporter.close()
    assert old.exists()
    assert exporter.state["status"] == "disabled"
    assert exporter.state["exported_iteration"] == 1
    assert len(list(exporter.root.glob("export-*"))) == 1


def test_worker_manager_only_exports_aggregate_best(tmp_path, monkeypatch):
    from deepdetect.pytorch_worker.checkpoints import CheckpointManager

    gate, started = fake_processes(monkeypatch)
    gate.set()
    manager = CheckpointManager(tmp_path, onnx_export={"enabled": True})

    def save(iteration):
        path = checkpoint(tmp_path, iteration)
        solver = tmp_path / f"solver-{iteration}.pt"
        solver.write_bytes(b"solver")
        return path, solver

    manager.step(1, save, metrics={"aggregate": 0.5, "0": 0.5})
    wait_for(lambda: manager.onnx_exporter.state["exported_iteration"] == 1)
    manager.step(2, save, metrics={"aggregate": 0.4, "0": 0.8})
    manager.step(3, save, metrics={"aggregate": 0.7, "0": 0.7})
    manager.close()
    assert [process.request["iteration"] for process in started] == [1, 3]
    assert all(process.request["backend"] == "pytorch" for process in started)
    assert all(len(process.request["checkpoints"]) == 1 for process in started)


def test_cancel_during_python_drain(tmp_path, monkeypatch):
    gate, started = fake_processes(monkeypatch)
    exporter = BestOnnxExporter(tmp_path, {"enabled": True})
    exporter.notify(1, [checkpoint(tmp_path, 1)])
    wait_for(lambda: started)
    exporter.close(cancellation=lambda: True)
    assert started[0].killed
    assert exporter.state["status"] == "cancelled"


def test_native_bridge_failure_latch_survives_resume(tmp_path, native_bridge):
    (tmp_path / 'start.json').write_text('{}')
    subprocess.run([str(native_bridge), str(tmp_path), '/missing/python'], check=True)
    (tmp_path / 'start.json').write_text(json.dumps({'resume': True}))
    result = subprocess.run([str(native_bridge), str(tmp_path), '/missing/python'], check=True, capture_output=True)
    assert not result.stderr
    state = json.loads((tmp_path / 'onnx/state.json').read_text())
    assert state['status'] == 'disabled'
    assert state['best_iteration'] == 1


def test_native_bridge_child_exits_without_reply(tmp_path, native_bridge):
    (tmp_path / 'start.json').write_text('{}')
    subprocess.run([str(native_bridge), str(tmp_path), '/bin/true'], check=True)
    assert json.loads((tmp_path / 'onnx/state.json').read_text())['status'] == 'disabled'


def test_native_bridge_from_relocated_wheel_tests(tmp_path, native_bridge):
    import shutil

    source_root = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
    root = (
        Path(source_root).resolve()
        if source_root
        else Path(__file__).resolve().parents[3]
    )
    copied_tests = tmp_path / "build/python-wheel-test-envs/cpu/wheel-tests"
    copied_tests.mkdir(parents=True)
    copied_module = copied_tests / "test_onnx_export.py"
    shutil.copy2(__file__, copied_module)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--import-mode=importlib",
            "-q",
            str(copied_module) + "::test_native_bridge_launch_failure_nonfatal",
        ],
        cwd=copied_tests.parent,
        env={
            **os.environ,
            "DEEPDETECT_WHEEL_TEST_SOURCE_ROOT": str(root),
            # Keep testing the same package after changing cwd, even when the
            # source suite was launched with a relative PYTHONPATH.
            "PYTHONPATH": str(Path(onnx_export.__file__).resolve().parents[1]),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
