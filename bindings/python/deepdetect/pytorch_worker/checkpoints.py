from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .sdk import WorkerDependencyError


@dataclass(frozen=True)
class TrainingCheckpointSelection:
    model: Path | None
    solver: Path | None
    source: str
    resume_from: str | None = None
    iteration: int | None = None


def resolve_training_checkpoint(
    mllib: Mapping[str, Any],
    repository: Path | None,
) -> TrainingCheckpointSelection:
    if bool(mllib.get("resume")):
        mode = str(mllib.get("resume_from") or "latest")
        return resolve_resume_checkpoint(repository, mode)

    raw = mllib.get("weights") or mllib.get("checkpoint")
    if raw:
        return TrainingCheckpointSelection(
            model=Path(str(raw)).expanduser().resolve(),
            solver=None,
            source="weights",
        )
    return TrainingCheckpointSelection(model=None, solver=None, source="none")


def resolve_resume_checkpoint(
    repository: Path | None,
    resume_from: str,
) -> TrainingCheckpointSelection:
    if repository is None:
        raise WorkerDependencyError("resuming training requires a model repository")
    repository = repository.expanduser().resolve()
    if not repository.is_dir():
        raise WorkerDependencyError(
            f"resume repository not found or not a directory: {repository}"
        )
    if resume_from == "latest":
        return _latest_complete_checkpoint(repository)
    if resume_from == "best":
        iteration = _best_model_iteration(repository)
        return _numbered_checkpoint(repository, iteration, resume_from="best")
    raise WorkerDependencyError("resume_from must be one of: latest, best")


def _latest_complete_checkpoint(repository: Path) -> TrainingCheckpointSelection:
    complete: list[int] = []
    for path in repository.glob("checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d+)\.pt", path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if (repository / f"solver-{iteration}.pt").is_file():
            complete.append(iteration)
    if complete:
        return _numbered_checkpoint(
            repository,
            max(complete),
            resume_from="latest",
        )

    model = repository / "checkpoint-latest.pt"
    solver = repository / "solver-latest.pt"
    if model.is_file() and solver.is_file():
        return TrainingCheckpointSelection(
            model=model,
            solver=solver,
            source="resume",
            resume_from="latest",
        )
    raise WorkerDependencyError(
        "resume repository has no complete model/solver checkpoint pair: "
        f"{repository}"
    )


def _numbered_checkpoint(
    repository: Path,
    iteration: int,
    *,
    resume_from: str,
) -> TrainingCheckpointSelection:
    model = repository / f"checkpoint-{iteration}.pt"
    solver = repository / f"solver-{iteration}.pt"
    missing = [str(path) for path in (model, solver) if not path.is_file()]
    if missing:
        raise WorkerDependencyError(
            f"{resume_from} resume checkpoint is incomplete; missing: "
            + ", ".join(missing)
        )
    return TrainingCheckpointSelection(
        model=model,
        solver=solver,
        source="resume",
        resume_from=resume_from,
        iteration=iteration,
    )


def _best_model_iteration(repository: Path) -> int:
    marker = repository / "best_model.txt"
    if not marker.is_file():
        raise WorkerDependencyError(f"best model marker not found: {marker}")
    for line in marker.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "iteration":
            raw_iteration = value.strip()
            if not re.fullmatch(r"\d+", raw_iteration):
                break
            return int(raw_iteration)
    raise WorkerDependencyError(
        f"best model marker has no valid iteration entry: {marker}"
    )


class CheckpointManager:
    """Opt-in retention for adapter-owned, complete checkpoint pairs."""

    def __init__(self, repository, *, snapshot=0, metric="map", maximize=True,
                 resume=False, log=None, onnx_export=None, inference=None):
        import json
        if isinstance(snapshot, bool) or not isinstance(snapshot, int) or snapshot < 0:
            raise ValueError("snapshot must be a non-negative integer")
        self.repository = Path(repository).resolve() if repository else None
        self.snapshot, self.metric, self.maximize = snapshot, metric, maximize
        self.log = log or (lambda message: None)
        self.state = {"version": 1, "metric": metric, "maximize": maximize,
                      "files": {}, "best": {}, "periodic": [], "latest": None}
        from ..onnx_export import BestOnnxExporter
        self.onnx_exporter = BestOnnxExporter(self.repository or ".", onnx_export,
            inference=inference, resume=resume, log=self.log, backend="pytorch")
        self.cleanup = True
        self.saved = set()
        if resume and self.repository:
            path = self.repository / "checkpoint-state.json"
            if path.exists():
                try:
                    state = json.loads(path.read_text())
                    assert state["version"] == 1 and state["metric"] == metric
                    assert state["maximize"] == maximize
                    assert isinstance(state["files"], dict)
                    assert isinstance(state["best"], dict)
                    assert isinstance(state["periodic"], list)
                    for number, names in state["files"].items():
                        assert str(int(number)) == number and int(number) > 0
                        assert isinstance(names, list) and len(names) == 2
                        for name in names:
                            assert isinstance(name, str) and not Path(name).is_absolute()
                            assert (self.repository / name).resolve().is_relative_to(self.repository)
                    for marker, best in state["best"].items():
                        assert re.fullmatch(r"best_model(?:_test_\d+)?\.txt", marker)
                        assert len(best) == 2 and isinstance(best[0], int)
                        assert isinstance(best[1], (float, int))
                    assert all(isinstance(i, int) for i in state["periodic"])
                    assert state["latest"] is None or isinstance(state["latest"], int)
                    self.state = state
                except Exception as error:
                    self.cleanup = False
                    self.log(f"Checkpoint cleanup disabled: {error}")
            else:
                for marker in self.repository.glob("best_model*.txt"):
                    try:
                        fields = dict(line.split(":", 1) for line in marker.read_text().splitlines())
                        self.state["best"][marker.name] = [int(fields["iteration"]), float(fields[metric])]
                    except (ValueError, KeyError):
                        pass

    def step(self, iteration, save, *, metrics=None, final=False):
        """Save callback takes iteration and returns model/solver paths or None."""
        import copy
        import json
        import math
        if not self.repository or iteration <= 0:
            return
        periodic = self.snapshot > 0 and iteration % self.snapshot == 0
        if metrics is None and not periodic and not final:
            return
        if iteration in self.saved:
            return
        paths = save(iteration)
        if paths is None:
            paths = [self.repository / f"checkpoint-{iteration}.pt",
                     self.repository / f"solver-{iteration}.pt"]
        paths = [Path(p).resolve() for p in paths]
        if len(paths) != 2 or not all(p.is_file() and p.is_relative_to(self.repository) for p in paths):
            self.log("Checkpoint hook did not supply a complete repository pair; skipping retention")
            return
        state = copy.deepcopy(self.state)
        state["files"][str(iteration)] = [str(p.relative_to(self.repository)) for p in paths]
        state["latest"] = iteration
        if periodic and iteration not in state["periodic"]:
            state["periodic"].append(iteration)
        for scope, value in (metrics or {}).items():
            if not math.isfinite(value):
                continue
            marker = "best_model.txt" if scope == "aggregate" else f"best_model_test_{int(scope)}.txt"
            previous = state["best"].get(marker)
            if previous is None or (value >= previous[1] if self.maximize else value <= previous[1]):
                state["best"][marker] = [iteration, value]
                _atomic_checkpoint_text(self.repository / marker,
                                        f"iteration:{iteration}\n{self.metric}:{value}\n")
        _atomic_checkpoint_text(self.repository / "checkpoint-state.json", json.dumps(state))
        self.state = state
        if state["best"].get("best_model.txt", [None])[0] == iteration:
            self.onnx_exporter.notify(iteration, paths[:1])
        self.saved.add(iteration)
        retained = set(state["periodic"]) | {state["latest"]} | {v[0] for v in state["best"].values()}
        if self.cleanup:
            for number, names in list(state["files"].items()):
                if int(number) in retained:
                    continue
                try:
                    for name in names:
                        path = (self.repository / name).resolve()
                        if not path.is_relative_to(self.repository):
                            raise ValueError("checkpoint outside repository")
                        path.unlink(missing_ok=True)
                    del state["files"][number]
                except (OSError, ValueError) as error:
                    self.log(f"Checkpoint cleanup failed: {error}")
            _atomic_checkpoint_text(self.repository / "checkpoint-state.json", json.dumps(state))


    def close(self, *, cancel=False, cancellation=None):
        self.onnx_exporter.close(cancel=cancel, cancellation=cancellation)

def _atomic_checkpoint_text(path, value):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointMetricReporter:
    """Forward the reporter interface while collecting evaluation scalars."""
    def __init__(self, reporter):
        self.reporter = reporter
        self.values = {}

    def __getattr__(self, name):
        return getattr(self.reporter, name)

    def metric(self, name, value, *, iteration=None):
        self.reporter.metric(name, value, iteration=iteration)
        self.values[name] = value

    def scores(self, count):
        values = {str(i): self.values[f"map_test{i}"] for i in range(count)
                  if f"map_test{i}" in self.values}
        if len(values) != count or not values:
            return {}
        return {**values, "aggregate": sum(values.values()) / len(values)}
