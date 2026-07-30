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
