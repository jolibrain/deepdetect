from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import deepdetect


def configure_gpu_compatibility(
    build_info: dict[str, Any], *, requested: bool
) -> None:
    if not requested or os.environ.get("TORCH_CUDNN_V8_API_DISABLED") is not None:
        return
    versions = str(build_info.get("dependency_versions", ""))
    cuda_match = re.search(r"\bCUDA_VERSION=(\d+)", versions)
    cudnn_match = re.search(r"\bCUDNN_VERSION=(\d+)", versions)
    if not cuda_match or not cudnn_match:
        return
    if int(cuda_match.group(1)) >= 13 and int(cudnn_match.group(1)) < 9:
        os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
        print(
            "CUDA 13 with cuDNN 8 detected; enabling the legacy cuDNN API.",
            file=sys.stderr,
        )


def stage_model(weights: Path, repository: Path) -> Path:
    weights = weights.expanduser().resolve()
    repository = repository.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"model weights not found: {weights}")
    repository.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        repository.glob("checkpoint-*.pt"), key=lambda path: path.stat().st_mtime
    )
    if checkpoints:
        return checkpoints[-1]
    destination = repository / weights.name
    if destination != weights:
        shutil.copyfile(weights, destination)
    return destination


def validate_resume_repository(repository: Path, resume: str) -> None:
    repository = repository.expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"model repository not found: {repository}")
    if resume == "latest":
        _require_any(repository, "solver-*.pt", "solver state")
        if not any(
            any(repository.glob(pattern))
            for pattern in ("checkpoint-*.pt", "checkpoint-*.npt", "checkpoint-*.ptw")
        ):
            raise FileNotFoundError(
                f"resume repository has no model checkpoint: {repository}"
            )
        return
    if resume == "best":
        iteration = best_model_iteration(repository)
        solver = repository / f"solver-{iteration}.pt"
        checkpoints = [
            repository / f"checkpoint-{iteration}.pt",
            repository / f"checkpoint-{iteration}.npt",
            repository / f"checkpoint-{iteration}.ptw",
        ]
        if not any(path.is_file() for path in checkpoints):
            expected = ", ".join(str(path) for path in checkpoints)
            raise FileNotFoundError(
                f"best checkpoint not found; expected one of: {expected}"
            )
        if not solver.is_file():
            raise FileNotFoundError(f"best solver state not found: {solver}")
        return
    raise ValueError("resume must be one of: latest, best")


def best_model_iteration(repository: Path) -> str:
    path = repository.expanduser().resolve() / "best_model.txt"
    if not path.is_file():
        raise FileNotFoundError(f"best model marker not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "iteration":
            iteration = value.strip()
            if iteration:
                return iteration
    raise ValueError(f"best model marker has no iteration entry: {path}")


def _require_any(repository: Path, pattern: str, description: str) -> None:
    if not any(repository.glob(pattern)):
        raise FileNotFoundError(
            f"resume repository has no {description} matching {pattern}: {repository}"
        )


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def report_error(error: BaseException) -> int:
    if isinstance(error, deepdetect.DeepDetectError):
        print(
            "DeepDetect error: "
            f"status={error.status_code} dd_code={error.dd_code} "
            f"message={error.message}",
            file=sys.stderr,
        )
    else:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    return 1
