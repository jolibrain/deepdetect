from pathlib import Path

import pytest

from deepdetect.pytorch_worker.checkpoints import resolve_training_checkpoint
from deepdetect.pytorch_worker.sdk import WorkerDependencyError


def _checkpoint_pair(repository: Path, iteration: int) -> None:
    (repository / f"checkpoint-{iteration}.pt").write_bytes(b"model")
    (repository / f"solver-{iteration}.pt").write_bytes(b"solver")


def test_weights_are_selected_without_resume(tmp_path):
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"weights")

    selection = resolve_training_checkpoint(
        {"weights": str(weights)},
        tmp_path,
    )

    assert selection.model == weights
    assert selection.solver is None
    assert selection.source == "weights"


def test_resume_precedes_weights_and_selects_highest_complete_pair(tmp_path):
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"weights")
    _checkpoint_pair(tmp_path, 3)
    _checkpoint_pair(tmp_path, 8)
    (tmp_path / "checkpoint-10.pt").write_bytes(b"incomplete")

    selection = resolve_training_checkpoint(
        {
            "resume": True,
            "resume_from": "latest",
            "weights": str(weights),
        },
        tmp_path,
    )

    assert selection.model == tmp_path / "checkpoint-8.pt"
    assert selection.solver == tmp_path / "solver-8.pt"
    assert selection.source == "resume"
    assert selection.resume_from == "latest"
    assert selection.iteration == 8


def test_latest_resume_uses_alias_pair_as_legacy_fallback(tmp_path):
    (tmp_path / "checkpoint-latest.pt").write_bytes(b"model")
    (tmp_path / "solver-latest.pt").write_bytes(b"solver")

    selection = resolve_training_checkpoint({"resume": True}, tmp_path)

    assert selection.model == tmp_path / "checkpoint-latest.pt"
    assert selection.solver == tmp_path / "solver-latest.pt"
    assert selection.resume_from == "latest"
    assert selection.iteration is None


def test_best_resume_selects_marker_iteration_pair(tmp_path):
    _checkpoint_pair(tmp_path, 4)
    _checkpoint_pair(tmp_path, 9)
    (tmp_path / "best_model.txt").write_text(
        "iteration:4\nmap-50:0.75\n",
        encoding="utf-8",
    )

    selection = resolve_training_checkpoint(
        {"resume": True, "resume_from": "best"},
        tmp_path,
    )

    assert selection.model == tmp_path / "checkpoint-4.pt"
    assert selection.solver == tmp_path / "solver-4.pt"
    assert selection.resume_from == "best"
    assert selection.iteration == 4


@pytest.mark.parametrize(
    ("mllib", "message"),
    [
        ({"resume": True}, "no complete model/solver checkpoint pair"),
        (
            {"resume": True, "resume_from": "best"},
            "best model marker not found",
        ),
        (
            {"resume": True, "resume_from": "unsupported"},
            "resume_from must be one of",
        ),
    ],
)
def test_invalid_resume_state_never_falls_back_to_weights(tmp_path, mllib, message):
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"weights")

    with pytest.raises(WorkerDependencyError, match=message):
        resolve_training_checkpoint(
            {**mllib, "weights": str(weights)},
            tmp_path,
        )
