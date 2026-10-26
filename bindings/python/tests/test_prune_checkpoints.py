"""Checkpoint cleanup tests use temporary files only."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prune_checkpoints", Path(__file__).resolve().parents[3] / "tools/prune_checkpoints.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_run(root, status="finished", nested=False):
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({"status": status}))
    (root / "best_model.txt").write_text("iteration:20\nforeground_iou:0.7\n")
    target = root / "checkpoints" if nested else root
    target.mkdir(exist_ok=True)
    for step in (10, 20, 30):
        for prefix in ("checkpoint", "solver"):
            (target / f"{prefix}-{step}.pt").write_bytes(b"test")
    return target


@pytest.mark.parametrize("nested", [False, True])
def test_preview_and_apply(tmp_path, nested):
    target = make_run(tmp_path / "trial/control", nested=nested)
    (target / "notes.txt").write_text("keep")
    (target / "checkpoint-999.pt").symlink_to(target / "checkpoint-20.pt")
    assert module.prune(tmp_path) == (4, 16)
    assert (target / "checkpoint-10.pt").exists()
    assert module.prune(tmp_path, apply=True) == (4, 16)
    assert (target / "checkpoint-20.pt").exists()
    assert (target / "solver-20.pt").exists()
    assert (target / "notes.txt").exists()
    assert (target / "checkpoint-999.pt").is_symlink()
    assert not (target / "checkpoint-30.pt").exists()
    assert module.prune(tmp_path, apply=True) == (0, 0)


@pytest.mark.parametrize("case", ["running", "unknown", "missing_marker", "invalid_marker",
                                  "missing_solver", "missing_manifest", "symlink_marker"])
def test_unsafe_runs_skipped(tmp_path, case):
    target = make_run(tmp_path / "run", status=case if case in {"running", "unknown"} else "finished")
    marker = target / "best_model.txt"
    if case == "missing_marker":
        marker.unlink()
    elif case == "invalid_marker":
        marker.write_text("iteration:wrong")
    elif case == "missing_solver":
        (target / "solver-20.pt").unlink()
    elif case == "missing_manifest":
        (target / "run.json").unlink()
    elif case == "symlink_marker":
        marker.rename(target / "real-marker")
        marker.symlink_to(target / "real-marker")
    assert module.prune(tmp_path, apply=True) == (0, 0)
    assert (target / "checkpoint-10.pt").exists()


def test_symlink_directories_not_followed(tmp_path):
    target = make_run(tmp_path / "outside")
    tree = tmp_path / "runs"
    tree.mkdir()
    (tree / "linked").symlink_to(target, target_is_directory=True)
    assert module.prune(tree, apply=True) == (0, 0)
    assert (target / "checkpoint-10.pt").exists()
