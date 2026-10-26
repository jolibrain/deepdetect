#!/usr/bin/env python3
"""Keep only each stopped run's recorded best checkpoint/solver pair.

Usage from the DeepDetect repository:
    python3 tools/prune_checkpoints.py /path/to/runs
    python3 tools/prune_checkpoints.py /path/to/runs --apply

The default is a dry run. Selection uses best_model.txt, not metric inference.
Only checkpoint-N.pt and solver-N.pt files are eligible; other formats and
artifacts are untouched. The best pair is kept for resuming training. Runs with
missing metadata, incomplete best pairs, or active/unknown status are skipped.
Stop training before applying cleanup and do not restart it during cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import re

CHECKPOINT = re.compile(r"(?:checkpoint|solver)-([0-9]+)\.pt")
STOPPED = {"finished", "cancelled", "error", "failed"}


def candidates(root):
    """Find checkpoint directories without traversing symbolic links."""
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs
                            if not (Path(directory) / d).is_symlink())
        if any(CHECKPOINT.fullmatch(name) for name in files):
            yield Path(directory)


def plan(directory):
    # Support both worker repositories and repositories with a checkpoints/ dir.
    repository = directory.parent if directory.name == "checkpoints" else directory
    marker = repository / "best_model.txt"
    manifest = repository / "run.json"
    if marker.is_symlink() or manifest.is_symlink():
        raise ValueError("metadata is a symbolic link")
    try:
        marker_text = marker.read_text()
        manifest_text = manifest.read_text()
        status = json.loads(manifest_text).get("status")
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"missing or invalid metadata: {exc}") from exc
    if status not in STOPPED:
        raise ValueError(f"run status is {status!r}; only stopped runs are eligible")
    matches = re.findall(r"^iteration:([0-9]+)\s*$", marker_text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("best_model.txt must contain exactly one iteration:N entry")
    iteration = int(matches[0])
    keep = {f"checkpoint-{iteration}.pt", f"solver-{iteration}.pt"}
    if any(not (directory / name).is_file() or (directory / name).is_symlink()
           for name in keep):
        raise ValueError(f"best checkpoint/solver pair at iteration {iteration} is missing")
    remove = []
    for path in sorted(directory.iterdir()):
        if (CHECKPOINT.fullmatch(path.name) and path.name not in keep
                and not path.is_symlink() and path.is_file()):
            stat = path.stat()
            remove.append((path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return iteration, remove, (marker_text, manifest_text)


def prune(root, apply=False):
    count = total = 0
    for directory in candidates(root):
        try:
            iteration, remove, metadata = plan(directory)
            print(f"KEEP {directory}: checkpoint-{iteration}.pt + solver-{iteration}.pt")
            for path, *_ in remove:
                print(f"  {'DELETE' if apply else 'WOULD DELETE'} {path}")
            if apply:
                # Refuse a stale plan if training/metadata/files changed during preview.
                if plan(directory) != (iteration, remove, metadata):
                    raise ValueError("run changed during inspection; nothing deleted")
                for path, *_ in remove:
                    path.unlink()
            count += len(remove)
            total += sum(item[3] for item in remove)
        except (OSError, ValueError) as exc:
            print(f"SKIP {directory}: {exc}")
    print(f"{'Deleted' if apply else 'Would delete'} {count} files, "
          f"{total / 1024**3:.2f} GiB."
          + ("" if apply else " Use --apply to delete."))
    return count, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("runs"),
                        help="runs tree or single run (default: runs)")
    parser.add_argument("--apply", action="store_true", help="actually delete obsolete pairs")
    args = parser.parse_args()
    if not args.root.is_dir() or args.root.is_symlink():
        parser.error("root must be an existing directory, not a symbolic link")
    prune(args.root, args.apply)


if __name__ == "__main__":
    main()
