#!/usr/bin/env python3
"""Run DeepDetect SAM2 bbox prompts for a joliGEN-style paths.txt manifest."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image
from tqdm import tqdm

STATE_DIRECTORY_NAME = ".sam2-manifest-run"
COMPLETION_FILE_NAME = "completed.jsonl"
OUTPUT_DATASET_NAME = "paths.txt"


@dataclass(frozen=True)
class ManifestEntry:
    image: Path
    bbox: Path

    @property
    def key(self) -> tuple[str, str]:
        return str(self.image), str(self.bbox)


@dataclass(frozen=True)
class CompletionArtifacts:
    mask: Path
    overlay: Path
    mask_value: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run DeepDetect SAM2 once for every pair in a paths.txt manifest."
    )
    parser.add_argument("manifest", type=Path, help="Two-column image/bbox paths.txt")
    parser.add_argument("output_dir", type=Path, help="Directory for SAM2 artifacts")
    parser.add_argument("--weights", required=True, type=Path, help="SAM2 checkpoint")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Base directory for relative manifest paths (default: manifest parent parent)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "extern/pytorch_workers/sam2/config.yaml",
        help="SAM2 DeepDetect CLI config",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        help="DeepDetect model repository (default: output state directory)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python interpreter with DeepDetect and sam2 installed",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gpuid", nargs="+")
    parser.add_argument("--limit", type=int, help="Maximum pending entries to run")
    parser.add_argument(
        "--class-mask-values",
        action="store_true",
        help="Set foreground mask pixels to the bbox class ID instead of 1",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the completion state and reprocess all entries",
    )
    parser.add_argument(
        "--show-cli-output",
        action="store_true",
        help="Forward every JSONL event from the DeepDetect CLI",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_manifest(args)


def run_manifest(args: argparse.Namespace) -> int:
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not weights.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {config}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else manifest.parent.parent.resolve()
    )
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    entries = read_manifest(manifest, data_root)
    if args.class_mask_values:
        for entry in entries:
            class_id = bbox_class_id(entry.bbox)
            if not 1 <= class_id <= 255:
                raise ValueError(
                    f"{entry.bbox}: class-mask-values requires a class ID "
                    f"between 1 and 255, got {class_id}"
                )
    state_dir = output_dir / STATE_DIRECTORY_NAME
    completion_path = state_dir / COMPLETION_FILE_NAME
    output_manifest = output_dir / OUTPUT_DATASET_NAME
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    if args.dry_run:
        completed = {} if args.overwrite else load_completed(completion_path)
        pending = [entry for entry in entries if entry.key not in completed]
        completed_count = len(entries) - len(pending)
        if args.limit is not None:
            pending = pending[: args.limit]
        print(
            f"manifest={manifest} data_root={data_root} total={len(entries)} "
            f"completed={completed_count} scheduled={len(pending)}"
        )
        return 0
    state_dir.mkdir(parents=True, exist_ok=True)
    with output_run_lock(state_dir):
        return run_manifest_locked(
            args,
            entries=entries,
            state_dir=state_dir,
            completion_path=completion_path,
            output_manifest=output_manifest,
            masks_dir=masks_dir,
            overlays_dir=overlays_dir,
            weights=weights,
            config=config,
        )


def run_manifest_locked(
    args: argparse.Namespace,
    *,
    entries: list[ManifestEntry],
    state_dir: Path,
    completion_path: Path,
    output_manifest: Path,
    masks_dir: Path,
    overlays_dir: Path,
    weights: Path,
    config: Path,
) -> int:
    if args.overwrite and completion_path.exists():
        completion_path.unlink()
    completed = load_completed(completion_path)
    pending = [entry for entry in entries if entry.key not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    completed = organize_completed_artifacts(
        entries,
        completed,
        completion_path=completion_path,
        masks_dir=masks_dir,
        overlays_dir=overlays_dir,
        class_mask_values=bool(args.class_mask_values),
    )
    write_output_manifest(output_manifest, entries, completed)
    if not pending:
        print("No pending SAM2 manifest entries")
        return 0

    image_list = state_dir / "images.txt"
    bbox_list = state_dir / "bbox-files.txt"
    write_path_list(image_list, [entry.image for entry in pending])
    write_path_list(bbox_list, [entry.bbox for entry in pending])
    repository = (
        args.repository.expanduser().resolve()
        if args.repository is not None
        else state_dir / "repository"
    )
    command = cli_command(
        args,
        config=config,
        weights=weights,
        repository=repository,
        output_dir=state_dir / "artifacts",
        image_list=image_list,
        bbox_list=bbox_list,
    )
    print(f"Running {len(pending)} SAM2 entries")
    if args.show_cli_output:
        print(shlex.join(command))
    completed_now = stream_cli(
        command,
        cwd=Path(__file__).resolve().parents[1],
        pending=pending,
        completion_path=completion_path,
        completed=completed,
        output_manifest=output_manifest,
        masks_dir=masks_dir,
        overlays_dir=overlays_dir,
        class_mask_values=bool(args.class_mask_values),
        show_cli_output=bool(args.show_cli_output),
    )
    missing = [entry for entry in pending if entry.key not in completed_now]
    if missing:
        raise RuntimeError(
            f"SAM2 completed without overlays for {len(missing)} manifest entries"
        )
    return 0


@contextmanager
def output_run_lock(state_dir: Path) -> Iterator[None]:
    lock_path = state_dir / "run.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another SAM2 manifest runner is using this output: "
                f"{state_dir.parent}"
            ) from error
        yield
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def read_manifest(manifest: Path, data_root: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen_stems: dict[str, Path] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{manifest}:{line_number}: expected image and bbox paths")
        image = resolve_manifest_path(data_root, fields[0])
        bbox = resolve_manifest_path(data_root, fields[1])
        if not image.is_file():
            raise FileNotFoundError(
                f"{manifest}:{line_number}: image not found: {image}"
            )
        if not bbox.is_file():
            raise FileNotFoundError(f"{manifest}:{line_number}: bbox not found: {bbox}")
        validate_bbox_file(bbox)
        previous = seen_stems.get(image.stem)
        if previous is not None:
            raise ValueError(
                f"{manifest}:{line_number}: output basename collision for "
                f"{image.name} and {previous.name}"
            )
        seen_stems[image.stem] = image
        entries.append(ManifestEntry(image=image, bbox=bbox))
    if not entries:
        raise ValueError(f"manifest contains no entries: {manifest}")
    return entries


def resolve_manifest_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else data_root / path).resolve()


def bbox_class_id(path: Path) -> int:
    rows = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty bbox row")
    fields = rows[0].split()
    if len(fields) != 5:
        raise ValueError(f"{path}: expected cls xmin ymin xmax ymax")
    try:
        class_id = int(fields[0])
        xmin, ymin, xmax, ymax = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"{path}: bbox fields must be numeric") from error
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"{path}: bbox values must be finite")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"{path}: bbox must have positive area")
    return class_id


def validate_bbox_file(path: Path) -> None:
    bbox_class_id(path)


def load_completed(path: Path) -> dict[tuple[str, str], CompletionArtifacts]:
    if not path.is_file():
        return {}
    completed: dict[tuple[str, str], CompletionArtifacts] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            image = Path(str(record["image"])).resolve()
            bbox = Path(str(record["bbox"])).resolve()
            overlay = Path(str(record["overlay"])).resolve()
            raw_mask = record.get("mask")
            raw_mask_value = record.get("mask_value")
            mask = (
                Path(str(raw_mask)).resolve()
                if raw_mask is not None
                else mask_path_for_overlay(overlay)
            )
            mask_value = (
                int(raw_mask_value) if raw_mask_value is not None else None
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid completion record"
            ) from error
        if overlay.is_file() and mask.is_file():
            completed[(str(image), str(bbox))] = CompletionArtifacts(
                mask=mask,
                overlay=overlay,
                mask_value=mask_value,
            )
    return completed


def write_path_list(path: Path, paths: list[Path]) -> None:
    payload = "".join(f"{item}\n" for item in paths)
    write_text_atomic(path, payload)


def write_completion(
    path: Path,
    entry: ManifestEntry,
    overlay: Path,
    mask: Path,
    mask_value: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(
        {
            "image": str(entry.image),
            "bbox": str(entry.bbox),
            "overlay": str(overlay),
            "mask": str(mask),
            "mask_value": mask_value,
        },
        sort_keys=True,
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def mask_path_for_overlay(overlay: Path) -> Path:
    base = overlay.stem.removesuffix("_overlay")
    return overlay.with_name(f"{base}_mask_0001.png")


def write_output_manifest(
    path: Path,
    entries: list[ManifestEntry],
    completed: dict[tuple[str, str], CompletionArtifacts],
) -> None:
    payload = "".join(
        f"{entry.image} {completed[entry.key].mask}\n"
        for entry in entries
        if entry.key in completed
    )
    write_text_atomic(path, payload)


def organize_completed_artifacts(
    entries: list[ManifestEntry],
    completed: dict[tuple[str, str], CompletionArtifacts],
    *,
    completion_path: Path,
    masks_dir: Path,
    overlays_dir: Path,
    class_mask_values: bool,
) -> dict[tuple[str, str], CompletionArtifacts]:
    organized = dict(completed)
    for entry in entries:
        artifacts = organized.get(entry.key)
        if artifacts is None:
            continue
        mask = copy_artifact(artifacts.mask, masks_dir)
        overlay = copy_overlay_as_jpeg(artifacts.overlay, overlays_dir)
        mask_value = artifacts.mask_value
        if class_mask_values:
            class_id = bbox_class_id(entry.bbox)
            if mask_value != class_id:
                apply_class_mask_value(mask, class_id)
            mask_value = class_id
        updated = CompletionArtifacts(
            mask=mask,
            overlay=overlay,
            mask_value=mask_value,
        )
        if updated == artifacts:
            continue
        write_completion(
            completion_path,
            entry,
            overlay,
            mask,
            mask_value=mask_value,
        )
        organized[entry.key] = updated
        remove_relocated_source(artifacts.mask, mask)
        remove_relocated_source(artifacts.overlay, overlay)
    return organized


def copy_artifact(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / source.name).resolve()
    if source.resolve() == destination:
        return destination
    shutil.copy2(source, destination)
    return destination


def copy_overlay_as_jpeg(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / source.with_suffix(".jpg").name).resolve()
    if source.resolve() == destination:
        return destination
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, destination)
    else:
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
    return destination


def remove_relocated_source(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve() and source.is_file():
        source.unlink()


def move_artifact(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / source.name).resolve()
    if source.resolve() != destination:
        source.replace(destination)
    return destination


def apply_class_mask_value(mask_path: Path, class_id: int) -> None:
    with Image.open(mask_path) as image:
        binary_mask = image.convert("L")
        class_mask = binary_mask.point(
            lambda value: class_id if value else 0,
            mode="L",
        )
    with tempfile.NamedTemporaryFile(
        "wb", dir=mask_path.parent, suffix=".png", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        class_mask.save(temporary_path, format="PNG")
        temporary_path.replace(mask_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def move_overlay_as_jpeg(source: Path, directory: Path) -> Path:
    destination = copy_overlay_as_jpeg(source, directory)
    remove_relocated_source(source, destination)
    return destination


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def cli_command(
    args: argparse.Namespace,
    *,
    config: Path,
    weights: Path,
    repository: Path,
    output_dir: Path,
    image_list: Path,
    bbox_list: Path,
) -> list[str]:
    command = [
        str(args.python_executable),
        "-m",
        "deepdetect.cli.main",
        "infer",
        "sam2",
        "--config",
        str(config),
        "--images-file",
        str(image_list),
        "--bbox-files-file",
        str(bbox_list),
        "--weights",
        str(weights),
        "--repository",
        str(repository),
        "--output",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--output-format",
        "jsonl",
    ]
    if args.gpu is True:
        command.append("--gpu")
    elif args.gpu is False:
        command.append("--no-gpu")
    if args.gpuid:
        command.extend(["--gpuid", *(str(value) for value in args.gpuid)])
    return command


def stream_cli(
    command: list[str],
    *,
    cwd: Path,
    pending: list[ManifestEntry],
    completion_path: Path,
    completed: dict[tuple[str, str], CompletionArtifacts],
    output_manifest: Path,
    masks_dir: Path,
    overlays_dir: Path,
    class_mask_values: bool,
    show_cli_output: bool,
) -> dict[tuple[str, str], CompletionArtifacts]:
    pending_by_image = {str(entry.image): entry for entry in pending}
    masks_by_image: dict[str, list[Path]] = {}
    invalid_mask_counts: dict[str, int] = {}
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("unable to capture DeepDetect CLI output")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("a", encoding="utf-8") as manifest_stream, tqdm(
        total=len(pending), desc="SAM2", unit="image", dynamic_ncols=True
    ) as progress:
        for line in process.stdout:
            if show_cli_output:
                sys.stdout.write(line)
            event = artifact_event(line)
            if event is None:
                if not line.lstrip().startswith("{"):
                    sys.stderr.write(line)
                continue
            image = str(Path(str(event["image"])).resolve())
            entry = pending_by_image.get(image)
            if entry is None:
                continue
            kind = event["kind"]
            artifact_path = Path(str(event["path"])).resolve()
            if kind == "mask":
                if artifact_path.is_file():
                    mask = move_artifact(artifact_path, masks_dir)
                    if class_mask_values:
                        apply_class_mask_value(mask, bbox_class_id(entry.bbox))
                    masks_by_image.setdefault(image, []).append(mask)
                continue
            if kind != "overlay":
                continue
            if not artifact_path.is_file() or entry.key in completed:
                continue
            mask_paths = masks_by_image.get(image, [])
            if len(mask_paths) != 1:
                invalid_mask_counts[image] = len(mask_paths)
                continue
            overlay = move_overlay_as_jpeg(artifact_path, overlays_dir)
            mask_value = bbox_class_id(entry.bbox) if class_mask_values else None
            write_completion(
                completion_path,
                entry,
                overlay,
                mask_paths[0],
                mask_value=mask_value,
            )
            completed[entry.key] = CompletionArtifacts(
                mask=mask_paths[0],
                overlay=overlay,
                mask_value=mask_value,
            )
            manifest_stream.write(f"{entry.image} {mask_paths[0]}\n")
            manifest_stream.flush()
            progress.update(1)
    exit_code = process.wait()
    if exit_code:
        raise subprocess.CalledProcessError(exit_code, command)
    if invalid_mask_counts:
        details = ", ".join(
            f"{image} ({count} masks)"
            for image, count in sorted(invalid_mask_counts.items())
        )
        raise RuntimeError(
            "SAM2 box prompts must produce exactly one binary mask before "
            f"their overlay: {details}"
        )
    return completed


def artifact_event(line: str) -> dict[str, Any] | None:
    if '"event": "artifact"' not in line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("event") != "artifact" or not isinstance(
        payload.get("kind"), str
    ):
        return None
    if not isinstance(payload.get("image"), str) or not isinstance(
        payload.get("path"), str
    ):
        return None
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
