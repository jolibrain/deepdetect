from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image


def load_runner():
    source_root = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
    root = (
        Path(source_root).resolve()
        if source_root
        else Path(__file__).resolve().parents[3]
    )
    path = root / "tools" / "run_sam2_manifest.py"
    spec = importlib.util.spec_from_file_location("run_sam2_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_manifest_dataset(tmp_path: Path, count: int = 2):
    dataset = tmp_path / "dataset"
    train = dataset / "trainA"
    images = dataset / "ring" / "img"
    boxes = dataset / "ring" / "bbox"
    train.mkdir(parents=True)
    images.mkdir(parents=True)
    boxes.mkdir(parents=True)
    rows = []
    entries = []
    for index in range(count):
        image = images / f"sample-{index}.png"
        bbox = boxes / f"sample-{index}.txt"
        Image.new("RGB", (8, 8), "white").save(image)
        bbox.write_text("1 1 1 7 7\n", encoding="utf-8")
        rows.append(f"ring/img/{image.name} ring/bbox/{bbox.name}")
        entries.append((image.resolve(), bbox.resolve()))
    manifest = train / "paths.txt"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dataset, manifest, entries


def test_manifest_paths_resolve_from_parent_of_train_directory(tmp_path):
    runner = load_runner()
    dataset, manifest, expected = make_manifest_dataset(tmp_path)

    entries = runner.read_manifest(manifest, manifest.parent.parent)

    assert [(entry.image, entry.bbox) for entry in entries] == expected
    assert dataset == manifest.parent.parent


def test_manifest_rejects_multiple_bbox_rows(tmp_path):
    runner = load_runner()
    dataset, manifest, entries = make_manifest_dataset(tmp_path, count=1)
    entries[0][1].write_text("1 1 1 7 7\n1 2 2 6 6\n", encoding="utf-8")

    try:
        runner.read_manifest(manifest, dataset)
    except ValueError as error:
        assert "exactly one non-empty bbox row" in str(error)
    else:
        raise AssertionError("expected a bbox validation error")


def test_manifest_rejects_duplicate_image_outputs(tmp_path):
    runner = load_runner()
    dataset, manifest, _ = make_manifest_dataset(tmp_path, count=1)
    duplicate_bbox = dataset / "ring" / "bbox" / "duplicate.txt"
    duplicate_bbox.write_text("1 1 1 7 7\n", encoding="utf-8")
    manifest.write_text(
        "\n".join(
            [
                "ring/img/sample-0.png ring/bbox/sample-0.txt",
                "ring/img/sample-0.png ring/bbox/duplicate.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        runner.read_manifest(manifest, dataset)
    except ValueError as error:
        assert "output basename collision" in str(error)
    else:
        raise AssertionError("expected an output collision error")


def test_runner_uses_list_files_and_resumes_completed_overlays(monkeypatch, tmp_path):
    runner = load_runner()
    _, manifest, entries = make_manifest_dataset(tmp_path)
    weights = tmp_path / "sam2.1_hiera_tiny.pt"
    config = tmp_path / "sam2.yaml"
    output = tmp_path / "output"
    weights.write_bytes(b"weights")
    config.write_text("{}\n", encoding="utf-8")
    state = output / runner.STATE_DIRECTORY_NAME
    first_overlay = output / "sample-0_sam2_overlay.png"
    first_mask = output / "sample-0_sam2_mask_0001.png"
    first_overlay.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(first_overlay)
    first_mask.write_bytes(b"mask")
    runner.write_completion(
        state / runner.COMPLETION_FILE_NAME,
        runner.ManifestEntry(*entries[0]),
        first_overlay,
        first_mask,
    )
    seen = {}

    def fake_stream(
        command,
        *,
        cwd,
        pending,
        completion_path,
        completed,
        output_manifest,
        masks_dir,
        overlays_dir,
        class_mask_values,
        show_cli_output,
    ):
        del cwd, class_mask_values, show_cli_output
        seen["command"] = command
        seen["pending"] = pending
        image_list = Path(command[command.index("--images-file") + 1])
        bbox_list = Path(command[command.index("--bbox-files-file") + 1])
        assert image_list.read_text(encoding="utf-8").splitlines() == [
            str(entries[1][0])
        ]
        assert bbox_list.read_text(encoding="utf-8").splitlines() == [
            str(entries[1][1])
        ]
        overlay = overlays_dir / "sample-1_sam2_overlay.jpg"
        mask = masks_dir / "sample-1_sam2_mask_0001.png"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(overlay)
        mask.write_bytes(b"mask")
        runner.write_completion(completion_path, pending[0], overlay, mask)
        completed[pending[0].key] = runner.CompletionArtifacts(
            mask=mask,
            overlay=overlay,
        )
        with output_manifest.open("a", encoding="utf-8") as stream:
            stream.write(f"{pending[0].image} {mask}\n")
        return completed

    monkeypatch.setattr(runner, "stream_cli", fake_stream)
    args = runner.parse_args(
        [
            str(manifest),
            str(output),
            "--weights",
            str(weights),
            "--config",
            str(config),
            "--no-gpu",
        ]
    )

    assert runner.run_manifest(args) == 0
    assert [entry.key for entry in seen["pending"]] == [
        runner.ManifestEntry(*entries[1]).key
    ]
    assert "--images-file" in seen["command"]
    assert "--bbox-files-file" in seen["command"]
    assert "--no-gpu" in seen["command"]
    assert set(runner.load_completed(state / runner.COMPLETION_FILE_NAME)) == {
        runner.ManifestEntry(*entries[0]).key,
        runner.ManifestEntry(*entries[1]).key,
    }
    assert (output / runner.OUTPUT_DATASET_NAME).read_text(
        encoding="utf-8"
    ).splitlines() == [
        f"{entries[0][0]} {output / 'masks' / first_mask.name}",
        f"{entries[1][0]} {output / 'masks' / 'sample-1_sam2_mask_0001.png'}",
    ]
    assert not first_mask.exists()
    migrated_overlay = output / "overlays" / first_overlay.with_suffix(".jpg").name
    assert migrated_overlay.is_file()
    assert Image.open(migrated_overlay).format == "JPEG"


def test_runner_overwrite_reprocesses_completed_entries(monkeypatch, tmp_path):
    runner = load_runner()
    _, manifest, entries = make_manifest_dataset(tmp_path)
    weights = tmp_path / "sam2.1_hiera_tiny.pt"
    config = tmp_path / "sam2.yaml"
    output = tmp_path / "output"
    weights.write_bytes(b"weights")
    config.write_text("{}\n", encoding="utf-8")
    state = output / runner.STATE_DIRECTORY_NAME
    previous_overlay = output / "previous_overlay.png"
    previous_mask = output / "previous_mask.png"
    previous_overlay.parent.mkdir(parents=True)
    previous_overlay.write_bytes(b"overlay")
    previous_mask.write_bytes(b"mask")
    runner.write_completion(
        state / runner.COMPLETION_FILE_NAME,
        runner.ManifestEntry(*entries[0]),
        previous_overlay,
        previous_mask,
    )
    seen = {}

    def fake_stream(
        command,
        *,
        cwd,
        pending,
        completion_path,
        completed,
        output_manifest,
        masks_dir,
        overlays_dir,
        class_mask_values,
        show_cli_output,
    ):
        del command, cwd, class_mask_values, show_cli_output
        seen["pending"] = pending
        for index, entry in enumerate(pending):
            overlay = overlays_dir / f"replacement-{index}.png"
            mask = masks_dir / f"replacement-{index}_mask.png"
            overlay.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            overlay.write_bytes(b"overlay")
            mask.write_bytes(b"mask")
            runner.write_completion(completion_path, entry, overlay, mask)
            completed[entry.key] = runner.CompletionArtifacts(
                mask=mask,
                overlay=overlay,
            )
            with output_manifest.open("a", encoding="utf-8") as stream:
                stream.write(f"{entry.image} {mask}\n")
        return completed

    monkeypatch.setattr(runner, "stream_cli", fake_stream)
    args = runner.parse_args(
        [
            str(manifest),
            str(output),
            "--weights",
            str(weights),
            "--config",
            str(config),
            "--overwrite",
        ]
    )

    assert runner.run_manifest(args) == 0
    assert [entry.key for entry in seen["pending"]] == [
        runner.ManifestEntry(*image_bbox).key for image_bbox in entries
    ]
    assert set(runner.load_completed(state / runner.COMPLETION_FILE_NAME)) == {
        runner.ManifestEntry(*image_bbox).key for image_bbox in entries
    }


def test_overlay_artifact_marks_entry_complete(monkeypatch, tmp_path):
    runner = load_runner()
    _, manifest, entries = make_manifest_dataset(tmp_path, count=1)
    entry = runner.ManifestEntry(*entries[0])
    overlay = tmp_path / "overlay.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), "white").save(overlay)
    mask.write_bytes(b"mask")

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                json.dumps(
                    {
                        "event": "artifact",
                        "kind": "mask",
                        "image": str(entry.image),
                        "path": str(mask),
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "artifact",
                        "kind": "overlay",
                        "image": str(entry.image),
                        "path": str(overlay),
                    }
                )
                + "\n"
            )

        def wait(self):
            return 0

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    completion = tmp_path / "completed.jsonl"

    completed = runner.stream_cli(
        ["deepdetect"],
        cwd=manifest.parent,
        pending=[entry],
        completion_path=completion,
        completed={},
        output_manifest=tmp_path / "paths.txt",
        masks_dir=tmp_path / "masks",
        overlays_dir=tmp_path / "overlays",
        class_mask_values=False,
        show_cli_output=False,
    )

    expected = runner.CompletionArtifacts(
        mask=tmp_path / "masks" / mask.name,
        overlay=tmp_path / "overlays" / overlay.with_suffix(".jpg").name,
    )
    assert completed == {entry.key: expected}
    assert runner.load_completed(completion) == {entry.key: expected}
    assert (tmp_path / "paths.txt").read_text(encoding="utf-8") == (
        f"{entry.image} {expected.mask}\n"
    )
    assert Image.open(expected.overlay).format == "JPEG"


def test_class_mask_values_use_bbox_class_id(monkeypatch, tmp_path):
    runner = load_runner()
    _, manifest, entries = make_manifest_dataset(tmp_path, count=1)
    entry = runner.ManifestEntry(*entries[0])
    entry.bbox.write_text("7 1 1 7 7\n", encoding="utf-8")
    overlay = tmp_path / "overlay.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (2, 2), "white").save(overlay)
    binary_mask = Image.new("L", (2, 2))
    binary_mask.putdata([0, 1, 255, 0])
    binary_mask.save(mask)

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                json.dumps(
                    {
                        "event": "artifact",
                        "kind": "mask",
                        "image": str(entry.image),
                        "path": str(mask),
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "artifact",
                        "kind": "overlay",
                        "image": str(entry.image),
                        "path": str(overlay),
                    }
                )
                + "\n"
            )

        def wait(self):
            return 0

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    completion = tmp_path / "completed.jsonl"

    completed = runner.stream_cli(
        ["deepdetect"],
        cwd=manifest.parent,
        pending=[entry],
        completion_path=completion,
        completed={},
        output_manifest=tmp_path / "paths.txt",
        masks_dir=tmp_path / "masks",
        overlays_dir=tmp_path / "overlays",
        class_mask_values=True,
        show_cli_output=False,
    )

    artifacts = completed[entry.key]
    assert artifacts.mask_value == 7
    assert list(Image.open(artifacts.mask).get_flattened_data()) == [0, 7, 7, 0]
    assert runner.load_completed(completion)[entry.key].mask_value == 7


def test_class_mask_values_upgrade_completed_binary_mask(tmp_path):
    runner = load_runner()
    _, _, entries = make_manifest_dataset(tmp_path, count=1)
    entry = runner.ManifestEntry(*entries[0])
    entry.bbox.write_text("11 1 1 7 7\n", encoding="utf-8")
    mask = tmp_path / "old-mask.png"
    overlay = tmp_path / "old-overlay.jpg"
    binary_mask = Image.new("L", (2, 2))
    binary_mask.putdata([1, 0, 0, 255])
    binary_mask.save(mask)
    Image.new("RGB", (2, 2), "white").save(overlay)
    completion = tmp_path / "state" / "completed.jsonl"

    organized = runner.organize_completed_artifacts(
        [entry],
        {
            entry.key: runner.CompletionArtifacts(
                mask=mask,
                overlay=overlay,
            )
        },
        completion_path=completion,
        masks_dir=tmp_path / "masks",
        overlays_dir=tmp_path / "overlays",
        class_mask_values=True,
    )

    artifacts = organized[entry.key]
    assert artifacts.mask_value == 11
    assert list(Image.open(artifacts.mask).get_flattened_data()) == [11, 0, 0, 11]
    assert runner.load_completed(completion) == organized


def test_output_lock_rejects_concurrent_runner(tmp_path):
    runner = load_runner()
    state = tmp_path / runner.STATE_DIRECTORY_NAME
    state.mkdir()

    with runner.output_run_lock(state):
        try:
            with runner.output_run_lock(state):
                raise AssertionError("second runner unexpectedly acquired the lock")
        except RuntimeError as error:
            assert "another SAM2 manifest runner" in str(error)
