from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def check_dataset_list(path: Path, *, expected_fields: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset list not found: {path}")
    samples = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if expected_fields is not None and len(fields) != expected_fields:
            raise ValueError(f"{path}:{line_number}: expected {expected_fields} fields")
        samples += 1
    if samples == 0:
        raise ValueError(f"dataset list contains no samples: {path}")
    return {"path": str(path), "samples": samples}


def validate_segmentation_lists(list_paths: list[Path], nclasses: int) -> dict[str, Any]:
    if nclasses <= 0:
        raise ValueError("number of classes must be positive")
    invalid_values: set[int] = set()
    checked_masks = 0
    for list_path in list_paths:
        check_dataset_list(list_path, expected_fields=2)
        for line_number, line in enumerate(
            list_path.expanduser().resolve().read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            mask_path = Path(fields[1])
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"{list_path}:{line_number}: mask not found: {mask_path}"
                )
            values = np.unique(np.asarray(Image.open(mask_path)))
            invalid_values.update(
                int(value) for value in values if int(value) >= nclasses
            )
            checked_masks += 1
    if checked_masks == 0:
        raise ValueError("segmentation dataset lists contain no samples")
    if invalid_values:
        values = ", ".join(str(value) for value in sorted(invalid_values))
        raise ValueError(
            f"the {nclasses}-class SegFormer configuration requires mask values "
            f"from 0 through {nclasses - 1}; found: {values}"
        )
    return {"checked_masks": checked_masks, "nclasses": nclasses}


def validate_detection_lists(list_paths: list[Path], nclasses: int) -> dict[str, Any]:
    if nclasses <= 0:
        raise ValueError("number of classes must be positive")
    checked_images = 0
    checked_bbox_files = 0
    empty_bbox_files = 0
    positive_boxes = 0
    class_counts: dict[int, int] = {}
    list_summaries = [
        (list_path, check_dataset_list(list_path, expected_fields=2))
        for list_path in list_paths
    ]
    total_samples = sum(summary["samples"] for _, summary in list_summaries)

    progress = _progress(
        total=total_samples,
        desc="checking detection dataset",
        unit="sample",
    )
    try:
        for list_path, _ in list_summaries:
            list_path = list_path.expanduser().resolve()
            for line_number, line in enumerate(
                list_path.read_text(encoding="utf-8").splitlines(),
                1,
            ):
                if not line.strip():
                    continue
                image_raw, bbox_raw = line.split()
                image_path = _resolve_dataset_entry_path(list_path.parent, image_raw)
                bbox_path = _resolve_dataset_entry_path(list_path.parent, bbox_raw)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"{list_path}:{line_number}: image not found: {image_path}"
                    )
                if not bbox_path.is_file():
                    raise FileNotFoundError(
                        f"{list_path}:{line_number}: bbox file not found: {bbox_path}"
                    )

                checked_images += 1
                checked_bbox_files += 1
                boxes_in_file = 0
                for bbox_line_number, bbox_line in enumerate(
                    bbox_path.read_text(encoding="utf-8").splitlines(),
                    1,
                ):
                    if not bbox_line.strip():
                        raise ValueError(
                            f"{bbox_path}:{bbox_line_number}: empty bbox line"
                        )
                    fields = bbox_line.split()
                    if len(fields) != 5:
                        raise ValueError(
                            f"{bbox_path}:{bbox_line_number}: expected 5 fields"
                        )
                    cls = int(fields[0])
                    if cls <= 0 or cls >= nclasses:
                        raise ValueError(
                            f"{bbox_path}:{bbox_line_number}: invalid class {cls} "
                            f"for nclasses={nclasses}"
                        )
                    for value in fields[1:]:
                        float(value)
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    positive_boxes += 1
                    boxes_in_file += 1
                if boxes_in_file == 0:
                    empty_bbox_files += 1
                progress.update(1)
    finally:
        progress.close()

    if checked_images == 0:
        raise ValueError("detection dataset lists contain no samples")
    return {
        "checked_images": checked_images,
        "checked_bbox_files": checked_bbox_files,
        "empty_bbox_files": empty_bbox_files,
        "positive_boxes": positive_boxes,
        "class_counts": class_counts,
        "nclasses": nclasses,
    }


def validate_keypoint_lists(
    list_paths: list[Path], nkeypoints: int, *, head: str
) -> dict[str, Any]:
    if nkeypoints <= 0:
        raise ValueError("number of keypoints must be positive")
    checked_images = 0
    checked_keypoint_files = 0
    instances = 0
    list_summaries = [
        (list_path, check_dataset_list(list_path, expected_fields=2))
        for list_path in list_paths
    ]
    total_samples = sum(summary["samples"] for _, summary in list_summaries)
    progress = _progress(
        total=total_samples,
        desc="checking keypoint dataset",
        unit="sample",
    )
    try:
        for list_path, _summary in list_summaries:
            list_path = list_path.expanduser().resolve()
            for line_number, line in enumerate(
                list_path.read_text(encoding="utf-8").splitlines(),
                1,
            ):
                if not line.strip():
                    continue
                image_raw, keypoints_raw = line.split()
                image_path = _resolve_dataset_entry_path(list_path.parent, image_raw)
                keypoints_path = _resolve_dataset_entry_path(
                    list_path.parent,
                    keypoints_raw,
                )
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"{list_path}:{line_number}: image not found: {image_path}"
                    )
                if not keypoints_path.is_file():
                    raise FileNotFoundError(
                        f"{list_path}:{line_number}: keypoint file not found: {keypoints_path}"
                    )
                checked_images += 1
                checked_keypoint_files += 1
                for keypoint_line_number, keypoint_line in enumerate(
                    keypoints_path.read_text(encoding="utf-8").splitlines(),
                    1,
                ):
                    if not keypoint_line.strip():
                        continue
                    fields = keypoint_line.split()
                    expected = nkeypoints * 2 + (5 if head == "topdown" else 0)
                    if len(fields) != expected:
                        raise ValueError(
                            f"{keypoints_path}:{keypoint_line_number}: "
                            f"expected {expected} coordinate fields"
                        )
                    offset = 0
                    if head == "topdown":
                        cls = int(fields[0])
                        if cls <= 0 or str(cls) != fields[0]:
                            raise ValueError(
                                f"{keypoints_path}:{keypoint_line_number}: "
                                "class must be a positive integer"
                            )
                        xmin, ymin, xmax, ymax = (float(value) for value in fields[1:5])
                        if xmax <= xmin or ymax <= ymin:
                            raise ValueError(
                                f"{keypoints_path}:{keypoint_line_number}: invalid bbox"
                            )
                        offset = 5
                    coordinates = [float(value) for value in fields[offset:]]
                    for index in range(0, len(coordinates), 2):
                        x, y = coordinates[index : index + 2]
                        if (x == -1.0) != (y == -1.0) or x < -1.0 or y < -1.0:
                            raise ValueError(
                                f"{keypoints_path}:{keypoint_line_number}: "
                                "missing keypoints must be -1 -1"
                            )
                    instances += 1
                progress.update(1)
    finally:
        progress.close()
    if checked_images == 0:
        raise ValueError("keypoint dataset lists contain no samples")
    return {
        "checked_images": checked_images,
        "checked_keypoint_files": checked_keypoint_files,
        "instances": instances,
        "nkeypoints": nkeypoints,
        "head": head,
    }


def _resolve_dataset_entry_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _progress(*, total: int, desc: str, unit: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
        leave=False,
    )


class _NullProgress:
    def update(self, count: int) -> None:
        return None

    def close(self) -> None:
        return None


def run_training_checks(task: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    train_data = Path(options["train_data"])
    test_data = _test_data_paths(options["test_data"])
    dataset_check = str(options.get("dataset_check", "full"))
    if dataset_check == "none":
        event = {
            "name": "dataset_validation",
            "mode": "none",
            "skipped": True,
            "train_path": str(train_data.expanduser()),
        }
        if len(test_data) == 1:
            event["test_path"] = str(test_data[0].expanduser())
        else:
            event["test_paths"] = [str(path.expanduser()) for path in test_data]
        return [event]
    if dataset_check != "full":
        raise ValueError("dataset_check must be one of: full, none")

    checks = [{"name": "train_list", **check_dataset_list(train_data)}]
    for index, path in enumerate(test_data):
        name = "test_list" if len(test_data) == 1 else f"test_list_test{index}"
        checks.append({"name": name, **check_dataset_list(path)})
    if task == "segmentation" and not options.get("skip_mask_validation", False):
        checks.append(
            {
                "name": "segmentation_masks",
                **validate_segmentation_lists(
                    [train_data, *test_data], int(options["nclasses"])
                ),
            }
        )
    if task == "detection":
        checks.append(
            {
                "name": "detection_bboxes",
                **validate_detection_lists(
                    [train_data, *test_data], int(options["nclasses"])
                ),
            }
        )
    if task == "keypoint":
        checks.append(
            {
                "name": "keypoint_targets",
                **validate_keypoint_lists(
                    [train_data, *test_data],
                    int(options["nkeypoints"]),
                    head=_keypoint_head(options),
                ),
            }
        )
    return checks


def _keypoint_head(options: dict[str, Any]) -> str:
    head = "topdown"
    for section_name in ("service_mllib", "mllib"):
        section = options.get(section_name)
        if not isinstance(section, dict):
            continue
        vitpose = section.get("vitpose")
        if isinstance(vitpose, dict) and "head" in vitpose:
            head = str(vitpose["head"])
    if head not in {"topdown", "slots"}:
        raise ValueError("vitpose.head must be topdown or slots")
    return head


def _test_data_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]
