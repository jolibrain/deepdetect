#!/usr/bin/env python3
"""Convert COCO keypoint annotations to DeepDetect keypoint lists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO person keypoints to DeepDetect keypoint format."
    )
    parser.add_argument("annotations", type=Path, help="COCO keypoint JSON file")
    parser.add_argument("image_root", type=Path, help="Directory containing images")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--list-name",
        default="keypoints.txt",
        help="Output list filename inside output_dir",
    )
    parser.add_argument(
        "--nkeypoints",
        type=int,
        default=None,
        help=(
            "Override expected keypoints per instance. By default this is "
            "detected from COCO category metadata or annotation lengths."
        ),
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category id to export",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to output_dir instead of absolute paths",
    )
    parser.add_argument(
        "--fail-on-missing-images",
        action="store_true",
        help="Fail instead of skipping annotations whose image file is missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    convert_coco_keypoints(
        annotations=args.annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        list_name=args.list_name,
        nkeypoints=args.nkeypoints,
        category_id=args.category_id,
        relative_paths=args.relative_paths,
        skip_missing_images=not args.fail_on_missing_images,
    )
    return 0


def convert_coco_keypoints(
    *,
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    list_name: str,
    nkeypoints: int | None,
    category_id: int,
    relative_paths: bool,
    skip_missing_images: bool = True,
) -> Path:
    annotations = annotations.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotations}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO image root not found: {image_root}")

    data = json.loads(annotations.read_text(encoding="utf-8"))
    nkeypoints = detect_nkeypoints(data, category_id, nkeypoints)
    images = {
        int(image["id"]): image
        for image in data.get("images", [])
        if isinstance(image, dict) and "id" in image and "file_name" in image
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for item in data.get("annotations", []):
        if not isinstance(item, dict):
            continue
        if int(item.get("category_id", -1)) != category_id:
            continue
        if int(item.get("iscrowd", 0)) != 0:
            continue
        if int(item.get("num_keypoints", 0)) <= 0:
            continue
        image_id = int(item.get("image_id", -1))
        if image_id not in images:
            continue
        keypoints = item.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) != nkeypoints * 3:
            raise ValueError(
                f"annotation {item.get('id', '<unknown>')}: expected "
                f"{nkeypoints * 3} COCO keypoint values"
            )
        annotations_by_image.setdefault(image_id, []).append(item)

    keypoint_dir = output_dir / "keypoints"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / list_name
    list_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    skipped_missing_images = 0
    for image_id in sorted(annotations_by_image):
        image = images[image_id]
        image_path = (image_root / str(image["file_name"])).resolve()
        if not image_path.is_file():
            if not skip_missing_images:
                raise FileNotFoundError(f"COCO image not found: {image_path}")
            skipped_missing_images += 1
            continue
        target_path = keypoint_dir / f"{image_id:012d}.txt"
        lines = [
            format_deepdetect_keypoint_line(item["keypoints"], nkeypoints)
            for item in sorted(
                annotations_by_image[image_id],
                key=lambda value: int(value.get("id", 0)),
            )
        ]
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append(
            "%s %s"
            % (
                output_path(image_path, list_path.parent, relative_paths),
                output_path(target_path, list_path.parent, relative_paths),
            )
        )

    if not rows:
        message = "COCO annotations contain no exportable keypoints"
        if skipped_missing_images:
            message += (
                f"; skipped {skipped_missing_images} image(s) missing from "
                f"{image_root}"
            )
        raise ValueError(message)
    if skipped_missing_images:
        print(
            f"Skipped {skipped_missing_images} image(s) missing from {image_root}",
            file=sys.stderr,
        )
    list_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return list_path


def detect_nkeypoints(
    data: dict[str, Any], category_id: int, requested_nkeypoints: int | None
) -> int:
    if requested_nkeypoints is not None:
        if requested_nkeypoints <= 0:
            raise ValueError("--nkeypoints must be positive")
        return requested_nkeypoints

    for category in data.get("categories", []):
        if not isinstance(category, dict):
            continue
        if int(category.get("id", -1)) != category_id:
            continue
        category_keypoints = category.get("keypoints")
        if isinstance(category_keypoints, list) and category_keypoints:
            return len(category_keypoints)

    detected_nkeypoints: int | None = None
    for item in data.get("annotations", []):
        if not isinstance(item, dict):
            continue
        if int(item.get("category_id", -1)) != category_id:
            continue
        keypoints = item.get("keypoints")
        if not isinstance(keypoints, list):
            continue
        if len(keypoints) == 0 or len(keypoints) % 3 != 0:
            raise ValueError(
                f"annotation {item.get('id', '<unknown>')}: COCO keypoint "
                "values must be a non-empty multiple of 3"
            )
        item_nkeypoints = len(keypoints) // 3
        if detected_nkeypoints is None:
            detected_nkeypoints = item_nkeypoints
        elif detected_nkeypoints != item_nkeypoints:
            raise ValueError(
                f"annotation {item.get('id', '<unknown>')}: inconsistent "
                f"keypoint count {item_nkeypoints}, expected "
                f"{detected_nkeypoints}"
            )

    if detected_nkeypoints is not None:
        return detected_nkeypoints

    raise ValueError(
        "unable to detect keypoint count from COCO dataset; pass --nkeypoints"
    )


def format_deepdetect_keypoint_line(keypoints: list[Any], nkeypoints: int) -> str:
    fields: list[str] = []
    for index in range(nkeypoints):
        x = float(keypoints[index * 3])
        y = float(keypoints[index * 3 + 1])
        visibility = int(keypoints[index * 3 + 2])
        if visibility == 0:
            fields.extend(["-1", "-1"])
        else:
            fields.extend([format_number(x), format_number(y)])
    return " ".join(fields)


def output_path(path: Path, base: Path, relative: bool) -> str:
    if not relative:
        return str(path)
    return os.path.relpath(path, base)


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
