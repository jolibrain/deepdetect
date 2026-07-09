from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def load_converter():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "coco_keypoints_to_dd.py"
    spec = importlib.util.spec_from_file_location("coco_keypoints_to_dd", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coco_keypoints_converter_writes_dd_format(tmp_path, capsys):
    converter = load_converter()
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 6), color="white").save(image_root / "sample.jpg")
    annotations = tmp_path / "person_keypoints.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 10,
                        "file_name": "sample.jpg",
                        "width": 8,
                        "height": 6,
                    },
                    {
                        "id": 11,
                        "file_name": "missing.jpg",
                        "width": 8,
                        "height": 6,
                    }
                ],
                "annotations": [
                    {
                        "id": 2,
                        "image_id": 10,
                        "category_id": 1,
                        "iscrowd": 0,
                        "num_keypoints": 2,
                        "keypoints": [1, 2, 2, 0, 0, 0, 5, 6, 1],
                    },
                    {
                        "id": 3,
                        "image_id": 10,
                        "category_id": 1,
                        "iscrowd": 1,
                        "num_keypoints": 3,
                        "keypoints": [1, 1, 2, 2, 2, 2, 3, 3, 2],
                    },
                    {
                        "id": 4,
                        "image_id": 11,
                        "category_id": 1,
                        "iscrowd": 0,
                        "num_keypoints": 3,
                        "keypoints": [1, 1, 2, 2, 2, 2, 3, 3, 2],
                    },
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": ["left", "center", "right"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    list_path = converter.convert_coco_keypoints(
        annotations=annotations,
        image_root=image_root,
        output_dir=tmp_path / "out",
        list_name="val.txt",
        nkeypoints=None,
        category_id=1,
        relative_paths=True,
    )

    rows = list_path.read_text(encoding="utf-8").splitlines()
    assert rows == ["../images/sample.jpg keypoints/000000000010.txt"]
    target = list_path.parent / "keypoints" / "000000000010.txt"
    assert target.read_text(encoding="utf-8").splitlines() == [
        "1 2 -1 -1 5 6"
    ]
    captured = capsys.readouterr()
    assert "Skipped 1 image(s)" in captured.err
