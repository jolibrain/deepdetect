from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WHEEL_TEST_SOURCE_ROOT = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
ROOT = (
    Path(WHEEL_TEST_SOURCE_ROOT).resolve()
    if WHEEL_TEST_SOURCE_ROOT
    else Path(__file__).resolve().parents[3]
)
SAM2_ROOT = ROOT / "extern" / "pytorch_workers" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from deepdetect.pytorch_worker.sdk import (
    PredictionContractError,
    WorkerDependencyError,
    WorkerContext,
    validate_prediction_result,
)
from sam2_worker.config import SAM2_VARIANTS, worker_config_from_mllib
from sam2_worker.rle import encode_coco_rle
from sam2_worker.worker_impl import (
    BBoxPrompt,
    DeepDetectWorker,
    MaskPrediction,
    _bbox_from_mask,
    _connector_prediction,
    _prompt_masks,
    _read_bbox_file,
)
import sam2_worker.worker_impl as worker_impl


def test_sam2_config_defaults_to_tiny_and_requires_checkpoint(tmp_path):
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.touch()

    config = worker_config_from_mllib({"weights": str(checkpoint), "sam2": {}})

    assert config.variant == "tiny"
    assert config.config_path == SAM2_VARIANTS["tiny"]
    assert config.automatic.points_per_side == 32
    assert config.automatic.max_masks == 0


def test_sam2_config_rejects_missing_checkpoint():
    with pytest.raises(WorkerDependencyError, match="checkpoint not found"):
        worker_config_from_mllib({"weights": "/missing/sam2.pt", "sam2": {}})


def test_coco_rle_uses_column_major_mask_order():
    mask = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)

    assert encode_coco_rle(mask) == {
        "encoding": "coco_rle",
        "size": [2, 3],
        "counts": [1, 3, 2],
    }


def test_sam2_bbox_sidecar_clips_source_coordinates(tmp_path):
    boxes = tmp_path / "boxes.txt"
    boxes.write_text("8 -2 1 9 6\n", encoding="utf-8")

    prompts = _read_bbox_file(boxes, width=8, height=5)

    assert prompts == [BBoxPrompt(category="8", bbox=(0.0, 1.0, 8.0, 5.0))]


def test_sam2_bbox_sidecar_rejects_degenerate_boxes(tmp_path):
    boxes = tmp_path / "boxes.txt"
    boxes.write_text("8 2 2 2 4\n", encoding="utf-8")

    with pytest.raises(PredictionContractError, match="positive clipped area"):
        _read_bbox_file(boxes, width=8, height=5)


def test_sam2_prompt_result_requires_one_mask_for_each_box():
    with pytest.raises(PredictionContractError, match="one mask per bounding box"):
        _prompt_masks(np.zeros((1, 3, 3), dtype=np.uint8), expected_count=2)


def test_sam2_connector_result_exposes_parallel_binary_masks(tmp_path):
    image = tmp_path / "image.png"
    mask = np.asarray([[0, 1], [1, 1]], dtype=np.uint8)
    prediction = MaskPrediction(
        mask=mask,
        bbox=_bbox_from_mask(mask),
        score=0.9,
        category="5",
        stability_score=0.8,
    )

    result = {"results": [_connector_prediction(image, [prediction])]}

    assert validate_prediction_result(result) == result
    item = result["results"][0]
    assert item["cats"] == ["5"]
    assert item["masks"][0]["size"] == [2, 2]


def test_prediction_contract_rejects_unaligned_masks():
    with pytest.raises(PredictionContractError, match="one entry per probability"):
        validate_prediction_result(
            {
                "results": [
                    {
                        "probs": [0.9],
                        "masks": [],
                    }
                ]
            }
        )


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeModel:
    def eval(self):
        return self


class _FakeAutomaticGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @staticmethod
    def generate(image):
        del image
        return [
            {
                "segmentation": np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
                "predicted_iou": 0.7,
                "stability_score": 0.8,
            },
            {
                "segmentation": np.asarray([[0, 1], [1, 1]], dtype=np.uint8),
                "predicted_iou": 0.9,
                "stability_score": 0.6,
            },
        ]


class _FakeImagePredictor:
    def __init__(self, model):
        self.model = model
        self.image = None
        self.image_batch = None
        self.predict_batch_calls = 0

    def set_image(self, image):
        self.image = image

    def set_image_batch(self, images):
        self.image_batch = images

    def predict_batch(self, *, box_batch, multimask_output):
        assert multimask_output is False
        self.predict_batch_calls += 1
        masks_batch = []
        scores_batch = []
        for boxes in box_batch:
            masks = np.zeros((len(boxes), 1, 2, 2), dtype=np.uint8)
            masks[:, 0, 0, 0] = 1
            masks_batch.append(masks)
            scores_batch.append(np.full((len(boxes), 1), 0.8))
        return masks_batch, scores_batch, None

    @staticmethod
    def predict(*, box, multimask_output):
        assert multimask_output is False
        masks = np.zeros((len(box), 1, 2, 2), dtype=np.uint8)
        masks[:, 0, 0, 0] = 1
        return masks, np.full((len(box), 1), 0.8), None


def _configured_fake_worker(monkeypatch, checkpoint: Path) -> DeepDetectWorker:
    worker = DeepDetectWorker()
    monkeypatch.setattr(worker, "_import_torch", lambda: _FakeTorch())
    monkeypatch.setattr(worker_impl, "select_device", lambda torch, mllib: ("cpu", False))

    def import_runtime():
        worker._build_sam2 = lambda *args, **kwargs: _FakeModel()
        worker._automatic_generator_class = _FakeAutomaticGenerator
        worker._image_predictor_class = _FakeImagePredictor

    monkeypatch.setattr(worker, "_import_sam2_runtime", import_runtime)
    worker.configure(
        WorkerContext(
            repository=str(checkpoint.parent),
            mllib={"weights": str(checkpoint), "gpu": False, "sam2": {}},
            raw={},
        )
    )
    return worker


def test_sam2_worker_automatic_prediction_sorts_masks(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.touch()
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image)
    worker = _configured_fake_worker(monkeypatch, checkpoint)

    result = worker.predict({"request": {"data": [str(image)]}})

    item = result["results"][0]
    assert item["probs"] == [0.9, 0.7]
    assert item["cats"] == ["0", "0"]
    assert item["masks"][0]["counts"] == [1, 3]


def test_sam2_worker_box_prediction_preserves_sidecar_category(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.touch()
    image = tmp_path / "image.png"
    boxes = tmp_path / "boxes.txt"
    Image.new("RGB", (2, 2), "white").save(image)
    boxes.write_text("9 0 0 2 2\n", encoding="utf-8")
    worker = _configured_fake_worker(monkeypatch, checkpoint)

    result = worker.predict(
        {
            "request": {
                "data": [str(image)],
                "parameters": {"input": {"bbox_files": [str(boxes)]}},
            }
        }
    )

    item = result["results"][0]
    assert item["cats"] == ["9"]
    assert item["probs"] == [0.8]
    assert item["bboxes"] == [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}]


def test_sam2_worker_batches_box_prompt_image_embeddings(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.touch()
    images = []
    boxes = []
    for index in range(2):
        image = tmp_path / f"image-{index}.png"
        bbox = tmp_path / f"boxes-{index}.txt"
        Image.new("RGB", (2, 2), "white").save(image)
        bbox.write_text(f"{index + 1} 0 0 2 2\n", encoding="utf-8")
        images.append(image)
        boxes.append(bbox)
    worker = _configured_fake_worker(monkeypatch, checkpoint)

    result = worker.predict(
        {
            "request": {
                "data": [str(image) for image in images],
                "parameters": {
                    "input": {"bbox_files": [str(bbox) for bbox in boxes]}
                },
            }
        }
    )

    predictor = worker._image_predictor
    assert isinstance(predictor, _FakeImagePredictor)
    assert predictor.image_batch is not None
    assert len(predictor.image_batch) == 2
    assert predictor.predict_batch_calls == 1
    assert [item["cats"] for item in result["results"]] == [["1"], ["2"]]
