from __future__ import annotations

import importlib.metadata
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from deepdetect.pytorch_worker.builtin.vision.detection.common import select_device
from deepdetect.pytorch_worker.sdk import (
    DeepDetectWorkerBase,
    PredictionContractError,
    WorkerContext,
    WorkerDependencyError,
)

from .config import Sam2WorkerConfig, worker_config_from_mllib
from .rle import encode_coco_rle


SAM2_PACKAGE_VERSION = "1.1.0"


@dataclass(frozen=True)
class MaskPrediction:
    mask: np.ndarray
    bbox: dict[str, float]
    score: float
    category: str
    stability_score: float


class DeepDetectWorker(DeepDetectWorkerBase):
    def __init__(self) -> None:
        super().__init__()
        self.config: Sam2WorkerConfig | None = None
        self.torch: Any = None
        self.device: Any = None
        self._build_sam2: Any = None
        self._automatic_generator_class: Any = None
        self._image_predictor_class: Any = None
        self._automatic_generator: Any = None
        self._image_predictor: Any = None

    def configure(self, context: WorkerContext) -> dict[str, Any]:
        super().configure(context)
        self.config = worker_config_from_mllib(context.mllib)
        self.torch = self._import_torch()
        self.device, multi_gpu_requested = select_device(self.torch, dict(context.mllib))
        if multi_gpu_requested:
            raise WorkerDependencyError("SAM2 inference supports one GPU per worker")
        self._import_sam2_runtime()
        return {
            "sam2": {
                "variant": self.config.variant,
                "weights": str(self.config.weights),
            }
        }

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.config is None or self.context is None:
            raise PredictionContractError("SAM2 worker is not configured")
        request = params.get("request", params)
        if not isinstance(request, Mapping):
            raise PredictionContractError("predict request must be an object")
        data = request.get("data", [])
        if not isinstance(data, list) or not data:
            raise PredictionContractError("predict data must be a non-empty list")
        image_paths = [Path(str(value)).expanduser().resolve() for value in data]
        for image_path in image_paths:
            if not image_path.is_file():
                raise PredictionContractError(f"input image not found: {image_path}")
        bbox_files = _bbox_files(request, len(image_paths))
        self._ensure_model()
        images = [_read_rgb_image(image_path) for image_path in image_paths]
        if bbox_files:
            predictions = self._predict_boxes_batch(images, bbox_files)
        else:
            predictions = [self._predict_automatic(image) for image in images]
        results = [
            _connector_prediction(image_path, masks)
            for image_path, masks in zip(image_paths, predictions)
        ]
        return {"results": results}

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.config is None or self.torch is None or self.device is None:
            raise WorkerDependencyError("SAM2 worker was not initialized")
        try:
            model = self._build_sam2(
                self.config.config_path,
                str(self.config.weights),
                device=self.device,
                apply_postprocessing=True,
            )
        except Exception as error:
            raise WorkerDependencyError(
                f"unable to load SAM2 {self.config.variant} checkpoint "
                f"{self.config.weights}: {error}"
            ) from error
        model.eval()
        self.model = model
        self._automatic_generator = self._automatic_generator_class(
            model,
            points_per_side=self.config.automatic.points_per_side,
            points_per_batch=self.config.automatic.points_per_batch,
            pred_iou_thresh=self.config.automatic.pred_iou_thresh,
            stability_score_thresh=self.config.automatic.stability_score_thresh,
            box_nms_thresh=self.config.automatic.box_nms_thresh,
            crop_n_layers=self.config.automatic.crop_n_layers,
            output_mode="binary_mask",
        )
        self._image_predictor = self._image_predictor_class(model)

    def _predict_automatic(self, image: np.ndarray) -> list[MaskPrediction]:
        if self._automatic_generator is None:
            raise WorkerDependencyError("SAM2 automatic generator was not initialized")
        with self._inference_context():
            generated = self._automatic_generator.generate(image)
        if not isinstance(generated, list):
            raise PredictionContractError("SAM2 automatic generator returned invalid masks")
        masks: list[MaskPrediction] = []
        for generated_mask in generated:
            if not isinstance(generated_mask, Mapping):
                raise PredictionContractError("SAM2 automatic mask must be an object")
            mask = _binary_mask(generated_mask.get("segmentation"))
            score = _finite_score(generated_mask.get("predicted_iou", 0.0), "predicted_iou")
            stability = _finite_score(
                generated_mask.get("stability_score", 0.0), "stability_score"
            )
            masks.append(
                MaskPrediction(
                    mask=mask,
                    bbox=_bbox_from_mask(mask),
                    score=score,
                    category="0",
                    stability_score=stability,
                )
            )
        masks.sort(
            key=lambda prediction: (
                -prediction.score,
                -prediction.stability_score,
                -int(np.count_nonzero(prediction.mask)),
                prediction.bbox["xmin"],
                prediction.bbox["ymin"],
                prediction.bbox["xmax"],
                prediction.bbox["ymax"],
            )
        )
        limit = self.config.automatic.max_masks if self.config is not None else 0
        return masks[:limit] if limit else masks

    def _predict_boxes_batch(
        self,
        images: list[np.ndarray],
        bbox_files: list[Path],
    ) -> list[list[MaskPrediction]]:
        if self._image_predictor is None:
            raise WorkerDependencyError("SAM2 image predictor was not initialized")
        predictions: list[list[MaskPrediction]] = [[] for _ in images]
        active_images: list[np.ndarray] = []
        active_boxes: list[np.ndarray] = []
        active_prompts: list[list[BBoxPrompt]] = []
        active_indexes: list[int] = []
        for index, (image, bbox_file) in enumerate(zip(images, bbox_files)):
            prompts = _read_bbox_file(bbox_file, image.shape[1], image.shape[0])
            if not prompts:
                continue
            active_images.append(image)
            active_boxes.append(
                np.asarray([prompt.bbox for prompt in prompts], dtype=np.float32)
            )
            active_prompts.append(prompts)
            active_indexes.append(index)
        if not active_images:
            return predictions
        with self._inference_context():
            self._image_predictor.set_image_batch(active_images)
            masks_batch, scores_batch, _low_res_masks = (
                self._image_predictor.predict_batch(
                    box_batch=active_boxes,
                    multimask_output=False,
                )
            )
        if len(masks_batch) != len(active_images) or len(scores_batch) != len(
            active_images
        ):
            raise PredictionContractError(
                "SAM2 batch prediction returned an unexpected image count"
            )
        for index, prompts, masks, scores in zip(
            active_indexes,
            active_prompts,
            masks_batch,
            scores_batch,
        ):
            predictions[index] = self._prompt_predictions(prompts, masks, scores)
        return predictions

    @staticmethod
    def _prompt_predictions(
        prompts: list[BBoxPrompt],
        masks: Any,
        scores: Any,
    ) -> list[MaskPrediction]:
        normalized_masks = _prompt_masks(masks, len(prompts))
        normalized_scores = _prompt_scores(scores, len(prompts))
        predictions: list[MaskPrediction] = []
        for prompt, mask, score in zip(prompts, normalized_masks, normalized_scores):
            predictions.append(
                MaskPrediction(
                    mask=mask,
                    bbox=_bbox_from_mask(mask, fallback=prompt.output_bbox),
                    score=score,
                    category=prompt.category,
                    stability_score=0.0,
                )
            )
        return predictions

    def _inference_context(self) -> Any:
        if self.torch is None or self.device is None:
            return nullcontext()
        inference_mode = self.torch.inference_mode()
        if str(self.device).startswith("cuda"):
            return _nested_context(
                inference_mode,
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16),
            )
        return inference_mode

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise WorkerDependencyError("SAM2 requires PyTorch") from error
        return torch

    def _import_sam2_runtime(self) -> None:
        try:
            version = importlib.metadata.version("sam2")
        except importlib.metadata.PackageNotFoundError as error:
            raise WorkerDependencyError(
                "SAM2 requires the external runtime 'sam2==1.1.0'; "
                "install extern/pytorch_workers/sam2/requirements.txt into the worker Python environment"
            ) from error
        if version != SAM2_PACKAGE_VERSION:
            raise WorkerDependencyError(
                f"SAM2 requires sam2=={SAM2_PACKAGE_VERSION}, found {version}"
            )
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise WorkerDependencyError(
                "sam2==1.1.0 does not expose the required SAM2 image inference API"
            ) from error
        self._build_sam2 = build_sam2
        self._automatic_generator_class = SAM2AutomaticMaskGenerator
        self._image_predictor_class = SAM2ImagePredictor


@dataclass(frozen=True)
class BBoxPrompt:
    category: str
    bbox: tuple[float, float, float, float]

    @property
    def output_bbox(self) -> dict[str, float]:
        return {
            "xmin": self.bbox[0],
            "ymin": self.bbox[1],
            "xmax": self.bbox[2],
            "ymax": self.bbox[3],
        }


def _bbox_files(params: Mapping[str, Any], image_count: int) -> list[Path]:
    parameters = params.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    input_parameters = parameters.get("input", {})
    input_parameters = input_parameters if isinstance(input_parameters, Mapping) else {}
    raw_files = input_parameters.get("bbox_files", [])
    if raw_files is None:
        return []
    if not isinstance(raw_files, list):
        raise PredictionContractError("bbox_files must be a list")
    if not raw_files:
        return []
    if len(raw_files) != image_count:
        raise PredictionContractError("bbox_files must contain one file per input image")
    paths = [Path(str(value)).expanduser().resolve() for value in raw_files]
    for path in paths:
        if not path.is_file():
            raise PredictionContractError(f"bbox file not found: {path}")
    return paths


def _read_rgb_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except OSError as error:
        raise PredictionContractError(f"unable to read input image {path}: {error}") from error


def _read_bbox_file(path: Path, width: int, height: int) -> list[BBoxPrompt]:
    prompts: list[BBoxPrompt] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise PredictionContractError(
                f"{path}:{line_number}: expected cls xmin ymin xmax ymax"
            )
        category = fields[0]
        try:
            int(category)
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise PredictionContractError(
                f"{path}:{line_number}: bbox fields must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise PredictionContractError(f"{path}:{line_number}: bbox values must be finite")
        xmin, ymin, xmax, ymax = values
        xmin = min(max(xmin, 0.0), float(width))
        ymin = min(max(ymin, 0.0), float(height))
        xmax = min(max(xmax, 0.0), float(width))
        ymax = min(max(ymax, 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            raise PredictionContractError(
                f"{path}:{line_number}: bbox must have positive clipped area"
            )
        prompts.append(BBoxPrompt(category=category, bbox=(xmin, ymin, xmax, ymax)))
    return prompts


def _binary_mask(value: Any) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise PredictionContractError("SAM2 mask must be a two-dimensional array")
    return np.asarray(mask > 0, dtype=np.uint8)


def _bbox_from_mask(
    mask: np.ndarray, *, fallback: dict[str, float] | None = None
) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        if fallback is not None:
            return fallback
        return {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0}
    return {
        "xmin": float(xs.min()),
        "ymin": float(ys.min()),
        "xmax": float(xs.max() + 1),
        "ymax": float(ys.max() + 1),
    }


def _finite_score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise PredictionContractError(f"SAM2 {name} must be numeric") from error
    if not math.isfinite(score):
        raise PredictionContractError(f"SAM2 {name} must be finite")
    return score


def _prompt_masks(value: Any, expected_count: int) -> list[np.ndarray]:
    masks = np.asarray(value)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    elif masks.ndim == 4:
        if masks.shape[1] != 1:
            raise PredictionContractError("SAM2 prompt prediction returned multiple masks")
        masks = masks[:, 0, :, :]
    if masks.ndim != 3 or masks.shape[0] != expected_count:
        raise PredictionContractError(
            "SAM2 prompt prediction did not return one mask per bounding box"
        )
    return [_binary_mask(mask) for mask in masks]


def _prompt_scores(value: Any, expected_count: int) -> list[float]:
    scores = np.asarray(value).reshape(-1)
    if scores.size != expected_count:
        raise PredictionContractError(
            "SAM2 prompt prediction did not return one score per bounding box"
        )
    return [_finite_score(score, "predicted_iou") for score in scores]


def _connector_prediction(
    image_path: Path, predictions: list[MaskPrediction]
) -> dict[str, Any]:
    return {
        "uri": str(image_path),
        "loss": 0.0,
        "probs": [prediction.score for prediction in predictions],
        "cats": [prediction.category for prediction in predictions],
        "bboxes": [prediction.bbox for prediction in predictions],
        "masks": [encode_coco_rle(prediction.mask) for prediction in predictions],
    }


class _nested_context:
    def __init__(self, first: Any, second: Any) -> None:
        self.first = first
        self.second = second

    def __enter__(self) -> None:
        self.first.__enter__()
        self.second.__enter__()

    def __exit__(self, *args: Any) -> None:
        self.second.__exit__(*args)
        self.first.__exit__(*args)
