from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from deepdetect.pytorch_worker.sdk import WorkerDependencyError


SAM2_VARIANTS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


@dataclass(frozen=True)
class AutomaticMaskConfig:
    points_per_side: int
    points_per_batch: int
    pred_iou_thresh: float
    stability_score_thresh: float
    box_nms_thresh: float
    crop_n_layers: int
    max_masks: int


@dataclass(frozen=True)
class Sam2WorkerConfig:
    variant: str
    config_path: str
    weights: Path
    automatic: AutomaticMaskConfig


def worker_config_from_mllib(mllib: Mapping[str, Any]) -> Sam2WorkerConfig:
    options = mllib.get("sam2", {})
    if not isinstance(options, Mapping):
        raise WorkerDependencyError("sam2 configuration must be an object")
    variant = str(options.get("variant", "tiny"))
    config_path = SAM2_VARIANTS.get(variant)
    if config_path is None:
        raise WorkerDependencyError(
            f"unsupported SAM2 variant {variant!r}; expected one of "
            f"{', '.join(sorted(SAM2_VARIANTS))}"
        )
    raw_weights = mllib.get("weights") or options.get("weights")
    if not raw_weights:
        raise WorkerDependencyError("SAM2 requires mllib.weights")
    weights = Path(str(raw_weights)).expanduser().resolve()
    if not weights.is_file():
        raise WorkerDependencyError(f"SAM2 checkpoint not found: {weights}")
    automatic = options.get("automatic", {})
    if not isinstance(automatic, Mapping):
        raise WorkerDependencyError("sam2.automatic must be an object")
    return Sam2WorkerConfig(
        variant=variant,
        config_path=config_path,
        weights=weights,
        automatic=AutomaticMaskConfig(
            points_per_side=_positive_int(
                automatic.get("points_per_side", 32), "sam2.automatic.points_per_side"
            ),
            points_per_batch=_positive_int(
                automatic.get("points_per_batch", 64), "sam2.automatic.points_per_batch"
            ),
            pred_iou_thresh=_probability(
                automatic.get("pred_iou_thresh", 0.8),
                "sam2.automatic.pred_iou_thresh",
            ),
            stability_score_thresh=_probability(
                automatic.get("stability_score_thresh", 0.95),
                "sam2.automatic.stability_score_thresh",
            ),
            box_nms_thresh=_probability(
                automatic.get("box_nms_thresh", 0.7),
                "sam2.automatic.box_nms_thresh",
            ),
            crop_n_layers=_non_negative_int(
                automatic.get("crop_n_layers", 0), "sam2.automatic.crop_n_layers"
            ),
            max_masks=_non_negative_int(
                automatic.get("max_masks", 0), "sam2.automatic.max_masks"
            ),
        ),
    )


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise WorkerDependencyError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise WorkerDependencyError(f"{name} must be non-negative")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise WorkerDependencyError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise WorkerDependencyError(f"{name} must be an integer") from error
    if result != value:
        raise WorkerDependencyError(f"{name} must be an integer")
    return result


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise WorkerDependencyError(f"{name} must be a number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise WorkerDependencyError(f"{name} must be between 0 and 1")
    return result
