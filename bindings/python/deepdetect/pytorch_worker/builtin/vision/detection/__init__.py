"""Object detection workers for the managed PyTorch backend."""

from .base import DetectionTrainingWorkerBase
from .common import (
    DetectionEvalBox,
    DetectionListDataset,
    DetectionSample,
    detection_map_metrics,
    detection_metric_thresholds,
    read_detection_list,
    report_detection_metrics,
)
from .reference_torch_detector import DeepDetectWorker as ReferenceTorchDetectorWorker

__all__ = [
    "DetectionEvalBox",
    "DetectionListDataset",
    "DetectionSample",
    "DetectionTrainingWorkerBase",
    "ReferenceTorchDetectorWorker",
    "detection_map_metrics",
    "detection_metric_thresholds",
    "read_detection_list",
    "report_detection_metrics",
]
