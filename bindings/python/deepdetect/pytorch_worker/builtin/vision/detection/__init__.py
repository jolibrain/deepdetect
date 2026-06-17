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
from .training import (
    DetectionCheckpointManager,
    DetectionProgressReporter,
    DetectionRepositoryContractWriter,
    DetectionTrainOptions,
    DetectionTrainRequest,
)

__all__ = [
    "DetectionCheckpointManager",
    "DetectionEvalBox",
    "DetectionListDataset",
    "DetectionProgressReporter",
    "DetectionRepositoryContractWriter",
    "DetectionSample",
    "DetectionTrainingWorkerBase",
    "DetectionTrainOptions",
    "DetectionTrainRequest",
    "ReferenceTorchDetectorWorker",
    "detection_map_metrics",
    "detection_metric_thresholds",
    "read_detection_list",
    "report_detection_metrics",
]
