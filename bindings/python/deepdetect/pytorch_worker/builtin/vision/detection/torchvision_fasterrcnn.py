from __future__ import annotations

from typing import Any

from ....sdk import WorkerDependencyError

from .base import DetectionTrainingWorkerBase
from .common import *  # noqa: F403
from .common import debug as detection_debug


class DeepDetectWorker(DetectionTrainingWorkerBase):
    worker_name = "torchvision-detector"
    debug_name = "torchvision-detector"

    def import_backend(self) -> tuple[Any, Any]:
        return import_torchvision()

    def backend_versions(self, *backend: Any) -> dict[str, Any]:
        torch, torchvision = backend
        return {
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),
        }

    def create_model(self, nclasses: int, *backend: Any) -> Any:
        _torch, torchvision = backend
        return create_model(nclasses, torchvision=torchvision)


def import_torchvision() -> tuple[Any, Any]:
    detection_debug(
        "import_torchvision: importing torch/torchvision/nms",
        tag="torchvision-detector",
    )
    try:
        import torch
        import torchvision
        from torchvision.ops import nms as _nms
    except Exception as error:
        raise WorkerDependencyError(
            "torchvision could not be imported with working custom ops; "
            "install a torchvision wheel matching the active torch build"
        ) from error
    detection_debug(
        "import_torchvision: torch=%s torchvision=%s"
        % (
            getattr(torch, "__version__", "unknown"),
            getattr(torchvision, "__version__", "unknown"),
        ),
        tag="torchvision-detector",
    )
    return torch, torchvision


def create_model(nclasses: int, *, torchvision: Any | None = None) -> Any:
    if torchvision is None:
        _torch, torchvision = import_torchvision()
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=nclasses,
    )
