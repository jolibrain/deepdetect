from __future__ import annotations

from typing import Any

from ....sdk import WorkerDependencyError

from .base import DetectionTrainingWorkerBase
from .common import debug as detection_debug


class DeepDetectWorker(DetectionTrainingWorkerBase):
    worker_name = "reference-torch-detector"
    debug_name = "reference-torch-detector"

    def import_backend(self) -> tuple[Any, ...]:
        return import_torch()

    def create_model(self, nclasses: int, *backend: Any) -> Any:
        (torch,) = backend
        return TinyTorchDetector(torch, nclasses)

    def training_losses(
        self,
        model: Any,
        images: list[Any],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return model.losses(images, targets)

    def predict_batch(self, model: Any, images: list[Any]) -> list[dict[str, Any]]:
        return model.predict(images)


def import_torch() -> tuple[Any, ...]:
    detection_debug("import_torch: importing torch", tag="reference-torch-detector")
    try:
        import torch
    except Exception as error:
        raise WorkerDependencyError("torch could not be imported") from error
    detection_debug(
        "import_torch: torch=%s" % getattr(torch, "__version__", "unknown"),
        tag="reference-torch-detector",
    )
    return (torch,)


class TinyTorchDetector:
    def __init__(self, torch: Any, nclasses: int) -> None:
        self.torch = torch
        self.nclasses = int(nclasses)
        nn = torch.nn
        self.module = nn.Module()
        self.module.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.module.class_head = nn.Linear(8, self.nclasses)
        self.module.box_head = nn.Linear(8, 4)

    def to(self, device: Any) -> "TinyTorchDetector":
        self.module.to(device)
        return self

    def train(self, mode: bool = True) -> "TinyTorchDetector":
        self.module.train(mode)
        return self

    def eval(self) -> "TinyTorchDetector":
        self.module.eval()
        return self

    def parameters(self) -> Any:
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any], strict: bool = False) -> Any:
        return self.module.load_state_dict(state, strict=strict)

    def losses(
        self,
        images: list[Any],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        torch = self.torch
        functional = torch.nn.functional
        features = self._features(images)
        logits = self.module.class_head(features)
        box_params = self.module.box_head(features)
        labels, boxes, positive = self._target_batch(targets, images)
        class_loss = functional.cross_entropy(logits, labels)
        predicted_boxes = self._boxes_from_params(box_params, images)
        if bool(positive.any().item()):
            box_loss = functional.smooth_l1_loss(
                predicted_boxes[positive],
                boxes[positive],
            )
        else:
            box_loss = box_params.sum() * 0.0
        return {
            "loss_classifier": class_loss,
            "loss_box_reg": box_loss,
        }

    def predict(self, images: list[Any]) -> list[dict[str, Any]]:
        torch = self.torch
        features = self._features(images)
        logits = self.module.class_head(features)
        probabilities = torch.softmax(logits, dim=1)
        if self.nclasses > 1:
            foreground_scores, foreground_labels = probabilities[:, 1:].max(dim=1)
            labels = foreground_labels + 1
        else:
            foreground_scores = probabilities[:, 0]
            labels = torch.zeros_like(foreground_scores, dtype=torch.int64)
        boxes = self._boxes_from_params(self.module.box_head(features), images)
        return [
            {
                "boxes": boxes[index].reshape(1, 4),
                "scores": foreground_scores[index].reshape(1),
                "labels": labels[index].reshape(1).to(dtype=torch.int64),
            }
            for index in range(len(images))
        ]

    def _features(self, images: list[Any]) -> Any:
        torch = self.torch
        rows = []
        for image in images:
            row = self.module.features(image.unsqueeze(0)).flatten(1)
            rows.append(row)
        return torch.cat(rows, dim=0)

    def _target_batch(
        self,
        targets: list[dict[str, Any]],
        images: list[Any],
    ) -> tuple[Any, Any, Any]:
        torch = self.torch
        labels = []
        boxes = []
        positive = []
        for target, image in zip(targets, images):
            target_labels = target.get("labels")
            target_boxes = target.get("boxes")
            if target_labels is not None and int(target_labels.numel()) > 0:
                labels.append(target_labels[0].clamp(0, max(0, self.nclasses - 1)))
                boxes.append(target_boxes[0])
                positive.append(True)
            else:
                labels.append(torch.tensor(0, dtype=torch.int64, device=image.device))
                boxes.append(torch.zeros((4,), dtype=image.dtype, device=image.device))
                positive.append(False)
        return (
            torch.stack(labels).to(dtype=torch.int64),
            torch.stack(boxes).to(dtype=images[0].dtype),
            torch.tensor(positive, dtype=torch.bool, device=images[0].device),
        )

    def _boxes_from_params(self, box_params: Any, images: list[Any]) -> Any:
        torch = self.torch
        boxes = []
        normalized = torch.sigmoid(box_params)
        for row, image in zip(normalized, images):
            height = float(image.shape[-2])
            width = float(image.shape[-1])
            x0 = row[0] * width
            y0 = row[1] * height
            x1 = x0 + (row[2] * (width - x0)).clamp(min=1.0)
            y1 = y0 + (row[3] * (height - y0)).clamp(min=1.0)
            boxes.append(
                torch.stack([x0, y0, x1.clamp(max=width), y1.clamp(max=height)])
            )
        return torch.stack(boxes)
