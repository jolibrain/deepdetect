"""Deterministic real-tensor workers used by native and Docker HTTP tests."""

import json
from pathlib import Path

import torch

from deepdetect.pytorch_worker.builtin.vision.detection.base import (
    DetectionTrainingWorkerBase,
)
from deepdetect.pytorch_worker.prediction import (
    prediction_batch,
    semantic_prediction_results,
)


class Model(torch.nn.Module):
    def __init__(self, repository):
        super().__init__()
        self.repository = Path(repository)

    def forward(self, batch):
        with (self.repository / "forwards.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {"shape": list(batch.shape), "pixels": batch[:, :, 0, 0].tolist()}
                )
                + "\n"
            )
        return batch[:, 0] > batch[:, 2]


class DeepDetectWorker(DetectionTrainingWorkerBase):
    def import_backend(self):
        return (torch,)

    def create_model(self, nclasses, *backend):
        return Model(self.context.repository)

    def load_model_for_prediction(self, checkpoint_manager, model):
        return None

    def predict_batch(self, model, images):
        batch = torch.stack(images)
        model(batch)
        return [
            dict(
                boxes=batch.new_tensor([[0, 0, image.shape[-1], image.shape[-2]]]),
                scores=batch.new_tensor([0.9]),
                labels=torch.tensor([1], device=batch.device),
            )
            for image in images
        ]


class SemanticWorker(DeepDetectWorker):
    def predict(self, params):
        with prediction_batch(params["request"], torch) as (batch, metas):
            if self.model is None:
                self.model = self.create_model(self.nclasses)
            classes = self.model(batch.to(self.device)).long()
            return semantic_prediction_results(
                classes, metas, torch=torch, confidences={"1": batch[:, 0]}
            )


class BrokenWorker(DeepDetectWorker):
    def predict(self, params):
        result = super().predict(params)
        if params["request"]["tensor_batch"]["meta"]["sample_ids"][0] >= 2:
            result["results"] = []
        return result


class MisassociatedWorker(DeepDetectWorker):
    def predict(self, params):
        result = super().predict(params)
        result["results"][0]["uri"] = "wrong-sample"
        return result


class TimeoutWorker(DeepDetectWorker):
    def predict(self, params):
        import time

        with prediction_batch(params["request"], torch):
            time.sleep(12)
        return {"results": []}
