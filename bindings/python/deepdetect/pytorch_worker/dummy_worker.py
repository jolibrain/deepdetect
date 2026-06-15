from __future__ import annotations

import time
from typing import Any

from .sdk import Cancellation, DeepDetectWorkerBase, WorkerContext, WorkerReporter


class DeepDetectWorker(DeepDetectWorkerBase):
    def __init__(self) -> None:
        super().__init__()
        self.torch_version = ""

    def configure(self, context: WorkerContext) -> dict[str, Any]:
        import torch

        super().configure(context)
        self.torch_version = str(getattr(torch, "__version__", "unknown"))
        return {
            "worker": "dummy",
            "torch_version": self.torch_version,
        }

    def train(
        self,
        params: dict[str, Any],
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
    ) -> dict[str, Any]:
        request = params.get("request", {})
        context_mllib = dict(self.context.mllib) if self.context is not None else {}
        request_mllib = request.get("parameters", {}).get("mllib", {})
        if not isinstance(request_mllib, dict):
            request_mllib = {}
        mllib = {**context_mllib, **request_mllib}
        solver = mllib.get("solver", {})
        iterations = int(solver.get("iterations", 5))
        iterations = max(1, min(iterations, 1000000))
        learning_rate = float(solver.get("base_lr", 0.001))

        for iteration in range(1, iterations + 1):
            if cancellation.requested:
                reporter.status(
                    phase="cancelled",
                    iteration=iteration - 1,
                    iterations=iterations,
                    test_active=0,
                )
                return {"status": "cancelled", "iteration": iteration - 1}
            loss = 1.0 / float(iteration)
            status_payload = {
                "phase": "train",
                "iteration": iteration,
                "iterations": iterations,
                "test_active": 0,
                "elapsed_time_ms": iteration,
                "remain_time": max(0, iterations - iteration),
            }
            if mllib.get("emit_test_predictions") and iteration == iterations:
                status_payload["test_predictions"] = {
                    "test0": {
                        "iteration": iteration,
                        "samples": [
                            {
                                "index": 0,
                                "imgsize": {"width": 64, "height": 64},
                                "classes": [
                                    {
                                        "cat": "dummy",
                                        "prob": 1.0,
                                        "bbox": {
                                            "xmin": 0.0,
                                            "ymin": 0.0,
                                            "xmax": 1.0,
                                            "ymax": 1.0,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                }
            reporter.status(**status_payload)
            reporter.metric("iteration", iteration, iteration=iteration)
            reporter.metric("train_loss", loss, iteration=iteration)
            reporter.metric("learning_rate", learning_rate, iteration=iteration)
            time.sleep(0.001)
        return {"status": "finished", "iteration": iterations}

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        request = params.get("request", {})
        data = request.get("data", []) or ["dummy"]
        output = request.get("parameters", {}).get("output", {})
        include_bbox = bool(output.get("bbox"))

        results = []
        for uri in data:
            result: dict[str, Any] = {
                "uri": str(uri),
                "loss": 0.0,
                "probs": [1.0],
                "cats": ["dummy"],
            }
            if include_bbox:
                result["bboxes"] = [
                    {
                        "xmin": 0.0,
                        "ymin": 0.0,
                        "xmax": 1.0,
                        "ymax": 1.0,
                    }
                ]
            results.append(result)
        return {"results": results}
