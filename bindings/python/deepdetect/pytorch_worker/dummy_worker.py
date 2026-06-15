from __future__ import annotations

import time
from typing import Any


class DeepDetectWorker:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.torch_version = ""

    def configure(self, context: dict[str, Any]) -> dict[str, Any]:
        import torch

        self.torch_version = str(getattr(torch, "__version__", "unknown"))
        self.context = dict(context)
        return {
            "worker": "dummy",
            "torch_version": self.torch_version,
        }

    def train(
        self,
        params: dict[str, Any],
        *,
        reporter: Any,
        cancellation: Any,
    ) -> dict[str, Any]:
        request = params.get("request", {})
        mllib = request.get("parameters", {}).get("mllib", {})
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
            reporter.status(
                phase="train",
                iteration=iteration,
                iterations=iterations,
                test_active=0,
                elapsed_time_ms=iteration,
                remain_time=max(0, iterations - iteration),
            )
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
