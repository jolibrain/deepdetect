from deepdetect.pytorch_worker.sdk import DeepDetectWorkerBase


class DeepDetectWorker(DeepDetectWorkerBase):
    def predict(self, params):
        return {
            "output_format": "semantic_segmentation_v1",
            "nclasses": 2,
            "results": [
                {"uri": uri, "imgsize": {"width": 3, "height": 2},
                 "vals": [0, 1, 0, 1, 0, 1],
                 "confidences": {"best": [0.75] * 6, "0": [1, 0, 1, 0, 1, 0]}}
                for uri in params["request"]["data"]
            ],
        }
