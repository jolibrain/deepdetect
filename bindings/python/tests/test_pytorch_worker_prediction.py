import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from deepdetect.pytorch_worker.prediction import (
    prediction_batch,
    semantic_prediction_results,
)
from deepdetect.pytorch_worker.sdk import (
    PredictionContractError,
    WorkerContext,
    validate_prediction_result,
)


def batch(n=2):
    return {
        "tensor_batch": {
            "kind": "tensor_batch",
            "inputs": [
                {
                    "kind": "tensor_ref",
                    "device": "cpu",
                    "dtype": "float32",
                    "shape": [n, 3, 2, 2],
                    "storage": {
                        "type": "inline_test_stub",
                        "offset": 0,
                        "nbytes": n * 48,
                        "values": (
                            [0.8] * 4 + [0.4] * 4 + [0.2] * 4
                            if n == 1
                            else ([0.8] * 4 + [0.4] * 4 + [0.2] * 4) * n
                        ),
                    },
                }
            ],
            "meta": {
                "paths": ["nonexistent.png"] * n,
                "sample_ids": list(range(n)),
                "original_widths": [6] * n,
                "original_heights": [4] * n,
                "preprocessed_widths": [2] * n,
                "preprocessed_heights": [2] * n,
            },
        }
    }


def fixture_worker(name):
    source_root = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "prediction_fixture",
        root / "tests/pytorch_prediction_worker.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)()


def test_detection_consumes_tensor_without_reader_and_rescales_once(tmp_path):
    worker = fixture_worker("DeepDetectWorker")
    worker.configure(WorkerContext(str(tmp_path), {"nclasses": 2}, {}))
    result = worker.predict({"request": batch()})
    validate_prediction_result(result)
    assert len(result["results"]) == 2
    assert result["results"][0]["bboxes"] == [
        {"xmin": 0.0, "ymin": 0.0, "xmax": 6.0, "ymax": 4.0}
    ]
    assert '"shape": [2, 3, 2, 2]' in (tmp_path / "forwards.jsonl").read_text()


def test_semantic_nearest_and_bilinear():
    with prediction_batch(batch(1), torch) as (_, metas):
        result = semantic_prediction_results(
            torch.tensor([[[0, 1], [2, 3]]]),
            metas,
            torch=torch,
            confidences={"1": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])},
        )
    validate_prediction_result(result)
    item = result["results"][0]
    assert item["vals"][:6] == [0, 0, 0, 1, 1, 1]
    assert item["vals"][-6:] == [2, 2, 2, 3, 3, 3]
    assert 0 < item["confidences"]["1"][2] < 1
    bad = copy.deepcopy(result)
    bad["results"][0]["vals"].pop()
    with pytest.raises(PredictionContractError):
        validate_prediction_result(bad)


def test_prediction_metadata_rejected_before_mapping():
    request = batch()
    request["tensor_batch"]["meta"]["paths"].pop()
    with pytest.raises(PredictionContractError):
        with prediction_batch(request, torch):
            pass


def test_shared_memory_closed_on_model_failure(tmp_path):
    request = batch(1)
    storage = request["tensor_batch"]["inputs"][0]["storage"]
    path = tmp_path / "input.bin"
    import numpy as np

    path.write_bytes(np.array(storage.pop("values"), dtype=np.float32).tobytes())
    storage.update(type="shared_memory", name=str(path))
    with pytest.raises(RuntimeError, match="model failed"):
        with prediction_batch(request, torch) as (tensor, metas):
            assert tensor.shape == (1, 3, 2, 2)
            raise RuntimeError("model failed")
    # /proc tracks actual live mappings even when the pathname is still present.
    assert str(path) not in Path("/proc/self/maps").read_text()
