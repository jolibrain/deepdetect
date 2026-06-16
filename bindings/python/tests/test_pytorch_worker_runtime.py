import json
import threading
import time
import sys
import types

import pytest
from PIL import Image

from deepdetect.pytorch_worker.protocol import read_frame, write_frame
from deepdetect.pytorch_worker.runtime import WorkerRuntime
from deepdetect.pytorch_worker.sdk import (
    Cancellation,
    DatasetContractError,
    DeepDetectWorkerBase,
    WorkerContext,
    WorkerReporter,
)
from deepdetect.pytorch_worker.builtin.vision.detection.base import (
    DetectionTrainingWorkerBase,
)
from deepdetect.pytorch_worker.builtin.vision.detection.common import (
    DetectionListDataset,
)
from deepdetect.pytorch_worker.builtin.vision.detection.torchvision_fasterrcnn import (
    DetectionEvalBox,
    DeepDetectWorker,
    detection_map_metrics,
    detection_metric_thresholds,
    read_detection_list,
    report_detection_metrics,
)
from deepdetect.pytorch_worker.templates.train_worker import (
    DeepDetectWorker as CompatibilityDeepDetectWorker,
)
from deepdetect.pytorch_worker.builtin.vision.detection.reference_torch_detector import (
    DeepDetectWorker as ReferenceTorchDetectorWorker,
)


class MemorySocket:
    def __init__(self):
        self.peer = None
        self.incoming = bytearray()
        self.condition = threading.Condition()
        self.timeout = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendall(self, data):
        with self.peer.condition:
            self.peer.incoming.extend(data)
            self.peer.condition.notify_all()

    def recv(self, size):
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        with self.condition:
            while not self.incoming and not self.closed:
                if deadline is None:
                    self.condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("memory socket timeout")
                    self.condition.wait(remaining)
            if not self.incoming:
                return b""
            chunk = bytes(self.incoming[:size])
            del self.incoming[:size]
            return chunk

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def socket_pair():
    left = MemorySocket()
    right = MemorySocket()
    left.peer = right
    right.peer = left
    return left, right


def request(sock, message_id, method, params=None):
    write_frame(sock, {"id": message_id, "method": method, "params": params or {}})


def read_until(sock, *, event=None, message_id=None, timeout=5.0):
    sock.settimeout(timeout)
    while True:
        message = read_frame(sock)
        if event is not None and message.get("event") == event:
            return message
        if message_id is not None and message.get("id") == message_id:
            return message


def install_worker_module(name, worker_class):
    module = types.ModuleType(name)
    module.DeepDetectWorker = worker_class
    sys.modules[name] = module
    return module


def test_protocol_roundtrip():
    left, right = socket_pair()
    try:
        write_frame(left, {"id": 1, "result": {"ok": True}})
        assert read_frame(right) == {"id": 1, "result": {"ok": True}}
    finally:
        left.close()
        right.close()


def test_dummy_worker_reports_metrics_and_predicts():
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "hello", {"protocol_version": 1})
        hello = read_until(server, message_id=1)
        assert hello["result"]["capabilities"]["train"] is True

        request(server, 2, "configure", {"repository": "/tmp/repo", "mllib": {}})
        configured = read_until(server, message_id=2)
        assert configured["result"]["worker"] == "dummy"
        assert configured["result"]["torch_version"]

        request(
            server,
            3,
            "train_start",
            {
                "request": {
                    "parameters": {
                        "mllib": {
                            "solver": {"iterations": 2, "base_lr": 0.01}
                        }
                    }
                }
            },
        )
        assert read_until(server, message_id=3)["result"]["status"] == "started"
        metric = read_until(server, event="metric")
        assert metric["payload"]["name"] in {"iteration", "train_loss"}
        result = read_until(server, event="train_result")
        assert result["payload"]["status"] == "finished"

        request(
            server,
            4,
            "predict",
            {
                "request": {
                    "data": ["image.jpg"],
                    "parameters": {"output": {"bbox": True}},
                }
            },
        )
        prediction = read_until(server, message_id=4)
        assert prediction["result"]["results"][0]["uri"] == "image.jpg"
        assert prediction["result"]["results"][0]["bboxes"][0]["xmax"] == 1.0
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)


def test_torchvision_worker_detection_list_resolves_relative_paths(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    bbox = tmp_path / "target.txt"
    bbox.write_text("1 0 0 4 4\n", encoding="utf-8")
    data = tmp_path / "train.txt"
    data.write_text("image.jpg target.txt\n", encoding="utf-8")

    samples = read_detection_list(data, nclasses=2)

    assert len(samples) == 1
    assert samples[0].index == 0
    assert samples[0].image == image.resolve()
    assert samples[0].target == bbox.resolve()


def test_detection_list_dataset_defers_file_and_target_validation(tmp_path):
    data = tmp_path / "train.txt"
    data.write_text("missing.jpg missing.txt\n", encoding="utf-8")

    dataset = DetectionListDataset(data, nclasses=2, torch=object())

    assert len(dataset) == 1
    assert dataset.samples[0].image == tmp_path / "missing.jpg"
    assert dataset.samples[0].target == tmp_path / "missing.txt"


def test_detection_base_configure_uses_adapter_metadata(tmp_path):
    class FakeTorch:
        __version__ = "fake-torch"

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return name

    class FakeDetectionWorker(DetectionTrainingWorkerBase):
        worker_name = "fake-detector"

        def import_backend(self):
            return (FakeTorch(),)

        def create_model(self, nclasses, *backend):
            raise AssertionError("configure should not create the model")

    context = WorkerContext(
        repository=str(tmp_path),
        mllib={"nclasses": 3},
        raw={},
    )

    result = FakeDetectionWorker().configure(context)

    assert result == {
        "worker": "fake-detector",
        "task": "detection",
        "nclasses": 3,
        "device": "cpu",
        "torch_version": "fake-torch",
    }


def write_detection_list(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (16, 12), color="white").save(image)
    bbox = tmp_path / "target.txt"
    bbox.write_text("1 1 2 8 9\n", encoding="utf-8")
    data = tmp_path / "train.txt"
    data.write_text("image.jpg target.txt\n", encoding="utf-8")
    return image, data


def test_reference_torch_detector_configures_without_torchvision(tmp_path):
    pytest.importorskip("torch")
    worker = ReferenceTorchDetectorWorker()
    result = worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={"nclasses": 2},
            raw={},
        )
    )

    assert result["worker"] == "reference-torch-detector"
    assert result["task"] == "detection"
    assert result["nclasses"] == 2
    assert result["device"] == "cpu"
    assert result["torch_version"]


def test_reference_torch_detector_trains_one_cpu_iteration(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    _image, data = write_detection_list(tmp_path)
    worker = ReferenceTorchDetectorWorker()
    worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={"nclasses": 2},
            raw={},
        )
    )
    events = []
    reporter = WorkerReporter(lambda event, payload: events.append((event, payload)))

    result = worker.train(
        {
            "request": {
                "data": [str(data), str(data)],
                "parameters": {
                    "mllib": {
                        "solver": {
                            "iterations": 1,
                            "test_interval": 1,
                            "base_lr": 0.001,
                        },
                        "net": {"batch_size": 1},
                    },
                    "output": {
                        "measure": ["map-50"],
                        "test_predictions": {"sample_count": 1},
                    },
                },
            }
        },
        reporter=reporter,
        cancellation=Cancellation(),
    )

    assert result["status"] == "finished"
    metric_names = {payload["name"] for event, payload in events if event == "metric"}
    assert {"train_loss", "loss_classifier", "loss_box_reg", "map_test0"} <= metric_names
    worker_config = json.loads(
        (tmp_path / "pytorch_worker_config.json").read_text(encoding="utf-8")
    )
    assert worker_config["worker"] == "reference-torch-detector"
    assert worker_config["task"] == "detection"
    assert worker_config["train_mllib"]["solver"]["iterations"] == 1
    manifest = json.loads(
        (tmp_path / "connector_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["boundary"] == "path-backed"
    assert manifest["nclasses"] == 2
    assert manifest["train"]["path"] == str(data.resolve())
    assert manifest["train"]["samples"] == 1
    assert manifest["tests"] == [
        {"index": 0, "path": str(data.resolve()), "samples": 1}
    ]
    class_mapping = json.loads(
        (tmp_path / "class_mapping.json").read_text(encoding="utf-8")
    )
    assert class_mapping == {"0": "background", "1": "1"}
    assert (tmp_path / "checkpoint-1.pt").is_file()
    assert (tmp_path / "checkpoint-latest.pt").is_file()
    assert (tmp_path / "solver-1.pt").is_file()
    assert (tmp_path / "solver-latest.pt").is_file()


def test_reference_torch_detector_predicts_detection_schema(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    image, _data = write_detection_list(tmp_path)
    worker = ReferenceTorchDetectorWorker()
    worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={"nclasses": 2},
            raw={},
        )
    )

    result = worker.predict(
        {
            "request": {
                "data": [str(image)],
                "parameters": {"output": {"bbox": True}},
            }
        }
    )

    prediction = result["results"][0]
    assert prediction["uri"] == str(image.resolve())
    assert prediction["probs"][0] >= 0.0
    assert prediction["cats"] == ["1"]
    bbox = prediction["bboxes"][0]
    assert 0.0 <= bbox["xmin"] < bbox["xmax"] <= 16.0
    assert 0.0 <= bbox["ymin"] < bbox["ymax"] <= 12.0


def test_runtime_loads_reference_detector_from_mllib_module(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    image, data = write_detection_list(tmp_path)
    module_name = (
        "deepdetect.pytorch_worker.builtin.vision.detection.reference_torch_detector"
    )
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(
            server,
            1,
            "configure",
            {
                "repository": str(tmp_path),
                "mllib": {
                    "module": module_name,
                    "class": "DeepDetectWorker",
                    "nclasses": 2,
                },
            },
        )
        configured = read_until(server, message_id=1)
        assert configured["result"]["worker"] == "reference-torch-detector"
        assert configured["result"]["task"] == "detection"

        request(
            server,
            2,
            "train_start",
            {
                "request": {
                    "data": [str(data), str(data)],
                    "parameters": {
                        "mllib": {
                            "solver": {
                                "iterations": 1,
                                "test_interval": 1,
                                "base_lr": 0.001,
                            },
                            "net": {"batch_size": 1},
                        },
                        "output": {
                            "measure": ["map-50"],
                            "test_predictions": {"sample_count": 1},
                        },
                    },
                }
            },
        )
        assert read_until(server, message_id=2)["result"]["status"] == "started"
        metrics = set()
        saw_test_predictions = False
        while True:
            message = read_frame(server)
            if message.get("event") == "metric":
                metrics.add(message["payload"]["name"])
            elif message.get("event") == "status":
                saw_test_predictions = saw_test_predictions or (
                    "test_predictions" in message["payload"]
                )
            elif message.get("event") == "failure":
                raise AssertionError(message["payload"])
            elif message.get("event") == "train_result":
                assert message["payload"]["status"] == "finished"
                break
        assert {"train_loss", "loss_classifier", "loss_box_reg", "map_test0"} <= metrics
        assert saw_test_predictions
        manifest = json.loads(
            (tmp_path / "connector_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["train"]["samples"] == 1
        assert manifest["tests"][0]["samples"] == 1
        assert json.loads(
            (tmp_path / "pytorch_worker_config.json").read_text(encoding="utf-8")
        )["worker"] == "reference-torch-detector"

        request(
            server,
            3,
            "predict",
            {
                "request": {
                    "data": [str(image)],
                    "parameters": {"output": {"bbox": True}},
                }
            },
        )
        prediction = read_until(server, message_id=3)
        assert prediction["result"]["results"][0]["uri"] == str(image.resolve())
        assert prediction["result"]["results"][0]["bboxes"]
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)


def test_torchvision_worker_old_template_import_path_is_compatible():
    assert CompatibilityDeepDetectWorker is DeepDetectWorker


def test_torchvision_worker_detection_list_rejects_invalid_class(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    bbox = tmp_path / "target.txt"
    bbox.write_text("2 0 0 4 4\n", encoding="utf-8")
    data = tmp_path / "train.txt"
    data.write_text("image.jpg target.txt\n", encoding="utf-8")

    try:
        read_detection_list(data, nclasses=2)
    except DatasetContractError as error:
        assert "invalid class" in str(error)
    else:
        raise AssertionError("invalid class was accepted")


def test_torchvision_detection_metric_thresholds_use_native_names():
    thresholds = detection_metric_thresholds(
        {"measure": ["map", "map-05", "map-50", "map-90"]}
    )

    assert thresholds == {
        "map-05": 0.05,
        "map-50": 0.5,
        "map-90": 0.9,
    }


def test_torchvision_detection_map_metrics_use_iou_thresholds():
    targets = [DetectionEvalBox(0, 1, (0.0, 0.0, 10.0, 10.0))]
    predictions = [DetectionEvalBox(0, 1, (5.0, 5.0, 15.0, 15.0), 0.9)]

    metrics = detection_map_metrics(predictions, targets)

    assert metrics["map-05"] == 1.0
    assert metrics["map-50"] == 0.0
    assert metrics["map-90"] == 0.0
    assert abs(metrics["map"] - (1.0 / 3.0)) < 1e-12


def test_torchvision_detection_map_metrics_penalize_early_false_positive():
    targets = [DetectionEvalBox(0, 1, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        DetectionEvalBox(0, 1, (20.0, 20.0, 30.0, 30.0), 0.9),
        DetectionEvalBox(0, 1, (0.0, 0.0, 10.0, 10.0), 0.8),
    ]

    metrics = detection_map_metrics(predictions, targets, {"map-50": 0.5})

    assert metrics == {"map": 0.5, "map-50": 0.5}


def test_torchvision_detection_map_metrics_without_targets_are_zero():
    predictions = [DetectionEvalBox(0, 1, (0.0, 0.0, 10.0, 10.0), 0.9)]

    metrics = detection_map_metrics(predictions, [])

    assert metrics == {
        "map": 0.0,
        "map-05": 0.0,
        "map-50": 0.0,
        "map-90": 0.0,
    }


def test_torchvision_detection_metrics_are_reported_per_test_set():
    events = []
    reporter = WorkerReporter(lambda event, payload: events.append((event, payload)))

    report_detection_metrics(
        reporter,
        {"map": 0.2, "map-50": 0.3},
        iteration=7,
        test_index=1,
    )

    assert events == [
        ("metric", {"name": "map_test1", "value": 0.2, "iteration": 7}),
        ("metric", {"name": "map-50_test1", "value": 0.3, "iteration": 7}),
    ]


def test_runtime_maps_non_finite_metric_to_contract_error():
    class BadMetricWorker(DeepDetectWorkerBase):
        def train(self, params, *, reporter, cancellation):
            reporter.metric("bad_metric", float("inf"), iteration=1)
            return {"status": "finished"}

        def predict(self, params):
            return {"results": []}

    module_name = "test_bad_metric_worker"
    install_worker_module(module_name, BadMetricWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        assert read_until(server, message_id=1)["result"] == {}
        request(server, 2, "train_start", {"request": {"parameters": {}}})
        assert read_until(server, message_id=2)["result"]["status"] == "started"
        failure = read_until(server, event="failure")
        assert failure["payload"]["category"] == "metric_contract_error"
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)
        sys.modules.pop(module_name, None)


def test_runtime_rejects_missing_worker_methods():
    class MissingMethodsWorker:
        def configure(self, context):
            return {}

    module_name = "test_missing_methods_worker"
    install_worker_module(module_name, MissingMethodsWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        response = read_until(server, message_id=1)
        assert response["error"]["category"] == "worker_contract_error"
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)
        sys.modules.pop(module_name, None)


def test_runtime_rejects_malformed_predictions():
    class BadPredictionWorker(DeepDetectWorkerBase):
        def train(self, params, *, reporter, cancellation):
            return {"status": "finished"}

        def predict(self, params):
            return {"not_results": []}

    module_name = "test_bad_prediction_worker"
    install_worker_module(module_name, BadPredictionWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        assert read_until(server, message_id=1)["result"] == {}
        request(server, 2, "predict", {"request": {"data": ["image.jpg"]}})
        response = read_until(server, message_id=2)
        assert response["error"]["category"] == "prediction_contract_error"
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)
        sys.modules.pop(module_name, None)
