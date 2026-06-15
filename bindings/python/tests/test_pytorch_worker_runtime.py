import threading
import time
import sys
import types

from deepdetect.pytorch_worker.protocol import read_frame, write_frame
from deepdetect.pytorch_worker.runtime import WorkerRuntime
from deepdetect.pytorch_worker.sdk import DeepDetectWorkerBase


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
