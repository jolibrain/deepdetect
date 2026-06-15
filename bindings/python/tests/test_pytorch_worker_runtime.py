import threading
import time

from deepdetect.pytorch_worker.protocol import read_frame, write_frame
from deepdetect.pytorch_worker.runtime import WorkerRuntime


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
