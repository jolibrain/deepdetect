import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepdetect import CapabilityError, DeepDetect, DeepDetectError, TrainingJob


def response(code=200, *, head=None, body=None, **status):
    value = {"status": {"code": code, "msg": "OK" if code < 300 else "error"}}
    value["status"].update(status)
    if head is not None:
        value["head"] = head
    if body is not None:
        value["body"] = body
    return json.dumps(value)


class FakeRuntime:
    def __init__(self, *, cuda=True):
        self.cuda = cuda
        self.calls = []
        self.statuses = []

    def build_info(self):
        return json.dumps({"version": "test", "cuda": self.cuda})

    def info(self, request):
        self.calls.append(("info", json.loads(request)))
        return response(head={"version": "test", "services": []})

    def create_service(self, name, request):
        self.calls.append(("create", name, json.loads(request)))
        return response(201)

    def service_info(self, name):
        self.calls.append(("service_info", name))
        return response(body={"name": name})

    def delete_service(self, name, request):
        self.calls.append(("delete", name, json.loads(request)))
        return response()

    def predict(self, request):
        self.calls.append(("predict", json.loads(request)))
        return response(body={"predictions": [{"cat": "ok"}]})

    def train(self, request):
        request = json.loads(request)
        self.calls.append(("train", request))
        if request["async"]:
            return response(201, head={"job": 7, "status": "running"})
        return response(201, body={"measure": {"acc": 1.0}})

    def training_status(self, request):
        self.calls.append(("status", json.loads(request)))
        status = self.statuses.pop(0) if self.statuses else "running"
        return response(head={"job": 7, "status": status, "time": 1.0}, body={})

    def cancel_training(self, request):
        self.calls.append(("cancel", json.loads(request)))
        return response()


def make_service(runtime=None, **mllib_parameters):
    runtime = runtime or FakeRuntime()
    dd = DeepDetect(_runtime=runtime)
    service = dd.create_service(
        "Classifier",
        model={"repository": "/models/classifier"},
        mllib="torch",
        input_parameters={"connector": "image"},
        mllib_parameters=mllib_parameters,
        output_parameters={},
    )
    return runtime, dd, service


def test_create_service_and_normalized_payloads():
    runtime, dd, service = make_service()
    call = runtime.calls[0]
    assert call[1] == "classifier"
    assert call[2]["parameters"]["input"]["connector"] == "image"
    assert dd.info()["version"] == "test"
    assert service.info() == {"name": "classifier"}
    assert service.predict(Path("image.png"))["predictions"][0]["cat"] == "ok"
    assert runtime.calls[-1][1]["data"] == ["image.png"]


def test_json_data_is_preserved():
    runtime, _, service = make_service()
    service.predict({"rows": [1, 2.5, "three"]})
    assert runtime.calls[-1][1]["data"] == {"rows": [1, 2.5, "three"]}


def test_blocking_and_async_training():
    runtime, _, service = make_service()
    assert service.train(["data"], asynchronous=False)["measure"]["acc"] == 1.0
    job = service.train(["data"], asynchronous=True)
    assert isinstance(job, TrainingJob)
    assert job.job == 7


def test_numpy_images_are_png_base64():
    runtime, _, service = make_service()
    service.predict(np.zeros((3, 4, 3), dtype=np.uint8))
    encoded = runtime.calls[-1][1]["data"][0]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert image.format == "PNG"
    assert image.size == (4, 3)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3, 1), dtype=np.uint8),
    ],
)
def test_invalid_numpy_images(array):
    _, _, service = make_service()
    with pytest.raises(TypeError):
        service.predict(array)


def test_cpu_sdk_rejects_gpu_before_dispatch():
    runtime = FakeRuntime(cuda=False)
    dd = DeepDetect(_runtime=runtime)
    with pytest.raises(CapabilityError):
        dd.create_service(
            "gpu",
            model={"repository": "/tmp/model"},
            mllib="torch",
            input_parameters={"connector": "image"},
            mllib_parameters={"gpu": True},
        )
    assert runtime.calls == []


def test_cpu_sdk_rejects_gpu_training_before_dispatch():
    runtime, _, service = make_service(FakeRuntime(cuda=False))
    with pytest.raises(CapabilityError):
        service.train(["data"], mllib_parameters={"gpu": True})
    assert runtime.calls[-1][0] == "create"


def test_error_mapping():
    runtime = FakeRuntime()
    runtime.service_info = lambda name: response(
        404, dd_code=1002, dd_msg="Service not found"
    )
    service = DeepDetect(_runtime=runtime).service("missing")
    with pytest.raises(DeepDetectError) as captured:
        service.info()
    assert captured.value.status_code == 404
    assert captured.value.dd_code == 1002
    assert captured.value.response["status"]["dd_msg"] == "Service not found"


def test_deleted_handle_is_invalid():
    _, _, service = make_service()
    service.delete(clear="full")
    with pytest.raises(RuntimeError, match="deleted"):
        service.info()


def test_service_context_manager_deletes_on_error():
    runtime, _, service = make_service()
    with pytest.raises(RuntimeError, match="prediction failed"):
        with service:
            raise RuntimeError("prediction failed")
    assert runtime.calls[-1] == ("delete", "classifier", {})
    with pytest.raises(RuntimeError, match="deleted"):
        service.info()


def test_job_wait_and_cancel(monkeypatch):
    runtime, _, service = make_service()
    runtime.statuses = ["running", "finished"]
    job = TrainingJob(service, 7)
    monkeypatch.setattr("deepdetect.client.time.sleep", lambda _: None)
    assert job.wait(poll_interval=0.01)["status"] == "finished"
    job.cancel()
    assert runtime.calls[-1] == ("cancel", {"service": "classifier", "job": 7})


def test_job_status_accepts_output_parameters():
    runtime, _, service = make_service()
    runtime.statuses = ["running"]
    job = TrainingJob(service, 7)

    job.status(output_parameters={"measure_hist": True, "max_hist_points": 10})

    assert runtime.calls[-1] == (
        "status",
        {
            "service": "classifier",
            "job": 7,
            "parameters": {
                "output": {"measure_hist": True, "max_hist_points": 10}
            },
        },
    )


def test_job_timeout(monkeypatch):
    _, _, service = make_service()
    job = TrainingJob(service, 7)
    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr("deepdetect.client.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("deepdetect.client.time.sleep", lambda _: None)
    with pytest.raises(TimeoutError):
        job.wait(timeout=0.5, poll_interval=0.01)
