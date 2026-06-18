import json
import struct
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
    WorkerConnector,
    WorkerContractError,
    WorkerDependencyError,
    WorkerReporter,
)
from deepdetect.pytorch_worker.tensors import parse_tensor_batch_ref, parse_tensor_ref
from deepdetect.pytorch_worker.tensors import (
    materialize_inline_tensor_ref,
    materialize_tensor_ref,
)
from deepdetect.pytorch_worker.builtin.vision.detection.base import (
    DetectionTrainingWorkerBase,
)
from deepdetect.pytorch_worker.builtin.vision.detection.common import (
    DetectionListDataset,
    DetectionTensorBatchDataset,
    make_loader,
)
from deepdetect.pytorch_worker.builtin.vision.detection.training import (
    DetectionProgressReporter,
    DetectionRepositoryContractWriter,
    DetectionTrainOptions,
    DetectionTrainRequest,
    class_mapping,
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


def test_runtime_routes_training_thread_connector_request():
    class ConnectorWorker(DeepDetectWorkerBase):
        def train(self, params, *, reporter, cancellation):
            info = self.context.connector.dataset_info()
            reporter.metric("train_samples", info["train_samples"], iteration=1)
            return {"status": "finished", "train_samples": info["train_samples"]}

        def predict(self, params):
            return {"results": []}

    module_name = "unit_test_connector_worker"
    install_worker_module(module_name, ConnectorWorker)
    runtime_sock, host_sock = socket_pair()
    runtime = WorkerRuntime(runtime_sock)
    thread = threading.Thread(target=runtime.serve, daemon=True)
    thread.start()
    try:
        request(
            host_sock,
            1,
            "configure",
            {"repository": "/tmp/dd-test", "mllib": {"module": module_name}},
        )
        configure_response = read_until(host_sock, message_id=1)
        assert configure_response["result"] == {}

        request(host_sock, 2, "train_start", {"request": {"data": []}})
        train_start_response = read_until(host_sock, message_id=2)
        assert train_start_response["result"] == {"status": "started"}

        connector_request = read_frame(host_sock)
        assert "method" in connector_request, connector_request
        assert connector_request["method"] == "connector_dataset_info"
        assert connector_request["id"] < 0
        write_frame(
            host_sock,
            {
                "id": connector_request["id"],
                "result": {
                    "task": "detection",
                    "boundary": "connector-tensor-pull",
                    "train_samples": 3,
                    "test_samples": [1],
                },
            },
        )

        metric = read_until(host_sock, event="metric")
        assert metric["payload"] == {
            "name": "train_samples",
            "value": 3,
            "iteration": 1,
        }
        result = read_until(host_sock, event="train_result")
        assert result["payload"] == {"status": "finished", "train_samples": 3}

        request(host_sock, 3, "shutdown")
        shutdown_response = read_until(host_sock, message_id=3)
        assert shutdown_response["result"] == {"status": "shutdown"}
    finally:
        host_sock.close()
        runtime_sock.close()
        thread.join(timeout=2)


def tensor_ref_payload(**overrides):
    payload = {
        "kind": "tensor_ref",
        "device": "cpu",
        "dtype": "float32",
        "shape": [2, 3, 4],
        "layout": "strided",
        "strides": [12, 4, 1],
        "storage": {
            "type": "shared_memory",
            "name": "dd-test-batch",
            "offset": 0,
            "nbytes": 96,
        },
        "lifetime": {
            "owner": "deepdetect",
            "valid_until_ack": "batch_done",
        },
        "cuda": {
            "ipc_handle": None,
            "stream": None,
        },
    }
    payload.update(overrides)
    return payload


def inline_tensor_ref_payload(shape, values, *, dtype="float32"):
    return tensor_ref_payload(
        dtype=dtype,
        shape=list(shape),
        strides=None,
        storage={
            "type": "inline_test_stub",
            "name": "unit-test",
            "offset": 0,
            "nbytes": 0,
            "values": list(values),
        },
    )


def detection_tensor_batch_payload(
    *,
    sample_id=7,
    width=16,
    height=12,
    value=0.5,
):
    return {
        "kind": "tensor_batch",
        "inputs": [
            inline_tensor_ref_payload(
                (1, 3, height, width),
                [value] * (1 * 3 * height * width),
            )
        ],
        "targets": {
            "boxes": [[[1, 2, min(8, width), min(9, height)]]],
            "labels": [[1]],
        },
        "meta": {
            "sample_ids": [sample_id],
            "paths": [f"tensor://sample{sample_id}"],
            "target_paths": [f"tensor://sample{sample_id}.txt"],
            "widths": [width],
            "heights": [height],
            "original_widths": [width],
            "original_heights": [height],
            "preprocessed_widths": [width],
            "preprocessed_heights": [height],
            "augmentation_applied": False,
            "augmentation_seed": None,
            "augmentation_policy": "none",
        },
    }


def cxx_compatible_detection_tensor_batch_payload(
    *,
    sample_id=7,
    width=16,
    height=12,
    value=0.5,
):
    payload = detection_tensor_batch_payload(
        sample_id=sample_id,
        width=width,
        height=height,
        value=value,
    )
    payload["targets"] = {
        "samples": [
            {
                "boxes": [
                    {
                        "xmin": 1,
                        "ymin": 2,
                        "xmax": min(8, width),
                        "ymax": min(9, height),
                    }
                ],
                "labels": [1],
            }
        ]
    }
    return payload


def test_tensor_ref_parses_cpu_metadata():
    ref = parse_tensor_ref(tensor_ref_payload())

    assert ref.device == "cpu"
    assert ref.dtype == "float32"
    assert ref.shape == (2, 3, 4)
    assert ref.strides == (12, 4, 1)
    assert ref.storage.type == "shared_memory"
    assert ref.storage.name == "dd-test-batch"
    assert ref.storage.nbytes == 96
    assert ref.lifetime["owner"] == "deepdetect"


def test_tensor_ref_accepts_inline_test_stub_storage():
    ref = parse_tensor_ref(
        tensor_ref_payload(
            storage={
                "type": "inline_test_stub",
                "name": "unit-test",
                "offset": 0,
                "nbytes": 0,
            },
            strides=None,
        )
    )

    assert ref.storage.type == "inline_test_stub"
    assert ref.strides is None


def test_inline_tensor_ref_materializes_torch_tensor():
    torch = pytest.importorskip("torch")
    ref = parse_tensor_ref(inline_tensor_ref_payload((2, 3), range(6)))

    tensor = materialize_inline_tensor_ref(ref, torch)

    assert tuple(tensor.shape) == (2, 3)
    assert tensor.dtype == torch.float32
    assert tensor[1, 2].item() == 5.0


def test_inline_tensor_ref_rejects_mismatched_value_count():
    torch = pytest.importorskip("torch")
    ref = parse_tensor_ref(inline_tensor_ref_payload((2, 3), [1, 2]))

    with pytest.raises(WorkerContractError) as error:
        materialize_inline_tensor_ref(ref, torch)

    assert "values count" in str(error.value)


def test_shared_memory_tensor_ref_materializes_torch_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    storage = tmp_path / "tensor.bin"
    storage.write_bytes(struct.pack("6f", 1, 2, 3, 4, 5, 6))
    ref = parse_tensor_ref(
        tensor_ref_payload(
            shape=[2, 3],
            strides=None,
            storage={
                "type": "shared_memory",
                "name": str(storage),
                "offset": 0,
                "nbytes": 24,
            },
        )
    )

    materialized = materialize_tensor_ref(ref, torch)
    try:
        tensor = materialized.tensor
        assert tuple(tensor.shape) == (2, 3)
        assert tensor.dtype == torch.float32
        assert tensor[1, 2].item() == 6.0
    finally:
        materialized.close()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"kind": "not_tensor_ref"}, "kind"),
        ({"dtype": "complex64"}, "dtype"),
        ({"shape": [1, 0, 3]}, "shape"),
        ({"strides": [3, 1]}, "strides length"),
        ({"layout": "channels_last"}, "layout"),
        ({"storage": {"type": "unknown"}}, "storage type"),
        ({"storage": {"type": "shared_memory", "nbytes": 4}}, "requires name"),
        ({"storage": {"type": "shared_memory", "name": "x", "nbytes": 0}}, "nbytes"),
    ],
)
def test_tensor_ref_rejects_invalid_metadata(override, message):
    with pytest.raises(WorkerContractError) as error:
        parse_tensor_ref(tensor_ref_payload(**override))

    assert message in str(error.value)


def test_tensor_ref_rejects_reserved_gpu_device_for_consumption():
    with pytest.raises(DatasetContractError) as error:
        parse_tensor_ref(tensor_ref_payload(device="cuda:0"))

    assert "reserved for future use" in str(error.value)


def test_tensor_batch_ref_parses_inputs_targets_and_metadata():
    batch = parse_tensor_batch_ref(
        {
            "kind": "tensor_batch",
            "inputs": [tensor_ref_payload()],
            "targets": {
                "labels": [1, 2],
            },
            "meta": {
                "sample_ids": [10, 11],
            },
        }
    )

    assert len(batch.inputs) == 1
    assert batch.targets == {"labels": [1, 2]}
    assert batch.meta == {"sample_ids": [10, 11]}


def test_tensor_batch_ref_requires_non_empty_inputs():
    with pytest.raises(WorkerContractError) as error:
        parse_tensor_batch_ref({"kind": "tensor_batch", "inputs": []})

    assert "non-empty" in str(error.value)


def test_detection_tensor_batch_dataset_returns_detection_sample_contract():
    torch = pytest.importorskip("torch")
    batch = parse_tensor_batch_ref(
        {
            "kind": "tensor_batch",
            "inputs": [
                inline_tensor_ref_payload(
                    (1, 3, 4, 5),
                    [0.25] * (1 * 3 * 4 * 5),
                )
            ],
            "targets": {
                "boxes": [[[1, 1, 4, 3]]],
                "labels": [[1]],
            },
            "meta": {
                "sample_ids": [42],
                "paths": ["tensor://sample0"],
                "widths": [5],
                "heights": [4],
            },
        }
    )

    dataset = DetectionTensorBatchDataset([batch], nclasses=2, torch=torch)
    image, target, meta = dataset[0]

    assert len(dataset) == 1
    assert tuple(image.shape) == (3, 4, 5)
    assert tuple(target["boxes"].shape) == (1, 4)
    assert target["labels"].tolist() == [1]
    assert target["image_id"].tolist() == [42]
    assert meta == {
        "index": 42,
        "path": "tensor://sample0",
        "width": 5,
        "height": 4,
    }


def test_detection_tensor_batch_dataset_accepts_cxx_compatible_targets():
    torch = pytest.importorskip("torch")
    batch = parse_tensor_batch_ref(
        cxx_compatible_detection_tensor_batch_payload(sample_id=43)
    )

    dataset = DetectionTensorBatchDataset([batch], nclasses=2, torch=torch)
    _image, target, meta = dataset[0]

    assert len(dataset) == 1
    assert target["boxes"].tolist() == [[1.0, 2.0, 8.0, 9.0]]
    assert target["labels"].tolist() == [1]
    assert target["image_id"].tolist() == [43]
    assert meta["path"] == "tensor://sample43"


def test_detection_tensor_batch_dataset_accepts_pull_response_metadata(
    monkeypatch, capsys
):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("DEEPDETECT_DEBUG", raising=False)
    monkeypatch.delenv("DEEPDETECT_WORKER_DEBUG", raising=False)
    payload = cxx_compatible_detection_tensor_batch_payload(sample_id=44)
    payload["meta"]["future_connector_field"] = {"ignored": True}

    def request(method, params):
        if method == "connector_batch_next":
            assert params["split"] == "train"
            return {
                "status": "ok",
                "end": False,
                "batch_id": "batch-44",
                "split": "train",
                "test_index": None,
                "epoch": 1,
                "cursor_start": 0,
                "cursor_end": 1,
                "requested_batch_size": 1,
                "sample_count": 1,
                "transport": "inline",
                "tensor_nbytes": 2304,
                "future_top_level_field": "ignored",
                "batch": payload,
            }
        if method == "connector_batch_done":
            return {"status": "ok"}
        raise AssertionError(method)

    connector = WorkerConnector(request)
    response = connector.next_batch(split="train", batch_size=1)
    connector.batch_done(response["batch_id"])
    batch = parse_tensor_batch_ref(response["batch"])
    dataset = DetectionTensorBatchDataset([batch], nclasses=2, torch=torch)
    _image, target, meta = dataset[0]

    assert response["sample_count"] == 1
    assert batch.meta["target_paths"] == ["tensor://sample44.txt"]
    assert batch.meta["augmentation_applied"] is False
    assert batch.meta["augmentation_seed"] is None
    assert target["image_id"].tolist() == [44]
    assert meta["path"] == "tensor://sample44"
    assert capsys.readouterr().err == ""


def test_detection_tensor_batch_dataset_feeds_reference_detector_loss():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    batch = parse_tensor_batch_ref(
        {
            "kind": "tensor_batch",
            "inputs": [
                inline_tensor_ref_payload(
                    (1, 3, 12, 16),
                    [0.5] * (1 * 3 * 12 * 16),
                )
            ],
            "targets": {
                "boxes": [[[1, 2, 8, 9]]],
                "labels": [[1]],
            },
            "meta": {"sample_ids": [7]},
        }
    )
    dataset = DetectionTensorBatchDataset([batch], nclasses=2, torch=torch)
    loader = make_loader(dataset, batch_size=1, shuffle=False, torch=torch)
    images, targets, _metas = next(iter(loader))
    worker = ReferenceTorchDetectorWorker()
    backend = worker.import_backend()
    model = worker.create_model(2, *backend)

    losses = worker.training_losses(model, images, targets)
    total = sum(losses.values())
    total.backward()

    assert {"loss_classifier", "loss_box_reg"} <= set(losses)


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


def test_detection_train_request_merges_config_and_train_overrides(tmp_path):
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    context = WorkerContext(
        repository=str(tmp_path),
        mllib={
            "nclasses": 2,
            "solver": {"iterations": 10, "test_interval": 5},
            "net": {"batch_size": 4},
            "base_only": True,
        },
        raw={},
    )

    parsed = DetectionTrainRequest.from_params(
        context,
        {
            "request": {
                "data": [str(train), str(test)],
                "parameters": {
                    "mllib": {
                        "solver": {"iterations": 3, "iter_size": 2},
                        "net": {"batch_size": 1},
                    }
                },
            }
        },
    )

    assert parsed.train_list == train
    assert parsed.test_lists == [test]
    assert parsed.effective_mllib["base_only"] is True
    assert parsed.options == DetectionTrainOptions(
        iterations=3,
        test_interval=3,
        batch_size=1,
        iter_size=2,
        base_lr=0.0001,
    )


def test_detection_train_request_accepts_tensor_batches(tmp_path):
    context = WorkerContext(
        repository=str(tmp_path),
        mllib={"nclasses": 2, "solver": {"iterations": 10}},
        raw={},
    )

    parsed = DetectionTrainRequest.from_params(
        context,
        {
            "request": {
                "data": [],
                "tensor_batches": {
                    "train": [detection_tensor_batch_payload(sample_id=1)],
                    "tests": [[detection_tensor_batch_payload(sample_id=2)]],
                },
                "parameters": {
                    "mllib": {
                        "solver": {"iterations": 1, "test_interval": 1},
                    }
                },
            }
        },
    )

    assert parsed.source == "tensor"
    assert parsed.train_list is None
    assert parsed.test_lists == []
    assert len(parsed.train_tensor_batches) == 1
    assert len(parsed.test_tensor_batches) == 1
    assert len(parsed.test_tensor_batches[0]) == 1
    assert parsed.options.iterations == 1


def test_detection_train_request_accepts_cxx_compatible_tensor_test_sets(tmp_path):
    parsed = DetectionTrainRequest.from_params(
        WorkerContext(repository=str(tmp_path), mllib={"nclasses": 2}, raw={}),
        {
            "request": {
                "data": [],
                "tensor_batches": {
                    "train": [
                        cxx_compatible_detection_tensor_batch_payload(sample_id=1)
                    ],
                    "tests": [
                        {
                            "batches": [
                                cxx_compatible_detection_tensor_batch_payload(
                                    sample_id=2
                                )
                            ]
                        }
                    ],
                },
            }
        },
    )

    assert parsed.source == "tensor"
    assert len(parsed.train_tensor_batches) == 1
    assert len(parsed.test_tensor_batches) == 1
    assert len(parsed.test_tensor_batches[0]) == 1


def test_detection_train_request_rejects_mixed_path_and_tensor_sources(tmp_path):
    train = tmp_path / "train.txt"

    with pytest.raises(DatasetContractError) as error:
        DetectionTrainRequest.from_params(
            None,
            {
                "request": {
                    "data": [str(train)],
                    "tensor_batches": {
                        "train": [detection_tensor_batch_payload()],
                    },
                }
            },
        )

    assert "must not mix" in str(error.value)


def test_detection_repository_contract_writer_persists_expected_artifacts(tmp_path):
    class FakeDataset:
        def __init__(self, path, size):
            self.list_path = path
            self.size = size

        def __len__(self):
            return self.size

    context = WorkerContext(
        repository=str(tmp_path),
        mllib={"nclasses": 3, "class_names": ["background", "ring", "hand"]},
        raw={},
    )
    train = FakeDataset(tmp_path / "train.txt", 12)
    tests = [FakeDataset(tmp_path / "test.txt", 3)]

    DetectionRepositoryContractWriter(
        context,
        worker_name="fake-detector",
        task_name="detection",
        nclasses=3,
    ).write(
        train_dataset=train,
        test_datasets=tests,
        request={
            "data": [str(train.list_path), str(tests[0].list_path)],
        },
        request_params={
            "input": {"width": 320},
            "output": {"measure": ["map-50"]},
        },
        effective_mllib=context.mllib,
    )

    config = json.loads(
        (tmp_path / "pytorch_worker_config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "connector_manifest.json").read_text(encoding="utf-8")
    )
    classes = json.loads(
        (tmp_path / "class_mapping.json").read_text(encoding="utf-8")
    )

    assert config["worker"] == "fake-detector"
    assert config["input_parameters"] == {"width": 320}
    assert manifest["train"] == {"path": str(train.list_path), "samples": 12}
    assert manifest["tests"] == [
        {"index": 0, "path": str(tests[0].list_path), "samples": 3}
    ]
    assert classes == {"0": "background", "1": "ring", "2": "hand"}


def test_detection_repository_contract_writer_persists_tensor_backed_artifacts(tmp_path):
    class FakeTensorDataset:
        def __init__(self, batches, size):
            self.batches = batches
            self.size = size

        def __len__(self):
            return self.size

    context = WorkerContext(
        repository=str(tmp_path),
        mllib={"nclasses": 2},
        raw={},
    )
    train = FakeTensorDataset([object(), object()], 4)
    tests = [FakeTensorDataset([object()], 2)]

    DetectionRepositoryContractWriter(
        context,
        worker_name="fake-detector",
        task_name="detection",
        nclasses=2,
    ).write(
        train_dataset=train,
        test_datasets=tests,
        source="tensor",
        request={
            "data": [],
            "tensor_batches": {"train": ["redacted"], "tests": [["redacted"]]},
        },
        request_params={},
        effective_mllib=context.mllib,
    )

    config = json.loads(
        (tmp_path / "pytorch_worker_config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "connector_manifest.json").read_text(encoding="utf-8")
    )

    assert config["data"] == []
    assert config["tensor_batches"] == {
        "train_batches": 2,
        "test_batches": [1],
    }
    assert manifest["boundary"] == "tensor-backed"
    assert manifest["train"] == {
        "source": "tensor-backed",
        "batches": 2,
        "samples": 4,
    }
    assert manifest["tests"] == [
        {
            "index": 0,
            "source": "tensor-backed",
            "batches": 1,
            "samples": 2,
        }
    ]


def test_detection_repository_contract_writer_persists_connector_summary(tmp_path):
    context = WorkerContext(
        repository=str(tmp_path),
        mllib={"nclasses": 2},
        raw={},
    )
    train = type("DatasetSummary", (), {"__len__": lambda self: 5})()
    tests = [type("DatasetSummary", (), {"__len__": lambda self: 2})()]

    DetectionRepositoryContractWriter(
        context,
        worker_name="fake-detector",
        task_name="detection",
        nclasses=2,
    ).write(
        train_dataset=train,
        test_datasets=tests,
        source="connector_pull",
        request={"data": ["train.txt", "test.txt"]},
        request_params={},
        effective_mllib=context.mllib,
        connector_info={
            "transport": "shared_memory",
            "input_width": 320,
            "input_height": 240,
            "train_shuffle": True,
            "train_samples": 5,
            "test_samples": [2],
            "augmentation_enabled": False,
            "ignored": "value",
        },
    )

    manifest = json.loads(
        (tmp_path / "connector_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["boundary"] == "connector-tensor-pull"
    assert manifest["connector"] == {
        "transport": "shared_memory",
        "input_width": 320,
        "input_height": 240,
        "train_shuffle": True,
        "train_samples": 5,
        "test_samples": [2],
        "augmentation_enabled": False,
    }


def test_detection_progress_reporter_emits_stable_status_and_metrics():
    events = []
    progress = DetectionProgressReporter(
        WorkerReporter(lambda event, payload: events.append((event, payload)))
    )

    progress.train_step(
        iteration=2,
        iterations=5,
        start_time=time.monotonic(),
        base_lr=0.01,
        train_loss=1.5,
        losses={"loss_box": 0.5},
    )
    progress.test_progress(
        iteration=2,
        test_index=1,
        test_sets_total=3,
        processed=4,
        total=10,
    )
    progress.test_finished(
        iteration=2,
        test_sets_total=3,
        predictions_payload={"test1": {"iteration": 2, "samples": []}},
    )

    status_payloads = [payload for event, payload in events if event == "status"]
    metric_payloads = [payload for event, payload in events if event == "metric"]
    assert status_payloads[0]["phase"] == "train"
    assert status_payloads[1]["test_set_index"] == 1
    assert status_payloads[1]["test_processed"] == 4
    assert status_payloads[2]["test_active"] == 0
    assert "test_predictions" in status_payloads[2]
    assert {"name": "train_loss", "value": 1.5, "iteration": 2} in metric_payloads
    assert {"name": "loss_box", "value": 0.5, "iteration": 2} in metric_payloads


def test_detection_class_mapping_accepts_dict_and_list_names():
    assert class_mapping(3, {"classes": ["background", "ring", "hand"]}) == {
        "0": "background",
        "1": "ring",
        "2": "hand",
    }
    assert class_mapping(3, {"class_names": {"1": "ring", 2: "hand"}}) == {
        "0": "background",
        "1": "ring",
        "2": "hand",
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


def test_runtime_trains_reference_detector_from_tensor_batches(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
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

        request(
            server,
            2,
            "train_start",
            {
                "request": {
                    "data": [],
                    "tensor_batches": {
                        "train": [detection_tensor_batch_payload(sample_id=10)],
                        "tests": [[detection_tensor_batch_payload(sample_id=11)]],
                    },
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
        assert manifest["boundary"] == "tensor-backed"
        assert manifest["train"] == {
            "source": "tensor-backed",
            "batches": 1,
            "samples": 1,
        }
        assert manifest["tests"] == [
            {
                "index": 0,
                "source": "tensor-backed",
                "batches": 1,
                "samples": 1,
            }
        ]
        assert (tmp_path / "checkpoint-latest.pt").is_file()
        assert (tmp_path / "solver-latest.pt").is_file()
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


def test_runtime_reports_missing_entrypoint_as_launch_error(tmp_path):
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(
            server,
            1,
            "configure",
            {"mllib": {"entrypoint": str(tmp_path / "missing_worker.py")}},
        )
        response = read_until(server, message_id=1)
        assert response["error"]["category"] == "worker_launch_error"
        assert response["error"]["method"] == "configure"
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)


def test_runtime_reports_configure_dependency_error():
    class DependencyWorker(DeepDetectWorkerBase):
        def configure(self, context):
            raise WorkerDependencyError("missing optional package")

        def train(self, params, *, reporter, cancellation):
            return {"status": "finished"}

        def predict(self, params):
            return {"results": []}

    module_name = "test_dependency_worker"
    install_worker_module(module_name, DependencyWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        response = read_until(server, message_id=1)
        assert response["error"]["category"] == "dependency_error"
        assert response["error"]["method"] == "configure"
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)
        sys.modules.pop(module_name, None)


def test_runtime_reports_generic_training_exception_as_training_error():
    class FailingTrainWorker(DeepDetectWorkerBase):
        def train(self, params, *, reporter, cancellation):
            raise RuntimeError("training exploded")

        def predict(self, params):
            return {"results": []}

    module_name = "test_failing_train_worker"
    install_worker_module(module_name, FailingTrainWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        assert read_until(server, message_id=1)["result"] == {}
        request(server, 2, "train_start", {"request": {"parameters": {}}})
        assert read_until(server, message_id=2)["result"]["status"] == "started"
        failure = read_until(server, event="failure")
        assert failure["payload"]["category"] == "training_error"
        assert failure["payload"]["method"] == "train"
        assert "training exploded" in failure["payload"]["traceback"]
    finally:
        request(server, 99, "shutdown")
        read_until(server, message_id=99)
        server.close()
        client.close()
        thread.join(timeout=2)
        sys.modules.pop(module_name, None)


def test_runtime_cooperative_cancellation_returns_cancelled_result():
    class CancellableWorker(DeepDetectWorkerBase):
        def train(self, params, *, reporter, cancellation):
            deadline = time.monotonic() + 2.0
            while not cancellation.requested and time.monotonic() < deadline:
                time.sleep(0.01)
            if cancellation.requested:
                return {"status": "cancelled"}
            return {"status": "finished"}

        def predict(self, params):
            return {"results": []}

    module_name = "test_cancellable_worker"
    install_worker_module(module_name, CancellableWorker)
    server, client = socket_pair()
    thread = threading.Thread(target=WorkerRuntime(client).serve, daemon=True)
    thread.start()
    try:
        request(server, 1, "configure", {"mllib": {"module": module_name}})
        assert read_until(server, message_id=1)["result"] == {}
        request(server, 2, "train_start", {"request": {"parameters": {}}})
        assert read_until(server, message_id=2)["result"]["status"] == "started"
        request(server, 3, "train_cancel")
        assert read_until(server, message_id=3)["result"]["status"] == "cancelling"
        result = read_until(server, event="train_result")
        assert result["payload"]["status"] == "cancelled"
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
