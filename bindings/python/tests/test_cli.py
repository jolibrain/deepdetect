import io
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image

from deepdetect import DeepDetect
from deepdetect.cli import config
from deepdetect.cli import inference
from deepdetect.cli import __file__ as cli_package_file
from deepdetect.cli import main as cli
from deepdetect.cli.checks import validate_detection_lists
from deepdetect.cli import results
from deepdetect.cli import runs
from deepdetect.cli import training
from deepdetect.cli.profiles import get_profile
from deepdetect.cli.sinks import VisdomMetricSink
from deepdetect.cli.terminal import LiveTrainingTerminalReporter
from deepdetect.cli.visualize import (
    detection_overlay_image,
    segmentation_overlay_images,
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value)


def response(code=200, *, head=None, body=None, **status):
    value = {"status": {"code": code, "msg": "OK" if code < 300 else "error"}}
    value["status"].update(status)
    if head is not None:
        value["head"] = head
    if body is not None:
        value["body"] = body
    return json.dumps(value)


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.statuses = []

    def build_info(self):
        return json.dumps({"version": "test", "cuda": True})

    def create_service(self, name, request):
        self.calls.append(("create", name, json.loads(request)))
        return response(201)

    def set_log_level(self, level):
        self.calls.append(("set_global_log_level", level))
        return response()

    def set_service_log_level(self, name, level):
        self.calls.append(("set_log_level", name, level))
        return response()

    def delete_service(self, name, request):
        self.calls.append(("delete", name, json.loads(request)))
        return response()

    def train(self, request):
        request = json.loads(request)
        self.calls.append(("train", request))
        if request["async"]:
            return response(201, head={"job": 7, "status": "running"})
        return response(201, body={"measure": {"train_loss": 1.0}})

    def training_status(self, request):
        self.calls.append(("status", json.loads(request)))
        status = self.statuses.pop(0)
        return response(
            head={
                "job": 7,
                "status": status["status"],
                "time": status.get("time", 1.0),
            },
            body=status.get("body", {}),
        )

    def predict(self, request):
        request = json.loads(request)
        self.calls.append(("predict", request))
        predictions = []
        for item in request["data"]:
            predictions.append(
                {
                    "uri": str(item),
                    "classes": [
                        {
                            "cat": "object",
                            "prob": 0.9,
                            "bbox": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                        }
                    ],
                }
            )
        return response(body={"predictions": predictions})


def write_training_files(tmp_path: Path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    target = tmp_path / "target.txt"
    target.write_text("1 0 0 4 4\n", encoding="utf-8")
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text(f"{image} {target}\n", encoding="utf-8")
    test.write_text(f"{image} {target}\n", encoding="utf-8")
    return weights, train, test


def test_detection_dataset_validation_resolves_relative_entries(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    image = images / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    bbox = labels / "target.txt"
    bbox.write_text("1 0 0 4 4\n", encoding="utf-8")
    train = dataset / "train.txt"
    train.write_text("images/image.jpg labels/target.txt\n", encoding="utf-8")

    summary = validate_detection_lists([train], nclasses=2)

    assert summary["checked_images"] == 1
    assert summary["checked_bbox_files"] == 1
    assert summary["positive_boxes"] == 1
    assert summary["class_counts"] == {1: 1}


def write_resume_repository(repository: Path, *, iteration: int = 10) -> Path:
    repository.mkdir(parents=True)
    (repository / f"solver-{iteration}.pt").write_bytes(b"solver")
    (repository / f"checkpoint-{iteration}.pt").write_bytes(b"checkpoint")
    return repository


def test_config_merge_and_set_overrides(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("solver:\n  base_lr: 0.1\nbatch_size: 2\n", encoding="utf-8")

    merged = config.deep_merge(
        {"solver": {"base_lr": 0.01, "iterations": 10}},
        config.load_config(cfg),
        config.parse_overrides(["solver.iterations=20", "gpu=true"]),
    )

    assert merged["solver"] == {"base_lr": 0.1, "iterations": 20}
    assert merged["batch_size"] == 2
    assert merged["gpu"] is True


def test_default_example_configs_load():
    root = Path(cli_package_file).resolve().parent

    yolox = config.load_config(root / "yolox-default.yaml")
    segformer = config.load_config(root / "segformer-default.yaml")
    torchvision = config.load_config(root / "torchvision-detector-default.yaml")
    external = config.load_config(root / "external-pytorch-detector-default.yaml")
    vitpose = config.load_config(root / "vitpose-default.yaml")

    assert yolox["width"] == 640
    assert yolox["height"] == 640
    assert yolox["confidence_threshold"] == 0.25
    assert yolox["iter_size"] == 2
    assert yolox["augmentation"]["crop_size"] == 0
    assert yolox["augmentation"]["geometry"]["prob"] == 0.1
    assert yolox["class_weights"] is None
    assert yolox["dataset_check"] == "full"
    assert segformer["width"] == 480
    assert segformer["height"] == 480
    assert segformer["batch_size"] == 4
    assert segformer["iter_size"] == 1
    assert segformer["augmentation"]["crop_size"] == 224
    assert segformer["augmentation"]["cutout"] == 0.5
    assert segformer["class_weights"] is None
    assert segformer["dataset_check"] == "full"
    assert torchvision["weights"] is None
    assert torchvision["width"] == 640
    assert torchvision["height"] == 640
    assert torchvision["batch_size"] == 1
    assert torchvision["augmentation"]["mirror"] is True
    assert torchvision["augmentation"]["geometry"]["prob"] == 0.1
    assert torchvision["mllib"]["data_source"] == "connector_tensor_pull"
    assert torchvision["class_weights"] is None
    assert torchvision["dataset_check"] == "full"
    assert external["weights"] is None
    assert external["service_mllib"]["entrypoint"] == (
        "extern/pytorch_workers/model_slug/worker.py"
    )
    assert external["service_mllib"]["class"] == "DeepDetectWorker"
    assert external["mllib"]["data_source"] == "connector_tensor_pull"
    assert external["dataset_check"] == "full"
    assert vitpose["weights"] is None
    assert vitpose["width"] == 192
    assert vitpose["height"] == 256
    assert vitpose["nkeypoints"] == 17
    assert vitpose["max_objects"] == 1
    assert vitpose["service_mllib"]["entrypoint"] == (
        "extern/pytorch_workers/vitpose/worker.py"
    )
    assert vitpose["service_mllib"]["task"] == "keypoint"
    assert vitpose["service_mllib"]["vitpose"]["head"] == "topdown"
    assert vitpose["mllib"]["data_source"] == "connector_tensor_pull"
    assert vitpose["dataset_check"] == "full"


def test_generated_training_run_name_is_repository_name_only():
    name = runs.generate_training_run_name(
        model="yolox",
        repository=Path("/checkpoints/ring hand detector"),
        timestamp="20260610-131415",
    )

    assert name == "ring-hand-detector"


def test_train_yolox_async_payload_and_manifest(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {"status": "running", "body": {"measure": {"iteration": 1, "train_loss": 2.0}}},
        {
            "status": "finished",
            "body": {"measure": {"iteration": 2, "train_loss": 1.0, "map-50": 0.5}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test = write_training_files(tmp_path)
    run_root = tmp_path / "runs"

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(run_root),
            "--iterations",
            "3",
            "--batch-size",
            "2",
            "--iter-size",
            "4",
            "--width",
            "320",
            "--height",
            "352",
            "--gpuid",
            "1,3",
            "--run-name",
            "ring-hand-yolox",
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["parameters"]["mllib"]["template"] == "yolox"
    assert create[2]["parameters"]["mllib"]["gpu"] is True
    assert create[2]["parameters"]["mllib"]["gpuid"] == [1, 3]
    assert create[2]["parameters"]["input"]["width"] == 320
    assert create[2]["parameters"]["input"]["height"] == 352
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["async"] is True
    assert train_call[1]["parameters"]["mllib"]["gpu"] is True
    assert train_call[1]["parameters"]["mllib"]["gpuid"] == [1, 3]
    assert train_call[1]["parameters"]["mllib"]["solver"]["iter_size"] == 4
    assert train_call[1]["parameters"]["input"]["width"] == 320
    assert train_call[1]["parameters"]["input"]["height"] == 352
    assert train_call[1]["parameters"]["output"]["measure"] == [
        "map-05",
        "map-50",
        "map-90",
    ]
    run_json = run_root / "ring-hand-yolox" / "run.json"
    manifest = json.loads(run_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "finished"
    assert manifest["job"] == 7
    assert manifest["run_name"] == "ring-hand-yolox"
    saved_config = config.load_config(tmp_path / "repo" / "config.yaml")
    assert saved_config["batch_size"] == 2
    assert saved_config["iter_size"] == 4
    assert saved_config["width"] == 320
    assert saved_config["gpuid"] == [1, 3]
    assert saved_config["run_name"] == "ring-hand-yolox"
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert "dataset_check" in {event["event"] for event in events}
    assert "metric" in {event["event"] for event in events}


def test_train_torchvision_detector_uses_pytorch_backend_without_weights(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    _weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "torchvision-detector",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--iterations",
            "1",
            "--dataset-check",
            "none",
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["mllib"] == "pytorch"
    assert create[2]["parameters"]["mllib"]["module"] == (
        "deepdetect.pytorch_worker.builtin.vision.detection.torchvision_fasterrcnn"
    )
    assert create[2]["parameters"]["mllib"]["task"] == "detection"
    assert create[2]["parameters"]["mllib"]["python"] == sys.executable
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert "weights" not in train_call[1]["parameters"]["mllib"]
    assert train_call[1]["parameters"]["output"]["measure"] == [
        "map-05",
        "map-50",
        "map-90",
    ]
    saved_config = config.load_config(tmp_path / "repo" / "config.yaml")
    assert saved_config["weights"] is None
    assert saved_config["batch_size"] == 1
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert "run_finished" in {event["event"] for event in events}


def test_train_external_pytorch_detector_passes_entrypoint_from_yaml(
    monkeypatch,
    tmp_path,
    capsys,
):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    _weights, train, test = write_training_files(tmp_path)
    entrypoint = tmp_path / "external_worker.py"
    entrypoint.write_text("class DeepDetectWorker: pass\n", encoding="utf-8")
    cfg = tmp_path / "external.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"train_data: {train}",
                "test_data:",
                f"  - {test}",
                f"repository: {tmp_path / 'repo'}",
                "service_mllib:",
                f"  entrypoint: {entrypoint}",
                "  class: DeepDetectWorker",
                "mllib:",
                "  data_source: connector_tensor_pull",
                "iterations: 1",
                "dataset_check: none",
            ]
        ),
        encoding="utf-8",
    )

    code = cli.main(
        [
            "train",
            "external-pytorch-detector",
            "--config",
            str(cfg),
            "--job-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    service_mllib = create[2]["parameters"]["mllib"]
    assert create[2]["mllib"] == "pytorch"
    assert service_mllib["entrypoint"] == str(entrypoint)
    assert service_mllib["class"] == "DeepDetectWorker"
    assert service_mllib["task"] == "detection"
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["parameters"]["mllib"]["data_source"] == (
        "connector_tensor_pull"
    )
    saved_config = config.load_config(tmp_path / "repo" / "config.yaml")
    assert saved_config["service_mllib"]["entrypoint"] == str(entrypoint)
    assert "run_finished" in {
        json.loads(line)["event"]
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    }


def test_vitpose_profile_passes_keypoint_worker_parameters():
    profile = get_profile("vitpose")
    options = profile.train_defaults()

    service = profile.service_parameters(options)
    train_params = profile.train_parameters(options)
    predict_params = profile.predict_parameters(options)

    service_mllib = service["mllib_parameters"]
    assert service["mllib"] == "pytorch"
    assert service["input_parameters"]["keypoints"] is True
    assert service_mllib["entrypoint"] == "extern/pytorch_workers/vitpose/worker.py"
    assert service_mllib["task"] == "keypoint"
    assert service_mllib["nkeypoints"] == 17
    assert service_mllib["max_objects"] == 1
    assert service_mllib["vitpose"]["head"] == "topdown"
    assert train_params["mllib_parameters"]["data_source"] == "connector_tensor_pull"
    assert train_params["mllib_parameters"]["nkeypoints"] == 17
    assert train_params["mllib_parameters"]["max_objects"] == 1
    assert predict_params["output_parameters"]["keypoints"] is True
    assert predict_params["output_parameters"]["bbox"] is True
    assert predict_params["output_parameters"]["confidence_threshold"] == 0.25
    assert predict_params["output_parameters"]["keypoint_threshold"] == 0.05


def test_train_accepts_multiple_test_data_paths(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 20, "train_loss": 1.0},
                "measure_hist": {
                    "iteration_test0_hist": [10.0, 20.0],
                    "iteration_test1_hist": [10.0, 20.0],
                    "map-50_test0_hist": [0.2, 0.4],
                    "map-50_test1_hist": [0.3, 0.5],
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test0 = write_training_files(tmp_path)
    test1 = tmp_path / "test1.txt"
    test1.write_text(test0.read_text(encoding="utf-8"), encoding="utf-8")
    run_root = tmp_path / "runs"

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test0),
            str(test1),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(run_root),
            "--run-name",
            "multi-test",
        ]
    )

    assert code == 0
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["data"] == [
        str(train.resolve()),
        str(test0.resolve()),
        str(test1.resolve()),
    ]
    manifest = json.loads(
        (run_root / "multi-test" / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["options"]["test_data"] == [str(test0), str(test1)]
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert {"test_list_test0", "test_list_test1"} <= {
        event["name"] for event in events if event["event"] == "dataset_check"
    }
    metrics = [event for event in events if event["event"] == "metric"]
    assert [
        (event["name"], event["iteration"], event["value"]) for event in metrics
    ] == [
        ("map-50_test0", 10.0, 0.2),
        ("map-50_test0", 20.0, 0.4),
        ("map-50_test1", 10.0, 0.3),
        ("map-50_test1", 20.0, 0.5),
    ]


def test_train_config_accepts_test_data_list(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test0 = write_training_files(tmp_path)
    test1 = tmp_path / "test1.txt"
    test1.write_text(test0.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"train_data: {train}",
                "test_data:",
                f"  - {test0}",
                f"  - {test1}",
                f"weights: {weights}",
                f"repository: {tmp_path / 'repo'}",
                f"job_dir: {tmp_path / 'runs'}",
                "augmentation:",
                "  crop_size: 32",
                "  geometry:",
                "    prob: 0.3",
                "    zoom_in: false",
                "  noise:",
                "    prob: 0.2",
                "class_weights: [1, 0.5]",
            ]
        ),
        encoding="utf-8",
    )

    code = cli.main(["train", "yolox", "--config", str(cfg)])

    assert code == 0
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["data"] == [
        str(train.resolve()),
        str(test0.resolve()),
        str(test1.resolve()),
    ]
    mllib = train_call[1]["parameters"]["mllib"]
    assert mllib["crop_size"] == 32
    assert mllib["geometry"]["prob"] == 0.3
    assert mllib["geometry"]["zoom_in"] is False
    assert mllib["geometry"]["zoom_out"] is True
    assert mllib["noise"]["prob"] == 0.2
    assert mllib["class_weights"] == [1.0, 0.5]
    saved_config = config.load_config(tmp_path / "repo" / "config.yaml")
    assert saved_config["augmentation"]["crop_size"] == 32
    assert saved_config["augmentation"]["geometry"]["prob"] == 0.3
    assert saved_config["class_weights"] == [1, 0.5]
    assert "dataset_check" in {
        json.loads(line)["event"]
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    }


def test_segformer_training_parameters_accept_augmentation_overrides():
    profile = get_profile("segformer")
    options = profile.train_defaults()
    options["augmentation"] = config.deep_merge(
        options["augmentation"],
        {
            "crop_size": 384,
            "cutout": 0.25,
            "distort": {"prob": 0.4},
        },
    )

    parameters = profile.train_parameters(options)
    mllib = parameters["mllib_parameters"]

    assert mllib["crop_size"] == 384
    assert mllib["cutout"] == 0.25
    assert mllib["distort"]["prob"] == 0.4
    assert mllib["geometry"]["zoom_in"] is True


def test_train_set_overrides_flat_cli_option(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--set",
            "base_lr=0.002",
        ]
    )

    assert code == 0
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["parameters"]["mllib"]["solver"]["base_lr"] == 0.002
    capsys.readouterr()


def test_train_defaults_job_dir_to_repository(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test = write_training_files(tmp_path)
    repository = tmp_path / "repo-default-job-dir"

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(repository),
        ]
    )

    assert code == 0
    assert (repository / "run.json").is_file()
    assert (repository / "metrics.jsonl").is_file()
    manifest = json.loads((repository / "run.json").read_text(encoding="utf-8"))
    assert manifest["run_name"] == "repo-default-job-dir"
    assert manifest["options"]["job_dir"] == str(repository)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert events[0]["run_dir"] == str(repository.resolve())


def test_train_resume_latest_uses_repository_state_without_weights(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 11, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    _weights, train, test = write_training_files(tmp_path)
    repository = write_resume_repository(tmp_path / "repo")
    run_root = tmp_path / "runs"

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--repository",
            str(repository),
            "--job-dir",
            str(run_root),
            "--resume",
            "latest",
            "--dataset-check",
            "none",
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["parameters"]["mllib"]["resume_from"] == "latest"
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["parameters"]["mllib"]["resume"] is True
    assert train_call[1]["parameters"]["mllib"]["resume_from"] == "latest"
    run_json = run_root / "repo" / "run.json"
    assert run_json.is_file()
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert events[0]["run_name"] == "repo"
    assert events[0]["resume"] == "latest"


def test_train_resume_best_replays_metrics_to_visdom_and_skips_old_live_history(
    monkeypatch, tmp_path, capsys
):
    class FakeVisdom:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lines = []
            FakeVisdom.instances.append(self)

        def check_connection(self):
            return True

        def line(self, **kwargs):
            self.lines.append(kwargs)

        def images(self, tensor, **kwargs):
            return None

    class FakeProgress:
        instances = []

        def __init__(self, total):
            self.total = total
            self.updates = []
            self.closed = False
            FakeProgress.instances.append(self)

        def update(self, count):
            self.updates.append(count)

        def close(self):
            self.closed = True

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 15, "train_loss": 0.5},
                "measure_hist": {
                    "iteration_hist": [5.0, 10.0, 15.0],
                    "train_loss_hist": [2.0, 1.0, 0.5],
                    "map-50_hist": [0.1, 0.2, 0.3],
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        training, "history_progress", lambda *, total: FakeProgress(total)
    )
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    _weights, train, test = write_training_files(tmp_path)
    repository = write_resume_repository(tmp_path / "repo", iteration=10)
    (repository / "best_model.txt").write_text(
        "iteration:10\nmap-50:0.2\n",
        encoding="utf-8",
    )
    (repository / "metrics.json").write_text(
        json.dumps(
            {
                "body": {
                    "measure_hist": {
                        "iteration_hist": [5.0, 10.0],
                        "train_loss_hist": [2.0, 1.0],
                        "map-50_hist": [0.1, 0.2],
                        "elapsed_time_ms_hist": [100.0, 200.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--repository",
            str(repository),
            "--job-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "resume-visdom",
            "--resume",
            "best",
            "--visdom",
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["parameters"]["mllib"]["resume_from"] == "best"
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["parameters"]["mllib"]["resume"] is True
    assert train_call[1]["parameters"]["mllib"]["resume_from"] == "best"
    lines = FakeVisdom.instances[0].lines
    assert [
        (line["name"], line["X"].tolist(), line["Y"].tolist()) for line in lines
    ] == [
        ("train_loss", [5.0, 10.0], [2.0, 1.0]),
        ("map-50", [5.0, 10.0], [0.1, 0.2]),
        ("train_loss", [15.0], [0.5]),
        ("map-50", [15.0], [0.3]),
    ]
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    replay = next(event for event in events if event["event"] == "history_replayed")
    assert replay["metrics"] == 6
    assert replay["visdom_metrics"] == 4
    assert replay["visdom_skipped_metrics"] == 2
    assert "history_replay_started" in {event["event"] for event in events}
    assert FakeProgress.instances[0].total == 6
    assert FakeProgress.instances[0].updates == [2, 2, 2]
    assert FakeProgress.instances[0].closed is True
    live_metrics = [event for event in events if event["event"] == "metric"]
    assert [(event["name"], event["iteration"]) for event in live_metrics] == [
        ("train_loss", 15.0),
        ("map-50", 15.0),
    ]


def test_train_yolox_rejects_missing_bbox_file(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    missing_bbox = tmp_path / "missing.txt"
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text(f"{image} {missing_bbox}\n", encoding="utf-8")
    test.write_text(f"{image} {missing_bbox}\n", encoding="utf-8")

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 1
    assert not runtime.calls
    captured = capsys.readouterr()
    assert "bbox file not found" in captured.err


def test_train_yolox_can_skip_dataset_validation(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    missing_bbox = tmp_path / "missing.txt"
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text(f"{image} {missing_bbox}\n", encoding="utf-8")
    test.write_text(f"{image} {missing_bbox}\n", encoding="utf-8")

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--dataset-check",
            "none",
        ]
    )

    assert code == 0
    assert any(call[0] == "train" for call in runtime.calls)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    skipped = [event for event in events if event["event"] == "dataset_check"]
    assert skipped == [
        {
            "event": "dataset_check",
            "name": "dataset_validation",
            "mode": "none",
            "run_id": skipped[0]["run_id"],
            "skipped": True,
            "test_path": str(test),
            "timestamp": skipped[0]["timestamp"],
            "train_path": str(train),
        }
    ]


def test_train_rejects_gpuid_with_no_gpu(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--no-gpu",
            "--gpuid",
            "0",
        ]
    )

    assert code == 1
    assert runtime.calls == []
    assert "--gpuid requires GPU execution" in capsys.readouterr().err


def test_train_emits_only_new_history_points(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "running",
            "body": {
                "measure": {"iteration": 9, "train_loss": 80.0},
                "measure_hist": {
                    "train_loss_hist": [float(index) for index in range(10)]
                },
            },
        },
        {
            "status": "running",
            "body": {
                "measure": {"iteration": 9, "train_loss": 80.0},
                "measure_hist": {
                    "train_loss_hist": [float(index) for index in range(10)]
                },
            },
        },
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 10, "train_loss": 10.0},
                "measure_hist": {
                    "train_loss_hist": [float(index) for index in range(11)]
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 0
    status_calls = [call for call in runtime.calls if call[0] == "status"]
    assert status_calls[0][1]["parameters"]["output"] == {
        "measure_hist": True,
        "max_hist_points": 10000,
    }
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    train_loss = [
        event
        for event in events
        if event["event"] == "metric" and event["name"] == "train_loss"
    ]
    assert [event["iteration"] for event in train_loss] == [
        float(index) for index in range(11)
    ]
    assert [event["value"] for event in train_loss] == [
        float(index) for index in range(11)
    ]


def test_train_uses_iteration_history_for_eval_metrics(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 20, "train_loss": 2.0},
                "measure_hist": {
                    "iteration_hist": [10.0, 20.0],
                    "iteration_test0_hist": [10.0, 20.0],
                    "map-50_hist": [0.25, 0.5],
                    "map-50_test0_hist": [0.2, 0.4],
                    "num_fg_hist": [0.9, 0.8, 0.7],
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 0
    assert not any(call[0] == "set_global_log_level" for call in runtime.calls)
    assert not any(call[0] == "set_log_level" for call in runtime.calls)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    metrics = [event for event in events if event["event"] == "metric"]
    assert "iteration" not in {event["name"] for event in metrics}
    assert [
        (event["iteration"], event["value"])
        for event in metrics
        if event["name"] == "map-50"
    ] == [(10.0, 0.25), (20.0, 0.5)]
    assert [
        (event["iteration"], event["value"])
        for event in metrics
        if event["name"] == "map-50_test0"
    ] == [(10.0, 0.2), (20.0, 0.4)]
    assert [event["iteration"] for event in metrics if event["name"] == "num_fg"] == [
        0.0,
        1.0,
        2.0,
    ]


def test_train_live_terminal_falls_back_to_jsonl_when_not_tty(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setattr(training.sys.stdout, "isatty", lambda: False)
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--terminal",
            "live",
            "--output-format",
            "text",
        ]
    )

    assert code == 0
    assert not any(call[0] == "set_global_log_level" for call in runtime.calls)
    assert not any(call[0] == "set_log_level" for call in runtime.calls)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert "training_status" in {event["event"] for event in events}
    assert "metric" in {event["event"] for event in events}


def test_train_live_terminal_keeps_metric_sinks(monkeypatch, tmp_path, capsys):
    class FakeLiveReporter:
        instances = []

        def __init__(self, *, total_iterations, gpu_ids=None):
            self.total_iterations = total_iterations
            self.gpu_ids = gpu_ids
            self.events = []
            self.closed = False
            FakeLiveReporter.instances.append(self)

        def emit(self, event, **payload):
            record = {"event": event, "timestamp": 1.0, **payload}
            self.events.append(record)
            return record

        def close(self):
            self.closed = True

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 2, "train_loss": 1.0, "map-50": 0.5}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setattr(training.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(training, "LiveTrainingTerminalReporter", FakeLiveReporter)
    weights, train, test = write_training_files(tmp_path)
    run_root = tmp_path / "runs"

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(run_root),
            "--iterations",
            "3",
            "--gpuid",
            "2",
            "--run-name",
            "live-run",
            "--terminal",
            "live",
        ]
    )

    assert code == 0
    assert ("set_global_log_level", "warn") in runtime.calls
    assert ("set_log_level", "python-yolox-train", "warn") in runtime.calls
    assert runtime.calls.index(("set_global_log_level", "warn")) < next(
        index for index, call in enumerate(runtime.calls) if call[0] == "create"
    )
    assert FakeLiveReporter.instances[0].total_iterations == 3
    assert FakeLiveReporter.instances[0].gpu_ids == 2
    assert FakeLiveReporter.instances[0].closed is True
    assert "training_status" in {
        event["event"] for event in FakeLiveReporter.instances[0].events
    }
    metric_lines = (
        (run_root / "live-run" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    metrics = [json.loads(line) for line in metric_lines]
    assert {metric["name"] for metric in metrics} == {"train_loss", "map-50"}
    assert capsys.readouterr().out == ""


def test_live_training_terminal_reporter_renders_progress_losses_and_metrics():
    class FakeGpuMonitor:
        def snapshot(self):
            return ["1 util=88% mem=13.2/24.0GB (55%)"]

    stream = io.StringIO()
    reporter = LiveTrainingTerminalReporter(
        total_iterations=10,
        gpu_monitor=FakeGpuMonitor(),
        stream=stream,
        force_terminal=True,
    )
    assert stream.getvalue() == ""
    reporter.emit("run_started", run_id="run")
    assert stream.getvalue() == ""
    reporter.emit(
        "training_status",
        status="running",
        measure={
            "iteration": 4,
            "remain_time_str": "00:01:30",
            "train_loss": 1.25,
            "cls_loss": 0.5,
            "test_active": 1,
            "test_set_index": 1,
            "test_sets_total": 3,
            "test_processed": 20,
            "test_total": 40,
            "map-50": 0.75,
            "elapsed_time_ms": 123.0,
        },
    )
    reporter.emit("metric", name="acc", value=0.9, iteration=4)
    reporter.close()

    output = strip_ansi(stream.getvalue())
    lines = output.splitlines()
    gpu_line = next(index for index, line in enumerate(lines) if "gpu 1 util" in line)
    train_line = next(
        index for index, line in enumerate(lines) if "train" in line and "4/10" in line
    )
    loss_line = next(
        index for index, line in enumerate(lines) if "train_loss=1.25" in line
    )
    metrics_line = next(
        index for index, line in enumerate(lines) if "map-50=0.75" in line
    )
    assert gpu_line < train_line
    assert any("─" in line for line in lines[loss_line + 1 : metrics_line])
    assert "1 util=88% mem=13.2/24.0GB (55%)" in output
    assert "train" in output
    assert "4/10" in output
    assert "train_loss=1.25" in output
    assert "cls_loss=0.5" in output
    assert "test set 2/3" in output
    assert "20/40" in output
    assert "map-50=0.75" in output
    assert "acc=0.9" in output
    assert "elapsed_time_ms" not in output


def test_live_training_terminal_reporter_keeps_completed_test_progress_visible():
    stream = io.StringIO()
    reporter = LiveTrainingTerminalReporter(
        total_iterations=10,
        gpu_monitor=None,
        stream=stream,
        force_terminal=True,
    )

    reporter.emit(
        "training_status",
        status="running",
        measure={
            "iteration": 5,
            "train_loss": 1.0,
            "test_active": 0,
            "test_set_index": 0,
            "test_sets_total": 1,
            "test_processed": 12,
            "test_total": 12,
        },
    )
    reporter.close()

    output = strip_ansi(stream.getvalue())
    assert "test set 1/1" in output
    assert "12/12" in output


def test_live_training_terminal_reporter_waits_for_first_loss_before_rendering():
    stream = io.StringIO()
    reporter = LiveTrainingTerminalReporter(
        total_iterations=10,
        gpu_monitor=None,
        stream=stream,
        force_terminal=True,
    )
    reporter.emit(
        "training_status",
        status="running",
        measure={"iteration": 1, "remain_time_str": "00:01:00"},
    )
    reporter.emit("metric", name="map-50", value=0.4, iteration=1)
    assert stream.getvalue() == ""

    reporter.emit(
        "training_status",
        status="running",
        measure={"iteration": 2, "train_loss": 1.5, "map-50": 0.5},
    )
    reporter.close()

    output = stream.getvalue()
    assert "2/10" in output
    assert "train_loss=1.5" in output
    assert "map-50=0.5" in output


def test_live_training_terminal_reporter_renders_sink_warning():
    stream = io.StringIO()
    reporter = LiveTrainingTerminalReporter(
        total_iterations=10,
        gpu_monitor=None,
        stream=stream,
        force_terminal=True,
    )
    reporter.emit(
        "training_status",
        status="running",
        measure={"iteration": 2, "train_loss": 1.5},
    )
    reporter.emit(
        "sink_warning",
        sink="VisdomResultSink",
        message="transient prediction failure",
    )
    reporter.close()

    output = stream.getvalue()
    assert "warning" in output
    assert "VisdomResultSink: transient prediction failure" in output


def test_explicit_run_name_collision_fails_before_training(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime()
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    weights, train, test = write_training_files(tmp_path)
    run_root = tmp_path / "runs"
    (run_root / "existing").mkdir(parents=True)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(run_root),
            "--run-name",
            "existing",
        ]
    )

    assert code == 1
    assert runtime.calls == []
    assert "FileExistsError" in capsys.readouterr().err


def test_train_visdom_sink_plots_losses_and_metrics(monkeypatch, tmp_path, capsys):
    class FakeVisdom:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lines = []
            self.saved = []
            FakeVisdom.instances.append(self)

        def check_connection(self):
            return True

        def line(self, **kwargs):
            self.lines.append(kwargs)

        def images(self, tensor, **kwargs):
            return None

        def save(self, envs):
            self.saved.append(envs)

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {
                "measure": {
                    "iteration": 5,
                    "train_loss": 1.25,
                    "conf_loss": 0.5,
                    "map": 0.7,
                    "map-05": 0.9,
                    "map-50": 0.4,
                    "map-90": 0.1,
                    "elapsed_time_ms": 1234.0,
                    "test_active": 1,
                    "test_processed": 2,
                    "fp": float("nan"),
                }
            },
        }
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "visdom-run",
            "--visdom",
            "--no-visdom-results",
            "--visdom-save",
        ]
    )

    assert code == 0
    assert FakeVisdom.instances[0].kwargs["env"] == "visdom-run"
    lines = FakeVisdom.instances[0].lines
    assert {line["win"] for line in lines} == {
        "loss-train-loss",
        "loss-conf-loss",
        "metric-map",
        "metric-map-05",
        "metric-map-50",
        "metric-map-90",
    }
    assert {line["name"] for line in lines} == {
        "train_loss",
        "conf_loss",
        "map",
        "map-05",
        "map-50",
        "map-90",
    }
    assert "fp" not in {line["name"] for line in lines}
    assert "elapsed_time_ms" not in {line["name"] for line in lines}
    assert "test_active" not in {line["name"] for line in lines}
    assert "test_processed" not in {line["name"] for line in lines}
    assert [line["X"].tolist() for line in lines] == [
        [5.0],
        [5.0],
        [5.0],
        [5.0],
        [5.0],
        [5.0],
    ]
    assert FakeVisdom.instances[0].saved == [["visdom-run"]]
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert "sink_warning" not in {event["event"] for event in events}


def test_train_visdom_uploads_detection_results_on_eval_metric(
    monkeypatch, tmp_path, capsys
):
    class FakeVisdom:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lines = []
            self.image_grids = []
            FakeVisdom.instances.append(self)

        def check_connection(self):
            return True

        def line(self, **kwargs):
            self.lines.append(kwargs)

        def images(self, tensor, **kwargs):
            self.image_grids.append((tensor, kwargs))

    rendered_class_counts = []
    rendered_paths = []

    def fake_result_image_array(model, image_path, prediction, *, image_size):
        rendered_class_counts.append(len(prediction["classes"]))
        rendered_paths.append(image_path)
        return np.zeros((3, 8, 8), dtype=np.uint8)

    monkeypatch.setattr(results, "result_image_array", fake_result_image_array)

    runtime = FakeRuntime()
    runtime.statuses = [
        {"status": "running", "body": {"measure": {"iteration": 1, "train_loss": 2.0}}},
        {
            "status": "running",
            "body": {
                "measure": {"iteration": 100, "train_loss": 1.0},
                "measure_hist": {
                    "iteration_test0_hist": [100.0],
                    "map-50_test0_hist": [0.2],
                },
                "test_predictions": {
                    "test0": {
                        "iteration": 100.0,
                        "samples": [
                            {
                                "index": 1,
                                "classes": [
                                    {
                                        "cat": "object",
                                        "prob": 0.9,
                                        "bbox": {
                                            "xmin": 0,
                                            "ymin": 0,
                                            "xmax": 1,
                                            "ymax": 1,
                                        },
                                    },
                                    {
                                        "cat": "object",
                                        "prob": 0.8,
                                        "bbox": {
                                            "xmin": 2,
                                            "ymin": 2,
                                            "xmax": 3,
                                            "ymax": 3,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                },
            },
        },
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 100, "train_loss": 1.0},
                "measure_hist": {
                    "iteration_test0_hist": [100.0],
                    "map-50_test0_hist": [0.2],
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)
    image2 = tmp_path / "image2.jpg"
    Image.new("RGB", (8, 8), color="white").save(image2)
    target2 = tmp_path / "target2.txt"
    target2.write_text("1 0 0 4 4\n", encoding="utf-8")
    with test.open("a", encoding="utf-8") as stream:
        stream.write(f"{image2} {target2}\n")

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--visdom",
            "--visdom-results-count",
            "1",
        ]
    )

    assert code == 0
    assert not any(call[0] == "predict" for call in runtime.calls)
    train_call = next(call for call in runtime.calls if call[0] == "train")
    assert train_call[1]["parameters"]["output"]["test_predictions"] == {
        "enabled": True,
        "confidence_threshold": 0.1,
        "sample_count": 1,
        "sample_seed": 12345,
    }
    status_calls = [call for call in runtime.calls if call[0] == "status"]
    assert status_calls[0][1]["parameters"]["output"]["test_predictions"] is True
    image_grids = FakeVisdom.instances[0].image_grids
    assert len(image_grids) == 1
    assert rendered_class_counts == [2]
    assert rendered_paths == [image2.resolve()]
    artifact_dir = tmp_path / "repo" / "visdom-results" / "iteration-000100" / "test0"
    assert (artifact_dir / "sample-000001.png").is_file()
    artifact = json.loads((artifact_dir / "sample-000001.json").read_text())
    assert artifact["sample_index"] == 1
    assert artifact["image"] == str(image2.resolve())
    assert len(artifact["prediction"]["classes"]) == 2
    tensor, kwargs = image_grids[0]
    assert tensor.shape == (1, 3, 8, 8)
    assert kwargs["win"] == "results-detection-test0"
    assert kwargs["opts"]["title"] == "detection test0 results iteration 100"
    assert kwargs["opts"]["jpgquality"] == 90
    assert "sink_warning" not in {
        json.loads(line)["event"]
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    }


def test_train_visdom_results_can_be_disabled(monkeypatch, tmp_path, capsys):
    class FakeVisdom:
        instances = []

        def __init__(self, **kwargs):
            self.image_grids = []
            FakeVisdom.instances.append(self)

        def check_connection(self):
            return True

        def line(self, **kwargs):
            return None

        def images(self, tensor, **kwargs):
            self.image_grids.append((tensor, kwargs))

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 100, "train_loss": 1.0},
                "measure_hist": {
                    "iteration_test0_hist": [100.0],
                    "map-50_test0_hist": [0.2],
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--visdom",
            "--no-visdom-results",
        ]
    )

    assert code == 0
    assert not any(call[0] == "predict" for call in runtime.calls)
    assert FakeVisdom.instances[0].image_grids == []
    capsys.readouterr()


def test_train_visdom_results_waits_for_backend_prediction_payload(
    monkeypatch, tmp_path, capsys
):
    class FakeVisdom:
        instances = []

        def __init__(self, **kwargs):
            self.image_grids = []
            FakeVisdom.instances.append(self)

        def check_connection(self):
            return True

        def line(self, **kwargs):
            return None

        def images(self, tensor, **kwargs):
            self.image_grids.append((tensor, kwargs))

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "running",
            "body": {
                "measure": {"iteration": 100, "train_loss": 1.0},
                "measure_hist": {
                    "iteration_test0_hist": [100.0],
                    "map-50_test0_hist": [0.2],
                },
            },
        },
        {
            "status": "running",
            "body": {
                "measure": {"iteration": 200, "train_loss": 0.8},
                "measure_hist": {
                    "iteration_test0_hist": [100.0, 200.0],
                    "map-50_test0_hist": [0.2, 0.3],
                },
                "test_predictions": {
                    "test0": {
                        "iteration": 200.0,
                        "samples": [
                            {
                                "index": 0,
                                "classes": [
                                    {
                                        "cat": "object",
                                        "prob": 0.9,
                                        "bbox": {
                                            "xmin": 0,
                                            "ymin": 0,
                                            "xmax": 1,
                                            "ymax": 1,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        },
        {
            "status": "finished",
            "body": {
                "measure": {"iteration": 300, "train_loss": 0.7},
                "test_predictions": {
                    "test0": {
                        "iteration": 300.0,
                        "samples": [
                            {
                                "index": 0,
                                "classes": [
                                    {
                                        "cat": "object",
                                        "prob": 0.95,
                                        "bbox": {
                                            "xmin": 2,
                                            "ymin": 2,
                                            "xmax": 3,
                                            "ymax": 3,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--visdom",
            "--visdom-results-count",
            "1",
        ]
    )

    assert code == 0
    assert not any(call[0] == "predict" for call in runtime.calls)
    assert len(FakeVisdom.instances[0].image_grids) == 2
    titles = [
        kwargs["opts"]["title"]
        for _tensor, kwargs in FakeVisdom.instances[0].image_grids
    ]
    assert titles == [
        "detection test0 results iteration 200",
        "detection test0 results iteration 300",
    ]
    warnings = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip() and json.loads(line)["event"] == "sink_warning"
    ]
    assert warnings[0]["sink"] == "VisdomResultSink"
    assert "no test_predictions payload" in warnings[0]["message"]


def test_segmentation_result_image_is_rgb_chw_overlay(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image)

    array = results.result_image_array(
        "segformer",
        image,
        {
            "imgsize": {"width": 2, "height": 2},
            "vals": [0, 1, 1, 0],
        },
        image_size=(2, 2),
    )

    assert array.shape == (3, 2, 2)
    assert array.dtype.name == "uint8"


def test_segmentation_overlay_uses_distinct_class_colors(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (3, 1), color="white").save(image)

    mask, overlay = segmentation_overlay_images(
        image,
        {
            "imgsize": {"width": 3, "height": 1},
            "vals": [0, 1, 2],
        },
    )

    mask_rgb = mask.convert("RGB")
    assert mask_rgb.getpixel((0, 0)) == (0, 0, 0)
    assert mask_rgb.getpixel((1, 0)) != (0, 0, 0)
    assert mask_rgb.getpixel((2, 0)) != (0, 0, 0)
    assert mask_rgb.getpixel((1, 0)) != mask_rgb.getpixel((2, 0))
    assert overlay.getpixel((0, 0)) == (255, 255, 255)
    assert overlay.getpixel((1, 0)) != overlay.getpixel((2, 0))


def test_detection_result_image_draws_normalized_bbox_on_original_size(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (10, 8), color="white").save(image)

    array = results.result_image_array(
        "yolox",
        image,
        {
            "classes": [
                {
                    "cat": "object",
                    "prob": 0.9,
                    "bbox": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.8, "ymax": 0.8},
                }
            ]
        },
        image_size=(640, 640),
    )

    assert array.shape == (3, 8, 10)
    assert (array != 255).any()


def test_detection_result_image_uses_prediction_coordinate_size(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 10), color="white").save(image)

    array = results.result_image_array(
        "yolox",
        image,
        {
            "imgsize": {"width": 10, "height": 5},
            "classes": [
                {
                    "cat": "object",
                    "prob": 0.9,
                    "bbox": {"xmin": 2, "ymin": 1, "xmax": 4, "ymax": 3},
                }
            ],
        },
        image_size=(20, 10),
    )

    assert tuple(array[:, 1, 2]) == (255, 255, 255)
    assert tuple(array[:, 2, 4]) != (255, 255, 255)


def test_visdom_result_images_are_resized_to_max_side():
    image = np.zeros((3, 600, 1200), dtype=np.uint8)

    resized = results.visdom_result_image_arrays([image])

    assert len(resized) == 1
    assert resized[0].shape == (3, 256, 512)


def test_visdom_keypoint_results_render_points_on_source_image(tmp_path):
    image_path = tmp_path / "pose.jpg"
    Image.new("RGB", (20, 20), color="white").save(image_path)
    prediction = {
        "index": 0,
        "imgsize": {"width": 20, "height": 20},
        "classes": [
            {
                "cat": "1",
                "prob": 1.0,
                "keypoints": [
                    {"x": 10.0, "y": 10.0, "prob": 0.9, "valid": True},
                    {"x": -1.0, "y": -1.0, "prob": 0.0, "valid": False},
                ],
            }
        ],
    }

    rendered = results.result_image_array(
        "keypoint", image_path, prediction, image_size=(20, 20)
    )

    assert rendered.shape == (3, 20, 20)
    assert not np.all(rendered[:, 10, 10] == 255)


def test_visdom_result_images_are_padded_without_distorting_aspect_ratio():
    wide = np.full((3, 600, 1200), 32, dtype=np.uint8)
    tall = np.full((3, 1200, 600), 64, dtype=np.uint8)

    resized = results.visdom_result_image_arrays([wide, tall])

    assert [image.shape for image in resized] == [(3, 512, 512), (3, 512, 512)]
    assert np.all(resized[0][:, :256, :] == 32)
    assert np.all(resized[0][:, 256:, :] == 255)
    assert np.all(resized[1][:, :, :256] == 64)
    assert np.all(resized[1][:, :, 256:] == 255)


def test_detection_overlay_uses_stable_class_colors(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 20), color="white").save(image)

    rendered = detection_overlay_image(
        image,
        {
            "classes": [
                {
                    "cat": "ring",
                    "bbox": {"xmin": 1, "ymin": 1, "xmax": 5, "ymax": 5},
                },
                {
                    "cat": "marker",
                    "bbox": {"xmin": 8, "ymin": 1, "xmax": 12, "ymax": 5},
                },
                {
                    "cat": "ring",
                    "bbox": {"xmin": 1, "ymin": 8, "xmax": 5, "ymax": 12},
                },
            ]
        },
    )

    assert rendered.getpixel((1, 1)) == rendered.getpixel((1, 8))
    assert rendered.getpixel((1, 1)) != rendered.getpixel((8, 1))


def test_visdom_sink_groups_test_set_metrics_on_base_metric_window():
    class FakeClient:
        def __init__(self):
            self.lines = []

        def line(self, **kwargs):
            self.lines.append(kwargs)

    client = FakeClient()
    sink = VisdomMetricSink(
        env="multi-test",
        server="http://localhost",
        port=8097,
        base_url="/",
        client=client,
    )

    sink.write({"name": "map-50_test0", "value": 0.2, "iteration": 10})
    sink.write({"name": "map-50_test1", "value": 0.3, "iteration": 10})
    sink.write({"name": "map-50_test0", "value": 0.4, "iteration": 20})
    sink.write({"name": "map-50_test1", "value": 0.5, "iteration": 20})
    written = sink.write_many(
        [
            {"name": "map-90_test0", "value": 0.1, "iteration": 10},
            {"name": "map-90_test0", "value": 0.2, "iteration": 20},
            {"name": "map-90_test1", "value": 0.15, "iteration": 10},
            {"name": "map-90_test1", "value": 0.25, "iteration": 20},
        ]
    )

    assert written == 4
    assert [
        (line["win"], line["name"], line["X"].tolist(), line["Y"].tolist())
        for line in client.lines
    ] == [
        ("metric-map-50", "test0", [10.0], [0.2]),
        ("metric-map-50", "test1", [10.0], [0.3]),
        ("metric-map-50", "test0", [20.0], [0.4]),
        ("metric-map-50", "test1", [20.0], [0.5]),
        ("metric-map-90", "test0", [10.0, 20.0], [0.1, 0.2]),
        ("metric-map-90", "test1", [10.0, 20.0], [0.15, 0.25]),
    ]
    assert [line["update"] for line in client.lines] == [
        None,
        "append",
        "append",
        "append",
        None,
        "append",
    ]
    assert [line["opts"]["legend"] for line in client.lines] == [
        ["test0"],
        ["test0", "test1"],
        ["test0", "test1"],
        ["test0", "test1"],
        ["test0", "test1"],
        ["test0", "test1"],
    ]


def test_train_visdom_unreachable_can_fail_fast(monkeypatch, tmp_path, capsys):
    class FakeVisdom:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def check_connection(self):
            return False

    runtime = FakeRuntime()
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--visdom",
            "--no-visdom-offline-ok",
        ]
    )

    assert code == 1
    assert runtime.calls == []
    assert "Visdom server is unreachable" in capsys.readouterr().err


def test_train_visdom_unreachable_warns_and_continues(monkeypatch, tmp_path, capsys):
    class FakeVisdom:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def check_connection(self):
            return False

    runtime = FakeRuntime()
    runtime.statuses = [
        {
            "status": "finished",
            "body": {"measure": {"iteration": 1, "train_loss": 1.0}},
        },
    ]
    monkeypatch.setattr(
        training.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    monkeypatch.setattr(training.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "visdom", SimpleNamespace(Visdom=FakeVisdom))
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
            "--visdom",
        ]
    )

    assert code == 0
    assert any(call[0] == "train" for call in runtime.calls)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert "sink_warning" in {event["event"] for event in events}


def test_train_reports_native_import_error(monkeypatch, tmp_path, capsys):
    def fail_runtime():
        raise ImportError("deepdetect native extension 'deepdetect._native' missing")

    monkeypatch.setattr(training.deepdetect, "DeepDetect", fail_runtime)
    weights, train, test = write_training_files(tmp_path)

    code = cli.main(
        [
            "train",
            "yolox",
            "--train-data",
            str(train),
            "--test-data",
            str(test),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--job-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert "ImportError: deepdetect native extension" in captured.err
    assert "Traceback" not in captured.err


def test_infer_yolox_batches_and_benchmark(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    monkeypatch.setattr(
        inference.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    image1 = tmp_path / "one.png"
    image2 = tmp_path / "two.png"
    Image.new("RGB", (2, 2), "white").save(image1)
    Image.new("RGB", (2, 2), "white").save(image2)

    code = cli.main(
        [
            "infer",
            "yolox",
            str(image1),
            str(image2),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
            "--batch-size",
            "2",
            "--width",
            "416",
            "--height",
            "384",
            "--gpuid",
            "2",
            "--benchmark",
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["parameters"]["input"]["width"] == 416
    assert create[2]["parameters"]["input"]["height"] == 384
    assert create[2]["parameters"]["mllib"]["gpu"] is True
    assert create[2]["parameters"]["mllib"]["gpuid"] == 2
    predict_calls = [call for call in runtime.calls if call[0] == "predict"]
    assert len(predict_calls) == 1
    assert len(predict_calls[0][1]["data"]) == 2
    assert predict_calls[0][1]["parameters"]["input"]["width"] == 416
    assert predict_calls[0][1]["parameters"]["input"]["height"] == 384
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events].count("prediction") == 2
    assert events[-1]["event"] == "benchmark"


def test_infer_vitpose_aligns_bbox_files_with_image_batches(
    monkeypatch, tmp_path, capsys
):
    runtime = FakeRuntime()
    monkeypatch.setattr(
        inference.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    images = []
    boxes = []
    for index in range(2):
        image = tmp_path / f"pose-{index}.png"
        bbox = tmp_path / f"pose-{index}.txt"
        Image.new("RGB", (8, 8), "white").save(image)
        bbox.write_text("1 1 1 7 7\n", encoding="utf-8")
        images.append(image)
        boxes.append(bbox)

    code = cli.main(
        [
            "infer",
            "vitpose",
            *(str(path) for path in images),
            "--bbox-files",
            *(str(path) for path in boxes),
            "--repository",
            str(tmp_path / "repo"),
            "--batch-size",
            "1",
        ]
    )

    assert code == 0
    predict_calls = [call for call in runtime.calls if call[0] == "predict"]
    assert len(predict_calls) == 2
    for index, call in enumerate(predict_calls):
        assert call[1]["parameters"]["input"]["bbox_files"] == [
            str(boxes[index].resolve())
        ]


def test_infer_segformer_keeps_default_size(monkeypatch, tmp_path, capsys):
    runtime = FakeRuntime()
    runtime.predict = lambda request: (
        runtime.calls.append(("predict", json.loads(request)))
        or response(
            body={
                "predictions": [
                    {
                        "imgsize": {"width": 480, "height": 480},
                        "vals": [0] * (480 * 480),
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        inference.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=runtime)
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    image = tmp_path / "one.png"
    Image.new("RGB", (2, 2), "white").save(image)

    code = cli.main(
        [
            "infer",
            "segformer",
            str(image),
            "--weights",
            str(weights),
            "--repository",
            str(tmp_path / "repo"),
        ]
    )

    assert code == 0
    create = next(call for call in runtime.calls if call[0] == "create")
    assert create[2]["parameters"]["input"]["width"] == 480
    assert create[2]["parameters"]["input"]["height"] == 480
    predict = next(call for call in runtime.calls if call[0] == "predict")
    assert predict[1]["parameters"]["input"]["width"] == 480
    assert predict[1]["parameters"]["input"]["height"] == 480
    capsys.readouterr()


def test_job_status_reads_manifest(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "r1", "status": "finished", "last_status": {"job": 7}}),
        encoding="utf-8",
    )

    assert cli.main(["job", "status", str(run_dir)]) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "training_status"
    assert event["last_status"]["job"] == 7


def test_inspect_models_is_discoverable(capsys):
    assert cli.main(["inspect", "models"]) == 0
    names = {
        json.loads(line)["name"]
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    }
    assert {"yolox", "segformer"} <= names
