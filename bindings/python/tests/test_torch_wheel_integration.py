from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import deepdetect


FIXTURES_ENV = "DEEPDETECT_TORCH_FIXTURES"
ITERATIONS_ENV = "DEEPDETECT_TORCH_TEST_ITERATIONS"
TORCH_LR = 1e-5


def fixtures_root() -> Path:
    value = os.environ.get(FIXTURES_ENV)
    if not value:
        pytest.skip(f"{FIXTURES_ENV} is not set")
    root = Path(value).resolve()
    if not root.exists():
        pytest.skip(f"{FIXTURES_ENV} does not exist: {root}")
    return root


def iteration_count(default: int) -> int:
    value = os.environ.get(ITERATIONS_ENV)
    return int(value) if value else default


def copy_fixture(root: Path, name: str, tmp_path: Path) -> Path:
    source = root / name
    if not source.exists():
        pytest.skip(f"missing Torch fixture: {source}")
    destination = tmp_path / name
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "train.lmdb",
            "test_0.lmdb",
            "checkpoint-*.pt",
            "checkpoint-*.ptw",
            "solver-*.json",
        ),
    )
    rewrite_fixture_lists(destination, name)
    return destination


def rewrite_fixture_lists(destination: Path, name: str) -> None:
    original_prefix = f"../examples/torch/{name}"
    replacement = str(destination)
    for list_file in destination.rglob("*.txt"):
        try:
            contents = list_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = contents.replace(original_prefix, replacement)
        if rewritten != contents:
            list_file.write_text(rewritten, encoding="utf-8")


def configure_torch_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = pytest.importorskip("torch")
    torch.manual_seed(1235)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True


def cleanup_service(service: deepdetect.Service) -> None:
    try:
        service.delete(clear="full")
    except Exception:
        pass


def assert_metric_range(measure: dict[str, Any], name: str, *, upper: float = 1.0) -> None:
    assert name in measure
    assert float(measure[name]) <= upper


def test_resnet_image_training_from_wheel(tmp_path: Path) -> None:
    configure_torch_determinism()
    root = fixtures_root()
    repo = copy_fixture(root, "resnet50_training_torch241_small", tmp_path)
    train_data = repo / "train"
    test_data = repo / "test"
    test_image = repo / "train" / "cats" / "cat.10097.jpg"
    iterations = iteration_count(200)

    dd = deepdetect.DeepDetect()
    use_gpu = bool(dd.build_info.get("cuda"))
    service = dd.create_service(
        "py-wheel-resnet",
        description="image",
        model={"repository": str(repo)},
        mllib="torch",
        input_parameters={
            "connector": "image",
            "width": 256,
            "height": 256,
            "db": True,
        },
        mllib_parameters={
            "nclasses": 2,
            "finetuning": True,
            "gpu": use_gpu,
        },
    )
    try:
        body = service.train(
            [str(train_data), str(test_data)],
            mllib_parameters={
                "solver": {
                    "iterations": iterations,
                    "base_lr": TORCH_LR,
                    "iter_size": 4,
                    "solver_type": "ADAM",
                    "test_interval": iterations,
                },
                "net": {"batch_size": 4},
                "resume": False,
                "mirror": True,
                "rotate": True,
                "crop_size": 224,
                "test_crop_samples": 10,
                "cutout": 0.5,
                "geometry": {
                    "prob": 0.1,
                    "persp_horizontal": True,
                    "persp_vertical": True,
                    "zoom_in": True,
                    "zoom_out": True,
                    "pad_mode": "constant",
                },
                "noise": {"prob": 0.01},
                "distort": {"prob": 0.01},
                "dataloader_threads": 4,
            },
            input_parameters={"seed": 12345, "db": True, "shuffle": True},
            output_parameters={"measure": ["f1", "acc"]},
        )
        measure = body["measure"]
        assert_metric_range(measure, "acc")
        assert_metric_range(measure, "f1")

        prediction = service.predict(
            [str(test_image)],
            output_parameters={"best": 1},
        )
        assert prediction["predictions"]
        assert float(prediction["predictions"][0]["classes"][0]["prob"]) > 0.0
    finally:
        cleanup_service(service)


@pytest.mark.skipif(
    os.environ.get("DEEPDETECT_EXPECTED_CUDA") == "false",
    reason="YOLOX wheel integration test requires a CUDA wheel",
)
def test_yolox_object_detection_training_from_wheel(tmp_path: Path) -> None:
    configure_torch_determinism()
    root = fixtures_root()
    repo = copy_fixture(root, "yolox_train_torch", tmp_path)
    fasterrcnn = copy_fixture(root, "fasterrcnn_train_torch111", tmp_path)
    train_data = fasterrcnn / "train.txt"
    test_data = fasterrcnn / "test.txt"
    predict_image = fasterrcnn / "imgs" / "la_melrose_ave-000020.jpg"

    dd = deepdetect.DeepDetect()
    if not dd.build_info.get("cuda"):
        pytest.skip("YOLOX wheel integration test requires a CUDA wheel")
    service = dd.create_service(
        "py-wheel-yolox",
        description="yolox",
        model={"repository": str(repo)},
        mllib="torch",
        input_parameters={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
            "db": True,
        },
        mllib_parameters={"template": "yolox", "gpu": True, "nclasses": 2},
    )
    try:
        body = service.train(
            [str(train_data), str(test_data)],
            mllib_parameters={
                "solver": {
                    "iterations": iteration_count(3),
                    "iter_size": 2,
                    "solver_type": "ADAM",
                    "test_interval": 200,
                },
                "net": {
                    "batch_size": 2,
                    "test_batch_size": 2,
                    "reg_weight": 0.5,
                },
                "resume": False,
                "mirror": True,
                "rotate": True,
                "crop_size": 512,
                "test_crop_samples": 10,
                "cutout": 0.1,
                "geometry": {
                    "prob": 0.1,
                    "persp_horizontal": True,
                    "persp_vertical": True,
                    "zoom_in": True,
                    "zoom_out": True,
                    "pad_mode": "constant",
                },
                "noise": {"prob": 0.01},
                "distort": {"prob": 0.01},
            },
            input_parameters={"seed": 12347, "db": True, "shuffle": True},
            output_parameters={"measure": ["map-05", "map-50", "map-90"]},
        )
        measure = body["measure"]
        for key in ("map", "map-05", "map-50", "map-90"):
            assert_metric_range(measure, key)
        for key in ("iou_loss", "conf_loss", "cls_loss", "l1_loss", "train_loss"):
            assert key in measure
        expected_train_loss = (
            float(measure["iou_loss"]) * 0.5
            + float(measure["cls_loss"])
            + float(measure["l1_loss"])
            + float(measure["conf_loss"])
        )
        assert abs(float(measure["train_loss"]) - expected_train_loss) < 0.0001

        prediction = service.predict(
            [str(predict_image)],
            input_parameters={"height": 640, "width": 640},
            output_parameters={"bbox": True, "confidence_threshold": 0.8},
        )
        assert "predictions" in prediction
    finally:
        cleanup_service(service)


@pytest.mark.skipif(
    os.environ.get("DEEPDETECT_EXPECTED_CUDA") == "false",
    reason="SegFormer wheel integration test requires a CUDA wheel",
)
def test_segformer_segmentation_training_from_wheel(tmp_path: Path) -> None:
    configure_torch_determinism()
    root = fixtures_root()
    repo = copy_fixture(root, "segformer_training_torch", tmp_path)
    deeplabv3 = copy_fixture(root, "deeplabv3_training_torch", tmp_path)
    camvid = deeplabv3 / "CamVid_square"
    train_data = camvid / "train.txt"
    test_data = camvid / "test50.txt"
    test_image = camvid / "test" / "Seq05VD_f00330.png"
    iterations = iteration_count(200)

    dd = deepdetect.DeepDetect()
    if not dd.build_info.get("cuda"):
        pytest.skip("SegFormer wheel integration test requires a CUDA wheel")
    service = dd.create_service(
        "py-wheel-segformer",
        description="image",
        model={"repository": str(repo)},
        mllib="torch",
        input_parameters={
            "connector": "image",
            "width": 480,
            "height": 480,
            "db": True,
            "segmentation": True,
        },
        mllib_parameters={"nclasses": 13, "gpu": True, "segmentation": True},
    )
    try:
        body = service.train(
            [str(train_data), str(test_data)],
            mllib_parameters={
                "solver": {
                    "iterations": iterations,
                    "base_lr": TORCH_LR,
                    "iter_size": 1,
                    "solver_type": "ADAM",
                    "test_interval": 100,
                },
                "net": {"batch_size": 4},
                "resume": False,
                "mirror": True,
                "rotate": True,
                "crop_size": 224,
                "cutout": 0.5,
                "geometry": {
                    "prob": 0.1,
                    "persp_horizontal": True,
                    "persp_vertical": True,
                    "zoom_in": True,
                    "zoom_out": True,
                    "pad_mode": "constant",
                },
                "noise": {"prob": 0.01},
                "distort": {"prob": 0.01},
            },
            input_parameters={
                "seed": 12345,
                "db": True,
                "shuffle": True,
                "segmentation": True,
                "scale": 0.0039,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            output_parameters={"measure": ["meaniou", "acc"]},
        )
        measure = body["measure"]
        assert_metric_range(measure, "meanacc")
        if iterations >= 200:
            assert float(measure["meanacc"]) >= 0.003
        assert_metric_range(measure, "meaniou")

        prediction = service.predict(
            [str(test_image)],
            input_parameters={
                "height": 480,
                "width": 480,
                "scale": 0.0039,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            output_parameters={"segmentation": True, "confidences": ["best"]},
        )
        assert prediction["predictions"]
    finally:
        cleanup_service(service)
