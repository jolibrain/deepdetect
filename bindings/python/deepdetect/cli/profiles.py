from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/data1/beniz/models/dd")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    task: str
    description: str
    backend: str
    default_weights: Path | None
    default_repository: Path
    default_service_name: str
    default_nclasses: int
    requires_weights: bool
    service_input: dict[str, Any]
    service_mllib: dict[str, Any]
    train_input: dict[str, Any]
    train_mllib: dict[str, Any]
    train_output: dict[str, Any]
    predict_input: dict[str, Any]
    predict_output: dict[str, Any]

    @property
    def default_width(self) -> int:
        return int(self.service_input["width"])

    @property
    def default_height(self) -> int:
        return int(self.service_input["height"])

    def train_defaults(self) -> dict[str, Any]:
        return {
            "weights": self.default_weights,
            "repository": self.default_repository,
            "service_name": self.default_service_name,
            "nclasses": self.default_nclasses,
            "nkeypoints": self.service_mllib.get(
                "nkeypoints", self.train_mllib.get("nkeypoints")
            ),
            "max_objects": self.service_mllib.get(
                "max_objects", self.train_mllib.get("max_objects")
            ),
            "width": self.default_width,
            "height": self.default_height,
            "iterations": 100,
            "batch_size": int(
                self.train_mllib.get("net", {}).get(
                    "batch_size", 2 if self.task == "detection" else 4
                )
            ),
            "iter_size": int(self.train_mllib.get("solver", {}).get("iter_size", 1)),
            "augmentation": _augmentation_defaults(self.train_mllib),
            "service_mllib": None,
            "class_weights": None,
            "base_lr": float(
                self.train_mllib.get("solver", {}).get("base_lr", 0.0001)
            ),
            "test_interval": 100,
            "gpu": False,
            "gpuid": None,
            "sync": False,
            "poll_interval": 0.5,
            "timeout": None,
            "job_dir": None,
            "run_name": None,
            "resume": None,
            "output_format": "jsonl",
            "terminal": "verbose",
            "dataset_check": "full",
            "skip_mask_validation": False,
            "visdom": False,
            "visdom_server": "http://localhost",
            "visdom_port": 8097,
            "visdom_base_url": "/",
            "visdom_offline_ok": True,
            "visdom_save": False,
            "visdom_results": True,
            "visdom_results_count": 10,
            "visdom_results_seed": 12345,
            "confidence_threshold": 0.25,
            "best_bbox": None,
        }

    def infer_defaults(self) -> dict[str, Any]:
        return {
            "weights": self.default_weights,
            "repository": self.default_repository,
            "service_name": self.default_service_name.replace("train", "infer"),
            "nclasses": self.default_nclasses,
            "nkeypoints": self.service_mllib.get(
                "nkeypoints", self.train_mllib.get("nkeypoints")
            ),
            "max_objects": self.service_mllib.get(
                "max_objects", self.train_mllib.get("max_objects")
            ),
            "width": self.default_width,
            "height": self.default_height,
            "batch_size": 1,
            "gpu": False,
            "gpuid": None,
            "output": None,
            "visualize": False,
            "benchmark": False,
            "warmup": 0,
            "output_format": "json",
            "confidence_threshold": 0.25,
            "best_bbox": None,
        }

    def service_parameters(self, options: dict[str, Any]) -> dict[str, Any]:
        _validate_mapping_option(options, "service_mllib")
        mllib = _deep_merge(self.service_mllib, options.get("service_mllib"))
        mllib["gpu"] = bool(options["gpu"])
        if options.get("gpuid") is not None:
            mllib["gpuid"] = copy.deepcopy(options["gpuid"])
        mllib["nclasses"] = int(options["nclasses"])
        if options.get("nkeypoints") is not None:
            mllib["nkeypoints"] = int(options["nkeypoints"])
        if options.get("max_objects") is not None:
            mllib["max_objects"] = int(options["max_objects"])
        if self.backend == "pytorch":
            mllib.setdefault("python", sys.executable)
        if options.get("resume"):
            mllib["resume_from"] = str(options["resume"])
        if self.task == "keypoint":
            _sync_keypoint_model_size(mllib, options)
        input_parameters = copy.deepcopy(self.service_input)
        input_parameters["width"] = int(options["width"])
        input_parameters["height"] = int(options["height"])
        return {
            "model": {"repository": str(Path(options["repository"]).resolve())},
            "mllib": self.backend,
            "description": self.description,
            "input_parameters": input_parameters,
            "mllib_parameters": mllib,
            "output_parameters": {},
        }

    def train_parameters(self, options: dict[str, Any]) -> dict[str, Any]:
        mllib = copy.deepcopy(self.train_mllib)
        _validate_mapping_option(options, "augmentation")
        _validate_mapping_option(options, "mllib")
        mllib = _deep_merge(mllib, options.get("augmentation"))
        mllib = _deep_merge(mllib, options.get("mllib"))
        mllib["gpu"] = bool(options["gpu"])
        if options.get("gpuid") is not None:
            mllib["gpuid"] = copy.deepcopy(options["gpuid"])
        mllib.setdefault("solver", {})
        mllib["solver"]["iterations"] = int(options["iterations"])
        mllib["solver"]["iter_size"] = int(options["iter_size"])
        mllib["solver"]["base_lr"] = float(options["base_lr"])
        mllib["solver"]["test_interval"] = int(options["test_interval"])
        if options.get("nkeypoints") is not None:
            mllib["nkeypoints"] = int(options["nkeypoints"])
        if options.get("max_objects") is not None:
            mllib["max_objects"] = int(options["max_objects"])
        if options.get("class_weights") is not None:
            mllib["class_weights"] = _float_list_option(
                options["class_weights"], "class_weights"
            )
        if options.get("resume"):
            mllib["resume"] = True
            mllib["resume_from"] = str(options["resume"])
        if options.get("weights") is not None:
            mllib["weights"] = str(Path(options["weights"]).expanduser().resolve())
        mllib.setdefault("net", {})
        mllib["net"]["batch_size"] = int(options["batch_size"])
        if self.task == "detection":
            mllib["net"]["test_batch_size"] = int(options["batch_size"])
        if self.task == "keypoint":
            _sync_keypoint_model_size(mllib, options)
        input_parameters = copy.deepcopy(self.train_input)
        if "width" in self.service_input:
            input_parameters["width"] = int(options["width"])
        if "height" in self.service_input:
            input_parameters["height"] = int(options["height"])
        return {
            "input_parameters": input_parameters,
            "mllib_parameters": mllib,
            "output_parameters": copy.deepcopy(self.train_output),
        }

    def predict_parameters(self, options: dict[str, Any]) -> dict[str, Any]:
        input_parameters = copy.deepcopy(self.predict_input)
        input_parameters["width"] = int(options["width"])
        input_parameters["height"] = int(options["height"])
        output = copy.deepcopy(self.predict_output)
        if self.task == "detection":
            output["confidence_threshold"] = float(options["confidence_threshold"])
            if options.get("best_bbox") is not None:
                output["best_bbox"] = int(options["best_bbox"])
        if self.task == "keypoint":
            output["confidence_threshold"] = float(options["confidence_threshold"])
        return {
            "input_parameters": input_parameters,
            "output_parameters": output,
        }


PROFILES = {
    "yolox": ModelProfile(
        name="yolox",
        task="detection",
        description="YOLOX object detection",
        backend="torch",
        default_weights=DEFAULT_MODEL_ROOT / "yolox/yolox-nano_cls2.pt",
        default_repository=Path("deepdetect-models/yolox"),
        default_service_name="python-yolox-train",
        default_nclasses=2,
        requires_weights=True,
        service_input={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
            "db": False,
        },
        service_mllib={"template": "yolox"},
        train_input={"seed": 12347, "db": False, "shuffle": True},
        train_mllib={
            "solver": {"iter_size": 2, "solver_type": "ADAM"},
            "net": {"reg_weight": 0.5},
            "resume": False,
            "mirror": True,
            "rotate": True,
            "crop_size": 0,
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
        train_output={"measure": ["map-05", "map-50", "map-90"]},
        predict_input={"height": 640, "width": 640},
        predict_output={"bbox": True},
    ),
    "segformer": ModelProfile(
        name="segformer",
        task="segmentation",
        description="SegFormer semantic segmentation",
        backend="torch",
        default_weights=DEFAULT_MODEL_ROOT / "segformer/segformer-b0-cls2.pt",
        default_repository=Path("deepdetect-models/segformer"),
        default_service_name="python-segformer-train",
        default_nclasses=2,
        requires_weights=True,
        service_input={
            "connector": "image",
            "width": 480,
            "height": 480,
            "db": False,
            "segmentation": True,
        },
        service_mllib={"segmentation": True},
        train_input={
            "seed": 12345,
            "db": False,
            "shuffle": True,
            "segmentation": True,
            "scale": 0.0039,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        train_mllib={
            "solver": {"iter_size": 1, "solver_type": "ADAM"},
            "net": {},
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
        train_output={"measure": ["meaniou", "acc"]},
        predict_input={
            "height": 480,
            "width": 480,
            "scale": 0.0039,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        predict_output={"segmentation": True, "confidences": ["best"]},
    ),
    "torchvision-detector": ModelProfile(
        name="torchvision-detector",
        task="detection",
        description="Torchvision Faster R-CNN object detection worker",
        backend="pytorch",
        default_weights=None,
        default_repository=Path("deepdetect-models/torchvision-detector"),
        default_service_name="python-torchvision-detector-train",
        default_nclasses=2,
        requires_weights=False,
        service_input={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
            "db": False,
        },
        service_mllib={
            "task": "detection",
            "module": "deepdetect.pytorch_worker.builtin.vision.detection.torchvision_fasterrcnn",
            "class": "DeepDetectWorker",
        },
        train_input={"seed": 12347, "db": False, "shuffle": True},
        train_mllib={
            "solver": {"iter_size": 1, "solver_type": "ADAM"},
            "net": {"batch_size": 1},
            "resume": False,
        },
        train_output={"measure": ["map-05", "map-50", "map-90"]},
        predict_input={"height": 640, "width": 640},
        predict_output={"bbox": True},
    ),
    "external-pytorch-detector": ModelProfile(
        name="external-pytorch-detector",
        task="detection",
        description="External PyTorch object detection worker",
        backend="pytorch",
        default_weights=None,
        default_repository=Path("deepdetect-models/external-pytorch-detector"),
        default_service_name="python-external-pytorch-detector-train",
        default_nclasses=2,
        requires_weights=False,
        service_input={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
            "db": False,
        },
        service_mllib={
            "task": "detection",
            "class": "DeepDetectWorker",
        },
        train_input={"seed": 12347, "db": False, "shuffle": True},
        train_mllib={
            "solver": {"iter_size": 1, "solver_type": "ADAM"},
            "net": {"batch_size": 1},
            "resume": False,
            "data_source": "connector_tensor_pull",
        },
        train_output={"measure": ["map-05", "map-50", "map-90"]},
        predict_input={"height": 640, "width": 640},
        predict_output={"bbox": True},
    ),
    "vitpose": ModelProfile(
        name="vitpose",
        task="keypoint",
        description="Self-contained ViTPose keypoint worker",
        backend="pytorch",
        default_weights=None,
        default_repository=Path("deepdetect-models/vitpose"),
        default_service_name="python-vitpose-train",
        default_nclasses=1,
        requires_weights=False,
        service_input={
            "connector": "image",
            "height": 256,
            "width": 192,
            "rgb": True,
            "keypoints": True,
            "db": False,
        },
        service_mllib={
            "task": "keypoint",
            "entrypoint": "extern/pytorch_workers/vitpose/worker.py",
            "class": "DeepDetectWorker",
            "nkeypoints": 17,
            "max_objects": 1,
            "vitpose": {
                "variant": "base",
                "image_size": [192, 256],
                "heatmap_size": [48, 64],
                "sigma": 2.0,
                "max_objects": 1,
                "objectness_threshold": 0.25,
                "keypoint_threshold": 0.05,
                "weight_decay": 0.1,
                "layer_decay": 0.75,
                "grad_clip": 1.0,
            },
        },
        train_input={"seed": 12347, "db": False, "shuffle": True},
        train_mllib={
            "solver": {"iter_size": 1, "solver_type": "ADAMW", "base_lr": 0.0005},
            "net": {"batch_size": 1},
            "resume": False,
            "data_source": "connector_tensor_pull",
            "nkeypoints": 17,
            "max_objects": 1,
            "vitpose": {
                "variant": "base",
                "image_size": [192, 256],
                "heatmap_size": [48, 64],
                "sigma": 2.0,
                "max_objects": 1,
            },
        },
        train_output={"measure": ["train_loss"]},
        predict_input={"height": 256, "width": 192},
        predict_output={"keypoints": True},
    ),
}


def get_profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown model profile: {name}") from error


def _deep_merge(*values: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if not value:
            continue
        for key, item in value.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(item, dict)
            ):
                result[key] = _deep_merge(result[key], item)
            else:
                result[key] = copy.deepcopy(item)
    return result


def _sync_keypoint_model_size(mllib: dict[str, Any], options: dict[str, Any]) -> None:
    width = int(options["width"])
    height = int(options["height"])
    vitpose = mllib.setdefault("vitpose", {})
    if not isinstance(vitpose, dict):
        return
    vitpose["image_size"] = [width, height]
    vitpose["heatmap_size"] = [max(1, width // 4), max(1, height // 4)]
    if options.get("max_objects") is not None:
        vitpose["max_objects"] = int(options["max_objects"])


def _augmentation_defaults(mllib: Mapping[str, Any]) -> dict[str, Any]:
    non_augmentation_keys = {
        "solver",
        "net",
        "resume",
        "resume_from",
        "data_source",
        "nkeypoints",
        "max_objects",
        "vitpose",
        "gpu",
        "gpuid",
        "weights",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in mllib.items()
        if key not in non_augmentation_keys
    }


def _validate_mapping_option(options: dict[str, Any], name: str) -> None:
    value = options.get(name)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")


def _float_list_option(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of numbers")
    return [float(item) for item in value]
