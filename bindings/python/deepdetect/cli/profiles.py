from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/data1/beniz/models/dd")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    task: str
    description: str
    default_weights: Path
    default_repository: Path
    default_service_name: str
    default_nclasses: int
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
            "width": self.default_width,
            "height": self.default_height,
            "iterations": 100,
            "batch_size": 2 if self.name == "yolox" else 4,
            "base_lr": 0.0001,
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
        }

    def infer_defaults(self) -> dict[str, Any]:
        return {
            "weights": self.default_weights,
            "repository": self.default_repository,
            "service_name": self.default_service_name.replace("train", "infer"),
            "nclasses": self.default_nclasses,
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
        mllib = copy.deepcopy(self.service_mllib)
        mllib["gpu"] = bool(options["gpu"])
        if options.get("gpuid") is not None:
            mllib["gpuid"] = copy.deepcopy(options["gpuid"])
        mllib["nclasses"] = int(options["nclasses"])
        if options.get("resume"):
            mllib["resume_from"] = str(options["resume"])
        input_parameters = copy.deepcopy(self.service_input)
        input_parameters["width"] = int(options["width"])
        input_parameters["height"] = int(options["height"])
        return {
            "model": {"repository": str(Path(options["repository"]).resolve())},
            "mllib": "torch",
            "description": self.description,
            "input_parameters": input_parameters,
            "mllib_parameters": mllib,
            "output_parameters": {},
        }

    def train_parameters(self, options: dict[str, Any]) -> dict[str, Any]:
        mllib = copy.deepcopy(self.train_mllib)
        mllib["gpu"] = bool(options["gpu"])
        if options.get("gpuid") is not None:
            mllib["gpuid"] = copy.deepcopy(options["gpuid"])
        mllib.setdefault("solver", {})
        mllib["solver"]["iterations"] = int(options["iterations"])
        mllib["solver"]["base_lr"] = float(options["base_lr"])
        mllib["solver"]["test_interval"] = int(options["test_interval"])
        if options.get("resume"):
            mllib["resume"] = True
            mllib["resume_from"] = str(options["resume"])
        mllib.setdefault("net", {})
        mllib["net"]["batch_size"] = int(options["batch_size"])
        if self.name == "yolox":
            mllib["net"]["test_batch_size"] = int(options["batch_size"])
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
        if self.name == "yolox":
            output["confidence_threshold"] = float(options["confidence_threshold"])
            if options.get("best_bbox") is not None:
                output["best_bbox"] = int(options["best_bbox"])
        return {
            "input_parameters": input_parameters,
            "output_parameters": output,
        }


PROFILES = {
    "yolox": ModelProfile(
        name="yolox",
        task="detection",
        description="YOLOX object detection",
        default_weights=DEFAULT_MODEL_ROOT / "yolox/yolox-nano_cls2.pt",
        default_repository=Path("deepdetect-models/yolox"),
        default_service_name="python-yolox-train",
        default_nclasses=2,
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
        train_output={"measure": ["map-05", "map-50", "map-90"]},
        predict_input={"height": 640, "width": 640},
        predict_output={"bbox": True},
    ),
    "segformer": ModelProfile(
        name="segformer",
        task="segmentation",
        description="SegFormer semantic segmentation",
        default_weights=DEFAULT_MODEL_ROOT / "segformer/segformer-b0-cls2.pt",
        default_repository=Path("deepdetect-models/segformer"),
        default_service_name="python-segformer-train",
        default_nclasses=2,
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
}


def get_profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown model profile: {name}") from error
