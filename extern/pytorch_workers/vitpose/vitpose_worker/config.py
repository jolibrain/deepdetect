from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepdetect.pytorch_worker.sdk import DatasetContractError

from .losses import PoseLossConfig
from .model import VARIANT_DEFAULTS, ViTPoseModelConfig
from .targets import PoseTargetConfig


@dataclass(frozen=True)
class ViTPoseWorkerConfig:
    model: ViTPoseModelConfig
    target: PoseTargetConfig
    loss: PoseLossConfig
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    objectness_threshold: float
    keypoint_threshold: float
    weight_decay: float
    adamw_betas: tuple[float, float]
    layer_decay: float
    grad_clip: float | None
    connector_prefetch_batches: int


def worker_config_from_mllib(mllib: dict[str, Any]) -> ViTPoseWorkerConfig:
    options = mllib.get("vitpose", {})
    options = options if isinstance(options, dict) else {}
    variant = str(options.get("variant", mllib.get("variant", "base")))
    if variant not in VARIANT_DEFAULTS:
        raise DatasetContractError(
            f"unsupported ViTPose variant {variant!r}; expected one of "
            f"{', '.join(sorted(VARIANT_DEFAULTS))}"
        )
    variant_defaults = dict(VARIANT_DEFAULTS[variant])
    nkeypoints = positive_int(mllib.get("nkeypoints", options.get("nkeypoints", 17)), "nkeypoints")
    max_objects = positive_int(options.get("max_objects", mllib.get("max_objects", 1)), "max_objects")
    image_size = size_tuple(
        options.get("image_size", mllib.get("image_size", [192, 256])),
        "image_size",
    )
    heatmap_size = size_tuple(
        options.get("heatmap_size", [max(1, image_size[0] // 4), max(1, image_size[1] // 4)]),
        "heatmap_size",
    )
    model_config = ViTPoseModelConfig(
        image_size=image_size,
        heatmap_size=heatmap_size,
        nkeypoints=nkeypoints,
        max_objects=max_objects,
        variant=variant,
        patch_size=positive_int(options.get("patch_size", 16), "patch_size"),
        embed_dim=positive_int(options.get("embed_dim", variant_defaults["embed_dim"]), "embed_dim"),
        depth=positive_int(options.get("depth", variant_defaults["depth"]), "depth"),
        num_heads=positive_int(options.get("num_heads", variant_defaults["num_heads"]), "num_heads"),
        mlp_ratio=float(options.get("mlp_ratio", 4.0)),
        qkv_bias=bool(options.get("qkv_bias", True)),
        drop_path_rate=float(options.get("drop_path_rate", variant_defaults["drop_path_rate"])),
        upsample=positive_int(options.get("upsample", 4), "upsample"),
        final_conv_kernel=int(options.get("final_conv_kernel", 3)),
        num_deconv_layers=int(options.get("num_deconv_layers", 0)),
        num_deconv_filters=int_tuple(options.get("num_deconv_filters", []), "num_deconv_filters"),
        num_deconv_kernels=int_tuple(options.get("num_deconv_kernels", []), "num_deconv_kernels"),
    )
    target_config = PoseTargetConfig(
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=float(options.get("sigma", 2.0)),
        max_objects=max_objects,
        nkeypoints=nkeypoints,
    )
    loss_config = PoseLossConfig(
        target=target_config,
        heatmap_weight=float(options.get("heatmap_loss_weight", 1.0)),
        objectness_weight=float(options.get("objectness_loss_weight", 1.0)),
    )
    return ViTPoseWorkerConfig(
        model=model_config,
        target=target_config,
        loss=loss_config,
        mean=float_tuple(options.get("mean", [0.485, 0.456, 0.406]), "mean", length=3),
        std=float_tuple(options.get("std", [0.229, 0.224, 0.225]), "std", length=3),
        objectness_threshold=float(options.get("objectness_threshold", 0.25)),
        keypoint_threshold=float(options.get("keypoint_threshold", 0.05)),
        weight_decay=float(options.get("weight_decay", 0.1)),
        adamw_betas=float_tuple(options.get("betas", [0.9, 0.999]), "betas", length=2),
        layer_decay=float(options.get("layer_decay", variant_layer_decay(variant))),
        grad_clip=optional_float(options.get("grad_clip", 1.0), "grad_clip"),
        connector_prefetch_batches=positive_int(
            mllib.get("connector_prefetch_batches", options.get("connector_prefetch_batches", 2)),
            "connector_prefetch_batches",
        ),
    )


def variant_layer_decay(variant: str) -> float:
    return {
        "tiny": 0.9,
        "small": 0.75,
        "base": 0.75,
        "large": 0.8,
        "huge": 0.85,
    }[variant]


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise DatasetContractError(f"{name} must be positive")
    return result


def optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result <= 0:
        raise DatasetContractError(f"{name} must be positive")
    return result


def size_tuple(value: Any, name: str) -> tuple[int, int]:
    values = int_tuple(value, name)
    if len(values) != 2:
        raise DatasetContractError(f"{name} must contain width and height")
    return int(values[0]), int(values[1])


def int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DatasetContractError(f"{name} must be a list")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise DatasetContractError(f"{name} entries must be positive")
    return result


def float_tuple(value: Any, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise DatasetContractError(f"{name} must contain {length} values")
    return tuple(float(item) for item in value)
