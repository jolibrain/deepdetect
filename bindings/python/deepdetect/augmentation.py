"""CPU augmentation backed by DeepDetect's shared C++ operators.

No loading, normalization, tensor conversion or model runtime is implicit.
See augmentation.md for operation units, compatibility and validity semantics.
"""

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Mapping
import numpy as np

_NOISE = dict(
    decolorize="Decolorize",
    gauss_blur="GaussianBlur",
    hist_eq="HistEq",
    clahe="Clahe",
    jpg="JPG",
    erosion="Erosion",
    posterize="Posterize",
    inverse="Inverse",
    saltpepper="Saltpepper",
    convert_to_hsv="ConvertHSV",
    convert_to_lab="ConvertLAB",
)
_DISTORT = dict(
    brightness="Brightness",
    contrast="Contrast",
    saturation="Saturation",
    hue="Hue",
    channel_order="OrderChannel",
)
_RANGES = dict(
    brightness_range=[-32.0, 32.0],
    contrast_range=[0.5, 1.5],
    saturation_range=[0.5, 1.5],
    hue_range=[-36.0, 36.0],
)
_GEOMETRY = dict(
    prob=0.0,
    persp_horizontal=True,
    persp_vertical=True,
    transl_horizontal=False,
    transl_vertical=False,
    zoom_out=True,
    zoom_in=True,
    pad_mode="constant",
    persp_factor=0.25,
    transl_factor=0.5,
    zoom_factor=0.25,
)


def _native():
    try:
        from . import _augmentation
    except ImportError as error:
        raise ImportError(
            "DeepDetect augmentation extension is unavailable; install a wheel built with the augmentation SDK target"
        ) from error
    return _augmentation


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value, allowed, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"unknown {name} options: {sorted(extra)}")
    return dict(value)


def _number(value, name, low=0.0, high=1.0):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _boolean(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _resolve(config, semantics, color_order):
    if semantics not in ("explicit_v1", "legacy_v1"):
        raise ValueError("unknown augmentation semantics")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR; grayscale uses HW arrays")
    cfg = _mapping(
        config,
        ("mirror", "rotate", "crop_size", "cutout", "geometry", "noise", "distort"),
        "augmentation",
    )
    out = dict(semantics=semantics, color_order=color_order)
    for key in ("mirror", "rotate"):
        out[key] = _boolean(cfg.get(key, False), key)
    size = cfg.get("crop_size", -1)
    if semantics == "legacy_v1" and type(size) is int and size <= 0:
        size = -1
    if isinstance(size, bool) or not isinstance(size, int) or (size != -1 and size < 1):
        raise ValueError("crop_size must be a positive integer or -1")
    out["crop_size"] = size
    out["cutout"] = _number(cfg.get("cutout", 0), "cutout")
    geo = {**_GEOMETRY, **_mapping(cfg.get("geometry", {}), _GEOMETRY, "geometry")}
    for key, value in geo.items():
        if isinstance(_GEOMETRY[key], bool):
            _boolean(value, key)
        elif key == "pad_mode":
            if value not in ("constant", "mirrored", "repeat_nearest"):
                raise ValueError("invalid geometry pad_mode")
        else:
            _number(value, key, 0, 1)
    out["geometry"] = geo
    out["operation_probabilities"] = {}
    for family, operations in (("noise", _NOISE), ("distort", _DISTORT)):
        extra = ("saltpepper_fraction",) if family == "noise" else tuple(_RANGES)
        allowed = (
            ("prob",)
            if semantics == "legacy_v1"
            else ("prob", "probabilities", *operations, *extra)
        )
        params = _mapping(cfg.get(family, {}), allowed, family)
        base = _number(params.get("prob", 0), family + ".prob")
        if semantics == "legacy_v1":
            out[family] = {"prob": base}
            continue
        probs = _mapping(
            params.get("probabilities", {}), operations, family + ".probabilities"
        )
        resolved = {"prob": base}
        for name, suffix in operations.items():
            enabled = _boolean(params.get(name, name in probs), name)
            probability = _number(
                probs.get(name, base if enabled else 0), name + " probability"
            )
            if not enabled and probability:
                raise ValueError(f"{name} is disabled but has nonzero probability")
            resolved[name] = enabled and probability > 0
            out["operation_probabilities"][
                "apply" + family.title() + suffix
            ] = probability
        resolved["prob"] = max(
            [base]
            + [
                p
                for n, p in out["operation_probabilities"].items()
                if n.startswith("apply" + family.title())
            ]
        )
        if base and not any(params.get(name, name in probs) for name in operations):
            raise ValueError(f"{family}.prob requires explicitly enabled operations")
        if family == "noise":
            resolved["saltpepper_fraction"] = _number(
                params.get("saltpepper_fraction", 0.1), "saltpepper_fraction"
            )
        else:
            for key, default in _RANGES.items():
                bounds = params.get(key, default)
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"{key} must be a two-element range")
                lo, hi = (
                    (0, 100)
                    if key in ("contrast_range", "saturation_range")
                    else (-255, 255)
                )
                bounds = [_number(v, key, lo, hi) for v in bounds]
                if bounds[0] > bounds[1]:
                    raise ValueError(f"{key} is reversed")
                resolved[key] = bounds
        out[family] = resolved
    return out


@dataclass(frozen=True)
class AugmentationPlan:
    """Immutable JSON replay record; use to_json/from_json for persistence."""

    _json: str

    @property
    def metadata(self):
        return json.loads(self._json)

    def to_json(self):
        return self._json

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported augmentation plan")
        return cls(_json(data))


@dataclass(frozen=True)
class AugmentationResult:
    image: np.ndarray
    masks: dict[str, np.ndarray]
    in_image: np.ndarray
    occluded: np.ndarray
    source_to_output: np.ndarray
    plan: AugmentationPlan


@dataclass(frozen=True, init=False)
class Augmenter:
    """Immutable policy. Explicit calls require a seed or a previously sampled plan."""

    _config_json: str

    def __init__(self, config=None, *, semantics="explicit_v1", color_order="RGB"):
        object.__setattr__(
            self,
            "_config_json",
            _json(_resolve({} if config is None else config, semantics, color_order)),
        )

    @property
    def config(self):
        return json.loads(self._config_json)

    @staticmethod
    def build_info():
        return dict(_native().build_info())

    def _run(self, image, masks, seed, record, appearance):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not isinstance(appearance, bool):
            raise TypeError("appearance must be boolean")
        targets = {} if masks is None else dict(masks)
        if any(
            not isinstance(k, str) or not isinstance(v, np.ndarray)
            for k, v in targets.items()
        ):
            raise TypeError("masks must map names to NumPy arrays")
        return _native().run(image, targets, self.config, seed, record, appearance)

    def __call__(self, image, *, seed, masks=None):
        raw = self._run(image, masks, seed, None, True)
        plan = AugmentationPlan(
            _json(
                dict(
                    version=1,
                    shape=list(image.shape),
                    seed=seed,
                    config=self.config,
                    build=self.build_info(),
                    record=raw["record"],
                )
            )
        )
        return self._result(raw, plan)

    def sample(self, shape, *, seed):
        """Sample using a synthetic image; no source image content is inspected.

        v1 executes the operators on this image to capture exact legacy draws.
        For a single view prefer __call__, which avoids this extra processing.
        """
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) not in (2, 3)
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape
            )
            or (len(shape) == 3 and shape[2] != 3)
        ):
            raise ValueError("shape must be (H,W) or (H,W,3)")
        return self(np.zeros(shape, dtype=np.uint8), seed=seed).plan

    def apply(self, image, *, plan, masks=None, appearance=True):
        if not isinstance(plan, AugmentationPlan):
            raise TypeError("plan must be an AugmentationPlan")
        meta = plan.metadata
        if (
            meta.get("version") != 1
            or meta.get("config") != self.config
            or meta.get("build") != self.build_info()
        ):
            raise ValueError("augmentation plan policy/build mismatch")
        if not isinstance(image, np.ndarray) or meta.get("shape") != list(image.shape):
            raise ValueError("augmentation plan input shape mismatch")
        raw = self._run(image, masks, meta["seed"], meta["record"], appearance)
        if _json(raw["record"]) != _json(meta["record"]):
            raise ValueError("augmentation replay record mismatch")
        return self._result(raw, plan)

    @staticmethod
    def _result(raw, plan):
        return AugmentationResult(
            raw["image"],
            raw["masks"],
            raw["in_image"].astype(bool),
            raw["occluded"].astype(bool),
            np.asarray(raw["record"]["source_to_output"], dtype=np.float64),
            plan,
        )
