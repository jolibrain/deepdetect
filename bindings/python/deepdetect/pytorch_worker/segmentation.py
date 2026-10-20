"""Shared class-index mask validation; no training or model dependencies."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _label_id(value):
    if isinstance(value, str):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    raise ValueError(f"label mappings require integer IDs, got {value!r}")


def decode_mask(mask, nclasses, *, ignore_label=None, label_mapping=None,
                color_mapping=None):
    """Map explicit source labels, preserving unmapped index values.

    Color keys are comma-separated RGB values, e.g. ``"255,0,0": 1``.
    Indexed PNG arrays must be passed without conversion to RGB.
    """
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] != 3 or not isinstance(color_mapping, Mapping):
            raise ValueError("color masks require an explicit RGB color_mapping")
        mapped = np.full(array.shape[:2], -1, dtype=np.int64)
        recognized = np.zeros(array.shape[:2], dtype=bool)
        for color, target in color_mapping.items():
            rgb = tuple(int(v) for v in str(color).split(","))
            if len(rgb) != 3 or any(v < 0 or v > 255 for v in rgb):
                raise ValueError(f"invalid RGB mapping key: {color}")
            selected = np.all(array == rgb, axis=-1)
            mapped[selected] = _label_id(target)
            recognized |= selected
        if not recognized.all():
            raise ValueError("unknown mask colors not present in color_mapping")
        array = mapped
    elif array.ndim == 2 and array.dtype.kind in "biu":
        source = array.astype(np.int64)
        array = source.copy()
        for value, target in (label_mapping or {}).items():
            array[source == _label_id(value)] = _label_id(target)
    else:
        raise ValueError("mask must contain integer indices or explicitly mapped RGB")
    invalid = (array < 0) | (array >= nclasses)
    if ignore_label is not None:
        invalid &= array != ignore_label
    if invalid.any():
        raise ValueError(f"unknown mask values: {np.unique(array[invalid]).tolist()}")
    return array.astype(np.int64)
