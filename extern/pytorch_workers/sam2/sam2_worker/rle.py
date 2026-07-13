from __future__ import annotations

from typing import Any

import numpy as np

from deepdetect.pytorch_worker.sdk import PredictionContractError


def encode_coco_rle(mask: Any) -> dict[str, Any]:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise PredictionContractError("SAM2 mask must have height and width dimensions")
    height, width = array.shape
    if height <= 0 or width <= 0:
        raise PredictionContractError("SAM2 mask dimensions must be positive")
    values = np.asarray(array, dtype=np.uint8).reshape(-1, order="F") > 0
    counts: list[int] = []
    value = False
    count = 0
    for item in values:
        item_bool = bool(item)
        if item_bool == value:
            count += 1
        else:
            counts.append(count)
            value = item_bool
            count = 1
    counts.append(count)
    return {
        "encoding": "coco_rle",
        "size": [int(height), int(width)],
        "counts": counts,
    }
