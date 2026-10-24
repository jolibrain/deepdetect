"""Shared connector prediction boundary for detection and semantic workers."""

from contextlib import contextmanager
from typing import Any

from .sdk import PredictionContractError
from .tensors import parse_tensor_batch_ref, materialize_tensor_ref


@contextmanager
def prediction_batch(request: dict, torch: Any):
    """Yield preprocessed NCHW tensors and ordered metadata without image I/O.

    Tensors are valid only inside the context. Do not retain views in a model.
    """
    batch = parse_tensor_batch_ref(request.get("tensor_batch"))
    if len(batch.inputs) != 1:
        raise PredictionContractError("prediction requires exactly one input tensor")
    ref = batch.inputs[0]
    if ref.dtype != "float32" or len(ref.shape) != 4 or ref.strides is not None:
        raise PredictionContractError("prediction requires contiguous float32 NCHW")
    meta = batch.meta
    keys = (
        "paths",
        "sample_ids",
        "original_widths",
        "original_heights",
        "preprocessed_widths",
        "preprocessed_heights",
    )
    if not isinstance(meta, dict) or any(
        not isinstance(meta.get(key), list) or len(meta[key]) != ref.shape[0]
        for key in keys
    ):
        raise PredictionContractError("prediction metadata must match batch size")
    metas = []
    for i in range(ref.shape[0]):
        dims = [meta[key][i] for key in keys[2:]]
        if any(type(d) is not int or d <= 0 for d in dims):
            raise PredictionContractError(
                "prediction dimensions must be positive integers"
            )
        if dims[2:] != [ref.shape[3], ref.shape[2]]:
            raise PredictionContractError("prediction dimensions differ from tensor")
        if (
            not isinstance(meta["paths"][i], str)
            or type(meta["sample_ids"][i]) is not int
        ):
            raise PredictionContractError("invalid prediction sample identity")
        metas.append(
            dict(
                path=meta["paths"][i],
                index=meta["sample_ids"][i],
                original_width=dims[0],
                original_height=dims[1],
                width=dims[2],
                height=dims[3],
            )
        )
    materialized = materialize_tensor_ref(ref, torch)
    try:
        yield materialized.tensor, metas
    finally:
        materialized.close()


def semantic_prediction_results(class_maps, metas, *, torch, confidences=None):
    """Convert NHW class maps and optional named NHW confidences to HTTP pixels."""
    if class_maps.ndim != 3 or class_maps.shape[0] != len(metas):
        raise PredictionContractError("semantic class maps must have shape [N,H,W]")
    if confidences is None:
        confidences = {}
    for values in confidences.values():
        if values.shape != class_maps.shape:
            raise PredictionContractError(
                "semantic confidence shape must match class maps"
            )
    results = []
    for i, meta in enumerate(metas):
        size = (meta["original_height"], meta["original_width"])
        vals = (
            torch.nn.functional.interpolate(
                class_maps[i][None, None].to(dtype=torch.float64),
                size=size,
                mode="nearest",
            )
            .flatten()
            .cpu()
            .tolist()
        )
        item = dict(
            uri=meta["path"], vals=vals, imgsize=dict(height=size[0], width=size[1])
        )
        if confidences:
            item["confidences"] = {
                str(key): torch.nn.functional.interpolate(
                    values[i][None, None].to(dtype=torch.float64),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten()
                .cpu()
                .tolist()
                for key, values in confidences.items()
            }
        results.append(item)
    return {"task": "segmentation", "results": results}
