from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .sdk import DatasetContractError, WorkerContractError


ALLOWED_TENSOR_DTYPES = {
    "bool",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8",
}
ALLOWED_STORAGE_TYPES = {"inline_test_stub", "shared_memory"}


@dataclass(frozen=True)
class TensorStorageRef:
    type: str
    name: str | None
    offset: int
    nbytes: int
    values: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class TensorRef:
    device: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    strides: tuple[int, ...] | None
    storage: TensorStorageRef
    lifetime: Mapping[str, Any]
    cuda: Mapping[str, Any]


@dataclass(frozen=True)
class TensorBatchRef:
    inputs: tuple[TensorRef, ...]
    targets: Any = None
    meta: Any = None


def parse_tensor_ref(value: Any) -> TensorRef:
    if not isinstance(value, Mapping):
        raise WorkerContractError("tensor ref must be an object")
    kind = value.get("kind")
    if kind != "tensor_ref":
        raise WorkerContractError("tensor ref kind must be 'tensor_ref'")
    device = _required_string(value, "device", "tensor ref")
    if device != "cpu":
        raise DatasetContractError(
            f"tensor ref device is reserved for future use: {device}"
        )
    dtype = _required_string(value, "dtype", "tensor ref")
    if dtype not in ALLOWED_TENSOR_DTYPES:
        raise WorkerContractError(f"unsupported tensor dtype: {dtype}")
    layout = value.get("layout", "strided")
    if layout != "strided":
        raise WorkerContractError("tensor ref layout must be 'strided'")
    shape = _positive_int_tuple(value.get("shape"), "tensor ref shape")
    strides = None
    if "strides" in value and value["strides"] is not None:
        strides = _positive_int_tuple(value["strides"], "tensor ref strides")
        if len(strides) != len(shape):
            raise WorkerContractError(
                "tensor ref strides length must match shape length"
            )
    storage = parse_tensor_storage_ref(value.get("storage"))
    lifetime = _optional_mapping(value.get("lifetime"), "tensor ref lifetime")
    cuda = _optional_mapping(value.get("cuda"), "tensor ref cuda metadata")
    return TensorRef(
        device=device,
        dtype=dtype,
        shape=shape,
        layout=layout,
        strides=strides,
        storage=storage,
        lifetime=lifetime,
        cuda=cuda,
    )


def parse_tensor_storage_ref(value: Any) -> TensorStorageRef:
    if not isinstance(value, Mapping):
        raise WorkerContractError("tensor storage must be an object")
    storage_type = _required_string(value, "type", "tensor storage")
    if storage_type not in ALLOWED_STORAGE_TYPES:
        raise WorkerContractError(f"unsupported tensor storage type: {storage_type}")
    name = value.get("name")
    if name is not None and not isinstance(name, str):
        raise WorkerContractError("tensor storage name must be a string")
    offset = _nonnegative_int(value.get("offset", 0), "tensor storage offset")
    nbytes = _nonnegative_int(value.get("nbytes", 0), "tensor storage nbytes")
    if storage_type == "shared_memory":
        if not name:
            raise WorkerContractError("shared_memory tensor storage requires name")
        if nbytes <= 0:
            raise WorkerContractError("shared_memory tensor storage requires nbytes")
    values = value.get("values")
    if values is not None and not isinstance(values, list):
        raise WorkerContractError("tensor storage values must be a list")
    return TensorStorageRef(
        type=storage_type,
        name=name,
        offset=offset,
        nbytes=nbytes,
        values=tuple(values) if values is not None else None,
    )


def parse_tensor_batch_ref(value: Any) -> TensorBatchRef:
    if not isinstance(value, Mapping):
        raise WorkerContractError("tensor batch must be an object")
    kind = value.get("kind", "tensor_batch")
    if kind != "tensor_batch":
        raise WorkerContractError("tensor batch kind must be 'tensor_batch'")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise WorkerContractError("tensor batch inputs must be a non-empty list")
    return TensorBatchRef(
        inputs=tuple(parse_tensor_ref(item) for item in inputs),
        targets=value.get("targets"),
        meta=value.get("meta"),
    )


def materialize_inline_tensor_ref(ref: TensorRef, torch: Any) -> Any:
    if ref.device != "cpu":
        raise DatasetContractError(
            f"tensor ref device is reserved for future use: {ref.device}"
        )
    if ref.storage.type != "inline_test_stub":
        raise DatasetContractError(
            f"tensor storage cannot be materialized in this slice: {ref.storage.type}"
        )
    if ref.storage.values is None:
        raise WorkerContractError("inline_test_stub tensor storage requires values")
    expected = math.prod(ref.shape)
    if len(ref.storage.values) != expected:
        raise WorkerContractError(
            "inline_test_stub tensor values count must match tensor shape"
        )
    dtype = _torch_dtype(torch, ref.dtype)
    return torch.tensor(list(ref.storage.values), dtype=dtype).reshape(ref.shape)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    mapping = {
        "bool": torch.bool,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise WorkerContractError(f"unsupported tensor dtype: {dtype}") from error


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise WorkerContractError(f"{label} {key} must be a non-empty string")
    return item


def _positive_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise WorkerContractError(f"{label} must be a non-empty list")
    return tuple(_positive_int(item, label) for item in value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerContractError(f"{label} entries must be positive integers")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerContractError(f"{label} must be a non-negative integer")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkerContractError(f"{label} must be an object")
    return value
