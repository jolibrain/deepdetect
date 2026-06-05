from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .errors import CapabilityError, DeepDetectError


def _loads(response: str | Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(response) if isinstance(response, str) else dict(response)
    except (TypeError, ValueError) as error:
        raise RuntimeError("the native runtime returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("the native runtime returned a non-object JSON response")
    return value


def _checked(response: str | Mapping[str, Any]) -> dict[str, Any]:
    value = _loads(response)
    status = value.get("status", {})
    code = int(status.get("code", 500))
    if not 200 <= code < 300:
        message = status.get("dd_msg") or status.get("msg") or "DeepDetect error"
        raise DeepDetectError(code, status.get("dd_code"), str(message), value)
    return value


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("service name must be a non-empty string")
    return value.strip().lower()


def _image_data(value: np.ndarray) -> str:
    if value.dtype != np.uint8:
        raise TypeError("NumPy image arrays must have dtype uint8")
    if value.ndim == 2:
        mode = "L"
    elif value.ndim == 3 and value.shape[2] == 3:
        mode = "RGB"
    elif value.ndim == 3 and value.shape[2] == 4:
        mode = "RGBA"
    else:
        raise TypeError("NumPy image arrays must have shape HxW, HxWx3, or HxWx4")
    output = io.BytesIO()
    Image.fromarray(value, mode=mode).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _adapt_data(value: Any, image_service: bool | None) -> Any:
    if isinstance(value, np.ndarray):
        if image_service is False:
            raise TypeError("NumPy arrays are supported only by image services")
        return _image_data(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {key: _adapt_data(item, image_service) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_adapt_data(item, image_service) for item in value]
    return value


def _data(values: Any, image_service: bool | None) -> Any:
    adapted = _adapt_data(values, image_service)
    if isinstance(values, (str, os.PathLike, np.ndarray)):
        return [adapted]
    return adapted


class DeepDetect:
    def __init__(self, *, _runtime: Any = None) -> None:
        if _runtime is None:
            from . import _native

            _runtime = _native.runtime()
        self._runtime = _runtime
        self._build_info: dict[str, Any] | None = None
        self._connectors: dict[str, str] = {}

    @property
    def build_info(self) -> dict[str, Any]:
        if self._build_info is None:
            self._build_info = _loads(self._runtime.build_info())
        return dict(self._build_info)

    def info(self) -> dict[str, Any]:
        return _checked(self._runtime.info("{}")).get("head", {})

    def _require_gpu(self, requested: bool) -> None:
        if requested and not self.build_info.get("cuda", False):
            response = {"status": {"code": 400, "dd_code": 2000}}
            raise CapabilityError(
                400, 2000, "the installed DeepDetect SDK is CPU-only", response
            )

    def create_service(
        self,
        name: str,
        *,
        model: Mapping[str, Any],
        mllib: str,
        input_parameters: Mapping[str, Any],
        mllib_parameters: Mapping[str, Any] | None = None,
        output_parameters: Mapping[str, Any] | None = None,
        service_type: str = "supervised",
        description: str | None = None,
    ) -> "Service":
        normalized = _name(name)
        mllib_parameters = dict(mllib_parameters or {})
        self._require_gpu(bool(mllib_parameters.get("gpu")))
        request: dict[str, Any] = {
            "mllib": mllib,
            "type": service_type,
            "model": dict(model),
            "parameters": {
                "input": dict(input_parameters),
                "mllib": mllib_parameters,
                "output": dict(output_parameters or {}),
            },
        }
        if description is not None:
            request["description"] = description
        _checked(self._runtime.create_service(normalized, _json(request)))
        connector = str(input_parameters.get("connector", ""))
        self._connectors[normalized] = connector
        return Service(self, normalized, connector=connector or None)

    def service(self, name: str) -> "Service":
        normalized = _name(name)
        return Service(self, normalized, connector=self._connectors.get(normalized))


class Service:
    def __init__(
        self, client: DeepDetect, name: str, *, connector: str | None = None
    ) -> None:
        self._client = client
        self.name = name
        self._connector = connector
        self._deleted = False

    def _active(self) -> None:
        if self._deleted:
            raise RuntimeError(f"service handle {self.name!r} has been deleted")

    def __enter__(self) -> "Service":
        self._active()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if not self._deleted:
            try:
                self.delete()
            except Exception:
                if exc_type is None:
                    raise

    def info(self) -> dict[str, Any]:
        self._active()
        return _checked(self._client._runtime.service_info(self.name)).get("body", {})

    def predict(
        self,
        data: Any,
        *,
        input_parameters: Mapping[str, Any] | None = None,
        output_parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._active()
        request = {
            "service": self.name,
            "parameters": {
                "input": dict(input_parameters or {}),
                "output": dict(output_parameters or {}),
            },
            "data": _data(data, self._connector == "image" if self._connector else None),
        }
        return _checked(self._client._runtime.predict(_json(request))).get("body", {})

    def train(
        self,
        data: Any,
        *,
        input_parameters: Mapping[str, Any] | None = None,
        mllib_parameters: Mapping[str, Any] | None = None,
        output_parameters: Mapping[str, Any] | None = None,
        asynchronous: bool = False,
    ) -> dict[str, Any] | "TrainingJob":
        self._active()
        mllib_parameters = dict(mllib_parameters or {})
        self._client._require_gpu(bool(mllib_parameters.get("gpu")))
        request = {
            "service": self.name,
            "async": asynchronous,
            "parameters": {
                "input": dict(input_parameters or {}),
                "mllib": mllib_parameters,
                "output": dict(output_parameters or {}),
            },
            "data": _data(data, self._connector == "image" if self._connector else None),
        }
        response = _checked(self._client._runtime.train(_json(request)))
        if asynchronous:
            try:
                job = int(response["head"]["job"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("asynchronous training returned no job id") from error
            return TrainingJob(self, job)
        return response.get("body", {})

    def delete(self, clear: str | None = None) -> None:
        self._active()
        request = {} if clear is None else {"clear": clear}
        _checked(self._client._runtime.delete_service(self.name, _json(request)))
        self._deleted = True
        self._client._connectors.pop(self.name, None)


class TrainingJob:
    _TERMINAL = {"finished", "error", "terminated", "cancelled"}

    def __init__(self, service: Service, job: int) -> None:
        self.service = service
        self.job = job

    def status(self) -> dict[str, Any]:
        self.service._active()
        request = _json({"service": self.service.name, "job": self.job})
        response = _checked(self.service._client._runtime.training_status(request))
        result = dict(response.get("body", {}))
        head = response.get("head", {})
        for key in ("job", "status", "time"):
            if key in head:
                result[key] = head[key]
        return result

    def wait(
        self, timeout: float | None = None, poll_interval: float = 0.5
    ) -> dict[str, Any]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        started = time.monotonic()
        while True:
            status = self.status()
            if status.get("status") in self._TERMINAL:
                return status
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"training job {self.job} did not finish in time")
            time.sleep(poll_interval)

    def cancel(self) -> None:
        self.service._active()
        request = _json({"service": self.service.name, "job": self.job})
        _checked(self.service._client._runtime.cancel_training(request))
