from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping


class WorkerSDKError(Exception):
    category = "internal_error"


class WorkerContractError(WorkerSDKError):
    category = "worker_contract_error"


class MetricContractError(WorkerSDKError):
    category = "metric_contract_error"


class PredictionContractError(WorkerSDKError):
    category = "prediction_contract_error"


class DatasetContractError(WorkerSDKError):
    category = "dataset_contract_error"


class WorkerDependencyError(WorkerSDKError):
    category = "dependency_error"


class WorkerLaunchError(WorkerSDKError):
    category = "worker_launch_error"


@dataclass
class Cancellation:
    requested: bool = False


@dataclass(frozen=True)
class WorkerContext:
    repository: str
    mllib: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def from_configure_params(cls, params: Mapping[str, Any]) -> "WorkerContext":
        repository = params.get("repository", "")
        if not isinstance(repository, str):
            raise WorkerContractError("configure.repository must be a string")
        mllib = params.get("mllib", {})
        if not isinstance(mllib, Mapping):
            raise WorkerContractError("configure.mllib must be an object")
        return cls(repository=repository, mllib=mllib, raw=params)

    @property
    def repository_path(self) -> Path:
        return Path(self.repository)

    def artifact_path(self, *parts: str) -> Path:
        return self.repository_path.joinpath(*parts)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


class WorkerReporter:
    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self._emit = emit

    def status(self, **payload: Any) -> None:
        validate_json_payload(payload, WorkerContractError, "status payload")
        self._emit("status", dict(payload))

    def metric(self, name: str, value: Any, *, iteration: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise MetricContractError("metric name must be a non-empty string")
        metric_value = finite_scalar(value, "metric value")
        payload: dict[str, Any] = {"name": name, "value": metric_value}
        if iteration is not None:
            payload["iteration"] = finite_scalar(iteration, "metric iteration")
        self._emit("metric", payload)

    def artifact(self, **payload: Any) -> None:
        validate_json_payload(payload, WorkerContractError, "artifact payload")
        self._emit("artifact", dict(payload))

    def log(self, level: str, message: str, **payload: Any) -> None:
        if not isinstance(level, str) or not level:
            raise WorkerContractError("log level must be a non-empty string")
        if not isinstance(message, str):
            raise WorkerContractError("log message must be a string")
        event_payload = {"level": level, "message": message, **payload}
        validate_json_payload(event_payload, WorkerContractError, "log payload")
        self._emit("log", event_payload)


class DeepDetectWorkerBase:
    def __init__(self) -> None:
        self.context: WorkerContext | None = None

    def configure(self, context: WorkerContext) -> dict[str, Any]:
        self.context = context
        return {}

    def train(
        self,
        params: dict[str, Any],
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
    ) -> dict[str, Any]:
        raise WorkerContractError("worker must implement train()")

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        raise WorkerContractError("worker must implement predict()")


def finite_scalar(
    value: Any,
    label: str,
    error_type: type[WorkerSDKError] = MetricContractError,
) -> float | int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{label} must be a finite numeric scalar")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise error_type(f"{label} must be finite")
    return value


def validate_json_payload(
    payload: Any, error_type: type[WorkerSDKError], label: str
) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise error_type(f"{label} must be JSON serializable") from error


def validate_optional_result_dict(
    result: Any, error_type: type[WorkerSDKError], label: str
) -> dict[str, Any]:
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise error_type(f"{label} must return a dict")
    validate_json_payload(result, error_type, label)
    return result


def validate_prediction_result(result: Any) -> dict[str, Any]:
    result = validate_optional_result_dict(
        result, PredictionContractError, "predict result"
    )
    results = result.get("results")
    if not isinstance(results, list):
        raise PredictionContractError("predict result must contain results list")
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise PredictionContractError(f"prediction {index} must be an object")
        _validate_bboxes(item, index)
    return result


def _validate_bboxes(item: Mapping[str, Any], prediction_index: int) -> None:
    if "bboxes" not in item:
        return
    bboxes = item["bboxes"]
    if not isinstance(bboxes, list):
        raise PredictionContractError(
            f"prediction {prediction_index} bboxes must be a list"
        )
    for bbox_index, bbox in enumerate(bboxes):
        if not isinstance(bbox, Mapping):
            raise PredictionContractError(
                f"prediction {prediction_index} bbox {bbox_index} must be an object"
            )
        for key in ("xmin", "ymin", "xmax", "ymax"):
            if key not in bbox:
                raise PredictionContractError(
                    f"prediction {prediction_index} bbox {bbox_index} missing {key}"
                )
            finite_scalar(bbox[key], f"bbox {key}", PredictionContractError)
