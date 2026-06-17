from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....artifacts import write_json_artifact
from ....sdk import DatasetContractError, WorkerContext, WorkerReporter

from .common import (
    checkpoint_path,
    latest_checkpoint,
    maybe_load_checkpoint,
    maybe_load_solver,
    report_train_step,
    save_checkpoint,
)


@dataclass(frozen=True)
class DetectionTrainOptions:
    iterations: int
    test_interval: int
    batch_size: int
    iter_size: int
    base_lr: float

    @classmethod
    def from_mllib(cls, mllib: dict[str, Any]) -> "DetectionTrainOptions":
        solver = dict(
            mllib.get("solver", {}) if isinstance(mllib.get("solver"), dict) else {}
        )
        net = dict(mllib.get("net", {}) if isinstance(mllib.get("net"), dict) else {})
        iterations = positive_int(solver.get("iterations", 1), "iterations")
        return cls(
            iterations=iterations,
            test_interval=positive_int(
                solver.get("test_interval", iterations), "test_interval"
            ),
            batch_size=positive_int(net.get("batch_size", 1), "batch_size"),
            iter_size=positive_int(solver.get("iter_size", 1), "iter_size"),
            base_lr=float(solver.get("base_lr", 0.0001)),
        )


@dataclass(frozen=True)
class DetectionTrainRequest:
    request: dict[str, Any]
    request_params: dict[str, Any]
    effective_mllib: dict[str, Any]
    train_list: Path
    test_lists: list[Path]
    options: DetectionTrainOptions

    @classmethod
    def from_params(
        cls, context: WorkerContext | None, params: dict[str, Any]
    ) -> "DetectionTrainRequest":
        request = request_dict(params)
        request_params = parameters_dict(request)
        effective_mllib = merged_mllib(context, request_params)
        data = request.get("data", [])
        if not isinstance(data, list) or not data:
            raise DatasetContractError(
                "train request data must contain a train list path"
            )
        return cls(
            request=request,
            request_params=request_params,
            effective_mllib=effective_mllib,
            train_list=Path(str(data[0])),
            test_lists=[Path(str(path)) for path in data[1:]],
            options=DetectionTrainOptions.from_mllib(effective_mllib),
        )


class DetectionCheckpointManager:
    def __init__(self, context: WorkerContext | None, torch: Any, device: Any) -> None:
        self.context = context
        self.torch = torch
        self.device = device

    def load_model_for_training(self, model: Any, mllib: dict[str, Any]) -> Path | None:
        return maybe_load_checkpoint(
            model,
            self.torch,
            self.device,
            checkpoint_path(mllib, self.context),
        )

    def load_model_for_prediction(self, model: Any) -> Path | None:
        return maybe_load_checkpoint(
            model,
            self.torch,
            self.device,
            latest_checkpoint(self.context),
        )

    def load_optimizer(self, optimizer: Any, mllib: dict[str, Any]) -> None:
        maybe_load_solver(optimizer, self.torch, self.device, self.context, mllib)

    def save(self, model: Any, optimizer: Any, iteration: int) -> None:
        save_checkpoint(self.context, model, optimizer, self.torch, iteration)


class DetectionRepositoryContractWriter:
    def __init__(
        self,
        context: WorkerContext | None,
        *,
        worker_name: str,
        task_name: str,
        nclasses: int,
    ) -> None:
        self.context = context
        self.worker_name = worker_name
        self.task_name = task_name
        self.nclasses = nclasses

    def write(
        self,
        *,
        train_dataset: Any,
        test_datasets: list[Any],
        request: dict[str, Any],
        request_params: dict[str, Any],
        effective_mllib: dict[str, Any],
    ) -> None:
        if self.context is None:
            return
        input_params = training_parameter_section(request_params, "input")
        output_params = training_parameter_section(request_params, "output")
        data = request.get("data", [])
        data_paths = [str(Path(str(path)).expanduser()) for path in data]
        test_manifests = [
            {
                "index": index,
                "path": str(dataset.list_path),
                "samples": len(dataset),
            }
            for index, dataset in enumerate(test_datasets)
        ]
        config_payload = {
            "worker": self.worker_name,
            "task": self.task_name,
            "repository": self.context.repository,
            "configure_mllib": dict(self.context.mllib),
            "train_mllib": effective_mllib,
            "input_parameters": input_params,
            "output_parameters": output_params,
            "data": data_paths,
        }
        manifest_payload = {
            "version": 1,
            "boundary": "path-backed",
            "task": self.task_name,
            "nclasses": self.nclasses,
            "repository": self.context.repository,
            "train": {
                "path": str(train_dataset.list_path),
                "samples": len(train_dataset),
            },
            "tests": test_manifests,
            "input_parameters": input_params,
            "output_parameters": output_params,
        }
        write_json_artifact(
            self.context.artifact_path("pytorch_worker_config.json"),
            config_payload,
        )
        write_json_artifact(
            self.context.artifact_path("connector_manifest.json"),
            manifest_payload,
        )
        write_json_artifact(
            self.context.artifact_path("class_mapping.json"),
            class_mapping(self.nclasses, effective_mllib),
        )


class DetectionProgressReporter:
    def __init__(self, reporter: WorkerReporter) -> None:
        self.reporter = reporter

    def cancelled(self, *, iteration: int, iterations: int) -> None:
        self.reporter.status(
            phase="cancelled",
            iteration=iteration,
            iterations=iterations,
            test_active=0,
        )

    def train_step(
        self,
        *,
        iteration: int,
        iterations: int,
        start_time: float,
        base_lr: float,
        train_loss: float,
        losses: dict[str, float],
    ) -> None:
        report_train_step(
            self.reporter,
            iteration=iteration,
            iterations=iterations,
            start_time=start_time,
            base_lr=base_lr,
            train_loss=train_loss,
            losses=losses,
        )

    def test_progress(
        self,
        *,
        iteration: int,
        test_index: int,
        test_sets_total: int,
        processed: int,
        total: int,
    ) -> None:
        self.reporter.status(
            phase="test",
            iteration=iteration,
            test_active=1,
            test_set_index=test_index,
            test_sets_total=test_sets_total,
            test_processed=processed,
            test_total=total,
        )

    def test_finished(
        self,
        *,
        iteration: int,
        test_sets_total: int,
        predictions_payload: dict[str, Any],
    ) -> None:
        self.reporter.status(
            phase="train",
            iteration=iteration,
            test_active=0,
            test_set_index=max(0, test_sets_total - 1),
            test_sets_total=test_sets_total,
            test_processed=0,
            test_total=0,
            test_predictions=predictions_payload,
        )

    def finished(self, *, iteration: int, iterations: int) -> None:
        self.reporter.status(
            phase="finished",
            iteration=iteration,
            iterations=iterations,
            test_active=0,
        )


def training_parameter_section(params: dict[str, Any], name: str) -> dict[str, Any]:
    value = params.get(name)
    if value is None:
        value = params.get(f"{name}_parameters")
    return dict(value) if isinstance(value, dict) else {}


def class_mapping(nclasses: int, mllib: dict[str, Any]) -> dict[str, str]:
    names = mllib.get("class_names")
    if names is None:
        names = mllib.get("classes")
    mapping = {"0": "background"}
    for index in range(1, int(nclasses)):
        label = str(index)
        if isinstance(names, list) and index < len(names):
            label = str(names[index])
        elif isinstance(names, dict):
            label = str(names.get(str(index), names.get(index, label)))
        mapping[str(index)] = label
    return mapping


def request_dict(params: dict[str, Any]) -> dict[str, Any]:
    request = params.get("request", {})
    return request if isinstance(request, dict) else {}


def parameters_dict(request: dict[str, Any]) -> dict[str, Any]:
    parameters = request.get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}


def merged_mllib(
    context: WorkerContext | None, request_params: dict[str, Any]
) -> dict[str, Any]:
    result = dict(context.mllib if context is not None else {})
    request_mllib = request_params.get("mllib", {})
    if isinstance(request_mllib, dict):
        result.update(request_mllib)
    return result


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise DatasetContractError(f"{name} must be positive")
    return result


def optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return positive_int(value, name)
