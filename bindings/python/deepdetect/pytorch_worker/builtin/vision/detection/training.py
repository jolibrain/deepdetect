from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....artifacts import write_json_artifact
from ....sdk import DatasetContractError, WorkerContext, WorkerReporter
from ....tensors import TensorBatchRef, parse_tensor_batch_ref

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
    source: str
    train_list: Path | None
    test_lists: list[Path]
    train_tensor_batches: list[TensorBatchRef]
    test_tensor_batches: list[list[TensorBatchRef]]
    options: DetectionTrainOptions

    @classmethod
    def from_params(
        cls, context: WorkerContext | None, params: dict[str, Any]
    ) -> "DetectionTrainRequest":
        request = request_dict(params)
        request_params = parameters_dict(request)
        effective_mllib = merged_mllib(context, request_params)
        data = request.get("data", [])
        tensor_batches = request.get("tensor_batches")
        data_source = str(effective_mllib.get("data_source", ""))
        if data_source == "connector_tensor_pull":
            if tensor_batches is not None:
                raise DatasetContractError(
                    "connector_tensor_pull train request must not include tensor_batches"
                )
            if not isinstance(data, list) or not data:
                raise DatasetContractError(
                    "connector_tensor_pull train request data must contain list paths"
                )
            return cls(
                request=request,
                request_params=request_params,
                effective_mllib=effective_mllib,
                source="connector_pull",
                train_list=Path(str(data[0])),
                test_lists=[Path(str(path)) for path in data[1:]],
                train_tensor_batches=[],
                test_tensor_batches=[],
                options=DetectionTrainOptions.from_mllib(effective_mllib),
            )
        if data and tensor_batches is not None:
            raise DatasetContractError(
                "train request must not mix path data and tensor_batches"
            )
        if isinstance(data, list) and data:
            return cls(
                request=request,
                request_params=request_params,
                effective_mllib=effective_mllib,
                source="path",
                train_list=Path(str(data[0])),
                test_lists=[Path(str(path)) for path in data[1:]],
                train_tensor_batches=[],
                test_tensor_batches=[],
                options=DetectionTrainOptions.from_mllib(effective_mllib),
            )
        if tensor_batches is not None:
            train_batches, test_batches = parse_tensor_batch_sections(tensor_batches)
            return cls(
                request=request,
                request_params=request_params,
                effective_mllib=effective_mllib,
                source="tensor",
                train_list=None,
                test_lists=[],
                train_tensor_batches=train_batches,
                test_tensor_batches=test_batches,
                options=DetectionTrainOptions.from_mllib(effective_mllib),
            )
        if not isinstance(data, list) or not data:
            raise DatasetContractError(
                "train request data must contain a train list path or "
                "tensor_batches.train"
            )
        raise DatasetContractError("train request data must be a list")


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
        source: str = "path",
        request: dict[str, Any],
        request_params: dict[str, Any],
        effective_mllib: dict[str, Any],
    ) -> None:
        if self.context is None:
            return
        input_params = training_parameter_section(request_params, "input")
        output_params = training_parameter_section(request_params, "output")
        if source == "path":
            data = request.get("data", [])
            data_paths = [str(Path(str(path)).expanduser()) for path in data]
            data_contract = {"data": data_paths}
            train_manifest = {
                "path": str(train_dataset.list_path),
                "samples": len(train_dataset),
            }
            test_manifests = [
                {
                    "index": index,
                    "path": str(dataset.list_path),
                    "samples": len(dataset),
                }
                for index, dataset in enumerate(test_datasets)
            ]
            boundary = "path-backed"
        elif source == "tensor":
            train_batch_count = dataset_batch_count(train_dataset)
            test_batch_counts = [
                dataset_batch_count(dataset) for dataset in test_datasets
            ]
            data_contract = {
                "data": [],
                "tensor_batches": {
                    "train_batches": train_batch_count,
                    "test_batches": test_batch_counts,
                },
            }
            train_manifest = {
                "source": "tensor-backed",
                "batches": train_batch_count,
                "samples": len(train_dataset),
            }
            test_manifests = [
                {
                    "index": index,
                    "source": "tensor-backed",
                    "batches": test_batch_counts[index],
                    "samples": len(dataset),
                }
                for index, dataset in enumerate(test_datasets)
            ]
            boundary = "tensor-backed"
        elif source == "connector_pull":
            data = request.get("data", [])
            data_paths = [str(Path(str(path)).expanduser()) for path in data]
            data_contract = {
                "data": data_paths,
                "data_source": "connector_tensor_pull",
            }
            train_manifest = {
                "source": "connector-tensor-pull",
                "samples": len(train_dataset),
            }
            test_manifests = [
                {
                    "index": index,
                    "source": "connector-tensor-pull",
                    "samples": len(dataset),
                }
                for index, dataset in enumerate(test_datasets)
            ]
            boundary = "connector-tensor-pull"
        else:
            raise DatasetContractError(f"unsupported train request source: {source}")
        config_payload = {
            "worker": self.worker_name,
            "task": self.task_name,
            "repository": self.context.repository,
            "configure_mllib": dict(self.context.mllib),
            "train_mllib": effective_mllib,
            "input_parameters": input_params,
            "output_parameters": output_params,
        }
        config_payload.update(data_contract)
        manifest_payload = {
            "version": 1,
            "boundary": boundary,
            "task": self.task_name,
            "nclasses": self.nclasses,
            "repository": self.context.repository,
            "train": train_manifest,
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


def parse_tensor_batch_sections(
    tensor_batches: Any,
) -> tuple[list[TensorBatchRef], list[list[TensorBatchRef]]]:
    if not isinstance(tensor_batches, dict):
        raise DatasetContractError("train request tensor_batches must be an object")
    train_batches = tensor_batches.get("train")
    if not isinstance(train_batches, list) or not train_batches:
        raise DatasetContractError(
            "train request tensor_batches.train must be a non-empty list"
        )
    tests = tensor_batches.get("tests", [])
    if tests is None:
        tests = []
    if not isinstance(tests, list):
        raise DatasetContractError("train request tensor_batches.tests must be a list")
    parsed_tests = []
    for index, test_set in enumerate(tests):
        test_batches = tensor_test_batches(test_set, index)
        parsed_tests.append([parse_tensor_batch_ref(item) for item in test_batches])
    return [parse_tensor_batch_ref(item) for item in train_batches], parsed_tests


def tensor_test_batches(test_set: Any, index: int) -> list[Any]:
    if isinstance(test_set, dict):
        test_set = test_set.get("batches")
    if not isinstance(test_set, list) or not test_set:
        raise DatasetContractError(
            "train request tensor_batches.tests[%s] must be a non-empty "
            "list or object with non-empty batches" % index
        )
    return test_set


def dataset_batch_count(dataset: Any) -> int:
    batches = getattr(dataset, "batches", None)
    if isinstance(batches, list):
        return len(batches)
    return len(dataset)


@dataclass(frozen=True)
class DetectionDatasetSummary:
    samples: int

    def __len__(self) -> int:
        return self.samples


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
