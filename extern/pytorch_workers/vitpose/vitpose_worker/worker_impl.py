from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from deepdetect.pytorch_worker.artifacts import write_json_artifact
from deepdetect.pytorch_worker.builtin.vision.detection.common import select_device
from deepdetect.pytorch_worker.builtin.vision.detection.training import (
    connector_session_summary,
    dataset_batch_count,
    merged_mllib,
    parameters_dict,
    parse_tensor_batch_sections,
    positive_int,
    request_dict,
    training_parameter_section,
)
from deepdetect.pytorch_worker.sdk import (
    Cancellation,
    DatasetContractError,
    DeepDetectWorkerBase,
    PredictionContractError,
    WorkerContext,
    WorkerDependencyError,
    WorkerReporter,
)
from deepdetect.pytorch_worker.tensors import parse_tensor_batch_ref

from .checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    load_model_checkpoint,
    load_optimizer_checkpoint,
    save_checkpoint,
)
from .config import ViTPoseWorkerConfig, worker_config_from_mllib
from .data import (
    PoseTensorBatchDataset,
    make_loader,
    move_pose_target,
    normalize_batch,
    read_image_tensor,
)
from .decode import (
    connector_predictions,
    decode_keypoint,
    decode_pose_outputs,
    decode_topdown_outputs,
    prediction_sample,
)
from .losses import PoseLossReduction, slot_pose_losses, topdown_pose_losses
from .model import ViTPoseSlots, ViTPoseTopDown
from .optim import create_layer_decay_adamw
from .targets import select_instances


@dataclass(frozen=True)
class PoseTrainOptions:
    iterations: int
    test_interval: int
    batch_size: int
    iter_size: int
    base_lr: float

    @classmethod
    def from_mllib(cls, mllib: dict[str, Any]) -> "PoseTrainOptions":
        solver = dict(mllib.get("solver", {}) if isinstance(mllib.get("solver"), dict) else {})
        net = dict(mllib.get("net", {}) if isinstance(mllib.get("net"), dict) else {})
        iterations = positive_int(solver.get("iterations", 1), "iterations")
        return cls(
            iterations=iterations,
            test_interval=positive_int(solver.get("test_interval", iterations), "test_interval"),
            batch_size=positive_int(net.get("batch_size", 1), "batch_size"),
            iter_size=positive_int(solver.get("iter_size", 1), "iter_size"),
            base_lr=float(solver.get("base_lr", 0.0005)),
        )


@dataclass(frozen=True)
class PoseTrainRequest:
    request: dict[str, Any]
    request_params: dict[str, Any]
    effective_mllib: dict[str, Any]
    source: str
    train_list: Path | None
    test_lists: list[Path]
    train_tensor_batches: list[Any]
    test_tensor_batches: list[list[Any]]
    options: PoseTrainOptions

    @classmethod
    def from_params(
        cls,
        context: WorkerContext | None,
        params: dict[str, Any],
    ) -> "PoseTrainRequest":
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
                options=PoseTrainOptions.from_mllib(effective_mllib),
            )
        if data and tensor_batches is not None:
            raise DatasetContractError("train request must not mix path data and tensor_batches")
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
                options=PoseTrainOptions.from_mllib(effective_mllib),
            )
        raise DatasetContractError(
            "ViTPose training requires connector_tensor_pull or tensor_batches"
        )


def merge_training_mllib(
    service_mllib: dict[str, Any], train_mllib: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(service_mllib)
    merged.update(train_mllib)
    service_vitpose = service_mllib.get("vitpose", {})
    train_vitpose = train_mllib.get("vitpose", {})
    if isinstance(service_vitpose, dict) and isinstance(train_vitpose, dict):
        merged["vitpose"] = {**service_vitpose, **train_vitpose}
    return merged


@dataclass(frozen=True)
class PoseDatasetSummary:
    samples: int

    def __len__(self) -> int:
        return self.samples


@dataclass
class PoseEvaluationLosses:
    heatmap_numerator: float = 0.0
    heatmap_denominator: float = 0.0
    objectness_numerator: float = 0.0
    objectness_denominator: int = 0
    keypoint_error_sum: float = 0.0
    keypoint_error_count: int = 0

    def update(self, reduction: PoseLossReduction) -> None:
        self.heatmap_numerator += float(
            reduction.heatmap_numerator.detach().cpu().item()
        )
        self.heatmap_denominator += float(
            reduction.heatmap_denominator.detach().cpu().item()
        )
        if reduction.objectness_numerator is not None:
            self.objectness_numerator += float(
                reduction.objectness_numerator.detach().cpu().item()
            )
            self.objectness_denominator += reduction.objectness_denominator

    def add_keypoint_error(self, x_error: float, y_error: float) -> None:
        self.keypoint_error_sum += math.hypot(x_error, y_error)
        self.keypoint_error_count += 1

    def metrics(self, config: ViTPoseWorkerConfig) -> dict[str, float]:
        heatmap_loss = (
            self.heatmap_numerator / self.heatmap_denominator
            if self.heatmap_denominator > 0.0
            else 0.0
        )
        heatmap_area = config.target.heatmap_size[0] * config.target.heatmap_size[1]
        metrics = {
            "heatmap_loss": heatmap_loss,
            "loss": float(config.loss.heatmap_weight) * heatmap_loss,
            "visible_keypoints": self.heatmap_denominator / float(heatmap_area),
            "mean_keypoint_error_px": (
                self.keypoint_error_sum / float(self.keypoint_error_count)
                if self.keypoint_error_count > 0
                else 0.0
            ),
        }
        if self.objectness_denominator > 0:
            objectness_loss = self.objectness_numerator / float(
                self.objectness_denominator
            )
            metrics["objectness_loss"] = objectness_loss
            metrics["loss"] += float(config.loss.objectness_weight) * objectness_loss
        return metrics


class DeepDetectWorker(DeepDetectWorkerBase):
    worker_name = "vitpose"
    task_name = "keypoint"
    debug_name = "vitpose"

    def __init__(self) -> None:
        super().__init__()
        self.nkeypoints = 17
        self.max_objects = 1
        self.device: Any = None
        self.model: Any = None
        self.multi_gpu_requested = False
        self.config: ViTPoseWorkerConfig | None = None

    def import_backend(self) -> tuple[Any, ...]:
        try:
            import torch
        except Exception as error:
            raise WorkerDependencyError("torch could not be imported") from error
        return (torch,)

    def configure(self, context: WorkerContext) -> dict[str, Any]:
        backend = self.import_backend()
        torch = backend[0]
        super().configure(context)
        mllib = dict(context.mllib)
        self.config = worker_config_from_mllib(mllib)
        self.nkeypoints = int(self.config.model.nkeypoints)
        self.max_objects = int(self.config.model.max_objects)
        self.device, self.multi_gpu_requested = select_device(torch, mllib)
        return {
            "worker": self.worker_name,
            "task": self.task_name,
            "nkeypoints": self.nkeypoints,
            "max_objects": self.max_objects,
            "head": self.config.head,
            "device": str(self.device),
            "torch_version": str(getattr(torch, "__version__", "unknown")),
        }

    def train(
        self,
        params: dict[str, Any],
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
    ) -> dict[str, Any]:
        backend = self.import_backend()
        torch = backend[0]
        train_request = PoseTrainRequest.from_params(self.context, params)
        train_request = self.configure_training_request(train_request)
        if self.multi_gpu_requested:
            reporter.log(
                "warning",
                "multiple GPU ids were requested; ViTPose worker uses the first id in this slice",
            )
        if train_request.source == "connector_pull":
            return self.train_connector_pull(
                train_request,
                reporter=reporter,
                cancellation=cancellation,
                torch=torch,
            )
        return self.train_tensor(
            train_request,
            reporter=reporter,
            cancellation=cancellation,
            torch=torch,
        )

    def configure_training_request(self, train_request: PoseTrainRequest) -> PoseTrainRequest:
        mllib = merge_training_mllib(
            self.context.mllib if self.context is not None else {},
            train_request.effective_mllib,
        )
        pretrained = checkpoint_path(
            mllib,
            self.context.repository_path if self.context is not None else None,
        ) is not None
        self.config = worker_config_from_mllib(mllib, pretrained=pretrained)
        self.nkeypoints = int(self.config.model.nkeypoints)
        self.max_objects = int(self.config.model.max_objects)
        return replace(train_request, effective_mllib=mllib)

    def train_tensor(
        self,
        train_request: PoseTrainRequest,
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
        torch: Any,
    ) -> dict[str, Any]:
        train_dataset = PoseTensorBatchDataset(
            train_request.train_tensor_batches,
            nkeypoints=self.nkeypoints,
            torch=torch,
        )
        test_datasets = [
            PoseTensorBatchDataset(batches, nkeypoints=self.nkeypoints, torch=torch)
            for batches in train_request.test_tensor_batches
        ]
        self.write_repository_contract(
            train_request,
            train_dataset=train_dataset,
            test_datasets=test_datasets,
            source="tensor",
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=train_request.options.batch_size,
            shuffle=True,
            torch=torch,
        )
        return self._run_training_loop(
            train_request,
            train_batches=RepeatingLoader(train_loader),
            test_datasets=test_datasets,
            reporter=reporter,
            cancellation=cancellation,
            torch=torch,
        )

    def train_connector_pull(
        self,
        train_request: PoseTrainRequest,
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
        torch: Any,
    ) -> dict[str, Any]:
        if self.context is None or self.context.connector is None:
            raise DatasetContractError("connector_tensor_pull requires a worker connector")
        connector = self.context.connector
        dataset_info = connector.dataset_info()
        train_samples = positive_int(dataset_info.get("train_samples", 0), "connector train_samples")
        test_samples = connector_test_samples(dataset_info)
        self.write_repository_contract(
            train_request,
            train_dataset=PoseDatasetSummary(train_samples),
            test_datasets=[PoseDatasetSummary(count) for count in test_samples],
            source="connector_pull",
            connector_info=dataset_info,
        )
        train_batches = RepeatingConnectorBatches(
            self,
            split="train",
            batch_size=train_request.options.batch_size,
            connector=connector,
            torch=torch,
            prefetch_batches=self.config.connector_prefetch_batches if self.config else 2,
        )
        try:
            return self._run_training_loop(
                train_request,
                train_batches=train_batches,
                test_samples=test_samples,
                connector=connector,
                reporter=reporter,
                cancellation=cancellation,
                torch=torch,
            )
        finally:
            train_batches.close()

    def _run_training_loop(
        self,
        train_request: PoseTrainRequest,
        *,
        train_batches: Any,
        reporter: WorkerReporter,
        cancellation: Cancellation,
        torch: Any,
        test_datasets: list[Any] | None = None,
        test_samples: list[int] | None = None,
        connector: Any = None,
    ) -> dict[str, Any]:
        options = train_request.options
        self.model = self.create_model(torch).to(self.device)
        loaded_path = checkpoint_path(
            train_request.effective_mllib,
            self.context.repository_path if self.context is not None else None,
        )
        load_model_checkpoint(torch, self.model, loaded_path, device=self.device)
        self.model.train()
        optimizer = self.create_optimizer(torch, self.model, base_lr=options.base_lr)
        load_optimizer_checkpoint(
            torch,
            optimizer,
            self.context.repository_path if self.context is not None else None,
            device=self.device,
            mllib=train_request.effective_mllib,
        )
        optimizer.zero_grad(set_to_none=True)
        start_time = time.monotonic()
        optimizer_steps = 0
        accumulated = 0
        latest_loss = 0.0
        dropped_total = 0.0
        while optimizer_steps < options.iterations:
            if cancellation.requested:
                save_checkpoint(
                    torch,
                    self.model,
                    optimizer,
                    self.context.repository_path if self.context is not None else None,
                    optimizer_steps,
                )
                reporter.status(
                    phase="cancelled",
                    iteration=optimizer_steps,
                    iterations=options.iterations,
                    test_active=0,
                )
                return {"status": "cancelled", "iteration": optimizer_steps}
            batch = next_pose_batch(train_batches)
            if batch is None:
                continue
            images, targets, _metas = batch
            images, targets = self.prepare_training_batch(torch, images, targets)
            outputs = self.model(images)
            loss_fn = (
                topdown_pose_losses
                if self.config.head == "topdown"
                else slot_pose_losses
            )
            loss_dict, stats = loss_fn(
                outputs,
                targets,
                config=self.config.loss,
                torch_module=torch,
                device=self.device,
            )
            total_loss = loss_dict["loss"]
            (total_loss / float(options.iter_size)).backward()
            accumulated += 1
            latest_loss = float(total_loss.detach().cpu().item())
            dropped_total += stats["dropped_objects"]
            if accumulated < options.iter_size:
                continue
            if self.config is not None and self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            optimizer_steps += 1
            loss_values = {
                name: float(value.detach().cpu().item())
                for name, value in loss_dict.items()
            }
            report_train_step(
                reporter,
                iteration=optimizer_steps,
                iterations=options.iterations,
                start_time=start_time,
                base_lr=options.base_lr,
                train_loss=latest_loss,
                losses=loss_values,
            )
            reporter.metric("assigned_objects", stats["assigned_objects"], iteration=optimizer_steps)
            if dropped_total:
                reporter.metric("dropped_objects", dropped_total, iteration=optimizer_steps)
            should_test = (
                (test_datasets or test_samples)
                and (optimizer_steps % options.test_interval == 0 or optimizer_steps == options.iterations)
            )
            if should_test:
                if test_datasets is not None:
                    self.evaluate_tensor(
                        test_datasets,
                        reporter=reporter,
                        iteration=optimizer_steps,
                        torch=torch,
                        cancellation=cancellation,
                    )
                elif connector is not None and test_samples is not None:
                    self.evaluate_connector(
                        test_samples,
                        connector=connector,
                        reporter=reporter,
                        iteration=optimizer_steps,
                        torch=torch,
                        cancellation=cancellation,
                        batch_size=options.batch_size,
                    )
                self.model.train()
                save_checkpoint(
                    torch,
                    self.model,
                    optimizer,
                    self.context.repository_path if self.context is not None else None,
                    optimizer_steps,
                )
        save_checkpoint(
            torch,
            self.model,
            optimizer,
            self.context.repository_path if self.context is not None else None,
            options.iterations,
        )
        reporter.status(
            phase="finished",
            iteration=options.iterations,
            iterations=options.iterations,
            test_active=0,
        )
        return {
            "status": "finished",
            "iteration": options.iterations,
            "train_loss": latest_loss,
        }

    def create_model(self, torch: Any) -> Any:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        if self.config.head == "topdown":
            return ViTPoseTopDown(self.config.model)
        return ViTPoseSlots(self.config.model)

    def create_optimizer(self, torch: Any, model: Any, *, base_lr: float) -> Any:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        return create_layer_decay_adamw(
            torch,
            model,
            base_lr=base_lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.adamw_betas,
            layer_decay=self.config.layer_decay,
        )

    def prepare_training_batch(
        self,
        torch: Any,
        images: list[Any],
        targets: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        batch = normalize_batch(
            images,
            torch=torch,
            device=self.device,
            mean=self.config.mean,
            std=self.config.std,
        )
        return batch, [move_pose_target(target, self.device) for target in targets]

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        backend = self.import_backend()
        torch = backend[0]
        request = request_dict(params)
        request_params = parameters_dict(request)
        output_params = request_params.get("output", {})
        output_params = output_params if isinstance(output_params, dict) else {}
        objectness_threshold = float(
            output_params.get(
                "confidence_threshold",
                self.config.objectness_threshold if self.config else 0.25,
            )
        )
        keypoint_threshold = float(
            output_params.get(
                "keypoint_threshold",
                self.config.keypoint_threshold if self.config else 0.05,
            )
        )
        data = request.get("data", [])
        if not isinstance(data, list):
            raise PredictionContractError("predict data must be a list")
        self.ensure_prediction_model(torch)
        if self.config.head == "topdown":
            return self.predict_topdown_request(
                request,
                torch=torch,
                keypoint_threshold=keypoint_threshold,
            )
        image_paths = [Path(str(path)).expanduser().resolve() for path in data]
        images = []
        image_sizes = []
        for image_path in image_paths:
            if not image_path.is_file():
                raise PredictionContractError(f"input image not found: {image_path}")
            image, original_size = read_image_tensor(
                image_path,
                torch,
                image_size=self.config.model.image_size,
            )
            images.append(image)
            image_sizes.append(self.config.model.image_size)
        with torch.no_grad():
            batch = normalize_batch(
                images,
                torch=torch,
                device=self.device,
                mean=self.config.mean,
                std=self.config.std,
            )
            outputs = self.model(batch)
            decoded = decode_pose_outputs(
                outputs,
                image_sizes=image_sizes,
                objectness_threshold=objectness_threshold,
                keypoint_threshold=keypoint_threshold,
            )
        return {"results": connector_predictions(image_paths, decoded)}

    def predict_topdown_request(
        self,
        request: dict[str, Any],
        *,
        torch: Any,
        keypoint_threshold: float,
    ) -> dict[str, Any]:
        raw_sources = request.get("pose_sources", request.get("data", []))
        if not isinstance(raw_sources, list):
            raise PredictionContractError("top-down pose sources must be a list")
        source_paths = [Path(str(path)) for path in raw_sources]
        grouped: list[list[dict[str, Any]]] = [[] for _ in source_paths]
        raw_batch = request.get("tensor_batch")
        if raw_batch is None:
            return {"results": connector_predictions(source_paths, grouped)}
        tensor_batch = parse_tensor_batch_ref(raw_batch)
        dataset = PoseTensorBatchDataset(
            [tensor_batch], nkeypoints=self.nkeypoints, torch=torch
        )
        loader = make_loader(dataset, batch_size=len(dataset), shuffle=False, torch=torch)
        images, _targets, metas = next(iter(loader))
        with torch.no_grad():
            batch = normalize_batch(
                images,
                torch=torch,
                device=self.device,
                mean=self.config.mean,
                std=self.config.std,
            )
            poses = decode_topdown_outputs(
                self.model(batch),
                metas=metas,
                keypoint_threshold=keypoint_threshold,
            )
        for pose in poses:
            source_index = int(pose.pop("source_index"))
            if source_index < 0 or source_index >= len(grouped):
                raise PredictionContractError("top-down source index is out of range")
            grouped[source_index].append(pose)
        for sample in grouped:
            sample.sort(key=lambda pose: int(pose.get("instance_id", 0)))
            for pose in sample:
                pose.pop("instance_id", None)
        return {"results": connector_predictions(source_paths, grouped)}

    def ensure_prediction_model(self, torch: Any) -> None:
        if self.config is None:
            if self.context is None:
                raise PredictionContractError("ViTPose worker is not configured")
            self.config = worker_config_from_mllib(dict(self.context.mllib))
        if self.model is None:
            self.model = self.create_model(torch).to(self.device)
            load_model_checkpoint(
                torch,
                self.model,
                latest_checkpoint(self.context.repository_path if self.context is not None else None),
                device=self.device,
            )
        self.model.eval()

    def evaluate_tensor(
        self,
        test_datasets: list[Any],
        *,
        reporter: WorkerReporter,
        iteration: int,
        torch: Any,
        cancellation: Cancellation,
    ) -> None:
        if self.model is None:
            raise PredictionContractError("model is not initialized")
        self.model.eval()
        predictions_payload: dict[str, Any] = {}
        with torch.no_grad():
            for test_index, dataset in enumerate(test_datasets):
                samples = []
                sample_lookup: dict[int, dict[str, Any]] = {}
                processed = 0
                evaluation_losses = PoseEvaluationLosses()
                loader = make_loader(dataset, batch_size=1, shuffle=False, torch=torch)
                reporter.status(
                    phase="test",
                    iteration=iteration,
                    test_active=1,
                    test_set_index=test_index,
                    test_sets_total=len(test_datasets),
                    test_processed=0,
                    test_total=len(dataset),
                )
                for images, targets, metas in loader:
                    if cancellation.requested:
                        break
                    poses = self.evaluate_pose_batch(
                        torch,
                        images,
                        targets,
                        metas,
                        evaluation_losses,
                    )
                    processed += len(images)
                    for meta, sample_poses in zip(metas, poses):
                        self.record_test_prediction(
                            samples, sample_lookup, meta, sample_poses
                        )
                    reporter.status(
                        phase="test",
                        iteration=iteration,
                        test_active=1,
                        test_set_index=test_index,
                        test_sets_total=len(test_datasets),
                        test_processed=processed,
                        test_total=len(dataset),
                    )
                predictions_payload[f"test{test_index}"] = {
                    "iteration": iteration,
                    "samples": samples,
                }
                if not cancellation.requested:
                    self.report_evaluation_metrics(
                        reporter,
                        iteration=iteration,
                        test_index=test_index,
                        losses=evaluation_losses,
                    )
                reporter.metric(f"pose_samples_test{test_index}", processed, iteration=iteration)
        reporter.status(
            phase="train",
            iteration=iteration,
            test_active=0,
            test_set_index=max(0, len(test_datasets) - 1),
            test_sets_total=len(test_datasets),
            test_processed=0,
            test_total=0,
            test_predictions=predictions_payload,
        )

    def evaluate_connector(
        self,
        test_samples: list[int],
        *,
        connector: Any,
        reporter: WorkerReporter,
        iteration: int,
        torch: Any,
        cancellation: Cancellation,
        batch_size: int,
    ) -> None:
        if self.model is None:
            raise PredictionContractError("model is not initialized")
        self.model.eval()
        predictions_payload: dict[str, Any] = {}
        with torch.no_grad():
            for test_index, total_samples in enumerate(test_samples):
                samples = []
                sample_lookup: dict[int, dict[str, Any]] = {}
                processed = 0
                evaluation_losses = PoseEvaluationLosses()
                prefetcher = self.connector_batch_prefetcher(
                    split="test",
                    batch_size=batch_size,
                    connector=connector,
                    torch=torch,
                    test_index=test_index,
                    reset_epoch=True,
                    prefetch_batches=self.config.connector_prefetch_batches if self.config else 2,
                )
                reporter.status(
                    phase="test",
                    iteration=iteration,
                    test_active=1,
                    test_set_index=test_index,
                    test_sets_total=len(test_samples),
                    test_processed=0,
                    test_total=total_samples,
                )
                try:
                    while not cancellation.requested:
                        batch = prefetcher.next()
                        if batch is None:
                            break
                        images, targets, metas = batch
                        poses = self.evaluate_pose_batch(
                            torch,
                            images,
                            targets,
                            metas,
                            evaluation_losses,
                        )
                        processed += len(images)
                        for meta, sample_poses in zip(metas, poses):
                            self.record_test_prediction(
                                samples, sample_lookup, meta, sample_poses
                            )
                        reporter.status(
                            phase="test",
                            iteration=iteration,
                            test_active=1,
                            test_set_index=test_index,
                            test_sets_total=len(test_samples),
                            test_processed=processed,
                            test_total=total_samples,
                        )
                finally:
                    prefetcher.close()
                predictions_payload[f"test{test_index}"] = {
                    "iteration": iteration,
                    "samples": samples,
                }
                if not cancellation.requested:
                    self.report_evaluation_metrics(
                        reporter,
                        iteration=iteration,
                        test_index=test_index,
                        losses=evaluation_losses,
                    )
                reporter.metric(f"pose_samples_test{test_index}", processed, iteration=iteration)
        reporter.status(
            phase="train",
            iteration=iteration,
            test_active=0,
            test_set_index=max(0, len(test_samples) - 1),
            test_sets_total=len(test_samples),
            test_processed=0,
            test_total=0,
            test_predictions=predictions_payload,
        )

    def predict_pose_batch(
        self,
        torch: Any,
        images: list[Any],
        metas: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        batch = normalize_batch(
            images,
            torch=torch,
            device=self.device,
            mean=self.config.mean,
            std=self.config.std,
        )
        outputs = self.model(batch)
        return self.decode_pose_batch(outputs, metas)

    def evaluate_pose_batch(
        self,
        torch: Any,
        images: list[Any],
        targets: list[dict[str, Any]],
        metas: list[dict[str, Any]],
        evaluation_losses: PoseEvaluationLosses,
    ) -> list[list[dict[str, Any]]]:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        batch, targets = self.prepare_training_batch(torch, images, targets)
        outputs = self.model(batch)
        loss_fn = (
            topdown_pose_losses if self.config.head == "topdown" else slot_pose_losses
        )
        _losses, _stats, reduction = loss_fn(
            outputs,
            targets,
            config=self.config.loss,
            torch_module=torch,
            device=self.device,
            return_reduction=True,
        )
        evaluation_losses.update(reduction)
        self.record_keypoint_errors(
            evaluation_losses,
            outputs=outputs,
            targets=targets,
            metas=metas,
            reduction=reduction,
        )
        return self.decode_pose_batch(outputs, metas)

    def record_keypoint_errors(
        self,
        evaluation_losses: PoseEvaluationLosses,
        *,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        metas: list[dict[str, Any]],
        reduction: PoseLossReduction,
    ) -> None:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        heatmaps = outputs["heatmaps"]
        if self.config.head == "topdown":
            for batch_index, (target, meta) in enumerate(zip(targets, metas)):
                if int(target["keypoints"].shape[0]) != 1:
                    continue
                self.record_instance_keypoint_errors(
                    evaluation_losses,
                    heatmaps=heatmaps[batch_index],
                    keypoints=target["keypoints"][0],
                    visible=target["visible"][0],
                    meta=meta,
                    topdown=True,
                )
            return

        selected_targets = []
        for target in targets:
            selected_targets.append(
                select_instances(
                    target["keypoints"],
                    target["visible"],
                    max_objects=self.config.target.max_objects,
                )
            )
        for batch_index, slot_index, object_index in reduction.assignments:
            if batch_index >= len(selected_targets) or batch_index >= len(metas):
                continue
            keypoints, visible, _dropped = selected_targets[batch_index]
            if object_index >= int(keypoints.shape[0]):
                continue
            self.record_instance_keypoint_errors(
                evaluation_losses,
                heatmaps=heatmaps[batch_index, slot_index],
                keypoints=keypoints[object_index],
                visible=visible[object_index],
                meta=metas[batch_index],
                topdown=False,
            )

    def record_instance_keypoint_errors(
        self,
        evaluation_losses: PoseEvaluationLosses,
        *,
        heatmaps: Any,
        keypoints: Any,
        visible: Any,
        meta: dict[str, Any],
        topdown: bool,
    ) -> None:
        for joint_index in range(int(heatmaps.shape[0])):
            if joint_index >= int(keypoints.shape[0]):
                break
            if float(visible[joint_index].item()) <= 0.0:
                continue
            target_x = float(keypoints[joint_index, 0].item())
            target_y = float(keypoints[joint_index, 1].item())
            if (
                not math.isfinite(target_x)
                or not math.isfinite(target_y)
                or target_x < 0.0
                or target_y < 0.0
            ):
                continue
            predicted_x, predicted_y, _confidence, _valid = decode_keypoint(
                heatmaps[joint_index],
                image_size=(int(meta["width"]), int(meta["height"])),
                threshold=float("-inf"),
            )
            predicted_x, predicted_y = self.source_keypoint_coordinates(
                predicted_x,
                predicted_y,
                meta=meta,
                topdown=topdown,
            )
            target_x, target_y = self.source_keypoint_coordinates(
                target_x,
                target_y,
                meta=meta,
                topdown=topdown,
            )
            evaluation_losses.add_keypoint_error(
                predicted_x - target_x,
                predicted_y - target_y,
            )

    @staticmethod
    def source_keypoint_coordinates(
        x: float,
        y: float,
        *,
        meta: dict[str, Any],
        topdown: bool,
    ) -> tuple[float, float]:
        if topdown:
            inverse = meta.get("inverse_affine")
            if not isinstance(inverse, list) or len(inverse) != 6:
                raise DatasetContractError(
                    "top-down keypoint evaluation requires inverse_affine metadata"
                )
            return (
                inverse[0] * x + inverse[1] * y + inverse[2],
                inverse[3] * x + inverse[4] * y + inverse[5],
            )
        width = int(meta["width"])
        height = int(meta["height"])
        original_width = int(meta.get("original_width", width))
        original_height = int(meta.get("original_height", height))
        return (
            x * float(original_width) / float(width),
            y * float(original_height) / float(height),
        )

    def decode_pose_batch(
        self,
        outputs: dict[str, Any],
        metas: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        if self.config.head == "topdown":
            return [
                [pose]
                for pose in decode_topdown_outputs(
                    outputs,
                    metas=metas,
                    keypoint_threshold=self.config.keypoint_threshold,
                )
            ]
        image_sizes = [(int(meta["width"]), int(meta["height"])) for meta in metas]
        return decode_pose_outputs(
            outputs,
            image_sizes=image_sizes,
            objectness_threshold=self.config.objectness_threshold,
            keypoint_threshold=self.config.keypoint_threshold,
        )

    def report_evaluation_metrics(
        self,
        reporter: WorkerReporter,
        *,
        iteration: int,
        test_index: int,
        losses: PoseEvaluationLosses,
    ) -> None:
        if self.config is None:
            raise DatasetContractError("ViTPose worker is not configured")
        for name, value in sorted(losses.metrics(self.config).items()):
            reporter.metric(f"{name}_test{test_index}", value, iteration=iteration)

    def record_test_prediction(
        self,
        samples: list[dict[str, Any]],
        sample_lookup: dict[int, dict[str, Any]],
        meta: dict[str, Any],
        poses: list[dict[str, Any]],
    ) -> None:
        if self.config.head != "topdown":
            if len(samples) < 10:
                samples.append(prediction_sample(meta, poses))
            return
        source_index = int(meta["index"])
        sample = sample_lookup.get(source_index)
        if sample is None:
            if len(samples) >= 10:
                return
            sample = {
                "index": source_index,
                "imgsize": {
                    "width": int(meta["original_width"]),
                    "height": int(meta["original_height"]),
                },
                "classes": [],
            }
            sample_lookup[source_index] = sample
            samples.append(sample)
        for pose in poses:
            cleaned = dict(pose)
            cleaned.pop("source_index", None)
            cleaned.pop("instance_id", None)
            sample["classes"].append(cleaned)

    def pull_pose_batch(
        self,
        *,
        split: str,
        batch_size: int,
        connector: Any,
        torch: Any,
        test_index: int | None = None,
        reset_epoch: bool = False,
    ) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
        response = connector.next_batch(
            split=split,
            batch_size=batch_size,
            test_index=test_index,
            reset_epoch=reset_epoch,
        )
        if response.get("end"):
            return None
        batch_id = response.get("batch_id")
        batch_payload = response.get("batch")
        if not isinstance(batch_payload, dict):
            raise DatasetContractError("connector_batch_next result missing batch")
        try:
            tensor_batch = parse_tensor_batch_ref(batch_payload)
            dataset = PoseTensorBatchDataset(
                [tensor_batch],
                nkeypoints=self.nkeypoints,
                torch=torch,
            )
            loader = make_loader(dataset, batch_size=len(dataset), shuffle=False, torch=torch)
            return next(iter(loader))
        finally:
            connector.batch_done(batch_id)

    def connector_batch_prefetcher(
        self,
        *,
        split: str,
        batch_size: int,
        connector: Any,
        torch: Any,
        reset_epoch: bool,
        prefetch_batches: int,
        test_index: int | None = None,
    ) -> "ConnectorBatchPrefetcher":
        return ConnectorBatchPrefetcher(
            lambda *, reset_epoch: self.pull_pose_batch(
                split=split,
                batch_size=batch_size,
                connector=connector,
                torch=torch,
                test_index=test_index,
                reset_epoch=reset_epoch,
            ),
            reset_epoch=reset_epoch,
            prefetch_batches=prefetch_batches,
        )

    def write_repository_contract(
        self,
        train_request: PoseTrainRequest,
        *,
        train_dataset: Any,
        test_datasets: list[Any],
        source: str,
        connector_info: dict[str, Any] | None = None,
    ) -> None:
        if self.context is None:
            return
        request_params = train_request.request_params
        payload = {
            "worker": self.worker_name,
            "task": self.task_name,
            "repository": self.context.repository,
            "configure_mllib": dict(self.context.mllib),
            "train_mllib": train_request.effective_mllib,
            "input_parameters": training_parameter_section(request_params, "input"),
            "output_parameters": training_parameter_section(request_params, "output"),
        }
        manifest = {
            "version": 1,
            "boundary": source,
            "task": self.task_name,
            "nkeypoints": self.nkeypoints,
            "max_objects": self.max_objects,
            "head": self.config.head if self.config is not None else "topdown",
            "repository": self.context.repository,
            "train": {"samples": len(train_dataset)},
            "tests": [
                {"index": index, "samples": len(dataset)}
                for index, dataset in enumerate(test_datasets)
            ],
        }
        if source == "connector_pull" and connector_info is not None:
            manifest["connector"] = connector_session_summary(connector_info)
        elif source == "tensor":
            payload["tensor_batches"] = {
                "train_batches": dataset_batch_count(train_dataset),
                "test_batches": [dataset_batch_count(dataset) for dataset in test_datasets],
            }
        write_json_artifact(self.context.artifact_path("pytorch_worker_config.json"), payload)
        write_json_artifact(self.context.artifact_path("connector_manifest.json"), manifest)


class ConnectorBatchPrefetcher:
    def __init__(
        self,
        pull: Any,
        *,
        reset_epoch: bool,
        prefetch_batches: int,
    ) -> None:
        self.pull = pull
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(prefetch_batches)))
        self.closed = threading.Event()
        self.reset_epoch = reset_epoch
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        reset_epoch = self.reset_epoch
        try:
            while not self.closed.is_set():
                batch = self.pull(reset_epoch=reset_epoch)
                reset_epoch = False
                if not self._put(batch):
                    return
                if batch is None:
                    return
        except BaseException as error:
            self._put(error)

    def _put(self, item: Any) -> bool:
        while not self.closed.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def next(self) -> Any:
        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class RepeatingLoader:
    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class RepeatingConnectorBatches:
    def __init__(
        self,
        worker: DeepDetectWorker,
        *,
        split: str,
        batch_size: int,
        connector: Any,
        torch: Any,
        prefetch_batches: int,
    ) -> None:
        self.worker = worker
        self.split = split
        self.batch_size = batch_size
        self.connector = connector
        self.torch = torch
        self.prefetch_batches = prefetch_batches
        self.prefetcher = self._new_prefetcher(reset_epoch=True)

    def _new_prefetcher(self, *, reset_epoch: bool) -> ConnectorBatchPrefetcher:
        return self.worker.connector_batch_prefetcher(
            split=self.split,
            batch_size=self.batch_size,
            connector=self.connector,
            torch=self.torch,
            reset_epoch=reset_epoch,
            prefetch_batches=self.prefetch_batches,
        )

    def __next__(self) -> Any:
        batch = self.prefetcher.next()
        if batch is not None:
            return batch
        self.prefetcher.close()
        self.prefetcher = self._new_prefetcher(reset_epoch=True)
        return self.prefetcher.next()

    def close(self) -> None:
        self.prefetcher.close()


def next_pose_batch(source: Any) -> Any:
    if isinstance(source, ConnectorBatchPrefetcher):
        batch = source.next()
        if batch is None:
            return None
        return batch
    try:
        return next(source)
    except StopIteration:
        return None


def connector_test_samples(info: dict[str, Any]) -> list[int]:
    raw = info.get("test_samples", [])
    if raw is None:
        return []
    if isinstance(raw, list):
        return [positive_int(item, "connector test_samples") for item in raw]
    return [positive_int(raw, "connector test_samples")]


def report_train_step(
    reporter: WorkerReporter,
    *,
    iteration: int,
    iterations: int,
    start_time: float,
    base_lr: float,
    train_loss: float,
    losses: dict[str, float],
) -> None:
    elapsed = time.monotonic() - start_time
    mean_step = elapsed / float(max(1, iteration))
    remain_time = max(0.0, (iterations - iteration) * mean_step)
    reporter.status(
        phase="train",
        iteration=iteration,
        iterations=iterations,
        test_active=0,
        elapsed_time_ms=elapsed * 1000.0,
        remain_time=remain_time,
    )
    reporter.metric("iteration", iteration, iteration=iteration)
    reporter.metric("train_loss", train_loss, iteration=iteration)
    reporter.metric("learning_rate", base_lr, iteration=iteration)
    for name, value in sorted(losses.items()):
        reporter.metric(str(name), value, iteration=iteration)
