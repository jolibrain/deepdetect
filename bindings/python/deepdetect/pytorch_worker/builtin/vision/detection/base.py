from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ....sdk import (
    Cancellation,
    DatasetContractError,
    DeepDetectWorkerBase,
    PredictionContractError,
    WorkerContext,
    WorkerReporter,
)
from ....tensors import parse_tensor_batch_ref

from .common import (
    DetectionEvalBox,
    DetectionListDataset,
    DetectionTensorBatchDataset,
    connector_prediction,
    debug as detection_debug,
    detection_map_metrics,
    detection_metric_thresholds,
    make_loader,
    move_target,
    parse_test_prediction_config,
    prediction_eval_boxes,
    prediction_sample,
    read_image_tensor,
    report_detection_metrics,
    sampled_indices,
    select_device,
    target_eval_boxes,
)
from .training import (
    DetectionCheckpointManager,
    DetectionDatasetSummary,
    DetectionProgressReporter,
    DetectionRepositoryContractWriter,
    DetectionTrainRequest,
    optional_positive_int,
    parameters_dict,
    positive_int,
    request_dict,
)


class DetectionTrainingWorkerBase(DeepDetectWorkerBase):
    worker_name = "detection-worker"
    task_name = "detection"
    debug_name = "detection-worker"

    def __init__(self) -> None:
        super().__init__()
        self.nclasses = 2
        self.device: Any = None
        self.model: Any = None
        self.multi_gpu_requested = False

    def configure(self, context: WorkerContext) -> dict[str, Any]:
        self.debug("configure: importing backend")
        backend = self.import_backend()
        torch = backend[0]
        self.debug("configure: backend imported")
        super().configure(context)
        self.nclasses = positive_int(context.mllib.get("nclasses", 2), "nclasses")
        if self.nclasses < 2:
            raise DatasetContractError(
                "detection nclasses must include background and at least one class"
            )
        self.device, self.multi_gpu_requested = select_device(torch, context.mllib)
        self.debug(f"configure: nclasses={self.nclasses} device={self.device}")
        result = {
            "worker": self.worker_name,
            "task": self.task_name,
            "nclasses": self.nclasses,
            "device": str(self.device),
        }
        result.update(self.backend_versions(*backend))
        return result

    def train(
        self,
        params: dict[str, Any],
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
    ) -> dict[str, Any]:
        self.debug("train: importing backend")
        backend = self.import_backend()
        torch = backend[0]
        train_request = DetectionTrainRequest.from_params(self.context, params)
        options = train_request.options
        self.debug(
            "train: options iterations=%s test_interval=%s batch_size=%s "
            "iter_size=%s base_lr=%s"
            % (
                options.iterations,
                options.test_interval,
                options.batch_size,
                options.iter_size,
                options.base_lr,
            )
        )

        progress = DetectionProgressReporter(reporter)
        if self.multi_gpu_requested:
            reporter.log(
                "warning",
                "multiple GPU ids were requested; detection worker uses "
                "the first id in this slice",
            )
        if train_request.source == "connector_pull":
            return self.train_connector_pull(
                train_request,
                reporter=reporter,
                cancellation=cancellation,
                torch=torch,
                backend=backend,
            )

        self.debug(f"train: loading {train_request.source} train dataset")
        train_dataset, test_datasets = self.create_training_datasets(
            train_request,
            torch=torch,
        )
        self.debug(f"train: train samples={len(train_dataset)}")
        self.debug(
            "train: test samples=%s" % [len(dataset) for dataset in test_datasets]
        )
        self.repository_contract_writer().write(
            train_dataset=train_dataset,
            test_datasets=test_datasets,
            source=train_request.source,
            request=train_request.request,
            request_params=train_request.request_params,
            effective_mllib=train_request.effective_mllib,
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            torch=torch,
        )
        train_batches = iter(train_loader)

        self.debug("train: creating model")
        self.model = self.create_model(self.nclasses, *backend).to(self.device)
        self.debug("train: model created")
        checkpoint_manager = DetectionCheckpointManager(self.context, torch, self.device)
        checkpoint_manager.load_model_for_training(
            self.model, train_request.effective_mllib
        )
        self.debug("train: checkpoint load checked")
        self.model.train()
        self.debug("train: creating optimizer")
        optimizer = self.create_optimizer(torch, self.model, base_lr=options.base_lr)
        checkpoint_manager.load_optimizer(optimizer, train_request.effective_mllib)
        optimizer.zero_grad(set_to_none=True)
        self.debug("train: entering training loop")

        start_time = time.monotonic()
        optimizer_steps = 0
        accumulated = 0
        latest_loss = 0.0
        while optimizer_steps < options.iterations:
            if cancellation.requested:
                progress.cancelled(
                    iteration=optimizer_steps,
                    iterations=options.iterations,
                )
                checkpoint_manager.save(self.model, optimizer, optimizer_steps)
                return {"status": "cancelled", "iteration": optimizer_steps}

            try:
                if optimizer_steps == 0 and accumulated == 0:
                    self.debug("train: loading first batch")
                images, targets, _meta = next(train_batches)
            except StopIteration:
                train_batches = iter(train_loader)
                images, targets, _meta = next(train_batches)
            images = [image.to(self.device) for image in images]
            targets = [move_target(target, self.device) for target in targets]
            if optimizer_steps == 0 and accumulated == 0:
                self.debug("train: running first forward/backward")
            loss_dict = self.training_losses(self.model, images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            (total_loss / float(options.iter_size)).backward()
            accumulated += 1
            latest_loss = float(total_loss.detach().cpu().item())

            if accumulated < options.iter_size:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            optimizer_steps += 1
            loss_values = {
                name: float(value.detach().cpu().item())
                for name, value in loss_dict.items()
            }
            progress.train_step(
                iteration=optimizer_steps,
                iterations=options.iterations,
                start_time=start_time,
                base_lr=options.base_lr,
                train_loss=latest_loss,
                losses=loss_values,
            )

            should_test = (
                test_datasets
                and (
                    optimizer_steps % options.test_interval == 0
                    or optimizer_steps == options.iterations
                )
            )
            if should_test:
                self.debug(f"train: evaluating at iteration {optimizer_steps}")
                self.evaluate(
                    test_datasets,
                    reporter=reporter,
                    iteration=optimizer_steps,
                    request_params=train_request.request_params,
                    torch=torch,
                    cancellation=cancellation,
                )
                self.model.train()
                checkpoint_manager.save(self.model, optimizer, optimizer_steps)
                self.debug(f"train: checkpoint saved at iteration {optimizer_steps}")

        checkpoint_manager.save(self.model, optimizer, options.iterations)
        self.debug("train: finished")
        progress.finished(iteration=options.iterations, iterations=options.iterations)
        return {
            "status": "finished",
            "iteration": options.iterations,
            "train_loss": latest_loss,
        }

    def train_connector_pull(
        self,
        train_request: DetectionTrainRequest,
        *,
        reporter: WorkerReporter,
        cancellation: Cancellation,
        torch: Any,
        backend: tuple[Any, ...],
    ) -> dict[str, Any]:
        if self.context is None or self.context.connector is None:
            raise DatasetContractError(
                "connector_tensor_pull requires a worker connector"
            )
        connector = self.context.connector
        options = train_request.options
        progress = DetectionProgressReporter(reporter)

        self.debug("train: requesting connector dataset info")
        dataset_info = connector.dataset_info()
        train_samples = positive_int(
            dataset_info.get("train_samples", 0),
            "connector train_samples",
        )
        test_samples = connector_test_samples(dataset_info)
        train_dataset = DetectionDatasetSummary(train_samples)
        test_datasets = [DetectionDatasetSummary(count) for count in test_samples]
        self.debug(f"train: connector train samples={train_samples}")
        self.debug(f"train: connector test samples={test_samples}")
        self.repository_contract_writer().write(
            train_dataset=train_dataset,
            test_datasets=test_datasets,
            source=train_request.source,
            request=train_request.request,
            request_params=train_request.request_params,
            effective_mllib=train_request.effective_mllib,
            connector_info=dataset_info,
        )

        self.debug("train: creating model")
        self.model = self.create_model(self.nclasses, *backend).to(self.device)
        self.debug("train: model created")
        checkpoint_manager = DetectionCheckpointManager(self.context, torch, self.device)
        checkpoint_manager.load_model_for_training(
            self.model, train_request.effective_mllib
        )
        self.debug("train: checkpoint load checked")
        self.model.train()
        self.debug("train: creating optimizer")
        optimizer = self.create_optimizer(torch, self.model, base_lr=options.base_lr)
        checkpoint_manager.load_optimizer(optimizer, train_request.effective_mllib)
        optimizer.zero_grad(set_to_none=True)
        self.debug("train: entering connector pull training loop")

        start_time = time.monotonic()
        optimizer_steps = 0
        accumulated = 0
        latest_loss = 0.0
        prefetcher = self.connector_batch_prefetcher(
            split="train",
            batch_size=options.batch_size,
            connector=connector,
            torch=torch,
            reset_epoch=True,
            prefetch_batches=connector_prefetch_batches(
                train_request.effective_mllib
            ),
        )
        while optimizer_steps < options.iterations:
            if cancellation.requested:
                prefetcher.close()
                progress.cancelled(
                    iteration=optimizer_steps,
                    iterations=options.iterations,
                )
                checkpoint_manager.save(self.model, optimizer, optimizer_steps)
                return {"status": "cancelled", "iteration": optimizer_steps}

            batch = prefetcher.next()
            if batch is None:
                prefetcher.close()
                prefetcher = self.connector_batch_prefetcher(
                    split="train",
                    batch_size=options.batch_size,
                    connector=connector,
                    torch=torch,
                    reset_epoch=True,
                    prefetch_batches=connector_prefetch_batches(
                        train_request.effective_mllib
                    ),
                )
                continue
            images, targets, _metas = batch
            images = [image.to(self.device) for image in images]
            targets = [move_target(target, self.device) for target in targets]
            if optimizer_steps == 0 and accumulated == 0:
                self.debug("train: running first connector forward/backward")
            loss_dict = self.training_losses(self.model, images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            (total_loss / float(options.iter_size)).backward()
            accumulated += 1
            latest_loss = float(total_loss.detach().cpu().item())

            if accumulated < options.iter_size:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            optimizer_steps += 1
            loss_values = {
                name: float(value.detach().cpu().item())
                for name, value in loss_dict.items()
            }
            progress.train_step(
                iteration=optimizer_steps,
                iterations=options.iterations,
                start_time=start_time,
                base_lr=options.base_lr,
                train_loss=latest_loss,
                losses=loss_values,
            )

            should_test = (
                bool(test_samples)
                and (
                    optimizer_steps % options.test_interval == 0
                    or optimizer_steps == options.iterations
                )
            )
            if should_test:
                self.debug(f"train: evaluating connector at iteration {optimizer_steps}")
                self.evaluate_connector_pull(
                    test_samples,
                    connector=connector,
                    reporter=reporter,
                    iteration=optimizer_steps,
                    request_params=train_request.request_params,
                    torch=torch,
                    cancellation=cancellation,
                    batch_size=options.batch_size,
                    prefetch_batches=connector_prefetch_batches(
                        train_request.effective_mllib
                    ),
                )
                self.model.train()
                checkpoint_manager.save(self.model, optimizer, optimizer_steps)
                self.debug(f"train: checkpoint saved at iteration {optimizer_steps}")

        checkpoint_manager.save(self.model, optimizer, options.iterations)
        prefetcher.close()
        self.debug("train: finished")
        progress.finished(iteration=options.iterations, iterations=options.iterations)
        return {
            "status": "finished",
            "iteration": options.iterations,
            "train_loss": latest_loss,
        }

    def evaluate_connector_pull(
        self,
        test_samples: list[int],
        *,
        connector: Any,
        reporter: WorkerReporter,
        iteration: int,
        request_params: dict[str, Any],
        torch: Any,
        cancellation: Cancellation,
        batch_size: int,
        prefetch_batches: int,
    ) -> None:
        if self.model is None:
            raise PredictionContractError("model is not initialized")
        output = request_params.get("output", {})
        output = output if isinstance(output, dict) else {}
        test_prediction_config = parse_test_prediction_config(output)
        metric_thresholds = detection_metric_thresholds(output)
        predictions_payload: dict[str, Any] = {}
        progress = DetectionProgressReporter(reporter)

        self.model.eval()
        with torch.no_grad():
            for test_index, total_samples in enumerate(test_samples):
                self.debug(
                    "evaluate: connector test%s samples=%s"
                    % (test_index, total_samples)
                )
                sample_indices = sampled_indices(
                    total_samples,
                    count=test_prediction_config["sample_count"],
                    seed=test_prediction_config["sample_seed"] + iteration + test_index,
                )
                samples_wanted = set(sample_indices)
                samples: list[dict[str, Any]] = []
                eval_predictions: list[DetectionEvalBox] = []
                eval_targets: list[DetectionEvalBox] = []
                processed = 0
                prefetcher = self.connector_batch_prefetcher(
                    split="test",
                    batch_size=batch_size,
                    connector=connector,
                    torch=torch,
                    test_index=test_index,
                    reset_epoch=True,
                    prefetch_batches=prefetch_batches,
                )
                progress.test_progress(
                    iteration=iteration,
                    test_index=test_index,
                    test_sets_total=len(test_samples),
                    processed=0,
                    total=total_samples,
                )
                try:
                    while not cancellation.requested:
                        batch = prefetcher.next()
                        if batch is None:
                            break
                        images, targets, metas = batch
                        images = [image.to(self.device) for image in images]
                        outputs = self.predict_batch(self.model, images)
                        for meta, target, output_item in zip(metas, targets, outputs):
                            processed += 1
                            eval_targets.extend(target_eval_boxes(meta, target))
                            eval_predictions.extend(
                                prediction_eval_boxes(meta, output_item)
                            )
                            if int(meta["index"]) in samples_wanted:
                                samples.append(
                                    prediction_sample(
                                        meta,
                                        output_item,
                                        confidence_threshold=test_prediction_config[
                                            "confidence_threshold"
                                        ],
                                        best_bbox=test_prediction_config["best_bbox"],
                                    )
                                )
                        progress.test_progress(
                            iteration=iteration,
                            test_index=test_index,
                            test_sets_total=len(test_samples),
                            processed=processed,
                            total=total_samples,
                        )
                finally:
                    prefetcher.close()
                predictions_payload[f"test{test_index}"] = {
                    "iteration": iteration,
                    "samples": sorted(samples, key=lambda item: int(item["index"])),
                }
                metrics = detection_map_metrics(
                    eval_predictions,
                    eval_targets,
                    metric_thresholds,
                )
                if not eval_targets:
                    reporter.log(
                        "warning",
                        "test set has no positive detection boxes; "
                        "reporting zero detection mAP",
                        iteration=iteration,
                        test_set_index=test_index,
                    )
                report_detection_metrics(
                    reporter,
                    metrics,
                    iteration=iteration,
                    test_index=test_index,
                )
                self.debug(f"evaluate: connector test{test_index} metrics reported")

        progress.test_finished(
            iteration=iteration,
            test_sets_total=len(test_samples),
            predictions_payload=predictions_payload,
        )

    def pull_detection_batch(
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
            dataset = DetectionTensorBatchDataset(
                [tensor_batch],
                nclasses=self.nclasses,
                torch=torch,
            )
            loader = make_loader(
                dataset,
                batch_size=len(dataset),
                shuffle=False,
                torch=torch,
            )
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
            lambda *, reset_epoch: self.pull_detection_batch(
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

    def evaluate(
        self,
        test_datasets: list[DetectionListDataset],
        *,
        reporter: WorkerReporter,
        iteration: int,
        request_params: dict[str, Any],
        torch: Any,
        cancellation: Cancellation,
    ) -> None:
        if self.model is None:
            raise PredictionContractError("model is not initialized")
        self.debug(
            "evaluate: iteration=%s test_sets=%s" % (iteration, len(test_datasets))
        )
        output = request_params.get("output", {})
        output = output if isinstance(output, dict) else {}
        test_prediction_config = parse_test_prediction_config(output)
        metric_thresholds = detection_metric_thresholds(output)
        predictions_payload: dict[str, Any] = {}
        progress = DetectionProgressReporter(reporter)

        self.model.eval()
        with torch.no_grad():
            for test_index, dataset in enumerate(test_datasets):
                self.debug("evaluate: test%s samples=%s" % (test_index, len(dataset)))
                sample_indices = sampled_indices(
                    len(dataset),
                    count=test_prediction_config["sample_count"],
                    seed=test_prediction_config["sample_seed"] + iteration + test_index,
                )
                samples_wanted = set(sample_indices)
                samples: list[dict[str, Any]] = []
                eval_predictions: list[DetectionEvalBox] = []
                eval_targets: list[DetectionEvalBox] = []
                processed = 0
                progress.test_progress(
                    iteration=iteration,
                    test_index=test_index,
                    test_sets_total=len(test_datasets),
                    processed=0,
                    total=len(dataset),
                )
                loader = make_loader(dataset, batch_size=1, shuffle=False, torch=torch)
                for images, targets, metas in loader:
                    if cancellation.requested:
                        break
                    images = [image.to(self.device) for image in images]
                    outputs = self.predict_batch(self.model, images)
                    for meta, target, output in zip(metas, targets, outputs):
                        processed += 1
                        eval_targets.extend(target_eval_boxes(meta, target))
                        eval_predictions.extend(prediction_eval_boxes(meta, output))
                        if int(meta["index"]) in samples_wanted:
                            samples.append(
                                prediction_sample(
                                    meta,
                                    output,
                                    confidence_threshold=test_prediction_config[
                                        "confidence_threshold"
                                    ],
                                    best_bbox=test_prediction_config["best_bbox"],
                                )
                            )
                    progress.test_progress(
                        iteration=iteration,
                        test_index=test_index,
                        test_sets_total=len(test_datasets),
                        processed=processed,
                        total=len(dataset),
                    )
                predictions_payload[f"test{test_index}"] = {
                    "iteration": iteration,
                    "samples": sorted(samples, key=lambda item: int(item["index"])),
                }
                metrics = detection_map_metrics(
                    eval_predictions,
                    eval_targets,
                    metric_thresholds,
                )
                if not eval_targets:
                    reporter.log(
                        "warning",
                        "test set has no positive detection boxes; "
                        "reporting zero detection mAP",
                        iteration=iteration,
                        test_set_index=test_index,
                    )
                report_detection_metrics(
                    reporter,
                    metrics,
                    iteration=iteration,
                    test_index=test_index,
                )
                self.debug(f"evaluate: test{test_index} metrics reported")

        progress.test_finished(
            iteration=iteration,
            test_sets_total=len(test_datasets),
            predictions_payload=predictions_payload,
        )

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        backend = self.import_backend()
        torch = backend[0]
        request = request_dict(params)
        request_params = parameters_dict(request)
        output_params = request_params.get("output", {})
        output_params = output_params if isinstance(output_params, dict) else {}
        threshold = float(output_params.get("confidence_threshold", 0.0))
        best_bbox = optional_positive_int(output_params.get("best_bbox"), "best_bbox")
        data = request.get("data", [])
        if not isinstance(data, list):
            raise PredictionContractError("predict data must be a list")
        if self.model is None:
            self.model = self.create_model(self.nclasses, *backend).to(self.device)
            checkpoint_manager = DetectionCheckpointManager(
                self.context, torch, self.device
            )
            loaded = checkpoint_manager.load_model_for_prediction(self.model)
            if loaded is None:
                self.model.eval()
        self.model.eval()
        results = []
        with torch.no_grad():
            for image_path in data:
                image_path = Path(str(image_path)).expanduser().resolve()
                image, _size = read_image_tensor(image_path, torch)
                output = self.predict_batch(self.model, [image.to(self.device)])[0]
                results.append(
                    connector_prediction(
                        image_path,
                        output,
                        confidence_threshold=threshold,
                        best_bbox=best_bbox,
                    )
                )
        return {"results": results}

    def create_dataset(self, list_path: Path, *, torch: Any) -> DetectionListDataset:
        return DetectionListDataset(list_path, nclasses=self.nclasses, torch=torch)

    def create_training_datasets(
        self,
        train_request: DetectionTrainRequest,
        *,
        torch: Any,
    ) -> tuple[Any, list[Any]]:
        if train_request.source == "path":
            if train_request.train_list is None:
                raise DatasetContractError("path-backed train request has no train list")
            return (
                self.create_dataset(train_request.train_list, torch=torch),
                [
                    self.create_dataset(path, torch=torch)
                    for path in train_request.test_lists
                ],
            )
        if train_request.source == "tensor":
            return (
                DetectionTensorBatchDataset(
                    train_request.train_tensor_batches,
                    nclasses=self.nclasses,
                    torch=torch,
                ),
                [
                    DetectionTensorBatchDataset(
                        batches,
                        nclasses=self.nclasses,
                        torch=torch,
                    )
                    for batches in train_request.test_tensor_batches
                ],
            )
        raise DatasetContractError(
            f"unsupported train request source: {train_request.source}"
        )

    def repository_contract_writer(self) -> DetectionRepositoryContractWriter:
        return DetectionRepositoryContractWriter(
            self.context,
            worker_name=self.worker_name,
            task_name=self.task_name,
            nclasses=self.nclasses,
        )

    def create_optimizer(self, torch: Any, model: Any, *, base_lr: float) -> Any:
        return torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr,
        )

    def training_losses(
        self,
        model: Any,
        images: list[Any],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return model(images, targets)

    def predict_batch(self, model: Any, images: list[Any]) -> list[dict[str, Any]]:
        return model(images)

    def backend_versions(self, *backend: Any) -> dict[str, Any]:
        return {"torch_version": str(getattr(backend[0], "__version__", "unknown"))}

    def debug(self, message: str) -> None:
        detection_debug(message, tag=self.debug_name)

    def import_backend(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def create_model(self, nclasses: int, *backend: Any) -> Any:
        raise NotImplementedError


def connector_test_samples(dataset_info: dict[str, Any]) -> list[int]:
    values = dataset_info.get("test_samples", [])
    if values is None:
        return []
    if not isinstance(values, list):
        raise DatasetContractError("connector test_samples must be a list")
    result = []
    for index, value in enumerate(values):
        count = int(value)
        if count < 0:
            raise DatasetContractError(
                f"connector test_samples[{index}] must be non-negative"
            )
        result.append(count)
    return result


def connector_prefetch_batches(mllib: dict[str, Any]) -> int:
    value = mllib.get("connector_prefetch_batches", 2)
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise DatasetContractError(
            "mllib.connector_prefetch_batches must be an integer"
        ) from error
    if count <= 0:
        raise DatasetContractError("mllib.connector_prefetch_batches must be positive")
    return count


class ConnectorBatchPrefetcher:
    def __init__(
        self,
        fetch: Callable[
            [bool],
            tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]] | None,
        ],
        *,
        reset_epoch: bool,
        prefetch_batches: int,
    ) -> None:
        self._fetch = fetch
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, prefetch_batches))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(reset_epoch,),
            daemon=True,
        )
        self._thread.start()

    def next(
        self,
    ) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
        start = time.monotonic()
        item = self._queue.get()
        elapsed = (time.monotonic() - start) * 1000.0
        detection_debug(
            "prefetch_queue_wait elapsed_ms=%.3f item_type=%s"
            % (elapsed, type(item).__name__),
            tag="connector",
        )
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()

    def _run(self, reset_epoch: bool) -> None:
        try:
            current_reset = reset_epoch
            while not self._stop.is_set():
                batch = self._fetch(reset_epoch=current_reset)
                current_reset = False
                self._put(batch)
                if batch is None:
                    return
        except BaseException as error:
            self._put(error)

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
