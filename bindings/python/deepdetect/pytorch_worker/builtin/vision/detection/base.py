from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ....sdk import (
    Cancellation,
    DatasetContractError,
    DeepDetectWorkerBase,
    PredictionContractError,
    WorkerContext,
    WorkerReporter,
)

from .common import (
    DetectionEvalBox,
    DetectionListDataset,
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

        self.debug(f"train: loading train dataset {train_request.train_list}")
        train_dataset = self.create_dataset(train_request.train_list, torch=torch)
        self.debug(f"train: train samples={len(train_dataset)}")
        self.debug(f"train: loading {len(train_request.test_lists)} test datasets")
        test_datasets = [
            self.create_dataset(path, torch=torch) for path in train_request.test_lists
        ]
        self.debug(
            "train: test samples=%s" % [len(dataset) for dataset in test_datasets]
        )
        self.repository_contract_writer().write(
            train_dataset=train_dataset,
            test_datasets=test_datasets,
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
