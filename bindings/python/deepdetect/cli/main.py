from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepdetect
import numpy as np
from PIL import Image

from .checks import run_training_checks
from .config import cli_options, deep_merge, load_config, parse_overrides
from .events import EventWriter, MetricEventExtractor, metric_events
from .profiles import PROFILES, get_profile
from .runs import (
    create_run,
    load_run,
    repository_name,
    summarize_timings,
)
from .sinks import CompositeMetricSink, JSONLMetricSink, VisdomMetricSink
from .terminal import LiveTrainingTerminalReporter
from .utils import (
    chunks,
    configure_gpu_compatibility,
    report_error,
    stage_model,
    validate_resume_repository,
)
from .visualize import (
    detection_overlay_image,
    output_path_for,
    render_detections,
    render_segmentation,
    segmentation_overlay_images,
)

VISDOM_RESULT_MAX_SIDE = 512


def _path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _parse_gpu_ids(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    ids: list[int] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.append(int(item))
            except ValueError as error:
                raise ValueError(f"GPU ids must be integers, got: {item}") from error
    if not ids:
        raise ValueError("--gpuid expects at least one GPU id")
    if -1 in ids and ids != [-1]:
        raise ValueError("--gpuid -1 cannot be combined with other ids")
    return ids


def _normalize_gpu_options(options: dict[str, Any], args: argparse.Namespace) -> None:
    gpuid = options.get("gpuid")
    if gpuid is None:
        return
    if args.gpu is False:
        raise ValueError("--gpuid requires GPU execution; remove --no-gpu")
    options["gpuid"] = _normalize_gpu_ids(gpuid)
    options["gpu"] = True


def _normalize_gpu_ids(value: Any) -> int | list[int]:
    if isinstance(value, bool):
        raise ValueError("gpuid must be an integer or a list of integers")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        ids = _parse_gpu_ids([value])
    elif isinstance(value, (list, tuple)):
        ids = []
        for item in value:
            parsed = _parse_gpu_ids([str(item)])
            if parsed is not None:
                ids.extend(parsed)
    else:
        raise ValueError("gpuid must be an integer or a list of integers")
    if ids is None:
        raise ValueError("gpuid must not be empty")
    if -1 in ids and ids != [-1]:
        raise ValueError("gpuid -1 cannot be combined with other ids")
    if len(ids) == 1:
        return ids[0]
    return ids


def _resolve_options(
    defaults: dict[str, Any],
    args: argparse.Namespace,
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    config = load_config(args.config)
    overrides = parse_overrides(args.overrides)
    options = deep_merge(defaults, config, cli_values, overrides)
    for key in ("weights", "repository", "job_dir", "output", "train_data"):
        if key in options and options[key] is not None:
            options[key] = Path(options[key])
    if "test_data" in options and options["test_data"] is not None:
        options["test_data"] = _normalize_test_data_paths(options["test_data"])
    return options


def _normalize_test_data_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, (list, tuple)):
        paths = [Path(item) for item in value]
        if not paths:
            raise ValueError("test-data requires at least one path")
        return paths
    raise ValueError("test-data must be a path or a list of paths")


def run_train(args: argparse.Namespace) -> int:
    profile = get_profile(args.model)
    cli_values = cli_options(
        train_data=args.train_data,
        test_data=args.test_data,
        weights=args.weights,
        repository=args.repository,
        service_name=args.service_name,
        nclasses=args.nclasses,
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        test_interval=args.test_interval,
        gpu=args.gpu,
        gpuid=_parse_gpu_ids(args.gpuid),
        sync=args.sync,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        job_dir=args.job_dir,
        run_name=args.run_name,
        resume=args.resume,
        output_format=args.output_format,
        terminal=args.terminal,
        dataset_check=args.dataset_check,
        skip_mask_validation=args.skip_mask_validation,
        visdom=args.visdom,
        visdom_server=args.visdom_server,
        visdom_port=args.visdom_port,
        visdom_base_url=args.visdom_base_url,
        visdom_offline_ok=args.visdom_offline_ok,
        visdom_save=args.visdom_save,
        visdom_results=args.visdom_results,
        visdom_results_count=args.visdom_results_count,
        visdom_results_seed=args.visdom_results_seed,
    )
    options = _resolve_options(profile.train_defaults(), args, cli_values)
    _normalize_gpu_options(options, args)
    for required in ("train_data", "test_data", "repository"):
        if not options.get(required):
            raise ValueError(f"{required.replace('_', '-')} is required")
    resume = options.get("resume")
    if resume is not None and resume not in {"latest", "best"}:
        raise ValueError("resume must be one of: latest, best")
    if options.get("terminal") not in {"verbose", "live"}:
        raise ValueError("terminal must be one of: verbose, live")
    if resume is None and not options.get("weights"):
        raise ValueError("weights is required unless --resume is used")
    if options.get("job_dir") is None:
        options["job_dir"] = Path(options["repository"])
    for numeric in (
        "width",
        "height",
        "iterations",
        "batch_size",
        "test_interval",
        "poll_interval",
    ):
        _validate_positive(numeric, options[numeric])
    if int(options["nclasses"]) <= 0:
        raise ValueError("nclasses must be positive")
    _validate_positive("visdom_port", int(options["visdom_port"]))
    if int(options["visdom_results_count"]) < 0:
        raise ValueError("visdom_results_count must be non-negative")
    run_name = str(options.get("run_name") or repository_name(Path(options["repository"])))
    options["run_name"] = run_name
    if resume:
        validate_resume_repository(Path(options["repository"]), str(resume))

    writer = _create_training_terminal_reporter(options)
    manifest = _create_or_resume_run_manifest(
        args.model,
        options,
        run_name,
        job_dir_is_run_dir=_path_is_repository(options["job_dir"], options["repository"]),
    )
    writer.emit(
        "run_started",
        run_id=manifest.data["run_id"],
        run_name=manifest.data["run_name"],
        run_dir=str(manifest.run_dir),
        resume=resume,
    )
    metric_sink, visdom_sink = _create_training_metric_sink(
        options,
        writer=writer,
        run_id=manifest.data["run_id"],
        run_dir=manifest.run_dir,
    )
    extractor = MetricEventExtractor()

    try:
        if resume:
            _replay_resume_history(
                Path(options["repository"]),
                writer=writer,
                run_id=manifest.data["run_id"],
                extractor=extractor,
                visdom_sink=visdom_sink,
            )
        for check in run_training_checks(args.model, options):
            writer.emit("dataset_check", run_id=manifest.data["run_id"], **check)

        if not resume:
            stage_model(options["weights"], options["repository"])
        dd = deepdetect.DeepDetect()
        configure_gpu_compatibility(dd.build_info, requested=bool(options["gpu"]))
        service_parameters = profile.service_parameters(options)
        train_parameters = profile.train_parameters(options)
        result_visualizer = _create_training_result_visualizer(
            profile,
            options,
            visdom_sink=visdom_sink,
            writer=writer,
            run_id=manifest.data["run_id"],
        )
        if result_visualizer is not None:
            train_parameters["output_parameters"].update(
                result_visualizer.train_output_parameters()
            )
        if _training_live_terminal_enabled(options):
            dd.set_log_level("warn")
        with dd.create_service(options["service_name"], **service_parameters) as service:
            if _training_live_terminal_enabled(options):
                dd.set_service_log_level(options["service_name"], "warn")
            result = service.train(
                [
                    Path(options["train_data"]).resolve(),
                    *[Path(path).resolve() for path in options["test_data"]],
                ],
                asynchronous=not bool(options["sync"]),
                **train_parameters,
            )
            if isinstance(result, deepdetect.TrainingJob):
                final_status = _monitor_training(
                    result,
                    writer=writer,
                    manifest=manifest,
                    metric_sink=metric_sink,
                    result_visualizer=result_visualizer,
                    extractor=extractor,
                    timeout=options["timeout"],
                    poll_interval=float(options["poll_interval"]),
                )
            else:
                final_status = {"status": "finished", **result}
                writer.emit(
                    "training_status",
                    run_id=manifest.data["run_id"],
                    status="finished",
                    measure=result.get("measure", {}),
                )
                events = _write_metric_events(
                    {"status": "finished", **result},
                    writer=writer,
                    manifest=manifest,
                    metric_sink=metric_sink,
                )
                if result_visualizer is not None:
                    result_visualizer.maybe_write({"status": "finished", **result}, events)
        writer.emit(
            "run_finished",
            run_id=manifest.data["run_id"],
            status=final_status.get("status", "finished"),
        )
        manifest.update(status=final_status.get("status", "finished"), last_status=final_status)
        return 0
    finally:
        metric_sink.close()
        writer.close()


def _create_training_terminal_reporter(options: dict[str, Any]):
    if _training_live_terminal_enabled(options):
        return LiveTrainingTerminalReporter(
            total_iterations=int(options["iterations"]),
            gpu_ids=_live_terminal_gpu_ids(options),
        )
    if str(options.get("terminal", "verbose")) == "live":
        return EventWriter(output_format="jsonl")
    return EventWriter(output_format=options["output_format"])


def _training_live_terminal_enabled(options: dict[str, Any]) -> bool:
    return str(options.get("terminal", "verbose")) == "live" and sys.stdout.isatty()


def _live_terminal_gpu_ids(options: dict[str, Any]) -> int | list[int] | None:
    if not bool(options.get("gpu")):
        return None
    gpuid = options.get("gpuid")
    return 0 if gpuid is None else gpuid


def _create_training_metric_sink(
    options: dict[str, Any],
    *,
    writer: EventWriter,
    run_id: str,
    run_dir: Path,
) -> tuple[CompositeMetricSink, VisdomMetricSink | None]:
    def warn(sink: str, error: BaseException) -> None:
        writer.emit(
            "sink_warning",
            run_id=run_id,
            sink=sink,
            message=str(error),
        )

    sinks = [JSONLMetricSink(run_dir / "metrics.jsonl")]
    visdom_sink = None
    if options.get("visdom"):
        try:
            visdom_sink = VisdomMetricSink(
                env=str(options["run_name"]),
                server=str(options["visdom_server"]),
                port=int(options["visdom_port"]),
                base_url=str(options["visdom_base_url"]),
                save=bool(options["visdom_save"]),
                warning_callback=warn,
            )
            sinks.append(visdom_sink)
        except Exception as error:
            if not bool(options["visdom_offline_ok"]):
                raise RuntimeError(str(error)) from error
            warn("VisdomMetricSink", error)
    return (
        CompositeMetricSink(
            sinks,
            warning_callback=warn,
            disable_failed=bool(options["visdom_offline_ok"]),
        ),
        visdom_sink,
    )


class TrainingResultVisualizer:
    def __init__(
        self,
        *,
        model: str,
        task: str,
        samples: list[list["TestResultSample"]],
        image_size: tuple[int, int],
        confidence_threshold: float,
        best_bbox: int | None,
        artifact_dir: Path,
        visdom_sink: VisdomMetricSink,
        warning_callback: Any,
        disable_failed: bool,
    ) -> None:
        self.model = model
        self.task = task
        self.samples = samples
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.best_bbox = best_bbox
        self.artifact_dir = artifact_dir
        self.visdom_sink = visdom_sink
        self.warning_callback = warning_callback
        self.disable_failed = disable_failed
        self._last_iteration: float | None = None
        self._disabled = False
        self._warned_missing_payload = False

    def train_output_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "test_predictions": {
                "enabled": True,
                "confidence_threshold": 0.1,
                "indices": {
                    f"test{test_index}": [sample.index for sample in samples]
                    for test_index, samples in enumerate(self.samples)
                    if samples
                },
            }
        }
        if self.best_bbox is not None:
            parameters["test_predictions"]["best_bbox"] = self.best_bbox
        return parameters

    def maybe_write(
        self,
        status: dict[str, Any],
        metric_events: list[dict[str, Any]],
    ) -> None:
        if self._disabled:
            return
        iteration = self._result_iteration(status, metric_events)
        if iteration is None or iteration == self._last_iteration:
            return
        try:
            if not self._write(status, iteration):
                if not self._warned_missing_payload:
                    self.warning_callback(
                        "VisdomResultSink",
                        RuntimeError(
                            "training status has no test_predictions payload; "
                            "skipping live result images"
                        ),
                    )
                    self._warned_missing_payload = True
                return
        except Exception as error:
            if not self.disable_failed:
                raise
            self.warning_callback("VisdomResultSink", error)
        else:
            self._last_iteration = iteration

    def _result_iteration(
        self,
        status: dict[str, Any],
        metric_events: list[dict[str, Any]],
    ) -> float | None:
        test_predictions = status.get("test_predictions")
        if isinstance(test_predictions, dict):
            iterations = []
            for test_index, samples in enumerate(self.samples):
                if not samples:
                    continue
                test_payload = test_predictions.get(f"test{test_index}")
                if not isinstance(test_payload, dict):
                    return None
                payload_samples = test_payload.get("samples")
                if not isinstance(payload_samples, list):
                    return None
                try:
                    iterations.append(float(test_payload["iteration"]))
                except (KeyError, TypeError, ValueError):
                    return None
            if iterations and len(set(iterations)) == 1:
                return iterations[0]
        return _result_visualization_iteration(metric_events)

    def _write(self, status: dict[str, Any], iteration: float) -> bool:
        test_predictions = status.get("test_predictions")
        if not isinstance(test_predictions, dict):
            return False
        wrote = False
        for test_index, samples in enumerate(self.samples):
            if not samples:
                continue
            test_payload = test_predictions.get(f"test{test_index}")
            if not isinstance(test_payload, dict):
                continue
            predictions = test_payload.get("samples", [])
            if not isinstance(predictions, list):
                continue
            sample_by_index = {sample.index: sample.path for sample in samples}
            images = []
            rendered: list[tuple[int, Path, dict[str, Any], np.ndarray]] = []
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    continue
                try:
                    sample_index = int(prediction["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                image_path = sample_by_index.get(sample_index)
                if image_path is None:
                    continue
                image = _result_image_array(
                    self.model,
                    image_path,
                    prediction,
                    image_size=self.image_size,
                )
                images.append(image)
                rendered.append((sample_index, image_path, prediction, image))
            if not images:
                continue
            self._write_artifacts(
                test_index=test_index,
                iteration=iteration,
                rendered=rendered,
            )
            self.visdom_sink.write_images(
                window=f"results-{self.task}-test{test_index}",
                title=f"{self.task} test{test_index} results iteration {iteration:g}",
                images=_visdom_result_image_arrays(images),
            )
            wrote = True
        return wrote

    def _write_artifacts(
        self,
        *,
        test_index: int,
        iteration: float,
        rendered: list[tuple[int, Path, dict[str, Any], np.ndarray]],
    ) -> None:
        iteration_dir = (
            self.artifact_dir
            / f"iteration-{_artifact_iteration_name(iteration)}"
            / f"test{test_index}"
        )
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, image_path, prediction, image in rendered:
            stem = f"sample-{sample_index:06d}"
            image_out = iteration_dir / f"{stem}.png"
            prediction_out = iteration_dir / f"{stem}.json"
            _save_chw_image(image, image_out)
            prediction_out.write_text(
                json.dumps(
                    {
                        "image": str(image_path),
                        "sample_index": sample_index,
                        "prediction": prediction,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _create_training_result_visualizer(
    profile: Any,
    options: dict[str, Any],
    *,
    visdom_sink: VisdomMetricSink | None,
    writer: EventWriter,
    run_id: str,
) -> TrainingResultVisualizer | None:
    def warn(sink: str, error: BaseException) -> None:
        writer.emit(
            "sink_warning",
            run_id=run_id,
            sink=sink,
            message=str(error),
        )

    if visdom_sink is None:
        return None
    if not bool(options.get("visdom_results", True)):
        return None
    count = int(options.get("visdom_results_count", 10))
    if count <= 0:
        return None

    try:
        samples = _sample_test_images(
            options["test_data"],
            count=count,
            seed=int(options.get("visdom_results_seed", 12345)),
        )
    except Exception as error:
        if not bool(options["visdom_offline_ok"]):
            raise
        warn("VisdomResultSink", error)
        return None

    if not any(samples):
        return None
    return TrainingResultVisualizer(
        model=profile.name,
        task=profile.task,
        samples=samples,
        image_size=(int(options["width"]), int(options["height"])),
        confidence_threshold=float(options.get("confidence_threshold", 0.0)),
        best_bbox=(
            int(options["best_bbox"])
            if options.get("best_bbox") is not None
            else None
        ),
        artifact_dir=Path(options["repository"]).expanduser().resolve()
        / "visdom-results",
        visdom_sink=visdom_sink,
        warning_callback=warn,
        disable_failed=bool(options["visdom_offline_ok"]),
    )


@dataclass(frozen=True)
class TestResultSample:
    index: int
    path: Path


def _sample_test_images(
    test_data: list[Path],
    *,
    count: int,
    seed: int,
) -> list[list[TestResultSample]]:
    samples = []
    for index, list_path in enumerate(test_data):
        images = _dataset_image_paths(Path(list_path))
        indexed_images = [
            TestResultSample(sample_index, image)
            for sample_index, image in enumerate(images)
        ]
        if len(indexed_images) > count:
            indexed_images = random.Random(seed + index).sample(
                indexed_images, count
            )
        samples.append(sorted(indexed_images, key=lambda sample: sample.index))
    return samples


def _dataset_image_paths(list_path: Path) -> list[Path]:
    resolved = list_path.expanduser().resolve()
    base = resolved.parent
    images = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image = Path(line.split()[0]).expanduser()
        if not image.is_absolute():
            image = base / image
        images.append(image.resolve())
    return images


def _artifact_iteration_name(iteration: float) -> str:
    if float(iteration).is_integer():
        return f"{int(iteration):06d}"
    return str(iteration).replace(".", "_")


def _save_chw_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, got shape {array.shape}")
    Image.fromarray(array.transpose(1, 2, 0), mode="RGB").save(path)


def _visdom_result_image_arrays(images: list[np.ndarray]) -> list[np.ndarray]:
    resized = [
        _resize_chw_image(image, max_side=VISDOM_RESULT_MAX_SIDE)
        for image in images
    ]
    if not resized:
        return resized
    height = max(int(image.shape[1]) for image in resized)
    width = max(int(image.shape[2]) for image in resized)
    if all(image.shape[1] == height and image.shape[2] == width for image in resized):
        return resized
    padded = []
    for image in resized:
        canvas = np.full((3, height, width), 255, dtype=np.uint8)
        canvas[:, : image.shape[1], : image.shape[2]] = image
        padded.append(canvas)
    return padded


def _resize_chw_image(image: np.ndarray, *, max_side: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, got shape {array.shape}")
    height = int(array.shape[1])
    width = int(array.shape[2])
    largest = max(width, height)
    if largest <= max_side:
        return array
    scale = max_side / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(array.transpose(1, 2, 0), mode="RGB")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = pil_image.resize((new_width, new_height), resampling)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)


def _result_visualization_iteration(
    metric_events: list[dict[str, Any]],
) -> float | None:
    iterations = []
    for event in metric_events:
        if not _is_result_visualization_metric(event.get("name")):
            continue
        try:
            iterations.append(float(event["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(iterations) if iterations else None


def _is_result_visualization_metric(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    base, separator, suffix = name.rpartition("_test")
    if separator and suffix.isdigit() and base:
        name = base
    normalized = name.lower()
    return normalized == "map" or normalized.startswith("map-") or normalized in {
        "acc",
        "meaniou",
        "meanacc",
    }


def _result_image_array(
    model: str,
    image_path: Path,
    prediction: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    if model == "yolox":
        image = detection_overlay_image(
            image_path,
            prediction,
            coordinate_size=_prediction_image_size(prediction) or image_size,
        )
    else:
        _mask, image = segmentation_overlay_images(
            image_path,
            prediction,
            original_size=True,
        )
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array.transpose(2, 0, 1)


def _prediction_image_size(prediction: dict[str, Any]) -> tuple[int, int] | None:
    imgsize = prediction.get("imgsize")
    if not isinstance(imgsize, dict):
        return None
    try:
        width = int(imgsize["width"])
        height = int(imgsize["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _create_or_resume_run_manifest(
    model: str,
    options: dict[str, Any],
    run_name: str,
    *,
    job_dir_is_run_dir: bool = False,
) -> Any:
    if options.get("resume"):
        run_dir = Path(options["job_dir"]).expanduser().resolve()
        if not job_dir_is_run_dir:
            run_dir = run_dir / run_name
        manifest_path = run_dir / "run.json"
        if manifest_path.is_file():
            manifest = load_run(manifest_path)
            manifest.update(
                command="train",
                model=model,
                service_name=options["service_name"],
                options=copy.deepcopy(options),
                status="resuming",
                resumed_at=time.time(),
            )
            return manifest
        return create_run(
            options["job_dir"],
            command="train",
            model=model,
            service_name=options["service_name"],
            options=copy.deepcopy(options),
            run_name=run_name,
            exist_ok=run_dir.exists(),
            root_is_run_dir=job_dir_is_run_dir,
        )
    return create_run(
        options["job_dir"],
        command="train",
        model=model,
        service_name=options["service_name"],
        options=copy.deepcopy(options),
        run_name=run_name,
        root_is_run_dir=job_dir_is_run_dir,
    )


def _path_is_repository(job_dir: Path, repository: Path) -> bool:
    return job_dir.expanduser().resolve() == repository.expanduser().resolve()


def _replay_resume_history(
    repository: Path,
    *,
    writer: EventWriter,
    run_id: str,
    extractor: MetricEventExtractor,
    visdom_sink: VisdomMetricSink | None,
) -> None:
    status = _load_repository_metrics_status(repository)
    if status is None:
        return
    replay_extractor = MetricEventExtractor()
    events = replay_extractor.events(status)
    extractor.prime(events)
    writer.emit(
        "history_replay_started",
        run_id=run_id,
        source=str(repository.expanduser().resolve() / "metrics.json"),
        metrics=len(events),
        visdom=visdom_sink is not None,
    )
    visdom_metrics = 0
    if visdom_sink is not None:
        progress = _history_progress(total=len(events))
        try:
            visdom_metrics = visdom_sink.write_many(
                [{"event": "metric", "run_id": run_id, **event} for event in events],
                progress_callback=progress.update,
            )
        finally:
            progress.close()
    writer.emit(
        "history_replayed",
        run_id=run_id,
        source=str(repository.expanduser().resolve() / "metrics.json"),
        metrics=len(events),
        visdom_metrics=visdom_metrics,
        visdom_skipped_metrics=len(events) - visdom_metrics,
        visdom=visdom_sink is not None,
    )


def _load_repository_metrics_status(repository: Path) -> dict[str, Any] | None:
    path = repository.expanduser().resolve() / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    body = data.get("body") if isinstance(data, dict) else None
    if not isinstance(body, dict):
        body = data if isinstance(data, dict) else {}
    if not isinstance(body.get("measure_hist"), dict):
        return None
    return {
        "measure": body.get("measure", {}),
        "measure_hist": body["measure_hist"],
        "measure_sampling": body.get("measure_sampling", {}),
    }


def _history_progress(*, total: int):
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgress()
    return tqdm(
        total=total,
        desc="replaying Visdom history",
        unit="point",
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
        leave=True,
    )


class _NullProgress:
    def update(self, count: int) -> None:
        return None

    def close(self) -> None:
        return None


def _write_metric_events(
    status: dict[str, Any],
    *,
    writer: EventWriter,
    manifest: Any,
    metric_sink: CompositeMetricSink,
    extractor: MetricEventExtractor | None = None,
) -> list[dict[str, Any]]:
    metrics = extractor.events(status) if extractor is not None else metric_events(status)
    events = []
    for metric in metrics:
        event = writer.emit("metric", run_id=manifest.data["run_id"], **metric)
        metric_sink.write(event)
        events.append(event)
    return events


def _monitor_training(
    job: deepdetect.TrainingJob,
    *,
    writer: EventWriter,
    manifest: Any,
    metric_sink: CompositeMetricSink,
    result_visualizer: TrainingResultVisualizer | None = None,
    timeout: float | None = None,
    poll_interval: float = 0.5,
    extractor: MetricEventExtractor | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    extractor = extractor or MetricEventExtractor()
    manifest.update(job=job.job, status="running")
    while True:
        output_parameters: dict[str, Any] = {
            "measure_hist": True,
            "max_hist_points": 10000,
        }
        if result_visualizer is not None:
            output_parameters["test_predictions"] = True
        status = job.status(output_parameters=output_parameters)
        state = str(status.get("status", "")).lower()
        writer.emit(
            "training_status",
            run_id=manifest.data["run_id"],
            job=job.job,
            status=state,
            time=status.get("time"),
            measure=status.get("measure", {}),
            measures=status.get("measures"),
        )
        events = _write_metric_events(
            status,
            writer=writer,
            manifest=manifest,
            metric_sink=metric_sink,
            extractor=extractor,
        )
        if result_visualizer is not None:
            result_visualizer.maybe_write(status, events)
        manifest.update(status=state or "running", last_status=status)
        if state in deepdetect.TrainingJob._TERMINAL:
            if state != "finished":
                raise RuntimeError(f"training ended with status {state!r}")
            return status
        if timeout is not None and time.monotonic() - started >= timeout:
            job.cancel()
            manifest.update(status="cancelled")
            raise TimeoutError(f"training job {job.job} timed out and was cancelled")
        time.sleep(poll_interval)


def run_infer(args: argparse.Namespace) -> int:
    profile = get_profile(args.model)
    cli_values = cli_options(
        images=args.images,
        weights=args.weights,
        repository=args.repository,
        service_name=args.service_name,
        nclasses=args.nclasses,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        gpu=args.gpu,
        gpuid=_parse_gpu_ids(args.gpuid),
        output=args.output,
        visualize=args.visualize,
        benchmark=args.benchmark,
        warmup=args.warmup,
        output_format=args.output_format,
        confidence_threshold=getattr(args, "confidence_threshold", None),
        best_bbox=getattr(args, "best_bbox", None),
    )
    options = _resolve_options(profile.infer_defaults(), args, cli_values)
    _normalize_gpu_options(options, args)
    images = [Path(image) for image in options.get("images", [])]
    if not images:
        raise ValueError("at least one image is required")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"input image not found: {image}")
    for numeric in ("width", "height", "batch_size"):
        _validate_positive(numeric, int(options[numeric]))
    if int(options["warmup"]) < 0:
        raise ValueError("warmup must be non-negative")
    if profile.name == "yolox":
        threshold = float(options["confidence_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        if options.get("best_bbox") is not None:
            _validate_positive("best_bbox", int(options["best_bbox"]))

    writer = EventWriter(output_format=options["output_format"])
    stage_model(options["weights"], options["repository"])
    dd = deepdetect.DeepDetect()
    configure_gpu_compatibility(dd.build_info, requested=bool(options["gpu"]))
    service_parameters = profile.service_parameters(options)
    predict_parameters = profile.predict_parameters(options)
    batch_times: list[float] = []
    all_predictions: list[dict[str, Any]] = []

    with dd.create_service(options["service_name"], **service_parameters) as service:
        first_batch = next(chunks([image.resolve() for image in images], int(options["batch_size"])))
        for _ in range(int(options["warmup"])):
            service.predict(first_batch, **predict_parameters)
        for batch in chunks([image.resolve() for image in images], int(options["batch_size"])):
            started = time.perf_counter()
            result = service.predict(batch, **predict_parameters)
            elapsed = time.perf_counter() - started
            batch_times.append(elapsed)
            predictions = result.get("predictions", [])
            if len(predictions) != len(batch):
                raise ValueError("DeepDetect returned an unexpected prediction count")
            per_image_ms = elapsed * 1000.0 / len(batch)
            for image, prediction in zip(batch, predictions):
                all_predictions.append(prediction)
                writer.emit(
                    "prediction",
                    image=str(image),
                    time_ms=per_image_ms,
                    prediction=prediction,
                )

    if options.get("visualize") or options.get("output") is not None:
        _write_visual_outputs(
            profile.name,
            images,
            all_predictions,
            Path(options["output"] or "deepdetect-output"),
            writer,
        )
    if options.get("benchmark"):
        writer.emit(
            "benchmark",
            batch_size=int(options["batch_size"]),
            warmup=int(options["warmup"]),
            **summarize_timings(batch_times, len(images)),
        )
    return 0


def _write_visual_outputs(
    model: str,
    images: list[Path],
    predictions: list[dict[str, Any]],
    output: Path,
    writer: EventWriter,
) -> None:
    multiple = len(images) > 1
    for image, prediction in zip(images, predictions):
        if model == "yolox":
            path = output_path_for(output, image, multiple=multiple, suffix="_detections")
            render_detections(image, prediction, path)
            writer.emit("artifact", kind="detections", image=str(image), path=str(path))
        else:
            overlay = output_path_for(output, image, multiple=multiple, suffix="_overlay")
            mask = overlay.with_name(f"{overlay.stem}_mask.png")
            render_segmentation(image, prediction, mask, overlay)
            writer.emit("artifact", kind="mask", image=str(image), path=str(mask))
            writer.emit("artifact", kind="overlay", image=str(image), path=str(overlay))


def run_job_status(args: argparse.Namespace) -> int:
    writer = EventWriter(output_format=args.output_format)
    manifest = load_run(args.run)
    writer.emit(
        "training_status",
        run_id=manifest.data.get("run_id"),
        status=manifest.data.get("status"),
        last_status=manifest.data.get("last_status", {}),
        run_dir=str(manifest.run_dir),
    )
    return 0


def run_inspect_models(args: argparse.Namespace) -> int:
    writer = EventWriter(output_format=args.output_format)
    for profile in PROFILES.values():
        writer.emit(
            "model",
            name=profile.name,
            task=profile.task,
            description=profile.description,
            default_nclasses=profile.default_nclasses,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepdetect")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="train a model")
    train_models = train.add_subparsers(dest="model", required=True)
    for model in PROFILES:
        _add_train_parser(train_models.add_parser(model, help=f"train {model}"))

    infer = subcommands.add_parser("infer", help="run inference")
    infer_models = infer.add_subparsers(dest="model", required=True)
    for model in PROFILES:
        _add_infer_parser(infer_models.add_parser(model, help=f"infer {model}"), model)

    job = subcommands.add_parser("job", help="inspect CLI run manifests")
    job_subcommands = job.add_subparsers(dest="job_command", required=True)
    status = job_subcommands.add_parser("status", help="read the latest run status")
    status.add_argument("run", type=Path, help="run directory or run.json path")
    status.add_argument("--output-format", choices=("json", "jsonl", "text"), default="json")
    status.set_defaults(func=run_job_status)

    inspect = subcommands.add_parser("inspect", help="inspect CLI capabilities")
    inspect_subcommands = inspect.add_subparsers(dest="inspect_command", required=True)
    models = inspect_subcommands.add_parser("models", help="list model profiles")
    models.add_argument("--output-format", choices=("json", "jsonl", "text"), default="json")
    models.set_defaults(func=run_inspect_models)
    return parser


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        help="override config values with dotted keys, e.g. solver.base_lr=0.001",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "jsonl", "text"),
        default=None,
        help="event output format",
    )


def _add_train_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_config(parser)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--service-name")
    parser.add_argument("--nclasses", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--base-lr", type=float)
    parser.add_argument("--test-interval", type=int)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gpuid",
        nargs="+",
        help="GPU id or ids to use, e.g. 0, 0 1, 0,1, or -1 for all GPUs",
    )
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        choices=("latest", "best"),
        default=None,
        help="resume training from the latest or best checkpoint in --repository",
    )
    parser.add_argument(
        "--terminal",
        choices=("verbose", "live"),
        default=None,
        help="terminal display mode for training progress",
    )
    parser.add_argument("--visdom", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visdom-server")
    parser.add_argument("--visdom-port", type=int)
    parser.add_argument("--visdom-base-url")
    parser.add_argument(
        "--visdom-results",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="upload sampled test-set prediction overlays to Visdom",
    )
    parser.add_argument("--visdom-results-count", type=int)
    parser.add_argument("--visdom-results-seed", type=int)
    parser.add_argument(
        "--dataset-check",
        choices=("full", "none"),
        default=None,
        help="dataset validation mode before training",
    )
    parser.add_argument(
        "--visdom-offline-ok",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--visdom-save", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--skip-mask-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.set_defaults(func=run_train)


def _add_infer_parser(parser: argparse.ArgumentParser, model: str) -> None:
    _add_common_config(parser)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--service-name")
    parser.add_argument("--nclasses", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gpuid",
        nargs="+",
        help="GPU id or ids to use, e.g. 0, 0 1, 0,1, or -1 for all GPUs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup", type=int)
    if model == "yolox":
        parser.add_argument("--confidence-threshold", type=float)
        parser.add_argument("--best-bbox", type=int)
    parser.set_defaults(func=run_infer)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (
        deepdetect.DeepDetectError,
        OSError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return report_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
