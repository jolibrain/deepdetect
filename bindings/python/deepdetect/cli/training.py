from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import deepdetect

from .checks import run_training_checks
from .config import cli_options, save_config
from .events import EventWriter, MetricEventExtractor, metric_events
from .options import (
    normalize_gpu_options,
    parse_gpu_ids,
    resolve_options,
    validate_positive,
)
from .profiles import get_profile
from .results import TrainingResultVisualizer, create_training_result_visualizer
from .runs import create_run, load_run, repository_name
from .sinks import CompositeMetricSink, JSONLMetricSink, VisdomMetricSink
from .terminal import LiveTrainingTerminalReporter
from .utils import (
    configure_gpu_compatibility,
    stage_model,
    validate_resume_repository,
)


def run_train(args: Any) -> int:
    _debug("run_train: resolving profile and options")
    profile = get_profile(args.model)
    cli_values = cli_options(
        train_data=args.train_data,
        test_data=args.test_data,
        weights=args.weights,
        repository=args.repository,
        service_name=args.service_name,
        nclasses=args.nclasses,
        nkeypoints=args.nkeypoints,
        max_objects=args.max_objects,
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        batch_size=args.batch_size,
        iter_size=args.iter_size,
        base_lr=args.base_lr,
        test_interval=args.test_interval,
        gpu=args.gpu,
        gpuid=parse_gpu_ids(args.gpuid),
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
    options = resolve_options(profile.train_defaults(), args, cli_values)
    normalize_gpu_options(options, gpu_disabled=args.gpu is False)
    for required in ("train_data", "test_data", "repository"):
        if not options.get(required):
            raise ValueError(f"{required.replace('_', '-')} is required")
    resume = options.get("resume")
    if resume is not None and resume not in {"latest", "best"}:
        raise ValueError("resume must be one of: latest, best")
    if options.get("terminal") not in {"verbose", "live"}:
        raise ValueError("terminal must be one of: verbose, live")
    if resume is None and profile.requires_weights and not options.get("weights"):
        raise ValueError("weights is required unless --resume is used")
    if options.get("job_dir") is None:
        options["job_dir"] = Path(options["repository"])
    for numeric in (
        "width",
        "height",
        "iterations",
        "batch_size",
        "iter_size",
        "test_interval",
        "poll_interval",
    ):
        validate_positive(numeric, options[numeric])
    if int(options["nclasses"]) <= 0:
        raise ValueError("nclasses must be positive")
    if options.get("nkeypoints") is not None and int(options["nkeypoints"]) <= 0:
        raise ValueError("nkeypoints must be positive")
    if options.get("max_objects") is not None and int(options["max_objects"]) <= 0:
        raise ValueError("max_objects must be positive")
    validate_positive("visdom_port", int(options["visdom_port"]))
    if int(options["visdom_results_count"]) < 0:
        raise ValueError("visdom_results_count must be non-negative")
    run_name = str(options.get("run_name") or repository_name(Path(options["repository"])))
    options["run_name"] = run_name
    if resume:
        validate_resume_repository(Path(options["repository"]), str(resume))

    writer = create_training_terminal_reporter(options)
    _debug(
        "run_train: terminal=%s visdom=%s dataset_check=%s"
        % (
            options.get("terminal"),
            options.get("visdom"),
            options.get("dataset_check"),
        )
    )
    manifest = create_or_resume_run_manifest(
        args.model,
        options,
        run_name,
        job_dir_is_run_dir=path_is_repository(options["job_dir"], options["repository"]),
    )
    writer.emit(
        "run_started",
        run_id=manifest.data["run_id"],
        run_name=manifest.data["run_name"],
        run_dir=str(manifest.run_dir),
        resume=resume,
    )
    metric_sink, visdom_sink = create_training_metric_sink(
        options,
        writer=writer,
        run_id=manifest.data["run_id"],
        run_dir=manifest.run_dir,
    )
    _debug("run_train: metric sinks initialized")
    extractor = MetricEventExtractor()

    try:
        if resume:
            _debug("run_train: replaying resume history")
            replay_resume_history(
                Path(options["repository"]),
                writer=writer,
                run_id=manifest.data["run_id"],
                extractor=extractor,
                visdom_sink=visdom_sink,
            )
        _debug("run_train: running dataset checks")
        for check in run_training_checks(profile.task, options):
            writer.emit("dataset_check", run_id=manifest.data["run_id"], **check)
        _debug("run_train: dataset checks complete")

        if not resume and options.get("weights") is not None:
            _debug("run_train: staging model weights")
            options["weights"] = stage_model(options["weights"], options["repository"])
        save_training_config(options)
        _debug("run_train: config saved")
        dd = deepdetect.DeepDetect()
        _debug("run_train: DeepDetect runtime created")
        configure_gpu_compatibility(dd.build_info, requested=bool(options["gpu"]))
        _debug("run_train: GPU compatibility checked")
        service_parameters = profile.service_parameters(options)
        train_parameters = profile.train_parameters(options)
        result_visualizer = create_training_result_visualizer(
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
            _debug("run_train: result visualizer enabled")
        if training_live_terminal_enabled(options):
            dd.set_log_level("warn")
        _debug("run_train: creating service")
        with dd.create_service(options["service_name"], **service_parameters) as service:
            _debug("run_train: service created")
            if training_live_terminal_enabled(options):
                dd.set_service_log_level(options["service_name"], "warn")
            _debug("run_train: starting service.train")
            result = service.train(
                [
                    Path(options["train_data"]).resolve(),
                    *[Path(path).resolve() for path in options["test_data"]],
                ],
                asynchronous=not bool(options["sync"]),
                **train_parameters,
            )
            _debug(
                "run_train: service.train returned %s" % type(result).__name__
            )
            if isinstance(result, deepdetect.TrainingJob):
                _debug("run_train: monitoring async training job")
                final_status = monitor_training(
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
                _debug("run_train: processing synchronous training result")
                final_status = {"status": "finished", **result}
                writer.emit(
                    "training_status",
                    run_id=manifest.data["run_id"],
                    status="finished",
                    measure=result.get("measure", {}),
                )
                events = write_metric_events(
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
        manifest.update(
            status=final_status.get("status", "finished"),
            last_status=final_status,
        )
        return 0
    finally:
        _debug("run_train: closing sinks and terminal")
        metric_sink.close()
        writer.close()


def _debug(message: str) -> None:
    if os.environ.get("DEEPDETECT_DEBUG") or os.environ.get("DEEPDETECT_CLI_DEBUG"):
        print(f"[deepdetect-debug][cli] {message}", file=sys.stderr, flush=True)


def create_training_terminal_reporter(options: dict[str, Any]):
    if training_live_terminal_enabled(options):
        return LiveTrainingTerminalReporter(
            total_iterations=int(options["iterations"]),
            gpu_ids=live_terminal_gpu_ids(options),
        )
    if str(options.get("terminal", "verbose")) == "live":
        return EventWriter(output_format="jsonl")
    return EventWriter(output_format=options["output_format"])


def save_training_config(options: dict[str, Any]) -> Path:
    path = Path(options["repository"]).expanduser().resolve() / "config.yaml"
    save_config(path, options)
    return path


def training_live_terminal_enabled(options: dict[str, Any]) -> bool:
    return str(options.get("terminal", "verbose")) == "live" and sys.stdout.isatty()


def live_terminal_gpu_ids(options: dict[str, Any]) -> int | list[int] | None:
    if not bool(options.get("gpu")):
        return None
    gpuid = options.get("gpuid")
    return 0 if gpuid is None else gpuid


def create_training_metric_sink(
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


def create_or_resume_run_manifest(
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


def path_is_repository(job_dir: Path, repository: Path) -> bool:
    return job_dir.expanduser().resolve() == repository.expanduser().resolve()


def replay_resume_history(
    repository: Path,
    *,
    writer: EventWriter,
    run_id: str,
    extractor: MetricEventExtractor,
    visdom_sink: VisdomMetricSink | None,
) -> None:
    status = load_repository_metrics_status(repository)
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
        progress = history_progress(total=len(events))
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


def load_repository_metrics_status(repository: Path) -> dict[str, Any] | None:
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


def history_progress(*, total: int):
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


def write_metric_events(
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


def monitor_training(
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
        _debug(f"monitor_training: polling job {job.job}")
        output_parameters: dict[str, Any] = {
            "measure_hist": True,
            "max_hist_points": 10000,
        }
        if result_visualizer is not None:
            output_parameters["test_predictions"] = True
        status = job.status(output_parameters=output_parameters)
        state = str(status.get("status", "")).lower()
        measure = status.get("measure") if isinstance(status.get("measure"), dict) else {}
        _debug(
            "monitor_training: status=%s iteration=%s train_loss=%s"
            % (
                state,
                measure.get("iteration"),
                measure.get("train_loss"),
            )
        )
        writer.emit(
            "training_status",
            run_id=manifest.data["run_id"],
            job=job.job,
            status=state,
            time=status.get("time"),
            measure=status.get("measure", {}),
            measures=status.get("measures"),
        )
        events = write_metric_events(
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
