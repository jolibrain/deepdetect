from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import deepdetect

from .config import cli_options
from .events import EventWriter
from .options import (
    normalize_gpu_options,
    parse_gpu_ids,
    resolve_options,
    validate_positive,
)
from .profiles import get_profile
from .runs import summarize_timings
from .utils import configure_gpu_compatibility, stage_model
from .visualize import (
    output_path_for,
    render_detections,
    render_instance_masks,
    render_keypoints,
    render_segmentation,
)


def run_infer(args: Any) -> int:
    profile = get_profile(args.model)
    cli_values = cli_options(
        images=args.images,
        images_file=getattr(args, "images_file", None),
        weights=args.weights,
        repository=args.repository,
        service_name=args.service_name,
        nclasses=args.nclasses,
        nkeypoints=args.nkeypoints,
        max_objects=args.max_objects,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        gpu=args.gpu,
        gpuid=parse_gpu_ids(args.gpuid),
        output=args.output,
        visualize=args.visualize,
        benchmark=args.benchmark,
        warmup=args.warmup,
        output_format=args.output_format,
        confidence_threshold=getattr(args, "confidence_threshold", None),
        keypoint_threshold=getattr(args, "keypoint_threshold", None),
        bbox_files=getattr(args, "bbox_files", None),
        bbox_files_file=getattr(args, "bbox_files_file", None),
        best_bbox=getattr(args, "best_bbox", None),
    )
    options = resolve_options(profile.infer_defaults(), args, cli_values)
    normalize_gpu_options(options, gpu_disabled=args.gpu is False)
    images = _input_paths(
        options.get("images", []),
        options.get("images_file"),
        label="images",
    )
    if not images:
        raise ValueError("at least one image is required")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"input image not found: {image}")
    if profile.requires_weights and not options.get("weights"):
        raise ValueError("weights is required")
    for numeric in ("width", "height", "batch_size"):
        validate_positive(numeric, int(options[numeric]))
    if int(options["warmup"]) < 0:
        raise ValueError("warmup must be non-negative")
    if profile.task in {"detection", "keypoint"}:
        threshold = float(options["confidence_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
    if profile.task == "keypoint":
        keypoint_threshold = float(options["keypoint_threshold"])
        if not 0.0 <= keypoint_threshold <= 1.0:
            raise ValueError("keypoint threshold must be between 0 and 1")
    if profile.task == "detection":
        if options.get("best_bbox") is not None:
            validate_positive("best_bbox", int(options["best_bbox"]))

    writer = EventWriter(output_format=options["output_format"], collect_events=False)
    if options.get("weights") is not None:
        options["weights"] = stage_model(options["weights"], options["repository"])
    dd = deepdetect.DeepDetect()
    configure_gpu_compatibility(dd.build_info, requested=bool(options["gpu"]))
    service_parameters = profile.service_parameters(options)
    predict_parameters = profile.predict_parameters(options)
    bbox_files = _input_paths(
        options.get("bbox_files") or [],
        options.get("bbox_files_file"),
        label="bbox files",
    )
    if profile.task == "keypoint":
        vitpose = service_parameters["mllib_parameters"].get("vitpose", {})
        head = (
            str(vitpose.get("head", "topdown"))
            if isinstance(vitpose, dict)
            else "topdown"
        )
        if head == "topdown":
            if len(bbox_files) != len(images):
                raise ValueError(
                    "top-down ViTPose requires one --bbox-files entry per image"
                )
            for bbox_file in bbox_files:
                if not bbox_file.is_file():
                    raise FileNotFoundError(f"bbox file not found: {bbox_file}")
        elif bbox_files:
            raise ValueError("--bbox-files is only valid for top-down ViTPose")
    elif profile.task == "instance-segmentation":
        if bbox_files and len(bbox_files) != len(images):
            raise ValueError(
                "SAM2 requires one --bbox-files entry per image when boxes are used"
            )
        for bbox_file in bbox_files:
            if not bbox_file.is_file():
                raise FileNotFoundError(f"bbox file not found: {bbox_file}")
    batch_times: list[float] = []
    write_artifacts = options.get("visualize") or options.get("output") is not None
    output_path = Path(options["output"] or "deepdetect-output")
    multiple_images = len(images) > 1

    with dd.create_service(options["service_name"], **service_parameters) as service:
        resolved_images = [image.resolve() for image in images]
        batch_size = int(options["batch_size"])
        first_batch = resolved_images[:batch_size]
        first_parameters = _prediction_batch_parameters(
            predict_parameters, bbox_files[:batch_size]
        )
        for _ in range(int(options["warmup"])):
            service.predict(first_batch, **first_parameters)
        for start in range(0, len(resolved_images), batch_size):
            batch = resolved_images[start : start + batch_size]
            batch_parameters = _prediction_batch_parameters(
                predict_parameters, bbox_files[start : start + batch_size]
            )
            started = time.perf_counter()
            result = service.predict(batch, **batch_parameters)
            elapsed = time.perf_counter() - started
            batch_times.append(elapsed)
            predictions = result.get("predictions", [])
            if len(predictions) != len(batch):
                raise ValueError("DeepDetect returned an unexpected prediction count")
            per_image_ms = elapsed * 1000.0 / len(batch)
            for image, prediction in zip(batch, predictions):
                writer.emit(
                    "prediction",
                    image=str(image),
                    time_ms=per_image_ms,
                    prediction=prediction,
                )
                if write_artifacts:
                    write_visual_outputs(
                        profile.task,
                        [image],
                        [prediction],
                        output_path,
                        writer,
                        multiple=multiple_images,
                    )
    if options.get("benchmark"):
        writer.emit(
            "benchmark",
            batch_size=int(options["batch_size"]),
            warmup=int(options["warmup"]),
            **summarize_timings(batch_times, len(images)),
        )
    return 0


def _prediction_batch_parameters(
    parameters: dict[str, Any], bbox_files: list[Path]
) -> dict[str, Any]:
    result = copy.deepcopy(parameters)
    if bbox_files:
        result["input_parameters"]["bbox_files"] = [
            str(path) for path in bbox_files
        ]
    return result


def _input_paths(
    values: list[Path] | tuple[Path, ...] | list[str] | tuple[str, ...],
    list_path: Path | str | None,
    *,
    label: str,
) -> list[Path]:
    direct = [Path(value).expanduser().resolve() for value in values]
    if list_path is None:
        return direct
    if direct:
        raise ValueError(
            f"{label} cannot be combined with a {label.replace(' ', '-')}-file"
        )
    path = Path(list_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    result: list[Path] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        raw = line.strip()
        if not raw:
            continue
        entry = Path(raw).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        if not entry.is_file():
            singular = label[:-1] if label.endswith("s") else label
            raise FileNotFoundError(
                f"{path}:{line_number}: {singular} not found: {entry}"
            )
        result.append(entry.resolve())
    if not result and label == "images":
        raise ValueError(f"{label} file contains no paths: {path}")
    return result


def write_visual_outputs(
    task: str,
    images: list[Path],
    predictions: list[dict[str, Any]],
    output: Path,
    writer: EventWriter,
    *,
    multiple: bool | None = None,
) -> None:
    if multiple is None:
        multiple = len(images) > 1
    for image, prediction in zip(images, predictions):
        if task == "detection":
            path = output_path_for(output, image, multiple=multiple, suffix="_detections")
            render_detections(image, prediction, path)
            writer.emit("artifact", kind="detections", image=str(image), path=str(path))
        elif task == "keypoint":
            path = output_path_for(output, image, multiple=multiple, suffix="_keypoints")
            render_keypoints(image, prediction, path)
            writer.emit("artifact", kind="keypoints", image=str(image), path=str(path))
        elif task == "instance-segmentation":
            overlay = output_path_for(
                output, image, multiple=multiple, suffix="_sam2_overlay"
            ).with_suffix(".jpg")
            mask_paths = render_instance_masks(image, prediction, overlay)
            for index, mask_path in enumerate(mask_paths, 1):
                writer.emit(
                    "artifact",
                    kind="mask",
                    image=str(image),
                    index=index,
                    path=str(mask_path),
                )
            writer.emit("artifact", kind="overlay", image=str(image), path=str(overlay))
        else:
            overlay = output_path_for(output, image, multiple=multiple, suffix="_overlay")
            mask = overlay.with_name(f"{overlay.stem}_mask.png")
            render_segmentation(image, prediction, mask, overlay)
            writer.emit("artifact", kind="mask", image=str(image), path=str(mask))
            writer.emit("artifact", kind="overlay", image=str(image), path=str(overlay))
