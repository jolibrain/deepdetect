from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .events import EventWriter
from .sinks import VisdomMetricSink
from .visualize import (
    detection_overlay_image,
    keypoint_overlay_image,
    segmentation_overlay_images,
)

VISDOM_RESULT_MAX_SIDE = 512
VISDOM_RESULT_CONFIDENCE_THRESHOLD = 0.1


class TrainingResultVisualizer:
    def __init__(
        self,
        *,
        model: str,
        task: str,
        samples: list[list["TestResultSample"]],
        sample_count: int,
        sample_seed: int,
        image_size: tuple[int, int],
        best_bbox: int | None,
        artifact_dir: Path,
        visdom_sink: VisdomMetricSink,
        warning_callback: Any,
        disable_failed: bool,
    ) -> None:
        self.model = model
        self.task = task
        self.samples = samples
        self.sample_count = sample_count
        self.sample_seed = sample_seed
        self.image_size = image_size
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
                "confidence_threshold": VISDOM_RESULT_CONFIDENCE_THRESHOLD,
                "sample_count": self.sample_count,
                "sample_seed": self.sample_seed,
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
                image = result_image_array(
                    self.task,
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
                images=visdom_result_image_arrays(images),
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


def create_training_result_visualizer(
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
        samples = test_result_samples(options["test_data"])
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
        sample_count=count,
        sample_seed=int(options.get("visdom_results_seed", 12345)),
        image_size=(int(options["width"]), int(options["height"])),
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


def test_result_samples(test_data: list[Path]) -> list[list[TestResultSample]]:
    samples = []
    for list_path in test_data:
        images = dataset_image_paths(Path(list_path))
        indexed_images = [
            TestResultSample(sample_index, image)
            for sample_index, image in enumerate(images)
        ]
        samples.append(indexed_images)
    return samples


def dataset_image_paths(list_path: Path) -> list[Path]:
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


def visdom_result_image_arrays(images: list[np.ndarray]) -> list[np.ndarray]:
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


def result_image_array(
    task: str,
    image_path: Path,
    prediction: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    if task == "detection" or task == "yolox" or task == "torchvision-detector":
        image = detection_overlay_image(
            image_path,
            prediction,
            coordinate_size=prediction_image_size(prediction) or image_size,
        )
    elif task == "keypoint":
        image = keypoint_overlay_image(
            image_path,
            prediction,
            coordinate_size=prediction_image_size(prediction) or image_size,
        )
    else:
        _mask, image = segmentation_overlay_images(
            image_path,
            prediction,
            original_size=True,
        )
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array.transpose(2, 0, 1)


def prediction_image_size(prediction: dict[str, Any]) -> tuple[int, int] | None:
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
