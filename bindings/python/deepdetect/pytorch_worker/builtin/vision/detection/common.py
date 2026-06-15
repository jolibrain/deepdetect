from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ....sdk import (
    DatasetContractError,
    WorkerContext,
    WorkerDependencyError,
    WorkerReporter,
)


@dataclass(frozen=True)
class DetectionSample:
    index: int
    image: Path
    target: Path


@dataclass(frozen=True)
class DetectionEvalBox:
    image_id: int
    label: int
    box: tuple[float, float, float, float]
    score: float = 1.0


DEFAULT_DETECTION_MAP_THRESHOLDS = {
    "map-05": 0.05,
    "map-50": 0.50,
    "map-90": 0.90,
}


class DetectionListDataset:
    def __init__(self, list_path: Path, *, nclasses: int, torch: Any):
        self.list_path = list_path.expanduser().resolve()
        self.nclasses = nclasses
        self.torch = torch
        self.samples = read_detection_list(
            self.list_path,
            nclasses=nclasses,
            validate_files=False,
            validate_targets=False,
        )
        if not self.samples:
            raise DatasetContractError(f"dataset list contains no samples: {list_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        sample = self.samples[position]
        image, size = read_image_tensor(sample.image, self.torch)
        boxes, labels = read_target_tensors(
            sample.target,
            self.torch,
            nclasses=self.nclasses,
            image_size=size,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": self.torch.tensor([sample.index], dtype=self.torch.int64),
        }
        meta = {
            "index": sample.index,
            "path": str(sample.image),
            "width": size[0],
            "height": size[1],
        }
        return image, target, meta


def read_detection_list(
    list_path: Path,
    *,
    nclasses: int,
    validate_files: bool = True,
    validate_targets: bool = True,
) -> list[DetectionSample]:
    if not list_path.is_file():
        raise DatasetContractError(f"dataset list not found: {list_path}")
    base = list_path.parent
    resolve_paths = validate_files or validate_targets
    samples: list[DetectionSample] = []
    debug(
        "read_detection_list: reading %s validate_files=%s validate_targets=%s"
        % (list_path, validate_files, validate_targets)
    )
    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise DatasetContractError(
                    f"{list_path}:{line_number}: expected image and bbox path"
                )
            image = resolve_dataset_path(base, fields[0], resolve=resolve_paths)
            target = resolve_dataset_path(base, fields[1], resolve=resolve_paths)
            if validate_files and not image.is_file():
                raise DatasetContractError(
                    f"{list_path}:{line_number}: image not found: {image}"
                )
            if validate_files and not target.is_file():
                raise DatasetContractError(
                    f"{list_path}:{line_number}: bbox file not found: {target}"
                )
            if validate_targets:
                validate_bbox_file(target, nclasses=nclasses)
            samples.append(DetectionSample(len(samples), image, target))
            if len(samples) % 50000 == 0:
                debug(f"read_detection_list: parsed {len(samples)} samples")
    debug(f"read_detection_list: parsed {len(samples)} samples total")
    return samples


def resolve_dataset_path(base: Path, value: str, *, resolve: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve() if resolve else path


def validate_bbox_file(path: Path, *, nclasses: int) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetContractError(f"{path}:{line_number}: expected 5 bbox fields")
        label = int(fields[0])
        if label <= 0 or label >= nclasses:
            raise DatasetContractError(
                f"{path}:{line_number}: invalid class {label} for nclasses={nclasses}"
            )
        xmin, ymin, xmax, ymax = (float(value) for value in fields[1:])
        if xmax <= xmin or ymax <= ymin:
            raise DatasetContractError(f"{path}:{line_number}: invalid bbox coordinates")


def read_image_tensor(path: Path, torch: Any) -> tuple[Any, tuple[int, int]]:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, image.size


def read_target_tensors(
    path: Path,
    torch: Any,
    *,
    nclasses: int,
    image_size: tuple[int, int],
) -> tuple[Any, Any]:
    boxes = []
    labels = []
    width, height = image_size
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetContractError(f"{path}:{line_number}: expected 5 bbox fields")
        label = int(fields[0])
        if label <= 0 or label >= nclasses:
            raise DatasetContractError(
                f"{path}:{line_number}: invalid class {label} for nclasses={nclasses}"
            )
        xmin, ymin, xmax, ymax = (float(value) for value in fields[1:])
        xmin = clamp(xmin, 0.0, float(width))
        xmax = clamp(xmax, 0.0, float(width))
        ymin = clamp(ymin, 0.0, float(height))
        ymax = clamp(ymax, 0.0, float(height))
        if xmax <= xmin or ymax <= ymin:
            raise DatasetContractError(f"{path}:{line_number}: invalid bbox coordinates")
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape((-1, 4)),
        torch.tensor(labels, dtype=torch.int64),
    )


def make_loader(
    dataset: DetectionListDataset,
    *,
    batch_size: int,
    shuffle: bool,
    torch: Any,
) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_detection_batch,
    )


def collate_detection_batch(
    batch: Iterable[tuple[Any, dict[str, Any], dict[str, Any]]],
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)


def move_target(target: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in target.items()}


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


def parse_test_prediction_config(output: dict[str, Any]) -> dict[str, Any]:
    value = output.get("test_predictions", {})
    if value is True:
        value = {}
    if not isinstance(value, dict):
        value = {}
    return {
        "sample_count": max(0, int(value.get("sample_count", 10))),
        "sample_seed": int(value.get("sample_seed", 12345)),
        "confidence_threshold": float(value.get("confidence_threshold", 0.1)),
        "best_bbox": optional_positive_int(value.get("best_bbox"), "best_bbox"),
    }


def sampled_indices(total: int, *, count: int, seed: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), min(count, total)))


def prediction_sample(
    meta: dict[str, Any],
    output: dict[str, Any],
    *,
    confidence_threshold: float,
    best_bbox: int | None,
) -> dict[str, Any]:
    return {
        "index": int(meta["index"]),
        "imgsize": {"width": int(meta["width"]), "height": int(meta["height"])},
        "classes": detection_classes(
            output,
            confidence_threshold=confidence_threshold,
            best_bbox=best_bbox,
        ),
    }


def connector_prediction(
    image_path: Path,
    output: dict[str, Any],
    *,
    confidence_threshold: float,
    best_bbox: int | None,
) -> dict[str, Any]:
    classes = detection_classes(
        output,
        confidence_threshold=confidence_threshold,
        best_bbox=best_bbox,
    )
    return {
        "uri": str(image_path),
        "loss": 0.0,
        "probs": [float(item["prob"]) for item in classes],
        "cats": [str(item["cat"]) for item in classes],
        "bboxes": [dict(item["bbox"]) for item in classes],
    }


def detection_classes(
    output: dict[str, Any],
    *,
    confidence_threshold: float,
    best_bbox: int | None,
) -> list[dict[str, Any]]:
    boxes = output.get("boxes")
    scores = output.get("scores")
    labels = output.get("labels")
    if boxes is None or scores is None or labels is None:
        return []
    boxes = boxes.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()
    labels = labels.detach().cpu().tolist()
    classes = []
    for box, score, label in zip(boxes, scores, labels):
        score = float(score)
        if score < confidence_threshold:
            continue
        classes.append(
            {
                "cat": str(int(label)),
                "prob": score,
                "bbox": {
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3]),
                },
            }
        )
    classes.sort(key=lambda item: float(item["prob"]), reverse=True)
    if best_bbox is not None:
        classes = classes[:best_bbox]
    return classes


def detection_metric_thresholds(output: dict[str, Any]) -> dict[str, float]:
    measures = output.get("measure")
    thresholds: dict[str, float] = {}
    if isinstance(measures, list):
        for measure in measures:
            if not isinstance(measure, str) or not measure.startswith("map-"):
                continue
            try:
                threshold = int(measure.split("-", 1)[1])
            except ValueError:
                continue
            if 0 < threshold <= 100:
                thresholds[measure] = float(threshold) / 100.0
    return thresholds or dict(DEFAULT_DETECTION_MAP_THRESHOLDS)


def target_eval_boxes(
    meta: dict[str, Any],
    target: dict[str, Any],
) -> list[DetectionEvalBox]:
    image_id = int(meta["index"])
    boxes = tensor_rows(target.get("boxes"))
    labels = tensor_values(target.get("labels"))
    items = []
    for box, label in zip(boxes, labels):
        parsed = parse_eval_box(box)
        if parsed is None:
            continue
        label = int(label)
        if label <= 0:
            continue
        items.append(DetectionEvalBox(image_id, label, parsed, 1.0))
    return items


def prediction_eval_boxes(
    meta: dict[str, Any],
    output: dict[str, Any],
) -> list[DetectionEvalBox]:
    image_id = int(meta["index"])
    boxes = tensor_rows(output.get("boxes"))
    labels = tensor_values(output.get("labels"))
    scores = tensor_values(output.get("scores"))
    items = []
    for box, label, score in zip(boxes, labels, scores):
        parsed = parse_eval_box(box)
        if parsed is None:
            continue
        label = int(label)
        score = float(score)
        if label <= 0 or not math.isfinite(score):
            continue
        items.append(DetectionEvalBox(image_id, label, parsed, score))
    return items


def detection_map_metrics(
    predictions: list[DetectionEvalBox],
    targets: list[DetectionEvalBox],
    thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    thresholds = thresholds or DEFAULT_DETECTION_MAP_THRESHOLDS
    metrics = {
        name: mean_ap_at_iou(predictions, targets, threshold)
        for name, threshold in thresholds.items()
    }
    map_value = sum(metrics.values()) / float(len(metrics)) if metrics else 0.0
    return {"map": map_value, **metrics}


def report_detection_metrics(
    reporter: WorkerReporter,
    metrics: dict[str, float],
    *,
    iteration: int,
    test_index: int,
) -> None:
    suffix = f"_test{test_index}"
    for name, value in metrics.items():
        reporter.metric(f"{name}{suffix}", float(value), iteration=iteration)


def mean_ap_at_iou(
    predictions: list[DetectionEvalBox],
    targets: list[DetectionEvalBox],
    threshold: float,
) -> float:
    labels = sorted({target.label for target in targets})
    if not labels:
        return 0.0
    aps = []
    for label in labels:
        label_targets = [target for target in targets if target.label == label]
        label_predictions = [
            prediction for prediction in predictions if prediction.label == label
        ]
        aps.append(
            average_precision_for_label(
                label_predictions,
                label_targets,
                threshold,
            )
        )
    return sum(aps) / float(len(aps)) if aps else 0.0


def average_precision_for_label(
    predictions: list[DetectionEvalBox],
    targets: list[DetectionEvalBox],
    threshold: float,
) -> float:
    if not targets:
        return 0.0
    targets_by_image: dict[int, list[DetectionEvalBox]] = {}
    for target in targets:
        targets_by_image.setdefault(target.image_id, []).append(target)
    matched = {
        image_id: [False] * len(image_targets)
        for image_id, image_targets in targets_by_image.items()
    }
    true_positives = []
    false_positives = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        image_targets = targets_by_image.get(prediction.image_id, [])
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(image_targets):
            if matched[prediction.image_id][index]:
                continue
            iou = box_iou(prediction.box, target.box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= threshold:
            matched[prediction.image_id][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    return average_precision(true_positives, false_positives, len(targets))


def average_precision(
    true_positives: list[int],
    false_positives: list[int],
    num_targets: int,
) -> float:
    if num_targets <= 0:
        return 0.0
    if not true_positives:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / float(num_targets))
        precisions.append(cumulative_tp / float(cumulative_tp + cumulative_fp))

    recall_points = [0.0, *recalls, 1.0]
    precision_points = [0.0, *precisions, 0.0]
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(
            precision_points[index],
            precision_points[index + 1],
        )

    ap = 0.0
    for index in range(1, len(recall_points)):
        delta = recall_points[index] - recall_points[index - 1]
        if delta > 0.0:
            ap += delta * precision_points[index]
    return ap


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    xmin = max(left[0], right[0])
    ymin = max(left[1], right[1])
    xmax = min(left[2], right[2])
    ymax = min(left[3], right[3])
    intersection = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def parse_eval_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def tensor_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def tensor_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return []


def select_device(torch: Any, mllib: Any) -> tuple[Any, bool]:
    requested_gpu = bool(mllib.get("gpu")) if isinstance(mllib, dict) else False
    if not requested_gpu:
        return torch.device("cpu"), False
    if not torch.cuda.is_available():
        raise WorkerDependencyError("gpu=true was requested but torch.cuda is not available")
    gpuid = mllib.get("gpuid") if isinstance(mllib, dict) else None
    multi_gpu_requested = isinstance(gpuid, list) and len(gpuid) > 1
    if isinstance(gpuid, list):
        gpu_index = 0 if gpuid == [-1] else int(gpuid[0])
    elif gpuid is None or int(gpuid) == -1:
        gpu_index = 0
    else:
        gpu_index = int(gpuid)
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}"), multi_gpu_requested


def checkpoint_path(mllib: dict[str, Any], context: WorkerContext | None) -> Path | None:
    raw = mllib.get("weights") or mllib.get("checkpoint")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    if mllib.get("resume") and context is not None:
        return latest_checkpoint(context)
    return None


def latest_checkpoint(context: WorkerContext | None) -> Path | None:
    if context is None:
        return None
    checkpoints = sorted(
        context.repository_path.glob("checkpoint-*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def maybe_load_checkpoint(model: Any, torch: Any, device: Any, path: Path | None) -> Path | None:
    if path is None or not path.is_file():
        return None
    payload = torch.load(path, map_location=device)
    state = payload
    if isinstance(payload, dict):
        state = payload.get("model_state", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=False)
    return path


def maybe_load_solver(
    optimizer: Any,
    torch: Any,
    device: Any,
    context: WorkerContext | None,
    mllib: dict[str, Any],
) -> None:
    if not mllib.get("resume") or context is None:
        return
    solvers = sorted(
        context.repository_path.glob("solver-*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not solvers:
        return
    payload = torch.load(solvers[-1], map_location=device)
    if isinstance(payload, dict) and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])


def save_checkpoint(
    context: WorkerContext | None,
    model: Any,
    optimizer: Any,
    torch: Any,
    iteration: int,
) -> None:
    if context is None or iteration <= 0:
        return
    context.repository_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "nclasses": int(getattr(model, "nclasses", 0) or 0),
            "model_state": model.state_dict(),
        },
        context.artifact_path(f"checkpoint-{iteration}.pt"),
    )
    torch.save(
        {
            "iteration": iteration,
            "optimizer_state": optimizer.state_dict(),
        },
        context.artifact_path(f"solver-{iteration}.pt"),
    )


def request_dict(params: dict[str, Any]) -> dict[str, Any]:
    request = params.get("request", {})
    return request if isinstance(request, dict) else {}


def parameters_dict(request: dict[str, Any]) -> dict[str, Any]:
    parameters = request.get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}


def merged_mllib(context: WorkerContext | None, request_params: dict[str, Any]) -> dict[str, Any]:
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def debug(message: str, *, tag: str = "detection-worker") -> None:
    if os.environ.get("DEEPDETECT_DEBUG") or os.environ.get("DEEPDETECT_WORKER_DEBUG"):
        print(
            f"[deepdetect-debug][{tag}] {message}",
            file=sys.stderr,
            flush=True,
        )
