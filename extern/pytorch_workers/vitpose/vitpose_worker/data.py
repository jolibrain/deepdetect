from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from deepdetect.pytorch_worker.sdk import DatasetContractError
from deepdetect.pytorch_worker.tensors import TensorBatchRef, materialize_tensor_ref


class PoseTensorBatchDataset:
    def __init__(self, batches: list[TensorBatchRef], *, nkeypoints: int, torch: Any):
        self.batches = batches
        self.nkeypoints = int(nkeypoints)
        self.torch = torch
        self.samples = self._materialize_samples()
        if not self.samples:
            raise DatasetContractError("pose tensor batch dataset contains no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        return self.samples[index]

    def _materialize_samples(self) -> list[tuple[Any, dict[str, Any], dict[str, Any]]]:
        samples = []
        for batch_index, batch in enumerate(self.batches):
            if len(batch.inputs) != 1:
                raise DatasetContractError(
                    "keypoint tensor batches must contain exactly one image input"
                )
            start = time.monotonic()
            materialized = materialize_tensor_ref(batch.inputs[0], self.torch)
            _elapsed_ms = (time.monotonic() - start) * 1000.0
            try:
                images = materialized.tensor
                if int(images.ndim) == 3:
                    images = images.unsqueeze(0)
                if int(images.ndim) != 4 or int(images.shape[1]) != 3:
                    raise DatasetContractError(
                        "keypoint tensor input must have shape [N, 3, H, W]"
                    )
                targets = batch.targets if isinstance(batch.targets, dict) else {}
                meta = batch.meta if isinstance(batch.meta, dict) else {}
                for sample_index in range(int(images.shape[0])):
                    image = images[sample_index].clone().contiguous()
                    height = int(image.shape[-2])
                    width = int(image.shape[-1])
                    global_index = len(samples)
                    sample_meta = {
                        "index": _meta_value(meta, "sample_ids", sample_index, global_index),
                        "path": _meta_value(
                            meta,
                            "paths",
                            sample_index,
                            f"tensor://batch{batch_index}/{sample_index}",
                        ),
                        "width": int(_meta_value(meta, "widths", sample_index, width)),
                        "height": int(_meta_value(meta, "heights", sample_index, height)),
                        "original_width": int(
                            _meta_value(meta, "original_widths", sample_index, width)
                        ),
                        "original_height": int(
                            _meta_value(meta, "original_heights", sample_index, height)
                        ),
                        "instance_id": int(
                            _meta_value(meta, "instance_ids", sample_index, 0)
                        ),
                        "label": int(_meta_value(meta, "labels", sample_index, 1)),
                        "bbox": _meta_value(meta, "bboxes", sample_index, None),
                        "inverse_affine": _affine_values(
                            _meta_value(meta, "inverse_affines", sample_index, None)
                        ),
                        "source_paths": meta.get("source_paths", []),
                        "source_count": int(meta.get("source_count", 0)),
                    }
                    target = tensor_pose_target(
                        _sample_instances(targets, sample_index),
                        torch=self.torch,
                        nkeypoints=self.nkeypoints,
                        image_id=int(sample_meta["index"]),
                    )
                    samples.append((image, target, sample_meta))
            finally:
                materialized.close()
        return samples


def make_loader(dataset: Any, *, batch_size: int, shuffle: bool, torch: Any) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pose_batch,
    )


def collate_pose_batch(
    batch: Iterable[tuple[Any, dict[str, Any], dict[str, Any]]],
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)


def tensor_pose_target(
    instances: list[Any],
    *,
    torch: Any,
    nkeypoints: int,
    image_id: int,
) -> dict[str, Any]:
    keypoints = []
    visible = []
    for instance in instances:
        points, weights = parse_instance_keypoints(instance, nkeypoints=nkeypoints)
        keypoints.append(points)
        visible.append(weights)
    if keypoints:
        keypoint_tensor = torch.tensor(keypoints, dtype=torch.float32).reshape(
            (-1, nkeypoints, 2)
        )
        visible_tensor = torch.tensor(visible, dtype=torch.float32).reshape(
            (-1, nkeypoints)
        )
    else:
        keypoint_tensor = torch.empty((0, nkeypoints, 2), dtype=torch.float32)
        visible_tensor = torch.empty((0, nkeypoints), dtype=torch.float32)
    return {
        "keypoints": keypoint_tensor,
        "visible": visible_tensor,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
    }


def parse_instance_keypoints(
    instance: Any,
    *,
    nkeypoints: int,
) -> tuple[list[list[float]], list[float]]:
    raw_keypoints = instance.get("keypoints") if isinstance(instance, dict) else instance
    if not isinstance(raw_keypoints, list):
        raise DatasetContractError("keypoint instance must contain a keypoints list")
    points: list[list[float]] = []
    visible: list[float] = []
    if raw_keypoints and all(isinstance(value, (int, float)) for value in raw_keypoints):
        if len(raw_keypoints) != nkeypoints * 2:
            raise DatasetContractError(
                f"flat keypoints must contain {nkeypoints * 2} coordinates"
            )
        iterator = iter(raw_keypoints)
        for x_value, y_value in zip(iterator, iterator):
            x = float(x_value)
            y = float(y_value)
            valid = x >= 0.0 and y >= 0.0
            points.append([x if valid else -1.0, y if valid else -1.0])
            visible.append(1.0 if valid else 0.0)
        return points, visible

    if len(raw_keypoints) != nkeypoints:
        raise DatasetContractError(
            f"expected {nkeypoints} keypoints, got {len(raw_keypoints)}"
        )
    for keypoint in raw_keypoints:
        if isinstance(keypoint, dict):
            x = float(keypoint.get("x", -1.0))
            y = float(keypoint.get("y", -1.0))
            valid = bool(keypoint.get("valid", x >= 0.0 and y >= 0.0))
        elif isinstance(keypoint, list) and len(keypoint) >= 2:
            x = float(keypoint[0])
            y = float(keypoint[1])
            valid = bool(keypoint[2]) if len(keypoint) >= 3 else x >= 0.0 and y >= 0.0
        else:
            raise DatasetContractError("keypoint entries must be objects or lists")
        valid = valid and x >= 0.0 and y >= 0.0
        points.append([x if valid else -1.0, y if valid else -1.0])
        visible.append(1.0 if valid else 0.0)
    return points, visible


def read_image_tensor(
    path: Path,
    torch: Any,
    *,
    image_size: tuple[int, int],
) -> tuple[Any, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    if image.size != image_size:
        image = image.resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, original_size


def normalize_batch(
    images: list[Any],
    *,
    torch: Any,
    device: Any,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Any:
    batch = images[0].new_empty((len(images), *images[0].shape))
    for index, image in enumerate(images):
        batch[index].copy_(image)
    batch = batch.to(device)
    mean_tensor = torch.tensor(mean, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean_tensor) / std_tensor


def move_pose_target(target: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in target.items()}


def _sample_instances(targets: dict[str, Any], index: int) -> list[Any]:
    samples = targets.get("samples")
    if isinstance(samples, list) and index < len(samples):
        sample = samples[index]
        if isinstance(sample, dict):
            instances = sample.get("instances", [])
            return instances if isinstance(instances, list) else []
    instances = targets.get("instances")
    if isinstance(instances, list) and index < len(instances):
        sample_instances = instances[index]
        if isinstance(sample_instances, list):
            return sample_instances
    return []


def _meta_value(meta: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = meta.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def _affine_values(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("values")
    if not isinstance(value, list) or len(value) != 6:
        return None
    return [float(item) for item in value]
