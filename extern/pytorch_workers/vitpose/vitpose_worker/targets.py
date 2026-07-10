from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PoseTargetConfig:
    image_size: tuple[int, int]
    heatmap_size: tuple[int, int]
    sigma: float
    max_objects: int
    nkeypoints: int


def select_instances(
    keypoints: torch.Tensor,
    visible: torch.Tensor,
    *,
    max_objects: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    count = int(keypoints.shape[0])
    if count <= max_objects:
        return keypoints, visible, 0
    valid_counts = visible.sum(dim=1).detach().cpu().tolist()
    order = sorted(range(count), key=lambda index: (-float(valid_counts[index]), index))
    keep = torch.tensor(order[:max_objects], dtype=torch.long, device=keypoints.device)
    return keypoints.index_select(0, keep), visible.index_select(0, keep), count - max_objects


def generate_targets_for_sample(
    target: dict[str, Any],
    *,
    config: PoseTargetConfig,
    torch_module: Any,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    keypoints = target["keypoints"].to(device)
    visible = target["visible"].to(device)
    keypoints, visible, dropped = select_instances(
        keypoints,
        visible,
        max_objects=config.max_objects,
    )
    heatmap_width, heatmap_height = config.heatmap_size
    heatmaps = torch_module.zeros(
        (
            config.max_objects,
            config.nkeypoints,
            heatmap_height,
            heatmap_width,
        ),
        dtype=torch_module.float32,
        device=device,
    )
    weights = torch_module.zeros(
        (config.max_objects, config.nkeypoints, 1),
        dtype=torch_module.float32,
        device=device,
    )
    object_mask = torch_module.zeros(
        (config.max_objects,),
        dtype=torch_module.float32,
        device=device,
    )
    object_count = min(int(keypoints.shape[0]), config.max_objects)
    for object_index in range(object_count):
        object_mask[object_index] = 1.0
        for joint_index in range(config.nkeypoints):
            if float(visible[object_index, joint_index].item()) <= 0.0:
                continue
            x = float(keypoints[object_index, joint_index, 0].item())
            y = float(keypoints[object_index, joint_index, 1].item())
            if not math.isfinite(x) or not math.isfinite(y) or x < 0.0 or y < 0.0:
                continue
            if _draw_udp_gaussian(
                heatmaps[object_index, joint_index],
                x,
                y,
                image_size=config.image_size,
                sigma=config.sigma,
            ):
                weights[object_index, joint_index, 0] = 1.0
    return heatmaps, weights, object_mask, dropped


def build_batch_targets(
    targets: list[dict[str, Any]],
    *,
    config: PoseTargetConfig,
    torch_module: Any,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    heatmaps = []
    weights = []
    masks = []
    dropped_total = 0
    for target in targets:
        sample_heatmaps, sample_weights, object_mask, dropped = generate_targets_for_sample(
            target,
            config=config,
            torch_module=torch_module,
            device=device,
        )
        heatmaps.append(sample_heatmaps)
        weights.append(sample_weights)
        masks.append(object_mask)
        dropped_total += dropped
    return (
        torch_module.stack(heatmaps, dim=0),
        torch_module.stack(weights, dim=0),
        torch_module.stack(masks, dim=0),
        dropped_total,
    )


def _draw_udp_gaussian(
    heatmap: torch.Tensor,
    x: float,
    y: float,
    *,
    image_size: tuple[int, int],
    sigma: float,
) -> bool:
    width, height = image_size
    heatmap_height = int(heatmap.shape[0])
    heatmap_width = int(heatmap.shape[1])
    if heatmap_width <= 1 or heatmap_height <= 1:
        return False
    feat_stride_x = (float(width) - 1.0) / (float(heatmap_width) - 1.0)
    feat_stride_y = (float(height) - 1.0) / (float(heatmap_height) - 1.0)
    mu_x_ac = x / feat_stride_x
    mu_y_ac = y / feat_stride_y
    mu_x = int(mu_x_ac + 0.5)
    mu_y = int(mu_y_ac + 0.5)
    tmp_size = int(sigma * 3)
    upper_left = [mu_x - tmp_size, mu_y - tmp_size]
    bottom_right = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
    if (
        upper_left[0] >= heatmap_width
        or upper_left[1] >= heatmap_height
        or bottom_right[0] < 0
        or bottom_right[1] < 0
    ):
        return False
    size = 2 * tmp_size + 1
    arange = torch.arange(size, dtype=heatmap.dtype, device=heatmap.device)
    yy = arange[:, None]
    xx = arange[None, :]
    x0 = size // 2 + mu_x_ac - mu_x
    y0 = size // 2 + mu_y_ac - mu_y
    gaussian = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    g_x0 = max(0, -upper_left[0])
    g_x1 = min(bottom_right[0], heatmap_width) - upper_left[0]
    g_y0 = max(0, -upper_left[1])
    g_y1 = min(bottom_right[1], heatmap_height) - upper_left[1]
    h_x0 = max(0, upper_left[0])
    h_x1 = min(bottom_right[0], heatmap_width)
    h_y0 = max(0, upper_left[1])
    h_y1 = min(bottom_right[1], heatmap_height)
    heatmap[h_y0:h_y1, h_x0:h_x1] = torch.maximum(
        heatmap[h_y0:h_y1, h_x0:h_x1],
        gaussian[g_y0:g_y1, g_x0:g_x1],
    )
    return True
