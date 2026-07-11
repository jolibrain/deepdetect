from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .assignment import hungarian_assign
from .targets import PoseTargetConfig, build_batch_targets, build_topdown_batch_targets


@dataclass(frozen=True)
class PoseLossConfig:
    target: PoseTargetConfig
    heatmap_weight: float = 1.0
    objectness_weight: float = 1.0


@dataclass(frozen=True)
class PoseLossReduction:
    heatmap_numerator: torch.Tensor
    heatmap_denominator: torch.Tensor
    objectness_numerator: torch.Tensor | None = None
    objectness_denominator: int = 0
    assignments: tuple[tuple[int, int, int], ...] = ()


def topdown_pose_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    config: PoseLossConfig,
    torch_module: Any,
    device: Any,
    return_reduction: bool = False,
) -> (
    tuple[dict[str, torch.Tensor], dict[str, float]]
    | tuple[dict[str, torch.Tensor], dict[str, float], PoseLossReduction]
):
    target_heatmaps, target_weights = build_topdown_batch_targets(
        targets,
        config=config.target,
        torch_module=torch_module,
        device=device,
    )
    if return_reduction:
        heatmap_numerator, heatmap_denominator = masked_heatmap_mse_reduction(
            outputs["heatmaps"], target_heatmaps, target_weights
        )
        heatmap_loss = reduce_heatmap_mse(heatmap_numerator, heatmap_denominator)
    else:
        heatmap_loss = masked_heatmap_mse(
            outputs["heatmaps"], target_heatmaps, target_weights
        )
    losses = {
        "loss": float(config.heatmap_weight) * heatmap_loss,
        "heatmap_loss": heatmap_loss,
    }
    stats = {"assigned_objects": float(len(targets)), "dropped_objects": 0.0}
    if return_reduction:
        return (
            losses,
            stats,
            PoseLossReduction(
                heatmap_numerator=heatmap_numerator,
                heatmap_denominator=heatmap_denominator,
            ),
        )
    return losses, stats


def slot_pose_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    config: PoseLossConfig,
    torch_module: Any,
    device: Any,
    return_reduction: bool = False,
) -> (
    tuple[dict[str, torch.Tensor], dict[str, float]]
    | tuple[dict[str, torch.Tensor], dict[str, float], PoseLossReduction]
):
    pred_heatmaps = outputs["heatmaps"]
    pred_objectness = outputs["objectness"]
    target_heatmaps, target_weights, object_mask, dropped = build_batch_targets(
        targets,
        config=config.target,
        torch_module=torch_module,
        device=device,
    )
    matched_heatmaps = torch.zeros_like(pred_heatmaps)
    matched_weights = torch.zeros(
        (
            pred_heatmaps.shape[0],
            pred_heatmaps.shape[1],
            pred_heatmaps.shape[2],
            1,
        ),
        dtype=pred_heatmaps.dtype,
        device=pred_heatmaps.device,
    )
    objectness_target = torch.zeros_like(pred_objectness)
    assignments = 0
    matched_assignments: list[tuple[int, int, int]] = []
    with torch.no_grad():
        positive_cost = F.binary_cross_entropy_with_logits(
            pred_objectness,
            torch.ones_like(pred_objectness),
            reduction="none",
        )
        for batch_index in range(int(pred_heatmaps.shape[0])):
            object_count = int(object_mask[batch_index].sum().item())
            if object_count <= 0:
                continue
            costs: list[list[float]] = []
            for slot_index in range(int(pred_heatmaps.shape[1])):
                row = []
                for object_index in range(object_count):
                    heatmap_cost = _visible_heatmap_cost(
                        pred_heatmaps[batch_index, slot_index],
                        target_heatmaps[batch_index, object_index],
                        target_weights[batch_index, object_index],
                    )
                    row.append(
                        heatmap_cost
                        + float(config.objectness_weight)
                        * float(positive_cost[batch_index, slot_index].item())
                    )
                costs.append(row)
            for slot_index, object_index in hungarian_assign(costs):
                matched_heatmaps[batch_index, slot_index].copy_(
                    target_heatmaps[batch_index, object_index]
                )
                matched_weights[batch_index, slot_index].copy_(
                    target_weights[batch_index, object_index]
                )
                objectness_target[batch_index, slot_index] = 1.0
                assignments += 1
                matched_assignments.append((batch_index, slot_index, object_index))

    if return_reduction:
        heatmap_numerator, heatmap_denominator = masked_heatmap_mse_reduction(
            pred_heatmaps,
            matched_heatmaps,
            matched_weights,
        )
        heatmap_loss = reduce_heatmap_mse(heatmap_numerator, heatmap_denominator)
        objectness_numerator = F.binary_cross_entropy_with_logits(
            pred_objectness,
            objectness_target,
            reduction="sum",
        )
        objectness_denominator = int(objectness_target.numel())
        objectness_loss = objectness_numerator / float(objectness_denominator)
    else:
        heatmap_loss = masked_heatmap_mse(
            pred_heatmaps,
            matched_heatmaps,
            matched_weights,
        )
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_objectness,
            objectness_target,
        )
    total = (
        float(config.heatmap_weight) * heatmap_loss
        + float(config.objectness_weight) * objectness_loss
    )
    losses = {
        "loss": total,
        "heatmap_loss": heatmap_loss,
        "objectness_loss": objectness_loss,
    }
    stats = {
        "assigned_objects": float(assignments),
        "dropped_objects": float(dropped),
    }
    if return_reduction:
        return (
            losses,
            stats,
            PoseLossReduction(
                heatmap_numerator=heatmap_numerator,
                heatmap_denominator=heatmap_denominator,
                objectness_numerator=objectness_numerator,
                objectness_denominator=objectness_denominator,
                assignments=tuple(matched_assignments),
            ),
        )
    return losses, stats


def masked_heatmap_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    numerator, denominator = masked_heatmap_mse_reduction(pred, target, weights)
    return reduce_heatmap_mse(numerator, denominator)


def masked_heatmap_mse_reduction(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = weights.to(dtype=pred.dtype).unsqueeze(-1)
    squared = (pred - target).pow(2) * weights
    denominator = weights.sum() * pred.shape[-1] * pred.shape[-2]
    return squared.sum(), denominator


def reduce_heatmap_mse(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0
    return numerator / denominator.clamp(min=1.0)


def _visible_heatmap_cost(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    visible = weights.reshape(-1) > 0.0
    if not bool(visible.any().item()):
        return 0.0
    diff = pred.detach()[visible] - target.detach()[visible]
    return float(diff.pow(2).mean().item())
