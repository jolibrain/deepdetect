from __future__ import annotations

from collections import defaultdict
from typing import Any


def create_layer_decay_adamw(
    torch: Any,
    model: Any,
    *,
    base_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    layer_decay: float,
) -> Any:
    num_layers = int(getattr(getattr(model, "backbone", None), "depth", 0))
    groups: dict[tuple[float, float], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay = 0.0 if _no_weight_decay(name, parameter) else float(weight_decay)
        layer_id = _layer_id(name, num_layers)
        lr_scale = float(layer_decay) ** max(0, num_layers + 1 - layer_id)
        key = (decay, lr_scale)
        group = groups.setdefault(
            key,
            {
                "params": [],
                "weight_decay": decay,
                "lr": float(base_lr) * lr_scale,
                "lr_scale": lr_scale,
            },
        )
        group["params"].append(parameter)
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(list(groups.values()), lr=float(base_lr), betas=betas)


def _no_weight_decay(name: str, parameter: Any) -> bool:
    return (
        len(parameter.shape) == 1
        or name.endswith(".bias")
        or "pos_embed" in name
        or ".norm" in name
        or "bn" in name
    )


def _layer_id(name: str, num_layers: int) -> int:
    if name.startswith("backbone.patch_embed") or name.startswith("backbone.pos_embed"):
        return 0
    if name.startswith("backbone.blocks."):
        parts = name.split(".")
        if len(parts) > 2:
            try:
                return int(parts[2]) + 1
            except ValueError:
                return num_layers
    return num_layers + 1
