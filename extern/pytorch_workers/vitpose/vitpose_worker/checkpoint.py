from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn.functional as F

from deepdetect.pytorch_worker.sdk import WorkerDependencyError


def checkpoint_path(mllib: dict[str, Any], repository: Path | None) -> Path | None:
    raw = mllib.get("weights") or mllib.get("checkpoint")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    vitpose = mllib.get("vitpose", {})
    if isinstance(vitpose, dict):
        raw = vitpose.get("pretrained_model") or vitpose.get("pretrained")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    if mllib.get("resume") and repository is not None:
        return latest_checkpoint(repository)
    return None


def latest_checkpoint(repository: Path | None) -> Path | None:
    if repository is None:
        return None
    checkpoints = sorted(
        repository.glob("checkpoint-*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def load_model_checkpoint(
    torch: Any,
    model: Any,
    path: Path | None,
    *,
    device: Any,
) -> Path | None:
    if path is None:
        return None
    if not path.is_file():
        raise WorkerDependencyError(f"ViTPose checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    state = _state_dict(payload)
    state = _strip_prefixes(state)
    if _is_bare_vit_backbone(state):
        return _load_backbone_checkpoint(torch, model, state, path)
    checkpoint_head = _checkpoint_head(payload, state)
    model_head = str(getattr(model, "head", "topdown"))
    if checkpoint_head is not None and checkpoint_head != model_head:
        raise WorkerDependencyError(
            f"ViTPose checkpoint head {checkpoint_head!r} is incompatible with "
            f"configured head {model_head!r}"
        )
    _interpolate_pos_embed(state, model, torch)
    model.load_state_dict(state, strict=False)
    return path


def load_optimizer_checkpoint(
    torch: Any,
    optimizer: Any,
    repository: Path | None,
    *,
    device: Any,
    mllib: dict[str, Any],
) -> None:
    if repository is None or not mllib.get("resume"):
        return
    solvers = sorted(
        repository.glob("solver-*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not solvers:
        return
    payload = torch.load(solvers[-1], map_location=device)
    if isinstance(payload, dict) and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])


def save_checkpoint(
    torch: Any,
    model: Any,
    optimizer: Any,
    repository: Path | None,
    iteration: int,
) -> None:
    if repository is None or iteration <= 0:
        return
    repository.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "iteration": int(iteration),
        "nkeypoints": int(getattr(model, "nkeypoints", 0)),
        "max_objects": int(getattr(model, "max_objects", 0)),
        "head": str(getattr(model, "head", "topdown")),
        "model_state": model.state_dict(),
    }
    solver_payload = {
        "iteration": int(iteration),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(model_payload, repository / f"checkpoint-{iteration}.pt")
    torch.save(model_payload, repository / "checkpoint-latest.pt")
    torch.save(solver_payload, repository / f"solver-{iteration}.pt")
    torch.save(solver_payload, repository / "solver-latest.pt")


def _state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _checkpoint_head(payload: Any, state: dict[str, Any]) -> str | None:
    if isinstance(payload, dict) and payload.get("head") in {"topdown", "slots"}:
        return str(payload["head"])
    if any(str(key).startswith("objectness_head.") for key in state):
        return "slots"
    if any(str(key).startswith("keypoint_head.") for key in state):
        return "topdown"
    return None


def _is_bare_vit_backbone(state: dict[str, Any]) -> bool:
    """Recognize MAE/timm-style ViT state dicts without a model prefix."""
    if any(str(key).startswith("backbone.") for key in state):
        return False
    return "patch_embed.proj.weight" in state and any(
        str(key).startswith("blocks.") for key in state
    )


def _load_backbone_checkpoint(
    torch: Any,
    model: Any,
    state: dict[str, Any],
    path: Path,
) -> Path:
    if not hasattr(model, "backbone"):
        raise WorkerDependencyError(
            "ViTPose backbone checkpoint requires a model with a backbone"
        )

    expected = model.backbone.state_dict()
    compatible: dict[str, Any] = {}
    for name, value in state.items():
        target_name = (
            "last_norm." + name[len("norm.") :]
            if name.startswith("norm.")
            else name
        )
        target = expected.get(target_name)
        if target is None or not hasattr(value, "shape"):
            continue
        if tuple(value.shape) != tuple(target.shape) and target_name != "pos_embed":
            continue
        compatible[f"backbone.{target_name}"] = value

    _interpolate_pos_embed(compatible, model, torch)
    backbone_state = {
        name[len("backbone.") :]: value
        for name, value in compatible.items()
        if name.startswith("backbone.")
    }
    if not backbone_state:
        raise WorkerDependencyError(
            f"ViTPose backbone checkpoint has no compatible ViT tensors: {path}"
        )
    model.backbone.load_state_dict(backbone_state, strict=False)
    return path


def _strip_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        result[name] = value
    return result


def _interpolate_pos_embed(state: dict[str, Any], model: Any, torch: Any) -> None:
    key = "backbone.pos_embed"
    if key not in state or not hasattr(model, "backbone"):
        return
    current = model.backbone.pos_embed
    loaded = state[key]
    if tuple(loaded.shape) == tuple(current.shape):
        return
    cls_pos = loaded[:, :1]
    patch_pos = loaded[:, 1:]
    old_size = int(patch_pos.shape[1] ** 0.5)
    new_h, new_w = model.backbone.patch_embed.patch_shape
    if old_size * old_size != int(patch_pos.shape[1]):
        return
    patch_pos = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    state[key] = torch.cat([cls_pos, patch_pos], dim=1)
