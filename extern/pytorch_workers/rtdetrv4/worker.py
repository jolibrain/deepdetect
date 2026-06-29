from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from deepdetect.pytorch_worker.builtin.vision.detection.base import (
    DetectionTrainingWorkerBase,
)
from deepdetect.pytorch_worker.builtin.vision.detection.common import (
    checkpoint_path,
)
from deepdetect.pytorch_worker.sdk import (
    DatasetContractError,
    WorkerDependencyError,
)


class DeepDetectWorker(DetectionTrainingWorkerBase):
    worker_name = "rtdetrv4-external"
    debug_name = "rtdetrv4-external"

    def __init__(self) -> None:
        super().__init__()
        self._cfg: Any = None
        self._postprocessor: Any = None

    def import_backend(self) -> tuple[Any, ...]:
        try:
            import torch
        except Exception as error:
            raise WorkerDependencyError("torch could not be imported") from error
        return (torch,)

    def backend_versions(self, *backend: Any) -> dict[str, Any]:
        (torch,) = backend
        return {
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "upstream": "rtdetrv4",
        }

    def create_model(self, nclasses: int, *backend: Any) -> Any:
        cfg = self._load_config(nclasses)
        model = self._build_with_single_process_distributed_fallback(
            lambda: getattr(cfg, "model", None)
        )
        if model is None and callable(getattr(cfg, "build_model", None)):
            model = self._build_with_single_process_distributed_fallback(
                cfg.build_model
            )
        if model is None:
            raise WorkerDependencyError(
                "RT-DETRv4 YAMLConfig did not expose model or build_model()"
            )
        self._cfg = cfg
        self._postprocessor = self._build_with_single_process_distributed_fallback(
            lambda: getattr(cfg, "postprocessor", None)
        )
        return model

    def prepare_training_batch(
        self,
        images: list[Any],
        targets: list[dict[str, Any]],
        metas: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        moved_images, moved_targets = super().prepare_training_batch(
            images,
            targets,
            metas,
        )
        batched = moved_images[0].new_empty((len(moved_images), *moved_images[0].shape))
        for index, image in enumerate(moved_images):
            batched[index].copy_(image)
        return batched, [
            self._target_to_rtdetr(target, meta)
            for target, meta in zip(moved_targets, metas)
        ]

    def training_losses(
        self,
        model: Any,
        images: Any,
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        output = model(images, targets)
        if (
            isinstance(output, dict)
            and output
            and all(hasattr(value, "backward") for value in output.values())
        ):
            return output
        criterion = getattr(self._cfg, "criterion", None)
        if criterion is None:
            raise DatasetContractError(
                "RT-DETRv4 model did not return a loss dict and no criterion is available"
            )
        losses = criterion(output, targets)
        if not isinstance(losses, dict):
            raise DatasetContractError("RT-DETRv4 criterion must return a loss dict")
        return losses

    def predict_batch(self, model: Any, images: list[Any]) -> list[dict[str, Any]]:
        batch = images[0].new_empty((len(images), *images[0].shape))
        for index, image in enumerate(images):
            batch[index].copy_(image)
        self._force_dynamic_position_embeddings(model)
        self._move_cached_position_embeddings(model, batch.device)
        outputs = model(batch)
        if self._postprocessor is not None:
            sizes = batch.new_tensor(
                [[int(image.shape[-1]), int(image.shape[-2])] for image in images],
                dtype=batch.dtype,
            )
            outputs = self._postprocessor(outputs, sizes)
        return outputs

    def convert_prediction_outputs(
        self,
        outputs: list[dict[str, Any]],
        images: list[Any],
        metas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted = []
        for output in outputs:
            labels = output.get("labels")
            if labels is not None:
                labels = labels.to(dtype=labels.dtype) + 1
            converted.append(
                {
                    "boxes": output.get("boxes"),
                    "scores": output.get("scores"),
                    "labels": labels,
                }
            )
        return converted

    def load_model_for_training(
        self,
        checkpoint_manager: Any,
        model: Any,
        mllib: dict[str, Any],
    ) -> Path | None:
        path = self._checkpoint_path_for_training(mllib)
        if path is None:
            return None
        return self._load_checkpoint_payload(checkpoint_manager.torch, model, path)

    def _checkpoint_path_for_training(self, mllib: dict[str, Any]) -> Path | None:
        path = checkpoint_path(mllib, self.context)
        if path is None:
            options = mllib.get("rtdetrv4", {})
            options = options if isinstance(options, dict) else {}
            raw = (
                options.get("pretrained_model")
                or options.get("pretrained_path")
                or options.get("weights")
                or options.get("checkpoint")
            )
            if not raw:
                return None
            path = Path(str(raw)).expanduser().resolve()
        if path.is_dir():
            return self._checkpoint_from_directory(path)
        if not path.is_file():
            raise WorkerDependencyError(f"RT-DETRv4 checkpoint not found: {path}")
        return path

    @staticmethod
    def _checkpoint_from_directory(path: Path) -> Path:
        candidates: list[Path] = []
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            candidates.extend(path.glob(pattern))
        files = sorted(item for item in candidates if item.is_file())
        if not files:
            raise WorkerDependencyError(
                f"RT-DETRv4 checkpoint directory has no .pth, .pt, or .ckpt files: {path}"
            )
        if len(files) == 1:
            return files[0]
        preferred_names = (
            "checkpoint-latest.pt",
            "model.pth",
            "model.pt",
            "weights.pth",
            "weights.pt",
        )
        by_name = {item.name: item for item in files}
        for name in preferred_names:
            if name in by_name:
                return by_name[name]
        return max(files, key=lambda item: item.stat().st_mtime)

    def _load_config(self, nclasses: int) -> Any:
        if self.context is None:
            raise WorkerDependencyError("RT-DETRv4 worker is not configured")
        options = self.context.mllib.get("rtdetrv4", {})
        options = options if isinstance(options, dict) else {}
        repo_path = options.get("repo_path") or os.environ.get("RTDETRV4_REPO")
        if not repo_path:
            raise WorkerDependencyError(
                "RT-DETRv4 repo path is required: set rtdetrv4.repo_path "
                "or RTDETRV4_REPO"
            )
        repo = Path(str(repo_path)).expanduser().resolve()
        if not repo.is_dir():
            raise WorkerDependencyError(f"RT-DETRv4 repo path not found: {repo}")
        config_path = options.get("config_path")
        if not config_path:
            raise WorkerDependencyError("rtdetrv4.config_path is required")
        config = Path(str(config_path)).expanduser()
        if not config.is_absolute():
            config = repo / config
        if not config.is_file():
            raise WorkerDependencyError(f"RT-DETRv4 config not found: {config}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from engine.core import YAMLConfig
        except Exception as error:
            raise WorkerDependencyError(
                "could not import RT-DETRv4 engine.core.YAMLConfig"
            ) from error
        cfg = YAMLConfig(str(config))
        self._patch_yaml_config(cfg, nclasses=nclasses, options=options)
        return cfg

    def _patch_yaml_config(
        self,
        cfg: Any,
        *,
        nclasses: int,
        options: dict[str, Any],
    ) -> None:
        yaml_cfg = getattr(cfg, "yaml_cfg", None)
        if not isinstance(yaml_cfg, dict):
            return
        foreground_classes = max(1, int(nclasses) - 1)
        remap_mscoco_category = bool(options.get("remap_mscoco_category", False))
        yaml_cfg["num_classes"] = foreground_classes
        yaml_cfg["remap_mscoco_category"] = remap_mscoco_category
        self._set_nested_key(yaml_cfg, "model", "num_classes", foreground_classes)
        self._set_nested_key(
            yaml_cfg, "postprocessor", "num_classes", foreground_classes
        )
        self._set_nested_key(
            yaml_cfg,
            "postprocessor",
            "remap_mscoco_category",
            remap_mscoco_category,
        )
        self._set_nested_key(yaml_cfg, "criterion", "num_classes", foreground_classes)
        for key, value in list(yaml_cfg.items()):
            if not isinstance(value, dict):
                continue
            key_lower = key.lower()
            if key_lower == "postprocessor":
                value["num_classes"] = foreground_classes
                value["remap_mscoco_category"] = remap_mscoco_category
            elif key_lower.endswith("criterion"):
                value["num_classes"] = foreground_classes
        if bool(options.get("disable_teacher", True)):
            yaml_cfg.pop("teacher_model", None)
            yaml_cfg.pop("teacher", None)
            yaml_cfg.pop("ema_teacher", None)
            for value in yaml_cfg.values():
                if isinstance(value, dict):
                    self._remove_distillation(value)
        if not bool(options.get("pretrained_backbone", False)):
            for key, value in yaml_cfg.items():
                if isinstance(value, dict):
                    self._disable_pretrained(value)
                elif key in {
                    "pretrained",
                    "pretrained_backbone",
                    "backbone_pretrained",
                }:
                    yaml_cfg[key] = False

    @staticmethod
    def _set_nested_key(
        values: dict[str, Any],
        name: str,
        key: str,
        replacement: Any,
    ) -> None:
        child = values.get(name)
        if isinstance(child, dict):
            child[key] = replacement

    def _disable_pretrained(self, values: dict[str, Any]) -> None:
        for key, value in list(values.items()):
            if key in {"pretrained", "pretrained_backbone", "backbone_pretrained"}:
                values[key] = False
            elif isinstance(value, dict):
                self._disable_pretrained(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._disable_pretrained(item)

    def _remove_distillation(self, values: dict[str, Any]) -> None:
        if isinstance(values.get("losses"), list):
            values["losses"] = [
                loss for loss in values["losses"] if "distill" not in str(loss)
            ]
        weight_dict = values.get("weight_dict")
        if isinstance(weight_dict, dict):
            for key in list(weight_dict):
                if "distill" in str(key):
                    weight_dict.pop(key, None)
        for key in list(values):
            if "distill" in str(key):
                values.pop(key, None)
        for value in values.values():
            if isinstance(value, dict):
                self._remove_distillation(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._remove_distillation(item)

    def _build_with_single_process_distributed_fallback(self, factory: Any) -> Any:
        try:
            import torch
        except Exception:
            return factory()
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not hasattr(distributed, "is_initialized"):
            return factory()
        try:
            if bool(distributed.is_initialized()):
                return factory()
        except Exception:
            return factory()

        original_get_rank = getattr(distributed, "get_rank", None)
        original_get_world_size = getattr(distributed, "get_world_size", None)
        original_barrier = getattr(distributed, "barrier", None)
        distributed.get_rank = lambda *args, **kwargs: 0
        distributed.get_world_size = lambda *args, **kwargs: 1
        distributed.barrier = lambda *args, **kwargs: None
        try:
            return factory()
        finally:
            if original_get_rank is not None:
                distributed.get_rank = original_get_rank
            if original_get_world_size is not None:
                distributed.get_world_size = original_get_world_size
            if original_barrier is not None:
                distributed.barrier = original_barrier

    @staticmethod
    def _move_cached_position_embeddings(model: Any, device: Any) -> None:
        for module in model.modules():
            for name, value in list(vars(module).items()):
                if name.startswith("pos_embed") and hasattr(value, "to"):
                    moved = value.to(device=device)
                    if moved is not value:
                        setattr(module, name, moved)

    @staticmethod
    def _force_dynamic_position_embeddings(model: Any) -> None:
        for module in model.modules():
            if hasattr(module, "eval_spatial_size"):
                module.eval_spatial_size = None

    def _target_to_rtdetr(
        self,
        target: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        boxes = target["boxes"]
        labels = target["labels"] - 1
        width = float(meta["width"])
        height = float(meta["height"])
        if int(boxes.numel()) == 0:
            norm_boxes = boxes.reshape((-1, 4))
        else:
            x0, y0, x1, y1 = boxes.unbind(dim=1)
            norm_boxes = boxes.new_empty(boxes.shape)
            norm_boxes[:, 0] = ((x0 + x1) * 0.5) / width
            norm_boxes[:, 1] = ((y0 + y1) * 0.5) / height
            norm_boxes[:, 2] = (x1 - x0) / width
            norm_boxes[:, 3] = (y1 - y0) / height
        converted = dict(target)
        converted["boxes"] = norm_boxes
        converted["labels"] = labels
        converted_size = target["labels"].new_tensor([int(width), int(height)])
        converted["orig_size"] = converted_size.clone()
        converted["size"] = converted_size.clone()
        return converted

    def _load_checkpoint_payload(
        self,
        torch: Any,
        model: Any,
        path: Path,
    ) -> Path:
        payload = torch.load(path, map_location=self.device)
        state = self._state_dict_from_checkpoint(payload)
        model_state = model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state.items():
            target = model_state.get(key)
            if target is None and key.startswith("module."):
                target = model_state.get(key[len("module.") :])
                if target is not None:
                    key = key[len("module.") :]
            if target is None:
                skipped.append(key)
                continue
            if hasattr(value, "shape") and hasattr(target, "shape"):
                if tuple(value.shape) != tuple(target.shape):
                    skipped.append(key)
                    continue
            compatible[key] = value
        model.load_state_dict(compatible, strict=False)
        if skipped:
            self.debug(
                "train: skipped incompatible RT-DETRv4 checkpoint tensors "
                f"count={len(skipped)} sample={skipped[:8]}"
            )
        return path

    @staticmethod
    def _state_dict_from_checkpoint(payload: Any) -> dict[str, Any]:
        state = payload
        if isinstance(payload, dict):
            if "model_state" in payload:
                state = payload["model_state"]
            elif "model" in payload:
                state = payload["model"]
            elif isinstance(payload.get("ema"), dict):
                ema = payload["ema"]
                state = ema.get("module", ema)
            elif "state_dict" in payload:
                state = payload["state_dict"]
        if not isinstance(state, dict):
            raise WorkerDependencyError(
                "RT-DETRv4 checkpoint did not contain a model state dict"
            )
        return state
