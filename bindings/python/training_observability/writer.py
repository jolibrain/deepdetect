"""Writers for the small, portable training-observability/v1 layout."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "training-observability/v1"
DEFAULT_MANIFEST = "training-observability.json"
DEFAULT_METRICS = "metrics.jsonl"
DEFAULT_ARTIFACTS = "artifacts.jsonl"


def _relative_path(root: Path, value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"path must be inside run root: {value}") from error
    if any(part == ".." for part in path.parts):
        raise ValueError(f"path must not escape run root: {value}")
    return path.as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_run_manifest(
    root: str | Path,
    *,
    run_id: str | None = None,
    metadata_path: str | Path | None = "run.json",
    config_path: str | Path | None = "config.yaml",
    metrics_path: str | Path = DEFAULT_METRICS,
    metric_format: str = "generic-metric-v1",
    artifacts_path: str | Path = DEFAULT_ARTIFACTS,
    plot_groups: list[Mapping[str, Any]] | None = None,
) -> Path:
    """Create or refresh a versioned manifest without touching data streams."""
    run_root = Path(root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {}
    if run_id is not None:
        run["id"] = str(run_id)
    if metadata_path is not None:
        run["metadata_path"] = _relative_path(run_root, metadata_path)
    if config_path is not None:
        run["config_path"] = _relative_path(run_root, config_path)
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "metrics": {
            "path": _relative_path(run_root, metrics_path),
            "format": str(metric_format),
        },
        "artifacts": {"path": _relative_path(run_root, artifacts_path)},
    }
    if plot_groups:
        data["plot_groups"] = [dict(group) for group in plot_groups]
    path = run_root / DEFAULT_MANIFEST
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def append_artifact(
    root: str | Path,
    *,
    kind: str,
    path: str | Path,
    step: float | int | None = None,
    split: str | None = None,
    metadata_path: str | Path | None = None,
    timestamp: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a visual/artifact record with run-root-contained paths."""
    run_root = Path(root).expanduser().resolve()
    record: dict[str, Any] = {
        "event": "artifact",
        "kind": str(kind),
        "path": _relative_path(run_root, path),
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    if step is not None:
        record["step"] = float(step)
    if split is not None:
        record["split"] = str(split)
    if metadata_path is not None:
        record["metadata_path"] = _relative_path(run_root, metadata_path)
    if extra:
        record["metadata"] = dict(extra)
    artifact_path = run_root / DEFAULT_ARTIFACTS
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
    return record


class RunWriter:
    """Minimal writer that another trainer can copy or depend on directly."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str | None = None,
        metadata_path: str | Path | None = "run.json",
        config_path: str | Path | None = "config.yaml",
        plot_groups: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        write_run_manifest(
            self.root,
            run_id=run_id,
            metadata_path=metadata_path,
            config_path=config_path,
            plot_groups=plot_groups,
        )

    def metric(
        self,
        name: str,
        value: float | int,
        *,
        step: float | int,
        timestamp: float | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        number = float(value)
        record: dict[str, Any] = {
            "event": "metric",
            "name": str(name),
            "value": number,
            "step": float(step),
            "timestamp": time.time() if timestamp is None else float(timestamp),
        }
        if tags:
            record["tags"] = {str(key): str(item) for key, item in tags.items()}
        metrics_path = self.root / DEFAULT_METRICS
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        return record

    def artifact(self, **kwargs: Any) -> dict[str, Any]:
        return append_artifact(self.root, **kwargs)
