from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


class RunManifest:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @property
    def run_dir(self) -> Path:
        return self.path.parent

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )


def sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "run"


def repository_name(repository: Path) -> str:
    name = repository.expanduser().name
    if name:
        return sanitize_run_name(name)
    return "repository"


def generate_training_run_name(
    *, model: str, repository: Path, timestamp: str | None = None
) -> str:
    return repository_name(repository)


def create_run(
    root: Path,
    *,
    command: str,
    model: str,
    service_name: str,
    options: dict[str, Any],
    run_name: str | None = None,
    exist_ok: bool = False,
    root_is_run_dir: bool = False,
) -> RunManifest:
    run_id = sanitize_run_name(run_name) if run_name else (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    run_dir = root.expanduser().resolve()
    if not root_is_run_dir:
        run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok or root_is_run_dir)
    manifest_path = run_dir / "run.json"
    if manifest_path.exists() and not exist_ok:
        raise FileExistsError(f"run manifest already exists: {manifest_path}")
    data = {
        "run_id": run_id,
        "run_name": run_id,
        "command": command,
        "model": model,
        "service_name": service_name,
        "created_at": time.time(),
        "options": options,
        "status": "created",
    }
    manifest = RunManifest(manifest_path, data)
    manifest.save()
    return manifest


def load_run(path: Path) -> RunManifest:
    path = path.expanduser()
    manifest_path = path / "run.json" if path.is_dir() else path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunManifest(manifest_path, data)


def summarize_timings(batch_times: list[float], image_count: int) -> dict[str, Any]:
    total = sum(batch_times)
    per_image = total / image_count if image_count else 0.0
    sorted_times = sorted(batch_times)
    median_batch = sorted_times[len(sorted_times) // 2] if sorted_times else 0.0
    return {
        "images": image_count,
        "total_seconds": total,
        "avg_ms_per_image": per_image * 1000.0,
        "median_batch_ms": median_batch * 1000.0,
        "min_batch_ms": (min(batch_times) * 1000.0) if batch_times else 0.0,
        "max_batch_ms": (max(batch_times) * 1000.0) if batch_times else 0.0,
        "throughput_images_per_sec": (image_count / total) if total > 0 else 0.0,
    }
