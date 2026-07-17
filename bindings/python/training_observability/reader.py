"""Read generic and legacy DeepDetect training-run artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .plots import metric_plot_key
from .writer import DEFAULT_ARTIFACTS, DEFAULT_MANIFEST, DEFAULT_METRICS, SCHEMA_VERSION


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    step: float
    timestamp: float


@dataclass(frozen=True)
class PlotSpec:
    id: str
    title: str
    ylabel: str
    traces: tuple[str, ...]


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: Path
    step: float | None
    split: str | None
    metadata_path: Path | None
    timestamp: float | None
    metadata: dict[str, Any] | None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RunReader:
    """Read an approved local run root without modifying it."""

    def __init__(self, root: str | Path) -> None:
        source = Path(root).expanduser().resolve()
        self.root = source.parent if source.is_file() else source
        if not self.root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {self.root}")
        self.warnings: list[str] = []
        self.manifest, self.layout = self._load_manifest()

    def _load_manifest(self) -> tuple[dict[str, Any], str]:
        path = self.root / DEFAULT_MANIFEST
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid observability manifest: {path}") from error
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported observability schema: {manifest.get('schema_version')!r}"
                )
            return manifest, "manifest"
        if (self.root / "run.json").is_file() or (self.root / DEFAULT_METRICS).is_file():
            return (
                {
                    "schema_version": "deepdetect-legacy-v1",
                    "run": {"metadata_path": "run.json"},
                    "metrics": {"path": DEFAULT_METRICS, "format": "deepdetect-metric-v1"},
                    "artifacts": {"path": DEFAULT_ARTIFACTS},
                },
                "deepdetect-legacy",
            )
        raise FileNotFoundError(
            f"no {DEFAULT_MANIFEST}, run.json, or {DEFAULT_METRICS} in {self.root}"
        )

    def _contained_path(self, value: Any, *, field: str) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            self.warnings.append(f"ignored {field} outside run root: {value}")
            return None
        return candidate

    def _stream_path(self, section: str, default: str) -> Path:
        value = self.manifest.get(section, {})
        path = self._contained_path(value.get("path", default), field=f"{section}.path")
        return path or self.root / default

    def run_metadata(self) -> dict[str, Any]:
        run = self.manifest.get("run", {})
        path = self._contained_path(run.get("metadata_path", "run.json"), field="run.metadata_path")
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.warnings.append(f"invalid run metadata: {path.name}")
            return {}
        return data if isinstance(data, dict) else {}

    def metric_points(
        self,
        *,
        names: set[str] | None = None,
        from_step: float | None = None,
        to_step: float | None = None,
    ) -> list[MetricPoint]:
        path = self._stream_path("metrics", DEFAULT_METRICS)
        if not path.is_file():
            return []
        points: list[MetricPoint] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    self.warnings.append(f"ignored invalid metric record at line {index + 1}")
                continue
            if not isinstance(record, dict) or record.get("event") != "metric":
                continue
            name = record.get("name")
            if not isinstance(name, str) or (names is not None and name not in names):
                continue
            value = _as_float(record.get("value"))
            step = _as_float(record.get("step", record.get("iteration")))
            if value is None or step is None:
                continue
            if from_step is not None and step < from_step:
                continue
            if to_step is not None and step > to_step:
                continue
            timestamp = _as_float(record.get("timestamp"))
            points.append(MetricPoint(name, value, step, timestamp or 0.0))
        return points

    def plots(self) -> list[PlotSpec]:
        grouped: dict[str, tuple[str, str, set[str]]] = {}
        for point in self.metric_points():
            key = metric_plot_key(point.name)
            entry = grouped.setdefault(key.plot_id, (key.title, key.ylabel, set()))
            entry[2].add(key.trace)
        return [
            PlotSpec(plot_id, title, ylabel, tuple(sorted(traces)))
            for plot_id, (title, ylabel, traces) in sorted(grouped.items())
        ]

    def _artifact_records(self) -> Iterable[dict[str, Any]]:
        path = self._stream_path("artifacts", DEFAULT_ARTIFACTS)
        if not path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.warnings.append(f"ignored invalid artifact record at line {index + 1}")
                continue
            if isinstance(record, dict) and record.get("event") == "artifact":
                records.append(record)
        return records

    def artifacts(
        self,
        *,
        kind: str | None = None,
        split: str | None = None,
        step: float | None = None,
    ) -> list[Artifact]:
        records = list(self._artifact_records())
        if not records and self.layout == "deepdetect-legacy":
            records = self._legacy_result_records()
        artifacts: list[Artifact] = []
        for record in records:
            record_kind = record.get("kind")
            if not isinstance(record_kind, str) or (kind is not None and record_kind != kind):
                continue
            record_split = record.get("split") if isinstance(record.get("split"), str) else None
            if split is not None and record_split != split:
                continue
            record_step = _as_float(record.get("step"))
            if step is not None and record_step != step:
                continue
            artifact_path = self._contained_path(record.get("path"), field="artifact.path")
            if artifact_path is None or not artifact_path.is_file():
                continue
            metadata_path = self._contained_path(record.get("metadata_path"), field="artifact.metadata_path")
            if metadata_path is not None and not metadata_path.is_file():
                metadata_path = None
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else None
            artifacts.append(
                Artifact(
                    record_kind,
                    artifact_path,
                    record_step,
                    record_split,
                    metadata_path,
                    _as_float(record.get("timestamp")),
                    metadata,
                )
            )
        return artifacts

    def _legacy_result_records(self) -> list[dict[str, Any]]:
        base = self.root / "visdom-results"
        if not base.is_dir():
            return []
        records = []
        for image_path in sorted(base.glob("iteration-*/test*/sample-*.png")):
            relative = image_path.relative_to(self.root).as_posix()
            metadata = image_path.with_suffix(".json")
            iteration = image_path.parents[1].name.removeprefix("iteration-")
            try:
                step = float(iteration)
            except ValueError:
                step = None
            records.append(
                {
                    "event": "artifact",
                    "kind": "prediction-overlay",
                    "path": relative,
                    "step": step,
                    "split": image_path.parent.name,
                    "metadata_path": metadata.relative_to(self.root).as_posix(),
                }
            )
        return records

    def summary(self) -> dict[str, Any]:
        metadata = self.run_metadata()
        grouped: dict[str, list[MetricPoint]] = defaultdict(list)
        for point in self.metric_points():
            grouped[point.name].append(point)
        metrics = []
        for name, points in sorted(grouped.items()):
            ordered = sorted(points, key=lambda point: (point.step, point.timestamp))
            values = [point.value for point in ordered]
            metrics.append(
                {
                    "name": name,
                    "count": len(ordered),
                    "first_step": ordered[0].step,
                    "last_step": ordered[-1].step,
                    "last_value": ordered[-1].value,
                    "min": min(values),
                    "max": max(values),
                }
            )
        summary = {
            "root": str(self.root),
            "layout": self.layout,
            "run_id": metadata.get("run_id", self.manifest.get("run", {}).get("id")),
            "status": metadata.get("status"),
            "metrics": metrics,
            "plot_count": len(self.plots()),
            "artifact_count": len(self.artifacts()),
            "warnings": list(dict.fromkeys(self.warnings)),
        }
        return summary


def point_dict(point: MetricPoint) -> dict[str, Any]:
    return asdict(point)


def plot_dict(plot: PlotSpec) -> dict[str, Any]:
    return {**asdict(plot), "traces": list(plot.traces)}


def artifact_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        "kind": artifact.kind,
        "path": str(artifact.path),
        "step": artifact.step,
        "split": artifact.split,
        "metadata_path": str(artifact.metadata_path) if artifact.metadata_path else None,
        "timestamp": artifact.timestamp,
        "metadata": artifact.metadata,
    }
