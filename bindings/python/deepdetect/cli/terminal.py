from __future__ import annotations

import math
import sys
import time
from collections.abc import Sequence
from typing import Any, TextIO


LOSS_PRIORITY = (
    "train_loss",
    "total_loss",
    "cls_loss",
    "conf_loss",
    "iou_loss",
    "l1_loss",
)

METRIC_PRIORITY = (
    "map",
    "map-05",
    "map-50",
    "map-90",
    "acc",
    "meaniou",
    "meanacc",
)

INTERNAL_METRIC_NAMES = {
    "iteration",
    "learning_rate",
    "batch_duration_ms",
    "iter_time",
    "iteration_duration_ms",
    "remain_time",
    "remain_time_str",
    "elapsed_time_ms",
    "test_active",
    "test_set_index",
    "test_sets_total",
    "test_processed",
    "test_total",
    "test_names",
}


class LiveTrainingTerminalReporter:
    def __init__(
        self,
        *,
        total_iterations: int,
        gpu_ids: int | Sequence[int] | None = None,
        gpu_monitor: Any = None,
        stream: TextIO | None = None,
        force_terminal: bool | None = None,
    ) -> None:
        from rich.console import Console

        self.total_iterations = max(int(total_iterations), 1)
        self.events: list[dict[str, Any]] = []
        self._status = "starting"
        self._measure: dict[str, Any] = {}
        self._latest_metrics: dict[str, float] = {}
        self._latest_warning: str | None = None
        self._gpu_monitor = (
            gpu_monitor
            if gpu_monitor is not None
            else NVMLGpuMonitor.create(gpu_ids)
        )
        self._stream = stream or sys.stdout
        self._console = Console(
            file=self._stream,
            force_terminal=force_terminal,
            soft_wrap=True,
        )
        self._live: Any = None

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"event": event, "timestamp": time.time(), **payload}
        self.events.append(record)
        if event == "training_status":
            self._status = str(record.get("status", self._status) or self._status)
            measure = record.get("measure")
            if isinstance(measure, dict):
                self._measure = measure
                self._update_latest_metrics(measure)
            if self._live is None and not self._ready_to_render():
                return record
            self._start_or_refresh()
        elif event == "metric":
            name = record.get("name")
            value = _finite_float(record.get("value"))
            if isinstance(name, str) and value is not None and _is_display_metric(name):
                self._latest_metrics[name] = value
                self._refresh()
        elif event == "run_finished":
            self._status = str(record.get("status", self._status) or self._status)
            self._refresh()
        elif event == "sink_warning":
            sink = record.get("sink", "sink")
            message = record.get("message", "")
            self._latest_warning = f"{sink}: {message}"
            self._refresh()
        return record

    def close(self) -> None:
        if self._live is None:
            return
        self._live.update(self._render())
        self._live.stop()

    def _start_or_refresh(self) -> None:
        if self._live is None:
            from rich.live import Live

            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
            return
        self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        self._live.update(self._render(), refresh=True)

    def _ready_to_render(self) -> bool:
        return bool(self._loss_values())

    def _render(self):
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.rule import Rule
        from rich.table import Table

        table = Table.grid(padding=(0, 1))
        table.expand = True
        gpu_text = self._gpu_text()
        if gpu_text:
            table.add_row(f"[bold]gpu[/] {gpu_text}")
        table.add_row(
            self._training_progress(
                Progress(
                    TextColumn("[bold]train[/]"),
                    BarColumn(bar_width=None),
                    TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                    TextColumn(self._eta_text()),
                    expand=True,
                )
            )
        )
        table.add_row(f"[bold]loss[/] {_format_values(self._loss_values(), empty='pending')}")
        table.add_row(Rule(style="dim"))
        if self._has_test_progress():
            table.add_row(self._test_progress(Progress, SpinnerColumn, TextColumn, BarColumn))
        table.add_row(
            f"[bold]metrics[/] {_format_values(self._metric_values(), empty='pending')}"
        )
        if self._latest_warning:
            table.add_row(f"[bold red]warning[/] {self._latest_warning}")
        return Panel(table, title=f"training {self._status}", border_style="cyan")

    def _gpu_text(self) -> str | None:
        if self._gpu_monitor is None:
            return None
        snapshots = self._gpu_monitor.snapshot()
        if not snapshots:
            return None
        return "  |  ".join(snapshots)

    def _training_progress(self, progress):
        iteration = _finite_float(self._measure.get("iteration")) or 0.0
        completed = min(max(iteration, 0.0), float(self.total_iterations))
        progress.add_task("train", total=float(self.total_iterations), completed=completed)
        return progress

    def _has_test_progress(self) -> bool:
        if _finite_float(self._measure.get("test_active")) == 1.0:
            return True
        processed = _finite_float(self._measure.get("test_processed"))
        total = _finite_float(self._measure.get("test_total"))
        return processed is not None and total is not None and total > 0

    def _test_progress(self, Progress, SpinnerColumn, TextColumn, BarColumn):
        index = int(_finite_float(self._measure.get("test_set_index")) or 0)
        total_sets = int(_finite_float(self._measure.get("test_sets_total")) or 0)
        processed = _finite_float(self._measure.get("test_processed")) or 0.0
        total = _finite_float(self._measure.get("test_total")) or 0.0
        label = f"test set {index + 1}/{total_sets}" if total_sets > 0 else "test"
        if total > 0:
            progress = Progress(
                TextColumn(f"[bold]{label}[/]"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                TextColumn("{task.percentage:>3.0f}%"),
                expand=True,
            )
            progress.add_task("test", total=total, completed=min(processed, total))
            return progress
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{label}[/] {processed:.0f} processed"),
            expand=True,
        )
        progress.add_task("test", total=None)
        return progress

    def _eta_text(self) -> str:
        eta = self._measure.get("remain_time_str")
        return f"ETA {eta}" if isinstance(eta, str) and eta else ""

    def _loss_values(self) -> dict[str, float]:
        losses: dict[str, float] = {}
        for name in LOSS_PRIORITY:
            value = _finite_float(self._measure.get(name))
            if value is not None:
                losses[name] = value
        for name in sorted(self._measure):
            if name in losses or "loss" not in name.lower():
                continue
            value = _finite_float(self._measure.get(name))
            if value is not None:
                losses[name] = value
        return losses

    def _metric_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in METRIC_PRIORITY:
            value = self._latest_metrics.get(name)
            if value is not None:
                values[name] = value
        for name in sorted(self._latest_metrics):
            if name in values:
                continue
            if name.startswith("fp") or _is_display_metric(name):
                values[name] = self._latest_metrics[name]
        return values

    def _update_latest_metrics(self, measure: dict[str, Any]) -> None:
        for name, raw_value in measure.items():
            if not _is_display_metric(name):
                continue
            value = _finite_float(raw_value)
            if value is not None:
                self._latest_metrics[name] = value


def _format_values(values: dict[str, float], *, empty: str) -> str:
    if not values:
        return empty
    return "  ".join(f"{name}={value:.6g}" for name, value in values.items())


class NVMLGpuMonitor:
    def __init__(self, nvml: Any, gpu_ids: list[int]) -> None:
        self._nvml = nvml
        self._gpu_ids = gpu_ids

    @classmethod
    def create(cls, gpu_ids: int | Sequence[int] | None) -> "NVMLGpuMonitor | None":
        ids = _normalize_gpu_ids(gpu_ids)
        if not ids:
            return None
        try:
            import pynvml

            pynvml.nvmlInit()
            if ids == [-1]:
                ids = list(range(pynvml.nvmlDeviceGetCount()))
            return cls(pynvml, ids)
        except Exception:
            return None

    def snapshot(self) -> list[str]:
        snapshots = []
        for gpu_id in self._gpu_ids:
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            except Exception:
                continue
            used_gb = memory.used / (1024**3)
            total_gb = memory.total / (1024**3)
            memory_pct = 0.0 if memory.total == 0 else memory.used * 100.0 / memory.total
            snapshots.append(
                f"{gpu_id} util={utilization.gpu:.0f}% "
                f"mem={used_gb:.1f}/{total_gb:.1f}GB ({memory_pct:.0f}%)"
            )
        return snapshots


def _normalize_gpu_ids(gpu_ids: int | Sequence[int] | None) -> list[int]:
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    return [int(gpu_id) for gpu_id in gpu_ids]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_display_metric(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    if name in INTERNAL_METRIC_NAMES or name.startswith("iteration_"):
        return False
    if "loss" in name.lower():
        return False
    return True
