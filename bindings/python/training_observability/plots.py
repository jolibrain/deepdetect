"""Shared plot naming and local static rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MetricPlotKey:
    plot_id: str
    title: str
    ylabel: str
    trace: str


def metric_plot_key(name: str) -> MetricPlotKey:
    base, separator, suffix = name.rpartition("_test")
    trace = name
    test_trace = None
    if separator and suffix.isdigit() and base:
        name = base
        test_trace = f"test{suffix}"
        trace = test_trace
    class_metric = _per_class_map_metric(name)
    if class_metric is not None:
        name, class_id = class_metric
        trace = f"class {class_id}"
        if test_trace is not None:
            trace += f" / {test_trace}"
    window = _window_for(name)
    return MetricPlotKey(window, _window_title(window), _window_ylabel(window, name), trace)


def _per_class_map_metric(name: str) -> tuple[str, str] | None:
    metric, separator, class_id = name.rpartition("_")
    if not separator or not class_id.isdigit() or int(class_id) <= 0:
        return None
    if metric == "map":
        return metric, class_id
    if not metric.startswith("map-") or not metric[4:].isdigit():
        return None
    return metric, class_id


def _window_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "loss"


def _window_for(name: str) -> str:
    normalized = name.lower()
    if "loss" in normalized:
        return f"loss-{_window_name(name)}"
    if normalized.startswith("map"):
        return f"metric-{_window_name(name)}"
    if normalized.startswith("fp"):
        return "metric-fp"
    if normalized in {"num_fg", "run_fg"}:
        return "metric-num-fg"
    if normalized in {"learning_rate", "lr"}:
        return "metric-learning-rate"
    if "time" in normalized or "duration" in normalized:
        return "metric-time"
    return f"metric-{_window_name(name)}"


def _window_title(window: str) -> str:
    if window.startswith("loss-"):
        return window[5:].replace("-", " ")
    if window == "metric-map":
        return "mAP metrics"
    if window.startswith("metric-map-"):
        return window[7:].replace("-", " ")
    if window == "metric-fp":
        return "false positive metrics"
    if window == "metric-num-fg":
        return "foreground count"
    if window == "metric-learning-rate":
        return "learning rate"
    if window == "metric-time":
        return "timing metrics"
    if window.startswith("metric-"):
        return window[7:].replace("-", " ")
    return window.replace("-", " ")


def _window_ylabel(window: str, name: str) -> str:
    if window.startswith("loss-"):
        return name
    if window == "metric-map" or window.startswith("metric-map-"):
        return "mAP"
    if window == "metric-fp":
        return "false positives"
    if window == "metric-num-fg":
        return "num_fg"
    if window == "metric-learning-rate":
        return "learning rate"
    if window == "metric-time":
        return "time"
    return name


def render_plot(plot, points: Iterable, output: str | Path) -> Path:
    """Render one plot spec to PNG, importing matplotlib only when needed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot rendering needs matplotlib; install the observability extra"
        ) from error

    grouped = defaultdict(list)
    for point in points:
        key = metric_plot_key(point.name)
        if key.plot_id == plot.id:
            grouped[key.trace].append(point)
    if not grouped:
        raise ValueError(f"plot has no finite points: {plot.id}")

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for trace, trace_points in sorted(grouped.items()):
        ordered = sorted(trace_points, key=lambda point: (point.step, point.timestamp))
        axis.plot(
            [point.step for point in ordered],
            [point.value for point in ordered],
            label=trace,
        )
    axis.set_title(plot.title)
    axis.set_xlabel("step")
    axis.set_ylabel(plot.ylabel)
    axis.grid(True, alpha=0.25)
    if len(grouped) > 1:
        axis.legend()
    figure.savefig(destination, dpi=144)
    plt.close(figure)
    return destination
