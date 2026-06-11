from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


class MetricSink(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullMetricSink:
    def write(self, event: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class JSONLMetricSink:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._stream.write(json.dumps(event, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class CompositeMetricSink:
    def __init__(
        self,
        sinks: list[MetricSink],
        *,
        warning_callback: Callable[[str, BaseException], None] | None = None,
        disable_failed: bool = True,
    ) -> None:
        self._sinks = list(sinks)
        self._warning_callback = warning_callback
        self._disable_failed = disable_failed

    def write(self, event: dict[str, Any]) -> None:
        active: list[MetricSink] = []
        for sink in self._sinks:
            try:
                sink.write(event)
            except Exception as error:
                if self._warning_callback is not None:
                    self._warning_callback(type(sink).__name__, error)
                if not self._disable_failed:
                    raise
            else:
                active.append(sink)
        self._sinks = active

    def close(self) -> None:
        errors: list[BaseException] = []
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as error:
                if self._warning_callback is not None:
                    self._warning_callback(type(sink).__name__, error)
                errors.append(error)
        if errors and not self._disable_failed:
            raise errors[0]


class VisdomMetricSink:
    def __init__(
        self,
        *,
        env: str,
        server: str,
        port: int,
        base_url: str,
        save: bool = False,
        client: Any = None,
        warning_callback: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self.env = env
        self.save = save
        self._warning_callback = warning_callback
        self._window_traces: dict[str, list[str]] = {}
        self._skipped_metrics: set[str] = set()
        self._fallback_step = 0
        if client is None:
            try:
                import visdom
            except ImportError as error:
                raise RuntimeError(
                    "Visdom sink requested but the 'visdom' Python package is "
                    "not installed"
                ) from error
            client = visdom.Visdom(
                server=server,
                port=port,
                base_url=base_url,
                env=env,
            )
        self.client = client
        check_connection = getattr(self.client, "check_connection", None)
        if callable(check_connection) and not check_connection():
            raise RuntimeError(
                f"Visdom server is unreachable at {server}:{port}{base_url}"
            )

    def write(self, event: dict[str, Any]) -> None:
        name = str(event.get("name", "metric"))
        if self._skip_metric(name):
            return
        raw_value = event.get("value")
        value = self._finite_float(raw_value)
        if value is None:
            if self._is_non_finite_number(raw_value):
                return
            self._warn_skip_once(name, TypeError(f"non-numeric metric value: {raw_value!r}"))
            return
        x = event.get("iteration")
        x_value = self._finite_float(x)
        if x_value is None:
            self._fallback_step += 1
            x_value = float(self._fallback_step)
        base_name, trace_name = self._trace_for(name)
        window = self._window_for(base_name)
        update = "append" if window in self._window_traces else None
        legend = self._register_trace(window, trace_name)
        self.client.line(
            X=np.array([x_value]),
            Y=np.array([value]),
            win=window,
            name=trace_name,
            update=update,
            opts={
                "title": f"{self.env} {self._window_title(window)}",
                "xlabel": "iteration",
                "ylabel": self._window_ylabel(window, base_name),
                "legend": legend,
            },
        )

    def write_many(
        self,
        events: list[dict[str, Any]],
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int:
        traces: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        skipped = 0
        for event in events:
            name = str(event.get("name", "metric"))
            if self._skip_metric(name):
                skipped += 1
                continue
            raw_value = event.get("value")
            value = self._finite_float(raw_value)
            if value is None:
                if not self._is_non_finite_number(raw_value):
                    self._warn_skip_once(
                        name,
                        TypeError(f"non-numeric metric value: {raw_value!r}"),
                    )
                skipped += 1
                continue
            x_value = self._finite_float(event.get("iteration"))
            if x_value is None:
                self._fallback_step += 1
                x_value = float(self._fallback_step)
            base_name, trace_name = self._trace_for(name)
            window = self._window_for(base_name)
            traces.setdefault((window, trace_name, base_name), []).append(
                (x_value, value)
            )

        written = 0
        if progress_callback is not None and skipped:
            progress_callback(skipped)
        windows_seen_before = set(self._window_traces)
        for window, trace_name, _base_name in traces:
            self._register_trace(window, trace_name)
        windows_written: set[str] = set()
        for (window, trace_name, base_name), points in traces.items():
            update = (
                "append"
                if window in windows_seen_before or window in windows_written
                else None
            )
            self.client.line(
                X=np.array([point[0] for point in points]),
                Y=np.array([point[1] for point in points]),
                win=window,
                name=trace_name,
                update=update,
                opts={
                    "title": f"{self.env} {self._window_title(window)}",
                    "xlabel": "iteration",
                    "ylabel": self._window_ylabel(window, base_name),
                    "legend": self._window_traces[window],
                },
            )
            windows_written.add(window)
            written += len(points)
            if progress_callback is not None:
                progress_callback(len(points))
        return written

    def write_images(
        self,
        *,
        window: str,
        title: str,
        images: list[np.ndarray],
    ) -> None:
        if not images:
            return
        self.client.images(
            np.stack(images, axis=0),
            win=window,
            opts={"title": title, "jpgquality": 90},
        )

    def close(self) -> None:
        if self.save:
            save = getattr(self.client, "save", None)
            if callable(save):
                save([self.env])

    def _warn_skip_once(self, name: str, error: BaseException) -> None:
        if name in self._skipped_metrics:
            return
        self._skipped_metrics.add(name)
        if self._warning_callback is not None:
            self._warning_callback("VisdomMetricSink", error)

    def _register_trace(self, window: str, trace_name: str) -> list[str]:
        traces = self._window_traces.setdefault(window, [])
        if trace_name not in traces:
            traces.append(trace_name)
        return list(traces)

    @staticmethod
    def _skip_metric(name: str) -> bool:
        return name == "elapsed_time_ms" or name.startswith("test_")

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    @staticmethod
    def _is_non_finite_number(value: Any) -> bool:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return False
        return not math.isfinite(result)

    @staticmethod
    def _window_name(value: str) -> str:
        cleaned = "".join(char if char.isalnum() else "-" for char in value.strip())
        cleaned = "-".join(part for part in cleaned.split("-") if part)
        return cleaned or "loss"

    @staticmethod
    def _trace_for(name: str) -> tuple[str, str]:
        base, separator, suffix = name.rpartition("_test")
        if separator and suffix.isdigit() and base:
            return base, f"test{suffix}"
        return name, name

    @classmethod
    def _window_for(cls, name: str) -> str:
        normalized = name.lower()
        if "loss" in normalized:
            return f"loss-{cls._window_name(name)}"
        if normalized.startswith("map"):
            return f"metric-{cls._window_name(name)}"
        if normalized.startswith("fp"):
            return "metric-fp"
        if normalized in {"num_fg", "run_fg"}:
            return "metric-num-fg"
        if normalized in {"learning_rate", "lr"}:
            return "metric-learning-rate"
        if "time" in normalized or "duration" in normalized:
            return "metric-time"
        return f"metric-{cls._window_name(name)}"

    @staticmethod
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

    @staticmethod
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
