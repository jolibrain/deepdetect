from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from training_observability.plots import metric_plot_key


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
        plot = metric_plot_key(name)
        window = plot.plot_id
        update = "append" if window in self._window_traces else None
        legend = self._register_trace(window, plot.trace)
        self.client.line(
            X=np.array([x_value]),
            Y=np.array([value]),
            win=window,
            name=plot.trace,
            update=update,
            opts={
                "title": f"{self.env} {plot.title}",
                "xlabel": "iteration",
                "ylabel": plot.ylabel,
                "legend": legend,
            },
        )

    def write_many(
        self,
        events: list[dict[str, Any]],
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int:
        traces: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
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
            plot = metric_plot_key(name)
            traces.setdefault(
                (plot.plot_id, plot.trace, plot.title, plot.ylabel), []
            ).append((x_value, value))

        written = 0
        if progress_callback is not None and skipped:
            progress_callback(skipped)
        windows_seen_before = set(self._window_traces)
        for window, trace_name, _title, _ylabel in traces:
            self._register_trace(window, trace_name)
        windows_written: set[str] = set()
        for (window, trace_name, title, ylabel), points in traces.items():
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
                    "title": f"{self.env} {title}",
                    "xlabel": "iteration",
                    "ylabel": ylabel,
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
