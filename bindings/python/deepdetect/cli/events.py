from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from typing import Any, TextIO


class EventWriter:
    def __init__(
        self,
        *,
        output_format: str = "jsonl",
        stream: TextIO | None = None,
        collect_events: bool = True,
    ):
        self.output_format = output_format
        self.stream = stream or sys.stdout
        self.collect_events = collect_events
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"event": event, "timestamp": time.time(), **payload}
        if self.collect_events:
            self.events.append(record)
        if self.output_format == "json":
            print(json.dumps(record, sort_keys=True), file=self.stream)
        elif self.output_format == "jsonl":
            print(json.dumps(record, sort_keys=True), file=self.stream)
        else:
            self._emit_text(record)
        self.stream.flush()
        return record

    def close(self) -> None:
        return None

    def _emit_text(self, record: dict[str, Any]) -> None:
        event = record.get("event", "event")
        if event == "training_status":
            measure = record.get("measure") or {}
            iteration = measure.get("iteration", "?")
            loss = measure.get("train_loss", "?")
            status = record.get("status", "?")
            print(
                f"training_status status={status} iteration={iteration} "
                f"train_loss={loss}",
                file=self.stream,
            )
            return
        if event == "prediction":
            print(
                f"prediction image={record.get('image')} "
                f"time_ms={record.get('time_ms')}",
                file=self.stream,
            )
            return
        if event == "benchmark":
            print(
                f"benchmark images={record.get('images')} "
                f"avg_ms_per_image={record.get('avg_ms_per_image')}",
                file=self.stream,
            )
            return
        print(f"{event}: {json.dumps(record, sort_keys=True)}", file=self.stream)


def metric_events(status: dict[str, Any]) -> list[dict[str, Any]]:
    measure = status.get("measure")
    if not isinstance(measure, dict):
        return []
    iteration = measure.get("iteration")
    events = []
    for name, value in measure.items():
        if _is_iteration_metric(name):
            continue
        events.append({"name": name, "value": value, "iteration": iteration})
    return events


class MetricEventExtractor:
    def __init__(self) -> None:
        self._history_iterations: dict[str, float] = {}

    def prime(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            name = event.get("name")
            if not isinstance(name, str):
                continue
            iteration = _float_or_default(event.get("iteration"), -1.0)
            self._history_iterations[name] = max(
                self._history_iterations.get(name, -1.0),
                iteration,
            )

    def events(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        history_events = self._history_events(status)
        if history_events:
            return history_events
        if isinstance(status.get("measure_hist"), dict):
            return []
        return metric_events(status)

    def _history_events(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        measure_hist = status.get("measure_hist")
        if not isinstance(measure_hist, dict):
            return []
        measure_sampling = status.get("measure_sampling")
        if not isinstance(measure_sampling, dict):
            measure_sampling = {}

        events = []
        for hist_name, values in measure_hist.items():
            if not hist_name.endswith("_hist") or not _is_sequence(values):
                continue
            name = hist_name[:-5]
            if _is_iteration_metric(name):
                continue
            sampling = _positive_int(measure_sampling.get(f"{name}_sampling"), default=1)
            iteration_values = _matching_iteration_history(name, values, measure_hist)
            last_iteration = self._history_iterations.get(name, -1.0)
            newest_iteration = last_iteration
            for index, value in enumerate(values):
                iteration = (
                    _float_or_default(iteration_values[index], float(index * sampling))
                    if iteration_values is not None
                    else float(index * sampling)
                )
                if iteration <= last_iteration:
                    continue
                events.append(
                    {
                        "name": name,
                        "value": value,
                        "iteration": iteration,
                    }
                )
                newest_iteration = iteration
            self._history_iterations[name] = newest_iteration
        return events


def _is_iteration_metric(name: Any) -> bool:
    return isinstance(name, str) and (
        name == "iteration" or name.startswith("iteration_")
    )


def _matching_iteration_history(
    name: str,
    values: Sequence[Any],
    measure_hist: dict[str, Any],
) -> Sequence[Any] | None:
    candidate_names = []
    suffix = _test_suffix(name)
    if suffix:
        candidate_names.append(f"iteration_{suffix}_hist")
    candidate_names.append("iteration_hist")

    for candidate_name in candidate_names:
        candidate = measure_hist.get(candidate_name)
        if _is_sequence(candidate) and len(candidate) == len(values):
            return candidate
    return None


def _test_suffix(name: str) -> str | None:
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    suffix = parts[1]
    return suffix if suffix.startswith("test") else None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _positive_int(value: Any, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
