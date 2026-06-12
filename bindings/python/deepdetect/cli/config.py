from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_merge(*values: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if not value:
            continue
        for key, item in value.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(item, Mapping)
            ):
                result[key] = deep_merge(result[key], item)
            else:
                result[key] = copy.deepcopy(item)
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return data


def save_config(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            _plain_data(values),
            default_flow_style=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_overrides(values: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--set expects KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError("--set key must not be empty")
        parsed = yaml.safe_load(raw)
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            if not part:
                raise ValueError(f"invalid --set key: {key}")
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"cannot assign nested key below {part!r}")
        target[parts[-1]] = parsed
    return result


def cli_options(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    return copy.deepcopy(value)
