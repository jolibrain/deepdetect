from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import deep_merge, load_config, parse_overrides


def validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def parse_gpu_ids(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    ids: list[int] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.append(int(item))
            except ValueError as error:
                raise ValueError(f"GPU ids must be integers, got: {item}") from error
    if not ids:
        raise ValueError("--gpuid expects at least one GPU id")
    if -1 in ids and ids != [-1]:
        raise ValueError("--gpuid -1 cannot be combined with other ids")
    return ids


def normalize_gpu_options(
    options: dict[str, Any], *, gpu_disabled: bool = False
) -> None:
    gpuid = options.get("gpuid")
    if gpuid is None:
        return
    if gpu_disabled:
        raise ValueError("--gpuid requires GPU execution; remove --no-gpu")
    options["gpuid"] = normalize_gpu_ids(gpuid)
    options["gpu"] = True


def normalize_gpu_ids(value: Any) -> int | list[int]:
    if isinstance(value, bool):
        raise ValueError("gpuid must be an integer or a list of integers")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        ids = parse_gpu_ids([value])
    elif isinstance(value, (list, tuple)):
        ids = []
        for item in value:
            parsed = parse_gpu_ids([str(item)])
            if parsed is not None:
                ids.extend(parsed)
    else:
        raise ValueError("gpuid must be an integer or a list of integers")
    if ids is None:
        raise ValueError("gpuid must not be empty")
    if -1 in ids and ids != [-1]:
        raise ValueError("gpuid -1 cannot be combined with other ids")
    if len(ids) == 1:
        return ids[0]
    return ids


def resolve_options(
    defaults: dict[str, Any],
    args: Any,
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    config = load_config(args.config)
    overrides = parse_overrides(args.overrides)
    options = deep_merge(defaults, config, cli_values, overrides)
    for key in ("weights", "repository", "job_dir", "output", "train_data"):
        if key in options and options[key] is not None:
            options[key] = Path(options[key])
    if "test_data" in options and options["test_data"] is not None:
        options["test_data"] = normalize_test_data_paths(options["test_data"])
    return options


def normalize_test_data_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, (list, tuple)):
        paths = [Path(item) for item in value]
        if not paths:
            raise ValueError("test-data requires at least one path")
        return paths
    raise ValueError("test-data must be a path or a list of paths")
