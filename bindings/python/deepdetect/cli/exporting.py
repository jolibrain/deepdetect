"""Inference bundles, with optional post-export inference verification."""
from __future__ import annotations

import copy
from contextlib import contextmanager, redirect_stdout
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import deepdetect
import yaml

from .config import deep_merge, load_config, save_config
from .events import EventWriter
from .profiles import get_profile

EXPORT_VERSION = 1
# Lifecycle and optimizer settings are not part of the inference service contract.
TRAINING_KEYS = {
    "solver", "resume", "resume_from", "weights", "checkpoint", "python",
    "distributed", "data_source", "net", "gpu", "gpuid", "augmentation",
    "mirror", "rotate", "crop_size", "test_crop_samples", "cutout", "geometry",
    "noise", "distort", "class_weights", "test_predictions",
}
INPUT_KEYS = {
    "connector", "width", "height", "rgb", "bw", "segmentation", "bbox",
    "keypoints", "scale", "mean", "std", "channels", "keep_orig", "interpolation",
}


def set_field(data: dict[str, Any], key: str, value: Any, *, remove: bool = False) -> None:
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid configuration field: {key!r}")
    target = data
    for part in parts[:-1]:
        if remove and part not in target:
            return
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"configuration field is not a mapping: {key}")
    if remove:
        target.pop(parts[-1], None)
    else:
        target[parts[-1]] = copy.deepcopy(value)


def get_field(data: dict[str, Any], key: str) -> Any:
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing exported configuration field: {key}")
        value = value[part]
    return value


def relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("bundle paths must be nonempty relative strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay inside the export: {value}")
    return path


def resolve_export_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    """Resolve only explicitly declared bundle paths, before user overrides."""
    metadata = config.get("_export")
    if metadata is None:
        return config
    if not isinstance(metadata, dict) or metadata.get("version") != EXPORT_VERSION:
        raise ValueError("unsupported inference export version")
    fields = metadata.get("paths")
    if not isinstance(fields, list) or not all(isinstance(key, str) for key in fields):
        raise ValueError("export paths must be a list of configuration fields")
    result = copy.deepcopy(config)
    root = path.resolve().parent
    for key in fields:
        resolved = (root / relative_path(get_field(result, key))).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"export path escapes the bundle: {key}")
        if not resolved.exists():
            raise FileNotFoundError(f"export asset missing for {key}: {resolved}")
        set_field(result, key, str(resolved))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _select_checkpoint(source: Path, selection: str, backend: str) -> tuple[str, list[Path], dict[str, Any]]:
    extensions = (".pt",) if backend == "pytorch" else (".pt", ".npt", ".ptw")
    marker: dict[str, Any] = {}
    if selection == "best":
        path = source / "best_model.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                if key.strip() in marker:
                    raise ValueError(f"duplicate best model marker key: {key.strip()}")
                marker[key.strip()] = value.strip()
        iteration = str(marker.get("iteration", ""))
        if not re.fullmatch(r"[0-9]+", iteration):
            raise ValueError(f"best model marker has no valid iteration: {path}")
    elif selection == "latest":
        numbers = []
        for path in source.iterdir():
            match = re.fullmatch(r"checkpoint-([0-9]+)(\..+)", path.name)
            if match and match[2] in extensions and path.is_file():
                numbers.append(int(match[1]))
        iteration = str(max(numbers)) if numbers else "latest"
    elif re.fullmatch(r"[0-9]+", selection):
        iteration = str(int(selection))
    else:
        raise ValueError("--checkpoint must be best, latest, or a nonnegative iteration")
    files = [source / f"checkpoint-{iteration}{ext}" for ext in extensions
             if (source / f"checkpoint-{iteration}{ext}").is_file()]
    if not files:
        raise FileNotFoundError(f"selected checkpoint-{iteration} not found in {source}")
    return iteration, files, marker


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _inference_config(profile: Any, saved: dict[str, Any], source: Path) -> dict[str, Any]:
    options = deep_merge(profile.train_defaults(), saved)
    service = profile.service_parameters(options)
    # config.json captures the actual service input, including connector settings.
    persisted = _mapping(_read_json(source / "config.json").get("parameters", {}), "config.json parameters")
    service_input = deep_merge(service["input_parameters"], persisted.get("input"))
    train_mllib = deep_merge(profile.train_mllib, saved.get("augmentation"), saved.get("mllib"))
    mllib = deep_merge(service["mllib_parameters"], persisted.get("mllib"), saved.get("mllib"))
    for key in TRAINING_KEYS:
        mllib.pop(key, None)
    result = {key: copy.deepcopy(saved[key]) for key in profile.infer_defaults()
              if key in saved and key not in {
                  "weights", "repository", "service_name", "batch_size", "gpu", "gpuid",
                  "output", "output_format", "visualize", "benchmark", "warmup",
              }}
    result.update(nclasses=options["nclasses"], width=options["width"], height=options["height"],
                  service_name=f"python-{profile.name}-infer", batch_size=1, gpu=False,
                  service_mllib=mllib)
    if profile.backend == "torch" and int(train_mllib.get("crop_size", 0) or 0) > 0:
        result["width"] = result["height"] = int(train_mllib["crop_size"])
    # Preserve normalization from training for native connectors.
    training_input = deep_merge(profile.train_input, saved.get("input")) if profile.backend == "torch" else {}
    input_options = {key: value for key, value in deep_merge(training_input, service_input).items()
                     if key in INPUT_KEYS}
    input_options.update(width=result["width"], height=result["height"], db=False)
    result["service_input"] = input_options
    prediction = profile.predict_parameters(deep_merge(profile.infer_defaults(), result))
    result["predict_input"] = deep_merge(prediction["input_parameters"], input_options, saved.get("predict_input"))
    result["predict_output"] = deep_merge(prediction["output_parameters"], saved.get("predict_output"))
    return result


def _native_assets(source: Path, selected: list[Path]) -> list[Path]:
    assets: list[Path] = []
    for pattern in ("*corresp*", "*.prototxt"):
        matches = [path for path in source.glob(pattern) if path.is_file()]
        if len(matches) > 1:
            raise ValueError(f"ambiguous inference assets matching {pattern} in {source}")
        assets.extend(matches)
    if any(path.suffix == ".ptw" for path in selected) and not any(path.suffix == ".pt" for path in selected):
        candidates = [path for path in source.glob("*.pt")
                      if not path.name.startswith(("checkpoint-", "solver-"))]
        if len(candidates) != 1:
            raise ValueError("head checkpoint requires one unambiguous base .pt graph in the repository")
        assets.extend(candidates)
    if any(path.suffix == ".npt" for path in selected) and any(path.suffix == ".prototxt" for path in assets):
        raise ValueError("native checkpoint cannot be combined with a graph prototxt")
    return assets


def _strings(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value


def _stat(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_directory(stage: Path, destination: Path) -> None:
    """Linux atomic rename without replacing even an empty concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("atomic export publishing requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _test_inference(destination: Path, profile: Any) -> dict[str, Any]:
    """Exercise a relocated bundle without retaining service scratch files."""
    import numpy as np
    from PIL import Image

    from .options import normalize_gpu_options
    from .utils import configure_gpu_compatibility, predictions_by_uri

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deepdetect-export-test-") as directory:
        root = Path(directory)
        bundle = root / "model"
        shutil.copytree(destination, bundle)
        path = bundle / "inference.yaml"
        options = deep_merge(profile.infer_defaults(), resolve_export_config(load_config(path), path))
        normalize_gpu_options(options)
        options["batch_size"] = 1
        width, height = int(options["width"]), int(options["height"])
        service_parameters = profile.service_parameters(options)
        inputs = service_parameters["input_parameters"]
        channels = 1 if inputs.get("bw") or inputs.get("channels") == 1 else 3
        shape = (height, width) if channels == 1 else (height, width, channels)
        pixels = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        image = root / "random.png"
        Image.fromarray(pixels).save(image)
        prediction_parameters = profile.predict_parameters(options)
        if profile.task == "keypoint" and options["service_mllib"].get("vitpose", {}).get("head", "topdown") == "topdown":
            bbox = root / "bbox.txt"
            bbox.write_text(f"1 0 0 {width} {height}\n", encoding="utf-8")
            prediction_parameters["input_parameters"]["bbox_files"] = [str(bbox)]
        dd = deepdetect.DeepDetect()
        configure_gpu_compatibility(dd.build_info, requested=bool(options["gpu"]))
        with dd.create_service(options["service_name"], **service_parameters) as service:
            result = service.predict([image], **prediction_parameters)
            predictions = predictions_by_uri([image], result.get("predictions", []))
        return {
            "status": "passed", "input": {"kind": "random", "seed": 0,
                "width": width, "height": height, "channels": channels},
            "gpu": bool(options["gpu"]), "gpuid": options.get("gpuid"),
            "predictions": len(predictions), "elapsed_seconds": time.perf_counter() - started,
        }


@contextmanager
def _inference_logs_to_stderr():
    """Keep Python, native engine and worker stdout out of JSON reports."""
    sys.stdout.flush()
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


def _save_report(destination: Path, report: dict[str, Any]) -> None:
    # Replace rather than truncate: interrupted writes retain the previous status.
    temporary = destination / ".export.json.tmp"
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination / "export.json")
    finally:
        temporary.unlink(missing_ok=True)


def run_export(args: Any) -> int:
    if args.test_inference and args.dry_run:
        raise ValueError("--test-inference cannot be combined with --dry-run")
    source = args.source.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source repository not found: {source}")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("source and destination must not overlap")
    if destination.exists() or args.output.expanduser().is_symlink():
        raise FileExistsError(f"export destination already exists: {destination}")
    manifest = _read_json(source / "run.json")
    model = args.model or manifest.get("model")
    if not model:
        raise ValueError("profile missing from run.json; supply --model")
    if args.model and manifest.get("model") and args.model != manifest["model"]:
        raise ValueError("--model conflicts with run.json")
    profile = get_profile(model)
    if not profile.supports_training:
        raise ValueError(f"profile does not support training exports: {model}")
    config_path = (args.config or source / "config.yaml").expanduser().resolve()
    saved = load_config(config_path)
    metadata = _mapping(saved.get("export", {}), "export")
    unknown = set(metadata) - {"version", "module", "inference", "assets", "requirements"}
    if unknown or metadata.get("version", 1) != 1:
        raise ValueError(f"unsupported export metadata: {sorted(unknown)}")
    iteration, selected, marker = _select_checkpoint(source, args.checkpoint, profile.backend)
    config = _inference_config(profile, saved, source)
    # Top-level replacement lets workers drop opaque training-only mappings.
    config.update(copy.deepcopy(_mapping(metadata.get("inference", {}), "export.inference")))
    for expression in args.overrides:
        key, separator, value = expression.partition("=")
        if not separator:
            raise ValueError("--set expects KEY=VALUE")
        set_field(config, key, yaml.safe_load(value))
    for key in args.unset:
        set_field(config, key, None, remove=True)
    mllib = _mapping(config.get("service_mllib"), "service_mllib")
    module = args.worker_module or metadata.get("module") or mllib.get("module")
    if profile.backend == "pytorch":
        if not isinstance(module, str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", module):
            raise ValueError("export requires an installed worker module; declare export.module or use --worker-module")
        mllib["module"] = module
        mllib.pop("entrypoint", None)
    for key in ("weights", "checkpoint", "resume", "resume_from", "python", "gpu", "gpuid", "distributed"):
        mllib.pop(key, None)
    for key in ("resume", "export"):
        config.pop(key, None)
    allowed = set(profile.infer_defaults()) | {"service_mllib", "service_input", "predict_input", "predict_output"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown inference configuration fields: {sorted(unknown)}")
    config["weights"] = selected[0].name
    config["repository"] = "."
    config["_export"] = {"version": EXPORT_VERSION, "profile": model, "paths": ["repository", "weights"]}

    files: dict[str, Path] = {}
    def add_file(path: Path, name: str) -> None:
        relative_path(name)
        if not Path(name).parts or Path(name).parts[0] in {"inference.yaml", "export.json"} or name in files:
            raise ValueError(f"export destination collision: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"export asset is not a file: {path}")
        files[name] = path

    for path in selected:
        add_file(path, path.name)
    if profile.backend == "torch":
        for path in _native_assets(source, selected):
            add_file(path, path.name)
    assets = metadata.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("export.assets must be a list")
    assets = copy.deepcopy(assets)
    for expression in args.asset:
        key, separator, value = expression.partition("=")
        if not separator or not value:
            raise ValueError("--asset expects CONFIG_KEY=SOURCE")
        assets.append({"key": key, "source": str(Path(value).expanduser().resolve())})
    for index, asset in enumerate(assets):
        asset = _mapping(asset, "export asset")
        if set(asset) - {"key", "source", "destination"} or not isinstance(asset.get("key"), str):
            raise ValueError("asset requires key and source, with optional destination")
        key = asset["key"]
        if key in {"repository", "weights"} or key.startswith("_export"):
            raise ValueError(f"reserved asset field: {key}")
        if key in config["_export"]["paths"]:
            raise ValueError(f"duplicate asset field: {key}")
        raw = asset.get("source")
        if not isinstance(raw, str) or not raw:
            raise ValueError("asset source must be a nonempty path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        name = relative_path(asset.get("destination", f"assets/{index}/{path.name}"))
        if not path.exists():
            raise FileNotFoundError(f"export asset missing: {path}")
        if path.is_dir():
            # Reject symlink directories (including cycles); ordinary file symlinks
            # are dereferenced into standalone copies below.
            for child in sorted(path.rglob("*")):
                if child.is_symlink() and child.is_dir():
                    raise ValueError(f"asset contains a symlink directory: {child}")
                if child.is_file():
                    add_file(child, (name / child.relative_to(path)).as_posix())
            if not any(item.startswith(name.as_posix() + "/") for item in files):
                raise ValueError(f"asset directory is empty: {path}")
        else:
            add_file(path, name.as_posix())
        set_field(config, key, name.as_posix())
        config["_export"]["paths"].append(key)
    for name in files:
        if any(parent.as_posix() in files for parent in Path(name).parents):
            raise ValueError(f"export file/directory collision: {name}")
    # Validate path fields even for dry runs, before any file is copied.
    for key in config["_export"]["paths"]:
        relative_path(get_field(config, key))
    # Catch visible machine-local dependencies without interpreting opaque formats.
    for key, value in _strings({key: value for key, value in config.items() if key != "_export"}):
        if key in config["_export"]["paths"]:
            continue
        if Path(value).is_absolute() or value.startswith("~/"):
            raise ValueError(f"unbundled path in {key}: {value}; use --asset or --unset/--set")
        if "/" in value and ((config_path.parent / value).exists() or Path(value).exists()):
            raise ValueError(f"unbundled local path in {key}: {value}; declare an asset")
    # Validate generated options without importing the worker or loading weights.
    options = deep_merge(profile.infer_defaults(), config)
    for key in ("nclasses", "width", "height", "batch_size"):
        if int(options[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    profile.service_parameters(options)
    profile.predict_parameters(options)
    requirements = metadata.get("requirements", [])
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        raise ValueError("export.requirements must be a list of package requirements")
    if not requirements and module:
        requirements = sorted(importlib.metadata.packages_distributions().get(module.split(".")[0], []))
    versions = {"deepdetect": deepdetect.__version__}
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if not match:
            raise ValueError(f"invalid package requirement: {requirement}")
        try:
            versions[match[0]] = importlib.metadata.version(match[0])
        except importlib.metadata.PackageNotFoundError:
            versions[match[0]] = None
    snapshots = {name: _stat(path) for name, path in files.items()}
    report = {
        "version": EXPORT_VERSION, "profile": model, "backend": profile.backend,
        "selection": args.checkpoint, "iteration": iteration, "best_model": marker,
        "worker": {key: mllib[key] for key in ("module", "class") if key in mllib},
        "requirements": requirements, "installed_versions": versions,
        "files": [{"path": name, "size": snapshots[name][2]} for name in sorted(files)],
    }
    if args.test_inference:
        report["inference_test"] = {"status": "pending"}
    writer = EventWriter(output_format=args.output_format)
    if args.dry_run:
        writer.emit("export_planned", status="planned", destination=str(destination), **report)
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for record in report["files"]:
            name = record["path"]
            path = files[name]
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if _stat(path) != snapshots[name]:
                raise RuntimeError(f"source changed during export: {path}")
            shutil.copyfile(path, target)
            record["sha256"] = _sha256(target)
        if any(_stat(files[name]) != state for name, state in snapshots.items()):
            raise RuntimeError("source files changed during export; retry after checkpoint writing completes")
        save_config(stage / "inference.yaml", config)
        resolve_export_config(config, stage / "inference.yaml")
        report["files"].append({"path": "inference.yaml", "size": (stage / "inference.yaml").stat().st_size,
                                "sha256": _sha256(stage / "inference.yaml")})
        _save_report(stage, report)
        if destination.exists():
            raise FileExistsError(f"export destination already exists: {destination}")
        _publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if args.test_inference:
        if args.output_format == "text":
            print("Testing exported model inference on random input...", flush=True)
        try:
            with _inference_logs_to_stderr():
                report["inference_test"] = _test_inference(destination, profile)
        except Exception as error:
            report["inference_test"] = {"status": "failed", "error": str(error)}
            _save_report(destination, report)
            writer.emit("export_test_failed", status="failed", destination=str(destination), **report)
            print(f"Inference test failed: {error}", file=sys.stderr)
            return 1
        _save_report(destination, report)
    writer.emit("export_finished", status="success", destination=str(destination), **report)
    return 0
