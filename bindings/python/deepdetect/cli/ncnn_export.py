"""Offline conversion of a selected inference graph into a portable NCNN service."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from .config import deep_merge, load_config, save_config
from .events import EventWriter
from .exporting import (
    EXPORT_VERSION, _inference_config, _inference_logs_to_stderr,
    _publish_directory, _read_json, _save_report, _select_checkpoint,
    _sha256, _stat, _test_inference, relative_path, set_field,
)
from .profiles import get_profile

RELEASE = "20261020"
# Executables from the tagged upstream Linux and Linux-aarch64 release archives.
PNNX_DIGESTS = {
    "4db234cb90a27dc15e04e0d0edf5706c7a488ab8e6235cf8c146c6ec80fd9355",
    "1f55df74080eabb048cb122215f290d16e1bf82309c1a67662c91c437e96471e",
}


def _check_output(shape, fmt, classes):
    if fmt == "segmentation":
        if len(shape) == 4 and shape[0] in (1, None):
            shape = shape[1:]
        if (len(shape) != 3 or shape[0] != classes or
                any(side is not None and side <= 0 for side in shape[1:])):
            raise ValueError(f"expected segmentation logits [classes,height,width], got {shape}")
        return
    expected_rank = 2 if fmt in ("yolox", "ssd", "rtdetrv4") else 1
    if len(shape) == expected_rank + 1 and shape[0] in (None, 1):
        shape = shape[1:]
    if fmt == "yolox":
        if len(shape) != 2 or shape[-1] != 5 + classes - 1:
            raise ValueError("expected decoded YOLOX rows of 5 + foreground classes")
    elif fmt in ("ssd", "rtdetrv4"):
        if len(shape) != 2 or (shape[-1] != 6 and not (fmt == "rtdetrv4" and shape[-1] is None)):
            raise ValueError("expected detection output with six columns")
    elif len(shape) != 1 or shape[0] != classes:
        raise ValueError(f"classification output must contain {classes} scores, got {shape}")


def _prepare_graph(source, work, shape, fmt, classes):
    if source.suffix.lower() == ".onnx":
        import onnx
        from onnx import external_data_helper as external

        # Resolve external tensors next to the original graph, before staging it
        # under a new basename for PNNX.
        graph = onnx.load(source, load_external_data=False)
        base = source.parent.resolve()
        for tensor in external._get_all_tensors(graph):
            if external.uses_external_data(tensor):
                location = external.ExternalDataInfo(tensor).location
                if not (base / relative_path(location)).resolve().is_relative_to(base):
                    raise ValueError("external ONNX weights must be inside the model directory")
        external.load_external_data_for_model(graph, str(base))
        onnx.checker.check_model(graph)
        initializers = {v.name for v in graph.graph.initializer}
        inputs = [v for v in graph.graph.input if v.name not in initializers]
        outputs = graph.graph.output
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("NCNN image export requires one input and one output")
        def dimensions(value):
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT:
                raise ValueError("NCNN export expects FP32 graph input and output")
            return [d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim]
        dims = dimensions(inputs[0])
        if len(dims) != 4 or any(d is not None and d != s for d, s in zip(dims, shape)):
            raise ValueError(f"ONNX input shape {dims} conflicts with {shape}")
        _check_output(dimensions(outputs[0]), fmt, classes)
        target = work / "graph.onnx"
        onnx.save(graph, target)
        digest = _sha256(target)
        if fmt != "rtdetrv4":
            from .ncnn_topk import keep_topk_outputs
            if keep_topk_outputs(graph):
                onnx.save(graph, target)
        return target, digest
    elif source.suffix.lower() in (".pt", ".torchscript", ".ts"):
        import torch

        target = work / "graph.pt"
        shutil.copyfile(source, target)
        model = torch.jit.load(str(target), map_location="cpu").eval()
        with torch.inference_mode():
            out = model(torch.zeros(shape))
        if not isinstance(out, torch.Tensor) or out.dtype != torch.float32:
            raise ValueError("export an inference wrapper returning one FP32 tensor")
        _check_output(list(out.shape), fmt, classes)
    else:
        raise ValueError("expected a standalone ONNX or TorchScript inference graph")
    return target, _sha256(target)


def _run_graph(run, selected, identities, iteration, backend, saved, config, work):
    state = _read_json(run / "onnx/state.json")
    if state.get("bundle"):
        root = (run / "onnx").resolve()
        bundle = (root / relative_path(state["bundle"])).resolve()
        if not bundle.is_relative_to(root):
            raise ValueError("invalid ONNX bundle path")
        metadata = _read_json(bundle / "metadata.json")
        if metadata.get("checkpoints") == identities and (bundle / "model.onnx").is_file():
            return bundle / "model.onnx"

    service = dict(config.get("service_mllib", {}), nclasses=config["nclasses"])
    effective = deep_merge(saved.get("mllib"), service)
    settings = effective.get("onnx_export", {})
    if not settings.get("provider"):
        if backend == "torch" and len(selected) == 1 and selected[0].suffix == ".pt":
            return selected[0]
        raise ValueError("selected checkpoint needs a matching ONNX export or an onnx_export.provider")
    copies = []
    for source, identity in zip(selected, identities):
        copy = work / source.name
        shutil.copyfile(source, copy)
        if _sha256(copy) != identity["sha256"]:
            raise RuntimeError("checkpoint changed during export")
        copies.append(str(copy))
    request = {
        "backend": backend, "iteration": iteration, "config": settings,
        "checkpoints": copies, "checkpoint_identity": identities,
        "inference": {"service": {"repository": str(run), "mllib": service},
                      "mllib": effective,
                      "parameters": {"input": config["service_input"], "mllib": effective},
                      "width": config["width"], "height": config["height"]},
    }
    request_path = work / "provider.json"
    request_path.write_text(json.dumps(request))
    python = service.get("python") or sys.executable
    try:
        subprocess.run([python, "-m", "deepdetect.onnx_export", str(request_path)],
                       stdout=sys.stderr, check=True, timeout=settings.get("timeout_seconds", 120))
    except subprocess.SubprocessError as error:
        raise ValueError(f"ONNX provider failed for checkpoint {iteration}: {error}") from error
    return work / "bundle/model.onnx"


def _convert(graph, binary, shapes, stage, output_shape=None):
    from .ncnn_topk import lower_topk
    inputshape = ",".join("[" + ",".join(map(str, shape)) + "]" for shape in shapes)
    log = subprocess.run(
        [binary, graph.name, f"inputshape={inputshape}", "fp16=0",
         "device=cpu", "ncnnparam=model.ncnn.param", "ncnnbin=model.ncnn.bin"],
        cwd=graph.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    diagnostics = [line for line in log.stdout.splitlines()
                   if "unsupported" in line.lower() or "not supported" in line.lower()]
    if log.returncode or diagnostics:
        raise ValueError("PNNX conversion failed: " + ("\n".join(diagnostics) or log.stdout[-2000:]))
    params = graph.parent / "model.ncnn.param"
    lower_topk(params, graph.with_suffix(".pnnx.param"), shapes[0], output_shape)
    lines = params.read_text().splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise ValueError("PNNX did not emit a valid NCNN graph")
    for line in lines[2:]:
        op = line.split()[0]
        if "." in op or "::" in op or op == "GatherElements":
            raise ValueError(f"operator {op} could not be lowered to NCNN")
    for name in ("model.ncnn.param", "model.ncnn.bin"):
        if graph.parent != stage:
            shutil.move(graph.parent / name, stage / name)


def _portable_config(config):
    fmt = config.get("service_mllib", {}).get("ncnn_output")
    if fmt not in ("logits", "probabilities", "yolox", "ssd", "rtdetrv4", "segmentation"):
        raise ValueError("set service_mllib.ncnn_output to logits, probabilities, yolox, ssd, rtdetrv4 or segmentation")
    if fmt == "segmentation":
        profile = get_profile("ncnn-segmenter")
    elif fmt in ("yolox", "ssd", "rtdetrv4"):
        profile = get_profile("ncnn-detector")
    else:
        profile = get_profile("ncnn-classifier")
    dimensions = ("width", "height", "nclasses")
    if any(key not in config for key in dimensions):
        raise ValueError("NCNN export requires width, height and nclasses")
    width, height, classes = (int(config[key]) for key in dimensions)
    if min(width, height, classes) <= 0:
        raise ValueError("width, height and nclasses must be positive")
    inputs = deep_merge(config.get("service_input"), config.get("predict_input"))
    if any(inputs.get(k) for k in ("crop_width", "crop_height", "scaled", "aspect_ratio_pad")):
        raise ValueError("NCNN bundle export supports stretch or top-left letterbox resizing")
    image = {k: v for k, v in inputs.items()
             if k in ("rgb", "bw", "mean", "std", "scale", "interp", "ncnn_letterbox")}
    if "interpolation" in inputs and "interp" not in image:
        image["interp"] = inputs["interpolation"]
    if "interp" in image and image["interp"] not in ("nearest", "linear", "cubic", "area", "lanczos4"):
        raise ValueError("unknown OpenCV interpolation")
    image.update(connector="image", width=width, height=height, ncnn_normalize=True)
    channels = int(inputs.get("channels", 1 if image.get("bw") else 3))
    if channels not in (1, 3) or (image.get("bw") and channels != 1):
        raise ValueError("expected a grayscale or a three-channel image")
    if channels == 1:
        image["bw"] = True
    for key in ("mean", "std"):
        values = image.get(key, [])
        if values and (len(values) != channels or not all(math.isfinite(v) for v in values)):
            raise ValueError(f"{key} must have one finite value per channel")
    if any(v <= 0 for v in image.get("std", [])) or not 0 < image.get("scale", 1) < math.inf:
        raise ValueError("NCNN normalization requires positive scale and std")
    if fmt == "yolox" and classes < 2:
        raise ValueError("YOLOX nclasses includes background and at least one foreground class")
    shape = [1, channels, height, width]
    mllib = {"ncnn_output": fmt, "datatype": "fp32"}
    if fmt == "segmentation":
        resize = config["service_mllib"].get("ncnn_segmentation_resize", "class_ids")
        if resize not in ("class_ids", "logits"):
            raise ValueError("ncnn_segmentation_resize must be class_ids or logits")
        if image.get("ncnn_letterbox"):
            raise ValueError("NCNN segmentation requires stretch resize")
        mllib["ncnn_segmentation_resize"] = resize
    if "threads" in config["service_mllib"]:
        mllib["threads"] = int(config["service_mllib"]["threads"])
        if mllib["threads"] < 1:
            raise ValueError("NCNN threads must be positive")
    prediction = {k: v for k, v in (config.get("predict_output") or {}).items()
                  if k in ("best", "confidence_threshold", "nms_threshold", "best_bbox")}
    if fmt == "segmentation" and "confidences" in (config.get("predict_output") or {}):
        prediction["confidences"] = config["predict_output"]["confidences"]
    if profile.task == "detection":
        prediction["bbox"] = True
    threshold = config.get("confidence_threshold", prediction.get("confidence_threshold"))
    if threshold is None:
        threshold = 0.0 if fmt == "segmentation" else 0.25
    portable = dict(repository=".", width=width, height=height, nclasses=classes, gpu=False,
                    service_input=image, service_mllib=mllib, predict_output=prediction,
                    confidence_threshold=threshold,
                    _export={"version": EXPORT_VERSION, "profile": profile.name, "paths": ["repository"]})
    return portable, profile, shape


def export_ncnn(args):
    source = args.source.expanduser().resolve()
    dest = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() or args.output.is_symlink() or dest.is_relative_to(source) or source.is_relative_to(dest):
        raise ValueError("NCNN export requires a new destination outside the source")
    if args.asset or args.worker_module:
        raise ValueError("use the worker ONNX provider and --labels for NCNN exports")
    run = source.is_dir()
    if not run and (args.model or args.checkpoint != "best"):
        raise ValueError("--model and --checkpoint select checkpoints in a training repository")
    if not run and not args.config:
        raise ValueError("standalone graph export requires --config")
    saved = load_config(args.config or source / "config.yaml")
    selected, iteration, backend = [], None, None
    if run:
        recorded = _read_json(source / "run.json").get("model")
        if args.model and recorded and args.model != recorded:
            raise ValueError("--model conflicts with run.json")
        model = args.model or recorded
        if not model:
            raise ValueError("profile missing from run.json; supply --model")
        profile = get_profile(model)
        if not profile.supports_training:
            raise ValueError(f"profile does not support training exports: {model}")
        backend = profile.backend
        iteration, selected, _ = _select_checkpoint(source, args.checkpoint, backend)
        config = _inference_config(profile, saved, source)
        if profile.name == "segformer":
            config["service_mllib"].setdefault("ncnn_output", "segmentation")
        # Fold prediction-time preprocessing into the portable service defaults.
        config["service_input"] = deep_merge(config["service_input"], config.pop("predict_input"))
        config = deep_merge(config, (saved.get("export") or {}).get("inference"))
    else:
        config = dict(saved)
    for override in args.overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("--set expects KEY=VALUE")
        set_field(config, key, yaml.safe_load(value))
    for key in args.unset:
        set_field(config, key, None, remove=True)
    portable, profile, shape = _portable_config(config)
    labels = args.labels
    if labels is None and run:
        matches = [p for p in source.glob("*corresp*") if p.is_file()]
        if len(matches) > 1:
            raise ValueError("ambiguous label file; specify --labels")
        labels = next(iter(matches), None)
    if labels is not None and not labels.is_file():
        raise FileNotFoundError(labels)
    identities = [{"name": p.name, "sha256": _sha256(p)} for p in selected]
    report = {"version": EXPORT_VERSION, "profile": profile.name, "backend": "ncnn",
              "converter": RELEASE, "precision": "fp32", "input_shape": shape,
              "source": source.name, "iteration": iteration, "checkpoints": identities,
              "selection": args.checkpoint if run else "standalone", "files": []}
    writer = EventWriter(output_format=args.output_format)
    if args.dry_run:
        writer.emit("export_planned", status="planned", destination=str(dest), **report)
        return 0
    binary = args.pnnx or shutil.which("pnnx")
    if not binary or not Path(binary).is_file() or _sha256(Path(binary)) not in PNNX_DIGESTS:
        raise ValueError(f"install the official Linux PNNX {RELEASE} executable or specify --pnnx")
    binary = str(Path(binary).resolve())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".ncnn-") as directory:
        work = Path(directory)
        stage = work / "publish"
        stage.mkdir()
        graph = source
        if run:
            config["service_input"] = portable["service_input"]
            graph = _run_graph(source, selected, identities, iteration, backend, saved, config, work)
        snapshot = _stat(graph)
        mllib = portable["service_mllib"]
        classes = portable["nclasses"]
        prepared, graph_digest = _prepare_graph(graph, work, shape, mllib["ncnn_output"], classes)
        if _stat(graph) != snapshot:
            raise RuntimeError("source graph changed during export")
        report["graph_sha256"] = graph_digest
        report["pnnx_sha256"] = _sha256(Path(binary))
        if mllib["ncnn_output"] == "rtdetrv4":
            from .ncnn_rtdetrv4 import convert_rtdetrv4
            convert_rtdetrv4(prepared, binary, shape, classes, stage, _convert)
        else:
            output_shapes = {"ssd": ["0=6", "1=-1"], "yolox": [f"0={4 + classes}", "1=-1"],
                             "segmentation": ["0=0", "1=0", f"2={classes}"]}
            _convert(prepared, binary, [shape], stage, output_shapes.get(mllib["ncnn_output"], []))
        if labels:
            shutil.copyfile(labels, stage / "corresp.txt")
        save_config(stage / "inference.yaml", portable)
        (stage / "service.json").write_text(json.dumps({
            "mllib": "ncnn", "model": {"repository": "."},
            "parameters": {"input": portable["service_input"], "mllib": dict(mllib, nclasses=classes)}
        }, indent=2) + "\n")
        if any(_sha256(p) != identity["sha256"] for p, identity in zip(selected, identities)):
            raise RuntimeError("selected checkpoint changed during export")
        report["files"] = [{"path": p.name, "size": p.stat().st_size, "sha256": _sha256(p)}
                           for p in sorted(stage.iterdir())]
        if args.test_inference:
            with _inference_logs_to_stderr():
                report["inference_test"] = _test_inference(stage, profile)
        _save_report(stage, report)
        _publish_directory(stage, dest)
    writer.emit("export_finished", status="success", destination=str(dest), **report)
    return 0
