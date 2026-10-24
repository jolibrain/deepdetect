#!/usr/bin/env python3
"""Opt-in real RT-DETRv4 HTTP acceptance; requires local adapter/upstream/weights.

All paths must be accessible to both this process and the server. No training or
original checkpoint writes occur. The output directory must be new.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from deepdetect.pytorch_worker.sdk import WorkerContext
from pytorch_prediction_http import call


def run(args):
    torch.set_num_threads(2)
    repo = (
        Path(args.output_directory)
        if args.output_directory
        else Path(tempfile.mkdtemp(prefix="dd-rtdetr-http-"))
    )
    if args.output_directory:
        repo.mkdir(parents=True, exist_ok=False)
    repo = repo.resolve()
    adapter = Path(args.adapter).resolve()
    spec = importlib.util.spec_from_file_location("rtdetr_adapter", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = dict(
        nclasses=81,
        rtdetrv4=dict(
            repo_path=str(Path(args.upstream).resolve()),
            config_path="configs/rtv4/rtv4_hgnetv2_m_coco.yml",
            disable_teacher=True,
            pretrained_backbone=False,
        ),
    )
    worker = module.DeepDetectWorker()
    worker.configure(WorkerContext(str(repo), params, {}))
    model = worker.create_model(81, torch)
    worker._load_checkpoint_payload(torch, model, Path(args.checkpoint).resolve())
    torch.save({"model_state": model.state_dict()}, repo / "checkpoint-latest.pt")
    del model, worker
    wrapper = repo / "worker.py"
    wrapper.write_text(
        """import importlib.util
import json
from pathlib import Path
spec = importlib.util.spec_from_file_location("rtdetr_adapter", """
        + repr(str(adapter))
        + """)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class DeepDetectWorker(module.DeepDetectWorker):
    def create_model(self, nclasses, *backend):
        model = super().create_model(nclasses, *backend)
        def record(module, args):
            with (Path(self.context.repository) / "forwards.jsonl").open("a") as handle:
                handle.write(json.dumps(list(args[0].shape)) + "\\n")
        model.register_forward_pre_hook(record)
        return model
"""
    )
    params.update(
        entrypoint=str(wrapper), net={"test_batch_size": 2}, gpu=args.gpu, gpuid=[0]
    )
    image = Image.open(args.image).convert("RGB")
    paths = []
    for i, size in enumerate([image.size, (400, 240), (300, 500), (500, 350)]):
        path = repo / f"image-{i}.png"
        image.resize(size).save(path)
        paths.append(str(path))
    paths.insert(2, paths[0])
    service = "rtdetr_prediction_acceptance"
    call(
        args.url,
        "PUT",
        "/services/" + service,
        dict(
            mllib="pytorch",
            type="supervised",
            model={"repository": str(repo)},
            parameters={
                "input": dict(connector="image", width=640, height=640, rgb=True),
                "mllib": params,
            },
        ),
        201,
    )
    try:
        request = dict(
            service=service,
            data=paths,
            parameters={
                "mllib": {"predict_input": "connector"},
                "output": dict(bbox=True, confidence_threshold=0.25, best_bbox=10),
            },
        )
        predictions = call(args.url, "POST", "/predict", request)["body"]["predictions"]
        forwards = [
            json.loads(line)
            for line in (repo / "forwards.jsonl").read_text().splitlines()
        ]
        assert [shape[0] for shape in forwards] == [2, 2, 1], forwards
        assert [p["uri"] for p in predictions] == paths
        (repo / "batch-results.json").write_text(json.dumps(predictions, indent=2))
        worst_box = worst_score = 0
        for path, expected in zip(paths, predictions):
            actual = call(args.url, "POST", "/predict", dict(request, data=[path]))[
                "body"
            ]["predictions"][0]
            assert len(actual.get("classes", [])) == len(expected.get("classes", []))
            for a, b in zip(actual.get("classes", []), expected.get("classes", [])):
                assert a["cat"] == b["cat"]
                worst_score = max(worst_score, abs(a["prob"] - b["prob"]))
                av = [a["bbox"][k] for k in ("xmin", "ymin", "xmax", "ymax")]
                bv = [b["bbox"][k] for k in ("xmin", "ymin", "xmax", "ymax")]
                worst_box = max(worst_box, max(abs(x - y) for x, y in zip(av, bv)))
                np.testing.assert_allclose(av, bv, atol=0.02, rtol=1e-4)
                np.testing.assert_allclose(a["prob"], b["prob"], atol=1e-4, rtol=1e-4)
        report = dict(
            repository=str(repo),
            gpu=args.gpu,
            forward_shapes=forwards,
            detections=[len(p.get("classes", [])) for p in predictions],
            max_box_error=worst_box,
            max_score_error=worst_score,
        )
        (repo / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        call(args.url, "DELETE", "/services/" + service)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output-directory")
    run(parser.parse_args())
