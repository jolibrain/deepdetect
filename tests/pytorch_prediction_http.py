#!/usr/bin/env python3
"""Run against a fresh HTTP server; paths are in the server's filesystem.

Docker CI executes this script inside the image; no pytest dependency required.
"""

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import base64
import json
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

from PIL import Image


@contextmanager
def image_url(path):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(Path(path).parent))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{Path(path).name}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def call(url, method, route, body=None, expected=200):
    request = urllib.request.Request(
        url + route,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        result = json.load(error)
    assert result["status"]["code"] == expected, result
    return result


def run(url, gpu=False):
    fixture = str(Path(__file__).with_name("pytorch_prediction_worker.py").resolve())
    with tempfile.TemporaryDirectory(prefix="dd-predict-http-") as root:
        root = Path(root)
        paths = []
        for i, size in enumerate([(6, 4), (3, 7), (8, 2), (5, 5)]):
            path = root / f"{i}.png"
            Image.new("RGB", size, (204, 102, 51)).save(path)
            paths.append(str(path))
        paths.insert(2, paths[0])
        for worker in [
            "DeepDetectWorker",
            "SemanticWorker",
            "BrokenWorker",
            "MisassociatedWorker",
            "TimeoutWorker",
        ]:
            service = "batch_" + worker.lower()
            repo = root / worker
            repo.mkdir()
            shm = repo / "shm"
            mllib = dict(
                entrypoint=fixture,
                **{"class": worker},
                task="segmentation" if worker == "SemanticWorker" else "detection",
                nclasses=2,
                gpu=gpu,
                net={"test_batch_size": 2},
                connector_shared_memory_dir=str(shm),
            )
            call(
                url,
                "PUT",
                "/services/" + service,
                dict(
                    mllib="pytorch",
                    type="supervised",
                    model={"repository": str(repo)},
                    parameters={
                        "input": dict(connector="image", width=4, height=4, rgb=True),
                        "mllib": mllib,
                    },
                ),
                201,
            )
            try:
                request = dict(
                    service=service,
                    data=paths,
                    parameters={"output": {"bbox": worker != "SemanticWorker"}},
                )
                if worker in ("BrokenWorker", "MisassociatedWorker", "TimeoutWorker"):
                    response = urllib.request.Request(
                        url + "/predict",
                        data=json.dumps(request).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    try:
                        urllib.request.urlopen(response)
                        raise AssertionError("partial batch failure was accepted")
                    except urllib.error.HTTPError:
                        pass
                    assert not list(shm.glob("session-*"))
                    continue
                predictions = call(url, "POST", "/predict", request)["body"][
                    "predictions"
                ]
                forwards = [
                    json.loads(line)
                    for line in (repo / "forwards.jsonl").read_text().splitlines()
                ]
                assert [f["shape"][0] for f in forwards] == [2, 2, 1], forwards
                assert [p["uri"] for p in predictions] == paths, predictions
                for forward in forwards:
                    assert forward["shape"][1:] == [3, 4, 4]
                    for pixel in forward["pixels"]:
                        assert all(
                            abs(a - b) < 1e-6 for a, b in zip(pixel, [0.8, 0.4, 0.2])
                        ), pixel
                assert not list(shm.glob("session-*"))
                for path, prediction in zip(paths, predictions):
                    width, height = Image.open(path).size
                    if worker == "SemanticWorker":
                        assert prediction["imgsize"] == dict(width=width, height=height)
                        assert prediction["vals"] == [1] * (width * height)
                        assert all(
                            abs(v - 0.8) < 1e-6 for v in prediction["confidences"]["1"]
                        )
                    else:
                        assert prediction["classes"][0]["bbox"] == dict(
                            xmin=0, ymin=0, xmax=width, ymax=height
                        ), prediction
                    single = call(url, "POST", "/predict", dict(request, data=[path]))[
                        "body"
                    ]["predictions"][0]
                    assert {k: v for k, v in single.items() if k != "last"} == {
                        k: v for k, v in prediction.items() if k != "last"
                    }
                override_batch = dict(
                    request,
                    parameters={
                        "mllib": {"net": {"test_batch_size": 3}},
                        "output": request["parameters"]["output"],
                    },
                )
                named = call(
                    url,
                    "POST",
                    "/predict",
                    dict(override_batch, ids=["a", "b", "c", "d", "e"]),
                )["body"]["predictions"]
                assert [p["uri"] for p in named] == ["a", "b", "c", "d", "e"]
                recent = [
                    json.loads(line)
                    for line in (repo / "forwards.jsonl").read_text().splitlines()[-2:]
                ]
                assert [f["shape"][0] for f in recent] == [3, 2]
                if worker == "DeepDetectWorker":
                    empty = call(
                        url,
                        "POST",
                        "/predict",
                        dict(
                            request,
                            data=[paths[0]],
                            parameters={
                                "output": {"bbox": True, "confidence_threshold": 1.0}
                            },
                        ),
                    )["body"]["predictions"][0]
                    assert not empty.get("classes"), empty
                    call(
                        url,
                        "POST",
                        "/predict",
                        dict(
                            request,
                            data=[paths[0]],
                            parameters={
                                "mllib": {"predict_input": "paths"},
                                "output": {"bbox": True},
                            },
                        ),
                    )
                    last = json.loads(
                        (repo / "forwards.jsonl").read_text().splitlines()[-1]
                    )
                    assert last["shape"] == [1, 3, 4, 6], last
                with image_url(paths[0]) as source_url:
                    remote = call(
                        url, "POST", "/predict", dict(request, data=[source_url])
                    )["body"]["predictions"][0]
                    assert remote["uri"] == source_url
                # Inline transport, explicit normalization/color overrides, base64 identity.
                encoded = base64.b64encode(Path(paths[0]).read_bytes()).decode()
                override = dict(
                    request,
                    data=[encoded],
                    parameters={
                        "input": {
                            "rgb": False,
                            "scale": 0.5,
                            "mean": [1.0, 2.0, 3.0],
                            "std": [2.0, 4.0, 8.0],
                        },
                        "mllib": {"connector_tensor_transport": "inline"},
                    },
                )
                call(url, "POST", "/predict", override)
                last = json.loads(
                    (repo / "forwards.jsonl").read_text().splitlines()[-1]
                )
                assert last["pixels"][0] == [12.25, 12.25, 12.375], last
                for bad in [
                    {"width": 0},
                    {"std": [0.0]},
                    {"crop_width": 2},
                    {"scaled": True},
                    {"aspect_ratio_pad": True},
                ]:
                    call(
                        url,
                        "POST",
                        "/predict",
                        dict(request, parameters={"input": bad}),
                        expected=400,
                    )
                    assert not list(shm.glob("session-*"))
                call(
                    url,
                    "POST",
                    "/predict",
                    dict(
                        request, parameters={"mllib": {"net": {"test_batch_size": 0}}}
                    ),
                    expected=400,
                )
                call(
                    url,
                    "POST",
                    "/predict",
                    dict(request, data=[str(root / "missing.png")]),
                    expected=400,
                )
            finally:
                call(url, "DELETE", "/services/" + service)
        service = "legacy_prediction"
        repo = root / "legacy"
        repo.mkdir()
        call(
            url,
            "PUT",
            "/services/" + service,
            dict(
                mllib="pytorch",
                type="supervised",
                model={"repository": str(repo)},
                parameters={"input": {"connector": "image"}, "mllib": {}},
            ),
            201,
        )
        try:
            request = dict(
                service=service,
                data=["not-a-real-image"],
                parameters={"output": {"bbox": True}},
            )
            for mode in ("auto", "auto", "paths"):
                response = call(
                    url,
                    "POST",
                    "/predict",
                    dict(
                        request,
                        parameters={
                            "mllib": {"predict_input": mode},
                            "output": {"bbox": True},
                        },
                    ),
                )
                assert response["body"]["predictions"][0]["uri"] == "not-a-real-image"
            call(
                url,
                "POST",
                "/predict",
                dict(request, parameters={"mllib": {"predict_input": "connector"}}),
                expected=400,
            )
        finally:
            call(url, "DELETE", "/services/" + service)
        print(
            "PASS: detection/semantic HTTP batching [2,2,1], pixels, duplicates, dimensions, equivalence, errors and cleanup"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    run(args.url, gpu=args.gpu)
