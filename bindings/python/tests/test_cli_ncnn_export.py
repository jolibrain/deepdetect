"""NCNN conversion, selection and publication contracts."""
import hashlib
import json
import shutil

import pytest
import yaml

from deepdetect.cli import ncnn_export
from deepdetect.cli.main import main


def test_cached_onnx_requires_matching_checkpoint(tmp_path):
    root = tmp_path / "onnx"
    bundle = root / "best"
    bundle.mkdir(parents=True)
    graph = bundle / "model.onnx"
    graph.touch()
    (root / "state.json").write_text(json.dumps({"bundle": "best"}))
    identity = {"name": "checkpoint-2.pt", "sha256": "1" * 64}
    (bundle / "metadata.json").write_text(json.dumps({"checkpoints": [identity]}))

    def select(digest):
        return ncnn_export._run_graph(tmp_path, [], [dict(identity, sha256=digest)],
                                      "2", "pytorch", {}, {"nclasses": 2}, tmp_path)

    assert select(identity["sha256"]) == graph
    with pytest.raises(ValueError, match="matching ONNX"):
        select("2" * 64)


@pytest.mark.parametrize("shape,fmt,classes", [
    ([1, 1000], "logits", 1000), ([None, 3], "probabilities", 3),
    ([1, 2, 4, 6], "segmentation", 2), ([150, None, None], "segmentation", 150),
    ([1, 6], "ssd", 3), ([1, 8400, 85], "yolox", 81),
])
def test_single_image_outputs(shape, fmt, classes):
    ncnn_export._check_output(shape, fmt, classes)
    with pytest.raises(ValueError):
        ncnn_export._check_output([2] + shape, fmt, classes)


@pytest.mark.parametrize("shape", [[1, 8, 8, 2], [1, 3, 8, 8], [2, 8, 0], [1, 2]])
def test_segmentation_requires_spatial_class_logits(shape):
    with pytest.raises(ValueError, match="segmentation logits"):
        ncnn_export._check_output(shape, "segmentation", 2)


def test_segmentation_bundle_contract():
    config = {"width": 512, "height": 512, "nclasses": 150,
              "service_mllib": {"ncnn_output": "segmentation"},
              "predict_output": {"confidences": ["best"]}}
    for mode in ("class_ids", "logits"):
        portable, profile, shape = ncnn_export._portable_config(config)
        assert shape == [1, 3, 512, 512]
        assert profile.name == portable["_export"]["profile"] == "ncnn-segmenter"
        assert portable["service_mllib"]["ncnn_segmentation_resize"] == mode
        options = {**profile.infer_defaults(), **portable}
        assert profile.predict_parameters(options)["output_parameters"] == {
            "confidences": ["best"], "confidence_threshold": 0.0}
        config["service_mllib"]["ncnn_segmentation_resize"] = "logits"
    config["service_input"] = {"ncnn_letterbox": True}
    with pytest.raises(ValueError, match="stretch"):
        ncnn_export._portable_config(config)
    config["service_input"] = {}
    config["service_mllib"]["ncnn_segmentation_resize"] = "unknown"
    with pytest.raises(ValueError, match="class_ids or logits"):
        ncnn_export._portable_config(config)


def test_onnx_export_and_relocation(tmp_path):
    onnx = pytest.importorskip("onnx")
    pnnx = shutil.which("pnnx")
    if not pnnx:
        pytest.skip("put the pinned Linux pnnx executable on PATH for the conversion test")
    source, dest = tmp_path / "mean.onnx", tmp_path / "bundle"
    info = onnx.helper.make_tensor_value_info
    fp32 = onnx.TensorProto.FLOAT
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("GlobalAveragePool", ["image"], ["pool"]),
         onnx.helper.make_node("Flatten", ["pool"], ["scores"], axis=1)],
        "classifier", [info("image", fp32, [1, 3, 2, 2])], [info("scores", fp32, [1, 3])])
    model = onnx.helper.make_model_gen_version(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    onnx.save(model, source)
    config = tmp_path / "source.yaml"
    config.write_text(yaml.safe_dump({"width": 2, "height": 2, "nclasses": 3,
                                      "service_mllib": {"ncnn_output": "logits"}}))
    args = ["export", str(source), "--config", str(config), "--format", "ncnn",
            "--pnnx", pnnx, "--output", str(dest)]
    assert main(args + ["--dry-run"]) == 0
    assert not dest.exists()
    assert main(args) == 0
    moved = dest.rename(tmp_path / "moved")
    assert {p.name for p in moved.iterdir()} == {
        "model.ncnn.param", "model.ncnn.bin", "service.json", "inference.yaml", "export.json"}
    service = json.loads((moved / "service.json").read_text())
    assert service["model"]["repository"] == "."
    report = json.loads((moved / "export.json").read_text())
    for item in report["files"]:
        assert hashlib.sha256((moved / item["path"]).read_bytes()).hexdigest() == item["sha256"]

    config.write_text(config.read_text().replace("nclasses: 3", "nclasses: 2"))
    assert main(args) == 1
    assert not dest.exists()


@pytest.mark.parametrize("extension,indices", [("ts", False), ("onnx", False), ("onnx", True)])
def test_topk_export(tmp_path, extension, indices):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pnnx = shutil.which("pnnx")
    if not pnnx:
        pytest.skip("put the pinned PNNX executable on PATH")
    from deepdetect.cli.ncnn_graph import read_layers

    class Select(torch.nn.Module):
        def forward(self, image):
            ranked = torch.topk(image.mean((2, 3)), 2, dim=1, largest=False)
            return ranked.indices.float() if indices else ranked.values

    graph, bundle = tmp_path / ("select." + extension), tmp_path / "bundle"
    model, image = Select().eval(), torch.zeros(1, 3, 4, 4)
    if extension == "ts":
        torch.jit.trace(model, image).save(str(graph))
    else:
        torch.onnx.export(model, (image,), str(graph), opset_version=17, dynamo=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"width": 4, "height": 4, "nclasses": 2,
                                      "service_mllib": {"ncnn_output": "logits"}}))
    assert main(["export", str(graph), "--config", str(config), "--format", "ncnn",
                 "--pnnx", pnnx, "--output", str(bundle)]) == 0
    assert json.loads((bundle / "export.json").read_text())["graph_sha256"] == hashlib.sha256(graph.read_bytes()).hexdigest()
    layers = read_layers(bundle / "model.ncnn.param")
    rank, = [n for n in layers if n.op == "DeepDetectTopK"]
    assert set(rank.params) >= {"0=2", "1=-1", "2=0", f"3={int(indices)}"}
    assert {s for n in layers for s in n.outputs} - {s for n in layers for s in n.inputs} == {"out0"}
    assert any(n.op == "DeepDetectIndicesToFloat" for n in layers) == indices
