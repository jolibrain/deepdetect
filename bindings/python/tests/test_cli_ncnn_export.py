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
    ([1, 6], "ssd", 3), ([1, 8400, 85], "yolox", 81),
])
def test_single_image_outputs(shape, fmt, classes):
    ncnn_export._check_output(shape, fmt, classes)
    with pytest.raises(ValueError):
        ncnn_export._check_output([2] + shape, fmt, classes)


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
