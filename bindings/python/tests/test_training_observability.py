import json

from PIL import Image
import pytest

from training_observability import RunReader, RunWriter
from training_observability import cli as observe


def test_generic_writer_reader_and_cli_artifacts(tmp_path, capsys):
    root = tmp_path / "run"
    writer = RunWriter(root, run_id="generic-run")
    writer.metric("train_loss", 1.0, step=1, timestamp=10)
    writer.metric("train_loss", 0.5, step=2, timestamp=11)
    writer.metric("map-50_test0", 0.25, step=2, timestamp=12)
    writer.metric("map-50_test1", 0.5, step=2, timestamp=13)
    writer.metric("map-50_1_test0", 0.75, step=2, timestamp=14)

    image = root / "visuals" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(image)
    metadata = image.with_suffix(".json")
    metadata.write_text('{"sample": 1}', encoding="utf-8")
    writer.artifact(
        kind="prediction-overlay",
        path=image,
        metadata_path=metadata,
        step=2,
        split="test0",
    )

    reader = RunReader(root)
    assert reader.summary()["run_id"] == "generic-run"
    assert {plot.id for plot in reader.plots()} == {"loss-train-loss", "metric-map-50"}
    map_plot = next(plot for plot in reader.plots() if plot.id == "metric-map-50")
    assert map_plot.traces == ("class 1 / test0", "test0", "test1")
    artifacts = reader.artifacts(kind="prediction-overlay", step=2)
    assert len(artifacts) == 1
    assert artifacts[0].path == image.resolve()

    assert observe.main(["plots", str(root)]) == 0
    plots = json.loads(capsys.readouterr().out)
    assert {plot["id"] for plot in plots["plots"]} == {
        "loss-train-loss",
        "metric-map-50",
    }
    assert observe.main(["artifacts", str(root), "--step", "latest"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["artifacts"][0]["path"] == str(image.resolve())


def test_reader_supports_legacy_deepdetect_results_and_partial_live_record(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps({"run_id": "legacy-run", "status": "running"}), encoding="utf-8"
    )
    (root / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "metric",
                        "name": "train_loss",
                        "value": 1.0,
                        "iteration": 5,
                        "timestamp": 1,
                    }
                ),
                '{"event": "metric",',
            ]
        ),
        encoding="utf-8",
    )
    result_dir = root / "visdom-results" / "iteration-000005" / "test0"
    result_dir.mkdir(parents=True)
    image = result_dir / "sample-000001.png"
    Image.new("RGB", (4, 4), "white").save(image)
    image.with_suffix(".json").write_text("{}", encoding="utf-8")

    reader = RunReader(root)
    assert reader.layout == "deepdetect-legacy"
    assert [point.name for point in reader.metric_points()] == ["train_loss"]
    artifacts = reader.artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].step == 5.0
    assert artifacts[0].split == "test0"


def test_render_writes_only_requested_output(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    root = tmp_path / "run"
    writer = RunWriter(root)
    writer.metric("train_loss", 1.0, step=1)
    writer.metric("train_loss", 0.5, step=2)
    output = tmp_path / "rendered" / "loss.png"

    assert observe.main(
        ["render", str(root), "--plot", "loss-train-loss", "--output", str(output)]
    ) == 0
    assert output.is_file()
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["rendered"] == [{"path": str(output.resolve()), "plot": "loss-train-loss"}]
