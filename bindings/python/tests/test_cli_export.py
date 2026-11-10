"""Export contracts: exact checkpoint selection and movable inference bundles."""
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from deepdetect.cli import exporting
from deepdetect.cli.main import main
from deepdetect.cli.options import resolve_options
from deepdetect.cli.profiles import get_profile


def run_repo(tmp_path, model="external-pytorch-segmentation", **extra):
    source = tmp_path / "run"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"model": model}))
    saved = dict(nclasses=2, width=32, height=32,
                 service_mllib={"module": "example_worker.worker", "class": "DeepDetectWorker"})
    saved.update(extra)
    (source / "config.yaml").write_text(yaml.safe_dump(saved))
    (source / "best_model.txt").write_text("iteration:2\nforeground_iou:0.8\n")
    for step in (2, 10):
        (source / f"checkpoint-{step}.pt").write_bytes(f"model {step}".encode())
    (source / "solver-2.pt").write_bytes(b"optimizer")
    (source / "metrics.jsonl").write_text("{}\n")
    return source


def export(source, destination, *args):
    return main(["export", str(source), "--output", str(destination), *args])


@pytest.mark.parametrize("selection,expected", [("best", "2"), ("latest", "10"), ("2", "2")])
def test_selection_and_minimal_files(tmp_path, selection, expected):
    source = run_repo(tmp_path)
    # Modification time must not select the best or latest checkpoint.
    os.utime(source / "checkpoint-2.pt", (2000000000, 2000000000))
    target = tmp_path / "export"
    assert export(source, target, "--checkpoint", selection) == 0
    assert set(p.name for p in target.iterdir()) == {
        f"checkpoint-{expected}.pt", "inference.yaml", "export.json"}
    report = json.loads((target / "export.json").read_text())
    assert report["iteration"] == expected
    assert all(len(item["sha256"]) == 64 for item in report["files"])
    config = yaml.safe_load((target / "inference.yaml").read_text())
    assert config["weights"] == f"checkpoint-{expected}.pt"
    assert config["repository"] == "."


def test_latest_alias_without_solver(tmp_path):
    source = run_repo(tmp_path)
    for path in source.glob("checkpoint-*.pt"):
        path.unlink()
    (source / "checkpoint-latest.pt").write_bytes(b"model")
    assert export(source, tmp_path / "export", "--checkpoint", "latest") == 0


@pytest.mark.parametrize("marker", ["iteration:../2", "iteration:-1", "score:0.9", "iteration:2\niteration:10"])
def test_bad_markers_do_not_publish(tmp_path, marker):
    source = run_repo(tmp_path)
    (source / "best_model.txt").write_text(marker)
    assert export(source, tmp_path / "export") == 1
    assert not (tmp_path / "export").exists()


def test_missing_best_does_not_fallback(tmp_path):
    source = run_repo(tmp_path)
    (source / "checkpoint-2.pt").unlink()
    assert export(source, tmp_path / "export") == 1


def test_file_worker_requires_module_and_drops_interpreter(tmp_path):
    source = run_repo(tmp_path, service_mllib={"entrypoint": "/old/worker.py", "python": "/old/python"})
    target = tmp_path / "export"
    assert export(source, target) == 1
    assert export(source, target, "--worker-module", "installed.worker") == 0
    config = yaml.safe_load((target / "inference.yaml").read_text())
    assert config["service_mllib"]["module"] == "installed.worker"
    assert "entrypoint" not in config["service_mllib"]
    assert "python" not in config["service_mllib"]


def test_opaque_paths_require_declaration_or_override(tmp_path):
    source = run_repo(tmp_path, service_mllib={"module": "example.worker", "custom": {"dataset": "/old/train.txt"}})
    assert export(source, tmp_path / "export") == 1
    assert export(source, tmp_path / "export", "--set", "service_mllib.custom={}") == 0
    config = yaml.safe_load((tmp_path / "export/inference.yaml").read_text())
    assert config["service_mllib"]["custom"] == {}


def test_tar_relocation_assets_and_override_precedence(tmp_path):
    source = run_repo(tmp_path, export={
        "version": 1,
        "module": "installed.worker",
        "inference": {"service_mllib": {"class": "DeepDetectWorker"}},
        "assets": [{"key": "service_mllib.labels", "source": "labels.txt"}],
        "requirements": ["nonexistent-test-worker==1.0"],
    })
    (source / "labels.txt").write_text("background\ncrack\n")
    target = tmp_path / "export"
    assert export(source, target) == 0
    archive = shutil.make_archive(str(tmp_path / "model"), "gztar", target)
    shutil.rmtree(source)
    shutil.rmtree(target)
    moved = tmp_path / "elsewhere"
    shutil.unpack_archive(archive, moved, filter="data")
    args = SimpleNamespace(config=moved / "inference.yaml", overrides=["width=64"], model="external-pytorch-segmentation")
    profile = get_profile(args.model)
    options = resolve_options(profile.infer_defaults(), args, {"height": 48})
    assert options["weights"] == moved / "checkpoint-2.pt"
    assert options["repository"] == moved
    assert Path(options["service_mllib"]["labels"]).read_text() == "background\ncrack\n"
    service = profile.service_parameters(options)
    assert service["mllib_parameters"]["module"] == "installed.worker"
    assert service["input_parameters"]["width"] == 64
    assert service["input_parameters"]["height"] == 48


def test_native_components_and_normalization(tmp_path):
    source = run_repo(tmp_path, "segformer", service_mllib={}, augmentation={"crop_size": 80})
    (source / "checkpoint-2.ptw").write_bytes(b"head")
    (source / "corresp.txt").write_text("0 background\n1 crack")
    assert export(source, tmp_path / "export") == 0
    config = yaml.safe_load((tmp_path / "export/inference.yaml").read_text())
    assert config["width"] == config["height"] == 80
    assert config["predict_input"]["mean"] == [0.485, 0.456, 0.406]
    assert config["service_input"]["db"] is False
    assert (tmp_path / "export/checkpoint-2.ptw").exists()
    assert (tmp_path / "export/corresp.txt").exists()


def test_native_head_missing_graph(tmp_path):
    source = run_repo(tmp_path, "yolox", service_mllib={})
    (source / "checkpoint-2.pt").rename(source / "checkpoint-2.ptw")
    assert export(source, tmp_path / "export") == 1
    (source / "base.pt").write_bytes(b"base")
    assert export(source, tmp_path / "export") == 0
    assert (tmp_path / "export/base.pt").exists()


def test_native_npt_without_solver(tmp_path):
    source = run_repo(tmp_path, "yolox", service_mllib={})
    (source / "checkpoint-2.pt").rename(source / "checkpoint-2.npt")
    (source / "solver-2.pt").unlink()
    assert export(source, tmp_path / "export") == 0


def test_dry_run_and_destination_protection(tmp_path):
    source = run_repo(tmp_path)
    target = tmp_path / "parent/export"
    assert export(source, target, "--dry-run") == 0
    assert not target.parent.exists()
    assert export(source, source / "export") == 1
    assert export(source, tmp_path) == 1
    target.mkdir(parents=True)
    assert export(source, target) == 1


def test_profile_discovery_and_conflict(tmp_path):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    assert export(source, target, "--model", "yolox") == 1
    (source / "run.json").unlink()
    assert export(source, target) == 1
    assert export(source, target, "--model", "external-pytorch-segmentation") == 0


def test_copy_failure_and_source_mutation_cleanup(tmp_path, monkeypatch):
    source = run_repo(tmp_path)
    original = shutil.copyfile
    def mutate(src, dst):
        original(src, dst)
        Path(src).write_bytes(b"changed while copying")
    monkeypatch.setattr(exporting.shutil, "copyfile", mutate)
    assert export(source, tmp_path / "export") == 1
    assert not list(tmp_path.glob(".export-*"))
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("destination", ["../outside", "/absolute", "inference.yaml", "inference.yaml/child", "checkpoint-2.pt"])
def test_asset_collision_and_traversal(tmp_path, destination):
    source = run_repo(tmp_path, export={"assets": [
        {"key": "service_mllib.asset", "source": "asset.txt", "destination": destination}]})
    (source / "asset.txt").write_text("asset")
    assert export(source, tmp_path / "export") == 1
    assert not (tmp_path / "export").exists()


def test_asset_directory_and_symlink_file(tmp_path):
    source = run_repo(tmp_path)
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / "real.txt").write_text("content")
    (directory / "alias.txt").symlink_to(directory / "real.txt")
    assert export(source, tmp_path / "export", "--asset", f"service_mllib.resources={directory}") == 0
    target = tmp_path / "export/assets/0/assets/alias.txt"
    assert target.read_text() == "content"
    assert not target.is_symlink()


def test_export_config_rejects_missing_assets_and_escape(tmp_path):
    config = {"weights": "../model.pt", "_export": {"version": 1, "paths": ["weights"]}}
    with pytest.raises(ValueError, match="inside"):
        exporting.resolve_export_config(config, tmp_path / "inference.yaml")
    config["weights"] = "missing.pt"
    with pytest.raises(FileNotFoundError):
        exporting.resolve_export_config(config, tmp_path / "inference.yaml")


def test_training_asset_paths_resolved_before_persisting(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"export": {"assets": [{"key": "service_mllib.labels", "source": "labels.txt"}]}}))
    args = SimpleNamespace(config=config, overrides=[])
    options = resolve_options({}, args, {})
    assert options["export"]["assets"][0]["source"] == str(tmp_path / "labels.txt")


def test_cli_inference_after_relocation_uses_worker_checkpoint(tmp_path, monkeypatch, capsys):
    """Exercise the CLI, worker loader and real tensor prediction across relocation."""
    import socket
    import textwrap
    import torch
    from PIL import Image
    from deepdetect import DeepDetect
    from deepdetect.cli import inference
    from deepdetect.pytorch_worker.runtime import WorkerRuntime

    # Installed code is deliberately outside the training and export directories.
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "export_fixture_worker.py").write_text(textwrap.dedent('''
        from pathlib import Path
        import torch
        from deepdetect.pytorch_worker.sdk import DeepDetectWorkerBase
        class DeepDetectWorker(DeepDetectWorkerBase):
            def configure(self, context):
                super().configure(context)
                self.weight = torch.load(context.mllib['weights'], weights_only=True)
                self.label = Path(context.mllib['labels']).read_text()
                return {}
            def predict(self, params):
                probability = torch.sigmoid(self.weight * torch.tensor(0.5)).item()
                return {'results': [{'uri': path, 'classes': [{'cat': self.label, 'prob': probability,
                    'bbox': {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}}]}
                    for path in params['request']['data']]}
    '''))
    monkeypatch.syspath_prepend(str(installed))
    source = run_repo(tmp_path, "external-pytorch-detector", service_mllib={"module": "export_fixture_worker"},
                      export={"assets": [{"key": "service_mllib.labels", "source": "labels.txt"}]})
    torch.save(torch.tensor(2.0), source / "checkpoint-2.pt")
    torch.save(torch.tensor(-2.0), source / "checkpoint-10.pt")
    (source / "labels.txt").write_text("crack")
    target = tmp_path / "export"
    assert export(source, target) == 0
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)

    class Runtime:
        def build_info(self):
            return {"version": "test", "cuda": False}
        def create_service(self, name, request):
            request = json.loads(request)
            self.sock, self.peer = socket.socketpair()
            self.worker_runtime = WorkerRuntime(self.sock)
            params = {"repository": request["model"]["repository"], "mllib": request["parameters"]["mllib"]}
            self.worker_runtime.worker = self.worker_runtime._load_worker(params)
            self.worker_runtime._configure_worker(params)
            return {"status": {"code": 201}}
        def set_log_level(self, level):
            return {"status": {"code": 200}}
        def set_service_log_level(self, name, level):
            return {"status": {"code": 200}}
        def predict(self, request):
            predictions = self.worker_runtime.worker.predict({"request": json.loads(request)})['results']
            return {"status": {"code": 200}, "body": {"predictions": predictions}}
        def delete_service(self, name, request):
            self.sock.close()
            self.peer.close()
            return {"status": {"code": 200}}

    monkeypatch.setattr(inference.deepdetect, "DeepDetect", lambda: DeepDetect(_runtime=Runtime()))
    tested = tmp_path / "tested-export"
    assert export(source, tested, "--test-inference") == 0
    assert json.loads((tested / "export.json").read_text())["inference_test"]["status"] == "passed"
    def predict(root):
        capsys.readouterr()
        assert main(["infer", "external-pytorch-detector", str(image), "--config", str(root / "inference.yaml")]) == 0
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        return next(event for event in events if event["event"] == "prediction")["prediction"]

    before = predict(target)
    archive = shutil.make_archive(str(tmp_path / "model"), "gztar", target)
    shutil.rmtree(source)
    shutil.rmtree(target)
    relocated = tmp_path / "relocated"
    shutil.unpack_archive(archive, relocated, filter="data")
    monkeypatch.chdir(installed)
    after = predict(relocated)
    assert before == after
    assert after["classes"][0]["prob"] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())


@pytest.mark.parametrize("profile", ["external-pytorch-image", "external-pytorch-detector", "torchvision-detector", "vitpose"])
def test_other_worker_profiles(tmp_path, profile):
    source = run_repo(tmp_path, profile)
    assert export(source, tmp_path / "export") == 0
    config = yaml.safe_load((tmp_path / "export/inference.yaml").read_text())
    assert config["_export"]["profile"] == profile


@pytest.mark.parametrize("kind,variable", [
    ("cracks", "DEEPDETECT_EXPORT_CRACKS_RUN"),
    ("native", "DEEPDETECT_EXPORT_NATIVE_RUN"),
])
def test_real_model_relocation_smoke(tmp_path, kind, variable):
    """Opt in with a completed run and an installed native wheel in this interpreter."""
    import subprocess
    import sys
    from PIL import Image
    import numpy as np

    run = os.environ.get(variable)
    if not run:
        pytest.skip(f"set {variable} to enable real-model export/inference")
    source = Path(run).resolve()
    model = json.loads((source / "run.json").read_text())["model"]
    target = tmp_path / "export"
    overrides = ["--worker-module", "cracks_deepdetect_worker.worker", "--set", "service_mllib.segmentation={}"] if kind == "cracks" else []
    # An actual installed native wheel supplies the engine; load this checkout's
    # CLI implementation without replacing that wheel's native client/libraries.
    code = '''
import sys
import deepdetect
deepdetect.__path__.insert(0, sys.argv.pop(1))
from deepdetect.cli.main import main
raise SystemExit(main(sys.argv[1:]))
'''
    source_package = str(Path(__file__).resolve().parents[1] / "deepdetect")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    completed = subprocess.run(
        [sys.executable, "-c", code, source_package, "export", str(source),
         "--output", str(target), *overrides, "--test-inference"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-6000:] + completed.stdout[-6000:]
    assert json.loads((target / "export.json").read_text())["inference_test"]["status"] == "passed"
    image = tmp_path / "tile.png"
    Image.fromarray(np.random.default_rng(123).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image)
    def predict(bundle):
        result = subprocess.run([sys.executable, "-c", code, source_package,
                                 "infer", model, str(image), "--config", str(bundle / "inference.yaml")],
                                cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-6000:]
        events = []
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                events.append(json.loads(line))
        return next(event["prediction"] for event in events if event.get("event") == "prediction")
    before = predict(target)
    archive = shutil.make_archive(str(tmp_path / "model"), "gztar", target)
    shutil.rmtree(target)
    moved = tmp_path / "elsewhere/model"
    shutil.unpack_archive(archive, moved, filter="data")
    after = predict(moved)
    assert before["vals"] == after["vals"]
    if "confidences" in before:
        for key, values in before["confidences"].items():
            np.testing.assert_allclose(values, after["confidences"][key], rtol=1e-5, atol=1e-6)


def test_saved_prediction_overrides_are_preserved(tmp_path):
    source = run_repo(tmp_path, predict_input={"scale": 0.125}, predict_output={"confidences": ["0", "1"]})
    assert export(source, tmp_path / "export") == 0
    config = yaml.safe_load((tmp_path / "export/inference.yaml").read_text())
    assert config["predict_input"]["scale"] == 0.125
    assert config["predict_output"]["confidences"] == ["0", "1"]


def test_unset_removes_worker_training_dependency(tmp_path):
    source = run_repo(tmp_path, service_mllib={"module": "example.worker", "dataset": "/old/data"})
    assert export(source, tmp_path / "export", "--unset", "service_mllib.dataset") == 0
    config = yaml.safe_load((tmp_path / "export/inference.yaml").read_text())
    assert "dataset" not in config["service_mllib"]


def test_asset_fields_cannot_overlap(tmp_path):
    source = run_repo(tmp_path)
    asset = tmp_path / "asset.txt"
    asset.write_text("asset")
    assert export(source, tmp_path / "export", "--asset", f"service_mllib.resource={asset}",
                  "--asset", f"service_mllib={asset}", "--dry-run") == 1
    assert not (tmp_path / "export").exists()


def test_generated_config_must_match_inference_profile(tmp_path):
    source = run_repo(tmp_path)
    assert export(source, tmp_path / "export") == 0
    args = SimpleNamespace(config=tmp_path / "export/inference.yaml", overrides=[], model="yolox")
    with pytest.raises(ValueError, match="profile"):
        resolve_options(get_profile("yolox").infer_defaults(), args, {})


def test_plain_config_paths_keep_cwd_semantics(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("weights: model.pt\nrepository: model\n")
    options = resolve_options({}, SimpleNamespace(config=config, overrides=[]), {})
    assert options["weights"] == Path("model.pt")
    assert options["repository"] == Path("model")


def test_unknown_export_setting_fails(tmp_path):
    source = run_repo(tmp_path, export={"inference": {"widht": 12}})
    assert export(source, tmp_path / "export") == 1


def test_atomic_publication_does_not_replace_concurrent_empty_directory(tmp_path):
    stage = tmp_path / "staging"
    stage.mkdir()
    (stage / "model.pt").write_bytes(b"model")
    target = tmp_path / "target"
    target.mkdir()
    original_inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        exporting._publish_directory(stage, target)
    assert target.stat().st_ino == original_inode
    assert (stage / "model.pt").is_file()


def test_training_asset_override_resolves_from_cwd(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("{}")
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(config=config, overrides=["export.assets=[{key: service_mllib.labels, source: labels.txt}]"])
    options = resolve_options({}, args, {})
    assert options["export"]["assets"][0]["source"] == str(tmp_path / "labels.txt")


def test_export_default_success_message_and_error_are_distinct(tmp_path, capsys):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    assert export(source, target) == 0
    output = capsys.readouterr()
    assert f"Export successful: {target}" in output.out
    assert "Checkpoint: 2 (best)" in output.out
    assert f"Inference configuration: {target}/inference.yaml" in output.out
    assert f"Full report: {target}/export.json" in output.out
    assert not output.err
    assert export(source, target) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "already exists" in output.err


@pytest.mark.parametrize("output_format", ["json", "jsonl"])
def test_export_json_has_explicit_success_status(tmp_path, capsys, output_format):
    source = run_repo(tmp_path)
    assert export(source, tmp_path / "export", "--output-format", output_format) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["event"] == "export_finished"
    assert report["status"] == "success"
    assert report["files"]
    assert not output.err


def test_export_dry_run_text_is_not_a_completion_message(tmp_path, capsys):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    assert export(source, target, "--dry-run") == 0
    output = capsys.readouterr()
    assert "no files written" in output.out
    assert "checkpoint-2.pt" in output.out
    assert "Export successful:" not in output.out
    assert not target.exists()
    assert export(source, target, "--dry-run", "--output-format", "json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned"
    assert report["event"] == "export_planned"


@pytest.mark.parametrize("model,gray", [("external-pytorch-segmentation", False), ("vitpose", False), ("external-pytorch-image", True)])
def test_export_test_inference_random_size_and_clean_bundle(tmp_path, monkeypatch, capsys, model, gray):
    from PIL import Image
    source = run_repo(tmp_path, model, width=48, height=32, service_input={"bw": gray})
    target = tmp_path / "export"
    seen = {}

    class Service:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            seen["closed"] = True
        def predict(self, images, **parameters):
            with Image.open(images[0]) as image:
                assert image.size == (48, 32)
                assert image.mode == ("L" if gray else "RGB")
                assert len(set(image.tobytes())) > 1
            if model == "vitpose":
                bbox = Path(parameters["input_parameters"]["bbox_files"][0])
                assert bbox.read_text() == "1 0 0 48 32\n"
            # Simulate a worker writing scratch files into its service repository.
            (seen["repository"] / "worker-output.json").write_text("{}")
            return {"predictions": [{"uri": str(images[0]), "classes": []}]}

    class Client:
        build_info = {"cuda": False}
        def create_service(self, name, **parameters):
            seen["repository"] = Path(parameters["model"]["repository"])
            assert seen["repository"] != target
            assert seen["repository"] != source
            assert Path(parameters["mllib_parameters"]["weights"]).parent == seen["repository"]
            return Service()

    monkeypatch.setattr(exporting.deepdetect, "DeepDetect", Client)
    assert export(source, target, "--test-inference") == 0
    assert "Inference test: PASSED (random 48x32 image)" in capsys.readouterr().out
    report = json.loads((target / "export.json").read_text())
    assert report["inference_test"]["status"] == "passed"
    assert report["inference_test"]["input"]["channels"] == (1 if gray else 3)
    assert report["inference_test"]["predictions"] == 1
    assert seen["closed"]
    assert not seen["repository"].exists()
    assert not (target / "worker-output.json").exists()
    assert set(p.name for p in target.iterdir()) == {"checkpoint-2.pt", "inference.yaml", "export.json"}


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_inference_failure_retains_export_and_returns_error(tmp_path, monkeypatch, capsys, output_format):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    def fail(*args):
        raise RuntimeError("model cannot load checkpoint")
    monkeypatch.setattr(exporting, "_test_inference", fail)
    assert export(source, target, "--test-inference", "--output-format", output_format) == 1
    output = capsys.readouterr()
    assert "Export successful:" not in output.out
    assert "model cannot load checkpoint" in output.err
    if output_format == "json":
        event = json.loads(output.out)
        assert event["status"] == "failed"
        assert event["event"] == "export_test_failed"
    else:
        assert "inference test FAILED" in output.out
    report = json.loads((target / "export.json").read_text())
    assert report["inference_test"] == {"status": "failed", "error": "model cannot load checkpoint"}
    assert (target / "checkpoint-2.pt").is_file()


def test_test_inference_and_dry_run_are_incompatible(tmp_path):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    assert export(source, target, "--test-inference", "--dry-run") == 1
    assert not target.exists()


def test_test_inference_uses_exported_gpu_selection(tmp_path, monkeypatch):
    source = run_repo(tmp_path)
    target = tmp_path / "export"
    def test_config(destination, profile):
        options = yaml.safe_load((destination / "inference.yaml").read_text())
        assert options["gpu"] is True
        assert options["gpuid"] == 2
        return {"status": "passed", "input": {"width": 32, "height": 32}}
    monkeypatch.setattr(exporting, "_test_inference", test_config)
    assert export(source, target, "--test-inference", "--set", "gpu=true", "--set", "gpuid=2") == 0


def test_inference_logs_do_not_corrupt_json_report(tmp_path, monkeypatch, capfd):
    source = run_repo(tmp_path)
    def noisy_test(*args):
        print("Python inference log")
        os.write(1, b"Native inference log\n")
        return {"status": "passed", "input": {"width": 32, "height": 32}}
    monkeypatch.setattr(exporting, "_test_inference", noisy_test)
    assert export(source, tmp_path / "export", "--test-inference", "--output-format", "json") == 0
    output = capfd.readouterr()
    assert json.loads(output.out)["inference_test"]["status"] == "passed"
    assert "Python inference log" in output.err
    assert "Native inference log" in output.err


def test_inference_test_rejects_missing_prediction(tmp_path, monkeypatch):
    source = run_repo(tmp_path)
    class Service:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def predict(self, *args, **kwargs):
            return {"predictions": []}
    class Client:
        build_info = {"cuda": False}
        def create_service(self, *args, **kwargs):
            return Service()
    monkeypatch.setattr(exporting.deepdetect, "DeepDetect", Client)
    target = tmp_path / "export"
    assert export(source, target, "--test-inference") == 1
    report = json.loads((target / "export.json").read_text())
    assert report["inference_test"]["status"] == "failed"
    assert "prediction count" in report["inference_test"]["error"]
