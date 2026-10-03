from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepdetect.cli.config import load_config
from deepdetect.cli.main import build_parser
from deepdetect.cli.options import resolve_options
from deepdetect.cli.profiles import get_profile
from deepdetect.pytorch_worker.sdk import WorkerDependencyError

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = ROOT / "extern/pytorch_workers/rtdetrv4/worker.py"
spec = importlib.util.spec_from_file_location("rtdetrv4_finetune_worker", WORKER_PATH)
assert spec is not None and spec.loader is not None
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)
DeepDetectWorker = worker_module.DeepDetectWorker


class TinyDetector(torch.nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Module()
        self.decoder.enc_score_head = torch.nn.Linear(2, classes)
        self.decoder.dec_score_head = torch.nn.ModuleList(
            [torch.nn.Linear(2, classes)]
        )
        self.decoder.denoising_class_embed = torch.nn.Embedding(classes + 1, 2)


def test_full_detector_checkpoint_reinitializes_only_class_heads(tmp_path):
    source = TinyDetector(classes=80)
    source.backbone.weight.data.fill_(0.25)
    source.encoder.weight.data.fill_(0.5)
    state = source.state_dict()
    state["encoder.feature_projector.0.weight"] = torch.ones((2, 2))
    path = tmp_path / "official.pth"
    torch.save({"model": state}, path)

    target = TinyDetector(classes=2)
    initial_head = target.decoder.enc_score_head.weight.detach().clone()
    worker = DeepDetectWorker()
    worker.device = torch.device("cpu")
    worker._load_checkpoint_payload(torch, target, path)

    assert torch.equal(target.backbone.weight, source.backbone.weight)
    assert torch.equal(target.encoder.weight, source.encoder.weight)
    assert torch.equal(target.decoder.enc_score_head.weight, initial_head)


@pytest.mark.parametrize(
    "damage", ["missing_decoder", "missing_class_head", "wrong_backbone", "extra_key"]
)
def test_incompatible_detector_checkpoint_fails_before_loading(tmp_path, damage):
    source = TinyDetector()
    state = source.state_dict()
    if damage == "missing_decoder":
        state.pop("decoder.enc_score_head.weight")
        state.pop("decoder.dec_score_head.0.weight")
        state.pop("decoder.denoising_class_embed.weight")
        state.pop("encoder.weight")
    elif damage == "missing_class_head":
        state.pop("decoder.enc_score_head.weight")
    elif damage == "wrong_backbone":
        state["backbone.weight"] = torch.ones((3, 2))
    else:
        state["teacher.weight"] = torch.ones((2, 2))
    path = tmp_path / "wrong.pth"
    torch.save({"model": state}, path)
    target = TinyDetector()
    initial = target.backbone.weight.detach().clone()
    worker = DeepDetectWorker()
    worker.device = torch.device("cpu")

    with pytest.raises(WorkerDependencyError, match="incomplete or incompatible"):
        worker._load_checkpoint_payload(torch, target, path)
    assert torch.equal(target.backbone.weight, initial)


def test_resume_requires_exact_class_head(tmp_path):
    path = tmp_path / "checkpoint-1.pt"
    torch.save({"model_state": TinyDetector(classes=80).state_dict()}, path)
    worker = DeepDetectWorker()
    worker.device = torch.device("cpu")
    with pytest.raises(WorkerDependencyError, match="incomplete or incompatible"):
        worker._load_checkpoint_payload(torch, TinyDetector(classes=2), path, resume=True)


def test_finetune_requires_full_weights_and_uses_backbone_lr(tmp_path):
    worker = DeepDetectWorker()
    worker.context = SimpleNamespace(repository_path=tmp_path)
    with pytest.raises(WorkerDependencyError, match="full detector checkpoint"):
        worker._checkpoint_path_for_training(
            {"rtdetrv4": {"require_pretrained": True}}
        )

    model = TinyDetector()
    worker._train_mllib = {
        "rtdetrv4": {"backbone_lr_multiplier": 0.1, "weight_decay": 0.0001}
    }
    optimizer = worker.create_optimizer(torch, model, base_lr=0.0004)
    groups = {
        (group["lr"], group["weight_decay"]): group["params"]
        for group in optimizer.param_groups
    }
    assert any(p is model.backbone.weight for p in groups[(0.00004, 0.0001)])
    assert any(p is model.backbone.bias for p in groups[(0.00004, 0.0)])
    assert any(p is model.encoder.weight for p in groups[(0.0004, 0.0001)])


@pytest.mark.parametrize("variant,channels,ratio", [("m", 1536, 0.1), ("l", 2048, 0.05)])
def test_m_l_recipes_select_upstream_variant(variant, channels, ratio):
    recipe = load_config(ROOT / f"extern/pytorch_workers/rtdetrv4/finetune-{variant}.yaml")
    assert recipe["service_mllib"]["rtdetrv4"]["config_path"].endswith(
        f"rtv4_hgnetv2_{variant}_coco.yml"
    )
    assert recipe["mllib"]["rtdetrv4"]["require_pretrained"] is True
    assert recipe["mllib"]["rtdetrv4"]["backbone_lr_multiplier"] == ratio
    repo = Path(recipe["service_mllib"]["rtdetrv4"]["repo_path"])
    if not repo.is_dir():
        pytest.skip("upstream RT-DETRv4 checkout unavailable")
    worker = DeepDetectWorker()
    worker.context = SimpleNamespace(mllib=recipe["service_mllib"])
    model = worker.create_model(3, torch)
    assert model.backbone._out_channels[-1] == channels
    assert model.decoder.enc_score_head.out_features == 2


def test_cli_recipe_overrides_keep_detector_config_and_optimizer_options(tmp_path):
    recipe = ROOT / "extern/pytorch_workers/rtdetrv4/finetune-m.yaml"
    weights = tmp_path / "official-m.pth"
    args = build_parser().parse_args(
        [
            "train",
            "external-pytorch-detector",
            "--config",
            str(recipe),
            "--weights",
            str(weights),
            "--nclasses",
            "3",
            "--set",
            "mllib.rtdetrv4.backbone_lr_multiplier=0.08",
        ]
    )
    profile = get_profile("external-pytorch-detector")
    options = resolve_options(
        profile.train_defaults(),
        args,
        {"weights": args.weights, "nclasses": args.nclasses},
    )
    service = profile.service_parameters(options)
    train = profile.train_parameters(options)

    assert service["mllib_parameters"]["rtdetrv4"]["config_path"].endswith(
        "rtv4_hgnetv2_m_coco.yml"
    )
    assert service["mllib_parameters"]["weights"] == str(weights.resolve())
    assert train["mllib_parameters"]["rtdetrv4"]["require_pretrained"] is True
    assert (
        train["mllib_parameters"]["rtdetrv4"]["backbone_lr_multiplier"]
        == 0.08
    )
