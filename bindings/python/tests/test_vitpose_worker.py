from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

WHEEL_TEST_SOURCE_ROOT = os.environ.get("DEEPDETECT_WHEEL_TEST_SOURCE_ROOT")
ROOT = (
    Path(WHEEL_TEST_SOURCE_ROOT).resolve()
    if WHEEL_TEST_SOURCE_ROOT
    else Path(__file__).resolve().parents[3]
)
VITPOSE_ROOT = ROOT / "extern" / "pytorch_workers" / "vitpose"
if str(VITPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(VITPOSE_ROOT))
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

torch = pytest.importorskip("torch")

from deepdetect.pytorch_worker.sdk import (
    Cancellation,
    WorkerContext,
    WorkerReporter,
    validate_prediction_result,
)
from coco_keypoints_to_dd import (
    format_deepdetect_keypoint_line,
    format_deepdetect_topdown_line,
)
from vitpose_worker.assignment import hungarian_assign
from vitpose_worker.checkpoint import (
    _atomic_torch_save,
    checkpoint_path,
    load_model_checkpoint,
    load_optimizer_checkpoint,
)
from vitpose_worker.config import worker_config_from_mllib
from vitpose_worker.decode import decode_topdown_outputs
from vitpose_worker.losses import (
    PoseLossConfig,
    masked_heatmap_mse,
    slot_pose_losses,
    topdown_pose_losses,
)
from vitpose_worker.model import ViTPoseModelConfig, ViTPoseSlots, ViTPoseTopDown
from vitpose_worker.targets import PoseTargetConfig, build_batch_targets
from vitpose_worker.worker_impl import ConnectorBatchPrefetcher, DeepDetectWorker
from vitpose_worker.worker_impl import PoseTrainOptions, PoseTrainRequest


def test_vitpose_atomic_checkpoint_preserves_previous_file_on_interruption(tmp_path):
    final_path = tmp_path / "solver-latest.pt"
    final_path.write_bytes(b"previous-solver")

    class InterruptedTorch:
        saved_path = None

        @classmethod
        def save(cls, _payload, path):
            cls.saved_path = path
            path.write_bytes(b"incomplete-solver")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _atomic_torch_save(InterruptedTorch, {"iteration": 2}, final_path)

    assert InterruptedTorch.saved_path.parent == final_path.parent
    assert final_path.read_bytes() == b"previous-solver"
    assert list(tmp_path.glob(".solver-latest.pt.*.tmp")) == []


def test_hungarian_assignment_is_permutation_invariant():
    assert hungarian_assign([[4.0, 1.0], [1.0, 4.0]]) == [(1, 0), (0, 1)]


def test_vitpose_resume_checkpoint_precedes_all_pretrained_sources(tmp_path):
    weights = tmp_path / "weights.pt"
    nested_weights = tmp_path / "nested-weights.pt"
    weights.write_bytes(b"weights")
    nested_weights.write_bytes(b"nested")
    (tmp_path / "checkpoint-6.pt").write_bytes(b"model")
    (tmp_path / "solver-6.pt").write_bytes(b"solver")

    selected = checkpoint_path(
        {
            "resume": True,
            "resume_from": "latest",
            "weights": str(weights),
            "vitpose": {"pretrained_model": str(nested_weights)},
        },
        tmp_path,
    )

    assert selected == tmp_path / "checkpoint-6.pt"


def test_vitpose_resume_loads_solver_from_selected_model_iteration(tmp_path):
    class FakeTorch:
        loaded = []

        @classmethod
        def load(cls, path, *, map_location):
            cls.loaded.append((path, map_location))
            return {"optimizer_state": {"iteration": 6}}

    class FakeOptimizer:
        state = None

        @classmethod
        def load_state_dict(cls, state):
            cls.state = state

    (tmp_path / "checkpoint-6.pt").write_bytes(b"model")
    (tmp_path / "solver-6.pt").write_bytes(b"solver")
    (tmp_path / "checkpoint-9.pt").write_bytes(b"incomplete")

    load_optimizer_checkpoint(
        FakeTorch,
        FakeOptimizer(),
        tmp_path,
        device="cpu",
        mllib={"resume": True, "resume_from": "latest"},
    )

    assert FakeTorch.loaded == [(tmp_path / "solver-6.pt", "cpu")]
    assert FakeOptimizer.state == {"iteration": 6}


def test_layer_decay_defaults_to_uniform_for_fresh_training():
    mllib = {
        "nkeypoints": 2,
        "vitpose": {
            "head": "topdown",
            "variant": "base",
            "image_size": [32, 32],
            "heatmap_size": [8, 8],
        },
    }

    assert worker_config_from_mllib(mllib).layer_decay == 1.0
    assert worker_config_from_mllib({**mllib, "weights": "mae.pth"}).layer_decay == 0.75


def test_layer_decay_explicit_value_overrides_pretraining_policy():
    mllib = {
        "weights": "mae.pth",
        "vitpose": {
            "head": "topdown",
            "variant": "base",
            "image_size": [32, 32],
            "heatmap_size": [8, 8],
            "layer_decay": 0.6,
        },
    }

    assert worker_config_from_mllib(mllib).layer_decay == 0.6


def test_heatmap_foreground_weight_defaults_by_head_and_can_be_overridden():
    common = {
        "nkeypoints": 2,
        "vitpose": {
            "variant": "tiny",
            "image_size": [32, 32],
            "heatmap_size": [8, 8],
        },
    }

    topdown = worker_config_from_mllib(
        {**common, "vitpose": {**common["vitpose"], "head": "topdown"}}
    )
    slots = worker_config_from_mllib(
        {**common, "vitpose": {**common["vitpose"], "head": "slots"}}
    )
    overridden = worker_config_from_mllib(
        {
            **common,
            "vitpose": {
                **common["vitpose"],
                "head": "slots",
                "heatmap_foreground_weight": 25,
            },
        }
    )

    assert topdown.loss.heatmap_foreground_weight == 1.0
    assert slots.loss.heatmap_foreground_weight == 100.0
    assert overridden.loss.heatmap_foreground_weight == 25.0


def test_foreground_weighted_heatmap_mse_emphasizes_target_peaks():
    pred = torch.zeros((1, 1, 1, 8, 8))
    target = torch.zeros_like(pred)
    target[..., 3, 4] = 1.0
    weights = torch.ones((1, 1, 1, 1))

    unweighted = masked_heatmap_mse(pred, target, weights)
    weighted = masked_heatmap_mse(
        pred,
        target,
        weights,
        foreground_weight=100.0,
    )

    assert weighted > 10.0 * unweighted
    assert masked_heatmap_mse(
        target,
        target,
        weights,
        foreground_weight=100.0,
    ).item() == 0.0


def test_slot_assignment_uses_weighted_heatmap_loss_configuration():
    target = {
        "keypoints": torch.tensor([[[16.0, 16.0]]]),
        "visible": torch.ones((1, 1)),
    }
    target_config = PoseTargetConfig(
        image_size=(32, 32),
        heatmap_size=(8, 8),
        sigma=1.0,
        max_objects=2,
        nkeypoints=1,
    )
    target_heatmaps, _weights, _mask, _dropped = build_batch_targets(
        [target],
        config=target_config,
        torch_module=torch,
        device=torch.device("cpu"),
    )
    object_heatmap = target_heatmaps[0, 0]
    outputs = {
        "heatmaps": torch.stack(
            (torch.zeros_like(object_heatmap), object_heatmap)
        ).unsqueeze(0),
        "objectness": torch.tensor([[4.0, 0.0]]),
    }

    _, _, objectness_only = slot_pose_losses(
        outputs,
        [target],
        config=PoseLossConfig(
            target=target_config,
            heatmap_weight=0.0,
            heatmap_foreground_weight=100.0,
        ),
        torch_module=torch,
        device=torch.device("cpu"),
        return_reduction=True,
    )
    _, _, heatmap_aware = slot_pose_losses(
        outputs,
        [target],
        config=PoseLossConfig(
            target=target_config,
            heatmap_weight=10.0,
            heatmap_foreground_weight=100.0,
        ),
        torch_module=torch,
        device=torch.device("cpu"),
        return_reduction=True,
    )

    assert objectness_only.assignments == ((0, 0, 0),)
    assert heatmap_aware.assignments == ((0, 1, 0),)


def test_worker_selects_layer_decay_for_existing_resume_checkpoint(tmp_path):
    service_mllib = {
        "gpu": False,
        "nkeypoints": 2,
        "vitpose": {
            "head": "topdown",
            "variant": "tiny",
            "image_size": [32, 32],
            "heatmap_size": [8, 8],
        },
    }
    (tmp_path / "checkpoint-1.pt").touch()
    (tmp_path / "solver-1.pt").touch()
    worker = DeepDetectWorker()
    worker.configure(WorkerContext(repository=str(tmp_path), mllib=service_mllib, raw={}))
    request = PoseTrainRequest(
        request={},
        request_params={},
        effective_mllib={"resume": True, "vitpose": {"head": "topdown"}},
        source="tensor",
        train_list=None,
        test_lists=[],
        train_tensor_batches=[],
        test_tensor_batches=[],
        options=PoseTrainOptions(
            iterations=1,
            test_interval=1,
            batch_size=1,
            iter_size=1,
            base_lr=0.001,
        ),
    )

    worker.configure_training_request(request)

    assert worker.config is not None
    assert worker.config.layer_decay == 0.9


def test_connector_prefetcher_waits_for_full_queue_instead_of_stopping():
    values = iter(["first", "second", "third", None])
    third_requested = threading.Event()

    def pull(*, reset_epoch: bool):
        value = next(values)
        if value == "third":
            third_requested.set()
        return value

    prefetcher = ConnectorBatchPrefetcher(
        pull,
        reset_epoch=True,
        prefetch_batches=2,
    )
    try:
        assert third_requested.wait(timeout=1.0)
        time.sleep(0.2)
        assert prefetcher.next() == "first"
        assert prefetcher.next() == "second"
        assert prefetcher.next() == "third"
    finally:
        prefetcher.close()


def test_targets_keep_duplicate_joint_ids_in_separate_slots():
    target = {
        "keypoints": torch.tensor(
            [
                [[8.0, 8.0]],
                [[48.0, 48.0]],
            ],
            dtype=torch.float32,
        ),
        "visible": torch.ones((2, 1), dtype=torch.float32),
    }
    config = PoseTargetConfig(
        image_size=(64, 64),
        heatmap_size=(16, 16),
        sigma=1.0,
        max_objects=2,
        nkeypoints=1,
    )

    heatmaps, weights, object_mask, dropped = build_batch_targets(
        [target],
        config=config,
        torch_module=torch,
        device=torch.device("cpu"),
    )

    assert dropped == 0
    assert object_mask.tolist() == [[1.0, 1.0]]
    assert weights.reshape(2).tolist() == [1.0, 1.0]
    peak0 = int(heatmaps[0, 0, 0].reshape(-1).argmax().item())
    peak1 = int(heatmaps[0, 1, 0].reshape(-1).argmax().item())
    assert peak0 != peak1


def test_tiny_vitpose_forward_loss_backward_is_finite():
    model_config = ViTPoseModelConfig(
        head="slots",
        image_size=(32, 32),
        heatmap_size=(8, 8),
        nkeypoints=3,
        max_objects=2,
        variant="tiny",
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        upsample=4,
        final_conv_kernel=3,
        num_deconv_layers=0,
        num_deconv_filters=(),
        num_deconv_kernels=(),
    )
    model = ViTPoseSlots(model_config)
    images = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    target = {
        "keypoints": torch.tensor(
            [[[8.0, 8.0], [16.0, 16.0], [-1.0, -1.0]]],
            dtype=torch.float32,
        ),
        "visible": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
    }

    outputs = model(images)
    losses, stats = slot_pose_losses(
        outputs,
        [target],
        config=PoseLossConfig(
            target=PoseTargetConfig(
                image_size=(32, 32),
                heatmap_size=(8, 8),
                sigma=1.0,
                max_objects=2,
                nkeypoints=3,
            )
        ),
        torch_module=torch,
        device=torch.device("cpu"),
    )
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"]).item()
    assert stats["assigned_objects"] == 1.0


def test_tiny_topdown_vitpose_forward_loss_and_inverse_decode():
    model_config = ViTPoseModelConfig(
        head="topdown",
        image_size=(32, 32),
        heatmap_size=(8, 8),
        nkeypoints=2,
        max_objects=1,
        variant="tiny",
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        upsample=4,
        final_conv_kernel=3,
        num_deconv_layers=0,
        num_deconv_filters=(),
        num_deconv_kernels=(),
    )
    model = ViTPoseTopDown(model_config)
    images = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    target = {
        "keypoints": torch.tensor([[[8.0, 8.0], [-1.0, -1.0]]]),
        "visible": torch.tensor([[1.0, 0.0]]),
    }
    outputs = model(images)
    assert tuple(outputs["heatmaps"].shape) == (1, 2, 8, 8)
    losses, stats = topdown_pose_losses(
        outputs,
        [target],
        config=PoseLossConfig(
            target=PoseTargetConfig(
                image_size=(32, 32),
                heatmap_size=(8, 8),
                sigma=1.0,
                max_objects=1,
                nkeypoints=2,
            )
        ),
        torch_module=torch,
        device=torch.device("cpu"),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"]).item()
    assert stats["assigned_objects"] == 1.0

    heatmaps = torch.zeros((1, 1, 4, 4))
    heatmaps[0, 0, 2, 1] = 1.0
    poses = decode_topdown_outputs(
        {"heatmaps": heatmaps},
        metas=[
            {
                "width": 4,
                "height": 4,
                "inverse_affine": [2.0, 0.0, 10.0, 0.0, 3.0, 20.0],
                "bbox": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
                "label": 2,
                "index": 0,
            }
        ],
        keypoint_threshold=0.1,
    )
    assert poses[0]["keypoints"][0]["x"] == pytest.approx(12.0)
    assert poses[0]["keypoints"][0]["y"] == pytest.approx(26.0)
    assert poses[0]["cat"] == "2"


def test_mae_style_weights_initialize_only_the_vit_backbone(tmp_path):
    model_config = ViTPoseModelConfig(
        head="topdown",
        image_size=(32, 32),
        heatmap_size=(8, 8),
        nkeypoints=2,
        max_objects=1,
        variant="tiny",
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        upsample=4,
        final_conv_kernel=3,
        num_deconv_layers=0,
        num_deconv_filters=(),
        num_deconv_kernels=(),
    )
    model = ViTPoseTopDown(model_config)
    head_before = model.keypoint_head.final_layer.weight.detach().clone()
    backbone_state = {
        name: torch.full_like(value, 0.125)
        for name, value in model.backbone.state_dict().items()
    }
    backbone_state["pos_embed"] = torch.full((1, 14 * 14 + 1, 32), 0.125)
    backbone_state["norm.weight"] = backbone_state.pop("last_norm.weight")
    backbone_state["norm.bias"] = backbone_state.pop("last_norm.bias")
    checkpoint = tmp_path / "mae-style.pth"
    torch.save({"model": backbone_state}, checkpoint)

    load_model_checkpoint(torch, model, checkpoint, device=torch.device("cpu"))

    assert torch.equal(
        model.backbone.patch_embed.proj.weight,
        torch.full_like(model.backbone.patch_embed.proj.weight, 0.125),
    )
    assert torch.equal(
        model.backbone.last_norm.weight,
        torch.full_like(model.backbone.last_norm.weight, 0.125),
    )
    assert torch.equal(model.keypoint_head.final_layer.weight, head_before)


def test_prediction_contract_accepts_keypoints():
    result = {
        "results": [
            {
                "uri": "image.jpg",
                "loss": 0.0,
                "probs": [0.9],
                "cats": ["pose"],
                "keypoints": [
                    {
                        "points": [
                            {"x": 1.0, "y": 2.0, "prob": 0.8, "valid": True},
                            {"x": -1.0, "y": -1.0, "prob": 0.0, "valid": False},
                        ]
                    }
                ],
            }
        ]
    }

    assert validate_prediction_result(result) == result


def test_coco_topdown_format_uses_deepdetect_bbox_order():
    keypoints = format_deepdetect_keypoint_line([2, 3, 2, 0, 0, 0], 2)
    line = format_deepdetect_topdown_line(
        {"id": 7, "bbox": [1, 2, 10, 20]},
        keypoint_line=keypoints,
        category_id=1,
        image_width=100,
        image_height=80,
    )
    assert line == "1 1 2 11 22 2 3 -1 -1"


def test_topdown_worker_predicts_connector_tensor_batch(tmp_path):
    worker = DeepDetectWorker()
    worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={
                "gpu": False,
                "nkeypoints": 2,
                "vitpose": {
                    "head": "topdown",
                    "variant": "tiny",
                    "image_size": [32, 32],
                    "heatmap_size": [8, 8],
                    "patch_size": 16,
                    "embed_dim": 32,
                    "depth": 1,
                    "num_heads": 4,
                    "drop_path_rate": 0.0,
                },
            },
            raw={},
        )
    )
    tensor_batch = {
        "kind": "tensor_batch",
        "inputs": [
            {
                "kind": "tensor_ref",
                "device": "cpu",
                "dtype": "float32",
                "shape": [1, 3, 32, 32],
                "layout": "strided",
                "storage": {
                    "type": "inline_test_stub",
                    "name": "test",
                    "offset": 0,
                    "nbytes": 0,
                    "values": [0.0] * (3 * 32 * 32),
                },
                "lifetime": {},
                "cuda": {},
            }
        ],
        "meta": {
            "sample_ids": [0],
            "instance_ids": [0],
            "labels": [3],
            "paths": ["image.jpg"],
            "widths": [32],
            "heights": [32],
            "original_widths": [64],
            "original_heights": [64],
            "bboxes": [{"xmin": 4.0, "ymin": 5.0, "xmax": 40.0, "ymax": 50.0}],
            "inverse_affines": [
                {"values": [2.0, 0.0, 4.0, 0.0, 2.0, 5.0]}
            ],
            "source_paths": ["image.jpg"],
            "source_count": 1,
        },
    }
    result = worker.predict(
        {
            "request": {
                "data": ["image.jpg"],
                "pose_sources": ["image.jpg"],
                "tensor_batch": tensor_batch,
                "parameters": {
                    "output": {"keypoint_threshold": 0.0}
                },
            }
        }
    )
    validate_prediction_result(result)
    assert result["results"][0]["cats"] == ["3"]
    assert result["results"][0]["bboxes"][0]["xmin"] == 4.0
    assert len(result["results"][0]["keypoints"][0]["points"]) == 2


def test_slot_worker_prediction_uses_source_image_coordinates(tmp_path):
    class FixedSlots(torch.nn.Module):
        def forward(self, images):
            heatmaps = images.new_zeros((len(images), 1, 1, 8, 8))
            heatmaps[:, 0, 0, 4, 2] = 1.0
            return {
                "heatmaps": heatmaps,
                "objectness": images.new_full((len(images), 1), 10.0),
            }

    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 96)).save(image_path)
    worker = DeepDetectWorker()
    worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={
                "gpu": False,
                "nkeypoints": 1,
                "max_objects": 1,
                "vitpose": {
                    "head": "slots",
                    "variant": "tiny",
                    "image_size": [32, 32],
                    "heatmap_size": [8, 8],
                    "patch_size": 16,
                    "embed_dim": 32,
                    "depth": 1,
                    "num_heads": 4,
                    "drop_path_rate": 0.0,
                    "max_objects": 1,
                },
            },
            raw={},
        )
    )
    worker.model = FixedSlots()

    result = worker.predict(
        {
            "request": {
                "data": [str(image_path)],
                "parameters": {
                    "output": {
                        "confidence_threshold": 0.0,
                        "keypoint_threshold": 0.0,
                    }
                },
            }
        }
    )

    point = result["results"][0]["keypoints"][0]["points"][0]
    assert point["x"] == pytest.approx((2.0 * 31.0 / 7.0) * (64.0 / 32.0))
    assert point["y"] == pytest.approx((4.0 * 31.0 / 7.0) * (96.0 / 32.0))


def test_topdown_evaluation_reports_globally_reduced_losses(tmp_path):
    class ZeroTopDown(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return {"heatmaps": images.new_zeros((len(images), 2, 8, 8))}

    worker = _configured_evaluation_worker(tmp_path, head="topdown")
    model = ZeroTopDown()
    worker.model = model
    targets = [
        {
            "keypoints": torch.tensor([[[8.0, 8.0], [-1.0, -1.0]]]),
            "visible": torch.tensor([[1.0, 0.0]]),
        },
        {
            "keypoints": torch.tensor([[[12.0, 12.0], [20.0, 20.0]]]),
            "visible": torch.tensor([[1.0, 1.0]]),
        },
    ]
    dataset = [
        (
            torch.zeros((3, 32, 32)),
            targets[0],
            _evaluation_meta(
                0,
                inverse_affine=[2.0, 0.0, 10.0, 0.0, 3.0, 20.0],
            ),
        ),
        (torch.zeros((3, 32, 32)), targets[1], _evaluation_meta(1)),
    ]
    events = []

    worker.evaluate_tensor(
        [dataset],
        reporter=WorkerReporter(lambda event, payload: events.append((event, payload))),
        iteration=7,
        torch=torch,
        cancellation=Cancellation(),
    )

    expected, _stats = topdown_pose_losses(
        {"heatmaps": torch.zeros((2, 2, 8, 8))},
        targets,
        config=worker.config.loss,
        torch_module=torch,
        device=torch.device("cpu"),
    )
    metrics = _metric_values(events)
    assert model.calls == 2
    assert metrics["loss_test0"] == pytest.approx(float(expected["loss"].item()))
    assert metrics["heatmap_loss_test0"] == pytest.approx(
        float(expected["heatmap_loss"].item())
    )
    assert metrics["visible_keypoints_test0"] == 3.0
    assert metrics["mean_keypoint_error_px_test0"] == pytest.approx(
        (
            math.hypot(16.0, 24.0)
            + math.hypot(12.0, 12.0)
            + math.hypot(20.0, 20.0)
        )
        / 3.0
    )
    assert metrics["pose_samples_test0"] == 2.0
    assert "objectness_loss_test0" not in metrics


def test_slot_evaluation_reports_objectness_loss(tmp_path):
    class ZeroSlots(torch.nn.Module):
        def forward(self, images):
            return {
                "heatmaps": images.new_zeros((len(images), 2, 2, 8, 8)),
                "objectness": images.new_zeros((len(images), 2)),
            }

    worker = _configured_evaluation_worker(tmp_path, head="slots")
    worker.model = ZeroSlots()
    target = {
        "keypoints": torch.tensor([[[8.0, 8.0], [-1.0, -1.0]]]),
        "visible": torch.tensor([[1.0, 0.0]]),
    }
    events = []

    worker.evaluate_tensor(
        [[(torch.zeros((3, 32, 32)), target, _evaluation_meta(0))]],
        reporter=WorkerReporter(lambda event, payload: events.append((event, payload))),
        iteration=7,
        torch=torch,
        cancellation=Cancellation(),
    )

    expected, _stats = slot_pose_losses(
        {
            "heatmaps": torch.zeros((1, 2, 2, 8, 8)),
            "objectness": torch.zeros((1, 2)),
        },
        [target],
        config=worker.config.loss,
        torch_module=torch,
        device=torch.device("cpu"),
    )
    metrics = _metric_values(events)
    assert metrics["loss_test0"] == pytest.approx(float(expected["loss"].item()))
    assert metrics["heatmap_loss_test0"] == pytest.approx(
        float(expected["heatmap_loss"].item())
    )
    assert metrics["objectness_loss_test0"] == pytest.approx(
        float(expected["objectness_loss"].item())
    )
    assert metrics["visible_keypoints_test0"] == 1.0
    assert metrics["mean_keypoint_error_px_test0"] == pytest.approx(
        math.hypot(8.0, 8.0)
    )


def _configured_evaluation_worker(tmp_path, *, head):
    worker = DeepDetectWorker()
    worker.configure(
        WorkerContext(
            repository=str(tmp_path),
            mllib={
                "gpu": False,
                "nkeypoints": 2,
                "max_objects": 2,
                "vitpose": {
                    "head": head,
                    "variant": "tiny",
                    "image_size": [32, 32],
                    "heatmap_size": [8, 8],
                    "patch_size": 16,
                    "embed_dim": 32,
                    "depth": 1,
                    "num_heads": 4,
                    "drop_path_rate": 0.0,
                    "max_objects": 2,
                },
            },
            raw={},
        )
    )
    return worker


def _evaluation_meta(index, *, inverse_affine=None):
    return {
        "index": index,
        "width": 32,
        "height": 32,
        "original_width": 32,
        "original_height": 32,
        "label": 1,
        "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": 31.0, "ymax": 31.0},
        "inverse_affine": inverse_affine or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def _metric_values(events):
    return {
        payload["name"]: payload["value"]
        for event, payload in events
        if event == "metric"
    }
