from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VITPOSE_ROOT = ROOT / "extern" / "pytorch_workers" / "vitpose"
if str(VITPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(VITPOSE_ROOT))
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

torch = pytest.importorskip("torch")

from deepdetect.pytorch_worker.sdk import WorkerContext, validate_prediction_result
from coco_keypoints_to_dd import (
    format_deepdetect_keypoint_line,
    format_deepdetect_topdown_line,
)
from vitpose_worker.assignment import hungarian_assign
from vitpose_worker.checkpoint import load_model_checkpoint
from vitpose_worker.decode import decode_topdown_outputs
from vitpose_worker.losses import PoseLossConfig, slot_pose_losses, topdown_pose_losses
from vitpose_worker.model import ViTPoseModelConfig, ViTPoseSlots, ViTPoseTopDown
from vitpose_worker.targets import PoseTargetConfig, build_batch_targets
from vitpose_worker.worker_impl import DeepDetectWorker
from vitpose_worker.worker_impl import ConnectorBatchPrefetcher


def test_hungarian_assignment_is_permutation_invariant():
    assert hungarian_assign([[4.0, 1.0], [1.0, 4.0]]) == [(1, 0), (0, 1)]


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
