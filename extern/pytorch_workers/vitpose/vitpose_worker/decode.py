from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def decode_pose_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    image_sizes: list[tuple[int, int]],
    objectness_threshold: float,
    keypoint_threshold: float,
) -> list[list[dict[str, Any]]]:
    heatmaps = outputs["heatmaps"].detach()
    objectness = outputs["objectness"].detach().sigmoid()
    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(int(heatmaps.shape[0])):
        image_width, image_height = image_sizes[batch_index]
        sample: list[dict[str, Any]] = []
        for slot_index in range(int(heatmaps.shape[1])):
            object_score = float(objectness[batch_index, slot_index].cpu().item())
            if object_score < objectness_threshold:
                continue
            points = []
            for joint_index in range(int(heatmaps.shape[2])):
                x, y, confidence, valid = decode_keypoint(
                    heatmaps[batch_index, slot_index, joint_index],
                    image_size=(image_width, image_height),
                    threshold=keypoint_threshold,
                )
                points.append(
                    {
                        "x": x,
                        "y": y,
                        "prob": confidence,
                        "valid": valid,
                    }
                )
            sample.append(
                {
                    "cat": "pose",
                    "prob": object_score,
                    "keypoints": points,
                }
            )
        sample.sort(key=lambda item: float(item["prob"]), reverse=True)
        decoded.append(sample)
    return decoded


def decode_topdown_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    metas: list[dict[str, Any]],
    keypoint_threshold: float,
) -> list[dict[str, Any]]:
    heatmaps = outputs["heatmaps"].detach()
    decoded: list[dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        inverse = meta.get("inverse_affine")
        if not isinstance(inverse, list) or len(inverse) != 6:
            raise ValueError("top-down prediction metadata requires inverse_affine")
        points = []
        confidences = []
        for joint_index in range(int(heatmaps.shape[1])):
            x, y, confidence, valid = decode_keypoint(
                heatmaps[batch_index, joint_index],
                image_size=(int(meta["width"]), int(meta["height"])),
                threshold=keypoint_threshold,
            )
            if valid:
                x, y = (
                    inverse[0] * x + inverse[1] * y + inverse[2],
                    inverse[3] * x + inverse[4] * y + inverse[5],
                )
                confidences.append(confidence)
            points.append({"x": x, "y": y, "prob": confidence, "valid": valid})
        decoded.append(
            {
                "cat": str(int(meta.get("label", 1))),
                "prob": 1.0,
                "bbox": meta.get("bbox"),
                "keypoints": points,
                "source_index": int(meta["index"]),
                "instance_id": int(meta.get("instance_id", 0)),
            }
        )
    return decoded


def decode_keypoint(
    heatmap: torch.Tensor,
    *,
    image_size: tuple[int, int],
    threshold: float,
) -> tuple[float, float, float, bool]:
    flat = heatmap.reshape(-1)
    confidence_tensor, index_tensor = torch.max(flat, dim=0)
    confidence = float(confidence_tensor.cpu().item())
    if confidence < threshold:
        return -1.0, -1.0, confidence, False
    index = int(index_tensor.cpu().item())
    heatmap_width = int(heatmap.shape[1])
    heatmap_height = int(heatmap.shape[0])
    px = index % heatmap_width
    py = index // heatmap_width
    image_width, image_height = image_size
    if heatmap_width > 1:
        x = float(px) * (float(image_width) - 1.0) / (float(heatmap_width) - 1.0)
    else:
        x = 0.0
    if heatmap_height > 1:
        y = float(py) * (float(image_height) - 1.0) / (float(heatmap_height) - 1.0)
    else:
        y = 0.0
    return x, y, confidence, True


def connector_predictions(
    image_paths: list[Path],
    decoded: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results = []
    for image_path, poses in zip(image_paths, decoded):
        result = {
            "uri": str(image_path),
            "loss": 0.0,
            "probs": [float(pose["prob"]) for pose in poses],
            "cats": [str(pose.get("cat", "pose")) for pose in poses],
            "keypoints": [{"points": pose["keypoints"]} for pose in poses],
        }
        if poses and all(pose.get("bbox") is not None for pose in poses):
            result["bboxes"] = [pose["bbox"] for pose in poses]
        results.append(result)
    return results


def prediction_sample(meta: dict[str, Any], poses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "index": int(meta["index"]),
        "imgsize": {"width": int(meta["width"]), "height": int(meta["height"])},
        "classes": poses,
    }
