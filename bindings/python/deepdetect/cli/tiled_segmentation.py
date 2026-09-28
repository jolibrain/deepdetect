"""Tiled semantic-segmentation inference and bounded visual output."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .events import EventWriter
from .utils import predictions_by_uri
from .visualize import _segmentation_palette


@dataclass(frozen=True)
class TiledImageResult:
    tiles: int
    inference_seconds: float


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _ownership_bounds(starts: list[int], tile: int, length: int) -> list[int]:
    centers = [start + min(tile, length - start) / 2 for start in starts]
    return [
        0,
        *(int((left + right) / 2) for left, right in zip(centers, centers[1:])),
        length,
    ]


def _prediction_array(
    prediction: dict[str, Any], key: str, width: int, height: int
) -> np.ndarray:
    if key == "vals":
        values = prediction.get("vals")
    else:
        confidences = prediction.get("confidences")
        values = confidences.get("best") if isinstance(confidences, dict) else None
    if values is None:
        raise ValueError(f"tiled segmentation prediction is missing {key}")
    array = np.asarray(values)
    if array.size != width * height:
        raise ValueError(
            f"tiled segmentation {key} contains {array.size} values for {width}x{height}"
        )
    return array.reshape(height, width)


def _atomic_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _preview_mask(
    mask: np.ndarray, size: tuple[int, int], nclasses: int
) -> Image.Image:
    height, width = mask.shape
    preview_width, preview_height = size
    if (width, height) == size:
        values = mask
    elif nclasses == 2:
        # Map every foreground pixel into its preview cell, including features
        # thinner than one preview pixel.
        values = np.zeros((preview_height, preview_width), dtype=np.uint8)
        rows, columns = np.nonzero(mask == 1)
        values[
            np.minimum(rows * preview_height // height, preview_height - 1),
            np.minimum(columns * preview_width // width, preview_width - 1),
        ] = 1
    else:
        values = np.asarray(
            Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)
        )
    preview = Image.fromarray(values, mode="P")
    preview.putpalette(_segmentation_palette())
    return preview


def run_tiled_image(
    *,
    service: Any,
    image_path: Path,
    output_dir: Path,
    predict_parameters: dict[str, Any],
    tile_width: int,
    tile_height: int,
    overlap: int,
    batch_size: int,
    nclasses: int,
    preview_max_side: int,
    confidence_maps: bool,
    writer: EventWriter,
    warmup: int = 0,
) -> TiledImageResult:
    with Image.open(image_path) as source:
        source = source.convert("RGB")
    width, height = source.size
    x_starts = _tile_starts(width, tile_width, overlap)
    y_starts = _tile_starts(height, tile_height, overlap)
    x_bounds = _ownership_bounds(x_starts, tile_width, width)
    y_bounds = _ownership_bounds(y_starts, tile_height, height)
    tiles = [
        (x, y, x_index, y_index)
        for y_index, y in enumerate(y_starts)
        for x_index, x in enumerate(x_starts)
    ]
    total_tiles = len(tiles)
    writer.emit(
        "tile_plan",
        image=str(image_path),
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tiles=total_tiles,
        batches=(total_tiles + batch_size - 1) // batch_size,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.uint16) if confidence_maps else None
    foreground = (
        np.zeros((height, width), dtype=np.uint16)
        if confidence_maps and nclasses == 2
        else None
    )
    inference_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="deepdetect-tiles-") as temporary:
        temporary_dir = Path(temporary)
        for batch_start in range(0, total_tiles, batch_size):
            batch = tiles[batch_start : batch_start + batch_size]
            paths: list[Path] = []
            for index, (x, y, _, _) in enumerate(batch, batch_start):
                crop = source.crop(
                    (x, y, min(x + tile_width, width), min(y + tile_height, height))
                )
                if crop.size != (tile_width, tile_height):
                    padded = Image.new("RGB", (tile_width, tile_height))
                    padded.paste(crop, (0, 0))
                    crop = padded
                path = temporary_dir / f"tile-{index:08d}.png"
                crop.save(path)
                paths.append(path)
            if batch_start == 0:
                for _ in range(warmup):
                    service.predict(paths, **predict_parameters)
            started = time.perf_counter()
            response = service.predict(paths, **predict_parameters)
            batch_seconds = time.perf_counter() - started
            inference_seconds += batch_seconds
            predictions = predictions_by_uri(paths, response.get("predictions", []))
            for (x, y, x_index, y_index), prediction in zip(batch, predictions):
                imgsize = prediction.get("imgsize")
                if not isinstance(imgsize, dict) or (
                    int(imgsize.get("width", -1)),
                    int(imgsize.get("height", -1)),
                ) != (tile_width, tile_height):
                    raise ValueError(
                        "tiled segmentation prediction has unexpected imgsize"
                    )
                values = _prediction_array(prediction, "vals", tile_width, tile_height)
                if (
                    not np.all(np.isfinite(values))
                    or np.any(values != np.floor(values))
                    or np.any(values < 0)
                    or np.any(values >= nclasses)
                ):
                    raise ValueError(
                        "tiled segmentation prediction has invalid class ids"
                    )
                left, right = x_bounds[x_index : x_index + 2]
                top, bottom = y_bounds[y_index : y_index + 2]
                tile_slice = np.s_[top - y : bottom - y, left - x : right - x]
                owned_values = values[tile_slice].astype(np.uint8)
                mask[top:bottom, left:right] = owned_values
                if confidence is not None:
                    probabilities = _prediction_array(
                        prediction, "confidences.best", tile_width, tile_height
                    )[tile_slice]
                    if (
                        not np.all(np.isfinite(probabilities))
                        or np.any(probabilities < 0)
                        or np.any(probabilities > 1)
                    ):
                        raise ValueError(
                            "tiled segmentation prediction has invalid confidences"
                        )
                    confidence[top:bottom, left:right] = np.rint(
                        probabilities * 65535
                    ).astype(np.uint16)
                    if foreground is not None:
                        foreground[top:bottom, left:right] = np.rint(
                            np.where(
                                owned_values == 1, probabilities, 1 - probabilities
                            )
                            * 65535
                        ).astype(np.uint16)
            for path in paths:
                path.unlink()
            writer.emit(
                "tile_progress",
                image=str(image_path),
                processed=min(batch_start + batch_size, total_tiles),
                total_tiles=total_tiles,
                batch_time_ms=batch_seconds * 1000,
            )

    output_dir = Path(output_dir)
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    preview_path = output_dir / f"{image_path.stem}_overlay_preview.png"
    mask_image = Image.fromarray(mask, mode="P")
    mask_image.putpalette(_segmentation_palette())
    _atomic_png(mask_image, mask_path)
    writer.emit("artifact", kind="mask", image=str(image_path), path=str(mask_path))

    scale = min(1.0, preview_max_side / max(width, height))
    preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    preview_mask = _preview_mask(mask, preview_size, nclasses)
    base = source.resize(preview_size, Image.Resampling.BILINEAR).convert("RGBA")
    color = preview_mask.convert("RGBA")
    color.putalpha(
        Image.fromarray(
            np.where(np.asarray(preview_mask) == 0, 0, 120).astype(np.uint8)
        )
    )
    overlay = Image.alpha_composite(base, color).convert("RGB")
    _atomic_png(overlay, preview_path)
    writer.emit(
        "artifact", kind="overlay", image=str(image_path), path=str(preview_path)
    )

    if confidence is not None:
        confidence_path = output_dir / f"{image_path.stem}_confidence.png"
        _atomic_png(Image.fromarray(confidence), confidence_path)
        writer.emit(
            "artifact",
            kind="confidence",
            image=str(image_path),
            path=str(confidence_path),
        )
    if foreground is not None:
        foreground_path = output_dir / f"{image_path.stem}_foreground_probability.png"
        _atomic_png(Image.fromarray(foreground), foreground_path)
        writer.emit(
            "artifact",
            kind="foreground_probability",
            image=str(image_path),
            path=str(foreground_path),
        )

    classes, counts = np.unique(mask, return_counts=True)
    writer.emit(
        "prediction",
        image=str(image_path),
        time_ms=inference_seconds * 1000,
        width=width,
        height=height,
        tiles=total_tiles,
        class_histogram={
            str(int(label)): int(count) for label, count in zip(classes, counts)
        },
    )
    return TiledImageResult(tiles=total_tiles, inference_seconds=inference_seconds)
