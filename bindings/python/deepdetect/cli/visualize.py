from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

from PIL import Image, ImageDraw


def output_path_for(
    requested: Path, image_path: Path, *, multiple: bool, suffix: str
) -> Path:
    if multiple or requested.suffix == "":
        return requested / f"{image_path.stem}{suffix}.png"
    return requested


def render_detections(
    image_path: Path, prediction: dict[str, Any], output_path: Path
) -> None:
    image = detection_overlay_image(image_path, prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def detection_overlay_image(
    image_path: Path,
    prediction: dict[str, Any],
    *,
    size: tuple[int, int] | None = None,
    coordinate_size: tuple[int, int] | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        image = image.resize(size)
    x_scale = 1.0
    y_scale = 1.0
    if coordinate_size is not None:
        coordinate_width, coordinate_height = coordinate_size
        if coordinate_width > 0 and coordinate_height > 0:
            x_scale = image.width / coordinate_width
            y_scale = image.height / coordinate_height
    draw = ImageDraw.Draw(image)
    colors = ("#00e5ff", "#ffca28", "#66bb6a", "#ef5350")
    for index, detected_class in enumerate(prediction.get("classes", [])):
        bbox = detected_class.get("bbox", {})
        raw_box = (
            float(bbox["xmin"]),
            float(bbox["ymin"]),
            float(bbox["xmax"]),
            float(bbox["ymax"]),
        )
        if max(abs(value) for value in raw_box) <= 1.5:
            box = (
                raw_box[0] * image.width,
                raw_box[1] * image.height,
                raw_box[2] * image.width,
                raw_box[3] * image.height,
            )
        else:
            box = (
                raw_box[0] * x_scale,
                raw_box[1] * y_scale,
                raw_box[2] * x_scale,
                raw_box[3] * y_scale,
            )
        box = _clamp_box(box, image.size)
        if box is None:
            continue
        label = str(detected_class.get("cat", "class"))
        color = colors[_stable_color_index(label, len(colors))]
        draw.rectangle(box, outline=color, width=3)
        probability = detected_class.get("prob")
        if probability is not None:
            label = f"{label} {float(probability):.3f}"
        draw.text((box[0] + 3, box[1] + 3), label, fill=color)
    return image


def _stable_color_index(value: str, count: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % count


def _clamp_box(
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    width, height = size
    xmin = max(0.0, min(float(width - 1), min(box[0], box[2])))
    ymin = max(0.0, min(float(height - 1), min(box[1], box[3])))
    xmax = max(0.0, min(float(width - 1), max(box[0], box[2])))
    ymax = max(0.0, min(float(height - 1), max(box[1], box[3])))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def render_segmentation(
    image_path: Path,
    prediction: dict[str, Any],
    mask_path: Path,
    overlay_path: Path,
) -> None:
    mask, overlay = segmentation_overlay_images(image_path, prediction)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)
    overlay.save(overlay_path)


def segmentation_overlay_images(
    image_path: Path,
    prediction: dict[str, Any],
    *,
    original_size: bool = False,
) -> tuple[Image.Image, Image.Image]:
    imgsize = prediction.get("imgsize", {})
    width = int(imgsize["width"])
    height = int(imgsize["height"])
    values = prediction["vals"]
    if len(values) != width * height:
        raise ValueError(
            f"segmentation contains {len(values)} values for {width}x{height}"
        )

    class_values = bytes(int(value) for value in values)
    mask = Image.frombytes("P", (width, height), class_values)
    mask.putpalette([0, 0, 0, 255, 64, 64] + [0, 0, 0] * 254)

    image = Image.open(image_path).convert("RGBA")
    if original_size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    color_mask = mask.convert("RGBA")
    color_mask.putalpha(
        Image.frombytes(
            "L",
            mask.size,
            bytes(0 if value == 0 else 120 for value in mask.tobytes()),
        )
    )
    if not original_size:
        image = image.resize((width, height))
    overlay = Image.alpha_composite(image, color_mask).convert("RGB")
    return mask, overlay
