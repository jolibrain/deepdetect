from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SEGMENTATION_CLASS_COLORS = (
    (255, 64, 64),
    (64, 160, 255),
    (76, 175, 80),
    (255, 193, 7),
    (156, 39, 176),
    (0, 188, 212),
    (255, 112, 67),
    (63, 81, 181),
    (139, 195, 74),
    (233, 30, 99),
    (121, 85, 72),
    (0, 150, 136),
    (205, 220, 57),
    (96, 125, 139),
    (255, 152, 0),
    (103, 58, 183),
)


def output_path_for(
    requested: Path, image_path: Path, *, multiple: bool, suffix: str
) -> Path:
    if multiple or requested.suffix == "" or requested.is_dir():
        return requested / f"{image_path.stem}{suffix}.png"
    return requested


def render_detections(
    image_path: Path, prediction: dict[str, Any], output_path: Path
) -> None:
    image = detection_overlay_image(image_path, prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_keypoints(
    image_path: Path, prediction: dict[str, Any], output_path: Path
) -> None:
    image = keypoint_overlay_image(image_path, prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_instance_masks(
    image_path: Path, prediction: dict[str, Any], overlay_path: Path
) -> list[Path]:
    image = Image.open(image_path).convert("RGBA")
    masks = instance_mask_images(prediction, image.size)
    overlay = image
    for index, mask in enumerate(masks, 1):
        color = _segmentation_class_color(index)
        layer = Image.new("RGBA", image.size, (*color, 0))
        layer.putalpha(mask.point(lambda value: 120 if value else 0))
        overlay = Image.alpha_composite(overlay, layer)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    mask_paths: list[Path] = []
    base = overlay_path.stem.removesuffix("_overlay")
    for index, mask in enumerate(masks, 1):
        mask_path = overlay_path.with_name(f"{base}_mask_{index:04d}.png")
        mask.save(mask_path)
        mask_paths.append(mask_path)
    overlay = overlay.convert("RGB")
    if overlay_path.suffix.lower() in {".jpg", ".jpeg"}:
        overlay.save(overlay_path, format="JPEG", quality=95)
    else:
        overlay.save(overlay_path)
    return mask_paths


def instance_mask_images(
    prediction: dict[str, Any], image_size: tuple[int, int]
) -> list[Image.Image]:
    masks: list[Image.Image] = []
    for index, detected_class in enumerate(prediction.get("classes", [])):
        raw_mask = detected_class.get("mask")
        if not isinstance(raw_mask, dict):
            raise ValueError(f"instance mask {index} must be an object")
        mask = decode_coco_rle_mask(raw_mask)
        if mask.size != image_size:
            raise ValueError(
                f"instance mask {index} is {mask.width}x{mask.height}, "
                f"expected {image_size[0]}x{image_size[1]}"
            )
        masks.append(mask)
    return masks


def decode_coco_rle_mask(payload: dict[str, Any]) -> Image.Image:
    encoding = payload.get("encoding", "coco_rle")
    if encoding != "coco_rle":
        raise ValueError(f"unsupported instance mask encoding: {encoding!r}")
    size = payload.get("size")
    counts = payload.get("counts")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
    ):
        raise ValueError("COCO RLE mask size must be [height, width]")
    if not isinstance(counts, list):
        raise ValueError("COCO RLE mask counts must be a list")
    height, width = int(size[0]), int(size[1])
    expected = height * width
    values = bytearray(expected)
    offset = 0
    bit = 0
    for count in counts:
        if not isinstance(count, int) or count < 0:
            raise ValueError("COCO RLE mask counts must be non-negative integers")
        end = offset + count
        if end > expected:
            raise ValueError("COCO RLE mask counts exceed its declared size")
        if bit:
            values[offset:end] = b"\x01" * count
        offset = end
        bit = 1 - bit
    if offset != expected:
        raise ValueError("COCO RLE mask counts do not match its declared size")
    # COCO RLE traverses each column before moving to the next one.
    return Image.frombytes("L", (height, width), bytes(values)).transpose(
        Image.Transpose.TRANSPOSE
    )


def keypoint_overlay_image(
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
    for pose_index, detected_class in enumerate(prediction.get("classes", [])):
        color = colors[pose_index % len(colors)]
        for point in detected_class.get("keypoints", []):
            if not bool(point.get("valid", True)):
                continue
            x = float(point.get("x", -1.0))
            y = float(point.get("y", -1.0))
            if x < 0.0 or y < 0.0:
                continue
            x *= x_scale
            y *= y_scale
            radius = 3
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=color,
            )
    return image


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
    mask.putpalette(_segmentation_palette())

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


def _segmentation_palette() -> list[int]:
    palette = [0, 0, 0]
    for class_index in range(1, 256):
        color = _segmentation_class_color(class_index)
        palette.extend(color)
    return palette


def _segmentation_class_color(class_index: int) -> tuple[int, int, int]:
    if 1 <= class_index <= len(SEGMENTATION_CLASS_COLORS):
        return SEGMENTATION_CLASS_COLORS[class_index - 1]
    hue = ((class_index - 1) * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)
