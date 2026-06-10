from __future__ import annotations

from pathlib import Path
from typing import Any

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
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("#00e5ff", "#ffca28", "#66bb6a", "#ef5350")
    for index, detected_class in enumerate(prediction.get("classes", [])):
        bbox = detected_class.get("bbox", {})
        box = tuple(float(bbox[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        label = str(detected_class.get("cat", "class"))
        probability = detected_class.get("prob")
        if probability is not None:
            label = f"{label} {float(probability):.3f}"
        draw.text((box[0] + 3, box[1] + 3), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_segmentation(
    image_path: Path,
    prediction: dict[str, Any],
    mask_path: Path,
    overlay_path: Path,
) -> None:
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
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)

    color_mask = mask.convert("RGBA")
    color_mask.putalpha(
        Image.frombytes(
            "L",
            (width, height),
            bytes(0 if int(value) == 0 else 120 for value in values),
        )
    )
    image = Image.open(image_path).convert("RGBA").resize((width, height))
    Image.alpha_composite(image, color_mask).convert("RGB").save(overlay_path)

