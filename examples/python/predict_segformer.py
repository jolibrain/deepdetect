#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import deepdetect

from _binding_example_utils import (
    DEFAULT_MODEL_ROOT,
    output_path_for,
    print_json,
    render_segmentation,
    report_error,
    stage_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-class SegFormer-B0 inference with DeepDetect."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "segformer/segformer-b0-cls2.pt",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("deepdetect-models/segformer-predict"),
    )
    parser.add_argument("--service-name", default="python-segformer-predict")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    for image in args.images:
        if not image.is_file():
            raise FileNotFoundError(f"input image not found: {image}")
    stage_model(args.weights, args.repository)

    dd = deepdetect.DeepDetect()
    with dd.create_service(
        args.service_name,
        model={"repository": str(args.repository.resolve())},
        mllib="torch",
        description="two-class SegFormer-B0 inference",
        input_parameters={
            "connector": "image",
            "width": 480,
            "height": 480,
            "segmentation": True,
        },
        mllib_parameters={
            "gpu": args.gpu,
            "nclasses": 2,
            "segmentation": True,
        },
        output_parameters={},
    ) as service:
        result = service.predict(
            [image.resolve() for image in args.images],
            input_parameters={
                "height": 480,
                "width": 480,
                "scale": 0.0039,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            output_parameters={
                "segmentation": True,
                "confidences": ["best"],
            },
        )
        print_json(result)

        if args.output is not None:
            predictions = result.get("predictions", [])
            if len(predictions) != len(args.images):
                raise ValueError("DeepDetect returned an unexpected prediction count")
            multiple = len(args.images) > 1
            for image, prediction in zip(args.images, predictions):
                overlay_path = output_path_for(
                    args.output, image, multiple=multiple, suffix="_overlay"
                )
                mask_path = overlay_path.with_name(f"{overlay_path.stem}_mask.png")
                render_segmentation(image, prediction, mask_path, overlay_path)
                print(f"wrote {mask_path}")
                print(f"wrote {overlay_path}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (
        deepdetect.DeepDetectError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return report_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
