#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import deepdetect

from _binding_example_utils import (
    DEFAULT_MODEL_ROOT,
    output_path_for,
    print_json,
    render_detections,
    report_error,
    stage_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-class YOLOX nano inference with DeepDetect."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "yolox/yolox-nano_cls2.pt",
    )
    parser.add_argument(
        "--repository", type=Path, default=Path("deepdetect-models/yolox-predict")
    )
    parser.add_argument("--service-name", default="python-yolox-predict")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--best-bbox", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    if args.best_bbox is not None and args.best_bbox <= 0:
        raise ValueError("best bbox must be positive")
    for image in args.images:
        if not image.is_file():
            raise FileNotFoundError(f"input image not found: {image}")
    stage_model(args.weights, args.repository)

    dd = deepdetect.DeepDetect()
    with dd.create_service(
        args.service_name,
        model={"repository": str(args.repository.resolve())},
        mllib="torch",
        description="two-class YOLOX nano inference",
        input_parameters={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
        },
        mllib_parameters={
            "template": "yolox",
            "gpu": args.gpu,
            "nclasses": 2,
        },
        output_parameters={},
    ) as service:
        output_parameters: dict[str, object] = {
            "bbox": True,
            "confidence_threshold": args.confidence_threshold,
        }
        if args.best_bbox is not None:
            output_parameters["best_bbox"] = args.best_bbox
        result = service.predict(
            [image.resolve() for image in args.images],
            input_parameters={"height": 640, "width": 640},
            output_parameters=output_parameters,
        )
        print_json(result)

        if args.output is not None:
            predictions = result.get("predictions", [])
            if len(predictions) != len(args.images):
                raise ValueError("DeepDetect returned an unexpected prediction count")
            multiple = len(args.images) > 1
            for image, prediction in zip(args.images, predictions):
                output_path = output_path_for(
                    args.output, image, multiple=multiple, suffix="_detections"
                )
                render_detections(image, prediction, output_path)
                print(f"wrote {output_path}")
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
