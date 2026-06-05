#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import deepdetect

from _binding_example_utils import (
    DEFAULT_MODEL_ROOT,
    configure_gpu_compatibility,
    print_json,
    report_error,
    run_training_job,
    stage_model,
    validate_segmentation_lists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SegFormer-B0 model with DeepDetect."
    )
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "segformer/segformer-b0-cls2.pt",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("deepdetect-models/segformer-train"),
    )
    parser.add_argument("--service-name", default="python-segformer-train")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--nclasses",
        type=int,
        required=True,
        help="class count; must match both mask values and the model head",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.0001)
    parser.add_argument("--test-interval", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--skip-mask-validation",
        action="store_true",
        help="skip checking that mask pixels fit within --nclasses",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch size must be positive")
    if args.nclasses <= 0:
        raise ValueError("number of classes must be positive")
    if args.poll_interval <= 0:
        raise ValueError("poll interval must be positive")
    if not args.skip_mask_validation:
        validate_segmentation_lists(
            [args.train_data, args.test_data], args.nclasses
        )
    stage_model(args.weights, args.repository)

    dd = deepdetect.DeepDetect()
    configure_gpu_compatibility(dd.build_info, requested=args.gpu)
    with dd.create_service(
        args.service_name,
        model={"repository": str(args.repository.resolve())},
        mllib="torch",
        description=f"{args.nclasses}-class SegFormer-B0 training",
        input_parameters={
            "connector": "image",
            "width": 480,
            "height": 480,
            "db": False,
            "segmentation": True,
        },
        mllib_parameters={
            "gpu": args.gpu,
            "nclasses": args.nclasses,
            "segmentation": True,
        },
        output_parameters={},
    ) as service:
        result = service.train(
            [args.train_data.resolve(), args.test_data.resolve()],
            asynchronous=args.asynchronous,
            input_parameters={
                "seed": 12345,
                "db": False,
                "shuffle": True,
                "segmentation": True,
                "scale": 0.0039,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            mllib_parameters={
                "gpu": args.gpu,
                "solver": {
                    "iterations": args.iterations,
                    "base_lr": args.base_lr,
                    "iter_size": 1,
                    "solver_type": "ADAM",
                    "test_interval": args.test_interval,
                },
                "net": {"batch_size": args.batch_size},
                "resume": False,
                "mirror": True,
                "rotate": True,
                "crop_size": 224,
                "cutout": 0.5,
                "geometry": {
                    "prob": 0.1,
                    "persp_horizontal": True,
                    "persp_vertical": True,
                    "zoom_in": True,
                    "zoom_out": True,
                    "pad_mode": "constant",
                },
                "noise": {"prob": 0.01},
                "distort": {"prob": 0.01},
            },
            output_parameters={"measure": ["meaniou", "acc"]},
        )
        if isinstance(result, deepdetect.TrainingJob):
            result = run_training_job(
                result, timeout=args.timeout, poll_interval=args.poll_interval
            )
        print_json(result)
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        return 130
    except (
        deepdetect.DeepDetectError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return report_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
