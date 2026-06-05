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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-class YOLOX nano model with DeepDetect."
    )
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "yolox/yolox-nano_cls2.pt",
    )
    parser.add_argument(
        "--repository", type=Path, default=Path("deepdetect-models/yolox-train")
    )
    parser.add_argument("--service-name", default="python-yolox-train")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.0001)
    parser.add_argument("--test-interval", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch size must be positive")
    if args.poll_interval <= 0:
        raise ValueError("poll interval must be positive")
    stage_model(args.weights, args.repository)

    dd = deepdetect.DeepDetect()
    configure_gpu_compatibility(dd.build_info, requested=args.gpu)
    with dd.create_service(
        args.service_name,
        model={"repository": str(args.repository.resolve())},
        mllib="torch",
        description="two-class YOLOX nano training",
        input_parameters={
            "connector": "image",
            "height": 640,
            "width": 640,
            "rgb": True,
            "bbox": True,
            "db": False,
        },
        mllib_parameters={
            "template": "yolox",
            "gpu": args.gpu,
            "nclasses": 2,
        },
        output_parameters={},
    ) as service:
        result = service.train(
            [args.train_data.resolve(), args.test_data.resolve()],
            asynchronous=args.asynchronous,
            input_parameters={"seed": 12347, "db": False, "shuffle": True},
            mllib_parameters={
                "gpu": args.gpu,
                "solver": {
                    "iterations": args.iterations,
                    "base_lr": args.base_lr,
                    "iter_size": 2,
                    "solver_type": "ADAM",
                    "test_interval": args.test_interval,
                },
                "net": {
                    "batch_size": args.batch_size,
                    "test_batch_size": args.batch_size,
                    "reg_weight": 0.5,
                },
                "resume": False,
                "mirror": True,
                "rotate": True,
                "crop_size": 512,
                "test_crop_samples": 10,
                "cutout": 0.1,
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
            output_parameters={"measure": ["map-05", "map-50", "map-90"]},
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
