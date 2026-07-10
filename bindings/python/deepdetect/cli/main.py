from __future__ import annotations

import argparse
from pathlib import Path

import deepdetect

from .commands import run_inspect_models, run_job_status
from .inference import run_infer
from .profiles import PROFILES
from .training import run_train
from .utils import report_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepdetect")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="train a model")
    train_models = train.add_subparsers(dest="model", required=True)
    for model in PROFILES:
        _add_train_parser(train_models.add_parser(model, help=f"train {model}"))

    infer = subcommands.add_parser("infer", help="run inference")
    infer_models = infer.add_subparsers(dest="model", required=True)
    for model in PROFILES:
        _add_infer_parser(infer_models.add_parser(model, help=f"infer {model}"), model)

    job = subcommands.add_parser("job", help="inspect CLI run manifests")
    job_subcommands = job.add_subparsers(dest="job_command", required=True)
    status = job_subcommands.add_parser("status", help="read the latest run status")
    status.add_argument("run", type=Path, help="run directory or run.json path")
    status.add_argument("--output-format", choices=("json", "jsonl", "text"), default="json")
    status.set_defaults(func=run_job_status)

    inspect = subcommands.add_parser("inspect", help="inspect CLI capabilities")
    inspect_subcommands = inspect.add_subparsers(dest="inspect_command", required=True)
    models = inspect_subcommands.add_parser("models", help="list model profiles")
    models.add_argument("--output-format", choices=("json", "jsonl", "text"), default="json")
    models.set_defaults(func=run_inspect_models)
    return parser


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        help="override top-level config values, e.g. base_lr=0.001",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "jsonl", "text"),
        default=None,
        help="event output format",
    )


def _add_train_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_config(parser)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--service-name")
    parser.add_argument("--nclasses", type=int)
    parser.add_argument("--nkeypoints", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--iter-size",
        type=int,
        help="gradient accumulation steps before each optimizer update",
    )
    parser.add_argument("--base-lr", type=float)
    parser.add_argument("--test-interval", type=int)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gpuid",
        nargs="+",
        help="GPU id or ids to use, e.g. 0, 0 1, 0,1, or -1 for all GPUs",
    )
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        choices=("latest", "best"),
        default=None,
        help="resume training from the latest or best checkpoint in --repository",
    )
    parser.add_argument(
        "--terminal",
        choices=("verbose", "live"),
        default=None,
        help="terminal display mode for training progress",
    )
    parser.add_argument("--visdom", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visdom-server")
    parser.add_argument("--visdom-port", type=int)
    parser.add_argument("--visdom-base-url")
    parser.add_argument(
        "--visdom-results",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="upload sampled test-set prediction overlays to Visdom",
    )
    parser.add_argument("--visdom-results-count", type=int)
    parser.add_argument("--visdom-results-seed", type=int)
    parser.add_argument(
        "--dataset-check",
        choices=("full", "none"),
        default=None,
        help="dataset validation mode before training",
    )
    parser.add_argument(
        "--visdom-offline-ok",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--visdom-save", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--skip-mask-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.set_defaults(func=run_train)


def _add_infer_parser(parser: argparse.ArgumentParser, model: str) -> None:
    _add_common_config(parser)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--service-name")
    parser.add_argument("--nclasses", type=int)
    parser.add_argument("--nkeypoints", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gpuid",
        nargs="+",
        help="GPU id or ids to use, e.g. 0, 0 1, 0,1, or -1 for all GPUs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup", type=int)
    if PROFILES[model].task in {"detection", "keypoint"}:
        parser.add_argument("--confidence-threshold", type=float)
    if PROFILES[model].task == "detection":
        parser.add_argument("--best-bbox", type=int)
    parser.set_defaults(func=run_infer)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (
        deepdetect.DeepDetectError,
        OSError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return report_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
