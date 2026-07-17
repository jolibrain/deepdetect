"""JSON-first command line interface for read-only training observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .plots import render_plot
from .reader import RunReader, artifact_dict, plot_dict, point_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training-observe")
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summary", help="summarize a local training run")
    summary.add_argument("run", type=Path)

    metrics = commands.add_parser("metrics", help="read raw scalar metric points")
    metrics.add_argument("run", type=Path)
    metrics.add_argument("--name", action="append", dest="names")
    metrics.add_argument("--from-step", type=float)
    metrics.add_argument("--to-step", type=float)

    plots = commands.add_parser("plots", help="list renderable metric plots")
    plots.add_argument("run", type=Path)

    render = commands.add_parser("render", help="render current metrics to PNG")
    render.add_argument("run", type=Path)
    selector = render.add_mutually_exclusive_group(required=True)
    selector.add_argument("--plot")
    selector.add_argument("--all", action="store_true")
    render.add_argument("--output", type=Path)
    render.add_argument("--output-dir", type=Path)

    artifacts = commands.add_parser("artifacts", help="list saved visual artifacts")
    artifacts.add_argument("run", type=Path)
    artifacts.add_argument("--kind")
    artifacts.add_argument("--split")
    artifacts.add_argument("--step", default=None)
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _latest_step(reader: RunReader) -> float | None:
    values = [artifact.step for artifact in reader.artifacts() if artifact.step is not None]
    return max(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        reader = RunReader(args.run)
        if args.command == "summary":
            _emit(reader.summary())
            return 0
        if args.command == "metrics":
            points = reader.metric_points(
                names=set(args.names) if args.names else None,
                from_step=args.from_step,
                to_step=args.to_step,
            )
            _emit({"root": str(reader.root), "metrics": [point_dict(point) for point in points]})
            return 0
        if args.command == "plots":
            _emit({"root": str(reader.root), "plots": [plot_dict(plot) for plot in reader.plots()]})
            return 0
        if args.command == "render":
            plots = reader.plots()
            if args.plot:
                if args.output is None or args.output_dir is not None:
                    raise ValueError("--plot requires --output and does not accept --output-dir")
                plot = next((candidate for candidate in plots if candidate.id == args.plot), None)
                if plot is None:
                    raise ValueError(f"unknown plot: {args.plot}")
                output = render_plot(plot, reader.metric_points(), args.output)
                _emit({"root": str(reader.root), "rendered": [{"plot": plot.id, "path": str(output)}]})
                return 0
            if args.output_dir is None or args.output is not None:
                raise ValueError("--all requires --output-dir and does not accept --output")
            output_dir = args.output_dir.expanduser().resolve()
            rendered = []
            points = reader.metric_points()
            for plot in plots:
                output = render_plot(plot, points, output_dir / f"{plot.id}.png")
                rendered.append({"plot": plot.id, "path": str(output)})
            _emit({"root": str(reader.root), "rendered": rendered})
            return 0
        step = _latest_step(reader) if args.step == "latest" else (
            float(args.step) if args.step is not None else None
        )
        artifacts = reader.artifacts(kind=args.kind, split=args.split, step=step)
        _emit({"root": str(reader.root), "artifacts": [artifact_dict(item) for item in artifacts]})
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        _emit({"error": type(error).__name__, "message": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
