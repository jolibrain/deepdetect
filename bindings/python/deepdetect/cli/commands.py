from __future__ import annotations

from typing import Any

from .events import EventWriter
from .profiles import PROFILES
from .runs import load_run


def run_job_status(args: Any) -> int:
    writer = EventWriter(output_format=args.output_format)
    manifest = load_run(args.run)
    writer.emit(
        "training_status",
        run_id=manifest.data.get("run_id"),
        status=manifest.data.get("status"),
        last_status=manifest.data.get("last_status", {}),
        run_dir=str(manifest.run_dir),
    )
    return 0


def run_inspect_models(args: Any) -> int:
    writer = EventWriter(output_format=args.output_format)
    for profile in PROFILES.values():
        writer.emit(
            "model",
            name=profile.name,
            task=profile.task,
            description=profile.description,
            default_nclasses=profile.default_nclasses,
        )
    return 0
