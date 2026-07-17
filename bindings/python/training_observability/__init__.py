"""Portable, read-only training-run observability helpers.

The package deliberately has no dependency on DeepDetect.  A run may expose a
``training-observability/v1`` manifest, or be recognised through the legacy
DeepDetect layout.
"""

from .reader import Artifact, MetricPoint, PlotSpec, RunReader
from .writer import RunWriter, append_artifact, write_run_manifest

__all__ = [
    "Artifact",
    "MetricPoint",
    "PlotSpec",
    "RunReader",
    "RunWriter",
    "append_artifact",
    "write_run_manifest",
]

__version__ = "0.1.0"
