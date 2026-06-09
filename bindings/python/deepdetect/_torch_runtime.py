from __future__ import annotations

import importlib
import re


_TORCH_VERSION_RE = re.compile(r"^2\.12(?:\.|$|\+)")


def ensure_torch_runtime() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "deepdetect requires torch==2.12.* so libtorch can be loaded "
            "from the PyTorch wheel"
        ) from error

    version = str(getattr(torch, "__version__", ""))
    if not _TORCH_VERSION_RE.match(version):
        raise ImportError(
            "deepdetect was built for torch==2.12.*; "
            f"found torch {version or 'with an unknown version'}"
        )
