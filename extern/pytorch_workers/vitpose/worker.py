from __future__ import annotations

import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from vitpose_worker.worker_impl import DeepDetectWorker  # noqa: E402,F401
