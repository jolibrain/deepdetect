from __future__ import annotations

from typing import Any


class DeepDetectError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        dd_code: int | None,
        message: str,
        response: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.dd_code = dd_code
        self.message = message
        self.response = response


class CapabilityError(DeepDetectError):
    pass
