from __future__ import annotations

import json
import socket
import struct
from typing import Any

_HEADER = struct.Struct(">I")


class ProtocolError(RuntimeError):
    pass


def read_frame(sock: socket.socket) -> dict[str, Any]:
    header = _read_exact(sock, _HEADER.size)
    if not header:
        raise EOFError("worker socket closed")
    (size,) = _HEADER.unpack(header)
    if size <= 0:
        raise ProtocolError(f"invalid frame size: {size}")
    payload = _read_exact(sock, size)
    if len(payload) != size:
        raise EOFError("worker socket closed while reading frame")
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON frame") from error
    if not isinstance(message, dict):
        raise ProtocolError("frame JSON payload must be an object")
    return message


def write_frame(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)
