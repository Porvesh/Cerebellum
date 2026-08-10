"""Versioned framed protocol shared by the C++ runtime and Python worker."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import BinaryIO

import numpy as np

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024

_FRAME = struct.Struct(">I")
_HELLO = struct.Struct(">4sHBBIHHI")
_REQUEST = struct.Struct(">4sHBBQqqII")
_RESPONSE = struct.Struct(">4sHBBQqQqIHHI")

HELLO_MAGIC = b"CBHI"
REQUEST_MAGIC = b"CBRQ"
RESPONSE_MAGIC = b"CBRS"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WireRequest:
    request_id: int
    first_step: int
    last_emitted_step: int
    action_count: int
    stitching: int
    seed: int


@dataclass(frozen=True, slots=True)
class WireResponse:
    request_id: int
    first_step: int
    observation_sequence: int
    capture_ns: int
    model_actions: np.ndarray
    robot_actions: np.ndarray


def read_frame(stream: BinaryIO) -> bytes | None:
    header = _read_exact(stream, _FRAME.size, eof_ok=True)
    if header is None:
        return None
    (size,) = _FRAME.unpack(header)
    if size > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is too large: {size} bytes")
    return _read_exact(stream, size)


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is too large: {len(payload)} bytes")
    stream.write(_FRAME.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def encode_hello(*, chunk_size: int, model_dim: int, robot_dim: int) -> bytes:
    return _HELLO.pack(
        HELLO_MAGIC, PROTOCOL_VERSION, 0, 0, chunk_size, model_dim, robot_dim, 0
    )


def encode_hello_error(message: str) -> bytes:
    detail = message.encode("utf-8")
    return _HELLO.pack(HELLO_MAGIC, PROTOCOL_VERSION, 1, 0, 0, 0, 0, len(detail)) + detail


def decode_request(payload: bytes) -> WireRequest:
    if len(payload) != _REQUEST.size:
        raise ProtocolError(f"request has {len(payload)} bytes, expected {_REQUEST.size}")
    magic, version, stitching, _reserved, request_id, first, last, count, seed = (
        _REQUEST.unpack(payload)
    )
    if magic != REQUEST_MAGIC:
        raise ProtocolError("bad request magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    return WireRequest(request_id, first, last, count, stitching, seed)


def encode_response(response: WireResponse) -> bytes:
    model = np.asarray(response.model_actions, dtype=np.float32)
    robot = np.asarray(response.robot_actions, dtype=np.float32)
    if model.ndim != 2 or robot.ndim != 2 or model.shape[0] != robot.shape[0]:
        raise ProtocolError("response action arrays have incompatible shapes")
    count, model_dim = model.shape
    robot_dim = robot.shape[1]
    header = _RESPONSE.pack(
        RESPONSE_MAGIC,
        PROTOCOL_VERSION,
        0,
        0,
        response.request_id,
        response.first_step,
        response.observation_sequence,
        response.capture_ns,
        count,
        model_dim,
        robot_dim,
        0,
    )
    return header + model.astype(">f4", copy=False).tobytes() + robot.astype(">f4", copy=False).tobytes()


def encode_response_error(request_id: int, message: str) -> bytes:
    detail = message.encode("utf-8")
    return _RESPONSE.pack(
        RESPONSE_MAGIC, PROTOCOL_VERSION, 1, 0, request_id, 0, 0, 0, 0, 0, 0, len(detail)
    ) + detail


def _read_exact(stream: BinaryIO, size: int, *, eof_ok: bool = False) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            if eof_ok and not data:
                return None
            raise EOFError("stream ended in the middle of a frame")
        data.extend(part)
    return bytes(data)
