"""Versioned framed protocol shared by the C++ runtime and Python worker."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from time import monotonic_ns
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

PROTOCOL_VERSION = 4
MAX_FRAME_BYTES = 16 * 1024 * 1024

_FRAME = struct.Struct(">I")
_HELLO = struct.Struct(">4sHBBIHHHHI")
_HELLO_IMAGE = struct.Struct(">HHHH")
_REQUEST = struct.Struct(">4sHBBQqqIIQqIHI")
_IMAGE = struct.Struct(">HHHHI")
_RESPONSE = struct.Struct(">4sHBBQqQqIHHQQQQI")
_RESPONSE_ENCODE_NS_OFFSET = struct.calcsize(">4sHBBQqQqIHHQQQ")

HELLO_MAGIC = b"CBHI"
REQUEST_MAGIC = b"CBRQ"
RESPONSE_MAGIC = b"CBRS"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WireImage:
    feature_name: str
    channels: int
    height: int
    width: int
    pixels: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class WireImageSpec:
    feature_name: str
    channels: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class WireHello:
    chunk_size: int
    model_dim: int
    robot_dim: int
    state_dim: int
    images: tuple[WireImageSpec, ...]


@dataclass(frozen=True, slots=True)
class WireObservation:
    sequence: int
    capture_ns: int
    state: NDArray[np.float32]
    images: tuple[WireImage, ...]
    task: str


@dataclass(frozen=True, slots=True)
class WireRequest:
    request_id: int
    first_step: int
    last_emitted_step: int
    action_count: int
    stitching: int
    seed: int
    observation: WireObservation


@dataclass(frozen=True, slots=True)
class WireTiming:
    request_decode_ns: int = 0
    observation_ns: int = 0
    model_ns: int = 0
    response_encode_ns: int = 0


@dataclass(frozen=True, slots=True)
class WireResponse:
    request_id: int
    first_step: int
    observation_sequence: int
    capture_ns: int
    model_actions: NDArray[np.float32]
    robot_actions: NDArray[np.float32]
    timing: WireTiming = WireTiming()


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


def encode_hello(
    *,
    chunk_size: int,
    model_dim: int,
    robot_dim: int,
    state_dim: int = 0,
    images: tuple[WireImageSpec, ...] = (),
) -> bytes:
    if state_dim > 0xFFFF or len(images) > 0xFFFF:
        raise ProtocolError("hello observation schema exceeds protocol limits")
    payload = bytearray(
        _HELLO.pack(
            HELLO_MAGIC,
            PROTOCOL_VERSION,
            0,
            0,
            chunk_size,
            model_dim,
            robot_dim,
            state_dim,
            len(images),
            0,
        )
    )
    for image in images:
        name = image.feature_name.encode("utf-8")
        dimensions_fit = all(
            0 < value <= 0xFFFF
            for value in (image.channels, image.height, image.width)
        )
        if not name or len(name) > 0xFFFF or not dimensions_fit:
            raise ProtocolError("invalid image specification")
        payload.extend(
            _HELLO_IMAGE.pack(
                len(name), image.channels, image.height, image.width
            )
        )
        payload.extend(name)
    return bytes(payload)


def encode_hello_error(message: str) -> bytes:
    detail = message.encode("utf-8")
    return _HELLO.pack(
        HELLO_MAGIC, PROTOCOL_VERSION, 1, 0, 0, 0, 0, 0, 0, len(detail)
    ) + detail


def decode_hello(payload: bytes) -> WireHello:
    if len(payload) < _HELLO.size:
        raise ProtocolError("hello header is truncated")
    (
        magic,
        version,
        status,
        _reserved,
        chunk_size,
        model_dim,
        robot_dim,
        state_dim,
        image_count,
        error_size,
    ) = _HELLO.unpack_from(payload)
    if magic != HELLO_MAGIC:
        raise ProtocolError("bad hello magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    if status != 0:
        detail = _take(payload, _HELLO.size, error_size, "hello error")
        raise ProtocolError(detail.decode("utf-8", errors="replace"))
    if error_size != 0:
        raise ProtocolError("successful hello contains an error")

    cursor = _HELLO.size
    images: list[WireImageSpec] = []
    for _ in range(image_count):
        descriptor = _take(payload, cursor, _HELLO_IMAGE.size, "image specification")
        cursor += _HELLO_IMAGE.size
        name_size, channels, height, width = _HELLO_IMAGE.unpack(descriptor)
        name_bytes = _take(payload, cursor, name_size, "image specification name")
        cursor += name_size
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("image specification name is not UTF-8") from exc
        images.append(WireImageSpec(name, channels, height, width))
    if cursor != len(payload):
        raise ProtocolError("hello contains trailing bytes")
    return WireHello(chunk_size, model_dim, robot_dim, state_dim, tuple(images))


def encode_request(request: WireRequest) -> bytes:
    observation = request.observation
    state = np.ascontiguousarray(observation.state, dtype=np.float32)
    if state.ndim != 1:
        raise ProtocolError("state must be rank 1")
    task = observation.task.encode("utf-8")
    if not task:
        raise ProtocolError("task is empty")
    if state.size > 0xFFFFFFFF or len(task) > 0xFFFFFFFF:
        raise ProtocolError("state or task exceeds protocol limits")
    if len(observation.images) > 0xFFFF:
        raise ProtocolError("too many images")

    payload = bytearray(
        _REQUEST.pack(
            REQUEST_MAGIC,
            PROTOCOL_VERSION,
            request.stitching,
            0,
            request.request_id,
            request.first_step,
            request.last_emitted_step,
            request.action_count,
            request.seed,
            observation.sequence,
            observation.capture_ns,
            state.size,
            len(observation.images),
            len(task),
        )
    )
    payload.extend(state.astype(">f4", copy=False).tobytes())
    payload.extend(task)
    for image in observation.images:
        name = image.feature_name.encode("utf-8")
        pixels = np.ascontiguousarray(image.pixels, dtype=np.uint8).reshape(-1)
        expected = image.channels * image.height * image.width
        dimensions_fit = all(0 < value <= 0xFFFF for value in (
            image.channels, image.height, image.width
        ))
        if (
            not name
            or len(name) > 0xFFFF
            or pixels.size > 0xFFFFFFFF
            or pixels.size != expected
            or not dimensions_fit
        ):
            raise ProtocolError("invalid image name, dimensions, or pixel count")
        payload.extend(
            _IMAGE.pack(
                len(name), image.channels, image.height, image.width, pixels.size
            )
        )
        payload.extend(name)
        payload.extend(pixels.tobytes())
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"request is too large: {len(payload)} bytes")
    return bytes(payload)


def decode_request(payload: bytes) -> WireRequest:
    if len(payload) < _REQUEST.size:
        raise ProtocolError("request header is truncated")
    (
        magic,
        version,
        stitching,
        _reserved,
        request_id,
        first,
        last,
        count,
        seed,
        sequence,
        capture_ns,
        state_count,
        image_count,
        task_size,
    ) = _REQUEST.unpack_from(payload)
    if magic != REQUEST_MAGIC:
        raise ProtocolError("bad request magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")

    cursor = _REQUEST.size
    state_bytes = _take(payload, cursor, state_count * 4, "state")
    cursor += state_count * 4
    state = np.frombuffer(state_bytes, dtype=">f4").astype(np.float32)
    task_bytes = _take(payload, cursor, task_size, "task")
    cursor += task_size
    try:
        task = task_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("task is not UTF-8") from exc

    images: list[WireImage] = []
    for _ in range(image_count):
        descriptor = _take(payload, cursor, _IMAGE.size, "image descriptor")
        cursor += _IMAGE.size
        name_size, channels, height, width, pixel_count = _IMAGE.unpack(descriptor)
        if pixel_count != channels * height * width:
            raise ProtocolError("image pixel count does not match dimensions")
        name_bytes = _take(payload, cursor, name_size, "image name")
        cursor += name_size
        pixels = np.frombuffer(
            _take(payload, cursor, pixel_count, "image pixels"), dtype=np.uint8
        ).copy()
        cursor += pixel_count
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("image name is not UTF-8") from exc
        images.append(WireImage(name, channels, height, width, pixels))
    if cursor != len(payload):
        raise ProtocolError("request contains trailing bytes")

    return WireRequest(
        request_id,
        first,
        last,
        count,
        stitching,
        seed,
        WireObservation(sequence, capture_ns, state, tuple(images), task),
    )


def encode_response(response: WireResponse) -> bytes:
    started = monotonic_ns()
    model = np.asarray(response.model_actions, dtype=np.float32)
    robot = np.asarray(response.robot_actions, dtype=np.float32)
    if model.ndim != 2 or robot.ndim != 2 or model.shape[0] != robot.shape[0]:
        raise ProtocolError("response action arrays have incompatible shapes")
    count, model_dim = model.shape
    robot_dim = robot.shape[1]
    payload = bytearray(
        _RESPONSE.pack(
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
            response.timing.request_decode_ns,
            response.timing.observation_ns,
            response.timing.model_ns,
            0,
            0,
        )
    )
    payload.extend(model.astype(">f4", copy=False).tobytes())
    payload.extend(robot.astype(">f4", copy=False).tobytes())
    struct.pack_into(
        ">Q", payload, _RESPONSE_ENCODE_NS_OFFSET, monotonic_ns() - started
    )
    return bytes(payload)


def encode_response_error(request_id: int, message: str) -> bytes:
    detail = message.encode("utf-8")
    return _RESPONSE.pack(
        RESPONSE_MAGIC,
        PROTOCOL_VERSION,
        1,
        0,
        request_id,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        len(detail),
    ) + detail


def _take(payload: bytes, offset: int, size: int, label: str) -> bytes:
    if size < 0 or offset > len(payload) or size > len(payload) - offset:
        raise ProtocolError(f"{label} is truncated")
    return payload[offset : offset + size]


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
