"""Wire protocol for the persistent MuJoCo/LIBERO simulator worker."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray

from .protocol import ProtocolError

PROTOCOL_VERSION = 1

HELLO_MAGIC = b"CBSH"
COMMAND_MAGIC = b"CBSC"
OBSERVATION_MAGIC = b"CBSO"

_HELLO = struct.Struct(">4sHBBHHHI")
_IMAGE_SPEC = struct.Struct(">HHHH")
_COMMAND = struct.Struct(">4sHBBQqH")
_OBSERVATION = struct.Struct(">4sHBBQQqfBBHHI")
_IMAGE = struct.Struct(">HHHHI")


@dataclass(frozen=True, slots=True)
class SimulatorImageSpec:
    feature_name: str
    channels: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class SimulatorHello:
    action_dim: int
    state_dim: int
    images: tuple[SimulatorImageSpec, ...]
    task: str


@dataclass(frozen=True, slots=True)
class SimulatorCommand:
    sequence: int
    step: int
    action: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SimulatorImage:
    feature_name: str
    channels: int
    height: int
    width: int
    pixels: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class SimulatorObservation:
    sequence: int
    capture_ns: int
    sim_step: int
    reward: float
    terminated: bool
    success: bool
    state: NDArray[np.float32]
    images: tuple[SimulatorImage, ...]


def encode_hello(hello: SimulatorHello) -> bytes:
    task = hello.task.encode("utf-8")
    if not task or len(task) > 0xFFFFFFFF:
        raise ProtocolError("invalid simulator task")
    if not 0 < hello.action_dim <= 0xFFFF or not 0 < hello.state_dim <= 0xFFFF:
        raise ProtocolError("invalid simulator dimensions")
    if not hello.images or len(hello.images) > 0xFFFF:
        raise ProtocolError("invalid simulator image count")
    payload = bytearray(
        _HELLO.pack(
            HELLO_MAGIC,
            PROTOCOL_VERSION,
            0,
            0,
            hello.action_dim,
            hello.state_dim,
            len(hello.images),
            len(task),
        )
    )
    for image in hello.images:
        name = image.feature_name.encode("utf-8")
        if not name or len(name) > 0xFFFF or not all(
            0 < value <= 0xFFFF
            for value in (image.channels, image.height, image.width)
        ):
            raise ProtocolError("invalid simulator image specification")
        payload.extend(
            _IMAGE_SPEC.pack(len(name), image.channels, image.height, image.width)
        )
        payload.extend(name)
    payload.extend(task)
    return bytes(payload)


def encode_hello_error(message: str) -> bytes:
    detail = message.encode("utf-8")
    return _HELLO.pack(
        HELLO_MAGIC, PROTOCOL_VERSION, 1, 0, 0, 0, 0, len(detail)
    ) + detail


def decode_hello(payload: bytes) -> SimulatorHello:
    if len(payload) < _HELLO.size:
        raise ProtocolError("simulator hello is truncated")
    magic, version, status, _, action_dim, state_dim, image_count, detail_size = (
        _HELLO.unpack_from(payload)
    )
    if magic != HELLO_MAGIC or version != PROTOCOL_VERSION:
        raise ProtocolError("bad simulator hello magic or version")
    offset = _HELLO.size
    if status:
        if len(payload) != offset + detail_size:
            raise ProtocolError("bad simulator hello error length")
        raise ProtocolError(payload[offset:].decode("utf-8", errors="replace"))
    images: list[SimulatorImageSpec] = []
    for _ in range(image_count):
        if len(payload) < offset + _IMAGE_SPEC.size:
            raise ProtocolError("simulator image specification is truncated")
        name_size, channels, height, width = _IMAGE_SPEC.unpack_from(payload, offset)
        offset += _IMAGE_SPEC.size
        if len(payload) < offset + name_size:
            raise ProtocolError("simulator image name is truncated")
        name = payload[offset : offset + name_size].decode("utf-8")
        offset += name_size
        images.append(SimulatorImageSpec(name, channels, height, width))
    if len(payload) != offset + detail_size:
        raise ProtocolError("bad simulator task length")
    task = payload[offset:].decode("utf-8")
    return SimulatorHello(action_dim, state_dim, tuple(images), task)


def encode_command(command: SimulatorCommand) -> bytes:
    action = np.asarray(command.action, dtype=np.float32)
    if action.ndim != 1 or not 0 < action.size <= 0xFFFF:
        raise ProtocolError("simulator action must be a nonempty vector")
    return _COMMAND.pack(
        COMMAND_MAGIC,
        PROTOCOL_VERSION,
        0,
        0,
        command.sequence,
        command.step,
        action.size,
    ) + action.astype(">f4", copy=False).tobytes()


def decode_command(payload: bytes) -> SimulatorCommand:
    if len(payload) < _COMMAND.size:
        raise ProtocolError("simulator command is truncated")
    magic, version, status, _, sequence, step, action_dim = _COMMAND.unpack_from(payload)
    if magic != COMMAND_MAGIC or version != PROTOCOL_VERSION or status != 0:
        raise ProtocolError("bad simulator command header")
    expected = _COMMAND.size + action_dim * 4
    if len(payload) != expected:
        raise ProtocolError("bad simulator action length")
    action = np.frombuffer(payload, dtype=">f4", offset=_COMMAND.size).astype(np.float32)
    if not np.isfinite(action).all():
        raise ProtocolError("simulator action contains non-finite values")
    return SimulatorCommand(sequence, step, action)


def encode_observation(observation: SimulatorObservation) -> bytes:
    state = np.asarray(observation.state, dtype=np.float32)
    if state.ndim != 1 or not 0 < state.size <= 0xFFFF:
        raise ProtocolError("simulator state must be a nonempty vector")
    if len(observation.images) > 0xFFFF:
        raise ProtocolError("too many simulator images")
    payload = bytearray(
        _OBSERVATION.pack(
            OBSERVATION_MAGIC,
            PROTOCOL_VERSION,
            0,
            0,
            observation.sequence,
            observation.capture_ns,
            observation.sim_step,
            observation.reward,
            observation.terminated,
            observation.success,
            state.size,
            len(observation.images),
            0,
        )
    )
    payload.extend(state.astype(">f4", copy=False).tobytes())
    for image in observation.images:
        name = image.feature_name.encode("utf-8")
        pixels = np.asarray(image.pixels, dtype=np.uint8).reshape(-1)
        expected = image.channels * image.height * image.width
        if not name or len(name) > 0xFFFF or pixels.size != expected:
            raise ProtocolError("invalid simulator image")
        payload.extend(
            _IMAGE.pack(
                len(name), image.channels, image.height, image.width, pixels.size
            )
        )
        payload.extend(name)
        payload.extend(pixels.tobytes())
    return bytes(payload)


def encode_observation_error(sequence: int, message: str) -> bytes:
    detail = message.encode("utf-8")
    return _OBSERVATION.pack(
        OBSERVATION_MAGIC,
        PROTOCOL_VERSION,
        1,
        0,
        sequence,
        0,
        -1,
        0.0,
        False,
        False,
        0,
        0,
        len(detail),
    ) + detail


def decode_observation(payload: bytes) -> SimulatorObservation:
    if len(payload) < _OBSERVATION.size:
        raise ProtocolError("simulator observation is truncated")
    (
        magic,
        version,
        status,
        _,
        sequence,
        capture_ns,
        sim_step,
        reward,
        terminated,
        success,
        state_dim,
        image_count,
        detail_size,
    ) = _OBSERVATION.unpack_from(payload)
    if magic != OBSERVATION_MAGIC or version != PROTOCOL_VERSION:
        raise ProtocolError("bad simulator observation header")
    offset = _OBSERVATION.size
    if status:
        if len(payload) != offset + detail_size:
            raise ProtocolError("bad simulator observation error length")
        raise ProtocolError(payload[offset:].decode("utf-8", errors="replace"))
    state_bytes = state_dim * 4
    if len(payload) < offset + state_bytes:
        raise ProtocolError("simulator state is truncated")
    state = np.frombuffer(payload, dtype=">f4", count=state_dim, offset=offset).astype(
        np.float32
    )
    offset += state_bytes
    images: list[SimulatorImage] = []
    for _ in range(image_count):
        if len(payload) < offset + _IMAGE.size:
            raise ProtocolError("simulator image header is truncated")
        name_size, channels, height, width, pixel_size = _IMAGE.unpack_from(
            payload, offset
        )
        offset += _IMAGE.size
        if len(payload) < offset + name_size + pixel_size:
            raise ProtocolError("simulator image is truncated")
        name = payload[offset : offset + name_size].decode("utf-8")
        offset += name_size
        pixels = np.frombuffer(payload, dtype=np.uint8, count=pixel_size, offset=offset).copy()
        offset += pixel_size
        if pixel_size != channels * height * width:
            raise ProtocolError("simulator image shape mismatch")
        images.append(SimulatorImage(name, channels, height, width, pixels))
    if detail_size != 0 or offset != len(payload):
        raise ProtocolError("simulator observation contains trailing bytes")
    return SimulatorObservation(
        sequence,
        capture_ns,
        sim_step,
        reward,
        bool(terminated),
        bool(success),
        state,
        tuple(images),
    )
