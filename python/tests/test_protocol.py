from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from cerebellum_model.protocol import (
    ProtocolError,
    WireImage,
    WireObservation,
    WireRequest,
    WireResponse,
    decode_request,
    encode_request,
    encode_response,
    read_frame,
    write_frame,
)


def test_frames_survive_partial_reads() -> None:
    target = io.BytesIO()
    write_frame(target, b"hello")

    class TwoBytesAtATime(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 2))

    assert read_frame(TwoBytesAtATime(target.getvalue())) == b"hello"


def request() -> WireRequest:
    return WireRequest(
        request_id=9,
        first_step=44,
        last_emitted_step=43,
        action_count=50,
        stitching=1,
        seed=7,
        observation=WireObservation(
            sequence=12,
            capture_ns=987654321,
            state=np.array([1.25, -2.5, 3.75], dtype=np.float32),
            images=(
                WireImage(
                    "observation.images.camera1",
                    channels=3,
                    height=2,
                    width=2,
                    pixels=np.arange(12, dtype=np.uint8),
                ),
                WireImage(
                    "observation.images.camera2",
                    channels=1,
                    height=1,
                    width=3,
                    pixels=np.array([255, 17, 99], dtype=np.uint8),
                ),
            ),
            task="pick up the blue block",
        ),
    )


def test_request_round_trips_every_observation_value() -> None:
    original = request()
    decoded = decode_request(encode_request(original))

    assert decoded.request_id == 9
    assert decoded.first_step == 44
    assert decoded.last_emitted_step == 43
    assert decoded.action_count == 50
    assert decoded.stitching == 1
    assert decoded.seed == 7
    assert decoded.observation.sequence == 12
    assert decoded.observation.capture_ns == 987654321
    assert decoded.observation.task == "pick up the blue block"
    np.testing.assert_array_equal(decoded.observation.state, original.observation.state)
    assert len(decoded.observation.images) == 2
    for actual, expected in zip(decoded.observation.images, original.observation.images):
        assert actual.feature_name == expected.feature_name
        assert (actual.channels, actual.height, actual.width) == (
            expected.channels,
            expected.height,
            expected.width,
        )
        np.testing.assert_array_equal(actual.pixels, expected.pixels)


def test_response_contains_both_action_spaces() -> None:
    model = np.arange(12, dtype=np.float32).reshape(2, 6)
    robot = (model[:, :2] + 100).copy()
    payload = encode_response(WireResponse(4, 10, 3, 99, model, robot))
    header_size = struct.calcsize(">4sHBBQqQqIHHI")
    assert len(payload) == header_size + model.nbytes + robot.nbytes
    assert payload[header_size:] == model.astype(">f4").tobytes() + robot.astype(">f4").tobytes()


def test_bad_request_magic_is_rejected() -> None:
    payload = b"NOPE" + encode_request(request())[4:]
    with pytest.raises(ProtocolError, match="magic"):
        decode_request(payload)


def test_truncated_image_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="pixels"):
        decode_request(encode_request(request())[:-1])
