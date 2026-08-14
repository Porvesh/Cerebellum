from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from cerebellum_model.protocol import (
    ProtocolError,
    WireImageSpec,
    WireImage,
    WireObservation,
    WireRequest,
    WireRtcConditioning,
    WireResponse,
    WireTiming,
    decode_hello,
    decode_request,
    encode_hello,
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


def test_hello_round_trips_checkpoint_observation_schema() -> None:
    images = (
        WireImageSpec("observation.images.camera1", 3, 256, 256),
        WireImageSpec("observation.images.camera2", 3, 256, 256),
    )
    hello = decode_hello(
        encode_hello(
            chunk_size=50,
            model_dim=32,
            robot_dim=6,
            state_dim=6,
            images=images,
        )
    )
    assert hello.chunk_size == 50
    assert hello.model_dim == 32
    assert hello.robot_dim == 6
    assert hello.state_dim == 6
    assert hello.images == images


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


def test_rtc_prefix_round_trips_in_model_space() -> None:
    original = request()
    prefix = np.arange(8 * 32, dtype=np.float32).reshape(8, 32)
    rtc_request = WireRequest(
        request_id=original.request_id,
        first_step=original.first_step,
        last_emitted_step=original.last_emitted_step,
        action_count=original.action_count,
        stitching=2,
        seed=original.seed,
        observation=original.observation,
        rtc=WireRtcConditioning(17, original.first_step, 5, 6, prefix),
    )
    decoded = decode_request(encode_request(rtc_request))
    assert decoded.rtc is not None
    assert decoded.rtc.source_chunk_id == 17
    assert decoded.rtc.inference_delay == 5
    assert decoded.rtc.execution_horizon == 6
    np.testing.assert_array_equal(decoded.rtc.prefix, prefix)


def test_response_contains_both_action_spaces() -> None:
    model = np.arange(12, dtype=np.float32).reshape(2, 6)
    robot = (model[:, :2] + 100).copy()
    timing = WireTiming(11, 22, 33)
    payload = encode_response(WireResponse(4, 10, 3, 99, model, robot, timing))
    header_format = ">4sHBBQqQqIHHQQQQI"
    header_size = struct.calcsize(header_format)
    header = struct.unpack_from(header_format, payload)
    assert header[11:14] == (11, 22, 33)
    assert header[14] > 0
    assert len(payload) == header_size + model.nbytes + robot.nbytes
    assert payload[header_size:] == model.astype(">f4").tobytes() + robot.astype(">f4").tobytes()


def test_bad_request_magic_is_rejected() -> None:
    payload = b"NOPE" + encode_request(request())[4:]
    with pytest.raises(ProtocolError, match="magic"):
        decode_request(payload)


def test_truncated_image_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="pixels"):
        decode_request(encode_request(request())[:-1])
