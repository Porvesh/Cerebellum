from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from cerebellum_model.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    WireResponse,
    decode_request,
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


def test_request_decodes_network_byte_order() -> None:
    payload = struct.pack(
        ">4sHBBQqqII", b"CBRQ", PROTOCOL_VERSION, 1, 0, 9, 44, 43, 50, 7
    )
    request = decode_request(payload)
    assert request.request_id == 9
    assert request.first_step == 44
    assert request.stitching == 1


def test_response_contains_both_action_spaces() -> None:
    model = np.arange(12, dtype=np.float32).reshape(2, 6)
    robot = (model[:, :2] + 100).copy()
    payload = encode_response(WireResponse(4, 10, 3, 99, model, robot))
    header_size = struct.calcsize(">4sHBBQqQqIHHI")
    assert len(payload) == header_size + model.nbytes + robot.nbytes
    assert payload[header_size:] == model.astype(">f4").tobytes() + robot.astype(">f4").tobytes()


def test_bad_request_magic_is_rejected() -> None:
    payload = struct.pack(
        ">4sHBBQqqII", b"NOPE", PROTOCOL_VERSION, 0, 0, 1, 0, -1, 50, 0
    )
    with pytest.raises(ProtocolError, match="magic"):
        decode_request(payload)
