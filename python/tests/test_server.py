from __future__ import annotations

import numpy as np

import pytest

from cerebellum_model.protocol import (
    ProtocolError,
    WireImage,
    WireImageSpec,
    WireObservation,
    WireRequest,
)
from cerebellum_model.server import (
    RunnerSpec,
    _observation_from_wire,
    _seed_for_observation,
    _validate_observation_schema,
)


def wire_request(first_pixel: int = 0) -> WireRequest:
    pixels = np.arange(12, dtype=np.uint8)
    pixels[0] = first_pixel
    return WireRequest(
        request_id=1,
        first_step=0,
        last_emitted_step=-1,
        action_count=50,
        stitching=0,
        seed=9,
        observation=WireObservation(
            sequence=17,
            capture_ns=123456,
            state=np.array([1.0, -2.0, 3.5], dtype=np.float32),
            images=(WireImage("camera", 3, 2, 2, pixels),),
            task="move the cube",
        ),
    )


def test_server_converts_hwc_bytes_to_chw_unit_floats() -> None:
    request = wire_request()
    converted = _observation_from_wire(request)

    assert converted.sequence == 17
    assert converted.capture_ns == 123456
    assert converted.task == "move the cube"
    np.testing.assert_array_equal(converted.state, request.observation.state)
    expected = request.observation.images[0].pixels.reshape(2, 2, 3).transpose(2, 0, 1)
    np.testing.assert_allclose(converted.images["camera"], expected / np.float32(255.0))


def test_synthetic_seed_depends_on_camera_payload() -> None:
    assert _seed_for_observation(wire_request(1)) != _seed_for_observation(wire_request(2))


def test_checkpoint_schema_rejects_wrong_observation_shape() -> None:
    spec = RunnerSpec(
        chunk_size=50,
        model_dim=32,
        robot_dim=6,
        state_dim=6,
        images=(WireImageSpec("camera", 3, 256, 256),),
    )
    with pytest.raises(ProtocolError, match="state shape"):
        _validate_observation_schema(wire_request(), spec)
