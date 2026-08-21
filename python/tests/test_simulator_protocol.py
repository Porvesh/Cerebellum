from __future__ import annotations

import io

import numpy as np
import pytest

from cerebellum_model.protocol import ProtocolError, read_frame, write_frame
from cerebellum_model.simulator_protocol import (
    SimulatorCommand,
    SimulatorHello,
    SimulatorImage,
    SimulatorImageSpec,
    SimulatorObservation,
    decode_command,
    decode_hello,
    decode_observation,
    encode_command,
    encode_hello,
    encode_observation,
)
from cerebellum_model.simulator_server import SyntheticSimulator, parse_args, serve


def test_simulator_messages_round_trip() -> None:
    hello = SimulatorHello(
        7,
        8,
        (SimulatorImageSpec("observation.images.image", 3, 2, 3),),
        "pick up the bowl",
    )
    assert decode_hello(encode_hello(hello)) == hello

    command = SimulatorCommand(12, 44, np.arange(7, dtype=np.float32))
    decoded_command = decode_command(encode_command(command))
    assert (decoded_command.sequence, decoded_command.step) == (12, 44)
    np.testing.assert_array_equal(decoded_command.action, command.action)

    observation = SimulatorObservation(
        12,
        999,
        4,
        1.25,
        True,
        False,
        np.arange(8, dtype=np.float32),
        (
            SimulatorImage(
                "observation.images.image", 3, 2, 3, np.arange(18, dtype=np.uint8)
            ),
        ),
    )
    decoded = decode_observation(encode_observation(observation))
    assert decoded.sequence == 12
    assert decoded.sim_step == 4
    assert decoded.reward == pytest.approx(1.25)
    assert decoded.terminated and not decoded.success
    np.testing.assert_array_equal(decoded.state, observation.state)
    np.testing.assert_array_equal(decoded.images[0].pixels, observation.images[0].pixels)


def test_synthetic_server_handshake_initial_observation_and_step() -> None:
    command = encode_command(
        SimulatorCommand(3, 2, np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32))
    )
    source = io.BytesIO()
    write_frame(source, command)
    source.seek(0)
    sink = io.BytesIO()

    class Streams:
        buffer: io.BytesIO

        def __init__(self, buffer: io.BytesIO) -> None:
            self.buffer = buffer

    import cerebellum_model.simulator_server as server

    old_stdin, old_stdout = server.sys.stdin, server.sys.stdout
    try:
        server.sys.stdin = Streams(source)  # type: ignore[assignment]
        server.sys.stdout = Streams(sink)  # type: ignore[assignment]
        serve(parse_args(["--backend", "synthetic", "--image-size", "4"]))
    finally:
        server.sys.stdin, server.sys.stdout = old_stdin, old_stdout

    sink.seek(0)
    hello = decode_hello(read_frame(sink) or b"")
    initial = decode_observation(read_frame(sink) or b"")
    stepped = decode_observation(read_frame(sink) or b"")
    assert hello.action_dim == 7 and hello.state_dim == 8
    assert initial.sequence == 0 and initial.sim_step == 0
    assert stepped.sequence == 3 and stepped.sim_step == 1
    np.testing.assert_array_equal(stepped.state[:7], np.arange(1, 8, dtype=np.float32))
    assert read_frame(sink) is None


def test_nonfinite_action_is_rejected() -> None:
    payload = encode_command(SimulatorCommand(1, 0, np.array([np.nan], dtype=np.float32)))
    with pytest.raises(ProtocolError, match="non-finite"):
        decode_command(payload)


def test_quaternion_conversion_matches_expected_rotation() -> None:
    from cerebellum_model.simulator_server import _quat_to_axis_angle

    half = np.sqrt(0.5)
    np.testing.assert_allclose(
        _quat_to_axis_angle(np.array([0.0, 0.0, half, half])),
        np.array([0.0, 0.0, np.pi / 2]),
        rtol=1e-5,
    )
