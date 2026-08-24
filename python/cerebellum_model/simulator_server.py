"""Persistent synthetic or LIBERO simulator process used by Cerebellum."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from time import monotonic_ns
from typing import Protocol

import numpy as np

from .protocol import read_frame, write_frame
from .simulator_protocol import (
    SimulatorCommand,
    SimulatorHello,
    SimulatorImage,
    SimulatorImageSpec,
    SimulatorObservation,
    decode_command,
    encode_hello,
    encode_hello_error,
    encode_observation,
    encode_observation_error,
)


class SimulatorBackend(Protocol):
    hello: SimulatorHello

    def initial_observation(self) -> SimulatorObservation: ...

    def step(self, command: SimulatorCommand) -> SimulatorObservation: ...

    def close(self) -> None: ...


class SyntheticSimulator:
    def __init__(self, *, action_dim: int = 7, image_size: int = 8) -> None:
        self.hello = SimulatorHello(
            action_dim,
            8,
            (
                SimulatorImageSpec("observation.images.image", 3, image_size, image_size),
                SimulatorImageSpec("observation.images.image2", 3, image_size, image_size),
            ),
            "move the object to the target",
        )
        self._state = np.zeros(8, dtype=np.float32)
        self._sim_step = 0

    def _observation(self, sequence: int, reward: float = 0.0) -> SimulatorObservation:
        images = []
        for camera, spec in enumerate(self.hello.images):
            pixels = np.full(
                spec.channels * spec.height * spec.width,
                (self._sim_step + camera * 37) % 256,
                dtype=np.uint8,
            )
            images.append(
                SimulatorImage(
                    spec.feature_name, spec.channels, spec.height, spec.width, pixels
                )
            )
        return SimulatorObservation(
            sequence,
            monotonic_ns(),
            self._sim_step,
            reward,
            False,
            False,
            self._state.copy(),
            tuple(images),
        )

    def initial_observation(self) -> SimulatorObservation:
        return self._observation(0)

    def step(self, command: SimulatorCommand) -> SimulatorObservation:
        if command.action.shape != (self.hello.action_dim,):
            raise ValueError("synthetic action dimension mismatch")
        step_started = monotonic_ns()
        self._sim_step += 1
        copy_count = min(self._state.size, command.action.size)
        self._state[:copy_count] += command.action[:copy_count]
        environment_step_ns = monotonic_ns() - step_started
        build_started = monotonic_ns()
        observation = self._observation(command.sequence, float(command.action.sum()))
        return replace(
            observation,
            environment_step_ns=environment_step_ns,
            observation_build_ns=monotonic_ns() - build_started,
        )

    def close(self) -> None:
        pass


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"expected a four-dimensional quaternion, got {quat.shape}")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] / denominator * (2.0 * np.arccos(w))).astype(np.float32)


class LiberoSimulator:
    CAMERA_KEYS = (
        ("agentview_image", "observation.images.image"),
        ("robot0_eye_in_hand_image", "observation.images.image2"),
    )

    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        init_state: int,
        control_hz: int,
        image_size: int,
    ) -> None:
        # These imports intentionally stay inside the real backend. Unit tests
        # can exercise the complete transport without installing MuJoCo/LIBERO.
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        if suite_name not in suites:
            raise ValueError(f"unknown LIBERO suite: {suite_name}")
        suite = suites[suite_name]()
        task = suite.get_task(task_id)
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=image_size,
            camera_widths=image_size,
            control_freq=control_hz,
        )
        raw = self._env.reset()
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        states = torch.load(init_path, map_location="cpu", weights_only=False)
        raw = self._env.set_init_state(states[init_state % len(states)])
        noop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(10):
            raw, _, _, _ = self._env.step(noop)
        self._raw = raw
        self._sim_step = 0
        self.hello = SimulatorHello(
            7,
            8,
            tuple(
                SimulatorImageSpec(feature, 3, image_size, image_size)
                for _, feature in self.CAMERA_KEYS
            ),
            task.language,
        )

    def _convert(
        self,
        sequence: int,
        reward: float,
        terminated: bool,
        success: bool,
    ) -> SimulatorObservation:
        raw = self._raw
        state = np.concatenate(
            (
                np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(raw["robot0_eef_quat"]),
                np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        images = []
        for raw_name, feature_name in self.CAMERA_KEYS:
            # This is the exact 180-degree convention applied by LeRobot's
            # LiberoProcessorStep before SmolVLA preprocessing.
            image = np.ascontiguousarray(raw[raw_name][::-1, ::-1, :], dtype=np.uint8)
            images.append(
                SimulatorImage(
                    feature_name, image.shape[2], image.shape[0], image.shape[1], image.reshape(-1)
                )
            )
        return SimulatorObservation(
            sequence,
            monotonic_ns(),
            self._sim_step,
            reward,
            terminated,
            success,
            state,
            tuple(images),
        )

    def initial_observation(self) -> SimulatorObservation:
        return self._convert(0, 0.0, False, False)

    def step(self, command: SimulatorCommand) -> SimulatorObservation:
        if command.action.shape != (7,):
            raise ValueError("LIBERO action dimension mismatch")
        step_started = monotonic_ns()
        self._raw, reward, done, _ = self._env.step(command.action)
        environment_step_ns = monotonic_ns() - step_started
        self._sim_step += 1
        success = bool(self._env.check_success())
        build_started = monotonic_ns()
        observation = self._convert(
            command.sequence, float(reward), bool(done or success), success
        )
        return replace(
            observation,
            environment_step_ns=environment_step_ns,
            observation_build_ns=monotonic_ns() - build_started,
        )

    def close(self) -> None:
        self._env.close()


def _backend(args: argparse.Namespace) -> SimulatorBackend:
    if args.backend == "synthetic":
        return SyntheticSimulator(action_dim=args.action_dim, image_size=args.image_size)
    return LiberoSimulator(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state=args.init_state,
        control_hz=args.control_hz,
        image_size=args.image_size,
    )


def serve(args: argparse.Namespace) -> None:
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    try:
        simulator = _backend(args)
    except Exception as exc:
        write_frame(sink, encode_hello_error(str(exc)))
        return

    try:
        write_frame(sink, encode_hello(simulator.hello))
        write_frame(sink, encode_observation(simulator.initial_observation()))
        while (payload := read_frame(source)) is not None:
            command: SimulatorCommand | None = None
            try:
                command = decode_command(payload)
                response = encode_observation(simulator.step(command))
            except Exception as exc:
                sequence = command.sequence if command is not None else 0
                response = encode_observation_error(sequence, str(exc))
            write_frame(sink, response)
    finally:
        simulator.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("synthetic", "libero"), default="synthetic")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--control-hz", type=int, default=30)
    return parser.parse_args(argv)


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
