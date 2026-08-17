"""Persistent stdin/stdout inference worker for the C++ bridge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import struct
import sys
from time import monotonic_ns
from typing import Any
import zlib

import numpy as np

from .messages import InferenceRequest, Observation, RtcConditioning
from .protocol import (
    ProtocolError,
    WireImageSpec,
    WireResponse,
    WireRequest,
    WireTiming,
    decode_request,
    encode_hello,
    encode_hello_error,
    encode_response,
    encode_response_error,
    read_frame,
    write_frame,
)
from .runner import SyntheticRunner

_STITCHING = {0: "discard", 1: "ensemble", 2: "rtc"}


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    chunk_size: int
    model_dim: int
    robot_dim: int
    state_dim: int = 0
    images: tuple[WireImageSpec, ...] = ()


def _make_runner(args: argparse.Namespace) -> tuple[Any, RunnerSpec, bool]:
    if args.runner == "synthetic":
        runner = SyntheticRunner(model_dim=args.model_dim, robot_dim=args.robot_dim)
        return runner, RunnerSpec(args.chunk_size, args.model_dim, args.robot_dim), True

    from .smolvla_runner import SmolVLARunner

    runner = SmolVLARunner.from_pretrained(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        rtc_denoise_steps=args.rtc_denoise_steps,
    )
    images = tuple(
        WireImageSpec(name, shape[0], shape[1], shape[2])
        for name, shape in runner.image_shapes.items()
    )
    return (
        runner,
        RunnerSpec(
            runner.chunk_size,
            runner.model_action_dim,
            runner.robot_action_dim,
            runner.state_dim,
            images,
        ),
        False,
    )


def _observation_from_wire(wire: WireRequest) -> Observation:
    observation = wire.observation
    images = {}
    for image in observation.images:
        hwc = image.pixels.reshape(image.height, image.width, image.channels)
        images[image.feature_name] = np.ascontiguousarray(
            hwc.transpose(2, 0, 1), dtype=np.float32
        ) / np.float32(255.0)
    return Observation(
        sequence=observation.sequence,
        capture_ns=observation.capture_ns,
        state=observation.state,
        images=images,
        task=observation.task,
    )


def _validate_observation_schema(wire: WireRequest, spec: RunnerSpec) -> None:
    if spec.state_dim == 0 and not spec.images:
        return
    observation = wire.observation
    if observation.state.shape != (spec.state_dim,):
        raise ProtocolError(
            f"state shape {observation.state.shape} does not match ({spec.state_dim},)"
        )
    received = {
        image.feature_name: (image.channels, image.height, image.width)
        for image in observation.images
    }
    if len(received) != len(observation.images):
        raise ProtocolError("observation contains duplicate image names")
    expected = {
        image.feature_name: (image.channels, image.height, image.width)
        for image in spec.images
    }
    if received != expected:
        raise ProtocolError(f"image schema {received} does not match {expected}")


def _seed_for_observation(wire: WireRequest) -> int:
    """Make synthetic output depend on every transported observation value."""

    observation = wire.observation
    checksum = zlib.crc32(observation.state.astype(">f4", copy=False).tobytes())
    checksum = zlib.crc32(observation.task.encode("utf-8"), checksum)
    for image in observation.images:
        checksum = zlib.crc32(image.feature_name.encode("utf-8"), checksum)
        checksum = zlib.crc32(
            struct.pack(">HHH", image.channels, image.height, image.width), checksum
        )
        checksum = zlib.crc32(image.pixels.tobytes(), checksum)
    return (wire.seed ^ checksum) & 0xFFFFFFFF


def serve(*, runner: Any, spec: RunnerSpec, synthetic_seed: bool) -> int:
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    write_frame(
        sink,
        encode_hello(
            chunk_size=spec.chunk_size,
            model_dim=spec.model_dim,
            robot_dim=spec.robot_dim,
            state_dim=spec.state_dim,
            images=spec.images,
        ),
    )

    while (payload := read_frame(source)) is not None:
        request_id = 0
        try:
            decode_started = monotonic_ns()
            wire = decode_request(payload)
            decode_ns = monotonic_ns() - decode_started
            request_id = wire.request_id
            if wire.action_count != spec.chunk_size:
                raise ProtocolError(
                    f"requested {wire.action_count} actions, worker provides {spec.chunk_size}"
                )
            try:
                stitching = _STITCHING[wire.stitching]
            except KeyError as exc:
                raise ProtocolError(f"unknown stitching value: {wire.stitching}") from exc
            observation_started = monotonic_ns()
            _validate_observation_schema(wire, spec)
            observation = _observation_from_wire(wire)
            observation_ns = monotonic_ns() - observation_started
            model_started = monotonic_ns()
            rtc = None
            if wire.rtc is not None:
                rtc = RtcConditioning(
                    source_chunk_id=wire.rtc.source_chunk_id,
                    prefix_first_step=wire.rtc.prefix_first_step,
                    inference_delay=wire.rtc.inference_delay,
                    execution_horizon=wire.rtc.execution_horizon,
                    prefix=wire.rtc.prefix,
                )
            result = runner.predict(
                InferenceRequest(
                    first_step=wire.first_step,
                    last_emitted_step=wire.last_emitted_step,
                    action_count=wire.action_count,
                    stitching=stitching,
                    rtc=rtc,
                ),
                observation,
                seed=_seed_for_observation(wire) if synthetic_seed else wire.seed,
            )
            model_ns = monotonic_ns() - model_started
            response = WireResponse(
                request_id=wire.request_id,
                first_step=result.first_step,
                observation_sequence=result.observation_sequence,
                capture_ns=observation.capture_ns,
                model_actions=result.model_actions,
                robot_actions=result.robot_actions,
                timing=WireTiming(decode_ns, observation_ns, model_ns),
            )
            write_frame(sink, encode_response(response))
        except Exception as exc:  # keep one bad request from killing the worker
            print(f"cerebellum model worker: {exc}", file=sys.stderr, flush=True)
            write_frame(sink, encode_response_error(request_id, str(exc)))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=("synthetic", "smolvla"), default="synthetic")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--robot-dim", type=int, default=6)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtc-denoise-steps", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        runner, spec, synthetic_seed = _make_runner(args)
        raise SystemExit(serve(runner=runner, spec=spec, synthetic_seed=synthetic_seed))
    except Exception as exc:
        try:
            write_frame(sys.stdout.buffer, encode_hello_error(str(exc)))
        except Exception:
            pass
        print(f"cerebellum model worker failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
