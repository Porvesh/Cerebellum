"""Persistent stdin/stdout inference worker for the C++ bridge."""

from __future__ import annotations

import argparse
import sys
from time import monotonic_ns

import numpy as np

from .messages import InferenceRequest, Observation
from .protocol import (
    ProtocolError,
    WireResponse,
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


def _synthetic_observation(sequence: int) -> Observation:
    return Observation(
        sequence=sequence,
        capture_ns=monotonic_ns(),
        state=np.zeros(6, dtype=np.float32),
        images={"observation.images.camera1": np.zeros((3, 8, 8), dtype=np.float32)},
        task="synthetic transport test",
    )


def serve(*, chunk_size: int, model_dim: int, robot_dim: int) -> int:
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    runner = SyntheticRunner(model_dim=model_dim, robot_dim=robot_dim)
    write_frame(sink, encode_hello(chunk_size=chunk_size, model_dim=model_dim, robot_dim=robot_dim))

    sequence = 0
    while (payload := read_frame(source)) is not None:
        request_id = 0
        try:
            wire = decode_request(payload)
            request_id = wire.request_id
            if wire.action_count != chunk_size:
                raise ProtocolError(
                    f"requested {wire.action_count} actions, worker provides {chunk_size}"
                )
            try:
                stitching = _STITCHING[wire.stitching]
            except KeyError as exc:
                raise ProtocolError(f"unknown stitching value: {wire.stitching}") from exc
            if stitching == "rtc":
                raise NotImplementedError("RTC conditioning is not implemented")
            observation = _synthetic_observation(sequence)
            sequence += 1
            result = runner.predict(
                InferenceRequest(
                    first_step=wire.first_step,
                    last_emitted_step=wire.last_emitted_step,
                    action_count=wire.action_count,
                    stitching=stitching,
                ),
                observation,
                seed=wire.seed,
            )
            response = WireResponse(
                request_id=wire.request_id,
                first_step=result.first_step,
                observation_sequence=result.observation_sequence,
                capture_ns=observation.capture_ns,
                model_actions=result.model_actions,
                robot_actions=result.robot_actions,
            )
            write_frame(sink, encode_response(response))
        except Exception as exc:  # keep one bad request from killing the worker
            print(f"cerebellum model worker: {exc}", file=sys.stderr, flush=True)
            write_frame(sink, encode_response_error(request_id, str(exc)))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=("synthetic",), default="synthetic")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--robot-dim", type=int, default=6)
    args = parser.parse_args()
    try:
        raise SystemExit(
            serve(
                chunk_size=args.chunk_size,
                model_dim=args.model_dim,
                robot_dim=args.robot_dim,
            )
        )
    except Exception as exc:
        try:
            write_frame(sys.stdout.buffer, encode_hello_error(str(exc)))
        except Exception:
            pass
        print(f"cerebellum model worker failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
