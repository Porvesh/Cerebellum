"""Controlled real-SmolVLA RTC denoising-step sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic_ns

import numpy as np

from .messages import InferenceRequest, RtcConditioning
from .smolvla_runner import DEFAULT_MODEL_ID, SmolVLARunner


def _summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _physical_gpu(args: argparse.Namespace) -> int | str | None:
    if args.physical_gpu is not None:
        return args.physical_gpu
    visible = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0].strip()
    )
    if not visible:
        return None
    return int(visible) if visible.isdecimal() else visible


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    runner = SmolVLARunner.from_pretrained(
        args.model,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    observation = runner.synthetic_observation(seed=args.seed)
    results: list[dict[str, object]] = []

    for steps in args.steps:
        runner.rtc_denoise_steps = steps
        first = runner.predict(
            InferenceRequest(0, -1, runner.chunk_size, stitching="rtc"),
            observation,
            seed=args.seed,
        )
        prefix = first.model_actions[args.inference_delay :].copy()
        request = InferenceRequest(
            args.inference_delay,
            args.inference_delay - 1,
            runner.chunk_size,
            stitching="rtc",
            rtc=RtcConditioning(
                1,
                args.inference_delay,
                args.inference_delay,
                args.execution_horizon,
                prefix,
            ),
        )

        for _ in range(args.warmup):
            runner.predict(request, observation, seed=args.seed)

        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        latency_ms: list[float] = []
        guided = None
        for _ in range(args.iterations):
            started = monotonic_ns()
            guided = runner.predict(request, observation, seed=args.seed)
            latency_ms.append((monotonic_ns() - started) / 1e6)
        assert guided is not None

        frozen = args.inference_delay
        baseline_mse = float(np.mean((first.model_actions[:frozen] - prefix[:frozen]) ** 2))
        guided_mse = float(np.mean((guided.model_actions[:frozen] - prefix[:frozen]) ** 2))
        peak_mib = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if args.device.startswith("cuda")
            else 0.0
        )
        results.append(
            {
                "denoise_steps": steps,
                "latency_ms": _summary(latency_ms),
                "baseline_prefix_mse": baseline_mse,
                "guided_prefix_mse": guided_mse,
                "guided_to_baseline_mse_ratio": (
                    guided_mse / baseline_mse if baseline_mse > 0.0 else 0.0
                ),
                "peak_allocated_mib": peak_mib,
            }
        )

    return {
        "runner": "smolvla",
        "device": args.device,
        "physical_gpu": _physical_gpu(args),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "inference_delay": args.inference_delay,
        "execution_horizon": args.execution_horizon,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        help="physical GPU index for report metadata; inferred from CUDA_VISIBLE_DEVICES when set",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 6, 8, 10])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--inference-delay", type=int, default=5)
    parser.add_argument("--execution-horizon", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if any(step <= 0 for step in args.steps):
        parser.error("all step counts must be positive")
    if args.execution_horizon < args.inference_delay:
        parser.error("execution horizon must cover inference delay")
    report = run(args)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
