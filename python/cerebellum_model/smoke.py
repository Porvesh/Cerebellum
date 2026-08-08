"""Run one deterministic SmolVLA action-chunk forward pass."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .messages import InferenceRequest
from .smolvla_runner import DEFAULT_MODEL_ID, SmolVLARunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    runner = SmolVLARunner.from_pretrained(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    observation = runner.synthetic_observation(seed=args.seed)
    request = InferenceRequest(
        first_step=0,
        last_emitted_step=-1,
        action_count=runner.chunk_size,
    )
    result = runner.predict(request, observation, seed=args.seed)
    timing = runner.last_timing
    print(
        json.dumps(
            {
                "model_actions_shape": list(result.model_actions.shape),
                "robot_actions_shape": list(result.robot_actions.shape),
                "finite": bool(
                    np.isfinite(result.model_actions).all()
                    and np.isfinite(result.robot_actions).all()
                ),
                "timing_ms": {
                    "preprocess": timing.preprocess_ns / 1e6 if timing else None,
                    "inference": timing.inference_ns / 1e6 if timing else None,
                    "postprocess": timing.postprocess_ns / 1e6 if timing else None,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
