"""Compare two Cerebellum replay action CSV files on their shared real steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _load(path: Path) -> tuple[dict[int, NDArray[np.float32]], set[int], int]:
    actions: dict[int, NDArray[np.float32]] = {}
    fallbacks: set[int] = set()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        action_fields = [name for name in reader.fieldnames if name.startswith("action_")]
        action_fields.sort(key=lambda name: int(name.removeprefix("action_")))
        if not action_fields:
            raise ValueError(f"{path}: no action columns")
        for row in reader:
            step = int(row["step"])
            if step in actions or step in fallbacks:
                raise ValueError(f"{path}: duplicate step {step}")
            if int(row["fallback"]):
                fallbacks.add(step)
                continue
            action = np.asarray([float(row[name]) for name in action_fields], dtype=np.float32)
            if not np.isfinite(action).all():
                raise ValueError(f"{path}: non-finite action at step {step}")
            actions[step] = action
    return actions, fallbacks, len(action_fields)


def _smoothness(actions: dict[int, NDArray[np.float32]]) -> dict[str, float | int]:
    differences = [
        actions[step] - actions[step - 1]
        for step in sorted(actions)
        if step - 1 in actions
    ]
    if not differences:
        return {
            "consecutive_pairs": 0,
            "mean_step_l2": 0.0,
            "p99_step_linf": 0.0,
            "max_step_linf": 0.0,
        }
    values = np.stack(differences)
    l2 = np.linalg.vector_norm(values, axis=1)
    linf = np.max(np.abs(values), axis=1)
    return {
        "consecutive_pairs": len(differences),
        "mean_step_l2": float(np.mean(l2)),
        "p99_step_linf": float(np.percentile(linf, 99)),
        "max_step_linf": float(np.max(linf)),
    }


def compare(
    left_path: Path,
    right_path: Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> dict[str, object]:
    left, left_fallbacks, left_dim = _load(left_path)
    right, right_fallbacks, right_dim = _load(right_path)
    if left_dim != right_dim:
        raise ValueError(f"action dimensions differ: {left_dim} != {right_dim}")
    paired_steps = sorted(left.keys() & right.keys())
    if not paired_steps:
        raise ValueError("the replays have no shared non-fallback actions")
    differences = np.stack([left[step] - right[step] for step in paired_steps])
    absolute = np.abs(differences)
    return {
        "format": "cerebellum-replay-comparison-v1",
        "left": left_label or str(left_path),
        "right": right_label or str(right_path),
        "action_dim": left_dim,
        "paired_nonfallback_actions": len(paired_steps),
        "left_fallback_actions": len(left_fallbacks),
        "right_fallback_actions": len(right_fallbacks),
        "disagreement": {
            "mean_absolute_error": float(np.mean(absolute)),
            "root_mean_square_error": float(np.sqrt(np.mean(differences * differences))),
            "max_linf_error": float(np.max(np.max(absolute, axis=1))),
        },
        "left_smoothness": _smoothness(left),
        "right_smoothness": _smoothness(right),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(
        compare(
            args.left,
            args.right,
            left_label=args.left_label,
            right_label=args.right_label,
        ),
        indent=2,
    ) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
