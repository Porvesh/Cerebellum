"""Convert a dependency-light NumPy episode into Cerebellum's CBR format."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

MAGIC = b"CBRPLY1\0"
IMAGE_PREFIX = "image."
MAX_TASK_BYTES = 1 << 20
MAX_IMAGES = 64
MAX_PADDED_ACTION_DIM = 32


def _write(output: BinaryIO, fmt: str, *values: object) -> None:
    output.write(struct.pack("<" + fmt, *values))


def _task(value: NDArray[np.generic]) -> str:
    if value.ndim != 0:
        raise ValueError("task must be a scalar string")
    task = str(value.item())
    if not task:
        raise ValueError("task must not be empty")
    if len(task.encode("utf-8")) > MAX_TASK_BYTES:
        raise ValueError("task is too large")
    return task


def export_npz(input_path: Path, output_path: Path) -> None:
    """Export one NPZ episode.

    Required arrays:
      task: scalar Unicode string
      offset_ns: (frames,) int64, beginning at zero and strictly increasing
      state: (frames, state_dim) finite numeric values
      image.<feature-name>: (frames, height, width, channels) uint8

    Optional arrays:
      sequence: (frames,) uint64; defaults to 1..frames
      action: (frames, action_dim) finite numeric reference commands
    """

    with np.load(input_path, allow_pickle=False) as data:
        required = {"task", "offset_ns", "state"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")

        task = _task(np.asarray(data["task"]))
        offsets = np.asarray(data["offset_ns"], dtype=np.int64)
        state = np.asarray(data["state"], dtype=np.float32)
        if offsets.ndim != 1 or offsets.size == 0:
            raise ValueError("offset_ns must be a nonempty rank-1 array")
        if offsets[0] != 0 or np.any(offsets < 0) or np.any(np.diff(offsets) <= 0):
            raise ValueError("offset_ns must start at zero and strictly increase")
        frames = int(offsets.size)
        if state.ndim != 2 or state.shape[0] != frames or state.shape[1] == 0:
            raise ValueError("state must have shape (frames, positive state_dim)")
        if state.shape[1] > np.iinfo(np.uint16).max or not np.isfinite(state).all():
            raise ValueError("state dimensions or values are invalid")

        if "sequence" in data.files:
            sequence = np.asarray(data["sequence"], dtype=np.uint64)
            if sequence.shape != (frames,) or np.any(np.diff(sequence) == 0):
                raise ValueError("sequence must have shape (frames,) and strictly increase")
            if frames > 1 and np.any(sequence[1:] <= sequence[:-1]):
                raise ValueError("sequence must strictly increase")
        else:
            sequence = np.arange(1, frames + 1, dtype=np.uint64)

        actions: NDArray[np.float32] | None = None
        if "action" in data.files:
            actions = np.asarray(data["action"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[0] != frames
                or not 0 < actions.shape[1] <= MAX_PADDED_ACTION_DIM
                or not np.isfinite(actions).all()
            ):
                raise ValueError("action must have shape (frames, 1..32) and be finite")

        image_names = sorted(name for name in data.files if name.startswith(IMAGE_PREFIX))
        if not image_names or len(image_names) > MAX_IMAGES:
            raise ValueError("episode must contain between 1 and 64 image arrays")
        images: list[tuple[str, NDArray[np.uint8]]] = []
        for key in image_names:
            feature_name = key[len(IMAGE_PREFIX) :]
            encoded_name = feature_name.encode("utf-8")
            image = np.asarray(data[key])
            if not feature_name or len(encoded_name) > np.iinfo(np.uint16).max:
                raise ValueError(f"invalid image feature name: {feature_name!r}")
            if image.dtype != np.uint8 or image.ndim != 4 or image.shape[0] != frames:
                raise ValueError(f"{key} must be uint8 NHWC with one image per frame")
            height, width, channels = image.shape[1:]
            if not all(0 < value <= np.iinfo(np.uint16).max for value in (channels, height, width)):
                raise ValueError(f"{key} dimensions exceed CBR storage")
            images.append((feature_name, np.ascontiguousarray(image)))

        task_bytes = task.encode("utf-8")
        with output_path.open("wb") as output:
            output.write(MAGIC)
            _write(
                output,
                "IHHHHI",
                frames,
                state.shape[1],
                len(images),
                0 if actions is None else actions.shape[1],
                0,
                len(task_bytes),
            )
            output.write(task_bytes)
            for feature_name, image in images:
                name = feature_name.encode("utf-8")
                height, width, channels = image.shape[1:]
                _write(output, "H", len(name))
                output.write(name)
                _write(output, "HHH", channels, height, width)
            for index in range(frames):
                _write(output, "QQ", int(sequence[index]), int(offsets[index]))
                output.write(np.asarray(state[index], dtype="<f4").tobytes())
                for _, image in images:
                    output.write(image[index].tobytes())
                if actions is not None:
                    output.write(np.asarray(actions[index], dtype="<f4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source NPZ episode")
    parser.add_argument("--output", type=Path, required=True, help="destination .cbr episode")
    args = parser.parse_args()
    export_npz(args.input, args.output)


if __name__ == "__main__":
    main()
