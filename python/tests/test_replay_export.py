from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from cerebellum_model.replay_export import MAGIC, export_npz


def episode(path: Path, *, offsets: np.ndarray | None = None) -> Path:
    source = path / "episode.npz"
    np.savez(
        source,
        task=np.asarray("move the block"),
        offset_ns=(
            np.asarray([0, 33_333_333, 66_666_666], dtype=np.int64)
            if offsets is None
            else offsets
        ),
        sequence=np.asarray([7, 8, 9], dtype=np.uint64),
        state=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32),
        action=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype=np.float32),
        **{
            "image.observation.images.camera1": np.arange(
                3 * 2 * 2 * 3, dtype=np.uint8
            ).reshape(3, 2, 2, 3)
        },
    )
    return source


def test_export_writes_versioned_little_endian_header(tmp_path: Path) -> None:
    output = tmp_path / "episode.cbr"
    export_npz(episode(tmp_path), output)
    payload = output.read_bytes()
    assert payload[:8] == MAGIC
    frames, state_dim, cameras, action_dim, reserved, task_size = struct.unpack_from(
        "<IHHHHI", payload, 8
    )
    assert (frames, state_dim, cameras, action_dim, reserved) == (3, 2, 1, 2, 0)
    assert payload[24 : 24 + task_size] == b"move the block"


def test_export_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    source = episode(tmp_path, offsets=np.asarray([0, 5, 4], dtype=np.int64))
    with pytest.raises(ValueError, match="strictly increase"):
        export_npz(source, tmp_path / "bad.cbr")


def test_export_rejects_non_uint8_images(tmp_path: Path) -> None:
    source = tmp_path / "bad-image.npz"
    np.savez(
        source,
        task=np.asarray("move"),
        offset_ns=np.asarray([0], dtype=np.int64),
        state=np.asarray([[0]], dtype=np.float32),
        **{"image.camera": np.zeros((1, 2, 2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="uint8 NHWC"):
        export_npz(source, tmp_path / "bad.cbr")
