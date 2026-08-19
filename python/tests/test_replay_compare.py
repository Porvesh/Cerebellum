from __future__ import annotations

from pathlib import Path

import pytest

from cerebellum_model.replay_compare import compare


def write_actions(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "step,deadline_offset_ns,emit_offset_ns,fallback,action_0,action_1\n"
        + "\n".join(rows)
        + "\n"
    )
    return path


def test_compare_excludes_fallbacks_and_measures_smoothness(tmp_path: Path) -> None:
    left = write_actions(
        tmp_path / "left.csv",
        ["0,0,0,1,0,0", "1,1,1,0,1,2", "2,2,2,0,2,4"],
    )
    right = write_actions(
        tmp_path / "right.csv",
        ["0,0,0,0,9,9", "1,1,1,0,0,0", "2,2,2,0,2,2"],
    )
    result = compare(left, right)
    assert result["paired_nonfallback_actions"] == 2
    assert result["left_fallback_actions"] == 1
    assert result["right_fallback_actions"] == 0
    assert result["disagreement"]["mean_absolute_error"] == pytest.approx(1.25)
    assert result["disagreement"]["root_mean_square_error"] == pytest.approx(1.5)
    assert result["left_smoothness"]["consecutive_pairs"] == 1
    assert result["right_smoothness"]["consecutive_pairs"] == 2


def test_compare_rejects_dimension_mismatch(tmp_path: Path) -> None:
    left = write_actions(tmp_path / "left.csv", ["0,0,0,0,1,2"])
    right = tmp_path / "right.csv"
    right.write_text(
        "step,deadline_offset_ns,emit_offset_ns,fallback,action_0\n0,0,0,0,1\n"
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        compare(left, right)
