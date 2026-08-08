from __future__ import annotations

import numpy as np
import pytest

from cerebellum_model.messages import ActionChunkResult, InferenceRequest, Observation
from cerebellum_model.runner import SyntheticRunner


def observation() -> Observation:
    return Observation(
        sequence=7,
        capture_ns=123,
        state=np.zeros(6, dtype=np.float32),
        images={"observation.images.camera1": np.zeros((3, 8, 8), dtype=np.float32)},
        task="pick up the block",
    )


def test_request_is_on_one_absolute_timeline() -> None:
    request = InferenceRequest(first_step=44, last_emitted_step=43, action_count=50)
    assert request.first_step == 44
    with pytest.raises(ValueError, match="immediately follow"):
        InferenceRequest(first_step=44, last_emitted_step=40, action_count=50)


def test_synthetic_runner_returns_both_action_spaces() -> None:
    request = InferenceRequest(first_step=44, last_emitted_step=43, action_count=50)
    runner = SyntheticRunner(model_dim=32, robot_dim=6)

    result = runner.predict(request, observation(), seed=9)

    assert result.first_step == 44
    assert result.observation_sequence == 7
    assert result.model_actions.shape == (50, 32)
    assert result.robot_actions.shape == (50, 6)
    np.testing.assert_array_equal(result.robot_actions, result.model_actions[:, :6])


def test_synthetic_runner_is_deterministic_for_fixed_noise() -> None:
    request = InferenceRequest(first_step=0, last_emitted_step=-1, action_count=4)
    runner = SyntheticRunner()
    first = runner.predict(request, observation(), seed=123)
    second = runner.predict(request, observation(), seed=123)
    np.testing.assert_array_equal(first.model_actions, second.model_actions)


def test_result_rejects_non_finite_actions() -> None:
    model = np.zeros((2, 32), dtype=np.float32)
    model[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ActionChunkResult(
            first_step=0,
            observation_sequence=0,
            model_actions=model,
            robot_actions=np.zeros((2, 6), dtype=np.float32),
        )
