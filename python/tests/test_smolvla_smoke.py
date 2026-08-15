from __future__ import annotations

import os

import numpy as np
import pytest

from cerebellum_model.messages import InferenceRequest, RtcConditioning
from cerebellum_model.smolvla_runner import SmolVLARunner


@pytest.mark.smolvla
@pytest.mark.skipif(
    os.environ.get("CEREBELLUM_RUN_SMOLVLA") != "1",
    reason="set CEREBELLUM_RUN_SMOLVLA=1 to load the real checkpoint",
)
def test_one_real_action_chunk() -> None:
    device = os.environ.get("CEREBELLUM_SMOLVLA_DEVICE", "cpu")
    runner = SmolVLARunner.from_pretrained(device=device, local_files_only=True)
    assert runner.state_dim == 6
    assert len(runner.image_shapes) == 3
    assert all(shape == (3, 256, 256) for shape in runner.image_shapes.values())
    observation = runner.synthetic_observation(seed=11)
    request = InferenceRequest(0, -1, runner.chunk_size)
    result = runner.predict(request, observation, seed=11)

    assert result.model_actions.shape == (50, 32)
    assert result.robot_actions.shape == (50, 6)
    assert np.isfinite(result.model_actions).all()
    assert np.isfinite(result.robot_actions).all()


@pytest.mark.smolvla
@pytest.mark.skipif(
    os.environ.get("CEREBELLUM_RUN_SMOLVLA_RTC") != "1",
    reason="set CEREBELLUM_RUN_SMOLVLA_RTC=1 to run real RTC guidance",
)
def test_real_rtc_conditioned_chunk_moves_toward_prefix() -> None:
    device = os.environ.get("CEREBELLUM_SMOLVLA_DEVICE", "cuda")
    runner = SmolVLARunner.from_pretrained(device=device, local_files_only=True)
    observation = runner.synthetic_observation(seed=11)

    first = runner.predict(
        InferenceRequest(0, -1, runner.chunk_size, stitching="rtc"),
        observation,
        seed=11,
    )
    prefix = first.model_actions[5:].copy()
    conditioned = runner.predict(
        InferenceRequest(
            5,
            4,
            runner.chunk_size,
            stitching="rtc",
            rtc=RtcConditioning(1, 5, 5, 6, prefix),
        ),
        observation,
        seed=11,
    )

    # With the same observation and noise, an unguided second request would
    # reproduce first[0:5]. RTC should move it toward first[5:10].
    baseline_error = np.mean((first.model_actions[:5] - prefix[:5]) ** 2)
    guided_error = np.mean((conditioned.model_actions[:5] - prefix[:5]) ** 2)
    assert np.isfinite(conditioned.model_actions).all()
    assert np.isfinite(conditioned.robot_actions).all()
    assert guided_error < baseline_error
