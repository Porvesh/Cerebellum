from __future__ import annotations

import os

import numpy as np
import pytest

from cerebellum_model.messages import InferenceRequest
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
