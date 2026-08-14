"""Runner contract plus a dependency-light synthetic implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .messages import ActionChunkResult, InferenceRequest, Observation
from .rtc import prefix_weights


class ActionChunkRunner(Protocol):
    def predict(
        self, request: InferenceRequest, observation: Observation, *, seed: int = 0
    ) -> ActionChunkResult: ...


class SyntheticRunner:
    """Deterministic stand-in used to test the transport contract without ML."""

    def __init__(self, *, model_dim: int = 32, robot_dim: int = 6) -> None:
        if not 0 < robot_dim <= model_dim:
            raise ValueError("expected 0 < robot_dim <= model_dim")
        self.model_dim = model_dim
        self.robot_dim = robot_dim

    def predict(
        self, request: InferenceRequest, observation: Observation, *, seed: int = 0
    ) -> ActionChunkResult:
        rng = np.random.default_rng(seed)
        model_actions = rng.standard_normal(
            (request.action_count, self.model_dim), dtype=np.float32
        )
        if request.rtc is not None:
            prefix = request.rtc.prefix
            if prefix.shape[1] != self.model_dim:
                raise ValueError("RTC prefix model dimension mismatch")
            overlap = min(prefix.shape[0], request.action_count)
            weights = prefix_weights(
                request.rtc.inference_delay,
                request.rtc.execution_horizon,
                request.action_count,
            )[:overlap, None]
            model_actions[:overlap] = (
                weights * prefix[:overlap] +
                (np.float32(1.0) - weights) * model_actions[:overlap]
            )
        # Synthetic mode has no normalization pipeline, so the executable view
        # is simply the real-dimensional slice.
        robot_actions = model_actions[:, : self.robot_dim].copy()
        return ActionChunkResult(
            first_step=request.first_step,
            observation_sequence=observation.sequence,
            model_actions=model_actions,
            robot_actions=robot_actions,
        )
