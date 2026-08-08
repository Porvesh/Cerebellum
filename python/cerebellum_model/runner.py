"""Runner contract plus a dependency-light synthetic implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .messages import ActionChunkResult, InferenceRequest, Observation


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
        # Synthetic mode has no normalization pipeline, so the executable view
        # is simply the real-dimensional slice.
        robot_actions = model_actions[:, : self.robot_dim].copy()
        return ActionChunkResult(
            first_step=request.first_step,
            observation_sequence=observation.sequence,
            model_actions=model_actions,
            robot_actions=robot_actions,
        )
