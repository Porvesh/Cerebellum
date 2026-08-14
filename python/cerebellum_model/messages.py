"""Value objects at the C++ control ↔ Python inference boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RtcConditioning:
    """Committed model-space actions aligned with the new chunk's first step."""

    source_chunk_id: int
    prefix_first_step: int
    inference_delay: int
    execution_horizon: int
    prefix: FloatArray

    def __post_init__(self) -> None:
        value = np.ascontiguousarray(self.prefix, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] == 0:
            raise ValueError("RTC prefix must have shape (steps, dimensions)")
        if self.source_chunk_id <= 0:
            raise ValueError("RTC source_chunk_id must be positive")
        if self.inference_delay <= 0:
            raise ValueError("RTC inference_delay must be positive")
        if self.execution_horizon < self.inference_delay:
            raise ValueError("RTC execution_horizon must cover inference_delay")
        if not np.isfinite(value).all():
            raise ValueError("RTC prefix contains NaN or infinity")
        object.__setattr__(self, "prefix", value)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Ask the model for one action chunk on the absolute control timeline."""

    first_step: int
    last_emitted_step: int
    action_count: int
    stitching: str = "discard"
    rtc: RtcConditioning | None = None

    def __post_init__(self) -> None:
        if self.first_step < 0:
            raise ValueError("first_step must be non-negative")
        if self.last_emitted_step != self.first_step - 1:
            raise ValueError("first_step must immediately follow last_emitted_step")
        if self.action_count <= 0:
            raise ValueError("action_count must be positive")
        if self.stitching not in {"discard", "ensemble", "rtc"}:
            raise ValueError(f"unknown stitching policy: {self.stitching}")
        if self.rtc is not None:
            if self.stitching != "rtc":
                raise ValueError("RTC conditioning requires rtc stitching")
            if self.rtc.prefix_first_step != self.first_step:
                raise ValueError("RTC prefix is not aligned with first_step")


@dataclass(frozen=True, slots=True)
class Observation:
    """One unbatched observation in the feature names expected by SmolVLA."""

    sequence: int
    capture_ns: int
    state: FloatArray
    images: Mapping[str, FloatArray]
    task: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"state must be rank 1, got {state.shape}")
        if not self.images:
            raise ValueError("at least one image is required")
        normalized_images: dict[str, FloatArray] = {}
        for name, image in self.images.items():
            value = np.asarray(image, dtype=np.float32)
            if value.ndim != 3:
                raise ValueError(f"image {name!r} must be CHW rank 3, got {value.shape}")
            normalized_images[name] = np.ascontiguousarray(value)
        if not self.task.strip():
            raise ValueError("task must not be empty")
        object.__setattr__(self, "state", np.ascontiguousarray(state))
        object.__setattr__(self, "images", normalized_images)


@dataclass(frozen=True, slots=True)
class ActionChunkResult:
    """Both representations of a generated chunk.

    model_actions stays in SmolVLA's padded 32-dimensional model space for RTC.
    robot_actions is sliced and postprocessed into the six executable dimensions.
    Mixing these spaces would either command normalized values or condition RTC
    on denormalized values, both silently wrong.
    """

    first_step: int
    observation_sequence: int
    model_actions: FloatArray
    robot_actions: FloatArray

    def __post_init__(self) -> None:
        model = np.ascontiguousarray(self.model_actions, dtype=np.float32)
        robot = np.ascontiguousarray(self.robot_actions, dtype=np.float32)
        if model.ndim != 2 or robot.ndim != 2:
            raise ValueError("action arrays must have shape (steps, dimensions)")
        if model.shape[0] != robot.shape[0]:
            raise ValueError("model and robot action counts differ")
        if robot.shape[1] > model.shape[1]:
            raise ValueError("robot action dimension exceeds padded model dimension")
        if model.shape[0] == 0:
            raise ValueError("an action chunk cannot be empty")
        if not np.isfinite(model).all() or not np.isfinite(robot).all():
            raise ValueError("action chunk contains NaN or infinity")
        object.__setattr__(self, "model_actions", model)
        object.__setattr__(self, "robot_actions", robot)

    @property
    def action_count(self) -> int:
        return int(self.model_actions.shape[0])
