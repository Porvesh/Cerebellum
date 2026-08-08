"""LeRobot 0.4.4 SmolVLA adapter.

The public policy API slices actions to the robot dimension. RTC needs the
pre-slice padded trajectory, so this adapter mirrors the small public
predict_action_chunk path and calls model.sample_actions directly. The LeRobot
version is pinned in requirements-smolvla.txt and checked at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import monotonic_ns
from typing import Any

import numpy as np

from .messages import ActionChunkResult, InferenceRequest, Observation

SUPPORTED_LEROBOT_VERSION = "0.4.4"
DEFAULT_MODEL_ID = "lerobot/smolvla_base"


@dataclass(frozen=True, slots=True)
class ForwardTiming:
    preprocess_ns: int
    inference_ns: int
    postprocess_ns: int


class SmolVLARunner:
    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        torch_module: Any,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.torch = torch_module
        self.last_timing: ForwardTiming | None = None

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL_ID,
        *,
        device: str = "cuda",
        local_files_only: bool = False,
    ) -> "SmolVLARunner":
        installed = version("lerobot")
        if installed != SUPPORTED_LEROBOT_VERSION:
            raise RuntimeError(
                f"SmolVLARunner targets lerobot {SUPPORTED_LEROBOT_VERSION}, got {installed}"
            )

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        model_ref = str(model)
        config = PreTrainedConfig.from_pretrained(
            model_ref, local_files_only=local_files_only
        )
        config.device = device
        policy = SmolVLAPolicy.from_pretrained(
            model_ref,
            config=config,
            local_files_only=local_files_only,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            model_ref,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        policy.eval()
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            torch_module=torch,
        )

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def model_action_dim(self) -> int:
        return int(self.policy.config.max_action_dim)

    @property
    def robot_action_dim(self) -> int:
        return int(self.policy.config.action_feature.shape[0])

    def synthetic_observation(self, *, seed: int = 0) -> Observation:
        """Build deterministic, correctly named inputs from the checkpoint config."""

        rng = np.random.default_rng(seed)
        state_dim = int(self.policy.config.robot_state_feature.shape[0])
        state = rng.standard_normal(state_dim, dtype=np.float32)
        images = {
            name: rng.random(feature.shape, dtype=np.float32)
            for name, feature in self.policy.config.image_features.items()
        }
        return Observation(
            sequence=0,
            capture_ns=monotonic_ns(),
            state=state,
            images=images,
            task="move the object to the target",
        )

    def predict(
        self, request: InferenceRequest, observation: Observation, *, seed: int = 0
    ) -> ActionChunkResult:
        if request.stitching == "rtc":
            raise NotImplementedError("RTC conditioning is the next model-runner milestone")
        if request.action_count != self.chunk_size:
            raise ValueError(
                f"checkpoint emits {self.chunk_size} actions, request asks for {request.action_count}"
            )

        torch = self.torch
        raw: dict[str, Any] = {
            "observation.state": torch.from_numpy(observation.state),
            "task": observation.task,
        }
        raw.update({name: torch.from_numpy(image) for name, image in observation.images.items()})

        t0 = monotonic_ns()
        batch = self.preprocessor(raw)
        t1 = monotonic_ns()

        batch = self.policy._prepare_batch(batch)
        images, image_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        language_tokens = batch["observation.language.tokens"]
        language_masks = batch["observation.language.attention_mask"]

        generator = torch.Generator(device=self.policy.config.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            1,
            self.chunk_size,
            self.model_action_dim,
            generator=generator,
            device=self.policy.config.device,
            dtype=state.dtype,
        )

        with torch.inference_mode():
            model_actions = self.policy.model.sample_actions(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state,
                noise=noise,
            )
        if self.policy.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = monotonic_ns()

        robot_actions = model_actions[:, :, : self.robot_action_dim]
        robot_actions = self.postprocessor(robot_actions[0])
        t3 = monotonic_ns()
        self.last_timing = ForwardTiming(t1 - t0, t2 - t1, t3 - t2)

        return ActionChunkResult(
            first_step=request.first_step,
            observation_sequence=observation.sequence,
            model_actions=model_actions[0].detach().float().cpu().numpy(),
            robot_actions=robot_actions.detach().float().cpu().numpy(),
        )
