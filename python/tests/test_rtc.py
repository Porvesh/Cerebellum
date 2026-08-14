from __future__ import annotations

import numpy as np
import pytest

from cerebellum_model.rtc import RtcProcessor, prefix_weights


def test_linear_prefix_weights_match_rtc_schedule() -> None:
    np.testing.assert_allclose(
        prefix_weights(2, 5, 8),
        np.array([1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        prefix_weights(7, 3, 5),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_prefix_weights_match_installed_lerobot_oracle() -> None:
    pytest.importorskip("lerobot")
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor as LeRobotRtcProcessor

    oracle = LeRobotRtcProcessor(
        RTCConfig(enabled=True, prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    )
    expected = oracle.get_prefix_weights(5, 9, 50).numpy()
    np.testing.assert_allclose(prefix_weights(5, 9, 50), expected)


def test_guidance_uses_denoiser_vjp_not_only_identity_path() -> None:
    torch = pytest.importorskip("torch")
    processor = RtcProcessor(torch, max_guidance_weight=5.0)
    latent = torch.ones((1, 3, 1), dtype=torch.float32)
    prefix = torch.zeros_like(latent)

    # v=2x makes predicted_clean=x-time*2x. At time=.25 its Jacobian is .5.
    # A detached denoiser would incorrectly use an identity Jacobian here.
    guided = processor.denoise_step(
        latent,
        prefix,
        inference_delay=3,
        execution_horizon=3,
        time=0.25,
        original_denoise_step_partial=lambda value: 2.0 * value,
    )
    base = 2.0 * latent
    assert torch.all(guided > base)
