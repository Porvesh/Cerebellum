"""Cerebellum's RTC prefix guidance, following arXiv:2506.07339.

This module owns the algorithm. LeRobot supplies SmolVLA's denoiser, but its RTC
processor is not used, which keeps the bridge contract and guidance testable here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def prefix_weights(start: int, end: int, total: int) -> NDArray[np.float32]:
    """One for committed actions, linear fade, then zero for the free suffix."""

    if total <= 0 or start < 0 or end < 0:
        raise ValueError("expected non-negative horizons and total > 0")
    end = min(end, total)
    start = min(start, end)
    weights = np.zeros(total, dtype=np.float32)
    weights[:start] = np.float32(1.0)
    fade = end - start
    if fade > 0:
        weights[start:end] = np.linspace(
            1.0, 0.0, fade + 2, dtype=np.float32
        )[1:-1]
    return weights


@dataclass(slots=True)
class RtcRuntimeConfig:
    enabled: bool = True
    max_guidance_weight: float = 5.0


class RtcProcessor:
    """Apply RTC's VJP correction to SmolVLA's reverse-time flow velocity."""

    def __init__(self, torch_module: Any, *, max_guidance_weight: float = 5.0) -> None:
        if max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        self.torch = torch_module
        self.max_guidance_weight = float(max_guidance_weight)

    def is_debug_enabled(self) -> bool:
        return False

    def track(self, **_: Any) -> None:
        return None

    def _nvtx(self, name: str, tensor: Any) -> Any:
        if tensor.is_cuda:
            return self.torch.cuda.nvtx.range(name)
        return nullcontext()

    def denoise_step(
        self,
        x_t: Any,
        prev_chunk_left_over: Any,
        inference_delay: int,
        time: float | Any,
        original_denoise_step_partial: Callable[[Any], Any],
        execution_horizon: int | None = None,
    ) -> Any:
        torch = self.torch
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)
        if inference_delay is None or execution_horizon is None:
            raise ValueError("RTC requires inference_delay and execution_horizon")

        squeezed = x_t.ndim == 2
        latent = x_t.detach().clone()
        if squeezed:
            latent = latent.unsqueeze(0)
        prefix = prev_chunk_left_over
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        horizon = min(int(execution_horizon), int(prefix.shape[1]))

        with self._nvtx("rtc/prepare_guidance", latent):
            padded = torch.zeros_like(latent)
            steps = min(int(prefix.shape[1]), int(latent.shape[1]))
            dims = min(int(prefix.shape[2]), int(latent.shape[2]))
            padded[:, :steps, :dims] = prefix[:, :steps, :dims].to(
                device=latent.device, dtype=latent.dtype
            )
            weights = torch.as_tensor(
                prefix_weights(int(inference_delay), horizon, int(latent.shape[1])),
                device=latent.device,
                dtype=latent.dtype,
            )[None, :, None]

        # x must require grad before the denoiser call: RTC needs the full VJP
        # through x_1(x_t), including the denoiser Jacobian.
        with torch.enable_grad():
            latent.requires_grad_(True)
            with self._nvtx("rtc/base_denoiser", latent):
                velocity = original_denoise_step_partial(latent)
            with self._nvtx("rtc/vjp", latent):
                predicted_clean = latent - torch.as_tensor(
                    time, device=latent.device, dtype=latent.dtype
                ) * velocity
                error = (padded - predicted_clean) * weights
                correction = torch.autograd.grad(
                    predicted_clean,
                    latent,
                    grad_outputs=error.detach(),
                    retain_graph=False,
                )[0]

        tau = 1.0 - torch.as_tensor(time, device=latent.device, dtype=latent.dtype)
        one_minus_tau_sq = (1.0 - tau) ** 2
        inv_r2 = (one_minus_tau_sq + tau**2) / one_minus_tau_sq
        cap = torch.as_tensor(
            self.max_guidance_weight, device=latent.device, dtype=latent.dtype
        )
        c = torch.nan_to_num((1.0 - tau) / tau, posinf=self.max_guidance_weight)
        guidance = torch.minimum(
            torch.nan_to_num(c * inv_r2, posinf=self.max_guidance_weight), cap
        )
        with self._nvtx("rtc/apply_correction", latent):
            result = velocity - guidance * correction
        return result.squeeze(0) if squeezed else result
