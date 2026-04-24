"""Wrapper for SONIC's encoder ONNX model.

The encoder compresses motion reference data into a 64-D token that the
decoder uses as a conditioning signal.  For locomotion-only training at a
fixed velocity range we cache a "walking token" and skip per-step encoding.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from robo_residual import OnnxBasePolicy


class SonicEncoderWrapper:
    """Thin wrapper around model_encoder.onnx.

    Modes:
        - ``cached``: compute the token once from a reference motion and reuse
          it every step.  Best for constant-velocity locomotion training.
        - ``live``: run the encoder every step with fresh motion reference data.
          Required for motion-tracking or teleop scenarios.
    """

    def __init__(
        self,
        encoder_onnx_path: str | Path,
        device: str = "cuda",
        mode: str = "cached",
    ) -> None:
        self.device = device
        self.mode = mode
        self._encoder = OnnxBasePolicy(str(encoder_onnx_path), device=device)
        self._cached_token: Tensor | None = None

    @property
    def token_dim(self) -> int:
        return self._encoder.num_actions  # encoder output = token

    def encode(self, encoder_obs: Tensor) -> Tensor:
        """Run the encoder on raw motion reference observations.

        Args:
            encoder_obs: (N, encoder_input_dim) motion reference data.

        Returns:
            (N, 64) token tensor on ``self.device``.
        """
        return self._encoder.forward(encoder_obs)

    def cache_token(self, encoder_obs: Tensor) -> None:
        """Compute and cache a token for reuse in ``cached`` mode."""
        with torch.no_grad():
            self._cached_token = self.encode(encoder_obs[:1]).detach()

    def get_token(
        self, num_envs: int, encoder_obs: Tensor | None = None
    ) -> Tensor:
        """Return the token for the current step.

        In ``cached`` mode, returns the pre-computed token broadcast to
        ``num_envs``.  In ``live`` mode, runs the encoder on ``encoder_obs``.
        """
        if self.mode == "cached":
            if self._cached_token is None:
                # Fallback: zero token (decoder still works, just not conditioned)
                return torch.zeros(num_envs, self.token_dim, device=self.device)
            return self._cached_token.expand(num_envs, -1)

        if encoder_obs is None:
            raise ValueError("encoder_obs required in 'live' mode")
        return self.encode(encoder_obs)


class ZeroTokenProvider:
    """Dummy provider that returns zero tokens (no encoder needed).

    Useful for testing or when the encoder ONNX is not available.
    """

    def __init__(self, token_dim: int = 64, device: str = "cuda") -> None:
        self._token_dim = token_dim
        self.device = device

    @property
    def token_dim(self) -> int:
        return self._token_dim

    def get_token(
        self, num_envs: int, encoder_obs: Tensor | None = None
    ) -> Tensor:
        return torch.zeros(num_envs, self._token_dim, device=self.device)
