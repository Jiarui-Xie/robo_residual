"""Composable multi-phase residual stacking.

Stack multiple residual phases on top of an ONNX base:
  Phase 0: ONNX base policy (frozen)
  Phase 1: Residual for objective A (trained, then frozen as ONNX)
  Phase 2: Residual for objective B (actively training)
  ...

Each phase has its own clamp limits. After training, fuse the active phase
into the base ONNX and start the next phase.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from robo_residual.config.residual_config import ResidualConfig
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import _build_mlp
from robo_residual.utils.zero_init import zero_init_output_layer


class ComposableResidual(nn.Module):
    """ONNX base + N frozen ONNX residual phases + 1 active PyTorch residual.

    The frozen phases are loaded as ONNX models (via OnnxBasePolicy).
    Only the active residual MLP and critic are trainable.

    Args:
        base_onnx_path: Path to the base policy ONNX.
        frozen_residual_onnx_paths: Paths to previously trained & fused residual ONNX files.
            Each is run in sequence, their outputs summed with the base.
        frozen_residual_limits: Per-joint clamp limits for each frozen residual phase.
        active_config: Config for the currently training residual phase.
        num_actor_obs: Observation dim for the actor. Auto-detected from base ONNX if None.
        num_critic_obs: Observation dim for the critic. Defaults to num_actor_obs.
        num_actions: Action dim. Auto-detected from base ONNX if None.
        critic_hidden_dims: Hidden dims for the critic MLP.
        device: Device for PyTorch modules.
    """

    def __init__(
        self,
        base_onnx_path: str | Path,
        frozen_residual_onnx_paths: list[str | Path],
        frozen_residual_limits: list[torch.Tensor],
        active_config: ResidualConfig,
        num_actor_obs: int | None = None,
        num_critic_obs: int | None = None,
        num_actions: int | None = None,
        critic_hidden_dims: list[int] | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._device = device

        # Load base
        self.base = OnnxBasePolicy(
            base_onnx_path,
            device=device,
            obs_input=active_config.base_obs_input,
            action_output=active_config.base_action_output,
        )

        _num_actor_obs = num_actor_obs if num_actor_obs is not None else self.base.num_obs
        _num_actions = num_actions if num_actions is not None else self.base.num_actions
        _num_critic_obs = num_critic_obs if num_critic_obs is not None else _num_actor_obs

        # Load frozen residual phases
        self.frozen_residuals: list[OnnxBasePolicy] = []
        for path in frozen_residual_onnx_paths:
            phase = OnnxBasePolicy(path, device=device)
            self.frozen_residuals.append(phase)

        # Store frozen limits as buffers
        for i, lim in enumerate(frozen_residual_limits):
            self.register_buffer(f"frozen_limits_{i}", lim)
        self._num_frozen = len(frozen_residual_limits)

        # Active residual (trainable)
        active_limits = active_config.build_residual_limits(_num_actions)
        self.register_buffer("active_limits", active_limits)

        self.active_residual = _build_mlp(
            _num_actor_obs,
            _num_actions,
            active_config.residual_hidden_dims,
            active_config.residual_activation,
        )
        zero_init_output_layer(self.active_residual)

        # Critic
        _critic_dims = critic_hidden_dims or active_config.residual_hidden_dims
        self.critic = _build_mlp(_num_critic_obs, 1, _critic_dims, active_config.residual_activation)

        # Noise
        if active_config.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(active_config.init_noise_std * torch.ones(_num_actions))
            )
        else:
            self.std = nn.Parameter(
                active_config.init_noise_std * torch.ones(_num_actions)
            )
        self._noise_std_type = active_config.noise_std_type
        self._num_actor_obs = _num_actor_obs
        self._num_actions = _num_actions

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _compute_action_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.base.forward(actor_obs)

            # Add frozen residuals
            for i, frozen_res in enumerate(self.frozen_residuals):
                lim = getattr(self, f"frozen_limits_{i}")
                delta = frozen_res.forward(actor_obs)
                delta = torch.clamp(delta, -lim, lim)
                action = action + delta

        # Active residual (has grad)
        active_delta = self.active_residual(actor_obs)
        active_delta = torch.clamp(active_delta, -self.active_limits, self.active_limits)
        return action + active_delta

    def _get_std(self) -> torch.Tensor:
        if self._noise_std_type == "log":
            return torch.exp(self.log_std)
        return self.std

    def act(self, actor_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = self._compute_action_mean(actor_obs)
        std = self._get_std().expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self._compute_action_mean(actor_obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass
