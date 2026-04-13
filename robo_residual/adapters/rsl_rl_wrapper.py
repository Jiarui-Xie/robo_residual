"""rsl_rl-compatible wrapper for ResidualActorCritic.

Wraps ResidualActorCritic with TensorDict obs_groups support, making it a
drop-in replacement for rsl_rl's ActorCritic in OnPolicyRunner.

Usage in rsl_rl:
    from robo_residual.adapters.rsl_rl_wrapper import RslRlResidualActorCritic

    policy = RslRlResidualActorCritic(
        obs=env.get_observations(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        onnx_path="base_policy.onnx",
        config=residual_config,
        device="cuda",
    )
    # Now use with OnPolicyRunner as normal
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from robo_residual.adapters.obs_adapter import ObsAdapter
from robo_residual.config.residual_config import ResidualConfig
from robo_residual.core.residual_actor_critic import ResidualActorCritic


class RslRlResidualActorCritic(nn.Module):
    """rsl_rl-compatible wrapper around ResidualActorCritic.

    Matches the interface expected by rsl_rl's OnPolicyRunner / PPO:
    - Constructor takes ``(obs: TensorDict, obs_groups, ...)``
    - ``act(obs: TensorDict)`` / ``act_inference(obs: TensorDict)``
    - ``evaluate(obs: TensorDict)``
    - ``update_normalization(obs: TensorDict)``
    - Properties: ``action_mean``, ``action_std``, ``entropy``
    - ``is_recurrent = False``

    Args:
        obs: Initial TensorDict observation (used to infer dimensions).
        obs_groups: Dict mapping roles to obs group names.
        onnx_path: Path to the base policy ONNX file.
        config: Residual training configuration.
        num_base_obs: Observation dim for the base ONNX. If None, auto-detected.
        critic_hidden_dims: Hidden dims for the critic MLP.
        actor_obs_normalization: Whether to normalize actor obs.
        critic_obs_normalization: Whether to normalize critic obs.
        device: Device for PyTorch modules.
        **kwargs: Extra kwargs (ignored, for rsl_rl compatibility).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: dict[str, torch.Tensor] | Any,
        obs_groups: dict[str, list[str]],
        onnx_path: str | Path,
        config: ResidualConfig,
        num_base_obs: int | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.adapter = ObsAdapter(obs_groups)
        self.obs_groups = obs_groups

        # Infer dimensions from the initial observation
        actor_obs = self.adapter.get_actor_obs(obs)
        critic_obs = self.adapter.get_critic_obs(obs)
        num_actor_obs = actor_obs.shape[-1]
        num_critic_obs = critic_obs.shape[-1]

        # Auto-detect num_actions from ONNX
        from robo_residual.core.onnx_base import OnnxBasePolicy
        _base = OnnxBasePolicy(onnx_path, device="cpu",
                               obs_input=config.base_obs_input,
                               action_output=config.base_action_output)
        num_actions = _base.num_actions

        self.inner = ResidualActorCritic(
            onnx_path=onnx_path,
            config=config,
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_base_obs=num_base_obs,
            critic_hidden_dims=critic_hidden_dims,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            device=device,
        )

    # ── rsl_rl interface ──────────────────────────────────────────────

    def act(self, obs: dict[str, torch.Tensor] | Any, **kwargs) -> torch.Tensor:
        actor_obs = self.adapter.get_actor_obs(obs)
        return self.inner.act(actor_obs, **kwargs)

    def act_inference(self, obs: dict[str, torch.Tensor] | Any) -> torch.Tensor:
        actor_obs = self.adapter.get_actor_obs(obs)
        return self.inner.act_inference(actor_obs)

    def evaluate(self, obs: dict[str, torch.Tensor] | Any, **kwargs) -> torch.Tensor:
        critic_obs = self.adapter.get_critic_obs(obs)
        return self.inner.evaluate(critic_obs, **kwargs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.inner.get_actions_log_prob(actions)

    def update_normalization(self, obs: dict[str, torch.Tensor] | Any) -> None:
        actor_obs = self.adapter.get_actor_obs(obs)
        critic_obs = self.adapter.get_critic_obs(obs)
        self.inner.update_normalization(actor_obs, critic_obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.inner.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.inner.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.inner.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.inner.entropy

    @property
    def distribution(self):
        return self.inner.distribution

    # ── Expose inner for checkpoint / export ──────────────────────────

    @property
    def residual(self) -> nn.Module:
        return self.inner.residual

    @property
    def critic(self) -> nn.Module:
        return self.inner.critic

    @property
    def base(self):
        return self.inner.base

    @property
    def max_residual_limits(self) -> torch.Tensor:
        return self.inner.max_residual_limits

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.inner.trainable_parameters
