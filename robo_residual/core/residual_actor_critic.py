"""Residual Actor-Critic: ONNX base + PyTorch residual MLP + critic.

Architecture (same obs):
    actor_obs → [ONNX Base Policy]  → base_action (frozen, no grad)
    actor_obs → [Residual MLP]      → δ (trainable, zero-initialized)
    action_mean = base_action + clamp(δ, -limits, +limits)

Architecture (different obs — residual sees more than base):
    base_obs  → [ONNX Base Policy]  → base_action (frozen)
    actor_obs → [Residual MLP]      → δ (trainable)
    action_mean = base_action + clamp(δ, -limits, +limits)

The ONNX base can be any exported policy (MLP, LSTM, etc.).
Only the residual MLP and critic are trainable.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from robo_residual.config.residual_config import ResidualConfig
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.utils.normalizer import EmpiricalNormalization
from robo_residual.utils.zero_init import zero_init_output_layer


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str = "elu",
) -> nn.Sequential:
    """Build a simple MLP with given hidden dims and activation."""
    activation_fn = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }
    if activation not in activation_fn:
        raise ValueError(f"Unknown activation: {activation}. Choose from {list(activation_fn)}")

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation_fn[activation]())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActorCritic(nn.Module):
    """ONNX base policy + trainable residual MLP + trainable critic.

    Drop-in module for PPO training. The base policy is loaded from ONNX and
    is completely frozen (runs via onnxruntime). Only the residual MLP, critic,
    and noise parameters receive gradients.

    The residual MLP can take different (typically larger) observations than the
    base policy. For example, the base might only see proprioception, while the
    residual additionally sees terrain heightmaps.

    Args:
        onnx_path: Path to the base policy ONNX file.
        config: Residual training configuration.
        num_actor_obs: Observation dim for the residual actor. If ``None``,
            auto-detected from the ONNX model's main input. Can be larger than
            the base's obs dim if the residual needs extra information.
        num_critic_obs: Observation dim for the critic. If ``None``, defaults
            to ``num_actor_obs``.
        num_actions: Action dim. If ``None``, auto-detected from the ONNX model's
            main output.
        num_base_obs: Observation dim for the base ONNX policy. If ``None``,
            auto-detected from the ONNX model. Use this when ``num_actor_obs``
            is larger than the base's input — specify which slice of the actor
            obs to feed to the base.
        critic_hidden_dims: Hidden dims for the critic MLP. Defaults to
            ``config.residual_hidden_dims``.
        actor_obs_normalization: Whether to normalize actor obs (residual MLP input).
        critic_obs_normalization: Whether to normalize critic obs.
        device: Device for PyTorch modules ("cpu" or "cuda").
    """

    def __init__(
        self,
        onnx_path: str | Path,
        config: ResidualConfig,
        num_actor_obs: int | None = None,
        num_critic_obs: int | None = None,
        num_actions: int | None = None,
        num_base_obs: int | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # Load frozen ONNX base
        self.base = OnnxBasePolicy(
            onnx_path,
            device=device,
            obs_input=config.base_obs_input,
            action_output=config.base_action_output,
        )

        # Resolve dimensions
        self._num_base_obs = num_base_obs if num_base_obs is not None else self.base.num_obs
        self._num_actor_obs = num_actor_obs if num_actor_obs is not None else self._num_base_obs
        self._num_actions = num_actions if num_actions is not None else self.base.num_actions
        self._num_critic_obs = num_critic_obs if num_critic_obs is not None else self._num_actor_obs
        self._device = device

        # Build per-joint residual limits
        limits = config.build_residual_limits(self._num_actions)
        self.register_buffer("max_residual_limits", limits)

        # Residual MLP (trainable, zero-initialized output)
        # Takes actor_obs (may be larger than base_obs)
        self.residual = _build_mlp(
            self._num_actor_obs,
            self._num_actions,
            config.residual_hidden_dims,
            config.residual_activation,
        )
        zero_init_output_layer(self.residual)

        # Critic (trainable, fresh)
        _critic_dims = critic_hidden_dims or config.residual_hidden_dims
        self.critic = _build_mlp(
            self._num_critic_obs,
            1,
            _critic_dims,
            config.residual_activation,
        )

        # Observation normalizers
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self._num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self._num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # Action noise
        self._noise_std_type = config.noise_std_type
        if config.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(config.init_noise_std * torch.ones(self._num_actions))
            )
        elif config.noise_std_type == "scalar":
            self.std = nn.Parameter(
                config.init_noise_std * torch.ones(self._num_actions)
            )
        else:
            raise ValueError(f"noise_std_type must be 'log' or 'scalar', got '{config.noise_std_type}'")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def num_actor_obs(self) -> int:
        return self._num_actor_obs

    @property
    def num_base_obs(self) -> int:
        return self._num_base_obs

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that should go into the optimizer (residual + critic + noise)."""
        return list(self.parameters())

    # ── Obs slicing ───────────────────────────────────────────────────

    def _get_base_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Extract the slice of actor_obs that the base ONNX policy expects.

        If actor_obs is larger than base_obs, takes the first ``num_base_obs`` dims.
        Override this method for custom slicing logic.
        """
        if self._num_actor_obs == self._num_base_obs:
            return actor_obs
        return actor_obs[..., :self._num_base_obs]

    # ── Core forward ──────────────────────────────────────────────────

    def _compute_action_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """action_mean = onnx_base(base_obs) + clamp(residual(actor_obs))."""
        base_obs = self._get_base_obs(actor_obs)
        with torch.no_grad():
            base_action = self.base.forward(base_obs)
        delta = self.residual(actor_obs)
        delta = torch.clamp(delta, -self.max_residual_limits, self.max_residual_limits)
        return base_action + delta

    def _get_std(self) -> torch.Tensor:
        if self._noise_std_type == "log":
            return torch.exp(self.log_std)
        return self.std

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self._compute_action_mean(actor_obs)
        std = self._get_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    # ── PPO interface ─────────────────────────────────────────────────

    def act(self, actor_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample an action (training mode)."""
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (inference mode)."""
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self._compute_action_mean(actor_obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Critic value estimate."""
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability of actions under current distribution."""
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

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> None:
        """Update running normalization statistics.

        Call this each rollout step during training.

        Args:
            actor_obs: Actor observations for this step.
            critic_obs: Critic observations. If None, uses actor_obs.
        """
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            if critic_obs is None:
                critic_obs = actor_obs
            self.critic_obs_normalizer.update(critic_obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset internal state (no-op for MLP residual)."""
        pass
