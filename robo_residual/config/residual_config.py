from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JointGroupConfig:
    """Per-joint-group residual clamp configuration.

    Like per-layer LoRA rank — different body parts get different residual budgets.
    """

    name: str
    indices: list[int]
    max_residual: float


@dataclass
class ResidualConfig:
    """Configuration for residual policy training on top of an ONNX base.

    Args:
        joint_groups: Per-group clamp limits. Joints not covered use ``default_max_residual``.
        residual_hidden_dims: Hidden layer sizes for the residual MLP.
        residual_activation: Activation function name (torch.nn convention).
        init_noise_std: Initial action noise standard deviation.
        noise_std_type: How noise std is parameterized — "log" (exp(param)) or "scalar" (raw param).
        default_max_residual: Clamp limit for joints not in any group.
        base_obs_input: Which ONNX input feeds the observation — index or name.
        base_action_output: Which ONNX output is the action — index or name.
    """

    joint_groups: list[JointGroupConfig] = field(default_factory=list)
    residual_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    residual_activation: str = "elu"
    init_noise_std: float = 0.1
    noise_std_type: str = "log"
    default_max_residual: float = 0.1
    base_obs_input: int | str = 0
    base_action_output: int | str = 0

    def validate(self, num_actions: int) -> None:
        """Check config consistency. Raises ValueError on problems."""
        all_indices: list[int] = []
        for group in self.joint_groups:
            for idx in group.indices:
                if idx < 0 or idx >= num_actions:
                    raise ValueError(
                        f"Joint group '{group.name}' has index {idx} "
                        f"outside action space [0, {num_actions})"
                    )
                if idx in all_indices:
                    raise ValueError(
                        f"Joint index {idx} appears in multiple groups"
                    )
                all_indices.append(idx)

    def build_residual_limits(self, num_actions: int) -> torch.Tensor:
        """Build per-joint clamp limit tensor from config.

        Returns:
            Tensor of shape ``(num_actions,)`` with per-joint max residual values.
        """
        self.validate(num_actions)
        limits = torch.full((num_actions,), self.default_max_residual)
        for group in self.joint_groups:
            for idx in group.indices:
                limits[idx] = group.max_residual
        return limits
