"""Reward functions for energy-efficient gait training.

All functions are stateless and operate on batched tensors (N, ...).
Signs are applied by the caller using weights from RewardConfig.
"""

from __future__ import annotations

import torch
from torch import Tensor


def velocity_tracking(
    base_lin_vel: Tensor,  # (N, 3) world-frame linear velocity
    cmd_vel: Tensor,       # (N, 3) commanded velocity (x, y, yaw)
    sigma: float = 0.25,
) -> Tensor:
    """Gaussian reward for matching commanded forward velocity.

    Returns 1.0 at perfect tracking, decays to 0 with increasing error.
    """
    error_sq = torch.sum((base_lin_vel[:, :2] - cmd_vel[:, :2]) ** 2, dim=-1)
    return torch.exp(-error_sq / (sigma ** 2))


def energy_consumption(
    torques: Tensor,                    # (N, num_joints)
    joint_vel: Tensor,                  # (N, num_joints)
    joint_weights: Tensor | None = None,  # (num_joints,) per-joint multiplier
) -> Tensor:
    """Mechanical power: sum of |torque * angular_velocity| per env.

    joint_weights allows double-penalizing specific joints (e.g. hip pitch, ankles).
    """
    power = torch.abs(torques * joint_vel)
    if joint_weights is not None:
        power = power * joint_weights
    return torch.sum(power, dim=-1)


def energy_balance(
    torques: Tensor,           # (N, num_joints)
    joint_vel: Tensor,         # (N, num_joints)
    left_indices: list[int],   # native indices of left-leg joints
    right_indices: list[int],  # native indices of right-leg joints
) -> Tensor:
    """|E_left - E_right| — penalize asymmetric power between left/right legs.

    Larger value = more limp-like gait. Used to discourage one-leg-big-step strategies.
    """
    power = torch.abs(torques * joint_vel)
    e_left  = power[:, left_indices].sum(dim=-1)
    e_right = power[:, right_indices].sum(dim=-1)
    return torch.abs(e_left - e_right)


def angular_velocity_penalty(
    base_ang_vel: Tensor,  # (N, 3)
) -> Tensor:
    """Squared norm of body angular velocity — penalizes wobble."""
    return torch.sum(base_ang_vel ** 2, dim=-1)


def foot_slip(
    foot_vel: Tensor,        # (N, num_feet, 3)
    contact_forces: Tensor,  # (N, num_feet)
    threshold: float = 1.0,  # contact force threshold (N)
) -> Tensor:
    """Penalize foot sliding when in contact with ground."""
    in_contact = (contact_forces > threshold).float()
    slip_speed_sq = torch.sum(foot_vel ** 2, dim=-1)  # (N, num_feet)
    return torch.sum(slip_speed_sq * in_contact, dim=-1)


def action_smoothness(
    actions: Tensor,       # (N, num_joints) current actions
    prev_actions: Tensor,  # (N, num_joints) previous actions
) -> Tensor:
    """Squared difference between consecutive actions."""
    return torch.sum((actions - prev_actions) ** 2, dim=-1)


def base_height_tracking(
    base_height: Tensor,      # (N,)
    target_height: float,
) -> Tensor:
    """Penalize deviation from target standing height."""
    return (base_height - target_height) ** 2


def residual_magnitude(
    residual_delta: Tensor,  # (N, num_joints) raw residual output
) -> Tensor:
    """Penalize large residual corrections — encourage staying close to base."""
    return torch.sum(residual_delta ** 2, dim=-1)


def compute_total_reward(
    components: dict[str, Tensor],
    weights: dict[str, float],
) -> Tensor:
    """Weighted sum of reward components.

    Args:
        components: name -> (N,) tensor of raw reward values.
        weights: name -> scalar weight (sign included).

    Returns:
        (N,) total reward tensor.
    """
    total = torch.zeros_like(next(iter(components.values())))
    for name, value in components.items():
        if name in weights:
            total = total + weights[name] * value
    return total
