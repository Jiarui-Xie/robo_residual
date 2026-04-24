"""Reward weights for energy-efficient gait training at 2-3 m/s."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Weights for each reward component.

    Signs are baked in: positive = reward, negative = penalty.
    """

    # Phase 1 weights: prioritize walking first, energy optimization second.
    # Once policy stably walks at 2-3 m/s, increase energy weight and reduce
    # velocity weight to focus on efficiency.

    # -- tracking (primary signal) --
    velocity_tracking: float = 5.0         # exp-shaped, dominant positive signal
    velocity_tracking_sigma: float = 0.5   # forgiving tracking

    # -- energy (optimization target) --
    energy_penalty: float = -0.002         # v7-level (known working). v9-a10 collapsed to slow gait,
                                           # v9-a5 collapsed to fallen robot. Return to -0.002 baseline.
    energy_balance_penalty: float = -0.005

    # -- stability --
    ang_vel_penalty: float = -0.05
    foot_slip_penalty: float = -0.2
    base_height_penalty: float = -0.5
    target_base_height: float = 0.72

    # -- smoothness --
    action_smoothness: float = -0.005

    # -- residual regularization (L2: sum residual², pulls residual toward 0) --
    residual_magnitude: float = -0.01      # high-speed needs more residual freedom; reduced from -0.05

    # -- survival --
    alive_bonus: float = 2.0


# Speed range sampled during training
# Set to match IsaacLab achievable range with recorded_run_fast.npz reference.
# Base SONIC achieves ~1.63 m/s with this reference; residual should push towards 2.0+.
CMD_VEL_MIN: float = 1.3   # m/s
CMD_VEL_MAX: float = 2.2   # m/s
