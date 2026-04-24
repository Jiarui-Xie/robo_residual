"""ResidualConfig for SONIC G1 energy-efficient gait at 2-3 m/s."""

from __future__ import annotations

from robo_residual import JointGroupConfig, ResidualConfig

from .g1_joints import ARM_INDICES, LEG_INDICES, WAIST_INDICES


def build_sonic_residual_config() -> ResidualConfig:
    """Build a ResidualConfig for G1 energy-efficient residual training.

    Joint budget rationale:
    - Legs (0.30 rad ≈ 17.2°): v1 (0.10) saturated at iter ~50 with energy overwhelming
      velocity gradient; bold clamp to let residual genuinely reshape gait pattern for
      dual speed+energy optimization at 3.0 m/s.
    - Waist (0.10 rad ≈ 5.7°): moderate — allow torso lean adjustments
      for dynamic balance at 2-3 m/s.
    - Arms (0.03 rad ≈ 1.7°): minimal — arms don't contribute much to
      locomotion energy; just dampen unnecessary swinging.
    """
    # v5-highspeed: leg clamp 0.45 — validated that residual doesn't destroy gait;
    # larger budget lets residual reshape stride for 3.0 m/s speed+energy optimization.
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("legs", LEG_INDICES, max_residual=0.35),
            JointGroupConfig("waist", WAIST_INDICES, max_residual=0.10),
            JointGroupConfig("arms", ARM_INDICES, max_residual=0.03),
        ],
        residual_hidden_dims=[512, 256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
        noise_std_type="log",
        default_max_residual=0.35,
    )
