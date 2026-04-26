"""Unitree Go2 12-DOF joint layout and pre-built ResidualConfigs."""

from __future__ import annotations

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig

# ── Joint indices ────────────────────────────────────────────────────────── #
# Ordering matches Unitree Go2 LowState motor convention: FR, FL, RR, RL
# (each leg: hip, thigh, calf). Verify against your policy's training ordering
# before using named constants for per-joint reward shaping.

FR_HIP = 0
FR_THIGH = 1
FR_CALF = 2

FL_HIP = 3
FL_THIGH = 4
FL_CALF = 5

RR_HIP = 6
RR_THIGH = 7
RR_CALF = 8

RL_HIP = 9
RL_THIGH = 10
RL_CALF = 11

NUM_JOINTS = 12

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

HIP_INDICES = [0, 3, 6, 9]
THIGH_INDICES = [1, 4, 7, 10]
CALF_INDICES = [2, 5, 8, 11]
ALL_INDICES = list(range(12))

# ── Pre-built configs ─────────────────────────────────────────────────────── #

def go2_conservative(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Small residual budget — safe starting point for Go2 policy fine-tuning."""
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("hips", HIP_INDICES, max_residual=0.05),
            JointGroupConfig("thighs", THIGH_INDICES, max_residual=0.08),
            JointGroupConfig("calves", CALF_INDICES, max_residual=0.05),
        ],
        residual_hidden_dims=hidden_dims or [256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
    )


def go2_energy_efficient(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Larger budget for energy or gait optimisation on Go2."""
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("hips", HIP_INDICES, max_residual=0.10),
            JointGroupConfig("thighs", THIGH_INDICES, max_residual=0.20),
            JointGroupConfig("calves", CALF_INDICES, max_residual=0.15),
        ],
        residual_hidden_dims=hidden_dims or [256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
    )
