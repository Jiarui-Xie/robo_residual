"""Unitree H1 19-DOF joint layout and pre-built ResidualConfigs."""

from __future__ import annotations

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig

# ── Joint indices (standard Unitree H1 ordering) ─────────────────────────── #

LEFT_HIP_YAW = 0
LEFT_HIP_ROLL = 1
LEFT_HIP_PITCH = 2
LEFT_KNEE = 3
LEFT_ANKLE = 4

RIGHT_HIP_YAW = 5
RIGHT_HIP_ROLL = 6
RIGHT_HIP_PITCH = 7
RIGHT_KNEE = 8
RIGHT_ANKLE = 9

WAIST_YAW = 10

LEFT_SHOULDER_PITCH = 11
LEFT_SHOULDER_ROLL = 12
LEFT_SHOULDER_YAW = 13
LEFT_ELBOW = 14

RIGHT_SHOULDER_PITCH = 15
RIGHT_SHOULDER_ROLL = 16
RIGHT_SHOULDER_YAW = 17
RIGHT_ELBOW = 18

NUM_JOINTS = 19

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]

LEG_INDICES = list(range(0, 10))
WAIST_INDICES = [10]
ARM_INDICES = list(range(11, 19))

# ── Pre-built configs ─────────────────────────────────────────────────────── #

def h1_conservative(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Small residual budget — safe starting point for H1 policy fine-tuning."""
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("legs", LEG_INDICES, max_residual=0.05),
            JointGroupConfig("waist", WAIST_INDICES, max_residual=0.05),
            JointGroupConfig("arms", ARM_INDICES, max_residual=0.10),
        ],
        residual_hidden_dims=hidden_dims or [256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
    )


def h1_energy_efficient(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Larger leg budget for energy / speed optimisation on H1."""
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("legs", LEG_INDICES, max_residual=0.30),
            JointGroupConfig("waist", WAIST_INDICES, max_residual=0.10),
            JointGroupConfig("arms", ARM_INDICES, max_residual=0.05),
        ],
        residual_hidden_dims=hidden_dims or [512, 256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
    )
