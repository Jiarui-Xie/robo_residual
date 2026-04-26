"""Unitree G1 29-DOF joint layout and pre-built ResidualConfigs."""

from __future__ import annotations

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig

# ── Joint indices ────────────────────────────────────────────────────────── #
# Ordering matches SONIC / IsaacLab training convention (hip_yaw=0).
# Note: the real-robot Unitree DDS LowState uses hip_pitch=0 — different order.
# Use these indices when building configs for SONIC-based ONNX policies.

LEFT_HIP_YAW = 0
LEFT_HIP_ROLL = 1
LEFT_HIP_PITCH = 2
LEFT_KNEE = 3
LEFT_ANKLE_PITCH = 4
LEFT_ANKLE_ROLL = 5

RIGHT_HIP_YAW = 6
RIGHT_HIP_ROLL = 7
RIGHT_HIP_PITCH = 8
RIGHT_KNEE = 9
RIGHT_ANKLE_PITCH = 10
RIGHT_ANKLE_ROLL = 11

WAIST_YAW = 12
WAIST_ROLL = 13
WAIST_PITCH = 14

LEFT_SHOULDER_PITCH = 15
LEFT_SHOULDER_ROLL = 16
LEFT_SHOULDER_YAW = 17
LEFT_ELBOW = 18
LEFT_WRIST_ROLL = 19
LEFT_WRIST_PITCH = 20
LEFT_WRIST_YAW = 21

RIGHT_SHOULDER_PITCH = 22
RIGHT_SHOULDER_ROLL = 23
RIGHT_SHOULDER_YAW = 24
RIGHT_ELBOW = 25
RIGHT_WRIST_ROLL = 26
RIGHT_WRIST_PITCH = 27
RIGHT_WRIST_YAW = 28

NUM_JOINTS = 29

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

LEG_INDICES = list(range(0, 12))
WAIST_INDICES = list(range(12, 15))
ARM_INDICES = list(range(15, 29))

# ── Pre-built configs ─────────────────────────────────────────────────────── #

def g1_conservative(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Small residual budget — safe starting point for any task.

    Use this when you want to fine-tune a G1 policy without risking gait
    destabilisation. Suitable for reward-shaping objectives (e.g. reducing
    arm swing, adjusting stepping frequency).
    """
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


def g1_energy_efficient(hidden_dims: list[int] | None = None) -> ResidualConfig:
    """Larger leg budget for energy / speed optimisation.

    Validated on SONIC base policy: -20% energy at 3 m/s, -22% at 1.7 m/s.
    Legs get a generous 0.35 rad budget to allow genuine stride reshaping;
    arms are kept tight to avoid unnecessary flailing.
    """
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("legs", LEG_INDICES, max_residual=0.35),
            JointGroupConfig("waist", WAIST_INDICES, max_residual=0.10),
            JointGroupConfig("arms", ARM_INDICES, max_residual=0.03),
        ],
        residual_hidden_dims=hidden_dims or [512, 256, 128],
        residual_activation="elu",
        init_noise_std=0.03,
    )
