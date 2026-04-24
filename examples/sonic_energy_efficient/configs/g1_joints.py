"""Unitree G1 29-DOF joint constants from GEAR-SONIC config.

Source: GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
#  Joint index layout (IsaacLab ordering, 29 DOF)
# --------------------------------------------------------------------------- #

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

# Named groups
LEG_INDICES = list(range(0, 12))       # 12 joints
WAIST_INDICES = list(range(12, 15))    # 3 joints
ARM_INDICES = list(range(15, 29))      # 14 joints

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

# --------------------------------------------------------------------------- #
#  PD gains (from MOTOR_KP / MOTOR_KD in SONIC config)
# --------------------------------------------------------------------------- #

MOTOR_KP = [
    150, 150, 150, 200, 40, 40,   # left leg
    150, 150, 150, 200, 40, 40,   # right leg
    250, 250, 250,                 # waist
    100, 100, 40, 40, 20, 20, 20, # left arm
    100, 100, 40, 40, 20, 20, 20, # right arm
]

MOTOR_KD = [
    2, 2, 2, 4, 2, 2,   # left leg
    2, 2, 2, 4, 2, 2,   # right leg
    5, 5, 5,             # waist
    5, 5, 2, 2, 2, 2, 2, # left arm
    5, 5, 2, 2, 2, 2, 2, # right arm
]

# --------------------------------------------------------------------------- #
#  Default standing pose (radians)
# --------------------------------------------------------------------------- #

DEFAULT_DOF_ANGLES = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
     0.0, 0.0, 0.0,                     # waist
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # left arm
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # right arm
]

# --------------------------------------------------------------------------- #
#  Observation layout for SONIC decoder (994D)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SonicObsLayout:
    """Describes the 994-D observation vector fed to model_decoder.onnx.

    Total: 64 + 30 + 290 + 290 + 290 + 30 = 994
    Verified against the actual released model_decoder.onnx input shape.
    The observation_config.yaml correctly says "10frame_step1" — it IS 10 frames.
    (The earlier 436D estimate of 4 frames was wrong; the deploy C++ code uses
    a trimmed observation, but the ONNX model itself expects the full 10-frame history.)
    """

    token_dim: int = 64
    ang_vel_hist_dim: int = 30      # 3 × 10 frames
    joint_pos_hist_dim: int = 290   # 29 × 10 frames
    joint_vel_hist_dim: int = 290   # 29 × 10 frames
    action_hist_dim: int = 290      # 29 × 10 frames
    gravity_hist_dim: int = 30      # 3 × 10 frames

    history_frames: int = 10

    @property
    def total_dim(self) -> int:
        return (
            self.token_dim
            + self.ang_vel_hist_dim
            + self.joint_pos_hist_dim
            + self.joint_vel_hist_dim
            + self.action_hist_dim
            + self.gravity_hist_dim
        )


OBS_LAYOUT = SonicObsLayout()
assert OBS_LAYOUT.total_dim == 994
