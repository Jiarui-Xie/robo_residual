"""Exact SONIC deployment parameters (transcribed from policy_parameters.hpp).

Source: GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp

Two joint orderings coexist in SONIC:
- MUJOCO order: URDF kinematic tree (used by sim, reference motions)
- ISAACLAB order: policy I/O, robot SDK

Both are 29-DOF.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# --- Motor armature constants ---
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

# Computed stiffness: armature × ω²
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

# Computed damping: 2 × ζ × armature × ω
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# Effort limits
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0


# --- Joint mapping arrays (ISAACLAB ↔ MUJOCO) ---
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)

MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)


# --- Default standing angles (ISAACLAB order) ---
SONIC_DEFAULT_ANGLES = np.array([
    -0.312,  # 0  left_hip_pitch
     0.0,    # 1  left_hip_roll
     0.0,    # 2  left_hip_yaw
     0.669,  # 3  left_knee
    -0.363,  # 4  left_ankle_pitch
     0.0,    # 5  left_ankle_roll
    -0.312,  # 6  right_hip_pitch
     0.0,    # 7  right_hip_roll
     0.0,    # 8  right_hip_yaw
     0.669,  # 9  right_knee
    -0.363,  # 10 right_ankle_pitch
     0.0,    # 11 right_ankle_roll
     0.0,    # 12 waist_yaw
     0.0,    # 13 waist_roll
     0.0,    # 14 waist_pitch
     0.2,    # 15 left_shoulder_pitch
     0.2,    # 16 left_shoulder_roll
     0.0,    # 17 left_shoulder_yaw
     0.6,    # 18 left_elbow
     0.0,    # 19 left_wrist_roll
     0.0,    # 20 left_wrist_pitch
     0.0,    # 21 left_wrist_yaw
     0.2,    # 22 right_shoulder_pitch
    -0.2,    # 23 right_shoulder_roll
     0.0,    # 24 right_shoulder_yaw
     0.6,    # 25 right_elbow
     0.0,    # 26 right_wrist_roll
     0.0,    # 27 right_wrist_pitch
     0.0,    # 28 right_wrist_yaw
], dtype=np.float32)


# --- Per-joint action scale (ISAACLAB order) ---
# Computed as 0.25 × effort_limit / stiffness per joint
def _scale(effort, stiff):
    return 0.25 * effort / stiff


SONIC_ACTION_SCALE = np.array([
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 0  left_hip_pitch
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 1  left_hip_roll
    _scale(EFFORT_LIMIT_7520_14, STIFFNESS_7520_14),  # 2  left_hip_yaw
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 3  left_knee
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 4  left_ankle_pitch
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 5  left_ankle_roll
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 6  right_hip_pitch
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 7  right_hip_roll
    _scale(EFFORT_LIMIT_7520_14, STIFFNESS_7520_14),  # 8  right_hip_yaw
    _scale(EFFORT_LIMIT_7520_22, STIFFNESS_7520_22),  # 9  right_knee
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 10 right_ankle_pitch
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 11 right_ankle_roll
    _scale(EFFORT_LIMIT_7520_14, STIFFNESS_7520_14),  # 12 waist_yaw
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 13 waist_roll
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 14 waist_pitch
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 15 left_shoulder_pitch
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 16 left_shoulder_roll
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 17 left_shoulder_yaw
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 18 left_elbow
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 19 left_wrist_roll
    _scale(EFFORT_LIMIT_4010, STIFFNESS_4010),        # 20 left_wrist_pitch
    _scale(EFFORT_LIMIT_4010, STIFFNESS_4010),        # 21 left_wrist_yaw
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 22 right_shoulder_pitch
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 23 right_shoulder_roll
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 24 right_shoulder_yaw
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 25 right_elbow
    _scale(EFFORT_LIMIT_5020, STIFFNESS_5020),        # 26 right_wrist_roll
    _scale(EFFORT_LIMIT_4010, STIFFNESS_4010),        # 27 right_wrist_pitch
    _scale(EFFORT_LIMIT_4010, STIFFNESS_4010),        # 28 right_wrist_yaw
], dtype=np.float32)


# --- Per-joint PD gains (SONIC GROUPED / MUJOCO order) ---
# Source: gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
# These are the real deployment KP/KD, computed from motor armature constants.
# The GR00T MuJoCo sim also uses these exact values via DDS (motor_cmd[i].kp = kps[i]).
SONIC_KP = np.array([
    STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
    STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
    STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
    STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
    STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
], dtype=np.float32)

SONIC_KD = np.array([
    DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
    DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
    DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
    DAMPING_5020, DAMPING_4010, DAMPING_4010,
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
    DAMPING_5020, DAMPING_4010, DAMPING_4010,
], dtype=np.float32)


# SONIC training MuJoCo XML joint-level physics (from
# gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml <default>):
#   <joint damping="0.05" armature="0.01" frictionloss="0.2"/>
# Applied uniformly to all 29 locomotion joints. These map onto IsaacLab
# actuator config fields: armature, viscous_friction, friction (static).
MUJOCO_JOINT_ARMATURE = 0.01
MUJOCO_JOINT_DAMPING = 0.05        # -> IsaacLab viscous_friction
MUJOCO_JOINT_FRICTIONLOSS = 0.2    # -> IsaacLab friction (static Coulomb)

# SONIC training ground friction from gear_sonic/.../scene_43dof.xml:
#   <default><geom friction="1.0"/></default>
# (Deploy-side scene_29dof.xml uses 0.5; we match TRAINING, not deploy.)
MUJOCO_GROUND_FRICTION = 1.0

# Match GR00T DDS sim: SIMULATE_DT=0.005, control_decimation=4 → 50 Hz control
MUJOCO_TIMESTEP = 0.005
MUJOCO_DECIMATION = 4


# ---------------------------------------------------------------------------
#  NVIDIA IsaacLab native G1 joint ordering (interleaved left/right)
#
#  SONIC's ONNX decoder outputs actions indexed in THIS order (not the SONIC
#  body-part-grouped order above).  Evidence: deploy code does
#  `floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i]` where i is SONIC
#  grouped and `isaaclab_to_mujoco[i]` gives the NVIDIA native position.
#
#  NATIVE_* arrays reorder SONIC_* arrays via MUJOCO_TO_ISAACLAB so they can
#  be passed directly to IsaacLab without further permutation.
# ---------------------------------------------------------------------------

# SONIC joint names in body-part-grouped order (matches SONIC_DEFAULT_ANGLES etc.)
SONIC_GROUPED_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# NVIDIA IsaacLab native G1 joint names (derived via MUJOCO_TO_ISAACLAB)
NATIVE_LOCOMOTION_JOINT_NAMES = [
    SONIC_GROUPED_JOINT_NAMES[MUJOCO_TO_ISAACLAB[j]] for j in range(29)
]

# Reordered parameter arrays (NVIDIA native order, usable directly with IsaacLab)
NATIVE_DEFAULT_ANGLES = SONIC_DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
NATIVE_ACTION_SCALE = SONIC_ACTION_SCALE[MUJOCO_TO_ISAACLAB]
NATIVE_KP = SONIC_KP[MUJOCO_TO_ISAACLAB]
NATIVE_KD = SONIC_KD[MUJOCO_TO_ISAACLAB]


# --- Locomotion modes ---
MODE_IDLE = 0
MODE_SLOW_WALK = 1
MODE_WALK = 2
MODE_RUN = 3
MODE_BOXING = 4


# --- Utility functions ---

def to_tensor(arr: np.ndarray, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, device=device, dtype=dtype)
