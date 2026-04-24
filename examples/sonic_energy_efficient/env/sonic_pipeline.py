"""Full SONIC inference pipeline: planner → encoder → token.

Transcribed from GR00T-WBC deploy C++ code.

Provides:
- SonicPlanner: wraps planner_sonic.onnx (batch=1, CPU/GPU)
- SonicEncoder: wraps model_encoder.onnx (batch=1)
- SonicPipeline: high-level API to generate tokens for velocity commands

Joint ordering conventions used by SONIC:
- planner context_mujoco_qpos[7+i]  : joint i in IsaacLab order
- planner output mujoco_qpos[f,7+i] : joint i in IsaacLab order
- encoder motion_joint_positions    : joints in MuJoCo order
- encoder motion_anchor_orientation : 6D flattened first-2-cols of rotation matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from ..configs.sonic_params import (
    ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB,
    SONIC_DEFAULT_ANGLES, MODE_WALK,
)

# --- Encoder 1762D layout offsets (from observation_config.yaml order) ---
ENCODER_OBS_DIM = 1762
OFF_ENCODER_MODE_4 = 0
LEN_ENCODER_MODE_4 = 4
OFF_MJP_10F_S5 = 4
LEN_MJP_10F_S5 = 290   # 10 × 29 joint positions
OFF_MJV_10F_S5 = 294
LEN_MJV_10F_S5 = 290   # 10 × 29 joint velocities
OFF_MRZP_10F_S5 = 584
LEN_MRZP_10F_S5 = 10   # 10 root z positions
OFF_MRZP = 594
LEN_MRZP = 1           # current root z
OFF_MAO = 595
LEN_MAO = 6            # current anchor orientation
OFF_MAO_10F_S5 = 601
LEN_MAO_10F_S5 = 60    # 10 × 6 anchor orientations
OFF_MJP_LB_10F_S5 = 661
LEN_MJP_LB_10F_S5 = 120  # 10 × 12 lower body joint positions
OFF_MJV_LB_10F_S5 = 781
LEN_MJV_LB_10F_S5 = 120
OFF_VR3P = 901
LEN_VR3P = 9
OFF_VR3P_ORN = 910
LEN_VR3P_ORN = 12
OFF_SMPL = 922
LEN_SMPL = 720
OFF_SMPL_ANCHOR = 1642
LEN_SMPL_ANCHOR = 60
OFF_MJP_WR = 1702
LEN_MJP_WR = 60

# Lower body joints (MuJoCo-indexed positions used to extract from mujoco-ordered array)
# From policy_parameters.hpp: lower_body_joint_mujoco_order_in_mujoco_index
LB_MUJOCO_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)

# Wrist joints in MuJoCo order: from wrist_joint_mujoco_order_in_mujoco_index
WRIST_MUJOCO_INDICES = np.array([19, 20, 21, 26, 27, 28], dtype=np.int64)


# ============================================================================
#  Quaternion utilities (wxyz convention)
# ============================================================================

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion (wxyz)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b (wxyz)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=a.dtype)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion wxyz → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=q.dtype)


def calc_heading_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from full quaternion (wxyz).

    Equivalent to the C++ `calc_heading_quat_d`: keeps only the yaw component.
    """
    w, x, y, z = q
    # Yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=q.dtype)


def calc_heading_quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of yaw-only quaternion."""
    return quat_conjugate(calc_heading_quat(q))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


# ============================================================================
#  Planner wrapper
# ============================================================================


class SonicPlanner:
    """Thin wrapper around planner_sonic.onnx.

    Inputs (all [1, ...]):
        context_mujoco_qpos [1, 4, 36]  — last 4 frames of state (pos/quat/joints)
        target_vel          [1]          — scalar m/s
        mode                [1]          — int64 mode ID (0..4)
        movement_direction  [1, 3]       — unit vector
        facing_direction    [1, 3]       — unit vector
        random_seed         [1]          — int64
        has_specific_target [1, 1]       — int64 flag
        specific_target_positions  [1, 4, 3]
        specific_target_headings   [1, 4]
        allowed_pred_num_tokens    [1, 11]  — int64 mask
        height              [1]          — target height in meters (−1 = default)

    Outputs:
        mujoco_qpos     [1, 64, 36]  — 64 frames @ 30 Hz
        num_pred_frames [1]           — int32 count of valid frames
    """

    def __init__(self, onnx_path: str | Path, providers: Optional[list[str]] = None) -> None:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]

    def run(
        self,
        target_vel: float,
        mode: int = MODE_WALK,
        default_height: float = 0.72,
        facing: tuple[float, float, float] = (1.0, 0.0, 0.0),
        movement: tuple[float, float, float] = (1.0, 0.0, 0.0),
        random_seed: int = 42,
        default_angles: Optional[np.ndarray] = None,
        context_qpos: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int]:
        """Run planner with reasonable defaults.

        Args:
            context_qpos: Optional (4, 36) array of 4 context frames (at 30 Hz)
                representing recent robot history.  If None, build 4 static
                frames of default pose (stationary start).

        Returns:
            (mujoco_qpos [64, 36], num_valid_frames)
            Joint columns 7..36 are in IsaacLab order.
        """
        if default_angles is None:
            default_angles = SONIC_DEFAULT_ANGLES

        ctx = np.zeros((1, 4, 36), dtype=np.float32)
        if context_qpos is not None:
            ctx[0] = context_qpos.astype(np.float32)
        else:
            # Default: 4 frames of static standing pose
            for f in range(4):
                ctx[0, f, 0] = 0.0
                ctx[0, f, 1] = 0.0
                ctx[0, f, 2] = default_height
                ctx[0, f, 3] = 1.0
                ctx[0, f, 4] = 0.0
                ctx[0, f, 5] = 0.0
                ctx[0, f, 6] = 0.0
                ctx[0, f, 7:36] = default_angles

        feeds = {
            "context_mujoco_qpos": ctx,
            "target_vel": np.array([target_vel], dtype=np.float32),
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": np.array([movement], dtype=np.float32),
            "facing_direction": np.array([facing], dtype=np.float32),
            "random_seed": np.array([random_seed], dtype=np.int64),
            "has_specific_target": np.array([[0]], dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.array(
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
            ),
            "height": np.array([-1.0], dtype=np.float32),
        }
        # Only pass inputs the model actually has
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}

        out_qpos, num_frames = self.session.run(None, feeds)
        return out_qpos[0], int(num_frames[0])


def resample_30hz_to_50hz(qpos_30hz: np.ndarray, num_valid: int) -> np.ndarray:
    """Resample planner output from 30 Hz to 50 Hz via slerp on quaternions, linear on everything else.

    Args:
        qpos_30hz: [N, 36] planner output (IsaacLab-ordered joints at cols 7..36)
        num_valid: number of valid frames

    Returns:
        qpos_50hz [M, 36] where M = floor(num_valid/30 * 50)
    """
    motion_seconds = num_valid / 30.0
    m = int(np.floor(motion_seconds * 50))
    out = np.zeros((m, 36), dtype=np.float32)

    for i in range(m):
        t = i / 50.0
        f_30 = t * 30.0
        f0 = int(np.floor(f_30))
        f1 = min(f0 + 1, num_valid - 1)
        w0 = 1.0 - (f_30 - f0)
        w1 = 1.0 - w0

        # Linear interp for pos + joints
        out[i, 0:3] = w0 * qpos_30hz[f0, 0:3] + w1 * qpos_30hz[f1, 0:3]
        out[i, 7:36] = w0 * qpos_30hz[f0, 7:36] + w1 * qpos_30hz[f1, 7:36]

        # Slerp for quaternion
        q0 = qpos_30hz[f0, 3:7].astype(np.float64)
        q1 = qpos_30hz[f1, 3:7].astype(np.float64)
        out[i, 3:7] = slerp(q0, q1, f_30 - f0).astype(np.float32)

    return out


# ============================================================================
#  Encoder input builder
# ============================================================================


def compute_joint_velocities_50hz(motion_50hz: np.ndarray) -> np.ndarray:
    """Finite-difference joint velocities at 50 Hz. Returns [M, 29] IsaacLab-ordered."""
    m = motion_50hz.shape[0]
    jvel = np.zeros((m, 29), dtype=np.float32)
    for f in range(m - 1):
        jvel[f] = (motion_50hz[f + 1, 7:36] - motion_50hz[f, 7:36]) * 50.0
    if m >= 2:
        jvel[m - 1] = jvel[m - 2]
    return jvel


def build_encoder_input(
    motion_50hz: np.ndarray,        # [M, 36] IsaacLab-ordered joints
    motion_joint_vel_50hz: np.ndarray,  # [M, 29] IsaacLab-ordered velocities
    current_frame: int,
    robot_base_quat: np.ndarray,     # [4] wxyz of robot's current orientation
    init_motion_quat: np.ndarray,    # [4] wxyz of motion frame 0 orientation
    init_heading_quat: np.ndarray,   # [4] wxyz of robot's initial heading (episode-start yaw-only)
    mode_id: int = 0,
    num_frames: int = 10,
    step: int = 5,
) -> np.ndarray:
    """Build 1762-D encoder input vector for a single env.

    Only the 4 required observations for mode g1 are filled; the rest are zero.
    """
    vec = np.zeros(ENCODER_OBS_DIM, dtype=np.float32)

    # 1) encoder_mode_4 (4D): [mode_id, 0, 0, 0]
    vec[OFF_ENCODER_MODE_4] = float(mode_id)

    # 2-3) motion_joint_positions / velocities (290D each, MUJOCO order joints × 10 frames)
    M = motion_50hz.shape[0]
    mjp = np.zeros((num_frames, 29), dtype=np.float32)
    mjv = np.zeros((num_frames, 29), dtype=np.float32)
    for fi in range(num_frames):
        target = min(current_frame + fi * step, M - 1)
        mjp[fi] = motion_50hz[target, 7:36][MUJOCO_TO_ISAACLAB]
        mjv[fi] = motion_joint_vel_50hz[target][MUJOCO_TO_ISAACLAB]
    vec[OFF_MJP_10F_S5:OFF_MJP_10F_S5 + LEN_MJP_10F_S5] = mjp.reshape(-1)
    vec[OFF_MJV_10F_S5:OFF_MJV_10F_S5 + LEN_MJV_10F_S5] = mjv.reshape(-1)

    # 7) motion_anchor_orientation_10frame_step5 (60D = 10 × 6)
    # new_ref_root_rot = apply_delta_heading * ref_data_root_rot
    # apply_delta_heading = init_heading * quat_inv(heading(init_motion_quat))
    # base_to_ref_quat = quat_conj(base_quat) * new_ref_root_rot
    data_heading_inv = calc_heading_quat_inv(init_motion_quat)
    apply_delta_heading = quat_multiply(init_heading_quat, data_heading_inv)

    anchor_frames = np.zeros((num_frames, 6), dtype=np.float32)
    for fi in range(num_frames):
        target = min(current_frame + fi * step, M - 1)
        ref_root = motion_50hz[target, 3:7].astype(np.float64)
        new_ref = quat_multiply(apply_delta_heading, ref_root)
        base_to_ref = quat_multiply(quat_conjugate(robot_base_quat), new_ref)
        R = quat_to_rotmat(base_to_ref)
        # First 2 columns, row-wise flatten
        anchor_frames[fi] = np.array(
            [R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]], dtype=np.float32
        )
    vec[OFF_MAO_10F_S5:OFF_MAO_10F_S5 + LEN_MAO_10F_S5] = anchor_frames.reshape(-1)

    return vec


# ============================================================================
#  Encoder wrapper
# ============================================================================


class SonicEncoder:
    """Wrapper for model_encoder.onnx. Batch=1 only."""

    def __init__(self, onnx_path: str | Path, providers: Optional[list[str]] = None) -> None:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

    def run(self, obs_1762: np.ndarray) -> np.ndarray:
        """Run encoder. Input can be [1762] or [1, 1762]. Returns [64]."""
        if obs_1762.ndim == 1:
            obs_1762 = obs_1762[None, :]
        assert obs_1762.shape == (1, 1762), f"Expected (1, 1762), got {obs_1762.shape}"
        token = self.session.run(None, {"obs_dict": obs_1762.astype(np.float32)})[0]
        return token[0]  # [64]


# ============================================================================
#  Convenience: end-to-end pipeline
# ============================================================================


def run_pipeline_once(
    planner: SonicPlanner,
    encoder: SonicEncoder,
    target_vel: float,
    mode: int = MODE_WALK,
    default_height: float = 0.72,
) -> dict:
    """Run planner + one encoder call. Returns dict with motion_50hz, token, metadata.

    Assumes robot starts aligned with motion (base_quat = init_motion_quat) so
    anchor_orientation is identity. Adjust per-env when integrating with training.
    """
    # 1) Plan
    qpos_30hz, num_valid = planner.run(
        target_vel=target_vel, mode=mode, default_height=default_height,
    )

    # 2) Resample to 50 Hz
    motion_50hz = resample_30hz_to_50hz(qpos_30hz, num_valid)

    # 3) Compute joint velocities
    motion_jvel = compute_joint_velocities_50hz(motion_50hz)

    # 4) Encoder input for frame 0, assuming perfect robot-motion alignment
    init_motion_quat = motion_50hz[0, 3:7].astype(np.float64)
    robot_quat = init_motion_quat.copy()  # identity relative
    init_heading = calc_heading_quat(robot_quat)

    enc_in = build_encoder_input(
        motion_50hz=motion_50hz,
        motion_joint_vel_50hz=motion_jvel,
        current_frame=0,
        robot_base_quat=robot_quat,
        init_motion_quat=init_motion_quat,
        init_heading_quat=init_heading,
    )

    # 5) Run encoder
    token = encoder.run(enc_in)

    return {
        "motion_50hz": motion_50hz,
        "motion_jvel": motion_jvel,
        "num_frames_50hz": motion_50hz.shape[0],
        "num_frames_30hz": num_valid,
        "token": token,
        "qpos_30hz": qpos_30hz,
    }
