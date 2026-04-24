"""Generate reference_dataset.npz using pure MuJoCo physics instead of IsaacLab.

Runs SONIC decoder + online planner in MuJoCo, records the same fields
as generate_reference_dataset.py so the output is a drop-in replacement.

Usage (from /home/lumi/robo_residual):
    python -u examples/sonic_energy_efficient/tools/mujoco_generate_dataset.py \
        --output /tmp/reference_dataset_mujoco.npz \
        --duration 300 --cmd-buckets 1.5 2.0 2.5 3.0 --cmd-hold-seconds 8

Add --viewer to open a live MuJoCo window while recording.
"""

from __future__ import annotations
import sys
sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

import argparse
import time
import numpy as np
import torch
import mujoco

from examples.sonic_energy_efficient.configs.sonic_params import (
    SONIC_GROUPED_JOINT_NAMES, SONIC_DEFAULT_ANGLES, SONIC_ACTION_SCALE,
    SONIC_KP, SONIC_KD,
    MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO,
    MUJOCO_TIMESTEP, MUJOCO_DECIMATION,
    NATIVE_DEFAULT_ANGLES,
)
from examples.sonic_energy_efficient.env.sonic_obs_builder import SonicObsBuilder
from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder
from robo_residual.core.onnx_base import OnnxBasePolicy

MJ_XML = "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
DECODER_ONNX = "examples/sonic_energy_efficient/models/model_decoder.onnx"
ENCODER_ONNX = "examples/sonic_energy_efficient/models/model_encoder_dyn.onnx"
PLANNER_ONNX = "examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx"
DEVICE = "cuda"

MJ_TORQUE_LIM = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float32)


def quat_rotate_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v from world frame to body frame (inverse of q)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def setup_mujoco(xml_path: str, dt: float):
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    qpos_idx, qvel_idx, ctrl_idx = [], [], []
    for jn in SONIC_GROUPED_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"joint not found: {jn}")
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])
        # Find the actuator that targets this joint (by matching joint name stem)
        # Actuator name is the joint name without '_joint' suffix
        act_name = jn.replace("_joint", "")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid < 0:
            raise RuntimeError(f"actuator not found for joint: {jn} (tried '{act_name}')")
        ctrl_idx.append(aid)

    return model, data, np.array(qpos_idx), np.array(qvel_idx), np.array(ctrl_idx)


def reset_to_default(data, qpos_idx):
    mujoco.mj_resetData(data.model if hasattr(data, 'model') else None, data)
    data.qpos[qpos_idx] = SONIC_DEFAULT_ANGLES.copy()
    data.qpos[2] = 0.72
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model if hasattr(data, 'model') else None, data)


def get_state(data, qpos_idx, qvel_idx):
    """Extract state in IsaacLab NATIVE order + body-frame signals."""
    q_grouped = data.qpos[qpos_idx].copy()
    q_native = q_grouped[MUJOCO_TO_ISAACLAB]
    jpos_delta = (q_native - NATIVE_DEFAULT_ANGLES).astype(np.float32)

    qd_grouped = data.qvel[qvel_idx].copy()
    qd_native = qd_grouped[MUJOCO_TO_ISAACLAB].astype(np.float32)

    quat_wxyz = data.qpos[3:7].copy()
    # qvel[3:6] for free joint = world-frame angular velocity
    ang_vel_world = data.qvel[3:6].copy()
    ang_vel_b = quat_rotate_inv(quat_wxyz, ang_vel_world).astype(np.float32)
    gravity_b = quat_rotate_inv(quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)

    # root_state_w: [pos(3), quat(4), linvel_world(3), angvel_world(3)]
    root_state = np.concatenate([
        data.qpos[0:3],
        quat_wxyz,
        data.qvel[0:3],   # world linvel
        ang_vel_world,    # world angvel (matches IsaacLab root_state_w format)
    ]).astype(np.float32)

    return q_native.astype(np.float32), jpos_delta, qd_native, ang_vel_b, gravity_b, root_state, quat_wxyz.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/reference_dataset_mujoco.npz")
    p.add_argument("--duration", type=float, default=300.0, help="Total recording seconds")
    p.add_argument("--cmd-buckets", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    p.add_argument("--cmd-hold-seconds", type=float, default=8.0)
    p.add_argument("--warmup-seconds", type=float, default=3.0)
    p.add_argument("--viewer", action="store_true", help="Open interactive MuJoCo viewer")
    args = p.parse_args()

    dt_ctrl = MUJOCO_TIMESTEP * MUJOCO_DECIMATION
    total_frames = int(args.duration / dt_ctrl)
    warmup_frames = int(args.warmup_seconds / dt_ctrl)
    hold_frames = int(args.cmd_hold_seconds / dt_ctrl)

    print(f"[mj_dataset] {total_frames} frames ({args.duration:.0f}s), "
          f"cmd_buckets={args.cmd_buckets}, hold={args.cmd_hold_seconds}s")

    # --- Setup ---
    model, data, qpos_idx, qvel_idx, ctrl_idx = setup_mujoco(MJ_XML, MUJOCO_TIMESTEP)
    data.qpos[qpos_idx] = SONIC_DEFAULT_ANGLES.copy()
    data.qpos[2] = 0.72
    mujoco.mj_forward(model, data)

    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=PLANNER_ONNX,
        encoder_onnx_dyn=ENCODER_ONNX,
        num_envs=1,
        target_vel=args.cmd_buckets[0],
        device=DEVICE,
        replan_interval_steps=5,
        motion_look_ahead_steps=5,
    )
    decoder = OnnxBasePolicy(onnx_path=DECODER_ONNX, device=DEVICE)
    obs_builder = SonicObsBuilder(num_envs=1, device=DEVICE)
    obs_builder.reset()

    # Prime obs history
    q_n, jdelta, qd_n, av_b, grav_b, _, quat = get_state(data, qpos_idx, qvel_idx)
    obs_builder.update(
        ang_vel=torch.tensor(av_b, device=DEVICE).unsqueeze(0),
        joint_pos=torch.tensor(jdelta, device=DEVICE).unsqueeze(0),
        joint_vel=torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
        actions=torch.zeros(1, 29, device=DEVICE),
        gravity=torch.tensor(grav_b, device=DEVICE).unsqueeze(0),
    )

    # --- Recording buffers ---
    buf_jpos_abs = np.zeros((total_frames, 29), dtype=np.float32)
    buf_jpos_delta = np.zeros((total_frames, 29), dtype=np.float32)
    buf_jvel = np.zeros((total_frames, 29), dtype=np.float32)
    buf_ang_vel_b = np.zeros((total_frames, 3), dtype=np.float32)
    buf_gravity_b = np.zeros((total_frames, 3), dtype=np.float32)
    buf_actions = np.zeros((total_frames, 29), dtype=np.float32)
    buf_root_state = np.zeros((total_frames, 13), dtype=np.float32)
    buf_tokens = np.zeros((total_frames, 64), dtype=np.float32)
    buf_cmd_vel = np.zeros((total_frames, 3), dtype=np.float32)

    # --- Warmup ---
    print(f"[mj_dataset] Warming up {warmup_frames} steps ({args.warmup_seconds:.1f}s)...")
    with torch.no_grad():
        for _ in range(warmup_frames):
            q_n, jdelta, qd_n, av_b, grav_b, _, quat = get_state(data, qpos_idx, qvel_idx)
            token = token_provider.get_tokens(
                velocity_cmd=torch.tensor([args.cmd_buckets[0]], dtype=torch.float32, device=DEVICE),
                robot_base_quat=torch.tensor(quat, device=DEVICE).unsqueeze(0),
            )
            obs = obs_builder.build(token)
            actions_native = decoder.forward(obs)
            a_grouped = actions_native[0].cpu().numpy()[ISAACLAB_TO_MUJOCO]
            target_q = SONIC_DEFAULT_ANGLES + a_grouped * SONIC_ACTION_SCALE
            for _ in range(MUJOCO_DECIMATION):
                q = data.qpos[qpos_idx]
                qd = data.qvel[qvel_idx]
                tau = np.clip(SONIC_KP * (target_q - q) - SONIC_KD * qd, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
                data.ctrl[ctrl_idx] = tau
                mujoco.mj_step(model, data)
            q_n, jdelta, qd_n, av_b, grav_b, _, quat = get_state(data, qpos_idx, qvel_idx)
            obs_builder.update(
                ang_vel=torch.tensor(av_b, device=DEVICE).unsqueeze(0),
                joint_pos=torch.tensor(jdelta, device=DEVICE).unsqueeze(0),
                joint_vel=torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
                actions=actions_native,
                gravity=torch.tensor(grav_b, device=DEVICE).unsqueeze(0),
            )
            token_provider.step_frame()

    # --- Main recording loop ---
    print(f"[mj_dataset] Recording {total_frames} frames...")
    WINDOW = 50
    pos_hist = []

    viewer_ctx = None
    if args.viewer:
        import mujoco.viewer as mjv
        viewer_ctx = mjv.launch_passive(model, data)

    cmd_idx = 0
    current_cmd = args.cmd_buckets[0]
    frames_at_cmd = 0
    token_provider.set_target_vel(np.array([0]), np.array([current_cmd]))

    with torch.no_grad():
        for t in range(total_frames):
            if viewer_ctx is not None and not viewer_ctx.is_running():
                print("[mj_dataset] Viewer closed, stopping early.")
                total_frames = t
                break

            # Advance cmd bucket
            if frames_at_cmd >= hold_frames:
                cmd_idx = (cmd_idx + 1) % len(args.cmd_buckets)
                current_cmd = args.cmd_buckets[cmd_idx]
                token_provider.set_target_vel(np.array([0]), np.array([current_cmd]))
                frames_at_cmd = 0

            q_n, jdelta, qd_n, av_b, grav_b, root_st, quat = get_state(data, qpos_idx, qvel_idx)
            token = token_provider.get_tokens(
                velocity_cmd=torch.tensor([current_cmd], dtype=torch.float32, device=DEVICE),
                robot_base_quat=torch.tensor(quat, device=DEVICE).unsqueeze(0),
            )
            obs = obs_builder.build(token)
            actions_native = decoder.forward(obs)

            # Record
            buf_jpos_abs[t] = q_n
            buf_jpos_delta[t] = jdelta
            buf_jvel[t] = qd_n
            buf_ang_vel_b[t] = av_b
            buf_gravity_b[t] = grav_b
            buf_actions[t] = actions_native[0].cpu().numpy()
            buf_root_state[t] = root_st
            buf_tokens[t] = token[0].cpu().numpy()
            buf_cmd_vel[t] = [current_cmd, 0.0, 0.0]

            # Step
            a_grouped = actions_native[0].cpu().numpy()[ISAACLAB_TO_MUJOCO]
            target_q = SONIC_DEFAULT_ANGLES + a_grouped * SONIC_ACTION_SCALE
            for _ in range(MUJOCO_DECIMATION):
                q = data.qpos[qpos_idx]
                qd = data.qvel[qvel_idx]
                tau = np.clip(SONIC_KP * (target_q - q) - SONIC_KD * qd, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
                data.ctrl[ctrl_idx] = tau
                mujoco.mj_step(model, data)

            if viewer_ctx is not None:
                viewer_ctx.sync()

            q_n, jdelta, qd_n, av_b, grav_b, _, quat = get_state(data, qpos_idx, qvel_idx)
            obs_builder.update(
                ang_vel=torch.tensor(av_b, device=DEVICE).unsqueeze(0),
                joint_pos=torch.tensor(jdelta, device=DEVICE).unsqueeze(0),
                joint_vel=torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
                actions=actions_native,
                gravity=torch.tensor(grav_b, device=DEVICE).unsqueeze(0),
            )
            token_provider.step_frame()
            frames_at_cmd += 1

            # Check fall (reset if needed)
            if data.qpos[2] < 0.35:
                print(f"[mj_dataset] t={t*dt_ctrl:.1f}s: fell (z={data.qpos[2]:.2f}), resetting...")
                data.qpos[qpos_idx] = SONIC_DEFAULT_ANGLES.copy()
                data.qpos[2] = 0.72
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                obs_builder.reset()
                token_provider.reset(np.array([0]))
                token_provider.set_target_vel(np.array([0]), np.array([current_cmd]))

            pos_hist.append(data.qpos[0])
            if t % 250 == 0:
                if len(pos_hist) >= WINDOW:
                    vx = (pos_hist[-1] - pos_hist[-WINDOW]) / ((WINDOW - 1) * dt_ctrl)
                else:
                    vx = float("nan")
                print(f"  t={t*dt_ctrl:6.1f}s  cmd={current_cmd:.1f}  vx={vx:+.3f}  z={data.qpos[2]:.3f}")

    if viewer_ctx is not None:
        viewer_ctx.close()

    # Trim to actually recorded frames
    n = total_frames
    np.savez_compressed(
        args.output,
        jpos_abs=buf_jpos_abs[:n],
        jpos_delta=buf_jpos_delta[:n],
        jvel=buf_jvel[:n],
        ang_vel_b=buf_ang_vel_b[:n],
        gravity_b=buf_gravity_b[:n],
        actions=buf_actions[:n],
        root_state_w=buf_root_state[:n],
        tokens=buf_tokens[:n],
        cmd_vel=buf_cmd_vel[:n],
        default_angles=NATIVE_DEFAULT_ANGLES.astype(np.float32),
        dt=np.array([dt_ctrl], dtype=np.float32),
        cmd_buckets=np.array(args.cmd_buckets, dtype=np.float32),
        cmd_hold_seconds=np.array([args.cmd_hold_seconds], dtype=np.float32),
    )
    print(f"\n[mj_dataset] Saved {n} frames → {args.output}")


if __name__ == "__main__":
    main()
