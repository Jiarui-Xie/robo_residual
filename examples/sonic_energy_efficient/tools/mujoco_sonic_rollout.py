"""Standalone SONIC rollout in MuJoCo with interactive viewer.

Opens a MuJoCo viewer window and runs SONIC decoder + online planner
(no IsaacLab). Used to check whether periodic slowdowns are policy-level
or IsaacLab-specific.

Usage (from /home/lumi/robo_residual):
    python -u examples/sonic_energy_efficient/tools/mujoco_sonic_rollout.py --cmd-vel 2.0
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
import mujoco.viewer

from examples.sonic_energy_efficient.configs.sonic_params import (
    SONIC_GROUPED_JOINT_NAMES, SONIC_DEFAULT_ANGLES, SONIC_ACTION_SCALE, SONIC_KP, SONIC_KD,
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
    """Rotate v by the inverse (conjugate) of unit quaternion q (wxyz)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd-vel", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=9999, help="Control steps at 50 Hz")
    args = p.parse_args()

    dt_ctrl = MUJOCO_TIMESTEP * MUJOCO_DECIMATION  # 0.002 * 10 = 0.02s (50 Hz)

    # --- MuJoCo setup ---
    model = mujoco.MjModel.from_xml_path(MJ_XML)
    model.opt.timestep = MUJOCO_TIMESTEP
    data = mujoco.MjData(model)

    body_qpos_idx = []
    body_qvel_idx = []
    body_ctrl_idx = []
    for jn in SONIC_GROUPED_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"joint not found in MuJoCo XML: {jn}")
        body_qpos_idx.append(model.jnt_qposadr[jid])
        body_qvel_idx.append(model.jnt_dofadr[jid])
        act_name = jn.replace("_joint", "")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid < 0:
            raise RuntimeError(f"actuator not found: {act_name}")
        body_ctrl_idx.append(aid)
    body_qpos_idx = np.array(body_qpos_idx)
    body_qvel_idx = np.array(body_qvel_idx)
    body_ctrl_idx = np.array(body_ctrl_idx)

    # Start at SONIC default standing pose
    mujoco.mj_resetData(model, data)
    data.qpos[body_qpos_idx] = SONIC_DEFAULT_ANGLES.copy()
    data.qpos[2] = 0.72
    mujoco.mj_forward(model, data)

    # --- SONIC pipeline ---
    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=PLANNER_ONNX,
        encoder_onnx_dyn=ENCODER_ONNX,
        num_envs=1,
        target_vel=args.cmd_vel,
        device=DEVICE,
        replan_interval_steps=5,
        motion_look_ahead_steps=5,
    )
    decoder = OnnxBasePolicy(onnx_path=DECODER_ONNX, device=DEVICE)
    obs_builder = SonicObsBuilder(num_envs=1, device=DEVICE)
    obs_builder.reset()

    def get_state():
        q_g = data.qpos[body_qpos_idx].copy()
        q_n = q_g[MUJOCO_TO_ISAACLAB]
        jpos_delta = (q_n - NATIVE_DEFAULT_ANGLES).astype(np.float32)

        qd_g = data.qvel[body_qvel_idx].copy()
        qd_n = qd_g[MUJOCO_TO_ISAACLAB].astype(np.float32)

        quat_wxyz = data.qpos[3:7].copy()
        # qvel[3:6] for free joint is world-frame angular velocity → rotate to body frame
        ang_vel_world = data.qvel[3:6].copy()
        ang_vel = quat_rotate_inv(quat_wxyz, ang_vel_world).astype(np.float32)
        gravity_b = quat_rotate_inv(quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)

        return (
            torch.tensor(jpos_delta, device=DEVICE).unsqueeze(0),
            torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
            torch.tensor(ang_vel, device=DEVICE).unsqueeze(0),
            torch.tensor(gravity_b, device=DEVICE).unsqueeze(0),
            torch.tensor(quat_wxyz, dtype=torch.float32, device=DEVICE).unsqueeze(0),
        )

    # Prime history with initial state
    jpos, jvel, ang_vel, gravity, quat = get_state()
    obs_builder.update(ang_vel=ang_vel, joint_pos=jpos, joint_vel=jvel,
                       actions=torch.zeros(1, 29, device=DEVICE), gravity=gravity)

    # --- Main loop with interactive viewer ---
    pos_x_hist = []
    WINDOW = 50

    print(f"\n[mujoco_rollout] cmd={args.cmd_vel} m/s  dt={dt_ctrl:.3f}s  Press Ctrl+C to quit")
    print(f"{'t(s)':>6}  {'pos_z':>6}  {'vx(d/t)':>9}")
    print("-" * 30)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with torch.no_grad():
            for t in range(args.steps):
                if not viewer.is_running():
                    break

                jpos, jvel, ang_vel, gravity, quat = get_state()
                token = token_provider.get_tokens(
                    velocity_cmd=torch.tensor([args.cmd_vel], dtype=torch.float32, device=DEVICE),
                    robot_base_quat=quat,
                )
                obs = obs_builder.build(token)
                actions_native = decoder.forward(obs)  # (1, 29) NVIDIA native order

                # Convert to SONIC grouped for MuJoCo PD
                a_grouped = actions_native[0].cpu().numpy()[ISAACLAB_TO_MUJOCO]
                target_q = SONIC_DEFAULT_ANGLES + a_grouped * SONIC_ACTION_SCALE

                # External PD substeps
                step_start = time.monotonic()
                for _ in range(MUJOCO_DECIMATION):
                    q = data.qpos[body_qpos_idx]
                    qd = data.qvel[body_qvel_idx]
                    tau = SONIC_KP * (target_q - q) - SONIC_KD * qd
                    tau_clipped = np.clip(tau, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
                    data.ctrl[body_ctrl_idx] = tau_clipped
                    mujoco.mj_step(model, data)

                viewer.sync()
                # Real-time pacing
                elapsed = time.monotonic() - step_start
                sleep_t = dt_ctrl - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

                # Update obs history
                jpos, jvel, ang_vel, gravity, quat = get_state()
                obs_builder.update(ang_vel=ang_vel, joint_pos=jpos, joint_vel=jvel,
                                   actions=actions_native, gravity=gravity)
                token_provider.step_frame()

                pos_x_hist.append(data.qpos[0])

                if t % 50 == 0:
                    if len(pos_x_hist) >= WINDOW:
                        vx = (pos_x_hist[-1] - pos_x_hist[-WINDOW]) / ((WINDOW - 1) * dt_ctrl)
                    else:
                        vx = float("nan")
                    print(f"{t*dt_ctrl:6.1f}  {data.qpos[2]:6.3f}  {vx:+9.3f}")


if __name__ == "__main__":
    main()
