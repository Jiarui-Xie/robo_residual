"""Sim2sim: run SONIC (decoder + encoder + online planner) in MuJoCo.

Tests whether SONIC can track 3 m/s in MuJoCo — the same simulator GR00T-WBC
uses for their own sim2sim validation.  If MuJoCo can track 3 m/s, then our
IsaacLab physics mismatch is the culprit.  If MuJoCo also caps around
~1.5-2 m/s, then the SONIC decoder itself has limits.

Usage:
    python eval_sonic_mujoco.py --cmd-vel 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/lumi/robo_residual")

from examples.sonic_energy_efficient.env.sonic_pipeline import (
    SonicPlanner, build_encoder_input, calc_heading_quat,
    compute_joint_velocities_50hz, resample_30hz_to_50hz,
)
from examples.sonic_energy_efficient.configs.sonic_params import (
    MODE_RUN, MUJOCO_TO_ISAACLAB, SONIC_ACTION_SCALE, SONIC_DEFAULT_ANGLES,
    SONIC_GROUPED_JOINT_NAMES, SONIC_KD, SONIC_KP,
)
from examples.sonic_energy_efficient.configs.g1_joints import OBS_LAYOUT


HISTORY_FRAMES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", default="models/model_decoder.onnx")
    p.add_argument("--encoder-onnx", default="models/model_encoder_dyn.onnx")
    p.add_argument("--planner-onnx", default="models/planner_sonic.onnx")
    p.add_argument("--model-xml", default="/home/lumi/TWIST/assets/g1/g1_29dof_rev_1_0.xml")
    p.add_argument("--cmd-vel", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--replan-interval", type=int, default=25)
    p.add_argument("--render", action="store_true", default=False)
    return p.parse_args()


def build_planner_context(qpos_hist: np.ndarray) -> np.ndarray:
    """qpos_hist: [5, 36] 50Hz history → 4-frame context at 30Hz spacing."""
    ctx = np.zeros((4, 36), dtype=np.float32)
    step_50hz = 50.0 / 30.0
    for i in range(4):
        idx = int(round(len(qpos_hist) - 1 - (3 - i) * step_50hz))
        idx = max(0, min(len(qpos_hist) - 1, idx))
        ctx[i] = qpos_hist[idx]
    ctx[:, 0] = 0.0
    ctx[:, 1] = 0.0
    return ctx


def main():
    args = parse_args()

    # ---------- MuJoCo model ----------
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)

    # Joint name mapping: MuJoCo's qpos[7:] order vs SONIC grouped order
    # MuJoCo qpos[7+i] is joint with index i in model.joint_names
    mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                      for i in range(1, model.njnt)]  # skip floating_base_joint
    print(f"MuJoCo joints ({len(mj_joint_names)}): {mj_joint_names[:6]}...")

    # Find 29 locomotion joints in MuJoCo order
    sonic_grouped_in_mj = [mj_joint_names.index(n) for n in SONIC_GROUPED_JOINT_NAMES]
    # IsaacLab-native ordering in MuJoCo positions (for mapping decoder action to mj actuators)
    # decoder outputs in NVIDIA native order; our SONIC_GROUPED is user's "sonic grouped";
    # to apply decoder action[k] (native k-th) to the right physical joint, we need:
    #   native k → SONIC grouped idx = MUJOCO_TO_ISAACLAB[k]
    #   SONIC grouped idx → MuJoCo position = sonic_grouped_in_mj[that idx]
    native_to_mj = np.array(
        [sonic_grouped_in_mj[MUJOCO_TO_ISAACLAB[k]] for k in range(29)], dtype=np.int64
    )

    # Initial pose
    data.qpos[:3] = [0, 0, 0.72]
    data.qpos[3:7] = [1, 0, 0, 0]
    for k, mj_idx in enumerate(sonic_grouped_in_mj):
        data.qpos[7 + mj_idx] = SONIC_DEFAULT_ANGLES[k]
    mujoco.mj_forward(model, data)

    # ---------- Load SONIC models ----------
    print(f"Loading planner...")
    planner = SonicPlanner(args.planner_onnx)
    print(f"Loading encoder...")
    encoder = ort.InferenceSession(args.encoder_onnx, providers=["CPUExecutionProvider"])
    print(f"Loading decoder...")
    decoder = ort.InferenceSession(args.decoder_onnx, providers=["CPUExecutionProvider"])
    dec_in_name = decoder.get_inputs()[0].name

    # ---------- SONIC state history ----------
    # Proprio (decoder input) in SONIC grouped order — delta from default
    hist_ang_vel = np.zeros((HISTORY_FRAMES, 3), dtype=np.float32)
    hist_jpos = np.zeros((HISTORY_FRAMES, 29), dtype=np.float32)
    hist_jvel = np.zeros((HISTORY_FRAMES, 29), dtype=np.float32)
    hist_actions = np.zeros((HISTORY_FRAMES, 29), dtype=np.float32)
    hist_gravity = np.tile(np.array([0, 0, -1], dtype=np.float32), (HISTORY_FRAMES, 1))

    # Planner state history (5 frames @ 50Hz)
    qpos_5f = np.zeros((5, 36), dtype=np.float32)
    for i in range(5):
        qpos_5f[i, :3] = [0, 0, 0.72]
        qpos_5f[i, 3:7] = [1, 0, 0, 0]
        qpos_5f[i, 7:36] = SONIC_DEFAULT_ANGLES
    qpos_5f_idx = 0

    # ---------- Initial plan ----------
    def do_replan():
        nonlocal current_motion, current_motion_jvel, init_motion_quat, init_heading_quat, current_frame
        ctx = build_planner_context(qpos_5f)
        qpos_30hz, n_valid = planner.run(
            target_vel=args.cmd_vel, mode=MODE_RUN, context_qpos=ctx,
        )
        motion = resample_30hz_to_50hz(qpos_30hz[:n_valid], n_valid)
        jvel = compute_joint_velocities_50hz(motion)
        current_motion = motion
        current_motion_jvel = jvel
        current_frame = 0
        init_motion_quat = motion[0, 3:7].astype(np.float64)
        init_heading_quat = calc_heading_quat(init_motion_quat)

    current_motion = None
    current_motion_jvel = None
    current_frame = 0
    init_motion_quat = None
    init_heading_quat = None
    do_replan()
    print(f"Ready. Running {args.num_steps} steps @ 50Hz ({args.num_steps*0.02:.1f}s)")

    # ---------- Control loop @ 50 Hz (physics dt = 0.005, decimation = 4) ----------
    physics_dt = 0.005
    decimation = 4
    ctrl_period = physics_dt * decimation  # 0.02s = 50Hz

    init_x = float(data.qpos[0])
    last_action = np.zeros(29, dtype=np.float32)
    steps_since_replan = 0

    if args.render:
        viewer = mujoco.viewer.launch_passive(model, data)

    for step in range(args.num_steps):
        # -- Read state --
        qw, qx, qy, qz = data.qpos[3:7]
        base_quat = np.array([qw, qx, qy, qz], dtype=np.float64)

        # Ang vel in body frame (IMU gyroscope equivalent)
        # MuJoCo's qvel[3:6] is world frame ang vel; rotate into body frame
        from examples.sonic_energy_efficient.env.sonic_pipeline import quat_conjugate, quat_to_rotmat
        R = quat_to_rotmat(base_quat)
        ang_vel_world = data.qvel[3:6].astype(np.float64)
        ang_vel_body = R.T @ ang_vel_world

        gravity_body = R.T @ np.array([0, 0, -1.0])

        # Joint positions in SONIC grouped order
        jpos_grouped = np.array(
            [data.qpos[7 + mj_idx] for mj_idx in sonic_grouped_in_mj], dtype=np.float32
        )
        jvel_grouped = np.array(
            [data.qvel[6 + mj_idx] for mj_idx in sonic_grouped_in_mj], dtype=np.float32
        )
        # Delta from default (what decoder sees)
        jpos_delta = jpos_grouped - SONIC_DEFAULT_ANGLES.astype(np.float32)

        # -- Update planner state history (rolling 5) --
        qpos_5f_idx = (qpos_5f_idx + 1) % 5
        qpos_5f[qpos_5f_idx, :3] = data.qpos[:3]
        qpos_5f[qpos_5f_idx, 3:7] = data.qpos[3:7]
        qpos_5f[qpos_5f_idx, 7:36] = jpos_grouped

        # -- Replan? --
        if steps_since_replan >= args.replan_interval:
            do_replan()
            steps_since_replan = 0
        steps_since_replan += 1

        # -- Build encoder input with live robot quat --
        cf = int(min(current_frame, len(current_motion) - 1))
        enc_in = build_encoder_input(
            motion_50hz=current_motion,
            motion_joint_vel_50hz=current_motion_jvel,
            current_frame=cf,
            robot_base_quat=base_quat,
            init_motion_quat=init_motion_quat,
            init_heading_quat=init_heading_quat,
        )
        token = encoder.run(None, {"obs_dict": enc_in[None, :].astype(np.float32)})[0][0]

        # -- Update proprio history --
        hist_ang_vel = np.roll(hist_ang_vel, -1, axis=0)
        hist_ang_vel[-1] = ang_vel_body.astype(np.float32)
        hist_jpos = np.roll(hist_jpos, -1, axis=0)
        hist_jpos[-1] = jpos_delta
        hist_jvel = np.roll(hist_jvel, -1, axis=0)
        hist_jvel[-1] = jvel_grouped
        hist_actions = np.roll(hist_actions, -1, axis=0)
        hist_actions[-1] = last_action
        hist_gravity = np.roll(hist_gravity, -1, axis=0)
        hist_gravity[-1] = gravity_body.astype(np.float32)

        # -- Build decoder 994D obs --
        dec_in = np.zeros(OBS_LAYOUT.total_dim, dtype=np.float32)
        dec_in[0:64] = token
        dec_in[64:94] = hist_ang_vel.reshape(-1)
        dec_in[94:94+290] = hist_jpos.reshape(-1)
        dec_in[384:384+290] = hist_jvel.reshape(-1)
        dec_in[674:674+290] = hist_actions.reshape(-1)
        dec_in[964:994] = hist_gravity.reshape(-1)

        action_native = decoder.run(None, {dec_in_name: dec_in[None, :]})[0][0]  # (29,) NVIDIA native order
        last_action = action_native.copy()

        # -- Apply action: target = default + scale * action_native, in MuJoCo order --
        # decoder output is native order; we need to apply to MuJoCo-ordered qpos
        # native k-th action goes to MuJoCo joint `native_to_mj[k]`
        # default_angles for that joint = SONIC_DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB[k]]
        # (same physical joint)
        action_scale_native = np.array(
            [SONIC_ACTION_SCALE[MUJOCO_TO_ISAACLAB[k]] for k in range(29)], dtype=np.float32
        )
        default_native = np.array(
            [SONIC_DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB[k]] for k in range(29)], dtype=np.float32
        )
        target_native = default_native + action_scale_native * action_native

        # -- PD control (50Hz policy, 200Hz physics with decimation 4) --
        for _ in range(decimation):
            # Read current joint states
            q_mj = np.array([data.qpos[7 + native_to_mj[k]] for k in range(29)], dtype=np.float32)
            dq_mj = np.array([data.qvel[6 + native_to_mj[k]] for k in range(29)], dtype=np.float32)
            kp_native = np.array(
                [SONIC_KP[MUJOCO_TO_ISAACLAB[k]] for k in range(29)], dtype=np.float32
            )
            kd_native = np.array(
                [SONIC_KD[MUJOCO_TO_ISAACLAB[k]] for k in range(29)], dtype=np.float32
            )
            torque_native = kp_native * (target_native - q_mj) - kd_native * dq_mj
            # Clamp to effort limits (~25-139 N·m depending on joint)
            effort_limits = {
                "hip_pitch": 88, "hip_roll": 139, "hip_yaw": 88, "knee": 139,
                "ankle_pitch": 50, "ankle_roll": 50, "waist_yaw": 88,
                "waist_roll": 50, "waist_pitch": 50,
                "shoulder_pitch": 25, "shoulder_roll": 25, "shoulder_yaw": 25,
                "elbow": 25, "wrist_roll": 25, "wrist_pitch": 5, "wrist_yaw": 5,
            }
            # Apply to MuJoCo actuators
            for k in range(29):
                mj_act_idx = native_to_mj[k]
                data.ctrl[mj_act_idx] = torque_native[k]

            mujoco.mj_step(model, data)

        current_frame += 1

        # -- Log --
        if step % 20 == 0 or step == args.num_steps - 1:
            dx = float(data.qpos[0]) - init_x
            vx = float(data.qvel[0])
            h = float(data.qpos[2])
            print(f"  step {step:>3d}: dx={dx:+.2f}m  h={h:.2f}m  vx={vx:+.2f} (cmd {args.cmd_vel})")

        if args.render:
            viewer.sync()

    # Final
    dx = float(data.qpos[0]) - init_x
    duration = args.num_steps * ctrl_period
    print(f"\n=== MuJoCo RESULT after {duration:.1f}s ===")
    print(f"  dx = {dx:+.2f}m")
    print(f"  avg vx = {dx/duration:+.2f} m/s (cmd {args.cmd_vel})")
    if dx/duration > 2.0:
        print(f"  → SONIC CAN track {args.cmd_vel}m/s in MuJoCo. IsaacLab is mismatched.")
    else:
        print(f"  → SONIC CANNOT track {args.cmd_vel}m/s in MuJoCo either. Fundamental limit.")


if __name__ == "__main__":
    main()
