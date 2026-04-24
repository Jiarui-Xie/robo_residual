"""Replay the IsaacLab action sequence in MuJoCo to isolate the physics gap.

Reads /tmp/align9_check.npz (the IsaacLab smoke-test rollout), plays the same
raw decoder actions into a fresh MuJoCo sim using SONIC's external PD, and
writes the MuJoCo-side joint trajectory to /tmp/mj_replay.npz.

Both pipelines feed the SAME actions through the SAME PD law. Any divergence
in joint positions / base velocity between the two rollouts is purely the
physics plant (contacts / solver / integrator).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/lumi/robo_residual")

from pathlib import Path
import numpy as np
import mujoco

from examples.sonic_energy_efficient.configs.sonic_params import (
    SONIC_GROUPED_JOINT_NAMES,
    SONIC_DEFAULT_ANGLES, SONIC_ACTION_SCALE, SONIC_KP, SONIC_KD,
    MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO,
    MUJOCO_TIMESTEP, MUJOCO_DECIMATION,
)

NPZ_IN = "/tmp/align9_check.npz"
NPZ_OUT = "/tmp/mj_replay.npz"
XML = "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

# MuJoCo actuatorfrcrange caps (same order as SONIC_GROUPED_JOINT_NAMES, first 29)
MJ_TORQUE_LIM = np.array([
    88, 88, 88, 139, 50, 50,      # left leg (pitch, roll, yaw, knee, ankle_pitch, ankle_roll)
    88, 88, 88, 139, 50, 50,      # right leg
    88, 50, 50,                   # waist yaw/roll/pitch
    25, 25, 25, 25, 25, 5, 5,     # left arm (shoulder_p/r/y, elbow, wrist r/p/y)
    25, 25, 25, 25, 25, 5, 5,     # right arm
], dtype=np.float32)


def main():
    d = np.load(NPZ_IN)
    actions_native = d["actions"]           # (T, 29) RAW decoder output in NVIDIA native order
    jpos_abs_il = d["jpos_abs"]             # (T, 29) absolute joint q, IsaacLab side
    jvel_il = d["jvel"]                     # (T, 29)
    rs_il = d["root_state_w"]               # (T, 13)
    cmd_il = d["cmd_vel"]                   # (T, 3)
    dt_control = float(d["dt"][0])          # 50 Hz control step
    T = actions_native.shape[0]
    print(f"Loaded {T} frames, dt_control={dt_control}")

    # Convert per-frame action NVIDIA-native → SONIC-grouped (MuJoCo actuator order)
    actions_grouped = actions_native[:, ISAACLAB_TO_MUJOCO]   # (T, 29)

    # --- MuJoCo setup ---
    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = MUJOCO_TIMESTEP
    data = mujoco.MjData(model)

    # Map SONIC_GROUPED joint names → qpos / qvel indices in MuJoCo
    body_qpos_idx = []   # indices into data.qpos for the 29 body joints
    body_qvel_idx = []
    for jn in SONIC_GROUPED_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"joint not found: {jn}")
        body_qpos_idx.append(model.jnt_qposadr[jid])
        body_qvel_idx.append(model.jnt_dofadr[jid])
    body_qpos_idx = np.asarray(body_qpos_idx)
    body_qvel_idx = np.asarray(body_qvel_idx)
    print(f"Total model DOFs: qpos={model.nq}  qvel={model.nv}  nu={model.nu}")
    print(f"Mapped {len(body_qpos_idx)} body joints")

    # --- Initial state from IsaacLab frame 0 ---
    init_pos_w = rs_il[0, 0:3]                 # (3,)
    init_quat_w = rs_il[0, 3:7]                # (4,) IsaacLab uses wxyz already
    init_linvel_w = rs_il[0, 7:10]
    init_angvel_w = rs_il[0, 10:13]
    init_q_native = jpos_abs_il[0]             # (29,) NVIDIA-native order
    init_q_grouped = init_q_native[ISAACLAB_TO_MUJOCO]
    init_qdot_native = jvel_il[0]
    init_qdot_grouped = init_qdot_native[ISAACLAB_TO_MUJOCO]

    # Free joint: qpos[0:3]=pos, qpos[3:7]=quat (w,x,y,z for MuJoCo), qvel[0:3]=linvel, qvel[3:6]=angvel
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = init_pos_w
    data.qpos[3:7] = init_quat_w
    data.qvel[0:3] = init_linvel_w
    data.qvel[3:6] = init_angvel_w
    for i, qidx in enumerate(body_qpos_idx):
        data.qpos[qidx] = init_q_grouped[i]
    for i, vidx in enumerate(body_qvel_idx):
        data.qvel[vidx] = init_qdot_grouped[i]
    mujoco.mj_forward(model, data)

    # --- Replay ---
    jpos_mj = np.zeros((T, 29), dtype=np.float32)
    jvel_mj = np.zeros((T, 29), dtype=np.float32)
    basepos_mj = np.zeros((T, 3), dtype=np.float32)
    basevel_w_mj = np.zeros((T, 3), dtype=np.float32)
    trq_applied = np.zeros((T, 29), dtype=np.float32)
    trq_saturated = np.zeros((T, 29), dtype=np.float32)  # 1 if clipped

    fallen_at = -1
    for t in range(T):
        # Build target for this control step
        target_q = SONIC_DEFAULT_ANGLES + actions_grouped[t] * SONIC_ACTION_SCALE
        # External PD substeps
        for _ in range(MUJOCO_DECIMATION):
            q_mj = data.qpos[body_qpos_idx]
            qdot_mj = data.qvel[body_qvel_idx]
            tau = SONIC_KP * (target_q - q_mj) - SONIC_KD * qdot_mj
            tau_clipped = np.clip(tau, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
            data.ctrl[:29] = tau_clipped
            if model.nu > 29:
                data.ctrl[29:] = 0.0
            mujoco.mj_step(model, data)

        # Post-decimation snapshot
        jpos_mj[t] = data.qpos[body_qpos_idx]
        jvel_mj[t] = data.qvel[body_qvel_idx]
        basepos_mj[t] = data.qpos[0:3]
        basevel_w_mj[t] = data.qvel[0:3]
        trq_applied[t] = tau_clipped
        trq_saturated[t] = (np.abs(tau) > MJ_TORQUE_LIM).astype(np.float32)

        if data.qpos[2] < 0.3 and fallen_at < 0:
            fallen_at = t
        if t % 200 == 0:
            print(f"[mj-replay] t={t}/{T}  pos_z={data.qpos[2]:.2f}  vx_w={data.qvel[0]:.2f}  cmd={cmd_il[t,0]:.1f}")

    if fallen_at >= 0:
        print(f"[mj-replay] NOTE: robot dropped below 0.3m at frame {fallen_at}")

    np.savez_compressed(
        NPZ_OUT,
        jpos_mj=jpos_mj,
        jvel_mj=jvel_mj,
        basepos_mj=basepos_mj,
        basevel_w_mj=basevel_w_mj,
        trq_applied=trq_applied,
        trq_saturated=trq_saturated,
        cmd_vel=cmd_il,
        dt=np.array([dt_control], dtype=np.float32),
        cmd_buckets=d["cmd_buckets"],
    )
    print(f"[mj-replay] wrote {NPZ_OUT}")


if __name__ == "__main__":
    main()
