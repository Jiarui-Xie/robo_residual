"""Per-step physics comparison: IsaacLab vs MuJoCo under identical state+action.

For each of the first ~200 frames of /tmp/align9_check.npz:
 1. Hard-set MuJoCo's full state to IsaacLab's pre-step state for that frame.
 2. Apply one decimation of external PD with the IsaacLab-recorded action.
 3. Compare the resulting joint positions / velocities / base velocity with
    IsaacLab's NEXT-frame recorded state.

Any systematic delta here is pure physics divergence (solver, integrator,
contact model). No policy drift because state is re-synced every step.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/lumi/robo_residual")

import numpy as np
import mujoco

from examples.sonic_energy_efficient.configs.sonic_params import (
    SONIC_GROUPED_JOINT_NAMES,
    SONIC_DEFAULT_ANGLES, SONIC_ACTION_SCALE, SONIC_KP, SONIC_KD,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TIMESTEP, MUJOCO_DECIMATION,
)

NPZ_IN = "/tmp/align9_check.npz"
XML = "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

MJ_TORQUE_LIM = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float32)


def main():
    d = np.load(NPZ_IN)
    actions_native = d["actions"]
    jpos_il = d["jpos_abs"]
    jvel_il = d["jvel"]
    rs_il = d["root_state_w"]
    cmd_il = d["cmd_vel"]

    actions_grouped = actions_native[:, ISAACLAB_TO_MUJOCO]
    jpos_grouped_il = jpos_il[:, ISAACLAB_TO_MUJOCO]
    jvel_grouped_il = jvel_il[:, ISAACLAB_TO_MUJOCO]

    # Pick the cmd=3.0 window (frames 1200-1599)
    t0 = 1200
    t1 = 1500

    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = MUJOCO_TIMESTEP
    data = mujoco.MjData(model)

    body_qpos_idx = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        for jn in SONIC_GROUPED_JOINT_NAMES
    ])
    body_qvel_idx = np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        for jn in SONIC_GROUPED_JOINT_NAMES
    ])

    N = t1 - t0
    dq_step = np.zeros((N, 29), dtype=np.float32)     # (post_mj - pre_il) joint q
    dqdot_step = np.zeros((N, 29), dtype=np.float32)
    dlinvel_step = np.zeros((N, 3), dtype=np.float32) # world-frame base linvel delta
    mj_linvel_post = np.zeros((N, 3), dtype=np.float32)
    il_linvel_post = np.zeros((N, 3), dtype=np.float32)
    mj_q_post = np.zeros((N, 29), dtype=np.float32)
    il_q_post = np.zeros((N, 29), dtype=np.float32)
    trq_sat_cnt = np.zeros(29, dtype=np.float32)
    trq_applied_log = np.zeros((N, 29), dtype=np.float32)

    for k in range(N):
        t = t0 + k
        # --- Hard-set MuJoCo state to IsaacLab's pre-step state (frame t) ---
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = rs_il[t, 0:3]
        data.qpos[3:7] = rs_il[t, 3:7]      # wxyz
        data.qvel[0:3] = rs_il[t, 7:10]     # world linvel
        data.qvel[3:6] = rs_il[t, 10:13]    # world angvel
        for i, qidx in enumerate(body_qpos_idx):
            data.qpos[qidx] = jpos_grouped_il[t, i]
        for i, vidx in enumerate(body_qvel_idx):
            data.qvel[vidx] = jvel_grouped_il[t, i]
        mujoco.mj_forward(model, data)

        # --- Apply one decimation with IsaacLab's action for frame t ---
        target_q = SONIC_DEFAULT_ANGLES + actions_grouped[t] * SONIC_ACTION_SCALE
        for _ in range(MUJOCO_DECIMATION):
            q_mj = data.qpos[body_qpos_idx]
            qdot_mj = data.qvel[body_qvel_idx]
            tau = SONIC_KP * (target_q - q_mj) - SONIC_KD * qdot_mj
            tau_clipped = np.clip(tau, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
            trq_sat_cnt += (np.abs(tau) > MJ_TORQUE_LIM).astype(np.float32)
            data.ctrl[:29] = tau_clipped
            if model.nu > 29:
                data.ctrl[29:] = 0.0
            mujoco.mj_step(model, data)

        mj_q_post[k] = data.qpos[body_qpos_idx]
        mj_linvel_post[k] = data.qvel[0:3]
        # IsaacLab's t+1 state is the "true" next in its own plant
        if t + 1 < len(jpos_grouped_il):
            il_q_post[k] = jpos_grouped_il[t + 1]
            il_linvel_post[k] = rs_il[t + 1, 7:10]
        else:
            il_q_post[k] = mj_q_post[k]
            il_linvel_post[k] = mj_linvel_post[k]

        dq_step[k] = mj_q_post[k] - il_q_post[k]
        dqdot_step[k] = data.qvel[body_qvel_idx] - jvel_grouped_il[t+1] if t+1 < len(jvel_grouped_il) else 0
        dlinvel_step[k] = mj_linvel_post[k] - il_linvel_post[k]
        trq_applied_log[k] = tau_clipped

    print(f"\n=== Per-step physics divergence at cmd=3.0 (frames {t0}..{t1}) ===")
    print(f"N steps: {N}")
    print(f"\nBase world-frame linear-velocity delta (MJ − IL) after one control step:")
    print(f"  mean Δvx = {dlinvel_step[:, 0].mean():+.4f} m/s     median = {np.median(dlinvel_step[:, 0]):+.4f}")
    print(f"  mean Δvy = {dlinvel_step[:, 1].mean():+.4f} m/s")
    print(f"  mean Δvz = {dlinvel_step[:, 2].mean():+.4f} m/s")
    print(f"\nIsaacLab vx (world, at cmd=3.0): mean {il_linvel_post[:, 0].mean():.2f}  p90 {np.percentile(il_linvel_post[:, 0], 90):.2f}")
    print(f"MuJoCo  vx (world, one-step post-IL state): mean {mj_linvel_post[:, 0].mean():.2f}  p90 {np.percentile(mj_linvel_post[:, 0], 90):.2f}")

    print(f"\nPer-joint q delta (|MJ_post - IL_post|), rad, worst offenders:")
    print(f"{'joint':<32} {'mean|Δq|':>10} {'p95|Δq|':>10} {'max|Δq|':>10}")
    order = np.argsort(-np.abs(dq_step).mean(axis=0))
    for j in order[:12]:
        n = SONIC_GROUPED_JOINT_NAMES[j]
        ad = np.abs(dq_step[:, j])
        print(f"{n:<32} {ad.mean():>10.5f} {np.percentile(ad, 95):>10.5f} {ad.max():>10.5f}")

    print(f"\nMuJoCo-side torque saturation (frames where tau_requested > MJ_LIM):")
    for j in range(29):
        frac = trq_sat_cnt[j] / (N * MUJOCO_DECIMATION)
        if frac > 0.01:
            print(f"  {SONIC_GROUPED_JOINT_NAMES[j]:<32} sat in {frac:.1%} of substeps")

    np.savez_compressed(
        "/tmp/perstep_compare.npz",
        mj_q_post=mj_q_post, il_q_post=il_q_post,
        mj_linvel_post=mj_linvel_post, il_linvel_post=il_linvel_post,
        dq_step=dq_step, dlinvel_step=dlinvel_step,
        trq_applied=trq_applied_log, trq_saturated_cnt=trq_sat_cnt,
        t0=np.array([t0]), t1=np.array([t1]),
    )
    print(f"\nSaved /tmp/perstep_compare.npz")


if __name__ == "__main__":
    main()
