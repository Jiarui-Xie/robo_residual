"""Per-joint mechanical power comparison: baseline SONIC vs SONIC+Residual.

Runs both rollouts in MuJoCo and plots mean |τ × ω| per joint,
highlighting hottest joints and showing which ones the residual reduced.

Usage (from /home/lumi/robo_residual):
    python -u examples/sonic_energy_efficient/tools/mujoco_joint_power_plot.py \\
        --fused-decoder runs/sonic_energy_3p0_isaaclab_v3/fused_best_speed.onnx \\
        --cmd-vel 3.0 \\
        --output-dir runs/sonic_energy_3p0_isaaclab_v3/joint_power_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

import mujoco
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from examples.sonic_energy_efficient.configs.sonic_params import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_DECIMATION,
    MUJOCO_TIMESTEP,
    MUJOCO_TO_ISAACLAB,
    NATIVE_DEFAULT_ANGLES,
    SONIC_ACTION_SCALE,
    SONIC_DEFAULT_ANGLES,
    SONIC_GROUPED_JOINT_NAMES,
    SONIC_KD,
    SONIC_KP,
)
from examples.sonic_energy_efficient.env.sonic_obs_builder import SonicObsBuilder
from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder
from robo_residual.core.onnx_base import OnnxBasePolicy

MJ_XML       = "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
ENCODER_ONNX = "examples/sonic_energy_efficient/models/model_encoder_dyn.onnx"
PLANNER_ONNX = "examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx"
BASELINE_DECODER = "examples/sonic_energy_efficient/models/model_decoder.onnx"

MJ_TORQUE_LIM = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float32)

DEVICE = "cuda"

# Joint group membership for colouring (indices into SONIC_GROUPED_JOINT_NAMES)
# G1 29-DOF: left_leg(0-5), right_leg(6-11), waist(12-14), left_arm(15-21), right_arm(22-28)
JOINT_GROUPS = {
    "left_leg":  list(range(0, 6)),
    "right_leg": list(range(6, 12)),
    "waist":     list(range(12, 15)),
    "left_arm":  list(range(15, 22)),
    "right_arm": list(range(22, 29)),
}
GROUP_COLORS = {
    "left_leg":  "#4C72B0",
    "right_leg": "#DD8452",
    "waist":     "#55A868",
    "left_arm":  "#C44E52",
    "right_arm": "#8172B2",
}


def quat_rotate_inv(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def build_mujoco_env():
    model = mujoco.MjModel.from_xml_path(MJ_XML)
    model.opt.timestep = MUJOCO_TIMESTEP
    data = mujoco.MjData(model)
    qpos_idx, qvel_idx, ctrl_idx = [], [], []
    for jn in SONIC_GROUPED_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn.replace("_joint", ""))
        ctrl_idx.append(aid)
    idx = (np.array(qpos_idx), np.array(qvel_idx), np.array(ctrl_idx))
    mujoco.mj_resetData(model, data)
    data.qpos[idx[0]] = SONIC_DEFAULT_ANGLES.copy()
    data.qpos[2] = 0.72
    mujoco.mj_forward(model, data)
    return model, data, idx


def get_state(data, idx):
    qpos_idx, qvel_idx, _ = idx
    q_n  = data.qpos[qpos_idx][MUJOCO_TO_ISAACLAB]
    jpos = (q_n - NATIVE_DEFAULT_ANGLES).astype(np.float32)
    qd_n = data.qvel[qvel_idx][MUJOCO_TO_ISAACLAB].astype(np.float32)
    quat = data.qpos[3:7].copy()
    ang_vel = quat_rotate_inv(quat, data.qvel[3:6].copy()).astype(np.float32)
    grav    = quat_rotate_inv(quat, np.array([0., 0., -1.])).astype(np.float32)
    return (
        torch.tensor(jpos, device=DEVICE).unsqueeze(0),
        torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
        torch.tensor(ang_vel, device=DEVICE).unsqueeze(0),
        torch.tensor(grav, device=DEVICE).unsqueeze(0),
        torch.tensor(quat, dtype=torch.float32, device=DEVICE).unsqueeze(0),
    )


def run_rollout_per_joint(decoder_onnx, cmd_vel, warmup_steps, record_steps, label):
    """Run rollout and return per-joint mean |tau * qdot| array (29,)."""
    model, data, idx = build_mujoco_env()
    _, qvel_idx, ctrl_idx = idx
    dt_ctrl = MUJOCO_TIMESTEP * MUJOCO_DECIMATION

    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=PLANNER_ONNX,
        encoder_onnx_dyn=ENCODER_ONNX,
        num_envs=1,
        target_vel=cmd_vel,
        replan_interval_steps=5,
        motion_look_ahead_steps=5,
        device=DEVICE,
    )
    token_provider.set_target_vel(np.array([0]), np.array([cmd_vel]))

    decoder = OnnxBasePolicy(decoder_onnx, device=DEVICE)
    obs_builder = SonicObsBuilder(num_envs=1, device=DEVICE)
    obs_builder.reset()

    jpos, jvel, ang_vel, grav, quat = get_state(data, idx)
    token = token_provider.get_tokens(
        velocity_cmd=torch.tensor([cmd_vel], dtype=torch.float32, device=DEVICE),
        robot_base_quat=quat,
    )
    obs = obs_builder.build(token)

    per_joint_power = []  # list of (29,) arrays
    pos_x_hist = []
    total_steps = warmup_steps + record_steps

    for t in range(total_steps):
        with torch.no_grad():
            actions_native = decoder.forward(obs)

        a_grouped = actions_native[0].cpu().numpy()[ISAACLAB_TO_MUJOCO]
        target_q = SONIC_DEFAULT_ANGLES + a_grouped * SONIC_ACTION_SCALE

        tau_buf = []
        for _ in range(MUJOCO_DECIMATION):
            q  = data.qpos[idx[0]]
            qd = data.qvel[qvel_idx]
            tau = SONIC_KP * (target_q - q) - SONIC_KD * qd
            tau = np.clip(tau, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
            data.ctrl[ctrl_idx] = tau
            mujoco.mj_step(model, data)
            tau_buf.append(tau.copy())

        jpos, jvel, ang_vel, grav, quat = get_state(data, idx)
        obs_builder.update(ang_vel=ang_vel, joint_pos=jpos, joint_vel=jvel,
                           actions=actions_native, gravity=grav)
        token_provider.step_frame()
        pos_x_hist.append(data.qpos[0])

        if t >= warmup_steps:
            tau_mean = np.mean(tau_buf, axis=0)          # (29,) in SONIC grouped order
            qd_grouped = data.qvel[qvel_idx].copy()       # (29,) in SONIC grouped order
            joint_power = np.abs(tau_mean * qd_grouped)   # (29,) per-joint |tau*qdot|
            per_joint_power.append(joint_power)

        token = token_provider.get_tokens(
            velocity_cmd=torch.tensor([cmd_vel], dtype=torch.float32, device=DEVICE),
            robot_base_quat=quat,
        )
        obs = obs_builder.build(token)

        if t % 50 == 0:
            h = data.qpos[2]
            if len(pos_x_hist) >= 50:
                vx = (pos_x_hist[-1] - pos_x_hist[-50]) / (49 * dt_ctrl)
            else:
                vx = float("nan")
            phase = "warmup" if t < warmup_steps else "record"
            print(f"  [{label}][{phase} t={t*dt_ctrl:5.1f}s] h={h:.3f}  vx={vx:+.3f}")

    arr = np.array(per_joint_power)   # (record_steps, 29)
    mean_power = arr.mean(axis=0)     # (29,)
    total_mean = mean_power.sum()
    print(f"  [{label}] total_mean_power={total_mean:.1f}W")
    return mean_power


def plot_joint_power(baseline_power, residual_power, output_dir, cmd_vel):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = [n.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
             for n in SONIC_GROUPED_JOINT_NAMES]
    n = len(names)
    idx = np.arange(n)

    # Per-joint colour by group
    colors = []
    for i in range(n):
        for g, members in JOINT_GROUPS.items():
            if i in members:
                colors.append(GROUP_COLORS[g])
                break

    delta = residual_power - baseline_power          # negative = improvement
    delta_pct = delta / (baseline_power + 1e-6) * 100

    # Sort by baseline power (hottest first)
    order = np.argsort(baseline_power)[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f"Per-Joint Power Analysis  |  cmd={cmd_vel}m/s  |  SONIC baseline vs Residual",
                 fontsize=14, fontweight="bold")

    # ---- Top: absolute power comparison ----
    ax = axes[0]
    w = 0.38
    x = np.arange(n)
    b_sorted = baseline_power[order]
    r_sorted = residual_power[order]
    c_sorted = [colors[i] for i in order]
    n_sorted = [names[i] for i in order]

    bars_b = ax.bar(x - w/2, b_sorted, width=w, color=c_sorted, alpha=0.5, label="Baseline")
    bars_r = ax.bar(x + w/2, r_sorted, width=w, color=c_sorted, alpha=0.9, label="Residual")

    ax.set_xticks(x)
    ax.set_xticklabels(n_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Power (W)")
    ax.set_title("Absolute per-joint power (sorted by baseline, hottest first)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate top-5 hottest with delta
    for i in range(min(5, n)):
        d = r_sorted[i] - b_sorted[i]
        sign = "+" if d > 0 else ""
        ax.annotate(f"{sign}{d:.1f}W", xy=(x[i], max(b_sorted[i], r_sorted[i]) + 1),
                    ha="center", va="bottom", fontsize=7,
                    color="red" if d > 0 else "green")

    # ---- Bottom: % change per joint ----
    ax2 = axes[1]
    dp_sorted = delta_pct[order]
    bar_colors = ["#CC3333" if d > 0 else "#336633" for d in dp_sorted]
    ax2.bar(x, dp_sorted, color=bar_colors, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_sorted, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Power change (%)")
    ax2.set_title("Per-joint power change: residual vs baseline  (green=reduced, red=increased)")
    ax2.grid(axis="y", alpha=0.3)

    # Group legend
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    axes[0].legend(handles=patches + [
        mpatches.Patch(color="gray", alpha=0.5, label="Baseline"),
        mpatches.Patch(color="gray", alpha=0.9, label="Residual"),
    ], fontsize=8, loc="upper right")

    plt.tight_layout()
    plot_path = out / "joint_power_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {plot_path}")

    # Save numerical report
    report = out / "joint_power_report.txt"
    with open(report, "w") as f:
        f.write(f"Per-Joint Power Report  cmd={cmd_vel}m/s\n")
        f.write("="*60 + "\n")
        f.write(f"{'Joint':<30} {'Base(W)':>8} {'Res(W)':>8} {'Delta(W)':>9} {'Delta%':>8}\n")
        f.write("-"*60 + "\n")
        for i in order:
            d = residual_power[i] - baseline_power[i]
            dp = d / (baseline_power[i] + 1e-6) * 100
            f.write(f"{names[i]:<30} {baseline_power[i]:>8.2f} {residual_power[i]:>8.2f} "
                    f"{d:>+9.2f} {dp:>+7.1f}%\n")
        f.write("-"*60 + "\n")
        f.write(f"{'TOTAL':<30} {baseline_power.sum():>8.2f} {residual_power.sum():>8.2f} "
                f"{(residual_power-baseline_power).sum():>+9.2f} "
                f"{(residual_power.sum()-baseline_power.sum())/baseline_power.sum()*100:>+7.1f}%\n")
    print(f"[report] saved → {report}")
    return str(plot_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fused-decoder", required=True)
    p.add_argument("--baseline-decoder", default=BASELINE_DECODER)
    p.add_argument("--cmd-vel", type=float, default=3.0)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--record-steps", type=int, default=750)
    p.add_argument("--output-dir", default="runs/joint_power_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Per-Joint Power Analysis  cmd={args.cmd_vel}m/s")
    print(f"{'='*60}\n")

    print("[BASELINE]")
    baseline_power = run_rollout_per_joint(
        args.baseline_decoder, args.cmd_vel,
        args.warmup_steps, args.record_steps, "BASELINE"
    )

    print("\n[RESIDUAL]")
    residual_power = run_rollout_per_joint(
        args.fused_decoder, args.cmd_vel,
        args.warmup_steps, args.record_steps, "RESIDUAL"
    )

    print("\n[plotting]")
    plot_joint_power(baseline_power, residual_power, args.output_dir, args.cmd_vel)

    total_b = baseline_power.sum()
    total_r = residual_power.sum()
    reduction = (total_b - total_r) / total_b * 100
    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_b:.1f}W → {total_r:.1f}W  ({reduction:+.1f}%)")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
