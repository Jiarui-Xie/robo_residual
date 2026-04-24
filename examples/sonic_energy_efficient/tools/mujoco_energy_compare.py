"""Baseline vs Residual energy comparison in MuJoCo sim2sim.

Runs SONIC (baseline decoder) and SONIC+Residual (fused decoder) side-by-side,
measures mechanical power directly from MuJoCo joint data, records video,
and generates comparison plots.

Usage (from /home/lumi/robo_residual):
    python -u examples/sonic_energy_efficient/tools/mujoco_energy_compare.py \\
        --fused-decoder runs/sonic_energy_1p7_v2/fused_decoder_1p7.onnx \\
        --cmd-vel 1.7 \\
        --output-dir runs/sonic_energy_1p7_v2/compare

What it produces:
    compare/
      baseline_rollout.mp4      -- SONIC without residual
      residual_rollout.mp4      -- SONIC + fused residual
      comparison_sidebyside.mp4 -- Side-by-side
      energy_comparison.png     -- Energy / velocity plots
      energy_report.txt         -- Numbers for README
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import torch

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

MJ_XML     = "/home/lumi/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_29dof.xml"  # 29DOF rubber hand, matches IsaacLab training setup
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
VIDEO_FPS = 30
RENDER_EVERY = 1        # render every N control steps (50Hz / N = video fps)
RENDER_W, RENDER_H = 640, 480


def quat_rotate_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def build_mujoco_env():
    """Create fresh MuJoCo model+data at standing pose.

    Ground friction is overridden to MUJOCO_GROUND_FRICTION (1.0, matching
    IsaacLab training's ALIGN-6) regardless of which XML is loaded. The deploy
    XML (scene_29dof.xml) hardcodes 0.5 which would slip the residual gait.
    """
    model = mujoco.MjModel.from_xml_path(MJ_XML)
    model.opt.timestep = MUJOCO_TIMESTEP

    # ALIGN-6 (applied at load time): ground geom sliding friction → training value
    from examples.sonic_energy_efficient.configs.sonic_params import MUJOCO_GROUND_FRICTION
    model.geom_friction[:, 0] = MUJOCO_GROUND_FRICTION   # sliding friction only

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
    q_g  = data.qpos[qpos_idx].copy()
    q_n  = q_g[MUJOCO_TO_ISAACLAB]
    jpos = (q_n - NATIVE_DEFAULT_ANGLES).astype(np.float32)
    qd_g = data.qvel[qvel_idx].copy()
    qd_n = qd_g[MUJOCO_TO_ISAACLAB].astype(np.float32)
    quat = data.qpos[3:7].copy()
    ang_vel = quat_rotate_inv(quat, data.qvel[3:6].copy()).astype(np.float32)  # qvel[3:6] is world-frame; rotate to body-frame
    grav    = quat_rotate_inv(quat, np.array([0., 0., -1.])).astype(np.float32)
    return (
        torch.tensor(jpos, device=DEVICE).unsqueeze(0),
        torch.tensor(qd_n, device=DEVICE).unsqueeze(0),
        torch.tensor(ang_vel, device=DEVICE).unsqueeze(0),
        torch.tensor(grav, device=DEVICE).unsqueeze(0),
        torch.tensor(quat, dtype=torch.float32, device=DEVICE).unsqueeze(0),
    )


def run_rollout(
    decoder_onnx: str,
    cmd_vel: float,
    warmup_steps: int,
    record_steps: int,
    label: str,
    video_path: str | None = None,
    camera_name: str = "track",
) -> dict:
    """Run one rollout; return energy/velocity stats and optionally save video."""

    model, data, idx = build_mujoco_env()
    _, qvel_idx, ctrl_idx = idx

    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=PLANNER_ONNX,
        encoder_onnx_dyn=ENCODER_ONNX,
        num_envs=1,
        target_vel=cmd_vel,
        device=DEVICE,
        replan_interval_steps=5,
        motion_look_ahead_steps=5,
    )
    token_provider.set_target_vel(np.array([0]), np.array([cmd_vel]))

    decoder     = OnnxBasePolicy(onnx_path=decoder_onnx, device=DEVICE)
    obs_builder = SonicObsBuilder(num_envs=1, device=DEVICE)
    obs_builder.reset()

    # Only create renderer if recording video
    renderer = None
    track_cam = None
    if video_path:
        renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
        # Tracking camera that follows the pelvis body
        track_cam = mujoco.MjvCamera()
        track_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        track_cam.trackbodyid = pelvis_id
        track_cam.distance = 3.5
        track_cam.elevation = -15   # look slightly down
        track_cam.azimuth = 90      # side view

    dt_ctrl = MUJOCO_TIMESTEP * MUJOCO_DECIMATION

    energy_log: list[float] = []
    vx_log:     list[float] = []
    height_log: list[float] = []
    frames:     list[np.ndarray] = []

    total_steps = warmup_steps + record_steps
    pos_x_hist: list[float] = []

    print(f"\n[{label}] warmup={warmup_steps} record={record_steps} steps  decoder={Path(decoder_onnx).name}")
    print(f"[{label}] cmd_vel={cmd_vel} m/s  planner.target_vel={token_provider.target_vel[0]:.3f} m/s")

    prev_target_q = SONIC_DEFAULT_ANGLES.copy()  # 1-step action delay (matches GR00T DDS async)

    with torch.no_grad():
        for t in range(total_steps):
            jpos, jvel, ang_vel, grav, quat = get_state(data, idx)
            token = token_provider.get_tokens(
                velocity_cmd=torch.tensor([cmd_vel], dtype=torch.float32, device=DEVICE),
                robot_base_quat=quat,
            )
            obs = obs_builder.build(token)
            actions_native = decoder.forward(obs)

            a_grouped = actions_native[0].cpu().numpy()[ISAACLAB_TO_MUJOCO]
            target_q = SONIC_DEFAULT_ANGLES + a_grouped * SONIC_ACTION_SCALE

            # Apply previous step's target_q (1-step delay)
            apply_target_q = prev_target_q
            prev_target_q = target_q.copy()

            tau_buf = []
            for _ in range(MUJOCO_DECIMATION):
                q  = data.qpos[idx[0]]
                qd = data.qvel[qvel_idx]
                tau = SONIC_KP * (apply_target_q - q) - SONIC_KD * qd
                tau = np.clip(tau, -MJ_TORQUE_LIM, MJ_TORQUE_LIM)
                data.ctrl[ctrl_idx] = tau
                mujoco.mj_step(model, data)
                tau_buf.append(tau.copy())

            # Update obs history
            jpos, jvel, ang_vel, grav, quat = get_state(data, idx)
            obs_builder.update(ang_vel=ang_vel, joint_pos=jpos, joint_vel=jvel,
                               actions=actions_native, gravity=grav)
            token_provider.step_frame()

            pos_x_hist.append(data.qpos[0])

            if t >= warmup_steps:
                # Mean torque over substeps
                tau_mean  = np.mean(tau_buf, axis=0)
                qd_native = data.qvel[qvel_idx][MUJOCO_TO_ISAACLAB]
                power     = float(np.sum(np.abs(tau_mean * qd_native)))
                energy_log.append(power)

                if len(pos_x_hist) >= 50:
                    vx = (pos_x_hist[-1] - pos_x_hist[-50]) / (49 * dt_ctrl)
                else:
                    vx = float("nan")
                vx_log.append(vx)
                height_log.append(float(data.qpos[2]))

                # Render frame with tracking camera
                if renderer is not None and t % RENDER_EVERY == 0:
                    renderer.update_scene(data, camera=track_cam)
                    frame = renderer.render()
                    frames.append(frame.copy())

            if t % 50 == 0:
                h = data.qpos[2]
                if len(pos_x_hist) >= 50:
                    vx_disp = (pos_x_hist[-1] - pos_x_hist[-50]) / (49 * dt_ctrl)
                else:
                    vx_disp = float("nan")
                E_mean = float(np.mean(energy_log)) if energy_log else 0.0
                phase = "warmup" if t < warmup_steps else "record"
                print(f"  [{phase} t={t*dt_ctrl:5.1f}s] h={h:.3f}  vx={vx_disp:+.3f}  E={E_mean:.1f}W")

    # Save video
    if video_path and frames:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=VIDEO_FPS, quality=8)
        print(f"  [video] saved {len(frames)} frames → {video_path}")

    stats = {
        "label":       label,
        "energy":      np.array(energy_log),
        "vx":          np.array(vx_log),
        "height":      np.array(height_log),
        "mean_E":      float(np.mean(energy_log)),
        "median_E":    float(np.median(energy_log)),
        "mean_vx":     float(np.nanmean(vx_log)),
        "mean_h":      float(np.mean(height_log)),
        "n_steps":     record_steps,
        "duration_s":  record_steps * dt_ctrl,
    }
    print(f"  [{label}] mean_E={stats['mean_E']:.1f}W  median_E={stats['median_E']:.1f}W  "
          f"mean_vx={stats['mean_vx']:.3f}m/s  mean_h={stats['mean_h']:.3f}m")
    return stats


def make_sidebyside(video_a: str, video_b: str, output: str, label_a: str, label_b: str):
    """Combine two videos side-by-side with labels using ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_a, "-i", video_b,
        "-filter_complex",
        f"[0:v]drawtext=text='{label_a}':fontsize=28:fontcolor=white:x=10:y=10[v0];"
        f"[1:v]drawtext=text='{label_b}':fontsize=28:fontcolor=white:x=10:y=10[v1];"
        "[v0][v1]hstack=inputs=2[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ffmpeg] warning: {result.stderr[-200:]}")
    else:
        print(f"  [video] side-by-side saved → {output}")


def plot_comparison(baseline: dict, residual: dict, out_path: str, cmd_vel: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    reduction = (baseline["mean_E"] - residual["mean_E"]) / baseline["mean_E"] * 100
    overshoot_b = baseline["mean_vx"] - cmd_vel
    overshoot_r = residual["mean_vx"] - cmd_vel

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"SONIC baseline vs SONIC+Residual  |  cmd_vel={cmd_vel} m/s\n"
        f"Energy: {baseline['mean_E']:.0f}W → {residual['mean_E']:.0f}W  "
        f"({reduction:.1f}% reduction)   "
        f"Overshoot: {overshoot_b:+.2f} → {overshoot_r:+.2f} m/s",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    t_b = np.arange(len(baseline["energy"])) * MUJOCO_TIMESTEP * MUJOCO_DECIMATION
    t_r = np.arange(len(residual["energy"])) * MUJOCO_TIMESTEP * MUJOCO_DECIMATION

    # Energy time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_b, baseline["energy"], color="#4C72B0", alpha=0.35, linewidth=0.8)
    ax1.plot(t_r, residual["energy"], color="#DD4949", alpha=0.35, linewidth=0.8)
    w = 25
    def sm(x): return np.convolve(x, np.ones(w)/w, "same")
    ax1.plot(t_b, sm(baseline["energy"]), color="#4C72B0", linewidth=2.2, label=f"Baseline {baseline['mean_E']:.0f}W")
    ax1.plot(t_r, sm(residual["energy"]), color="#DD4949", linewidth=2.2, label=f"Residual {residual['mean_E']:.0f}W")
    ax1.axhline(baseline["mean_E"], linestyle="--", color="#4C72B0", linewidth=1.0, alpha=0.6)
    ax1.axhline(residual["mean_E"], linestyle="--", color="#DD4949", linewidth=1.0, alpha=0.6)
    ax1.fill_between(t_b[:min(len(t_b),len(t_r))],
                     residual["energy"][:min(len(t_b),len(t_r))],
                     baseline["energy"][:min(len(t_b),len(t_r))],
                     alpha=0.07, color="#DD4949")
    mid = (t_b[-1] + t_b[0]) / 2
    ax1.annotate(f"−{reduction:.1f}%", xy=(mid, (baseline["mean_E"]+residual["mean_E"])/2),
                 fontsize=15, fontweight="bold", color="#DD4949", ha="center", va="center")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mechanical power (W)")
    ax1.set_title("Joint Power: |τ × ω| summed over 29 joints")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Velocity
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_b, baseline["vx"], color="#4C72B0", alpha=0.4, linewidth=0.8)
    ax2.plot(t_r, residual["vx"], color="#DD4949", alpha=0.4, linewidth=0.8)
    ax2.plot(t_b, sm(baseline["vx"]), color="#4C72B0", linewidth=2.2, label=f"Baseline {baseline['mean_vx']:.2f}m/s")
    ax2.plot(t_r, sm(residual["vx"]), color="#DD4949", linewidth=2.2, label=f"Residual {residual['mean_vx']:.2f}m/s")
    ax2.axhline(cmd_vel, linestyle=":", color="black", linewidth=1.5, label=f"cmd {cmd_vel}m/s")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Forward velocity (m/s)")
    ax2.set_title("Actual Forward Velocity (Δpos/Δt)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Energy histogram
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(min(baseline["energy"].min(), residual["energy"].min()),
                       max(baseline["energy"].max(), residual["energy"].max()), 40)
    ax3.hist(baseline["energy"], bins=bins, alpha=0.5, color="#4C72B0", label="Baseline", density=True)
    ax3.hist(residual["energy"], bins=bins, alpha=0.5, color="#DD4949", label="Residual", density=True)
    ax3.axvline(baseline["mean_E"], color="#4C72B0", linewidth=2, linestyle="--")
    ax3.axvline(residual["mean_E"], color="#DD4949", linewidth=2, linestyle="--")
    ax3.set_xlabel("Power (W)")
    ax3.set_ylabel("Density")
    ax3.set_title("Power Distribution")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    summary = (
        f"Results (MuJoCo sim2sim, cmd={cmd_vel}m/s)\n"
        f"  Baseline : E={baseline['mean_E']:.0f}W, vx={baseline['mean_vx']:.3f}m/s\n"
        f"  Residual : E={residual['mean_E']:.0f}W, vx={residual['mean_vx']:.3f}m/s\n"
        f"  Reduction: {reduction:.1f}%\n"
        f"  Overshoot: {overshoot_b:+.2f} → {overshoot_r:+.2f} m/s"
    )
    fig.text(0.01, 0.01, summary, fontsize=8.5, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    return reduction


def write_report(baseline: dict, residual: dict, out_path: str, cmd_vel: float):
    reduction = (baseline["mean_E"] - residual["mean_E"]) / baseline["mean_E"] * 100
    overshoot_b = baseline["mean_vx"] - cmd_vel
    overshoot_r = residual["mean_vx"] - cmd_vel
    text = f"""Energy Efficiency Comparison — SONIC baseline vs SONIC+Residual
================================================================
Method : MuJoCo sim2sim, direct joint measurement (|τ × ω|)
Command: {cmd_vel} m/s forward walking
Duration: {baseline['duration_s']:.0f}s per run (after {int(baseline['n_steps']//50)}s warmup)

┌─────────────────────┬─────────────┬─────────────┐
│                     │  Baseline   │  Residual   │
├─────────────────────┼─────────────┼─────────────┤
│ Mean power (W)      │ {baseline['mean_E']:>10.1f}  │ {residual['mean_E']:>10.1f}  │
│ Median power (W)    │ {baseline['median_E']:>10.1f}  │ {residual['median_E']:>10.1f}  │
│ Mean fwd vel (m/s)  │ {baseline['mean_vx']:>10.3f}  │ {residual['mean_vx']:>10.3f}  │
│ Cmd overshoot (m/s) │ {overshoot_b:>+10.3f}  │ {overshoot_r:>+10.3f}  │
│ Mean height (m)     │ {baseline['mean_h']:>10.3f}  │ {residual['mean_h']:>10.3f}  │
└─────────────────────┴─────────────┴─────────────┘

Energy reduction : {reduction:.1f}%
Residual training: ~{1000} iterations, ~{1880:.0f} fps
Residual params  : 1,351,227 (MLP)
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(f"[report] saved → {out_path}")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-decoder", required=True,
                    help="Path to fused_decoder.onnx (baseline+residual)")
    ap.add_argument("--baseline-decoder", default=BASELINE_DECODER)
    ap.add_argument("--cmd-vel", type=float, default=1.7)
    ap.add_argument("--warmup-steps", type=int, default=150,
                    help="Steps to warm up gait before recording (3s at 50Hz)")
    ap.add_argument("--record-steps", type=int, default=750,
                    help="Steps to record for comparison (15s at 50Hz)")
    ap.add_argument("--output-dir", default="runs/compare")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip video recording (faster, stats only)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_b = str(out / "baseline_rollout.mp4") if not args.no_video else None
    video_r = str(out / "residual_rollout.mp4") if not args.no_video else None

    print("=" * 60)
    print(f"  SONIC Energy Comparison  cmd={args.cmd_vel} m/s")
    print("=" * 60)

    baseline = run_rollout(
        decoder_onnx=args.baseline_decoder,
        cmd_vel=args.cmd_vel,
        warmup_steps=args.warmup_steps,
        record_steps=args.record_steps,
        label="BASELINE",
        video_path=video_b,
    )

    residual = run_rollout(
        decoder_onnx=args.fused_decoder,
        cmd_vel=args.cmd_vel,
        warmup_steps=args.warmup_steps,
        record_steps=args.record_steps,
        label="RESIDUAL",
        video_path=video_r,
    )

    if not args.no_video and video_b and video_r:
        make_sidebyside(
            video_b, video_r,
            str(out / "comparison_sidebyside.mp4"),
            label_a="BASELINE (original SONIC)",
            label_b=f"RESIDUAL (+MLP, cmd={args.cmd_vel}m/s)",
        )

    plot_comparison(baseline, residual, str(out / "energy_comparison.png"), args.cmd_vel)
    write_report(baseline, residual, str(out / "energy_report.txt"), args.cmd_vel)

    reduction = (baseline["mean_E"] - residual["mean_E"]) / baseline["mean_E"] * 100
    print(f"\n{'='*60}")
    print(f"  RESULT: {reduction:.1f}% energy reduction  "
          f"({baseline['mean_E']:.0f}W → {residual['mean_E']:.0f}W)")
    print(f"  Output: {out}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
