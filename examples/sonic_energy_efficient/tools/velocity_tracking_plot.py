"""Velocity tracking stability: plot vx(t), tracking error, std over time.

Runs baseline + residual rollouts in MuJoCo and creates stability figure showing:
  (a) vx(t) trajectories overlaid — see oscillation amplitude
  (b) rolling-window std of vx — see stability improvement/degradation
  (c) tracking error |vx - cmd| distribution (histogram)

Usage:
    python examples/sonic_energy_efficient/tools/velocity_tracking_plot.py \\
        --fused runs/sonic_energy_3p0_v9/fused_best_iter1000.onnx \\
        --cmd-vel 3.0 \\
        --output runs/sonic_energy_3p0_v9/velocity_tracking.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.sonic_energy_efficient.tools import mujoco_energy_compare as M


def rolling_std(x, w):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    for i in range(w, len(x)):
        out[i] = x[i-w:i].std()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fused", required=True)
    p.add_argument("--cmd-vel", type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mj-xml", default=M.MJ_XML,
                   help="MuJoCo scene XML (default: scene_29dof.xml deploy target with friction auto-patched to 1.0)")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--record-steps", type=int, default=1000)
    args = p.parse_args()

    M.MJ_XML = args.mj_xml

    print(f"\n[velocity_tracking] Running baseline at cmd={args.cmd_vel}...")
    r_base = M.run_rollout(decoder_onnx=M.BASELINE_DECODER, cmd_vel=args.cmd_vel,
                           warmup_steps=args.warmup_steps, record_steps=args.record_steps,
                           label="baseline", video_path=None)
    print(f"\n[velocity_tracking] Running residual at cmd={args.cmd_vel}...")
    r_res = M.run_rollout(decoder_onnx=args.fused, cmd_vel=args.cmd_vel,
                          warmup_steps=args.warmup_steps, record_steps=args.record_steps,
                          label="residual", video_path=None)

    dt = 0.005 * 4  # MUJOCO_TIMESTEP * MUJOCO_DECIMATION = 0.02 s (50 Hz control)
    t = np.arange(len(r_base["vx"])) * dt
    vx_b = np.asarray(r_base["vx"])
    vx_r = np.asarray(r_res["vx"])
    err_b = vx_b - args.cmd_vel
    err_r = vx_r - args.cmd_vel
    WIN = 50  # rolling 1s window
    std_b = rolling_std(vx_b, WIN)
    std_r = rolling_std(vx_r, WIN)

    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.2], hspace=0.35, wspace=0.25)

    # (a) vx(t) overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, vx_b, color="#888", lw=1.0, label=f"baseline  μ={vx_b.mean():.2f}±{vx_b.std():.2f}")
    ax1.plot(t, vx_r, color="#2a9d8f", lw=1.0, label=f"v9 best   μ={vx_r.mean():.2f}±{vx_r.std():.2f}")
    ax1.axhline(args.cmd_vel, color="red", lw=1.0, ls="--", label=f"cmd={args.cmd_vel}")
    ax1.fill_between(t, vx_b.mean()-vx_b.std(), vx_b.mean()+vx_b.std(), alpha=0.1, color="#888")
    ax1.fill_between(t, vx_r.mean()-vx_r.std(), vx_r.mean()+vx_r.std(), alpha=0.15, color="#2a9d8f")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("forward velocity  vx (m/s)")
    ax1.set_title(f"Velocity tracking stability @ cmd={args.cmd_vel} m/s (MuJoCo 43DOF)")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # (b) rolling std
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, std_b, color="#888", lw=1.2, label=f"baseline  rolling σ (mean={np.nanmean(std_b):.3f})")
    ax2.plot(t, std_r, color="#2a9d8f", lw=1.2, label=f"v9 best   rolling σ (mean={np.nanmean(std_r):.3f})")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel(f"{WIN*dt:.0f}s rolling σ(vx)  (m/s)")
    ax2.set_title("Rolling velocity std — lower = smoother")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

    # (c) error distribution
    ax3 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(min(err_b.min(), err_r.min()), max(err_b.max(), err_r.max()), 40)
    ax3.hist(err_b, bins=bins, alpha=0.5, color="#888", label=f"baseline  μ={err_b.mean():+.2f}")
    ax3.hist(err_r, bins=bins, alpha=0.5, color="#2a9d8f", label=f"v9 best   μ={err_r.mean():+.2f}")
    ax3.axvline(0, color="red", lw=1.0, ls="--")
    ax3.set_xlabel("tracking error  vx - cmd (m/s)"); ax3.set_ylabel("frequency")
    ax3.set_title("Tracking error distribution")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # (d) per-step energy (sanity)
    ax4 = fig.add_subplot(gs[2, 1])
    E_b = np.asarray(r_base["energy"])
    E_r = np.asarray(r_res["energy"])
    ax4.plot(t, E_b, color="#888", lw=0.8, alpha=0.7,
             label=f"baseline  μ={E_b.mean():.0f}W")
    ax4.plot(t, E_r, color="#e76f51", lw=0.8, alpha=0.7,
             label=f"v9 best   μ={E_r.mean():.0f}W")
    dE = (E_r.mean() - E_b.mean()) / E_b.mean() * 100
    ax4.set_xlabel("t (s)"); ax4.set_ylabel("mechanical power (W)")
    ax4.set_title(f"Per-step power  (ΔE = {dE:+.1f}%)")
    ax4.legend(fontsize=8, loc="upper right"); ax4.grid(alpha=0.3)

    plt.suptitle(f"SONIC + Residual Locomotion (cmd={args.cmd_vel} m/s) — Tracking & Energy",
                 fontsize=12, y=0.995)
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"\n[velocity_tracking] Saved → {args.output}")

    # Print summary
    print(f"\n  baseline: μ(vx)={vx_b.mean():.3f}  σ={vx_b.std():.3f}  μ(err)={err_b.mean():+.3f}  μ(|err|)={np.abs(err_b).mean():.3f}")
    print(f"  residual: μ(vx)={vx_r.mean():.3f}  σ={vx_r.std():.3f}  μ(err)={err_r.mean():+.3f}  μ(|err|)={np.abs(err_r).mean():.3f}")
    dsig = (vx_r.std() - vx_b.std()) / vx_b.std() * 100
    derr = (np.abs(err_r).mean() - np.abs(err_b).mean()) / np.abs(err_b).mean() * 100
    print(f"  Δσ = {dsig:+.1f}%   Δ|err| = {derr:+.1f}%")


if __name__ == "__main__":
    main()
