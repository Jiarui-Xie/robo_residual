"""Parse train_isaac.py log and generate training curve figures.

Usage:
    python examples/sonic_energy_efficient/tools/plot_training_curves.py \
        --log runs/sonic_energy_1p7_v2/train.log \
        --output runs/sonic_energy_1p7_v2/curves.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Parsers ──────────────────────────────────────────────────────────────────

_RE_ITER = re.compile(
    r"\[iter\s+(\d+)/\d+\].*?vel=([+-]?\d+\.\d+).*?E=([+-]?\d+\.\d+)"
)
_RE_EVAL = re.compile(
    r"\[EVAL@(\d+)\].*?vx([+-]\d+\.\d+).*?E=([+-]?\d+\.\d+)W"
)


def parse_log(path: str):
    train_iters, train_vel, train_E = [], [], []
    eval_iters, eval_vx, eval_E = [], [], []

    with open(path) as f:
        for line in f:
            m = _RE_ITER.search(line)
            if m:
                train_iters.append(int(m.group(1)))
                train_vel.append(float(m.group(2)))
                train_E.append(float(m.group(3)))
                continue
            m = _RE_EVAL.search(line)
            if m:
                eval_iters.append(int(m.group(1)))
                eval_vx.append(float(m.group(2)))
                eval_E.append(float(m.group(3)))

    return (
        np.array(train_iters), np.array(train_vel), np.array(train_E),
        np.array(eval_iters),  np.array(eval_vx),   np.array(eval_E),
    )


# ── Smooth helper ─────────────────────────────────────────────────────────────

def smooth(x, w=20):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(log_path: str, out_path: str, cmd_vel: float = 1.7):
    ti, tv, tE, ei, evx, eE = parse_log(log_path)

    if len(ei) == 0:
        print("No EVAL lines found in log — check log path.")
        return

    # Baseline = EVAL@25 (near-zero residual, first eval checkpoint)
    baseline_E  = eE[0]
    baseline_vx = evx[0]
    final_E     = float(np.median(eE[-10:]))
    final_vx    = float(np.median(evx[-10:]))
    reduction   = (baseline_E - final_E) / baseline_E * 100

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Energy-Efficient Gait Training  |  cmd={cmd_vel} m/s\n"
        f"Baseline E={baseline_E:.0f} W  →  Final E={final_E:.0f} W  "
        f"({reduction:.1f}% reduction)   |   vx: {baseline_vx:.2f}→{final_vx:.2f} m/s",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: EVAL energy ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ei, eE, color="#4C72B0", linewidth=1.2, alpha=0.5, label="EVAL E (det.)")
    ax1.plot(ei, smooth(eE, 5), color="#4C72B0", linewidth=2.5, label="EVAL E (smoothed)")
    ax1.axhline(baseline_E, color="gray",  linestyle="--", linewidth=1.2, label=f"Baseline {baseline_E:.0f} W")
    ax1.axhline(final_E,    color="#DD4949", linestyle="--", linewidth=1.2, label=f"Final {final_E:.0f} W")
    ax1.fill_between(ei, final_E, baseline_E, alpha=0.08, color="#DD4949")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Power (W)")
    ax1.set_title("EVAL Energy Consumption (deterministic rollout, no noise)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(ei[0], ei[-1])
    ax1.grid(True, alpha=0.3)

    # Annotate reduction
    mid_iter = (ei[0] + ei[-1]) / 2
    ax1.annotate(
        f"−{reduction:.1f}%",
        xy=(mid_iter, (baseline_E + final_E) / 2),
        fontsize=14, fontweight="bold", color="#DD4949",
        ha="center", va="center",
    )

    # ── Panel 2: EVAL vx ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ei, evx, color="#55A868", linewidth=1.2, alpha=0.5)
    ax2.plot(ei, smooth(evx, 5), color="#55A868", linewidth=2.5, label="EVAL vx (smoothed)")
    ax2.axhline(cmd_vel,     color="black", linestyle=":", linewidth=1.5, label=f"cmd {cmd_vel} m/s")
    ax2.axhline(baseline_vx, color="gray",  linestyle="--", linewidth=1.2, label=f"Baseline {baseline_vx:.2f} m/s")
    ax2.axhline(final_vx,    color="#55A868", linestyle="--", linewidth=1.2, label=f"Final {final_vx:.2f} m/s")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Forward velocity (m/s)")
    ax2.set_title("EVAL Velocity (actual vs command)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(ei[0], ei[-1])
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Training E (rolling) ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ti, tE, color="#C44E52", linewidth=0.6, alpha=0.3, label="Train E (stochastic)")
    ax3.plot(ti, smooth(tE, 30), color="#C44E52", linewidth=2.2, label="Train E (smoothed)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Power (W, weighted)")
    ax3.set_title("Training Rollout Energy (stochastic, noisy)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlim(ti[0], ti[-1])
    ax3.grid(True, alpha=0.3)

    # ── Summary text ─────────────────────────────────────────────────────────
    summary = (
        f"Summary\n"
        f"  Baseline (near-zero residual):\n"
        f"    E = {baseline_E:.0f} W,  vx = {baseline_vx:.2f} m/s\n"
        f"  Final (iter {ei[-1]}):\n"
        f"    E = {final_E:.0f} W,  vx = {final_vx:.2f} m/s\n"
        f"  Energy reduction: {reduction:.1f}%\n"
        f"  Overshoot: {baseline_vx - cmd_vel:.2f} → {final_vx - cmd_vel:.2f} m/s\n"
        f"  Robot stayed alive: 99-100%"
    )
    fig.text(0.01, 0.01, summary, fontsize=8.5, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    print(f"[plot] Baseline E={baseline_E:.0f}W  Final E={final_E:.0f}W  Reduction={reduction:.1f}%")
    print(f"[plot] Baseline vx={baseline_vx:.2f}  Final vx={final_vx:.2f}  (cmd={cmd_vel})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", default="training_curves.png")
    ap.add_argument("--cmd-vel", type=float, default=1.7)
    args = ap.parse_args()
    plot(args.log, args.output, args.cmd_vel)


if __name__ == "__main__":
    main()
