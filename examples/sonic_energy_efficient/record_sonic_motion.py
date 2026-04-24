"""Record a long continuous SONIC motion by chaining planner calls with
PROPER temporal context (4 past frames at 30 Hz).

Prior version passed 4 identical static frames as context, making the planner
think "robot just standing still" every chunk, producing repeated ramp-up from
rest (the observable "step-pause-step" pattern).  This version feeds the
planner the actual 4-frame tail of the previous motion at 30 Hz spacing, so
the planner continues steady-state running rather than restarting.

Saves a single .npz with:
    motion_50hz: [T, 36]
    jvel_50hz  : [T, 29]
    fps = 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lumi/robo_residual")

from examples.sonic_energy_efficient.env.sonic_pipeline import (
    SonicPlanner, resample_30hz_to_50hz, compute_joint_velocities_50hz,
)
from examples.sonic_energy_efficient.configs.sonic_params import (
    MODE_RUN, MODE_WALK, SONIC_DEFAULT_ANGLES,
)


def build_context_from_tail(motion_50hz: np.ndarray) -> np.ndarray:
    """Build 4-frame context (30 Hz spacing) from tail of 50 Hz motion.

    Returns (4, 36) — 4 recent qpos frames at positions t-5, t-3, t-1, t (30 Hz ≈ every 1.67 frames of 50 Hz).
    We also zero-out X/Y so the planner sees "robot centered at origin"
    (planner is translation-invariant; only joint angles + quat carry info).
    """
    # 30 Hz step ≈ 50/30 = 1.667 frames of 50 Hz
    step_50hz = 50.0 / 30.0
    tail = len(motion_50hz) - 1
    idx = [int(round(tail - i * step_50hz)) for i in range(3, -1, -1)]
    idx = [max(0, min(len(motion_50hz) - 1, i)) for i in idx]
    ctx = motion_50hz[idx].copy()  # (4, 36)
    # Zero out XY translation (keep only relative pose) so planner's output is centered
    ctx[:, 0] = 0.0
    ctx[:, 1] = 0.0
    return ctx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--planner-onnx", type=str, required=True)
    p.add_argument("--target-vel", type=float, default=2.5)
    p.add_argument("--mode", type=int, default=MODE_RUN)
    p.add_argument("--num-chunks", type=int, default=10)
    p.add_argument("--output", type=str, default="models/recorded_run_motion.npz")
    p.add_argument("--frames-per-chunk", type=int, default=40,
                   help="How many steady-state frames to take from each chunk's tail (50 Hz). "
                        "Ignoring initial ramp-up.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    planner = SonicPlanner(args.planner_onnx)

    # ---------- Chunk 0: stationary start ----------
    qpos_30hz, n_valid = planner.run(
        target_vel=args.target_vel, mode=args.mode,
        default_angles=SONIC_DEFAULT_ANGLES,
    )
    qpos_30hz = qpos_30hz[:n_valid]
    motion_50hz = resample_30hz_to_50hz(qpos_30hz, n_valid)
    print(f"Chunk 0: {n_valid} 30Hz → {len(motion_50hz)} 50Hz frames (includes ramp-up)")

    # Keep only the LAST `frames_per_chunk` frames of chunk 0 (steady-state tail)
    if len(motion_50hz) > args.frames_per_chunk:
        motion_50hz = motion_50hz[-args.frames_per_chunk:]
    print(f"  Kept last {len(motion_50hz)} frames as steady-state start")

    # ---------- Subsequent chunks: proper 4-frame temporal context ----------
    for chunk in range(1, args.num_chunks):
        ctx = build_context_from_tail(motion_50hz)

        qpos_30hz, n_valid = planner.run(
            target_vel=args.target_vel, mode=args.mode,
            context_qpos=ctx,
        )
        qpos_30hz = qpos_30hz[:n_valid]
        new_motion_50hz = resample_30hz_to_50hz(qpos_30hz, n_valid)

        # Take only steady-state tail
        if len(new_motion_50hz) > args.frames_per_chunk:
            new_motion_50hz = new_motion_50hz[-args.frames_per_chunk:]

        # Align XY translation: continue from prev tail
        new_motion_50hz = new_motion_50hz.copy()
        new_motion_50hz[:, 0] += motion_50hz[-1, 0] - new_motion_50hz[0, 0]
        new_motion_50hz[:, 1] += motion_50hz[-1, 1] - new_motion_50hz[0, 1]

        # Blend joint angles (cols 7:36) at the chunk boundary over `blend_frames`
        # to avoid ~1 rad discontinuities when planner emits a different phase of gait.
        blend_frames = 10
        prev_tail = motion_50hz[-1, 7:36].copy()
        new_head = new_motion_50hz[0, 7:36].copy()
        offset = prev_tail - new_head  # additive correction at frame 0
        for i in range(min(blend_frames, len(new_motion_50hz))):
            w = 1.0 - (i / blend_frames)  # 1 → 0 over blend_frames
            new_motion_50hz[i, 7:36] += offset * w
        # Same blend for base quaternion (slerp-approximation via linear + normalize)
        from examples.sonic_energy_efficient.env.sonic_pipeline import slerp
        for i in range(min(blend_frames, len(new_motion_50hz))):
            w = 1.0 - (i / blend_frames)
            q_tail = motion_50hz[-1, 3:7].astype(np.float64)
            q_head = new_motion_50hz[i, 3:7].astype(np.float64)
            new_motion_50hz[i, 3:7] = slerp(q_head, q_tail, w).astype(np.float32)

        motion_50hz = np.concatenate([motion_50hz, new_motion_50hz], axis=0)

        # Measure chunk velocity
        dt_chunk = len(new_motion_50hz) / 50.0
        dx_chunk = new_motion_50hz[-1, 0] - new_motion_50hz[0, 0]
        print(f"Chunk {chunk}: +{len(new_motion_50hz)} frames, chunk vx={dx_chunk/dt_chunk:+.2f}m/s")

    jvel_50hz = compute_joint_velocities_50hz(motion_50hz)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, motion_50hz=motion_50hz, jvel_50hz=jvel_50hz,
        fps=np.array([50.0], dtype=np.float32),
        target_vel=np.array([args.target_vel], dtype=np.float32),
        mode=np.array([args.mode], dtype=np.int64),
    )

    dx = motion_50hz[-1, 0] - motion_50hz[0, 0]
    dy = motion_50hz[-1, 1] - motion_50hz[0, 1]
    dt = len(motion_50hz) / 50.0
    print(f"\n=== Recorded Motion ===")
    print(f"  Total: {len(motion_50hz)} frames = {dt:.1f}s @ 50Hz")
    print(f"  Distance: dx={dx:+.2f}m dy={dy:+.2f}m")
    print(f"  Avg velocity: {dx/dt:+.2f}m/s (target {args.target_vel})")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
