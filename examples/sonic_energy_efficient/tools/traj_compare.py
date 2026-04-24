"""Time-series trajectory comparison between IsaacLab and MuJoCo at cmd=3.0.

Reads /tmp/perstep_compare.npz (per-step reset comparison saved by
mujoco_per_step_compare.py). For the most divergent joints, prints the actual
angle time-series IL_q vs MJ_q side-by-side so we can see the pattern of the
divergence (lag? bias? phase shift? saturation?).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/lumi/robo_residual")
import numpy as np

from examples.sonic_energy_efficient.configs.sonic_params import (
    SONIC_GROUPED_JOINT_NAMES,
)

d = np.load("/tmp/perstep_compare.npz")
mj_q = d["mj_q_post"]      # (N, 29)
il_q = d["il_q_post"]      # (N, 29)
mj_v = d["mj_linvel_post"] # (N, 3)
il_v = d["il_linvel_post"] # (N, 3)
dq   = d["dq_step"]        # (N, 29)  mj - il
t0   = int(d["t0"][0])
N    = mj_q.shape[0]

# Rank joints by mean|Δq|
mean_abs = np.abs(dq).mean(axis=0)
rank = np.argsort(-mean_abs)

print(f"=== Per-step reset comparison, {N} frames starting at IL frame {t0} ===\n")

# --- 1. Sign of bias (MJ consistently > IL or balanced?) for top joints ---
print("BIAS DIRECTION (per joint, mean signed Δq = MJ - IL):")
print(f"{'joint':<32} {'mean Δq':>10} {'mean |Δq|':>10} {'sign-consistency':>18}")
for j in rank[:10]:
    n = SONIC_GROUPED_JOINT_NAMES[j]
    m_signed = dq[:, j].mean()
    m_abs = np.abs(dq[:, j]).mean()
    # frac of frames where sign matches mean sign
    sgn = np.sign(m_signed) if m_signed != 0 else 1.0
    frac_same = (np.sign(dq[:, j]) == sgn).mean()
    print(f"{n:<32} {m_signed:>+10.5f} {m_abs:>10.5f} {frac_same:>18.1%}")

print()
print("BASE LINVEL BIAS:")
print(f"  mean Δvx = {(mj_v[:,0]-il_v[:,0]).mean():+.4f}  (consistent {np.mean(np.sign(mj_v[:,0]-il_v[:,0]) == np.sign((mj_v[:,0]-il_v[:,0]).mean())):.1%})")
print(f"  mean Δvy = {(mj_v[:,1]-il_v[:,1]).mean():+.4f}")
print(f"  mean Δvz = {(mj_v[:,2]-il_v[:,2]).mean():+.4f}")

# --- 2. Side-by-side time series for top-5 divergent joints ---
top_joints = rank[:5]
print("\n\nFULL TIME-SERIES (IL_q / MJ_q / Δq), first 40 of 300 frames:\n")
for j in top_joints:
    n = SONIC_GROUPED_JOINT_NAMES[j]
    print(f"--- {n}  (mean|Δq|={mean_abs[j]:.4f}) ---")
    print(f"{'frame':>5} {'IL_q':>10} {'MJ_q':>10} {'Δq':>10}")
    for k in range(0, 40):
        print(f"{t0+k:>5} {il_q[k,j]:>10.4f} {mj_q[k,j]:>10.4f} {dq[k,j]:>+10.4f}")
    print()

# --- 3. When IL_q changes (Δq_il = il[t+1]-il[t]), does MJ change by more/less?
#   This tells us per-step JOINT-LEVEL acceleration bias.
print("\nPER-STEP JOINT-ACCELERATION BIAS:")
print(f"  For each joint, under identical starting state and action, MJ step moves by Δmj,")
print(f"  IL step moves by Δil. Report mean(Δmj - Δil) and |Δmj - Δil|.")
print(f"{'joint':<32} {'mean Δmj-Δil':>14} {'mean |Δmj-Δil|':>16} {'peak':>8}")
# IL step = il[t+1] - il[t], but il_q is il[t+1] (post). The pre state is the one we reset MJ from.
# Reconstruct: for k, IL pre-state at frame t0+k, IL post-state at frame t0+k+1.
# il_q[k] is already the post-state (t0+k+1). We need pre-state = IL value at t0+k.
# It's not in the file, but dq[k] = mj_q[k] - il_q[k] already is Δmj-Δil when both start from same pre.
# So dq IS the per-step acceleration bias.
for j in rank[:12]:
    n = SONIC_GROUPED_JOINT_NAMES[j]
    print(f"{n:<32} {dq[:,j].mean():>+14.5f} {np.abs(dq[:,j]).mean():>16.5f} {np.abs(dq[:,j]).max():>8.4f}")
