"""Analyze saturation during cmd=3.0 segment of /tmp/align9_check.npz.

We already saved per-step actions, joint positions, joint velocities. Check:
1. Are actions saturated? (decoder outputs in scaled units — normalized
   against NATIVE_ACTION_SCALE to show |action|/scale as a fraction of clip)
2. Are joint velocities hitting their limits (IsaacLab DCMotor velocity_limit
   after ALIGN-5 = 1000 rad/s, so should never saturate — but we show peaks).
3. Are joint torques (inferred from PD law: τ = Kp*(target - q) - Kd*qdot)
   saturating at actuator effort_limit?
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/lumi/robo_residual")
import numpy as np

NPZ = "/tmp/align9_check.npz"

from examples.sonic_energy_efficient.configs.sonic_params import (
    NATIVE_ACTION_SCALE, NATIVE_LOCOMOTION_JOINT_NAMES, NATIVE_KP, NATIVE_KD,
)

# MuJoCo torque ranges (actuatorfrcrange) by joint type
# leg_motor class: hip/pitch/roll/yaw → ±88,  knee → ±139
# ankle_motor class: ankle_pitch/roll → ±50
# arm_motor class: arms/shoulders → ±25
# wrist_motor / torso_motor / finger_motor → ±25
# Just use a lookup per joint name keyword.
MJ_TORQUE_LIMIT = {}
for n in NATIVE_LOCOMOTION_JOINT_NAMES:
    if "knee" in n: lim = 139.0
    elif "ankle" in n: lim = 50.0
    elif "hip" in n: lim = 88.0
    elif "waist_yaw" in n: lim = 88.0
    elif "waist" in n: lim = 50.0
    elif "shoulder" in n or "elbow" in n: lim = 25.0
    elif "wrist" in n: lim = 25.0
    else: lim = 25.0
    MJ_TORQUE_LIMIT[n] = lim


d = np.load(NPZ)
actions = d["actions"]           # (T, 29)  scaled decoder output (action units)
jpos_abs = d["jpos_abs"]         # (T, 29)  absolute joint positions
jpos_delta = d["jpos_delta"]     # (T, 29)  relative to default
jvel = d["jvel"]                 # (T, 29)  joint velocities
cmd_vx = d["cmd_vel"][:, 0]

# Find the cmd=3.0 window
mask_cmd3 = np.isclose(cmd_vx, 3.0)
idxs = np.where(mask_cmd3)[0]
if len(idxs) < 100:
    print("No cmd=3.0 window")
    raise SystemExit(0)
window = idxs[50:]  # skip transient
print(f"cmd=3.0 window: {len(window)} frames")

act_scale = np.asarray(NATIVE_ACTION_SCALE)          # (29,)
kp = np.asarray(NATIVE_KP)                           # (29,)
kd = np.asarray(NATIVE_KD)
default_angles = d["default_angles"]                 # (29,)

# 1. Action saturation
#   action output is scaled by action_scale to get a target joint pos offset
#   from default: q_target = default_angle + action
#   So raw decoder output (pre-scale) is action / act_scale. Saturate at ±1 of raw.
act_win = actions[window]                             # (N, 29)
act_raw = act_win / act_scale[None, :]
print(f"\n=== ACTION SATURATION (|raw_decoder_output| close to ±1 means clip) ===")
print(f"{'joint':<32} {'peak|act_raw|':>14} {'mean|act_raw|':>14} {'frac>0.95':>10}")
for i, n in enumerate(NATIVE_LOCOMOTION_JOINT_NAMES):
    peak = np.abs(act_raw[:, i]).max()
    mean = np.abs(act_raw[:, i]).mean()
    frac = (np.abs(act_raw[:, i]) > 0.95).mean()
    if peak > 0.5 or frac > 0.05:
        print(f"{n:<32} {peak:>14.3f} {mean:>14.3f} {frac:>10.2%}")

# 2. Joint velocity peaks
print(f"\n=== JOINT VELOCITY PEAKS (vs IsaacLab velocity_limit=1000 rad/s after ALIGN-5) ===")
print(f"{'joint':<32} {'|jvel| max':>12} {'|jvel| p95':>12}")
for i, n in enumerate(NATIVE_LOCOMOTION_JOINT_NAMES):
    if "hip" not in n and "knee" not in n and "ankle" not in n and "waist" not in n:
        continue
    pk = np.abs(jvel[window, i]).max()
    p95 = np.percentile(np.abs(jvel[window, i]), 95)
    print(f"{n:<32} {pk:>12.2f} {p95:>12.2f}")

# 3. Estimated torque (external PD: τ = Kp*(target - q) - Kd*qdot)
#    target = default + action
target = default_angles[None, :] + act_win            # (N, 29)
q = jpos_abs[window]
qdot = jvel[window]
tau_est = kp[None, :] * (target - q) - kd[None, :] * qdot

print(f"\n=== ESTIMATED TORQUE (|τ_est| vs MuJoCo actuatorfrcrange) ===")
print(f"{'joint':<32} {'|τ| peak':>10} {'|τ| p95':>10} {'mj_lim':>8} {'frac>lim':>10}")
for i, n in enumerate(NATIVE_LOCOMOTION_JOINT_NAMES):
    pk = np.abs(tau_est[:, i]).max()
    p95 = np.percentile(np.abs(tau_est[:, i]), 95)
    lim = MJ_TORQUE_LIMIT[n]
    frac_over = (np.abs(tau_est[:, i]) > lim).mean()
    if pk > lim * 0.5 or frac_over > 0.01:
        print(f"{n:<32} {pk:>10.2f} {p95:>10.2f} {lim:>8.1f} {frac_over:>10.2%}")

# 4. Body velocity the decoder SEES (base_lin_vel feeds into planner context, not decoder —
#    the decoder gets motion tokens + proprio). But the planner context uses root_state.
#    We can check the vx the decoder is reacting to.
rs = d["root_state_w"]
vx_world = rs[window, 7]
print(f"\n=== BODY VX (world frame) AT cmd=3.0 ===")
print(f"  mean = {vx_world.mean():.3f} m/s")
print(f"  max  = {vx_world.max():.3f} m/s")
print(f"  p90  = {np.percentile(vx_world, 90):.3f} m/s")

# 5. Joint position excursions — are we at joint pos limits?
print(f"\n=== JOINT POS EXCURSIONS (|q_delta|  peaks at cmd=3.0) ===")
q_delta = jpos_delta[window]  # already relative to default
for i, n in enumerate(NATIVE_LOCOMOTION_JOINT_NAMES):
    if "hip" in n or "knee" in n or "ankle" in n:
        pk = np.abs(q_delta[:, i]).max()
        p95 = np.percentile(np.abs(q_delta[:, i]), 95)
        scale = act_scale[i]
        print(f"{n:<32} |q-q0| peak={pk:.3f} p95={p95:.3f}  (action_scale={scale:.3f})")
