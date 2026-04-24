#!/usr/bin/env python3
import os
os.environ.setdefault("CYCLONEDDS_HOME", "/usr/local")

"""
Real-time terminal monitor for Unitree G1 29-dof joint state.

Displays:
  - Body velocity (from HighState)
  - Joint position / velocity / torque
  - Instantaneous power (tau * dq) and cumulative energy per joint
  - Heat warning when cumulative energy exceeds threshold

Usage:
  python3 monitor_state.py              # loopback, domain 0
  python3 monitor_state.py eth0         # specify interface
  python3 monitor_state.py eth0 1       # specify interface & domain id
"""

import sys
import time
import curses
import threading
import yaml
from pathlib import Path

# ── Read config.yaml for domain_id / interface ─────────────────────────────
_cfg_path = Path(__file__).parent / "simulate" / "config.yaml"
_cfg = {}
if _cfg_path.exists():
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f) or {}

_robot = _cfg.get("robot", "g1")
_is_hg = _robot in ("g1",)

if _is_hg:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# ── Joint names for supported robots ───────────────────────────────────────
G1_29DOF_NAMES = [
    # Left leg (0-5)
    "L_hip_pitch ", "L_hip_roll  ", "L_hip_yaw   ",
    "L_knee      ", "L_ankle_pit ", "L_ankle_rol ",
    # Right leg (6-11)
    "R_hip_pitch ", "R_hip_roll  ", "R_hip_yaw   ",
    "R_knee      ", "R_ankle_pit ", "R_ankle_rol ",
    # Waist (12-14)
    "waist_yaw   ", "waist_roll  ", "waist_pitch ",
    # Left arm (15-21)
    "L_shldr_pit ", "L_shldr_rol ", "L_shldr_yaw ",
    "L_elbow     ", "L_wrist_rol ", "L_wrist_pit ", "L_wrist_yaw ",
    # Right arm (22-28)
    "R_shldr_pit ", "R_shldr_rol ", "R_shldr_yaw ",
    "R_elbow     ", "R_wrist_rol ", "R_wrist_pit ", "R_wrist_yaw ",
]

GO2_NAMES = [
    "FR_hip  ", "FR_thigh", "FR_calf ",
    "FL_hip  ", "FL_thigh", "FL_calf ",
    "RR_hip  ", "RR_thigh", "RR_calf ",
    "RL_hip  ", "RL_thigh", "RL_calf ",
]

JOINT_NAMES = G1_29DOF_NAMES if _is_hg else GO2_NAMES
NUM_JOINTS = len(JOINT_NAMES)

# Group boundaries for visual separation
if _is_hg:
    GROUPS = [
        ("Left Leg",  0,  6),
        ("Right Leg", 6,  12),
        ("Waist",     12, 15),
        ("Left Arm",  15, 22),
        ("Right Arm", 22, 29),
    ]
else:
    GROUPS = [
        ("FR", 0, 3), ("FL", 3, 6), ("RR", 6, 9), ("RL", 9, 12),
    ]

# ── Shared state ──────────────────────────────────────────────────────────
low_state = None
high_state = None
lock = threading.Lock()

# Cumulative energy per joint (Joules)
energy = [0.0] * NUM_JOINTS
_prev_time = None


def low_state_handler(msg):
    global low_state, _prev_time, energy
    now = time.monotonic()
    with lock:
        if low_state is not None and _prev_time is not None:
            dt = now - _prev_time
            for i in range(NUM_JOINTS):
                tau = msg.motor_state[i].tau_est
                dq = msg.motor_state[i].dq
                energy[i] += abs(tau * dq) * dt
        low_state = msg
        _prev_time = now


def high_state_handler(msg):
    global high_state
    with lock:
        high_state = msg


def draw(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,   -1)
    curses.init_pair(2, curses.COLOR_GREEN,  -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_WHITE,  -1)
    curses.init_pair(5, curses.COLOR_RED,    -1)

    HEADER = curses.color_pair(1) | curses.A_BOLD
    LABEL  = curses.color_pair(3)
    NORMAL = curses.color_pair(4)
    GREEN  = curses.color_pair(2)
    HIGH   = curses.color_pair(5) | curses.A_BOLD

    POWER_WARN = 50.0    # W  — highlight power above this
    ENERGY_WARN = 500.0  # J  — highlight cumulative energy (heat proxy)

    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        if key == ord('r'):
            with lock:
                for i in range(NUM_JOINTS):
                    energy[i] = 0.0

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        row = 0

        def put(r, c, text, attr=NORMAL):
            if 0 <= r < h - 1:
                stdscr.addnstr(r, c, text, w - c - 1, attr)

        title = f" Unitree {_robot.upper()} — State Monitor  [q]uit  [r]eset energy "
        put(row, max(0, (w - len(title)) // 2), title, HEADER)
        row += 1
        put(row, 0, "─" * min(w - 1, 90), HEADER)
        row += 1

        # ── Body velocity ──────────────────────────────────────────────
        put(row, 0, "Body Velocity", LABEL)
        row += 1

        with lock:
            hs = high_state
        if hs is not None:
            vx, vy, vz = hs.velocity[0], hs.velocity[1], hs.velocity[2]
            speed = (vx**2 + vy**2 + vz**2) ** 0.5
            put(row, 2,
                f"vx:{vx:+7.3f}  vy:{vy:+7.3f}  vz:{vz:+7.3f}  |v|:{speed:6.3f} m/s",
                NORMAL)
        else:
            put(row, 2, "waiting...", NORMAL)
        row += 2

        # ── Joint table header ─────────────────────────────────────────
        hdr = f"  {'Joint':<14} {'q(rad)':>9} {'dq(rad/s)':>10} {'tau(Nm)':>9} {'Power(W)':>9} {'Energy(J)':>10}"
        put(row, 0, hdr, LABEL)
        row += 1
        put(row, 0, "  " + "─" * (len(hdr) - 2), LABEL)
        row += 1

        with lock:
            ls = low_state
            e_snap = energy[:]

        if ls is not None:
            for gname, gstart, gend in GROUPS:
                put(row, 0, f"  [{gname}]", LABEL)
                row += 1
                for i in range(gstart, gend):
                    if row >= h - 3:
                        break
                    ms = ls.motor_state[i]
                    q   = ms.q
                    dq  = ms.dq
                    tau = ms.tau_est
                    pwr = tau * dq
                    ej  = e_snap[i]

                    pwr_style = HIGH if abs(pwr) > POWER_WARN else NORMAL
                    ej_style  = HIGH if ej > ENERGY_WARN else GREEN

                    line = f"  {JOINT_NAMES[i]}  {q:+9.4f} {dq:+10.4f} {tau:+9.3f}"
                    put(row, 0, line, NORMAL)
                    # power
                    pwr_str = f" {pwr:+9.2f}"
                    put(row, len(line), pwr_str, pwr_style)
                    # energy
                    ej_str = f" {ej:10.1f}"
                    put(row, len(line) + len(pwr_str), ej_str, ej_style)
                    row += 1
                if row >= h - 3:
                    break

            # ── Total power ────────────────────────────────────────────
            row += 1
            total_pwr = sum(
                ls.motor_state[i].tau_est * ls.motor_state[i].dq
                for i in range(NUM_JOINTS)
            )
            total_e = sum(e_snap)
            put(row, 2,
                f"Total:  Power={total_pwr:+.1f} W   Energy={total_e:.1f} J",
                LABEL)
            row += 1
        else:
            put(row, 2, "waiting for LowState...", NORMAL)
            row += 1

        row += 1
        ts = time.strftime("%H:%M:%S")
        put(min(row, h - 2), 0,
            f"  {ts}  Power>{POWER_WARN:.0f}W=red  Energy>{ENERGY_WARN:.0f}J=red(heat risk)",
            curses.color_pair(3))

        stdscr.refresh()
        time.sleep(0.05)


def main():
    # CLI args override config.yaml
    interface = sys.argv[1] if len(sys.argv) >= 2 else _cfg.get("interface", "lo")
    domain_id = int(sys.argv[2]) if len(sys.argv) >= 3 else _cfg.get("domain_id", 0)

    ChannelFactoryInitialize(domain_id, interface)

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(low_state_handler, 10)

    sub_high = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub_high.Init(high_state_handler, 10)

    time.sleep(0.3)
    curses.wrapper(draw)


if __name__ == "__main__":
    main()
