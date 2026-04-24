"""Record an official SONIC sim2sim trajectory into reference_dataset.npz.

Architecture (gamepad-via-DDS, no keyboard stdin faking):

    Terminal A ────────────────────────────────
      python -m examples.sonic_energy_efficient.tools.sim_with_gamepad
        ↓ monkey-patches UnitreeSdk2Bridge.PublishLowState so every tick it
          reads 40 bytes from /tmp/wireless_remote_inject.bin and stuffs
          them into low_state.wireless_remote

    Terminal B (this script) ──────────────────
      1. starts deploy.sh as a subprocess (pipes 'y\\n' on stdin to answer
         bash's "Proceed?" prompt; captures stdout).
      2. watches deploy stdout for "Init Done".
      3. writes a sequence of gamepad frames to /tmp/wireless_remote_inject.bin:
            F1 pressed (2 s)  → Gamepad.use_planner = true
            release (0.5 s)
            start pressed (0.5 s) → start_control → WAIT_FOR_CONTROL → CONTROL
            release + ly=1.0 held (persistent) → forward motion
      4. DDS-subscribes to rt/lowstate / rt/lowcmd / rt/odostate at 50 Hz.
      5. saves dataset after `--duration` seconds.

The gamepad path avoids keyboard_handler.hpp's momentum decay entirely:
`ly` is a persistent analog value so forward motion holds forever as long
as we keep writing ly=1.0 into the inject file.

Must run in .venv_sim (needs unitree_sdk2py + CycloneDDS).
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

GEAR_SONIC_ROOT = Path("/home/lumi/GR00T-WholeBodyControl")
DEPLOY_DIR = GEAR_SONIC_ROOT / "gear_sonic_deploy"
DEPLOY_SH = DEPLOY_DIR / "deploy.sh"

INJECT_PATH = Path(os.environ.get("WIRELESS_REMOTE_INJECT", "/tmp/wireless_remote_inject.bin"))

# Hard-copied from examples/sonic_energy_efficient/configs/sonic_params.py so this
# script has no dependency on robo_residual being importable from .venv_sim.
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)

SONIC_DEFAULT_ANGLES_GROUPED = np.array([
    -0.312, 0.0,   0.0,   0.669, -0.363, 0.0,
    -0.312, 0.0,   0.0,   0.669, -0.363, 0.0,
     0.0,   0.0,   0.0,
     0.2,   0.2,   0.0,   0.6,   0.0,    0.0, 0.0,
     0.2,  -0.2,   0.0,   0.6,   0.0,    0.0, 0.0,
], dtype=np.float32)

SONIC_ACTION_SCALE_GROUPED = np.array([
    0.5, 0.5, 0.5, 0.5, 0.25, 0.25,
    0.5, 0.5, 0.5, 0.5, 0.25, 0.25,
    0.5, 0.25, 0.25,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25,
], dtype=np.float32)

NATIVE_DEFAULT_ANGLES = SONIC_DEFAULT_ANGLES_GROUPED[MUJOCO_TO_ISAACLAB]
NATIVE_ACTION_SCALE = SONIC_ACTION_SCALE_GROUPED[MUJOCO_TO_ISAACLAB]

RECORD_HZ = 50.0
RECORD_DT = 1.0 / RECORD_HZ

# xKeySwitchUnion bit layout (from gear_sonic_deploy/.../gamepad.hpp):
#   byte 0 (wireless_remote[2]): bit0=R1 bit1=L1 bit2=start bit3=select
#                                bit4=R2 bit5=L2 bit6=F1 bit7=F2
#   byte 1 (wireless_remote[3]): bit0=A bit1=B bit2=X bit3=Y
#                                bit4=up bit5=right bit6=down bit7=left
BTN_R1    = 1 << 0   # planner mode: R1 increments movement_mode (1=slow_walk→2=walk→3=run)
BTN_R2    = 1 << 4   # planner mode: R2 held → speed += 0.02/tick (50 Hz); run range [1.5, 3.0]
BTN_START = 1 << 2
BTN_F1    = 1 << 6


def make_wireless_remote(lx: float = 0.0, ly: float = 0.0,
                         rx: float = 0.0, ry: float = 0.0,
                         l2: float = 0.0,
                         btn_byte0: int = 0, btn_byte1: int = 0) -> bytes:
    """Encode 40-byte xRockerBtnDataStruct little-endian.

    Layout: head[2] | btn(2) | lx(4) | rx(4) | ry(4) | L2(4) | ly(4) | idle[16]
    """
    buf = bytearray(40)
    buf[2] = btn_byte0 & 0xFF
    buf[3] = btn_byte1 & 0xFF
    struct.pack_into("<f", buf, 4, lx)
    struct.pack_into("<f", buf, 8, rx)
    struct.pack_into("<f", buf, 12, ry)
    struct.pack_into("<f", buf, 16, l2)
    struct.pack_into("<f", buf, 20, ly)
    return bytes(buf)


def write_gamepad(packet: bytes) -> None:
    # Atomic-ish replace: write to temp then rename.
    tmp = INJECT_PATH.with_suffix(INJECT_PATH.suffix + ".tmp")
    tmp.write_bytes(packet)
    tmp.replace(INJECT_PATH)


# --------------------------------------------------------------------------- #
# Deploy subprocess (no pty — stdin just gets 'y\n' once for bash prompt)
# --------------------------------------------------------------------------- #

class DeploySubprocess:
    MARKER_INIT_DONE = "Init Done"
    MARKER_PROMPT = "Proceed with deployment?"

    def __init__(self, env_type: str = "sim",
                 init_timeout: float = 240.0,
                 mirror_output: bool = True) -> None:
        self.env_type = env_type
        self.init_timeout = init_timeout
        self.mirror_output = mirror_output
        self.proc: subprocess.Popen | None = None
        self.init_done_event = threading.Event()
        self.prompt_seen_event = threading.Event()
        self.stop_event = threading.Event()
        self._mirror_thread: threading.Thread | None = None

    def start(self) -> None:
        # --input-type gamepad bypasses InterfaceManager so the Gamepad class's
        # update()/handle_input() runs every tick.  (In manager mode, the active
        # sub-interface defaults to KEYBOARD (index 0) and gamepad bytes would
        # be copied but never processed.)
        self.proc = subprocess.Popen(
            ["bash", str(DEPLOY_SH), "--input-type", "gamepad", self.env_type],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(DEPLOY_DIR),
            preexec_fn=os.setsid,
            bufsize=0,
        )
        # Pre-answer the "Proceed? [Y/n]" prompt before the mirror thread starts.
        # bash's `read -p "..."` outputs the prompt WITHOUT a trailing newline, so
        # readline() would block forever waiting for \n — the answer would never
        # be sent and deploy would hang at the prompt.  Writing y\n into the pipe
        # buffer now means bash's `read` consumes it as soon as it runs.
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b"y\n")
            self.proc.stdin.flush()
            print("[record] pre-answered deploy Proceed prompt with 'y'", flush=True)
        except Exception as e:
            print(f"[record] WARNING: failed to pre-answer stdin: {e}", flush=True)
        self._mirror_thread = threading.Thread(target=self._mirror_loop, daemon=True)
        self._mirror_thread.start()

    def _mirror_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                if self.stop_event.is_set():
                    break
                line = raw.decode(errors="replace")
                if self.mirror_output:
                    sys.stdout.write("[deploy] " + line)
                    sys.stdout.flush()
                if not self.prompt_seen_event.is_set() and self.MARKER_PROMPT in line:
                    self.prompt_seen_event.set()
                    print("[record] deploy Proceed prompt seen (already pre-answered)", flush=True)
                if self.MARKER_INIT_DONE in line:
                    self.init_done_event.set()
        except Exception:
            pass

    def wait_init_done(self) -> bool:
        deadline = time.monotonic() + self.init_timeout
        while not self.init_done_event.is_set():
            if self.stop_event.is_set():
                return False
            if time.monotonic() > deadline:
                print(f"[record] TIMEOUT {self.init_timeout:.0f}s waiting for "
                      f"'{self.MARKER_INIT_DONE}'", flush=True)
                return False
            if self.proc is not None and self.proc.poll() is not None:
                print(f"[record] deploy exited early (rc={self.proc.returncode}) "
                      "before Init Done", flush=True)
                return False
            time.sleep(0.1)
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=2)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Gamepad state-machine: after "Init Done", press F1, then start, then hold ly
# --------------------------------------------------------------------------- #

def gamepad_sequence(ly_forward: float = 1.0, target_speed: float = 1.5,
                     stop_event: threading.Event | None = None) -> None:
    stop = stop_event or threading.Event()

    def hold(packet: bytes, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not stop.is_set() and time.monotonic() < deadline:
            write_gamepad(packet)
            time.sleep(0.02)  # 50 Hz rewrite so any reader reads fresh bytes

    idle        = make_wireless_remote()
    f1_pressed  = make_wireless_remote(btn_byte0=BTN_F1)
    r1_pressed  = make_wireless_remote(btn_byte0=BTN_R1)
    r2_pressed  = make_wireless_remote(btn_byte0=BTN_R2)
    # start WITHOUT ly so WBC stabilizes in standing before any velocity command
    start_only  = make_wireless_remote(btn_byte0=BTN_START)
    running     = make_wireless_remote(ly=ly_forward)

    # R2 hold time to ramp speed from RUN-mode default (1.5) to target_speed.
    # Each tick at 50 Hz adds 0.02 m/s; first tick clamps from -1 to 1.5.
    r2_hold_s = max(0.0, (target_speed - 1.5) / (0.02 * RECORD_HZ) + 0.1)

    # 1. Start button press → start_control → WBC engages in standing mode.
    #    No ly yet — let the robot stabilize on its feet first.
    print("[record] gamepad: pressing start (ly=0) → start_control", flush=True)
    hold(start_only, 0.5)

    # 2. Release start; give the WBC 2 s to stabilize in standing before
    #    switching to a running-gait reference.
    print("[record] gamepad: releasing start, standing settle 2.0 s", flush=True)
    hold(idle, 2.0)

    # 3. F1 press (edge: released→pressed) to toggle use_planner ON.
    #    Robot is already stably controlled, so switching the motion reference
    #    to the planner is a smooth transition rather than a cold jump.
    print("[record] gamepad: pressing F1 → toggle use_planner ON", flush=True)
    hold(f1_pressed, 1.0)

    # 4. Release F1 (give edge detector a frame with F1=0).
    print("[record] gamepad: releasing F1", flush=True)
    hold(idle, 0.5)

    # 5. Wait for planner to initialize (TRT compilation can take several
    #    seconds; handle_input also blocks up to 5 s waiting for planner_motion).
    print("[record] gamepad: waiting 5.0 s for planner init", flush=True)
    hold(idle, 5.0)

    # 6. Press R1 twice: mode 1 (slow_walk) → 2 (walk) → 3 (run).
    #    Each press needs a clean release so the edge detector fires once.
    print("[record] gamepad: R1 press #1 → mode 2 (walk)", flush=True)
    hold(r1_pressed, 0.3)
    hold(idle, 0.3)
    print("[record] gamepad: R1 press #2 → mode 3 (run)", flush=True)
    hold(r1_pressed, 0.3)
    hold(idle, 0.3)

    # 7. Start running immediately; hold R2 simultaneously so ly > dead_zone
    #    while speed ramps.  (If R2 is held with ly=0, deploy's dead-zone check
    #    forces final_speed→0 each tick, blocking accumulation to the planner.)
    if r2_hold_s > 0.01:
        r2_running = make_wireless_remote(btn_byte0=BTN_R2, ly=ly_forward)
        print(f"[record] gamepad: running + R2 held {r2_hold_s:.2f}s "
              f"→ ramp to {target_speed:.1f} m/s", flush=True)
        hold(r2_running, r2_hold_s)

    # 8. Release R2; hold ly forever at target speed.
    print(f"[record] gamepad: running (ly={ly_forward:.2f} held)", flush=True)
    while not stop.is_set():
        write_gamepad(running)
        time.sleep(0.02)


def quat_rotate_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


# --------------------------------------------------------------------------- #
# DDS recorder
# --------------------------------------------------------------------------- #

class SimRecorder:
    def __init__(self, target_vel: float) -> None:
        self.target_vel = target_vel
        self._low_state = None
        self._low_cmd = None
        self._odo_state = None
        self.frames: list[dict] = []
        self.last_record_time = -1.0
        self.enabled = False

    def set_low_state(self, msg) -> None: self._low_state = msg
    def set_low_cmd(self, msg) -> None:   self._low_cmd   = msg
    def set_odo_state(self, msg) -> None: self._odo_state = msg

    def enable(self) -> None:
        self.enabled = True
        self.last_record_time = time.monotonic()
        print("[record] recorder enabled", flush=True)

    def tick(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_record_time < RECORD_DT:
            return
        if self._low_state is None or self._low_cmd is None or self._odo_state is None:
            return
        self.last_record_time += RECORD_DT
        if self.last_record_time < now - 0.5:
            self.last_record_time = now

        ls, lc, odo = self._low_state, self._low_cmd, self._odo_state

        body_q_grouped = np.array([ls.motor_state[i].q for i in range(29)], dtype=np.float32)
        body_dq_grouped = np.array([ls.motor_state[i].dq for i in range(29)], dtype=np.float32)
        cmd_q_grouped = np.array([lc.motor_cmd[i].q for i in range(29)], dtype=np.float32)

        body_q = body_q_grouped[MUJOCO_TO_ISAACLAB]
        body_dq = body_dq_grouped[MUJOCO_TO_ISAACLAB]
        cmd_q = cmd_q_grouped[MUJOCO_TO_ISAACLAB]

        jpos_delta = body_q - NATIVE_DEFAULT_ANGLES
        action = (cmd_q - NATIVE_DEFAULT_ANGLES) / NATIVE_ACTION_SCALE

        quat_wxyz = np.array(ls.imu_state.quaternion, dtype=np.float32)
        ang_vel_world = np.array(ls.imu_state.gyroscope, dtype=np.float32)
        ang_vel_b = quat_rotate_inv(quat_wxyz, ang_vel_world).astype(np.float32)
        gravity_b = quat_rotate_inv(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

        pos_w = np.array(odo.position, dtype=np.float32)
        linvel_w = np.array(odo.linear_velocity, dtype=np.float32)
        root_state = np.concatenate([pos_w, quat_wxyz, linvel_w, ang_vel_world]).astype(np.float32)

        self.frames.append({
            "jpos_abs": body_q.astype(np.float32),
            "jpos_delta": jpos_delta.astype(np.float32),
            "jvel": body_dq.astype(np.float32),
            "ang_vel_b": ang_vel_b,
            "gravity_b": gravity_b,
            "actions": action.astype(np.float32),
            "root_state_w": root_state,
            "cmd_vel": np.array([self.target_vel, 0.0, 0.0], dtype=np.float32),
        })


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--output", type=str, default="/tmp/reference_dataset_official.npz")
    ap.add_argument("--target-vel", type=float, default=2.0,
                    help="Value stored in cmd_vel field of dataset (documentation / label).")
    ap.add_argument("--ly-forward", type=float, default=1.0,
                    help="Left-stick Y value held during recording (must be > dead_zone=0.05).")
    ap.add_argument("--target-speed", type=float, default=1.5,
                    help="Desired run speed in m/s (RUN mode range 1.5–3.0). "
                         "Values >1.5 hold R2 to ramp up. Also sets --target-vel if unspecified.")
    ap.add_argument("--init-timeout", type=float, default=240.0,
                    help="Max seconds to wait for deploy 'Init Done'.")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--interface", type=str, default="lo")
    ap.add_argument("--no-mirror-deploy", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true",
                    help="Don't spawn deploy — assume user already ran it manually.")
    args = ap.parse_args()

    # Reset inject file to neutral (all-zero) before anything subscribes.
    write_gamepad(make_wireless_remote())
    print(f"[record] gamepad inject file: {INJECT_PATH}", flush=True)

    # --- DDS init ---
    from unitree_sdk2py.core.channel import (
        ChannelSubscriber, ChannelFactoryInitialize,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, OdoState_

    print(f"[record] initializing DDS (domain={args.domain_id}, interface={args.interface})", flush=True)
    ChannelFactoryInitialize(args.domain_id, args.interface)

    recorder = SimRecorder(target_vel=args.target_vel)
    sub_state = ChannelSubscriber("rt/lowstate", LowState_); sub_state.Init(recorder.set_low_state, 10)
    sub_cmd   = ChannelSubscriber("rt/lowcmd", LowCmd_);     sub_cmd.Init(recorder.set_low_cmd, 10)
    sub_odo   = ChannelSubscriber("rt/odostate", OdoState_); sub_odo.Init(recorder.set_odo_state, 10)

    # --- Spawn deploy (or skip) ---
    deploy: DeploySubprocess | None = None
    if not args.skip_deploy:
        deploy = DeploySubprocess(
            env_type="sim",
            init_timeout=args.init_timeout,
            mirror_output=not args.no_mirror_deploy,
        )
        deploy.start()
    else:
        print("[record] --skip-deploy: assuming deploy is already running", flush=True)

    # Ctrl+C → clean shutdown
    stop_main = threading.Event()
    def handle_sigint(signum, frame):
        print("\n[record] Ctrl+C received — shutting down", flush=True)
        stop_main.set()
    signal.signal(signal.SIGINT, handle_sigint)

    # --- Wait for deploy Init Done ---
    if deploy is not None:
        print("[record] waiting for deploy 'Init Done'...", flush=True)
        if not deploy.wait_init_done():
            deploy.stop()
            sys.exit(1)
        if stop_main.is_set():
            deploy.stop()
            sys.exit(130)

    # --- Kick off gamepad state machine in a thread ---
    gp_stop = threading.Event()
    gp_thread = threading.Thread(
        target=gamepad_sequence,
        kwargs=dict(ly_forward=args.ly_forward, target_speed=args.target_speed,
                    stop_event=gp_stop),
        daemon=True,
    )
    gp_thread.start()

    # Let start + stand-settle + F1 + planner-init run before we begin recording.
    r2_ramp_s = max(0.0, (args.target_speed - 1.5) / (0.02 * RECORD_HZ) + 0.3)
    settle_s = 14.0 + r2_ramp_s
    print(f"[record] letting control engage for {settle_s:.1f}s before enabling recording", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_s:
        if stop_main.is_set():
            break
        time.sleep(0.1)

    recorder.enable()
    record_start = time.monotonic()
    deadline = record_start + args.duration
    last_report = record_start
    while not stop_main.is_set():
        now = time.monotonic()
        if now >= deadline:
            print(f"[record] duration reached ({args.duration:.0f}s)", flush=True)
            break
        recorder.tick()
        if now - last_report >= 10.0:
            print(f"[record] t={now - record_start:6.1f}s  frames={len(recorder.frames)}", flush=True)
            last_report = now
        time.sleep(0.005)

    # --- Shutdown order: stop gamepad (zero ly) → stop deploy ---
    gp_stop.set()
    write_gamepad(make_wireless_remote())  # neutral packet
    time.sleep(0.2)
    if deploy is not None:
        deploy.stop()

    # --- Save ---
    frames = recorder.frames
    if len(frames) < 10:
        print(f"[record] only {len(frames)} frames captured — refusing to save", flush=True)
        sys.exit(2)

    T = len(frames)
    out = {
        "jpos_abs":     np.stack([f["jpos_abs"]     for f in frames]),
        "jpos_delta":   np.stack([f["jpos_delta"]   for f in frames]),
        "jvel":         np.stack([f["jvel"]         for f in frames]),
        "ang_vel_b":    np.stack([f["ang_vel_b"]    for f in frames]),
        "gravity_b":    np.stack([f["gravity_b"]    for f in frames]),
        "actions":      np.stack([f["actions"]      for f in frames]),
        "root_state_w": np.stack([f["root_state_w"] for f in frames]),
        "cmd_vel":      np.stack([f["cmd_vel"]      for f in frames]),
        "tokens":       np.zeros((T, 64), dtype=np.float32),
        "default_angles":    NATIVE_DEFAULT_ANGLES.astype(np.float32),
        "dt":                np.array([RECORD_DT], dtype=np.float32),
        "cmd_buckets":       np.array([args.target_vel], dtype=np.float32),
        "cmd_hold_seconds":  np.array([args.duration], dtype=np.float32),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    pos_x = out["root_state_w"][:, 0]
    vx_avg = (pos_x[-1] - pos_x[0]) / (T * RECORD_DT)
    print(f"\n[record] saved {T} frames → {out_path}", flush=True)
    print(f"[record] duration: {T * RECORD_DT:.1f}s  avg vx: {vx_avg:+.3f} m/s", flush=True)
    print(f"[record] NOTE: tokens are zeros — run add_tokens_to_dataset.py in robo_residual venv", flush=True)


if __name__ == "__main__":
    main()
