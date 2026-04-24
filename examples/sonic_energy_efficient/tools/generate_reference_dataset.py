"""Generate offline reference dataset for RSI training.

Runs the SONIC online planner in IsaacLab (num_envs=1) across a sweep of
command velocities. Saves every 50Hz frame with everything needed to
re-initialize a training episode from that state:

  - Robot state (for IsaacLab RSI writeback):
      jpos_abs     (29,)  : joint positions (native order, absolute)
      jvel         (29,)  : joint velocities (native order)
      root_state_w (13,)  : base pos (3) + quat wxyz (4) + linvel_w (3) + angvel_w (3)

  - Obs-builder history ingredients (for SonicObsBuilder warm-start):
      jpos_delta   (29,)  : jpos_abs - default_angles (decoder expects delta)
      ang_vel_b    (3,)
      gravity_b    (3,)
      action       (29,)  : action that led to this post-step state

  - Pre-computed token (skips encoder at training time, 10-100x faster):
      token        (64,)

  - Command velocity sampled at this frame:
      cmd_vel      (3,)

Usage:
    cd examples/sonic_energy_efficient
    OMNI_KIT_ACCEPT_EULA=yes conda run -n robo --no-capture-output \\
      python -u tools/generate_reference_dataset.py \\
        --decoder-onnx models/model_decoder.onnx \\
        --planner-onnx models/planner_sonic_dyn.onnx \\
        --encoder-onnx models/model_encoder_dyn.onnx \\
        --output models/reference_dataset.npz \\
        --duration 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--planner-onnx", type=str, required=True)
    p.add_argument("--encoder-onnx", type=str, required=True)
    p.add_argument("--output", type=str, required=True,
                   help="Output .npz path")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Seconds of simulated time to record (50Hz → N frames)")
    p.add_argument("--warmup-seconds", type=float, default=3.0,
                   help="Drop this many seconds at the start (cold-start ramp)")
    p.add_argument("--cmd-buckets", type=float, nargs="+",
                   default=[1.5, 2.0, 2.5, 3.0],
                   help="Command velocities to sweep (m/s)")
    p.add_argument("--cmd-hold-seconds", type=float, default=15.0,
                   help="Seconds to hold each cmd before switching")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Preload CUDA libs before onnxruntime
    sys.path.insert(0, "/home/lumi/robo_residual")
    import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

    # ---- Start IsaacSim ----
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    # ---- Imports (require app running) ----
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from robo_residual import ResidualActorCritic

    from examples.sonic_energy_efficient.configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from examples.sonic_energy_efficient.configs.reward_config import RewardConfig
    from examples.sonic_energy_efficient.configs.sonic_params import NATIVE_DEFAULT_ANGLES
    from examples.sonic_energy_efficient.configs.sonic_residual_config import build_sonic_residual_config
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg
    from examples.sonic_energy_efficient.env.isaaclab_sonic_bridge import IsaacLabSonicBridge
    from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder

    device = "cuda"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dt = 0.02  # 50 Hz
    total_steps = int(args.duration / dt)
    warmup_steps = int(args.warmup_seconds / dt)
    hold_steps = int(args.cmd_hold_seconds / dt)
    record_steps = total_steps - warmup_steps

    print(f"[dataset-gen] Duration: {args.duration}s ({total_steps} steps @ 50Hz)")
    print(f"[dataset-gen] Warmup drop: {args.warmup_seconds}s ({warmup_steps} steps)")
    print(f"[dataset-gen] Recording: {record_steps} frames")
    print(f"[dataset-gen] Cmd buckets: {args.cmd_buckets}, hold {args.cmd_hold_seconds}s each")

    # ---- Env (num_envs=1, online planner is batch=1) ----
    env_cfg = G1_29DOF_Sonic_FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 5.0  # avoid auto-reset during generation
    # Disable command auto-resampling so our manual set_cmd() holds exactly.
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[dataset-gen] IsaacLab env ready, device={isaac_env.device}")

    # ---- Online planner (will auto-pick initial target_vel, overridden below) ----
    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=args.planner_onnx,
        encoder_onnx_dyn=args.encoder_onnx,
        num_envs=1,
        target_vel=args.cmd_buckets[0],
        replan_interval_steps=5,        # deploy faithful: 10Hz planner @ 50Hz sim
        motion_look_ahead_steps=5,
        device=device,
    )

    bridge = IsaacLabSonicBridge(isaac_env, token_provider, RewardConfig(), device=device)

    # ---- Policy: pure SONIC base (zero residual, no noise) ----
    residual_cfg = build_sonic_residual_config()
    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=residual_cfg,
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)

    # ---- Storage (pre-allocate) ----
    jpos_abs     = np.zeros((record_steps, 29), dtype=np.float32)
    jpos_delta   = np.zeros((record_steps, 29), dtype=np.float32)
    jvel         = np.zeros((record_steps, 29), dtype=np.float32)
    ang_vel_b    = np.zeros((record_steps, 3), dtype=np.float32)
    gravity_b    = np.zeros((record_steps, 3), dtype=np.float32)
    actions      = np.zeros((record_steps, 29), dtype=np.float32)
    root_state_w = np.zeros((record_steps, 13), dtype=np.float32)
    tokens       = np.zeros((record_steps, 64), dtype=np.float32)
    cmd_vel      = np.zeros((record_steps, 3), dtype=np.float32)

    default_angles_np = np.array(NATIVE_DEFAULT_ANGLES, dtype=np.float32)

    # ---- Helper: override cmd velocity ----
    def set_cmd(vx: float) -> None:
        """Inject velocity command into both the IsaacLab command manager and
        the online planner's target_vel."""
        # IsaacLab command manager
        cm = isaac_env.command_manager.get_term("base_velocity")
        cm.vel_command_b[:, 0] = vx
        cm.vel_command_b[:, 1] = 0.0
        cm.vel_command_b[:, 2] = 0.0
        # Online planner
        token_provider.target_vel[:] = vx

    # ---- Run ----
    obs = bridge.reset()
    set_cmd(args.cmd_buckets[0])

    t0 = time.time()
    robot = isaac_env.scene["robot"]

    for step in range(total_steps):
        # Cmd schedule: step through buckets, holding each for hold_steps
        bucket_idx = (step // hold_steps) % len(args.cmd_buckets)
        current_cmd = args.cmd_buckets[bucket_idx]
        if step > 0 and step % hold_steps == 0:
            set_cmd(current_cmd)

        with torch.no_grad():
            action = policy.act_inference(obs)

        # Pre-step snapshot of token used for THIS action
        # The token has already been built into obs for this step's policy call.
        # For the saved-token to correspond to the POST-step frame we record below,
        # we query it AFTER the bridge has updated its internal state.
        obs, _, _, info = bridge.step(action)

        if step < warmup_steps:
            if step % 50 == 0:
                print(f"[dataset-gen] warmup {step}/{warmup_steps}")
            continue

        rec_idx = step - warmup_steps

        # Post-step state
        jpos_all = robot.data.joint_pos[0]  # (43,) — we take 29 via bridge perm
        jvel_all = robot.data.joint_vel[0]
        native_idx = bridge._grouped_to_native  # tensor of 29 indices
        jpos_native = jpos_all[native_idx]       # (29,) absolute
        jvel_native = jvel_all[native_idx]

        jpos_abs[rec_idx]   = jpos_native.cpu().numpy()
        jpos_delta[rec_idx] = jpos_native.cpu().numpy() - default_angles_np
        jvel[rec_idx]       = jvel_native.cpu().numpy()
        ang_vel_b[rec_idx]  = robot.data.root_ang_vel_b[0].cpu().numpy()
        gravity_b[rec_idx]  = robot.data.projected_gravity_b[0].cpu().numpy()
        actions[rec_idx]    = action[0].cpu().numpy()

        # Root state in WORLD frame for write_root_state_to_sim:
        # pos(3) + quat_wxyz(4) + linvel_w(3) + angvel_w(3) = 13
        pos_w = robot.data.root_pos_w[0]
        quat_w = robot.data.root_quat_w[0]
        linvel_w = robot.data.root_lin_vel_w[0]
        angvel_w = robot.data.root_ang_vel_w[0]
        root_state_w[rec_idx] = torch.cat([pos_w, quat_w, linvel_w, angvel_w]).cpu().numpy()

        # Token used to produce next obs — re-query from provider for record.
        token = bridge._get_token()[0]
        tokens[rec_idx] = token.cpu().numpy()

        # Record the scheduled cmd directly — get_command() returns the
        # IsaacLab-internal resampled value which doesn't reflect our manual
        # vel_command_b override and always reads as zero.
        cmd_vel[rec_idx, 0] = current_cmd
        cmd_vel[rec_idx, 1] = 0.0
        cmd_vel[rec_idx, 2] = 0.0

        if rec_idx % 500 == 0:
            vx_actual = robot.data.root_lin_vel_w[0, 0].item()
            elapsed = time.time() - t0
            sim_s = (step + 1) * dt
            print(f"[dataset-gen] frame {rec_idx}/{record_steps}  "
                  f"cmd={current_cmd:.2f} vx={vx_actual:+.3f} "
                  f"sim={sim_s:.1f}s real={elapsed:.1f}s")

    # ---- Save ----
    np.savez_compressed(
        output_path,
        jpos_abs=jpos_abs,
        jpos_delta=jpos_delta,
        jvel=jvel,
        ang_vel_b=ang_vel_b,
        gravity_b=gravity_b,
        actions=actions,
        root_state_w=root_state_w,
        tokens=tokens,
        cmd_vel=cmd_vel,
        default_angles=default_angles_np,
        dt=np.array([dt], dtype=np.float32),
        cmd_buckets=np.array(args.cmd_buckets, dtype=np.float32),
        cmd_hold_seconds=np.array([args.cmd_hold_seconds], dtype=np.float32),
    )
    print(f"[dataset-gen] Saved {record_steps} frames to {output_path}")
    print(f"[dataset-gen] File size: {output_path.stat().st_size / 1e6:.1f} MB")

    isaac_env.close()
    app.close()


if __name__ == "__main__":
    main()
