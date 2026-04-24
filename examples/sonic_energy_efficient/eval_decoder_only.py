"""Pure rollout: decoder-only (no residual, no training, no exploration noise).

Purpose: verify that SONIC decoder + our IsaacLab setup produces walking behavior.
If this fails, no amount of residual training will help.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--planner-onnx", type=str, required=True)
    p.add_argument("--encoder-onnx", type=str, required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--recorded-motion", type=str, default=None,
                   help="Path to pre-recorded motion .npz (from record_sonic_motion.py)")
    p.add_argument("--online-planner", action="store_true",
                   help="Run planner online every N steps with live robot state as context")
    p.add_argument("--replan-interval", type=int, default=25,
                   help="Control steps between replans (50Hz; default 25 = 0.5s)")
    p.add_argument("--headless", action="store_true", default=False,
                   help="Run without GUI (default: open viewer)")
    p.add_argument("--cmd-vel", type=float, default=None,
                   help="Override command velocity (constant, m/s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    import sys
    sys.path.insert(0, "/home/lumi/robo_residual")

    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from robo_residual import OnnxBasePolicy

    from examples.sonic_energy_efficient.configs.g1_joints import OBS_LAYOUT
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg
    from examples.sonic_energy_efficient.env.isaaclab_sonic_bridge import IsaacLabSonicBridge
    from examples.sonic_energy_efficient.env.sonic_live_encoder import SonicLiveEncoder
    from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder

    device = "cuda"

    env_cfg = G1_29DOF_Sonic_FlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 3.0
    env_cfg.episode_length_s = args.num_steps * 0.02 + 1.0
    # Disable corruption noise for cleaner demo
    env_cfg.observations.policy.enable_corruption = False
    # Fixed velocity if requested
    if args.cmd_vel is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.cmd_vel, args.cmd_vel)
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)

    if args.online_planner:
        target_vel = args.cmd_vel if args.cmd_vel is not None else 2.5
        token_provider = SonicOnlinePlannerEncoder(
            planner_onnx=args.planner_onnx,
            encoder_onnx_dyn=args.encoder_onnx,
            num_envs=args.num_envs,
            target_vel=target_vel,
            replan_interval_steps=args.replan_interval,
            device=device,
        )
    else:
        token_provider = SonicLiveEncoder(
            planner_onnx=args.planner_onnx,
            encoder_onnx_dyn=args.encoder_onnx,
            num_envs=args.num_envs,
            device=device,
            recorded_motion_path=args.recorded_motion,
        )

    bridge = IsaacLabSonicBridge(isaac_env, token_provider, device=device)

    decoder = OnnxBasePolicy(args.decoder_onnx, device=device)
    print(f"\n[decoder_only] Decoder loaded: {decoder.num_obs}D → {decoder.num_actions}D")

    # Record initial positions
    robot = isaac_env.scene["robot"]
    init_pos = robot.data.root_pos_w[:, :3].clone()
    print(f"[decoder_only] Initial robot pos (first env): {init_pos[0].cpu().numpy()}")

    obs = bridge.reset()
    print(f"[decoder_only] Obs shape: {obs.shape}")

    forward_distances = []
    fallen_envs = 0

    for step in range(args.num_steps):
        with torch.no_grad():
            action = decoder.forward(obs)  # decoder-only, no residual
        obs, reward, done, info = bridge.step(action)

        if step % 20 == 0 or step == args.num_steps - 1:
            cur_pos = robot.data.root_pos_w[:, :3]
            dx = (cur_pos[:, 0] - init_pos[:, 0]).cpu().numpy()
            dy = (cur_pos[:, 1] - init_pos[:, 1]).cpu().numpy()
            base_h = cur_pos[:, 2].cpu().numpy()
            cmd_v = info["commands"][:, 0].cpu().numpy()
            lin_vel = robot.data.root_lin_vel_b[:, 0].cpu().numpy()
            n_alive = (~done).sum().item()
            print(f"  step {step:>3d}: dx={dx.mean():+.2f}m (±{dx.std():.2f})  "
                  f"dy={dy.mean():+.2f}m  h={base_h.mean():.2f}m  "
                  f"v_body_x={lin_vel.mean():+.2f} (cmd {cmd_v.mean():.2f})  "
                  f"alive={n_alive}/{args.num_envs}")

    # Final stats
    cur_pos = robot.data.root_pos_w[:, :3]
    total_dx = (cur_pos[:, 0] - init_pos[:, 0]).cpu().numpy()
    duration_s = args.num_steps * 0.02  # 50 Hz
    avg_vx = total_dx.mean() / duration_s
    print(f"\n[decoder_only] RESULT after {duration_s:.1f}s:")
    print(f"  avg forward distance: {total_dx.mean():+.2f}m")
    print(f"  avg forward velocity: {avg_vx:+.2f} m/s  (command was 2-3 m/s)")
    if avg_vx > 0.3:
        print(f"  → Robot IS WALKING. SONIC base works. Residual can optimize.")
    elif avg_vx > 0.05:
        print(f"  → Marginal forward motion. Physics close but needs tuning.")
    else:
        print(f"  → NOT walking. Deeper issue (physics/obs/action mismatch).")

    isaac_env.close()
    app.close()


if __name__ == "__main__":
    main()
