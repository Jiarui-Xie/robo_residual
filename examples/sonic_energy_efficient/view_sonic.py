"""Visualize pure SONIC base policy in IsaacLab with a live viewport window.

Opens an IsaacSim window, runs the base SONIC decoder (residual = 0) at a
fixed command velocity, and prints per-frame velocity stats to the terminal.

Usage:
    cd /home/lumi/robo_residual
    OMNI_KIT_ACCEPT_EULA=yes python -u examples/sonic_energy_efficient/view_sonic.py \
        --decoder-onnx examples/sonic_energy_efficient/models/model_decoder.onnx \
        --encoder-onnx examples/sonic_energy_efficient/models/model_encoder_dyn.onnx \
        --planner-onnx examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx \
        --cmd-vel 2.0
"""
from __future__ import annotations
import argparse, sys

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", required=True)
    p.add_argument("--encoder-onnx", default=None)
    p.add_argument("--planner-onnx", default=None)
    p.add_argument("--token-cache", default=None, help="Pre-computed token cache .npz (fastest, matches training)")
    p.add_argument("--cmd-vel", type=float, default=2.0, help="Fixed forward velocity command (m/s)")
    p.add_argument("--steps", type=int, default=9999)
    args = p.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False, "width": 1280, "height": 720})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from robo_residual import ResidualActorCritic

    from examples.sonic_energy_efficient.configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from examples.sonic_energy_efficient.configs.reward_config import RewardConfig
    from examples.sonic_energy_efficient.configs.sonic_residual_config import build_sonic_residual_config
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg
    from examples.sonic_energy_efficient.env.isaaclab_sonic_bridge import IsaacLabSonicBridge
    from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder

    device = "cuda"

    env_cfg = G1_29DOF_Sonic_FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.cmd_vel, args.cmd_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (3600.0, 3600.0)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # replan_interval=5 matches generate_reference_dataset.py (deploy faithful: 10Hz @ 50Hz sim).
    # Longer intervals let tracking error accumulate and cause instability.
    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=args.planner_onnx,
        encoder_onnx_dyn=args.encoder_onnx,
        num_envs=1,
        target_vel=args.cmd_vel,
        device=device,
        replan_interval_steps=5,
        motion_look_ahead_steps=5,
    )
    bridge = IsaacLabSonicBridge(env, token_provider, RewardConfig(), device=device)

    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=build_sonic_residual_config(),
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)
    policy.eval()

    obs = bridge.reset()
    robot = env.scene["robot"]
    dt = env.step_dt

    # warmup: let planner/decoder stabilize before we care about behavior
    warmup_steps = int(3.0 / dt)
    print(f"\n[view_sonic] Warming up {warmup_steps} steps (3s)...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            actions = policy.act_inference(obs)
            obs, _, dones, _ = bridge.step(actions)
            if dones.any():
                obs = bridge.reset()

    # sliding window for distance/time velocity (50 frames = 1 s)
    WINDOW = 50
    pos_history = []

    print(f"\n[view_sonic] cmd_vel={args.cmd_vel:.1f} m/s  step_dt={dt:.3f}s")
    print(f"[view_sonic] Press Ctrl+C to quit\n")
    print(f"{'t(s)':>6}  {'cmd_vx':>7}  {'vx(d/t)':>9}  {'vy(d/t)':>9}  {'h':>6}  {'wz':>6}")
    print("-" * 58)

    with torch.no_grad():
        for t in range(args.steps):
            actions = policy.act_inference(obs)
            obs, rewards, dones, info = bridge.step(actions)
            # NOTE: no app.update() — causes double physics step; viewport refreshes via env.step()

            pos_w = robot.data.root_pos_w[0].cpu()
            pos_history.append(pos_w.clone())
            if len(pos_history) > WINDOW:
                pos_history.pop(0)

            if t % 10 == 0:
                if len(pos_history) >= WINDOW:
                    elapsed = (WINDOW - 1) * dt
                    vx_dt = (pos_history[-1][0] - pos_history[0][0]).item() / elapsed
                    vy_dt = (pos_history[-1][1] - pos_history[0][1]).item() / elapsed
                else:
                    vx_dt = vy_dt = float("nan")

                ang_vel_b = robot.data.root_ang_vel_b[0]
                h = pos_w[2].item()
                cmd = env.command_manager.get_command("base_velocity")[0]
                print(
                    f"{t*dt:6.1f}  "
                    f"{cmd[0].item():+7.2f}  "
                    f"{vx_dt:+9.3f}  "
                    f"{vy_dt:+9.3f}  "
                    f"{h:6.3f}  "
                    f"{ang_vel_b[2].item():+6.3f}"
                )

            if dones.any():
                print(f"\n[view_sonic] episode ended at t={t*dt:.1f}s, resetting...")
                pos_history.clear()
                obs = bridge.reset()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
