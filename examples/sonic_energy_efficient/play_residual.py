"""Visualize a trained residual policy in IsaacLab with a live viewport.

Loads a checkpoint.pt or a fused_*.onnx. In checkpoint mode the residual is
re-attached to the ONNX base decoder and runs in PyTorch; in fused mode the
single ONNX model is used end-to-end (same numerics as deployment).

Usage:
    OMNI_KIT_ACCEPT_EULA=yes python -u examples/sonic_energy_efficient/play_residual.py \
        --decoder-onnx examples/sonic_energy_efficient/models/model_decoder.onnx \
        --encoder-onnx examples/sonic_energy_efficient/models/model_encoder_dyn.onnx \
        --planner-onnx examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx \
        --checkpoint runs/sonic_energy_3p0_v9/checkpoint_best.pt \
        --cmd-vel 3.0
"""
from __future__ import annotations
import argparse, sys

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", required=True,
                   help="Base SONIC decoder ONNX (ignored if --fused is set)")
    p.add_argument("--encoder-onnx", default=None)
    p.add_argument("--planner-onnx", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Residual checkpoint .pt to load on top of base decoder")
    p.add_argument("--fused", default=None,
                   help="Pre-fused ONNX (base+residual). Overrides --checkpoint.")
    p.add_argument("--cmd-vel", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=9999)
    p.add_argument("--warmup-seconds", type=float, default=3.0)
    args = p.parse_args()

    if args.checkpoint is None and args.fused is None:
        raise SystemExit("Must provide either --checkpoint or --fused")

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

    decoder_path = args.fused if args.fused else args.decoder_onnx
    policy = ResidualActorCritic(
        onnx_path=decoder_path,
        config=build_sonic_residual_config(),
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)
    policy.eval()

    if args.checkpoint and not args.fused:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.residual.load_state_dict(ckpt["residual_state_dict"])
        iter_num = ckpt.get("iteration", "?")
        mean_rew = ckpt.get("mean_reward", float("nan"))
        print(f"[play_residual] Loaded residual from checkpoint iter={iter_num} rew={mean_rew:+.3f}")
    elif args.fused:
        print(f"[play_residual] Running fused ONNX (residual baked in, no external loading)")

    obs = bridge.reset()
    robot = env.scene["robot"]
    dt = env.step_dt

    warmup_steps = int(args.warmup_seconds / dt)
    print(f"\n[play_residual] Warming up {warmup_steps} steps ({args.warmup_seconds:.1f}s)...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            actions = policy.act_inference(obs)
            obs, _, dones, _ = bridge.step(actions)
            if dones.any():
                obs = bridge.reset()

    WINDOW = 50
    pos_history = []

    print(f"\n[play_residual] cmd_vel={args.cmd_vel:.1f} m/s  step_dt={dt:.3f}s")
    print(f"[play_residual] Press Ctrl+C to quit\n")
    print(f"{'t(s)':>6}  {'cmd':>5}  {'vx(d/t)':>9}  {'vy':>6}  {'h':>6}  {'wz':>6}  {'|res|':>7}")
    print("-" * 62)

    with torch.no_grad():
        for t in range(args.steps):
            actions = policy.act_inference(obs)
            res_mag = policy.last_residual_delta.abs().mean().item() if policy.last_residual_delta is not None else 0.0
            obs, _, dones, _ = bridge.step(actions, policy.last_residual_delta)

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
                print(f"{t*dt:6.1f}  {cmd[0].item():+5.2f}  {vx_dt:+9.3f}  {vy_dt:+6.3f}  "
                      f"{h:6.3f}  {ang_vel_b[2].item():+6.3f}  {res_mag:7.4f}")

            if dones.any():
                print(f"\n[play_residual] episode ended at t={t*dt:.1f}s, resetting...")
                pos_history.clear()
                obs = bridge.reset()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
