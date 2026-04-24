"""Trapezoid velocity command rollout with online planner.

Each env runs: 1.5s ramp-up → 10s steady (random 2–3 m/s) → 1.5s ramp-down → 1s idle.
Injects cmd into IsaacLab command_manager every step AND updates online planner's
per-env target_vel. Logs per-phase velocity tracking.

Usage:
    OMNI_KIT_ACCEPT_EULA=yes DISPLAY=:0 python -u -m sonic_energy_efficient.rollout_trapezoid \
        --decoder-onnx sonic_energy_efficient/models/model_decoder.onnx \
        --encoder-onnx sonic_energy_efficient/models/model_encoder_dyn.onnx \
        --planner-onnx sonic_energy_efficient/models/planner_sonic.onnx \
        --num-envs 4 --headless false
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--encoder-onnx", type=str, required=True)
    p.add_argument("--planner-onnx", type=str, required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--ramp-up-s", type=float, default=1.5)
    p.add_argument("--steady-s", type=float, default=10.0)
    p.add_argument("--ramp-down-s", type=float, default=1.5)
    p.add_argument("--idle-s", type=float, default=1.0)
    p.add_argument("--vel-min", type=float, default=2.0)
    p.add_argument("--vel-max", type=float, default=3.0)
    p.add_argument("--headless", type=lambda s: s.lower() in ("true","1","yes"), default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    sys.path.insert(0, "/home/lumi/robo_residual")
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
    env_cfg.scene.num_envs = args.num_envs
    total_s = args.ramp_up_s + args.steady_s + args.ramp_down_s + args.idle_s
    env_cfg.episode_length_s = total_s + 1.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=args.planner_onnx,
        encoder_onnx_dyn=args.encoder_onnx,
        num_envs=args.num_envs,
        target_vel=(args.vel_min + args.vel_max) / 2,
        device=device,
        replan_interval_steps=25,
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

    # Sample steady velocity per env
    rng = np.random.default_rng(args.seed)
    steady_vel = rng.uniform(args.vel_min, args.vel_max, size=args.num_envs).astype(np.float32)
    print(f"[trap] Per-env steady velocities: {steady_vel.tolist()}")

    # Phase boundaries in steps (50 Hz)
    STEP_DT = 0.02
    ramp_up_steps = int(args.ramp_up_s / STEP_DT)
    steady_end_step = ramp_up_steps + int(args.steady_s / STEP_DT)
    ramp_down_end = steady_end_step + int(args.ramp_down_s / STEP_DT)
    idle_end = ramp_down_end + int(args.idle_s / STEP_DT)
    total_steps = idle_end
    print(f"[trap] Phases (50 Hz): ramp-up 0..{ramp_up_steps}, steady ..{steady_end_step}, ramp-down ..{ramp_down_end}, idle ..{idle_end}")

    def cmd_at_step(t: int) -> np.ndarray:
        """Return (num_envs,) x-velocity command for step t."""
        vx = np.zeros(args.num_envs, dtype=np.float32)
        if t < ramp_up_steps:
            w = t / ramp_up_steps
            vx = steady_vel * w
        elif t < steady_end_step:
            vx = steady_vel.copy()
        elif t < ramp_down_end:
            w = 1.0 - (t - steady_end_step) / (ramp_down_end - steady_end_step)
            vx = steady_vel * w
        # else idle → zeros
        return vx

    obs = bridge.reset()
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("base_velocity")
    print(f"[trap] cmd term type: {type(cmd_term).__name__}, command buf shape: {cmd_term.command.shape}")

    # Per-env velocity tracking log
    logs = []
    phase_names = ["ramp_up", "steady", "ramp_down", "idle"]

    with torch.no_grad():
        for t in range(total_steps):
            # Compute trapezoid cmd for this step
            vx_cmd = cmd_at_step(t)

            # 1. Push cmd into IsaacLab command manager (overrides stochastic sampling)
            cmd_term.command[:, 0] = torch.from_numpy(vx_cmd).to(device)
            cmd_term.command[:, 1] = 0.0
            cmd_term.command[:, 2] = 0.0

            # 2. Update online planner per-env target velocity
            token_provider.set_target_vel(
                np.arange(args.num_envs), np.maximum(vx_cmd, 0.1)  # planner doesn't like 0
            )

            # 3. Policy step (residual = 0 → pure decoder baseline)
            actions = policy.act_inference(obs)
            obs, rewards, dones, info = bridge.step(actions)

            if t % 25 == 0 or t == total_steps - 1:
                phase = "ramp_up" if t < ramp_up_steps else \
                        "steady" if t < steady_end_step else \
                        "ramp_down" if t < ramp_down_end else "idle"
                lin_vx = robot.data.root_lin_vel_b[:, 0].cpu().numpy()
                base_h = robot.data.root_pos_w[:, 2].cpu().numpy()
                print(f"[t={t:4d} {t*STEP_DT:5.2f}s phase={phase:9s}] cmd={vx_cmd.round(2).tolist()} actual={lin_vx.round(2).tolist()} h={base_h.round(2).tolist()}")
                logs.append((t, phase, vx_cmd.copy(), lin_vx.copy()))

            if dones.any():
                print(f"[t={t}] env reset (dones={dones.cpu().numpy()})")

    # Summary: mean tracking error per phase
    print("\n=== phase-wise velocity tracking summary ===")
    for ph in phase_names:
        errs = []
        for t, phase, cmd, act in logs:
            if phase == ph:
                errs.append(np.abs(cmd - act).mean())
        if errs:
            print(f"  {ph:9s}: {len(errs)} samples, |cmd-actual| mean = {np.mean(errs):.3f} m/s")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
