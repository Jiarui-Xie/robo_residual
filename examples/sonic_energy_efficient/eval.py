"""Evaluate a trained residual checkpoint.

Usage:
    python eval.py --decoder-onnx models/model_decoder.onnx \
                   --checkpoint runs/sonic_energy/checkpoint_best.pt \
                   --num-episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from robo_residual import ResidualActorCritic

try:
    from .configs.g1_joints import OBS_LAYOUT
    from .configs.sonic_residual_config import build_sonic_residual_config
    from .configs.train_config import TrainConfig
    from .env.g1_energy_env import DummyG1Env
    from .env.g1_energy_env_cfg import G1EnergyEnvCfg
    from .env.sonic_encoder_wrapper import ZeroTokenProvider
except ImportError:
    from configs.g1_joints import OBS_LAYOUT
    from configs.sonic_residual_config import build_sonic_residual_config
    from configs.train_config import TrainConfig
    from env.g1_energy_env import DummyG1Env
    from env.g1_energy_env_cfg import G1EnergyEnvCfg
    from env.sonic_encoder_wrapper import ZeroTokenProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SONIC residual checkpoint")
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Rebuild policy
    residual_cfg = build_sonic_residual_config()
    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=residual_cfg,
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)

    policy.residual.load_state_dict(ckpt["residual_state_dict"])
    policy.log_std.data = ckpt["log_std"]

    # Environment (dummy for now)
    train_cfg = TrainConfig(num_envs=1)
    env_cfg = G1EnergyEnvCfg(train=train_cfg)
    token_provider = ZeroTokenProvider(device=device)
    env = DummyG1Env(env_cfg, token_provider, device=device)

    print(f"[eval] Loaded checkpoint: iter={ckpt['iteration']}, "
          f"train_reward={ckpt.get('mean_reward', 'N/A')}")
    print(f"[eval] Running {args.num_episodes} episodes, max {args.max_steps} steps each\n")

    episode_rewards = []
    episode_lengths = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(args.max_steps):
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards.item()
            steps += 1
            if dones.any():
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    print(f"\n[eval] Mean reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"[eval] Mean length: {sum(episode_lengths) / len(episode_lengths):.0f}")


if __name__ == "__main__":
    main()
