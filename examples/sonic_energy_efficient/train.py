"""Train a residual MLP on top of SONIC decoder for energy-efficient gait.

Usage:
    # With IsaacLab (full simulation):
    python train.py --decoder-onnx models/model_decoder.onnx \
                    --encoder-onnx models/model_encoder.onnx

    # Dry-run without IsaacLab (random sensor data):
    python train.py --decoder-onnx models/model_decoder.onnx --dummy-env
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from robo_residual import ResidualActorCritic

try:
    from .configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from .configs.reward_config import RewardConfig
    from .configs.sonic_residual_config import build_sonic_residual_config
    from .configs.train_config import TrainConfig
    from .env.g1_energy_env import DummyG1Env, G1EnergyEnv
    from .env.g1_energy_env_cfg import G1EnergyEnvCfg
    from .env.sonic_encoder_wrapper import SonicEncoderWrapper, ZeroTokenProvider
except ImportError:
    from configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from configs.reward_config import RewardConfig
    from configs.sonic_residual_config import build_sonic_residual_config
    from configs.train_config import TrainConfig
    from env.g1_energy_env import DummyG1Env, G1EnergyEnv
    from env.g1_energy_env_cfg import G1EnergyEnvCfg
    from env.sonic_encoder_wrapper import SonicEncoderWrapper, ZeroTokenProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SONIC residual training")
    p.add_argument("--decoder-onnx", type=str, required=True,
                   help="Path to model_decoder.onnx")
    p.add_argument("--encoder-onnx", type=str, default=None,
                   help="Path to model_encoder.onnx (optional; uses zero token if omitted)")
    p.add_argument("--output-dir", type=str, default="runs/sonic_energy",
                   help="Directory for checkpoints and logs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dummy-env", action="store_true",
                   help="Use DummyG1Env (random data) instead of IsaacLab")
    p.add_argument("--num-envs", type=int, default=None,
                   help="Override number of parallel envs")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Override total training timesteps")
    return p.parse_args()


def compute_gae(
    rewards: torch.Tensor,     # (T, N)
    values: torch.Tensor,      # (T, N)
    dones: torch.Tensor,       # (T, N)
    next_value: torch.Tensor,  # (N,)
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Returns:
        advantages (T, N), returns (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: ResidualActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,       # (B, obs_dim)
    actions_batch: torch.Tensor,   # (B, act_dim)
    log_probs_old: torch.Tensor,   # (B,)
    advantages: torch.Tensor,      # (B,)
    returns: torch.Tensor,         # (B,)
    clip_range: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Single PPO epoch on a mini-batch."""
    # Forward pass
    actions_pred = policy.act(obs_batch)
    log_probs_new = policy.get_actions_log_prob(actions_batch)
    values_pred = policy.evaluate(obs_batch).squeeze(-1)
    entropy = policy.entropy.mean()

    # Policy loss (clipped surrogate)
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = ((values_pred - returns) ** 2).mean()

    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.trainable_parameters, max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": loss.item(),
    }


def main() -> None:
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- configs --
    train_cfg = TrainConfig()
    if args.num_envs is not None:
        train_cfg.num_envs = args.num_envs
    if args.total_timesteps is not None:
        train_cfg.total_timesteps = args.total_timesteps

    env_cfg = G1EnergyEnvCfg(train=train_cfg)
    residual_cfg = build_sonic_residual_config()

    # -- token provider --
    if args.encoder_onnx:
        token_provider = SonicEncoderWrapper(args.encoder_onnx, device=device, mode="cached")
    else:
        token_provider = ZeroTokenProvider(device=device)

    # -- environment --
    if args.dummy_env:
        env = DummyG1Env(env_cfg, token_provider, device=device)
        print("[train] Using DummyG1Env (random sensor data)")
    else:
        env = G1EnergyEnv(env_cfg, token_provider, device=device)
        print("[train] Using IsaacLab G1 environment")

    # -- residual actor-critic --
    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=residual_cfg,
        num_actor_obs=OBS_LAYOUT.total_dim,   # 436
        num_critic_obs=OBS_LAYOUT.total_dim,  # same obs for critic
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(policy.trainable_parameters, lr=train_cfg.learning_rate)

    # -- training loop --
    num_steps = train_cfg.num_steps_per_env
    num_envs = train_cfg.num_envs
    total_iters = train_cfg.total_timesteps // (num_steps * num_envs)

    print(f"[train] Residual params: {sum(p.numel() for p in policy.trainable_parameters):,}")
    print(f"[train] Total iterations: {total_iters:,} "
          f"({num_steps} steps × {num_envs} envs × {total_iters} iters = "
          f"{num_steps * num_envs * total_iters:,} timesteps)")
    print(f"[train] Output dir: {output_dir}")

    obs = env.reset()  # (N, 436)
    best_mean_reward = float("-inf")

    for iteration in range(1, total_iters + 1):
        t_start = time.time()

        # -- rollout collection --
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        with torch.no_grad():
            for _ in range(num_steps):
                actions = policy.act(obs)
                log_prob = policy.get_actions_log_prob(actions)
                value = policy.evaluate(obs).squeeze(-1)

                next_obs, rewards, dones, info = env.step(actions)

                rollout_obs.append(obs)
                rollout_actions.append(actions)
                rollout_log_probs.append(log_prob)
                rollout_rewards.append(rewards)
                rollout_dones.append(dones)
                rollout_values.append(value)

                obs = next_obs

            # Bootstrap value for GAE
            next_value = policy.evaluate(obs).squeeze(-1)

        # Stack rollouts: (T, N, ...)
        rollout_obs = torch.stack(rollout_obs)
        rollout_actions = torch.stack(rollout_actions)
        rollout_log_probs = torch.stack(rollout_log_probs)
        rollout_rewards = torch.stack(rollout_rewards)
        rollout_dones = torch.stack(rollout_dones)
        rollout_values = torch.stack(rollout_values)

        # -- GAE --
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones,
            next_value, train_cfg.gamma, train_cfg.lam,
        )

        # Flatten: (T*N, ...)
        flat_obs = rollout_obs.reshape(-1, OBS_LAYOUT.total_dim)
        flat_actions = rollout_actions.reshape(-1, NUM_JOINTS)
        flat_log_probs = rollout_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # -- PPO update --
        batch_size = num_steps * num_envs
        mini_batch_size = batch_size // train_cfg.num_mini_batches

        epoch_losses = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        num_updates = 0

        for _ in range(train_cfg.num_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                losses = ppo_update(
                    policy, optimizer,
                    flat_obs[idx], flat_actions[idx], flat_log_probs[idx],
                    flat_advantages[idx], flat_returns[idx],
                    train_cfg.clip_range, train_cfg.entropy_coef,
                    train_cfg.value_coef, train_cfg.max_grad_norm,
                )
                for k in epoch_losses:
                    epoch_losses[k] += losses[k]
                num_updates += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(num_updates, 1)

        elapsed = time.time() - t_start
        fps = (num_steps * num_envs) / elapsed
        mean_reward = rollout_rewards.mean().item()

        # -- logging --
        if iteration % train_cfg.log_interval == 0:
            print(
                f"[iter {iteration:>6d}/{total_iters}] "
                f"reward={mean_reward:+.4f}  "
                f"ploss={epoch_losses['policy_loss']:.4f}  "
                f"vloss={epoch_losses['value_loss']:.4f}  "
                f"ent={epoch_losses['entropy']:.4f}  "
                f"fps={fps:.0f}"
            )

        # -- checkpointing --
        if iteration % train_cfg.save_interval == 0 or mean_reward > best_mean_reward:
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                tag = "best"
            else:
                tag = f"iter_{iteration}"

            ckpt = {
                "iteration": iteration,
                "residual_state_dict": policy.residual.state_dict(),
                "critic_state_dict": policy.critic.state_dict(),
                "log_std": policy.log_std.data,
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": mean_reward,
                "config": {
                    "decoder_onnx": args.decoder_onnx,
                    "residual_hidden_dims": residual_cfg.residual_hidden_dims,
                    "joint_groups": [
                        {"name": g.name, "indices": g.indices, "max_residual": g.max_residual}
                        for g in residual_cfg.joint_groups
                    ],
                },
            }
            torch.save(ckpt, output_dir / f"checkpoint_{tag}.pt")

    print(f"\n[train] Done. Best mean reward: {best_mean_reward:.4f}")
    print(f"[train] Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
