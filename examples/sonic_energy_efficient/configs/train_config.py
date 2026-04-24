"""PPO training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # -- environment --
    num_envs: int = 4096
    episode_length_s: float = 10.0
    sim_dt: float = 0.005          # 200 Hz physics
    decimation: int = 4            # policy at 50 Hz (matches SONIC deploy)

    # -- PPO --
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95              # GAE lambda

    num_steps_per_env: int = 24    # rollout length
    num_mini_batches: int = 4
    num_epochs: int = 5
    total_timesteps: int = 50_000_000

    # -- checkpointing --
    save_interval: int = 500       # iterations
    log_interval: int = 10         # iterations
    eval_interval: int = 200       # iterations

    # -- device --
    device: str = "cuda"
