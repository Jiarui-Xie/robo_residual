"""IsaacLab environment wrapper for G1 energy-efficient residual training.

This module provides a ``G1EnergyEnv`` that:
1. Wraps an IsaacLab ``ManagerBasedRLEnv`` for the Unitree G1.
2. Builds the 436-D SONIC decoder observation via ``SonicObsBuilder``.
3. Computes reward terms from ``rewards.py``.
4. Provides the standard ``reset()`` / ``step()`` interface expected by
   the training loop.

If IsaacLab is not installed, import this module to get the
``G1EnergyEnvCfg`` and protocol definitions; instantiation will raise
a clear error.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from ..configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
from ..configs.reward_config import RewardConfig
from .g1_energy_env_cfg import G1EnergyEnvCfg
from .rewards import (
    action_smoothness,
    angular_velocity_penalty,
    base_height_tracking,
    compute_total_reward,
    energy_consumption,
    foot_slip,
    residual_magnitude,
    velocity_tracking,
)
from .sonic_obs_builder import SonicObsBuilder


class TokenProvider(Protocol):
    """Anything that can produce a (N, 64) token tensor."""

    def get_token(
        self, num_envs: int, encoder_obs: Tensor | None = None
    ) -> Tensor: ...


class G1EnergyEnv:
    """Vectorized environment for G1 residual training.

    Wraps IsaacLab simulation and translates its raw sensor output into
    SONIC's 436-D decoder observation format.
    """

    def __init__(
        self,
        cfg: G1EnergyEnvCfg,
        token_provider: TokenProvider,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.train.num_envs
        self.num_obs = OBS_LAYOUT.total_dim   # 436
        self.num_actions = NUM_JOINTS          # 29

        self._token_provider = token_provider
        self._obs_builder = SonicObsBuilder(self.num_envs, device=device)
        self._prev_actions = torch.zeros(
            self.num_envs, NUM_JOINTS, device=device
        )

        # Command buffer: (N, 3) = [vx, vy, yaw_rate]
        self._commands = torch.zeros(self.num_envs, 3, device=device)

        # Track step count per env for command resampling
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._resample_steps = int(
            cfg.command.resample_interval_s / (cfg.sim.dt * cfg.sim.decimation)
        )

        self._isaac_env = None  # lazily initialized

    # ----- IsaacLab backend ------------------------------------------------ #

    def _init_isaac(self) -> None:
        """Initialize the IsaacLab simulation backend.

        Raises ImportError with setup instructions if IsaacLab is not available.
        """
        try:
            from isaaclab.envs import ManagerBasedRLEnv
        except ImportError as e:
            raise ImportError(
                "IsaacLab is required to run simulation. "
                "Install from: https://github.com/isaac-sim/IsaacLab\n"
                "For testing without IsaacLab, use DummyG1Env instead."
            ) from e

        # TODO: register the actual IsaacLab env config here
        # self._isaac_env = ManagerBasedRLEnv(cfg=...)
        raise NotImplementedError(
            "IsaacLab integration pending. Use DummyG1Env for unit testing."
        )

    # ----- command sampling ------------------------------------------------ #

    def _resample_commands(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        vx_lo, vx_hi = self.cfg.command.vel_x_range
        vy_lo, vy_hi = self.cfg.command.vel_y_range
        yr_lo, yr_hi = self.cfg.command.yaw_rate_range

        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(vx_lo, vx_hi)
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(vy_lo, vy_hi)
        self._commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(yr_lo, yr_hi)

    # ----- public interface ------------------------------------------------ #

    def reset(self) -> Tensor:
        """Reset all environments and return initial observation (N, 436)."""
        self._obs_builder.reset()
        self._prev_actions.zero_()
        self._step_count.zero_()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        token = self._token_provider.get_token(self.num_envs)
        return self._obs_builder.build(token)

    def step(
        self,
        actions: Tensor,
        # Sensor data — from IsaacLab or injected for testing
        base_lin_vel: Tensor | None = None,
        base_ang_vel: Tensor | None = None,
        joint_pos: Tensor | None = None,
        joint_vel: Tensor | None = None,
        torques: Tensor | None = None,
        base_height: Tensor | None = None,
        foot_vel: Tensor | None = None,
        contact_forces: Tensor | None = None,
        gravity: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Step the environment.

        When sensor data kwargs are ``None``, they should come from the
        IsaacLab backend (not yet connected).  Providing them explicitly
        allows testing the reward/obs pipeline in isolation.

        Returns:
            obs (N, 436), rewards (N,), dones (N,), info dict.
        """
        # -- bookkeeping --
        self._step_count += 1

        # Resample commands periodically
        needs_resample = (self._step_count % self._resample_steps == 0)
        if needs_resample.any():
            self._resample_commands(needs_resample.nonzero(as_tuple=False).squeeze(-1))

        # -- zeros fallback for testing --
        N = self.num_envs
        if base_lin_vel is None:
            base_lin_vel = torch.zeros(N, 3, device=self.device)
        if base_ang_vel is None:
            base_ang_vel = torch.zeros(N, 3, device=self.device)
        if joint_pos is None:
            joint_pos = torch.zeros(N, NUM_JOINTS, device=self.device)
        if joint_vel is None:
            joint_vel = torch.zeros(N, NUM_JOINTS, device=self.device)
        if torques is None:
            torques = torch.zeros(N, NUM_JOINTS, device=self.device)
        if base_height is None:
            base_height = torch.full((N,), self.cfg.reward.target_base_height, device=self.device)
        if foot_vel is None:
            foot_vel = torch.zeros(N, 2, 3, device=self.device)  # 2 feet
        if contact_forces is None:
            contact_forces = torch.zeros(N, 2, device=self.device)
        if gravity is None:
            gravity = torch.zeros(N, 3, device=self.device)
            gravity[:, 2] = -1.0
        if dones is None:
            dones = torch.zeros(N, dtype=torch.bool, device=self.device)

        # -- update observation history --
        self._obs_builder.update(base_ang_vel, joint_pos, joint_vel, actions, gravity)

        # -- build observation --
        token = self._token_provider.get_token(N)
        obs = self._obs_builder.build(token)

        # -- compute rewards --
        rcfg = self.cfg.reward
        components = {
            "velocity_tracking": velocity_tracking(
                base_lin_vel, self._commands, rcfg.velocity_tracking_sigma
            ),
            "energy_penalty": energy_consumption(torques, joint_vel),
            "ang_vel_penalty": angular_velocity_penalty(base_ang_vel),
            "foot_slip_penalty": foot_slip(foot_vel, contact_forces),
            "action_smoothness": action_smoothness(actions, self._prev_actions),
            "base_height_penalty": base_height_tracking(
                base_height, rcfg.target_base_height
            ),
            "alive_bonus": torch.full((N,), 1.0, device=self.device),
        }

        weights = {
            "velocity_tracking": rcfg.velocity_tracking,
            "energy_penalty": rcfg.energy_penalty,
            "ang_vel_penalty": rcfg.ang_vel_penalty,
            "foot_slip_penalty": rcfg.foot_slip_penalty,
            "action_smoothness": rcfg.action_smoothness,
            "base_height_penalty": rcfg.base_height_penalty,
            "alive_bonus": rcfg.alive_bonus,
        }

        rewards = compute_total_reward(components, weights)

        # -- termination --
        fallen = base_height < rcfg.target_base_height * 0.4
        tilt_limit = math.radians(self.cfg.termination.max_body_tilt_deg)
        excessive_tilt = torch.norm(base_ang_vel, dim=-1) > tilt_limit  # simplified
        dones = dones | fallen | excessive_tilt

        # -- reset envs that are done --
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self._obs_builder.reset(done_ids)
            self._prev_actions[done_ids] = 0.0
            self._step_count[done_ids] = 0
            self._resample_commands(done_ids)

        self._prev_actions = actions.clone()

        info = {"reward_components": components, "commands": self._commands.clone()}
        return obs, rewards, dones, info


class DummyG1Env:
    """Lightweight stand-in for testing without IsaacLab.

    Returns random sensor data so the full training pipeline can be
    exercised offline.
    """

    def __init__(
        self,
        cfg: G1EnergyEnvCfg,
        token_provider: TokenProvider,
        device: str = "cuda",
    ) -> None:
        self._env = G1EnergyEnv(cfg, token_provider, device)
        self.num_envs = self._env.num_envs
        self.num_obs = self._env.num_obs
        self.num_actions = self._env.num_actions
        self.device = device

    def reset(self) -> Tensor:
        return self._env.reset()

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Step with random sensor data."""
        N = self.num_envs
        d = self.device
        return self._env.step(
            actions,
            base_lin_vel=torch.randn(N, 3, device=d) * 0.5 + torch.tensor([2.5, 0, 0], device=d),
            base_ang_vel=torch.randn(N, 3, device=d) * 0.1,
            joint_pos=torch.randn(N, NUM_JOINTS, device=d) * 0.1,
            joint_vel=torch.randn(N, NUM_JOINTS, device=d) * 0.5,
            torques=torch.randn(N, NUM_JOINTS, device=d) * 10,
            base_height=torch.full((N,), 0.72, device=d) + torch.randn(N, device=d) * 0.05,
            foot_vel=torch.randn(N, 2, 3, device=d) * 0.1,
            contact_forces=torch.abs(torch.randn(N, 2, device=d)) * 100,
            gravity=torch.tensor([[0.0, 0.0, -1.0]], device=d).expand(N, -1),
        )
