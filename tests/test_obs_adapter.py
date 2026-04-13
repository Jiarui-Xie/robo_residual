from __future__ import annotations

import torch
import pytest

from robo_residual.adapters.obs_adapter import ObsAdapter


class TestObsAdapter:
    def test_single_group(self, device):
        adapter = ObsAdapter({"policy": ["obs1"], "critic": ["obs1"]})
        obs = {"obs1": torch.randn(4, 64, device=device)}
        actor_obs = adapter.get_actor_obs(obs)
        assert actor_obs.shape == (4, 64)

    def test_multiple_groups_concat(self, device):
        adapter = ObsAdapter({
            "policy": ["proprio", "terrain"],
            "critic": ["proprio", "terrain", "privileged"],
        })
        obs = {
            "proprio": torch.randn(4, 48, device=device),
            "terrain": torch.randn(4, 100, device=device),
            "privileged": torch.randn(4, 20, device=device),
        }
        actor_obs = adapter.get_actor_obs(obs)
        critic_obs = adapter.get_critic_obs(obs)
        assert actor_obs.shape == (4, 148)   # 48 + 100
        assert critic_obs.shape == (4, 168)  # 48 + 100 + 20

    def test_different_actor_critic_groups(self, device):
        adapter = ObsAdapter({
            "policy": ["policy"],
            "critic": ["critic"],
        })
        obs = {
            "policy": torch.ones(2, 10, device=device),
            "critic": torch.ones(2, 20, device=device) * 2,
        }
        actor_obs = adapter.get_actor_obs(obs)
        critic_obs = adapter.get_critic_obs(obs)
        assert actor_obs.shape == (2, 10)
        assert critic_obs.shape == (2, 20)
        assert (actor_obs == 1).all()
        assert (critic_obs == 2).all()
