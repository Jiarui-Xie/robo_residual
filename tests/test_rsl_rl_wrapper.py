from __future__ import annotations

import torch
import pytest

from robo_residual.adapters.rsl_rl_wrapper import RslRlResidualActorCritic
from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig


class TestRslRlWrapper:
    @pytest.fixture
    def obs_groups(self):
        return {"policy": ["policy"], "critic": ["critic"]}

    @pytest.fixture
    def tensordict_obs(self, device):
        return {
            "policy": torch.randn(4, 64, device=device),
            "critic": torch.randn(4, 80, device=device),
        }

    @pytest.fixture
    def wrapper(self, dummy_onnx, simple_config, obs_groups, tensordict_obs, device):
        return RslRlResidualActorCritic(
            obs=tensordict_obs,
            obs_groups=obs_groups,
            onnx_path=dummy_onnx,
            config=simple_config,
            device=device,
        ).to(device)

    def test_act_with_tensordict(self, wrapper, tensordict_obs):
        actions = wrapper.act(tensordict_obs)
        assert actions.shape == (4, 29)

    def test_act_inference_with_tensordict(self, wrapper, tensordict_obs):
        actions = wrapper.act_inference(tensordict_obs)
        assert actions.shape == (4, 29)

    def test_evaluate_with_tensordict(self, wrapper, tensordict_obs):
        values = wrapper.evaluate(tensordict_obs)
        assert values.shape == (4, 1)

    def test_log_prob(self, wrapper, tensordict_obs):
        actions = wrapper.act(tensordict_obs)
        log_prob = wrapper.get_actions_log_prob(actions)
        assert log_prob.shape == (4,)
        assert torch.isfinite(log_prob).all()

    def test_is_not_recurrent(self, wrapper):
        assert not wrapper.is_recurrent

    def test_properties_accessible(self, wrapper, tensordict_obs):
        wrapper.act(tensordict_obs)
        assert wrapper.action_mean is not None
        assert wrapper.action_std is not None
        assert wrapper.entropy is not None
        assert wrapper.distribution is not None

    def test_trainable_parameters(self, wrapper):
        params = wrapper.trainable_parameters
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_residual_exposed(self, wrapper):
        assert wrapper.residual is not None
        assert wrapper.critic is not None
        assert wrapper.base is not None
        assert wrapper.max_residual_limits is not None

    def test_multi_group_actor(self, dummy_onnx, simple_config, device):
        """Actor can use multiple obs groups (e.g. proprio + terrain)."""
        obs_groups = {
            "policy": ["proprio", "extra"],
            "critic": ["proprio", "extra"],
        }
        obs = {
            "proprio": torch.randn(4, 50, device=device),
            "extra": torch.randn(4, 14, device=device),  # 50+14=64 = base obs dim
        }
        wrapper = RslRlResidualActorCritic(
            obs=obs,
            obs_groups=obs_groups,
            onnx_path=dummy_onnx,
            config=simple_config,
            device=device,
        ).to(device)
        actions = wrapper.act(obs)
        assert actions.shape == (4, 29)

    def test_with_normalization(self, dummy_onnx, simple_config, device):
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        obs = {
            "policy": torch.randn(4, 64, device=device),
            "critic": torch.randn(4, 64, device=device),
        }
        wrapper = RslRlResidualActorCritic(
            obs=obs,
            obs_groups=obs_groups,
            onnx_path=dummy_onnx,
            config=simple_config,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            device=device,
        ).to(device)
        # Should not crash
        wrapper.update_normalization(obs)
        actions = wrapper.act(obs)
        values = wrapper.evaluate(obs)
        assert actions.shape == (4, 29)
        assert values.shape == (4, 1)
