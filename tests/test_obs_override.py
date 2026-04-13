from __future__ import annotations

import torch
import pytest

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic


class TestObsOverride:
    """Test that residual can take larger obs than base ONNX."""

    def test_residual_larger_obs(self, dummy_onnx, simple_config, device):
        """Residual MLP takes 128-dim obs, base ONNX takes 64-dim obs."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=128,  # larger than base's 64
            num_base_obs=64,
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 128, device=device)
        actions = rac.act(actor_obs)
        assert actions.shape == (4, 29)

    def test_base_gets_correct_slice(self, dummy_onnx, simple_config, device):
        """Base should only see first 64 dims of a 128-dim actor obs."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=128,
            num_base_obs=64,
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 128, device=device)
        base_obs = rac._get_base_obs(actor_obs)
        assert base_obs.shape == (4, 64)
        torch.testing.assert_close(base_obs, actor_obs[:, :64])

    def test_same_obs_no_slicing(self, dummy_onnx, simple_config, device):
        """When actor_obs == base_obs, no slicing happens."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 64, device=device)
        base_obs = rac._get_base_obs(actor_obs)
        assert base_obs.data_ptr() == actor_obs.data_ptr()  # same tensor, no copy

    def test_zero_init_with_larger_obs(self, dummy_onnx, simple_config, device):
        """Even with larger obs, zero-init means act_inference == base output."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=128,
            num_base_obs=64,
            device=device,
        ).to(device)
        base = OnnxBasePolicy(dummy_onnx, device=device)

        actor_obs = torch.randn(4, 128, device=device)
        rac_out = rac.act_inference(actor_obs)
        base_out = base.forward(actor_obs[:, :64])
        torch.testing.assert_close(rac_out, base_out, atol=1e-5, rtol=1e-5)

    def test_gradient_with_larger_obs(self, dummy_onnx, simple_config, device):
        """Gradient should flow through residual even with larger obs."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=128,
            num_base_obs=64,
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 128, device=device)
        actions = rac.act(actor_obs)
        log_prob = rac.get_actions_log_prob(actions)
        loss = -log_prob.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in rac.residual.parameters()
        )
        assert has_grad

    def test_separate_critic_obs(self, dummy_onnx, simple_config, device):
        """Critic can have a completely different obs dim."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=128,
            num_critic_obs=200,
            num_base_obs=64,
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 128, device=device)
        critic_obs = torch.randn(4, 200, device=device)
        actions = rac.act(actor_obs)
        values = rac.evaluate(critic_obs)
        assert actions.shape == (4, 29)
        assert values.shape == (4, 1)
