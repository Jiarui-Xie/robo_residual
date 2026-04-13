from __future__ import annotations

import torch
import pytest

from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic


class TestResidualActorCritic:
    def test_zero_init_matches_base(self, dummy_onnx, simple_config, device):
        """At init, residual is zero so act_inference() should match base output."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        base = OnnxBasePolicy(dummy_onnx, device=device)

        obs = torch.randn(4, 64, device=device)
        rac_out = rac.act_inference(obs)
        base_out = base.forward(obs)
        torch.testing.assert_close(rac_out, base_out, atol=1e-5, rtol=1e-5)

    def test_per_joint_clamping_legs(self, dummy_onnx, simple_config, device):
        """Residual for leg joints (0-11) should be clamped to 0.05."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        # Force large residual weights
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(10.0)

        obs = torch.randn(4, 64, device=device)
        base_out = rac.base.forward(obs)
        full_out = rac.act_inference(obs)
        delta = full_out - base_out

        # Leg joints should be clamped to [-0.05, 0.05]
        assert (delta[:, :12].abs() <= 0.05 + 1e-6).all()

    def test_per_joint_clamping_arms(self, dummy_onnx, simple_config, device):
        """Residual for arm joints (12-28) should be clamped to 0.15."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(10.0)

        obs = torch.randn(4, 64, device=device)
        base_out = rac.base.forward(obs)
        full_out = rac.act_inference(obs)
        delta = full_out - base_out

        assert (delta[:, 12:29].abs() <= 0.15 + 1e-6).all()

    def test_residual_params_trainable(self, dummy_onnx, simple_config, device):
        """Residual MLP, critic, and noise should all require grad."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        for name, p in rac.named_parameters():
            assert p.requires_grad, f"Parameter {name} should require grad"

    def test_act_shape(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        actions = rac.act(obs)
        assert actions.shape == (8, 29)

    def test_evaluate_shape(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        values = rac.evaluate(obs)
        assert values.shape == (8, 1)

    def test_log_prob_finite(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        actions = rac.act(obs)
        log_prob = rac.get_actions_log_prob(actions)
        assert torch.isfinite(log_prob).all()

    def test_entropy_finite(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        rac.act(obs)
        assert torch.isfinite(rac.entropy).all()

    def test_gradient_flows_through_residual(self, dummy_onnx, simple_config, device):
        """Backward pass via log_prob should update residual weights."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        actions = rac.act(obs)
        log_prob = rac.get_actions_log_prob(actions)
        loss = -log_prob.mean()
        loss.backward()

        has_grad = False
        for p in rac.residual.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Residual should receive gradients"

    def test_gradient_does_not_flow_to_base(self, dummy_onnx, simple_config, device):
        """ONNX base has no PyTorch params, so no grad concern."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        # Base is an OnnxBasePolicy (not nn.Module), so it has no .parameters()
        # Just verify the module's parameters are only from residual/critic/noise
        param_names = [n for n, _ in rac.named_parameters()]
        assert all(
            n.startswith("residual.") or n.startswith("critic.") or "std" in n
            for n in param_names
        ), f"Unexpected params: {param_names}"

    def test_act_inference_deterministic(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        out1 = rac.act_inference(obs)
        out2 = rac.act_inference(obs)
        torch.testing.assert_close(out1, out2)

    def test_auto_detect_dims(self, dummy_onnx, simple_config, device):
        """num_actor_obs and num_actions should auto-detect from ONNX."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        assert rac.num_actor_obs == 64
        assert rac.num_actions == 29

    def test_custom_dims_override(self, dummy_onnx, simple_config, device):
        """Explicit dims should override ONNX auto-detection."""
        rac = ResidualActorCritic(
            dummy_onnx, simple_config,
            num_actor_obs=64, num_critic_obs=128, num_actions=29,
            critic_hidden_dims=[64, 32],
            device=device,
        ).to(device)

        actor_obs = torch.randn(4, 64, device=device)
        critic_obs = torch.randn(4, 128, device=device)
        actions = rac.act(actor_obs)
        values = rac.evaluate(critic_obs)
        assert actions.shape == (4, 29)
        assert values.shape == (4, 1)
