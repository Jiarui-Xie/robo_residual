from __future__ import annotations

import torch
import pytest

from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic
from robo_residual.core.residual_nets import GRUResidual, LSTMResidual, MLPResidual
from robo_residual.config.residual_config import ResidualConfig


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

    def test_residual_stats_after_act(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        rac.act(obs)
        stats = rac.residual_stats()
        assert "residual/mean_abs_delta" in stats
        assert "residual/saturation_rate" in stats
        assert stats["residual/mean_abs_delta"].shape == (29,)
        assert stats["residual/saturation_rate"].shape == (29,)
        assert isinstance(stats["residual/mean_abs_delta_scalar"], float)
        assert isinstance(stats["residual/max_saturation_rate"], float)
        assert (stats["residual/mean_abs_delta"] >= 0).all()
        assert (stats["residual/saturation_rate"] >= 0).all()
        assert (stats["residual/saturation_rate"] <= 1).all()

    def test_residual_stats_empty_before_forward(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        assert rac.residual_stats() == {}

    def test_residual_stats_saturation_at_clamp(self, dummy_onnx, simple_config, device):
        """When residual is forced large, saturation should be 1.0 for all joints."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(100.0)
        obs = torch.randn(16, 64, device=device)
        rac.act_inference(obs)
        stats = rac.residual_stats()
        assert (stats["residual/saturation_rate"] == 1.0).all()

    def test_residual_stats_no_saturation_at_zero(self, dummy_onnx, simple_config, device):
        """Zero-init residual should have zero delta and zero saturation."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        rac.act_inference(obs)
        stats = rac.residual_stats()
        torch.testing.assert_close(stats["residual/mean_abs_delta"], torch.zeros(29, device=device))
        torch.testing.assert_close(stats["residual/saturation_rate"], torch.zeros(29, device=device))

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


# ── Unit tests for residual network types ─────────────────────────────────────

class TestMLPResidual:
    def test_output_shape(self, device):
        net = MLPResidual(64, 29, [32, 16]).to(device)
        obs = torch.randn(8, 64, device=device)
        assert net(obs).shape == (8, 29)

    def test_reset_is_noop(self, device):
        net = MLPResidual(64, 29, [32, 16]).to(device)
        obs = torch.randn(4, 64, device=device)
        out1 = net(obs)
        net.reset(torch.ones(4, device=device))
        out2 = net(obs)
        torch.testing.assert_close(out1, out2)

    def test_params_trainable(self, device):
        net = MLPResidual(64, 29, [32, 16]).to(device)
        assert all(p.requires_grad for p in net.parameters())


class TestLSTMResidual:
    def test_output_shape(self, device):
        net = LSTMResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(8, 64, device=device)
        assert net(obs).shape == (8, 29)

    def test_hidden_state_persists(self, device):
        """Calling forward twice should give different results (LSTM is stateful)."""
        net = LSTMResidual(64, 29, hidden_dim=32).to(device)
        # Use non-zero weights so the LSTM actually changes state
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out1 = net(obs)
        out2 = net(obs)
        assert not torch.allclose(out1, out2), "LSTM output should differ on second call"

    def test_reset_all_clears_state(self, device):
        """After reset(None), a fresh forward should reproduce initial output."""
        net = LSTMResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(4, 64, device=device)
        out_fresh = net(obs).clone()
        # Dirty the hidden state
        net(obs)
        net(obs)
        # Full reset then re-run from same obs
        net.reset(None)
        out_after_reset = net(obs)
        torch.testing.assert_close(out_fresh, out_after_reset)

    def test_reset_selective(self, device):
        """Selective reset should zero only done envs' hidden state."""
        batch = 4
        net = LSTMResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(batch, 64, device=device)
        # Build up some hidden state
        net(obs)
        net(obs)
        h_before, c_before = net._hidden

        # Reset only env 0
        dones = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
        net.reset(dones)
        h_after, c_after = net._hidden

        # Env 0 hidden state should be zero
        assert h_after[:, 0, :].abs().max() == 0.0
        assert c_after[:, 0, :].abs().max() == 0.0
        # Envs 1-3 should be unchanged
        torch.testing.assert_close(h_after[:, 1:, :], h_before[:, 1:, :])
        torch.testing.assert_close(c_after[:, 1:, :], c_before[:, 1:, :])

    def test_params_trainable(self, device):
        net = LSTMResidual(64, 29, hidden_dim=32).to(device)
        assert all(p.requires_grad for p in net.parameters())


# ── Integration: ResidualActorCritic with LSTM residual ───────────────────────

class TestResidualActorCriticLSTM:
    def test_config_builds_lstm_residual(self, dummy_onnx, lstm_config, device):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        assert isinstance(rac.residual, LSTMResidual)

    def test_mlp_config_builds_mlp_residual(self, dummy_onnx, simple_config, device):
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        assert isinstance(rac.residual, MLPResidual)

    def test_invalid_residual_type_raises(self, dummy_onnx, device):
        bad_config = ResidualConfig(residual_type="cnn")
        with pytest.raises(ValueError, match="Unknown residual_type"):
            ResidualActorCritic(dummy_onnx, bad_config, device=device)

    def test_lstm_zero_init_matches_base(self, dummy_onnx, lstm_config, device):
        """At init, LSTM residual output is zero, so act_inference matches base."""
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        base = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(4, 64, device=device)
        torch.testing.assert_close(
            rac.act_inference(obs), base.forward(obs), atol=1e-5, rtol=1e-5
        )

    def test_lstm_act_shape(self, dummy_onnx, lstm_config, device):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        assert rac.act(obs).shape == (8, 29)

    def test_lstm_stateful(self, dummy_onnx, lstm_config, device):
        """Sequential act_inference calls should give different outputs (LSTM is stateful)."""
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        # Give non-zero weights so state actually matters
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out1 = rac.act_inference(obs).clone()
        out2 = rac.act_inference(obs)
        assert not torch.allclose(out1, out2), "LSTM RAC should be stateful"

    def test_lstm_reset_full(self, dummy_onnx, lstm_config, device):
        """After reset(), outputs should match a fresh RAC."""
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out_fresh = rac.act_inference(obs).clone()
        # Dirty state
        rac.act_inference(obs)
        # Reset then replay
        rac.reset()
        out_after = rac.act_inference(obs)
        torch.testing.assert_close(out_fresh, out_after)

    def test_lstm_reset_selective(self, dummy_onnx, lstm_config, device):
        """Selective reset via dones tensor should zero only done envs' hidden state."""
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        batch = 4
        obs = torch.randn(batch, 64, device=device)
        rac.act_inference(obs)
        rac.act_inference(obs)
        h_before, c_before = rac.residual._hidden

        dones = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
        rac.reset(dones)
        h_after, c_after = rac.residual._hidden

        # Env 0 should be zeroed
        assert h_after[:, 0, :].abs().max() == 0.0
        assert c_after[:, 0, :].abs().max() == 0.0
        # Envs 1-3 unchanged
        torch.testing.assert_close(h_after[:, 1:, :], h_before[:, 1:, :])
        torch.testing.assert_close(c_after[:, 1:, :], c_before[:, 1:, :])

    def test_lstm_clamping(self, dummy_onnx, lstm_config, device):
        """Per-joint clamping should apply to LSTM residual."""
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(10.0)
        obs = torch.randn(4, 64, device=device)
        base_out = rac.base.forward(obs)
        full_out = rac.act_inference(obs)
        delta = full_out - base_out
        assert (delta[:, :12].abs() <= 0.05 + 1e-6).all(), "leg joints not clamped"
        assert (delta[:, 12:29].abs() <= 0.15 + 1e-6).all(), "arm joints not clamped"

    def test_lstm_trainable_params(self, dummy_onnx, lstm_config, device):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        for name, p in rac.named_parameters():
            assert p.requires_grad, f"{name} should require grad"

    def test_lstm_gradient_flows(self, dummy_onnx, lstm_config, device):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        actions = rac.act(obs)
        loss = -rac.get_actions_log_prob(actions).mean()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in rac.residual.parameters()
        )
        assert has_grad, "LSTM residual should receive gradients"

    def test_lstm_residual_stats(self, dummy_onnx, lstm_config, device):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        rac.act(obs)
        stats = rac.residual_stats()
        assert "residual/mean_abs_delta" in stats
        assert stats["residual/mean_abs_delta"].shape == (29,)


# ── Unit tests for GRUResidual ────────────────────────────────────────────────

class TestGRUResidual:
    def test_output_shape(self, device):
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(8, 64, device=device)
        assert net(obs).shape == (8, 29)

    def test_hidden_state_persists(self, device):
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out1 = net(obs)
        out2 = net(obs)
        assert not torch.allclose(out1, out2), "GRU output should differ on second call"

    def test_reset_all_clears_state(self, device):
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(4, 64, device=device)
        out_fresh = net(obs).clone()
        net(obs)
        net(obs)
        net.reset(None)
        out_after_reset = net(obs)
        torch.testing.assert_close(out_fresh, out_after_reset)

    def test_reset_selective(self, device):
        """Selective reset should zero only done envs' hidden state."""
        batch = 4
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        obs = torch.randn(batch, 64, device=device)
        net(obs)
        net(obs)
        h_before = net._hidden

        dones = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
        net.reset(dones)
        h_after = net._hidden

        assert h_after[:, 0, :].abs().max() == 0.0
        torch.testing.assert_close(h_after[:, 1:, :], h_before[:, 1:, :])

    def test_params_trainable(self, device):
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        assert all(p.requires_grad for p in net.parameters())

    def test_no_cell_state(self, device):
        """GRU should have only h, not (h, c)."""
        net = GRUResidual(64, 29, hidden_dim=32).to(device)
        net(torch.randn(4, 64, device=device))
        assert isinstance(net._hidden, torch.Tensor), "GRU hidden state should be a plain tensor"


# ── Integration: ResidualActorCritic with GRU residual ────────────────────────

class TestResidualActorCriticGRU:
    def test_config_builds_gru_residual(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        assert isinstance(rac.residual, GRUResidual)

    def test_gru_zero_init_matches_base(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        base = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(4, 64, device=device)
        torch.testing.assert_close(
            rac.act_inference(obs), base.forward(obs), atol=1e-5, rtol=1e-5
        )

    def test_gru_act_shape(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        obs = torch.randn(8, 64, device=device)
        assert rac.act(obs).shape == (8, 29)

    def test_gru_stateful(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out1 = rac.act_inference(obs).clone()
        out2 = rac.act_inference(obs)
        assert not torch.allclose(out1, out2)

    def test_gru_reset(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(0.01)
        obs = torch.randn(4, 64, device=device)
        out_fresh = rac.act_inference(obs).clone()
        rac.act_inference(obs)
        rac.reset()
        out_after = rac.act_inference(obs)
        torch.testing.assert_close(out_fresh, out_after)

    def test_gru_clamping(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(10.0)
        obs = torch.randn(4, 64, device=device)
        base_out = rac.base.forward(obs)
        delta = rac.act_inference(obs) - base_out
        assert (delta[:, :12].abs() <= 0.05 + 1e-6).all()
        assert (delta[:, 12:29].abs() <= 0.15 + 1e-6).all()

    def test_gru_gradient_flows(self, dummy_onnx, gru_config, device):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        obs = torch.randn(4, 64, device=device)
        loss = -rac.get_actions_log_prob(rac.act(obs)).mean()
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in rac.residual.parameters()
        )
