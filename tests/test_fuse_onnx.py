from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig
from robo_residual.core.fuse_onnx import fuse_residual_to_onnx
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic
from robo_residual.core.residual_nets import GRUResidual, LSTMResidual


class TestFuseOnnx:
    def _train_residual(self, onnx_path, config, device, steps=5):
        """Quick training to get non-zero residual weights."""
        rac = ResidualActorCritic(onnx_path, config, device=device).to(device)
        optimizer = torch.optim.Adam(rac.trainable_parameters, lr=1e-2)
        for _ in range(steps):
            obs = torch.randn(4, rac.num_actor_obs, device=device)
            actions = rac.act(obs)
            log_prob = rac.get_actions_log_prob(actions)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return rac

    def test_fuse_produces_valid_onnx(self, dummy_onnx, simple_config, device, tmp_path):
        rac = self._train_residual(dummy_onnx, simple_config, device)
        out_path = tmp_path / "fused.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)

    def test_fused_output_matches_training(self, dummy_onnx, simple_config, device, tmp_path):
        rac = self._train_residual(dummy_onnx, simple_config, device)

        # Capture training-time output BEFORE fuse moves residual to CPU
        obs = torch.randn(4, 64, device=device)
        training_out = rac.act_inference(obs)

        out_path = tmp_path / "fused.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )

        sess = ort.InferenceSession(str(out_path))
        fused_out = sess.run(None, {"obs": obs.cpu().numpy().astype(np.float32)})[0]
        fused_out = torch.from_numpy(fused_out).to(device)

        torch.testing.assert_close(training_out, fused_out, atol=1e-3, rtol=1e-3)

    def test_fused_single_input_single_output(self, dummy_onnx, simple_config, device, tmp_path):
        rac = self._train_residual(dummy_onnx, simple_config, device)
        out_path = tmp_path / "fused.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        model = onnx.load(str(out_path))
        assert len(model.graph.input) == 1
        assert len(model.graph.output) == 1

    def test_fused_shape_correct(self, dummy_onnx, simple_config, device, tmp_path):
        rac = self._train_residual(dummy_onnx, simple_config, device)
        out_path = tmp_path / "fused.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        sess = ort.InferenceSession(str(out_path))
        obs = np.random.randn(8, 64).astype(np.float32)
        out = sess.run(None, {"obs": obs})[0]
        assert out.shape == (8, 29)

    def test_fused_clamping_applied(self, dummy_onnx, simple_config, device, tmp_path):
        """Even with large residual weights, fused output should be clamped."""
        rac = ResidualActorCritic(dummy_onnx, simple_config, device=device).to(device)
        # Force huge residual
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(100.0)

        out_path = tmp_path / "fused.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )

        obs = np.random.randn(4, 64).astype(np.float32)
        sess = ort.InferenceSession(str(out_path))
        fused_out = sess.run(None, {"obs": obs})[0]

        base = OnnxBasePolicy(dummy_onnx, device="cpu")
        base_out = base.forward(torch.from_numpy(obs)).numpy()

        delta = fused_out - base_out
        # Legs clamped to 0.05, arms to 0.15
        assert np.all(np.abs(delta[:, :12]) <= 0.05 + 1e-4)
        assert np.all(np.abs(delta[:, 12:29]) <= 0.15 + 1e-4)


class TestFuseOnnxLSTM:
    def _train_lstm_residual(self, onnx_path, lstm_config, device, steps=5):
        rac = ResidualActorCritic(onnx_path, lstm_config, device=device).to(device)
        optimizer = torch.optim.Adam(rac.trainable_parameters, lr=1e-2)
        for _ in range(steps):
            obs = torch.randn(4, rac.num_actor_obs, device=device)
            actions = rac.act(obs)
            loss = -rac.get_actions_log_prob(actions).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rac.reset(torch.zeros(4, device=device))
        return rac

    def test_lstm_fuse_produces_valid_onnx(self, dummy_onnx, lstm_config, device, tmp_path):
        rac = self._train_lstm_residual(dummy_onnx, lstm_config, device)
        out_path = tmp_path / "fused_lstm.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)

    def test_lstm_fused_has_hidden_state_io(self, dummy_onnx, lstm_config, device, tmp_path):
        """Fused LSTM model should expose h_in/c_in inputs and h_out/c_out outputs."""
        rac = self._train_lstm_residual(dummy_onnx, lstm_config, device)
        out_path = tmp_path / "fused_lstm.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        model = onnx.load(str(out_path))
        input_names = {inp.name for inp in model.graph.input}
        output_names = {out.name for out in model.graph.output}
        assert "residual_h_in" in input_names
        assert "residual_c_in" in input_names
        assert any("h_out" in n for n in output_names)
        assert any("c_out" in n for n in output_names)

    def test_lstm_fused_action_output_matches(self, dummy_onnx, lstm_config, device, tmp_path):
        """Fused LSTM action output should match training-time act_inference."""
        rac = self._train_lstm_residual(dummy_onnx, lstm_config, device)
        # Reset and capture training-time output with zeroed hidden state
        rac.reset()
        obs = torch.randn(1, 64, device=device)
        training_out = rac.act_inference(obs)

        out_path = tmp_path / "fused_lstm.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        lstm_res = rac.residual
        sess = ort.InferenceSession(str(out_path))
        h0 = np.zeros((lstm_res.num_layers, 1, lstm_res.hidden_dim), dtype=np.float32)
        c0 = np.zeros((lstm_res.num_layers, 1, lstm_res.hidden_dim), dtype=np.float32)
        fused_out = sess.run(None, {
            "obs": obs.cpu().numpy().astype(np.float32),
            "residual_h_in": h0,
            "residual_c_in": c0,
        })[0]
        fused_tensor = torch.from_numpy(fused_out).to(device)
        torch.testing.assert_close(training_out, fused_tensor, atol=1e-3, rtol=1e-3)

    def test_lstm_fused_clamping_applied(self, dummy_onnx, lstm_config, device, tmp_path):
        rac = ResidualActorCritic(dummy_onnx, lstm_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(100.0)
        out_path = tmp_path / "fused_lstm.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits,
            out_path, num_obs=64,
        )
        lstm_res = rac.residual
        obs = np.random.randn(4, 64).astype(np.float32)
        h0 = np.zeros((lstm_res.num_layers, 4, lstm_res.hidden_dim), dtype=np.float32)
        c0 = np.zeros((lstm_res.num_layers, 4, lstm_res.hidden_dim), dtype=np.float32)
        sess = ort.InferenceSession(str(out_path))
        fused_out = sess.run(None, {
            "obs": obs,
            "residual_h_in": h0,
            "residual_c_in": c0,
        })[0]
        base = OnnxBasePolicy(dummy_onnx, device="cpu")
        base_out = base.forward(torch.from_numpy(obs)).numpy()
        delta = fused_out - base_out
        assert np.all(np.abs(delta[:, :12]) <= 0.05 + 1e-4)
        assert np.all(np.abs(delta[:, 12:29]) <= 0.15 + 1e-4)


class TestFuseOnnxGRU:
    def _train_gru_residual(self, onnx_path, gru_config, device, steps=5):
        rac = ResidualActorCritic(onnx_path, gru_config, device=device).to(device)
        optimizer = torch.optim.Adam(rac.trainable_parameters, lr=1e-2)
        for _ in range(steps):
            obs = torch.randn(4, rac.num_actor_obs, device=device)
            loss = -rac.get_actions_log_prob(rac.act(obs)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rac.reset(torch.zeros(4, device=device))
        return rac

    def test_gru_fuse_produces_valid_onnx(self, dummy_onnx, gru_config, device, tmp_path):
        rac = self._train_gru_residual(dummy_onnx, gru_config, device)
        out_path = tmp_path / "fused_gru.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits, out_path, num_obs=64
        )
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)

    def test_gru_fused_has_hidden_state_io(self, dummy_onnx, gru_config, device, tmp_path):
        """GRU fused model should have h_in input and h_out output (no c)."""
        rac = self._train_gru_residual(dummy_onnx, gru_config, device)
        out_path = tmp_path / "fused_gru.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits, out_path, num_obs=64
        )
        model = onnx.load(str(out_path))
        input_names = {inp.name for inp in model.graph.input}
        output_names = {out.name for out in model.graph.output}
        assert "residual_h_in" in input_names
        assert "residual_c_in" not in input_names, "GRU should NOT have c_in"
        assert any("h_out" in n for n in output_names)
        assert not any("c_out" in n for n in output_names), "GRU should NOT have c_out"

    def test_gru_fused_action_output_matches(self, dummy_onnx, gru_config, device, tmp_path):
        rac = self._train_gru_residual(dummy_onnx, gru_config, device)
        rac.reset()
        obs = torch.randn(1, 64, device=device)
        training_out = rac.act_inference(obs)

        out_path = tmp_path / "fused_gru.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits, out_path, num_obs=64
        )
        gru_res = rac.residual
        sess = ort.InferenceSession(str(out_path))
        h0 = np.zeros((gru_res.num_layers, 1, gru_res.hidden_dim), dtype=np.float32)
        fused_out = sess.run(None, {
            "obs": obs.cpu().numpy().astype(np.float32),
            "residual_h_in": h0,
        })[0]
        torch.testing.assert_close(
            training_out, torch.from_numpy(fused_out).to(device), atol=1e-3, rtol=1e-3
        )

    def test_gru_fused_clamping_applied(self, dummy_onnx, gru_config, device, tmp_path):
        rac = ResidualActorCritic(dummy_onnx, gru_config, device=device).to(device)
        with torch.no_grad():
            for p in rac.residual.parameters():
                p.fill_(100.0)
        out_path = tmp_path / "fused_gru.onnx"
        fuse_residual_to_onnx(
            dummy_onnx, rac.residual, rac.max_residual_limits, out_path, num_obs=64
        )
        gru_res = rac.residual
        obs = np.random.randn(4, 64).astype(np.float32)
        h0 = np.zeros((gru_res.num_layers, 4, gru_res.hidden_dim), dtype=np.float32)
        sess = ort.InferenceSession(str(out_path))
        fused_out = sess.run(None, {"obs": obs, "residual_h_in": h0})[0]
        base_out = OnnxBasePolicy(dummy_onnx, device="cpu").forward(torch.from_numpy(obs)).numpy()
        delta = fused_out - base_out
        assert np.all(np.abs(delta[:, :12]) <= 0.05 + 1e-4)
        assert np.all(np.abs(delta[:, 12:29]) <= 0.15 + 1e-4)
