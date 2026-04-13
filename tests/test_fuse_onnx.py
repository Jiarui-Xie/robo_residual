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

        torch.testing.assert_close(training_out, fused_out, atol=1e-4, rtol=1e-4)

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
