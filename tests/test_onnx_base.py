from __future__ import annotations

import torch
import pytest

from robo_residual.core.onnx_base import OnnxBasePolicy


class TestOnnxBasePolicy:
    def test_load_onnx(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        assert policy.num_obs == 64
        assert policy.num_actions == 29

    def test_forward_shape(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(8, 64, device=device)
        out = policy.forward(obs)
        assert out.shape == (8, 29)
        assert out.device.type == device

    def test_num_obs_actions(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        assert policy.num_obs == 64
        assert policy.num_actions == 29
        assert policy.input_names == ["obs"]
        assert policy.output_names == ["actions"]

    def test_deterministic(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(4, 64, device=device)
        out1 = policy.forward(obs)
        out2 = policy.forward(obs)
        torch.testing.assert_close(out1, out2)

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            OnnxBasePolicy("/nonexistent/model.onnx")

    def test_batch_size_one(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(1, 64, device=device)
        out = policy.forward(obs)
        assert out.shape == (1, 29)

    def test_large_batch(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(256, 64, device=device)
        out = policy.forward(obs)
        assert out.shape == (256, 29)

    def test_is_stateful_false_for_mlp(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        assert not policy.is_stateful

    def test_forward_full(self, dummy_onnx, device):
        policy = OnnxBasePolicy(dummy_onnx, device=device)
        obs = torch.randn(4, 64, device=device)
        result = policy.forward_full(obs=obs)
        assert "actions" in result
        assert result["actions"].shape == (4, 29)
