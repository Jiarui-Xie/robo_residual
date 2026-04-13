from __future__ import annotations

from pathlib import Path

import torch
import pytest

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig
from robo_residual.core.composable import ComposableResidual
from robo_residual.core.fuse_onnx import fuse_residual_to_onnx
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic
from tests.conftest import _create_onnx_mlp


class TestComposable:
    def _make_frozen_residual_onnx(self, base_onnx, config, device, tmp_path, name="frozen_res.onnx"):
        """Train a residual, fuse it, return the fused ONNX path."""
        rac = ResidualActorCritic(base_onnx, config, device=device).to(device)
        # Train briefly
        optimizer = torch.optim.Adam(rac.trainable_parameters, lr=1e-2)
        for _ in range(3):
            obs = torch.randn(4, rac.num_actor_obs, device=device)
            actions = rac.act(obs)
            log_prob = rac.get_actions_log_prob(actions)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Export just the residual as ONNX (not fused with base — just the residual MLP)
        residual_path = tmp_path / name
        residual_cpu = rac.residual.cpu().eval()
        dummy = torch.zeros(1, rac.num_actor_obs)
        torch.onnx.export(
            residual_cpu, dummy, str(residual_path),
            input_names=["obs"], output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
        )
        return residual_path

    def test_stack_two_residuals(self, dummy_onnx_small, small_config, device, tmp_path):
        """Base + 1 frozen residual + 1 active residual should all contribute."""
        frozen_path = self._make_frozen_residual_onnx(
            dummy_onnx_small, small_config, device, tmp_path
        )
        frozen_limits = small_config.build_residual_limits(4)

        comp = ComposableResidual(
            base_onnx_path=dummy_onnx_small,
            frozen_residual_onnx_paths=[frozen_path],
            frozen_residual_limits=[frozen_limits],
            active_config=small_config,
            device=device,
        ).to(device)

        obs = torch.randn(4, 10, device=device)
        out = comp.act_inference(obs)
        assert out.shape == (4, 4)

        # Should differ from base alone (frozen residual is non-zero after training)
        base = OnnxBasePolicy(dummy_onnx_small, device=device)
        base_out = base.forward(obs)
        assert not torch.allclose(out, base_out, atol=1e-6)

    def test_independent_clamps(self, dummy_onnx_small, small_config, device, tmp_path):
        """Active residual should respect its own clamp limits."""
        frozen_path = self._make_frozen_residual_onnx(
            dummy_onnx_small, small_config, device, tmp_path
        )
        frozen_limits = small_config.build_residual_limits(4)

        # Active config with tighter limits
        tight_config = ResidualConfig(
            joint_groups=[
                JointGroupConfig("all", [0, 1, 2, 3], 0.01),
            ],
            residual_hidden_dims=[16, 8],
            init_noise_std=0.1,
        )

        comp = ComposableResidual(
            base_onnx_path=dummy_onnx_small,
            frozen_residual_onnx_paths=[frozen_path],
            frozen_residual_limits=[frozen_limits],
            active_config=tight_config,
            device=device,
        ).to(device)

        # Force large active residual weights
        with torch.no_grad():
            for p in comp.active_residual.parameters():
                p.fill_(10.0)

        obs = torch.randn(4, 10, device=device)

        # Get output with frozen residuals only (set active to zero)
        with torch.no_grad():
            for module in comp.active_residual.modules():
                if hasattr(module, 'weight'):
                    module.weight.zero_()
                if hasattr(module, 'bias'):
                    module.bias.zero_()

        base_plus_frozen = comp.act_inference(obs)

        # Restore large weights
        with torch.no_grad():
            for p in comp.active_residual.parameters():
                p.fill_(10.0)

        full_out = comp.act_inference(obs)
        active_delta = full_out - base_plus_frozen

        # Active delta should be clamped to 0.01
        assert (active_delta.abs() <= 0.01 + 1e-5).all()

    def test_new_phase_zero_init(self, dummy_onnx_small, small_config, device, tmp_path):
        """Active residual starts at zero, so output == base + frozen."""
        frozen_path = self._make_frozen_residual_onnx(
            dummy_onnx_small, small_config, device, tmp_path
        )
        frozen_limits = small_config.build_residual_limits(4)

        comp = ComposableResidual(
            base_onnx_path=dummy_onnx_small,
            frozen_residual_onnx_paths=[frozen_path],
            frozen_residual_limits=[frozen_limits],
            active_config=small_config,
            device=device,
        ).to(device)

        obs = torch.randn(4, 10, device=device)

        # Compute base + frozen manually
        base = OnnxBasePolicy(dummy_onnx_small, device=device)
        frozen_res = OnnxBasePolicy(frozen_path, device=device)
        expected = base.forward(obs) + torch.clamp(
            frozen_res.forward(obs), -frozen_limits.to(device), frozen_limits.to(device)
        )

        actual = comp.act_inference(obs)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_only_active_trainable(self, dummy_onnx_small, small_config, device, tmp_path):
        """Only active residual + critic + noise should have requires_grad."""
        frozen_path = self._make_frozen_residual_onnx(
            dummy_onnx_small, small_config, device, tmp_path
        )
        frozen_limits = small_config.build_residual_limits(4)

        comp = ComposableResidual(
            base_onnx_path=dummy_onnx_small,
            frozen_residual_onnx_paths=[frozen_path],
            frozen_residual_limits=[frozen_limits],
            active_config=small_config,
            device=device,
        ).to(device)

        param_names = [n for n, p in comp.named_parameters() if p.requires_grad]
        assert all(
            n.startswith("active_residual.") or n.startswith("critic.") or "std" in n
            for n in param_names
        ), f"Unexpected trainable params: {param_names}"
