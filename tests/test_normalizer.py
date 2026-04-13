from __future__ import annotations

import torch
import pytest

from robo_residual.utils.normalizer import EmpiricalNormalization


class TestEmpiricalNormalization:
    def test_initial_passthrough(self, device):
        """Before any updates, normalizes with mean=0, std=1 (identity-ish)."""
        norm = EmpiricalNormalization(10).to(device)
        x = torch.randn(4, 10, device=device)
        out = norm(x)
        # running_mean=0, running_var=1 → output ≈ input
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_update_changes_stats(self, device):
        norm = EmpiricalNormalization(10).to(device)
        data = torch.randn(100, 10, device=device) * 5 + 3
        norm.update(data)
        assert norm.count.item() == 100
        # Mean should be close to 3
        assert (norm.running_mean - 3.0).abs().mean() < 1.0

    def test_normalized_output_centered(self, device):
        norm = EmpiricalNormalization(5).to(device)
        data = torch.randn(1000, 5, device=device) * 2 + 10
        norm.update(data)
        out = norm(data)
        # Output should be roughly zero-mean
        assert out.mean(dim=0).abs().max() < 0.5

    def test_multiple_updates(self, device):
        norm = EmpiricalNormalization(5).to(device)
        for _ in range(10):
            batch = torch.randn(50, 5, device=device) * 3 + 1
            norm.update(batch)
        assert norm.count.item() == 500
