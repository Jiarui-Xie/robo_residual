from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from robo_residual.utils.zero_init import zero_init_output_layer


class TestZeroInit:
    def test_last_layer_zeroed(self):
        mlp = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
        zero_init_output_layer(mlp)
        last = mlp[-1]
        assert (last.weight == 0).all()
        assert (last.bias == 0).all()

    def test_output_is_zero(self):
        mlp = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
        zero_init_output_layer(mlp)
        x = torch.randn(4, 10)
        out = mlp(x)
        torch.testing.assert_close(out, torch.zeros(4, 5))

    def test_other_layers_nonzero(self):
        mlp = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
        first_weight_before = mlp[0].weight.clone()
        zero_init_output_layer(mlp)
        # First layer should be unchanged
        torch.testing.assert_close(mlp[0].weight, first_weight_before)

    def test_no_linear_raises(self):
        m = nn.Sequential(nn.ReLU(), nn.Tanh())
        with pytest.raises(ValueError, match="No nn.Linear"):
            zero_init_output_layer(m)
