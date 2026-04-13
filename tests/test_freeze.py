from __future__ import annotations

import torch.nn as nn

from robo_residual.utils.freeze import freeze_module, unfreeze_module


class TestFreeze:
    def test_freeze_disables_grad(self):
        m = nn.Linear(10, 5)
        freeze_module(m)
        for p in m.parameters():
            assert not p.requires_grad

    def test_freeze_sets_eval(self):
        m = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))
        freeze_module(m)
        assert not m.training

    def test_unfreeze_restores(self):
        m = nn.Linear(10, 5)
        freeze_module(m)
        unfreeze_module(m)
        for p in m.parameters():
            assert p.requires_grad
        assert m.training
