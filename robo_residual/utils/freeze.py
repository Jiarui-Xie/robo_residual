from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Freeze a module: disable gradients and set to eval mode."""
    module.requires_grad_(False)
    module.eval()


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze a module: enable gradients and set to train mode."""
    module.requires_grad_(True)
    module.train()
