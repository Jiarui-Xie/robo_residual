from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_output_layer(module: nn.Module) -> None:
    """Zero-initialize the last Linear layer in a module.

    This makes the module output zeros for any input, which is critical for
    residual policies — the initial behavior is identical to the base policy.
    """
    last_linear: nn.Linear | None = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child

    if last_linear is None:
        raise ValueError("No nn.Linear found in module")

    with torch.no_grad():
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
