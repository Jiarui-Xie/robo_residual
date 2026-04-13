"""Empirical observation normalization (running mean/std).

Ported from rsl_rl's EmpiricalNormalization for use in the residual toolbox.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Online running mean/std normalization.

    Normalizes input to zero mean and unit variance using running statistics
    collected during training. At eval time, uses the frozen statistics.

    Args:
        input_dim: Dimension of the input features.
        epsilon: Small constant to avoid division by zero.
    """

    def __init__(self, input_dim: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a new batch of observations.

        Args:
            x: Batch of observations, shape ``(batch, input_dim)``.
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.running_mean

        # Welford's online algorithm for combining mean/variance
        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.running_mean = new_mean
        self.running_var = new_var
        self.count = total_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        return (x - self.running_mean) / (self.running_var.sqrt() + self.epsilon)
