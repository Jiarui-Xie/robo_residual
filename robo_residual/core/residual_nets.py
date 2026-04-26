"""Residual network architectures for use with ResidualActorCritic.

Supported types:
- ``MLPResidual``: stateless feed-forward MLP (default)
- ``LSTMResidual``: single-step recurrent LSTM with managed hidden state (h, c)
- ``GRUResidual``: single-step recurrent GRU with managed hidden state (h only)

All expose the same interface:
    forward(obs: Tensor) -> Tensor
    reset(dones: Tensor | None) -> None
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def _build_mlp_sequential(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str,
) -> nn.Sequential:
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {activation}. Choose from {list(_ACTIVATIONS)}"
        )
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(_ACTIVATIONS[activation]())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class MLPResidual(nn.Module):
    """Stateless MLP residual: obs → hidden layers → delta."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.net = _build_mlp_sequential(input_dim, output_dim, hidden_dims, activation)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)

    def reset(self, dones: Tensor | None = None) -> None:
        pass  # stateless — nothing to reset


class LSTMResidual(nn.Module):
    """Recurrent LSTM residual network with managed per-environment hidden state.

    The hidden state is stored internally. Call ``reset(dones)`` at each
    rollout step to zero out hidden state for environments that just terminated.

    Args:
        input_dim: Observation feature dimension.
        output_dim: Action delta dimension.
        hidden_dim: LSTM hidden state size.
        num_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._hidden: tuple[Tensor, Tensor] | None = None

    def forward(self, obs: Tensor) -> Tensor:
        """Single-step forward. Hidden state is updated and detached from graph.

        Detaching after each step implements 1-step TBPTT: gradients flow through
        the current step's computation but not through historical hidden states.
        This prevents "backward through freed graph" errors across optimizer steps.
        """
        x = obs.unsqueeze(1)  # (B, 1, input_dim)
        out, (h_n, c_n) = self.lstm(x, self._hidden)
        self._hidden = (h_n.detach(), c_n.detach())
        return self.output_layer(out.squeeze(1))  # (B, output_dim)

    def reset(self, dones: Tensor | None = None) -> None:
        """Zero hidden state for done environments.

        Args:
            dones: Bool/float tensor of shape ``(B,)`` or ``(B, 1)``.
                   Entries that are True/1 will have their hidden state
                   cleared. Pass ``None`` to clear all environments.
        """
        if dones is None or self._hidden is None:
            self._hidden = None
            return
        h, c = self._hidden
        # mask broadcasts over (num_layers, B, hidden_dim)
        mask = dones.bool().view(1, -1, 1).to(h.device)
        self._hidden = (h.masked_fill(mask, 0.0), c.masked_fill(mask, 0.0))


class GRUResidual(nn.Module):
    """Recurrent GRU residual network with managed per-environment hidden state.

    GRU has no cell state (only ``h``), making it ~25% lighter than LSTM at
    the same hidden dim. Prefer over LSTM when deployment latency is critical.

    Args:
        input_dim: Observation feature dimension.
        output_dim: Action delta dimension.
        hidden_dim: GRU hidden state size.
        num_layers: Number of GRU layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._hidden: Tensor | None = None

    def forward(self, obs: Tensor) -> Tensor:
        """Single-step forward. Hidden state is updated and detached from graph."""
        x = obs.unsqueeze(1)  # (B, 1, input_dim)
        out, h_n = self.gru(x, self._hidden)
        self._hidden = h_n.detach()
        return self.output_layer(out.squeeze(1))  # (B, output_dim)

    def reset(self, dones: Tensor | None = None) -> None:
        """Zero hidden state for done environments.

        Args:
            dones: Bool/float tensor of shape ``(B,)`` or ``(B, 1)``.
                   Pass ``None`` to clear all environments.
        """
        if dones is None or self._hidden is None:
            self._hidden = None
            return
        mask = dones.bool().view(1, -1, 1).to(self._hidden.device)
        self._hidden = self._hidden.masked_fill(mask, 0.0)


class GRUStatelessWrapper(nn.Module):
    """Stateless wrapper around GRUResidual for ONNX export.

    Inputs : (obs, h)
    Outputs: (delta, h_n)
    """

    def __init__(self, gru_residual: GRUResidual) -> None:
        super().__init__()
        self.gru = gru_residual.gru
        self.output_layer = gru_residual.output_layer

    def forward(self, obs: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Single-step forward with explicit hidden state.

        Args:
            obs: ``(B, input_dim)``
            h:   ``(num_layers, B, hidden_dim)``

        Returns:
            delta: ``(B, output_dim)``
            h_n:   ``(num_layers, B, hidden_dim)``
        """
        x = obs.unsqueeze(1)  # (B, 1, input_dim)
        out, h_n = self.gru(x, h)
        return self.output_layer(out.squeeze(1)), h_n


class LSTMStatelessWrapper(nn.Module):
    """Stateless wrapper around LSTMResidual for ONNX export.

    Accepts and returns the hidden state explicitly so the ONNX graph has
    deterministic I/O — required for ONNX graph surgery in fuse_onnx.

    Inputs : (obs, h, c)
    Outputs: (delta, h_n, c_n)
    """

    def __init__(self, lstm_residual: LSTMResidual) -> None:
        super().__init__()
        self.lstm = lstm_residual.lstm
        self.output_layer = lstm_residual.output_layer

    def forward(
        self,
        obs: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-step forward with explicit hidden state.

        Args:
            obs: ``(B, input_dim)``
            h: ``(num_layers, B, hidden_dim)``
            c: ``(num_layers, B, hidden_dim)``

        Returns:
            delta: ``(B, output_dim)``
            h_n:   ``(num_layers, B, hidden_dim)``
            c_n:   ``(num_layers, B, hidden_dim)``
        """
        x = obs.unsqueeze(1)  # (B, 1, input_dim)
        out, (h_n, c_n) = self.lstm(x, (h, c))
        return self.output_layer(out.squeeze(1)), h_n, c_n
