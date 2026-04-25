"""Inference-time residual router.

One frozen base ONNX + N residual ONNXs. Switch or blend residuals at runtime
without re-training or re-fusing.

Switching modes
---------------
hard
    One residual active at a time. ``switch()`` takes effect immediately.
soft
    All residuals run every step; their clamped deltas are weighted and summed.
    ``switch()`` triggers a linear interpolation from current weights to the
    new one-hot target over ``blend_steps`` control steps.

Obs-stacking note
-----------------
The router does NOT maintain an observation history buffer — that is the
caller's responsibility (e.g. an env wrapper that stacks K frames). The router
handles obs-dim mismatches automatically: each slot sees ``obs[:, :slot_num_obs]``,
so slots trained on different obs widths work transparently from the same input.

LSTM / stateful base
--------------------
If the base ONNX has extra inputs beyond the main obs (e.g. ``h_in``, ``c_in``),
the router manages those hidden states internally between ``step()`` calls.
Pass ``base_hidden_io_map={"h_out": "h_in", "c_out": "c_in"}`` explicitly, or
let the router infer the mapping positionally. Use ``reset()`` to clear hidden
states between episodes, and ``reset_base_hidden=True`` on ``switch()`` if the
new residual was trained from a different hidden-state distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from robo_residual.core.onnx_base import OnnxBasePolicy


class SwitchMode(str, Enum):
    """Residual routing strategy."""
    HARD = "hard"
    SOFT = "soft"


@dataclass
class ResidualSlotConfig:
    """Configuration for one residual slot loaded into the router.

    Args:
        onnx_path: Path to the trained residual ONNX. May be a standalone
            residual MLP or the output of ``fuse_residual_to_onnx()``.
        limits: Per-joint clamp limits, shape ``(num_actions,)``. Must match
            the limits used during training of this residual.
        name: Human-readable mode name used in :meth:`ResidualRouter.switch`.
            Defaults to the slot index as a string.
        obs_mean: Running obs mean for normalisation, shape ``(num_obs,)``.
            If ``None``, obs is passed through without normalisation.
        obs_std: Running obs std paired with ``obs_mean``.
        obs_input: Which ONNX input is the main obs — index or name. Default ``0``.
        action_output: Which ONNX output is the action — index or name. Default ``0``.
    """
    onnx_path: str | Path
    limits: torch.Tensor
    name: str = ""
    obs_mean: Optional[torch.Tensor] = None
    obs_std: Optional[torch.Tensor] = None
    obs_input: int | str = 0
    action_output: int | str = 0


@dataclass
class _Slot:
    name: str
    policy: OnnxBasePolicy
    limits: torch.Tensor
    obs_mean: Optional[torch.Tensor]
    obs_std: Optional[torch.Tensor]


class ResidualRouter:
    """Inference-time 1-base + N-residual router with hard/soft switching.

    Parameters
    ----------
    base_onnx_path:
        Path to the base policy ONNX.
    residual_configs:
        One :class:`ResidualSlotConfig` per residual mode. At least one required.
    switch_mode:
        ``"hard"`` — exclusive, instantaneous. One residual active at a time.
        ``"soft"`` — weighted blend. All residuals run; their weighted deltas sum.
    default_blend_steps:
        Number of ``step()`` calls to interpolate over during a soft switch.
    base_obs_input:
        Which base ONNX input is the obs (index or name).
    base_action_output:
        Which base ONNX output is the action (index or name).
    base_obs_mean / base_obs_std:
        Optional obs normalisation for the base policy.
    base_hidden_io_map:
        For stateful bases (LSTM). Maps output tensor name → corresponding
        input tensor name, e.g. ``{"h_out": "h_in", "c_out": "c_in"}``.
        If ``None`` and the base is stateful, the mapping is inferred
        positionally: extra outputs are paired with extra inputs in declaration
        order.
    device:
        Device for all tensor operations (``"cpu"`` or ``"cuda"``).

    Examples
    --------
    Hard switching (instantaneous, exclusive)::

        router = ResidualRouter(
            base_onnx_path="base.onnx",
            residual_configs=[
                ResidualSlotConfig("energy.onnx",  limits_energy,  name="energy"),
                ResidualSlotConfig("terrain.onnx", limits_terrain, name="terrain"),
            ],
            switch_mode="hard",
        )
        action = router.step(obs)
        router.switch("terrain")

    Soft switching driven by terrain confidence::

        router = ResidualRouter(..., switch_mode="soft", default_blend_steps=30)
        # Programmatic blend: 70 % energy, 30 % terrain
        router.set_weights(torch.tensor([0.7, 0.3]))
        action = router.step(obs)

        # Smooth transition to pure terrain over 50 steps
        router.switch("terrain", blend_steps=50)
        for _ in range(50):
            action = router.step(obs)   # weights interpolate each step
    """

    def __init__(
        self,
        base_onnx_path: str | Path,
        residual_configs: list[ResidualSlotConfig],
        switch_mode: str | SwitchMode = SwitchMode.HARD,
        default_blend_steps: int = 20,
        base_obs_input: int | str = 0,
        base_action_output: int | str = 0,
        base_obs_mean: Optional[torch.Tensor] = None,
        base_obs_std: Optional[torch.Tensor] = None,
        base_hidden_io_map: Optional[dict[str, str]] = None,
        device: str = "cpu",
    ) -> None:
        if not residual_configs:
            raise ValueError("Provide at least one ResidualSlotConfig.")

        self._device = device
        self._switch_mode = SwitchMode(switch_mode)
        self._default_blend_steps = default_blend_steps

        self._base = OnnxBasePolicy(
            base_onnx_path,
            device=device,
            obs_input=base_obs_input,
            action_output=base_action_output,
        )
        self._base_obs_mean = base_obs_mean.to(device) if base_obs_mean is not None else None
        self._base_obs_std = base_obs_std.to(device) if base_obs_std is not None else None
        self._base_hidden_io_map: dict[str, str] = self._build_hidden_io_map(base_hidden_io_map)

        self._slots: list[_Slot] = []
        self._name_to_idx: dict[str, int] = {}
        for i, cfg in enumerate(residual_configs):
            name = cfg.name or str(i)
            if name in self._name_to_idx:
                raise ValueError(f"Duplicate slot name: '{name}'")
            slot = _Slot(
                name=name,
                policy=OnnxBasePolicy(
                    cfg.onnx_path, device=device,
                    obs_input=cfg.obs_input, action_output=cfg.action_output,
                ),
                limits=cfg.limits.to(device),
                obs_mean=cfg.obs_mean.to(device) if cfg.obs_mean is not None else None,
                obs_std=cfg.obs_std.to(device) if cfg.obs_std is not None else None,
            )
            self._slots.append(slot)
            self._name_to_idx[name] = i

        n = len(self._slots)
        self._active_idx: int = 0

        # Soft-mode blend weights, shape (n,)
        self._weights = torch.zeros(n, device=device)
        self._weights[0] = 1.0

        # Blend transition state (soft mode)
        self._src_weights: Optional[torch.Tensor] = None
        self._tgt_weights: Optional[torch.Tensor] = None
        self._blend_total: int = 0
        self._blend_done: int = 0

        # LSTM hidden state: keyed by ONNX *input* name for ready-to-feed access
        self._base_hidden: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def switch_mode(self) -> SwitchMode:
        return self._switch_mode

    @switch_mode.setter
    def switch_mode(self, mode: str | SwitchMode) -> None:
        self._switch_mode = SwitchMode(mode)

    @property
    def active_mode(self) -> str:
        """Name of the dominant mode (exclusive for hard; highest-weight for soft)."""
        if self._switch_mode == SwitchMode.HARD:
            return self._slots[self._active_idx].name
        return self._slots[int(self._weights.argmax())].name

    @property
    def weights(self) -> torch.Tensor:
        """Current blend weights, shape ``(n_residuals,)``. Read-only snapshot."""
        if self._switch_mode == SwitchMode.HARD:
            w = torch.zeros(len(self._slots), device=self._device)
            w[self._active_idx] = 1.0
            return w
        return self._weights.clone()

    @property
    def is_blending(self) -> bool:
        """True while a soft blend transition is in progress."""
        return self._tgt_weights is not None

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def switch(
        self,
        mode: str | int,
        blend_steps: Optional[int] = None,
        reset_base_hidden: bool = False,
    ) -> None:
        """Switch to a different residual mode.

        Hard mode: new mode is active immediately on the next ``step()``.
        Soft mode: begins a linear interpolation from the current weights to a
        one-hot target over ``blend_steps`` steps. Calling ``switch()`` again
        mid-blend starts a new blend from the current interpolated weights.

        Args:
            mode: Target mode — name (str) or slot index (int).
            blend_steps: Blend duration in steps (soft mode only). Defaults to
                ``default_blend_steps`` set at construction.
            reset_base_hidden: Clear the base LSTM hidden state on switch.
                Use this when the episode is resetting or when the new residual
                was trained from a very different behavioural distribution.
        """
        idx = self._resolve(mode)
        if reset_base_hidden:
            self._base_hidden = {}

        if self._switch_mode == SwitchMode.HARD:
            self._active_idx = idx
        else:
            self._src_weights = self._weights.clone()
            self._tgt_weights = torch.zeros(len(self._slots), device=self._device)
            self._tgt_weights[idx] = 1.0
            self._blend_total = blend_steps if blend_steps is not None else self._default_blend_steps
            self._blend_done = 0

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set blend weights directly, bypassing the transition (soft mode only).

        Cancels any ongoing blend. Weights are clamped to [0, ∞) and
        L1-normalised so they sum to 1.

        Args:
            weights: Shape ``(n_residuals,)``.
        """
        if self._switch_mode != SwitchMode.SOFT:
            raise RuntimeError("set_weights() requires switch_mode='soft'.")
        if weights.shape != (len(self._slots),):
            raise ValueError(
                f"Expected weights of shape ({len(self._slots)},), got {weights.shape}"
            )
        w = weights.to(self._device).clamp(min=0.0)
        total = w.sum()
        self._weights = w / total if total > 0 else w
        # Cancel any running blend
        self._tgt_weights = None
        self._src_weights = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(
        self,
        obs: torch.Tensor,
        **base_extra_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run one inference step.

        The base always runs once. In hard mode, only the active residual
        runs. In soft mode, all residuals with non-negligible weight run and
        their clamped deltas are summed.

        Obs-stacking is the caller's responsibility. Pass the full stacked
        obs; each slot auto-slices ``obs[:, :slot_num_obs]``.

        Args:
            obs: Observation tensor ``(batch, obs_dim)``.
            **base_extra_inputs: Extra named inputs for the base (e.g. manually
                managed LSTM hidden states). If omitted and the base is
                stateful, the router manages hidden states internally.

        Returns:
            Action tensor ``(batch, num_actions)``.
        """
        obs = obs.to(self._device)

        if self._switch_mode == SwitchMode.SOFT and self._tgt_weights is not None:
            self._advance_blend()

        base_obs = self._apply_norm(obs, self._base_obs_mean, self._base_obs_std, self._base.num_obs)
        if self._base.is_stateful:
            base_action = self._stateful_base_step(base_obs, base_extra_inputs)
        else:
            base_action = self._base.forward(base_obs)

        if self._switch_mode == SwitchMode.HARD:
            return self._hard_step(obs, base_action)
        return self._soft_step(obs, base_action)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset LSTM hidden states between episodes.

        Args:
            env_ids: Indices of envs to reset in a vectorised setting.
                If ``None``, all hidden states are cleared.
        """
        if env_ids is None:
            self._base_hidden = {}
        else:
            for tensor in self._base_hidden.values():
                tensor[env_ids] = 0.0

    def list_modes(self) -> list[str]:
        """Return all registered mode names in slot order."""
        return [s.name for s in self._slots]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, mode: str | int) -> int:
        if isinstance(mode, int):
            if not 0 <= mode < len(self._slots):
                raise IndexError(f"Slot index {mode} out of range [0, {len(self._slots)}).")
            return mode
        if mode not in self._name_to_idx:
            raise KeyError(f"Unknown mode '{mode}'. Available: {self.list_modes()}")
        return self._name_to_idx[mode]

    def _apply_norm(
        self,
        obs: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        num_obs: int,
    ) -> torch.Tensor:
        x = obs[..., :num_obs]
        if mean is not None and std is not None:
            x = (x - mean) / (std + 1e-8)
        return x

    def _build_hidden_io_map(self, user_map: Optional[dict[str, str]]) -> dict[str, str]:
        """Return output_name → input_name mapping for LSTM base, or {} for MLP."""
        if not self._base.is_stateful:
            return {}
        if user_map is not None:
            return user_map
        # Positional inference: zip extra outputs with extra inputs
        extra_inputs = [n for n in self._base.input_names if n != self._base._obs_input_name]
        extra_outputs = [n for n in self._base.output_names if n != self._base._action_output_name]
        return dict(zip(extra_outputs, extra_inputs))

    def _stateful_base_step(
        self,
        base_obs: torch.Tensor,
        caller_extras: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        feeds: dict[str, torch.Tensor] = {self._base._obs_input_name: base_obs}

        for out_name, in_name in self._base_hidden_io_map.items():
            if in_name in caller_extras:
                # Caller is managing hidden state manually
                feeds[in_name] = caller_extras[in_name].to(self._device)
            elif in_name in self._base_hidden:
                feeds[in_name] = self._base_hidden[in_name]
            else:
                # First step of episode: zero-initialise
                meta = next(m for m in self._base._inputs if m.name == in_name)
                shape = [base_obs.shape[0]] + [int(d) for d in meta.shape[1:]]
                feeds[in_name] = torch.zeros(shape, device=self._device)

        all_out = self._base.forward_full(**feeds)
        action = all_out[self._base._action_output_name]

        # Store updated hidden state keyed by input name for next step
        for out_name, in_name in self._base_hidden_io_map.items():
            if out_name in all_out:
                self._base_hidden[in_name] = all_out[out_name].detach()

        return action

    def _hard_step(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        slot = self._slots[self._active_idx]
        res_obs = self._apply_norm(obs, slot.obs_mean, slot.obs_std, slot.policy.num_obs)
        delta = slot.policy.forward(res_obs)
        return base_action + torch.clamp(delta, -slot.limits, slot.limits)

    def _soft_step(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        action = base_action.clone()
        for i, slot in enumerate(self._slots):
            w = self._weights[i].item()
            if w < 1e-6:
                continue
            res_obs = self._apply_norm(obs, slot.obs_mean, slot.obs_std, slot.policy.num_obs)
            delta = slot.policy.forward(res_obs)
            action = action + w * torch.clamp(delta, -slot.limits, slot.limits)
        return action

    def _advance_blend(self) -> None:
        """Advance soft-blend interpolation by one step."""
        self._blend_done += 1
        t = min(self._blend_done / self._blend_total, 1.0)
        self._weights = (1.0 - t) * self._src_weights + t * self._tgt_weights
        if t >= 1.0:
            self._src_weights = self._tgt_weights = None
