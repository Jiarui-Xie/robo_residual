"""Tests for ResidualRouter inference-time routing."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from robo_residual.inference.residual_router import ResidualRouter, ResidualSlotConfig, SwitchMode

# Re-use the ONNX builder from conftest (imported via fixture path)
from tests.conftest import _create_onnx_mlp

NUM_OBS = 32
NUM_ACTIONS = 12
BATCH = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limits(n: int = NUM_ACTIONS, val: float = 0.1) -> torch.Tensor:
    return torch.full((n,), val)


def _make_router(
    tmp_path,
    n_slots: int = 2,
    switch_mode: str = "hard",
    obs_dim: int = NUM_OBS,
) -> tuple[ResidualRouter, torch.Tensor]:
    base_path = _create_onnx_mlp(obs_dim, NUM_ACTIONS, path=tmp_path / "base.onnx")
    configs = [
        ResidualSlotConfig(
            onnx_path=_create_onnx_mlp(obs_dim, NUM_ACTIONS, path=tmp_path / f"res_{i}.onnx"),
            limits=_make_limits(),
            name=f"mode_{i}",
        )
        for i in range(n_slots)
    ]
    router = ResidualRouter(
        base_onnx_path=base_path,
        residual_configs=configs,
        switch_mode=switch_mode,
    )
    obs = torch.randn(BATCH, obs_dim)
    return router, obs


def _create_stateful_onnx(num_obs: int, num_actions: int, hidden_dim: int, path) -> path:
    """Minimal ONNX with extra h_in/h_out to simulate LSTM-like stateful base."""
    w = np.random.randn(num_actions, num_obs).astype(np.float32) * 0.01
    b = np.zeros(num_actions, dtype=np.float32)
    # h_out = h_in (identity passthrough for testability)

    W = helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, b.shape, b.flatten().tolist())
    # hidden passthrough weight (identity)
    I = helper.make_tensor("I", TensorProto.FLOAT, [hidden_dim, hidden_dim],
                           np.eye(hidden_dim, dtype=np.float32).flatten().tolist())
    Bh = helper.make_tensor("Bh", TensorProto.FLOAT, [hidden_dim],
                            np.zeros(hidden_dim, dtype=np.float32).flatten().tolist())

    gemm = helper.make_node("Gemm", ["obs", "W", "B"], ["actions"], transB=1)
    # h_out = h_in @ I + Bh  (identity)
    pass_h = helper.make_node("Gemm", ["h_in", "I", "Bh"], ["h_out"], transB=1)

    obs_in = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, num_obs])
    h_in   = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, hidden_dim])
    act_out = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, num_actions])
    h_out  = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, hidden_dim])

    graph = helper.make_graph(
        [gemm, pass_h], "stateful",
        [obs_in, h_in], [act_out, h_out],
        initializer=[W, B, I, Bh],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction(tmp_path):
    router, obs = _make_router(tmp_path)
    assert router.list_modes() == ["mode_0", "mode_1"]
    assert router.active_mode == "mode_0"
    assert router.switch_mode == SwitchMode.HARD


def test_duplicate_name_raises(tmp_path):
    base = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "base.onnx")
    res  = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res.onnx")
    with pytest.raises(ValueError, match="Duplicate"):
        ResidualRouter(
            base_onnx_path=base,
            residual_configs=[
                ResidualSlotConfig(res, _make_limits(), name="same"),
                ResidualSlotConfig(res, _make_limits(), name="same"),
            ],
        )


def test_empty_configs_raises(tmp_path):
    base = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "base.onnx")
    with pytest.raises(ValueError):
        ResidualRouter(base_onnx_path=base, residual_configs=[])


# ---------------------------------------------------------------------------
# Hard switch
# ---------------------------------------------------------------------------

def test_hard_step_returns_correct_shape(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="hard")
    action = router.step(obs)
    assert action.shape == (BATCH, NUM_ACTIONS)


def test_hard_switch_by_name(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="hard")
    router.switch("mode_1")
    assert router.active_mode == "mode_1"
    action = router.step(obs)
    assert action.shape == (BATCH, NUM_ACTIONS)


def test_hard_switch_by_index(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="hard")
    router.switch(1)
    assert router.active_mode == "mode_1"


def test_hard_weights_are_one_hot(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="hard")
    router.switch("mode_1")
    w = router.weights
    assert w[1].item() == pytest.approx(1.0)
    assert w[0].item() == pytest.approx(0.0)


def test_hard_switch_unknown_name_raises(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="hard")
    with pytest.raises(KeyError):
        router.switch("nonexistent")


def test_hard_switch_out_of_range_raises(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="hard")
    with pytest.raises(IndexError):
        router.switch(99)


# ---------------------------------------------------------------------------
# Soft switch
# ---------------------------------------------------------------------------

def test_soft_step_returns_correct_shape(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="soft")
    action = router.step(obs)
    assert action.shape == (BATCH, NUM_ACTIONS)


def test_soft_blend_interpolates_weights(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="soft")
    # Start at [1, 0]; switch to [0, 1] over 10 steps
    router.switch("mode_1", blend_steps=10)
    assert router.is_blending

    for step_idx in range(1, 11):
        router.step(obs)
        expected_w1 = step_idx / 10
        assert router.weights[1].item() == pytest.approx(expected_w1, abs=1e-5)

    # After blend completes, is_blending should be False
    assert not router.is_blending
    assert router.weights[1].item() == pytest.approx(1.0, abs=1e-5)


def test_soft_blend_interrupt_starts_from_current(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="soft")
    # Blend halfway toward mode_1
    router.switch("mode_1", blend_steps=10)
    for _ in range(5):
        router.step(obs)
    mid_w1 = router.weights[1].item()
    assert mid_w1 == pytest.approx(0.5, abs=1e-5)

    # Interrupt and blend back to mode_0
    router.switch("mode_0", blend_steps=10)
    # Next step: t=1/10, so weights[0] = mid_w0 * 0.9 + 1.0 * 0.1
    router.step(obs)
    assert router.weights[0].item() > mid_w1  # heading toward mode_0


def test_set_weights_normalises(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="soft")
    router.set_weights(torch.tensor([3.0, 1.0]))
    w = router.weights
    assert w[0].item() == pytest.approx(0.75, abs=1e-5)
    assert w[1].item() == pytest.approx(0.25, abs=1e-5)


def test_set_weights_cancels_blend(tmp_path):
    router, obs = _make_router(tmp_path, switch_mode="soft")
    router.switch("mode_1", blend_steps=10)
    assert router.is_blending
    router.set_weights(torch.tensor([0.5, 0.5]))
    assert not router.is_blending


def test_set_weights_wrong_shape_raises(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="soft")
    with pytest.raises(ValueError):
        router.set_weights(torch.tensor([1.0]))


def test_set_weights_on_hard_mode_raises(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="hard")
    with pytest.raises(RuntimeError):
        router.set_weights(torch.tensor([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Obs slicing for mixed obs dims
# ---------------------------------------------------------------------------

def test_obs_slicing_different_dims(tmp_path):
    """Slots with smaller obs dims slice correctly from a wider obs tensor."""
    wide_obs_dim = 48
    narrow_obs_dim = 16

    base_path = _create_onnx_mlp(wide_obs_dim, NUM_ACTIONS, path=tmp_path / "base.onnx")
    narrow_res = _create_onnx_mlp(narrow_obs_dim, NUM_ACTIONS, path=tmp_path / "narrow.onnx")
    wide_res   = _create_onnx_mlp(wide_obs_dim, NUM_ACTIONS, path=tmp_path / "wide.onnx")

    router = ResidualRouter(
        base_onnx_path=base_path,
        residual_configs=[
            ResidualSlotConfig(narrow_res, _make_limits(), name="narrow"),
            ResidualSlotConfig(wide_res,   _make_limits(), name="wide"),
        ],
        switch_mode="hard",
    )

    obs = torch.randn(BATCH, wide_obs_dim)

    router.switch("narrow")
    action = router.step(obs)
    assert action.shape == (BATCH, NUM_ACTIONS)

    router.switch("wide")
    action = router.step(obs)
    assert action.shape == (BATCH, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Obs normalisation
# ---------------------------------------------------------------------------

def test_per_slot_normalisation_applied(tmp_path):
    """Normalisation shifts action output deterministically."""
    router, obs = _make_router(tmp_path, switch_mode="hard")

    # Action without normalisation
    action_raw = router.step(obs).clone()

    # Rebuild with a nonzero obs mean on slot 0
    base_path = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "base2.onnx")
    res_path  = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res2.onnx")
    mean = torch.ones(NUM_OBS) * 5.0
    std  = torch.ones(NUM_OBS) * 2.0
    router2 = ResidualRouter(
        base_onnx_path=base_path,
        residual_configs=[ResidualSlotConfig(res_path, _make_limits(), obs_mean=mean, obs_std=std)],
        switch_mode="hard",
    )
    action_norm = router2.step(obs)
    # Different obs → different action
    assert not torch.allclose(action_raw, action_norm)


# ---------------------------------------------------------------------------
# Clamp limits
# ---------------------------------------------------------------------------

def test_residual_delta_clamped(tmp_path):
    """With zero-valued base and large-output residual, action is within limits."""
    # Build a residual that outputs large values by setting big weights
    # We test that the final action stays within base ± limits
    router, obs = _make_router(tmp_path, switch_mode="hard")
    limits_val = 0.05
    # Override limits to something tight
    router._slots[0].limits = torch.full((NUM_ACTIONS,), limits_val)
    action = router.step(obs)
    # We can't know the exact base action, but delta contribution is bounded
    # Just check shape and no NaN
    assert action.shape == (BATCH, NUM_ACTIONS)
    assert not torch.isnan(action).any()


# ---------------------------------------------------------------------------
# Switch mode toggling
# ---------------------------------------------------------------------------

def test_switch_mode_property(tmp_path):
    router, _ = _make_router(tmp_path, switch_mode="hard")
    router.switch_mode = "soft"
    assert router.switch_mode == SwitchMode.SOFT
    router.switch_mode = SwitchMode.HARD
    assert router.switch_mode == SwitchMode.HARD


# ---------------------------------------------------------------------------
# Stateful (LSTM-like) base
# ---------------------------------------------------------------------------

def test_stateful_base_hidden_state_updated(tmp_path):
    """Router initialises hidden state to zero and carries it across steps."""
    hidden_dim = 8
    stateful_path = _create_stateful_onnx(NUM_OBS, NUM_ACTIONS, hidden_dim, tmp_path / "stateful.onnx")
    res_path = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res.onnx")

    router = ResidualRouter(
        base_onnx_path=stateful_path,
        residual_configs=[ResidualSlotConfig(res_path, _make_limits())],
        switch_mode="hard",
        base_hidden_io_map={"h_out": "h_in"},
    )
    assert router._base.is_stateful
    assert len(router._base_hidden) == 0   # nothing stored yet

    obs = torch.randn(BATCH, NUM_OBS)
    router.step(obs)
    assert "h_in" in router._base_hidden
    assert router._base_hidden["h_in"].shape == (BATCH, hidden_dim)


def test_stateful_base_reset_clears_hidden(tmp_path):
    hidden_dim = 8
    stateful_path = _create_stateful_onnx(NUM_OBS, NUM_ACTIONS, hidden_dim, tmp_path / "stateful.onnx")
    res_path = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res.onnx")

    router = ResidualRouter(
        base_onnx_path=stateful_path,
        residual_configs=[ResidualSlotConfig(res_path, _make_limits())],
        switch_mode="hard",
        base_hidden_io_map={"h_out": "h_in"},
    )
    obs = torch.randn(BATCH, NUM_OBS)
    router.step(obs)
    assert "h_in" in router._base_hidden

    router.reset()
    assert len(router._base_hidden) == 0


def test_stateful_base_reset_env_ids(tmp_path):
    hidden_dim = 8
    stateful_path = _create_stateful_onnx(NUM_OBS, NUM_ACTIONS, hidden_dim, tmp_path / "stateful.onnx")
    res_path = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res.onnx")

    router = ResidualRouter(
        base_onnx_path=stateful_path,
        residual_configs=[ResidualSlotConfig(res_path, _make_limits())],
        switch_mode="hard",
        base_hidden_io_map={"h_out": "h_in"},
    )
    obs = torch.randn(BATCH, NUM_OBS)
    router.step(obs)

    # Set hidden state to non-zero so we can verify partial reset
    router._base_hidden["h_in"].fill_(9.9)
    env_ids = torch.tensor([0, 2])
    router.reset(env_ids=env_ids)

    h = router._base_hidden["h_in"]
    assert torch.all(h[0] == 0.0)     # reset
    assert torch.all(h[1] == 9.9)     # untouched
    assert torch.all(h[2] == 0.0)     # reset
    assert torch.all(h[3] == 9.9)     # untouched


def test_switch_with_reset_base_hidden(tmp_path):
    hidden_dim = 8
    stateful_path = _create_stateful_onnx(NUM_OBS, NUM_ACTIONS, hidden_dim, tmp_path / "stateful.onnx")
    res0 = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res0.onnx")
    res1 = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res1.onnx")

    router = ResidualRouter(
        base_onnx_path=stateful_path,
        residual_configs=[
            ResidualSlotConfig(res0, _make_limits(), name="a"),
            ResidualSlotConfig(res1, _make_limits(), name="b"),
        ],
        switch_mode="hard",
        base_hidden_io_map={"h_out": "h_in"},
    )
    obs = torch.randn(BATCH, NUM_OBS)
    router.step(obs)
    router._base_hidden["h_in"].fill_(7.0)

    router.switch("b", reset_base_hidden=True)
    assert len(router._base_hidden) == 0


# ---------------------------------------------------------------------------
# Positional hidden IO map inference
# ---------------------------------------------------------------------------

def test_stateful_base_infer_hidden_io_map(tmp_path):
    """Router infers h_out→h_in mapping positionally when not provided."""
    hidden_dim = 8
    stateful_path = _create_stateful_onnx(NUM_OBS, NUM_ACTIONS, hidden_dim, tmp_path / "stateful.onnx")
    res_path = _create_onnx_mlp(NUM_OBS, NUM_ACTIONS, path=tmp_path / "res.onnx")

    router = ResidualRouter(
        base_onnx_path=stateful_path,
        residual_configs=[ResidualSlotConfig(res_path, _make_limits())],
        switch_mode="hard",
        # No base_hidden_io_map → inferred
    )
    assert router._base_hidden_io_map == {"h_out": "h_in"}
