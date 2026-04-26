from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _create_onnx_mlp(
    num_obs: int,
    num_actions: int,
    hidden_dim: int = 32,
    path: Path | None = None,
    input_name: str = "obs",
    output_name: str = "actions",
) -> Path:
    """Create a simple Linear ONNX model: obs → Linear → ReLU → Linear → actions."""
    # Layer 1: Linear(num_obs, hidden_dim)
    w1 = np.random.randn(hidden_dim, num_obs).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    # Layer 2: Linear(hidden_dim, num_actions)
    w2 = np.random.randn(num_actions, hidden_dim).astype(np.float32) * 0.1
    b2 = np.zeros(num_actions, dtype=np.float32)

    W1 = helper.make_tensor("W1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist())
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, b1.shape, b1.flatten().tolist())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist())
    B2 = helper.make_tensor("B2", TensorProto.FLOAT, b2.shape, b2.flatten().tolist())

    # Gemm1: hidden = obs @ W1^T + B1
    gemm1 = helper.make_node("Gemm", [input_name, "W1", "B1"], ["hidden"], transB=1)
    relu = helper.make_node("Relu", ["hidden"], ["hidden_relu"])
    gemm2 = helper.make_node("Gemm", ["hidden_relu", "W2", "B2"], [output_name], transB=1)

    obs_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, num_obs])
    action_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, num_actions])

    graph = helper.make_graph(
        [gemm1, relu, gemm2],
        "test_mlp",
        [obs_input],
        [action_output],
        initializer=[W1, B1, W2, B2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    if path is None:
        path = Path("/tmp/test_base.onnx")
    onnx.save(model, str(path))
    return path


@pytest.fixture
def dummy_onnx(tmp_path) -> Path:
    """Create a simple 64→29 ONNX MLP model."""
    return _create_onnx_mlp(64, 29, path=tmp_path / "base.onnx")


@pytest.fixture
def dummy_onnx_small(tmp_path) -> Path:
    """Create a small 10→4 ONNX MLP model."""
    return _create_onnx_mlp(10, 4, path=tmp_path / "base_small.onnx")


@pytest.fixture
def simple_joint_groups():
    return [
        JointGroupConfig("legs", list(range(12)), 0.05),
        JointGroupConfig("arms", list(range(12, 29)), 0.15),
    ]


@pytest.fixture
def simple_config(simple_joint_groups):
    return ResidualConfig(
        joint_groups=simple_joint_groups,
        residual_hidden_dims=[64, 32],
        residual_activation="elu",
        init_noise_std=0.1,
    )


@pytest.fixture
def small_config():
    return ResidualConfig(
        joint_groups=[
            JointGroupConfig("group_a", [0, 1], 0.1),
            JointGroupConfig("group_b", [2, 3], 0.3),
        ],
        residual_hidden_dims=[16, 8],
        residual_activation="elu",
        init_noise_std=0.1,
    )


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture
def lstm_config(simple_joint_groups):
    return ResidualConfig(
        joint_groups=simple_joint_groups,
        residual_type="lstm",
        residual_lstm_hidden_dim=32,
        residual_lstm_num_layers=1,
        init_noise_std=0.1,
    )


@pytest.fixture
def gru_config(simple_joint_groups):
    return ResidualConfig(
        joint_groups=simple_joint_groups,
        residual_type="gru",
        residual_gru_hidden_dim=32,
        residual_gru_num_layers=1,
        init_noise_std=0.1,
    )
