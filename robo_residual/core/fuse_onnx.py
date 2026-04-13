"""Fuse base ONNX + trained residual MLP → single deployable ONNX.

Two fusion strategies:
1. **Graph surgery** (default): Merges ONNX graphs at the graph level using
   namespace prefixing + Add/Clip nodes. Works with ANY base architecture
   (MLP, LSTM, CNN, multi-input).
2. **PyTorch re-export** (fallback): Rebuilds both models as PyTorch modules
   and re-exports. Only works for simple MLPs.

After training, call ``fuse_residual_to_onnx()`` to merge the frozen base and
the trained residual into one ONNX file that can be deployed directly.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper


def fuse_residual_to_onnx(
    base_onnx_path: str | Path,
    residual_module: nn.Module,
    max_residual_limits: torch.Tensor,
    output_path: str | Path,
    num_obs: int,
    obs_input_name: str | None = None,
    action_output_name: str = "actions",
) -> None:
    """Merge base.onnx + trained residual MLP into a single fused ONNX model.

    Uses ONNX graph surgery — works with any base architecture (MLP, LSTM, CNN).

    The fused model computes: ``action = base(obs) + clamp(residual(obs), -limits, +limits)``

    For bases with multiple inputs (e.g. LSTM with hidden states), all original
    inputs are preserved. The residual shares the main obs input with the base.

    Args:
        base_onnx_path: Path to the base policy ONNX file.
        residual_module: Trained residual MLP (PyTorch nn.Module).
        max_residual_limits: Per-joint clamp limits tensor of shape ``(num_actions,)``.
        output_path: Where to save the fused ONNX model.
        num_obs: Number of observation features for the residual input.
        obs_input_name: Name of the obs input in the base ONNX. If None,
            uses the first input name from the base model.
        action_output_name: Desired name for the fused model's action output.
    """
    base_onnx_path = Path(base_onnx_path)
    output_path = Path(output_path)
    limits_cpu = max_residual_limits.detach().cpu()

    # Export residual MLP to temp ONNX
    residual_cpu = copy.deepcopy(residual_module).cpu().eval()
    dummy_input = torch.zeros(1, num_obs)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        residual_onnx_path = Path(f.name)

    torch.onnx.export(
        residual_cpu,
        dummy_input,
        str(residual_onnx_path),
        input_names=["residual_obs"],
        output_names=["residual_delta"],
        dynamic_axes={"residual_obs": {0: "batch"}, "residual_delta": {0: "batch"}},
        opset_version=17,
    )

    # Fuse via graph surgery
    _fuse_graphs(
        base_onnx_path=str(base_onnx_path),
        residual_onnx_path=str(residual_onnx_path),
        limits=limits_cpu,
        output_path=str(output_path),
        obs_input_name=obs_input_name,
        action_output_name=action_output_name,
    )

    residual_onnx_path.unlink(missing_ok=True)


def _prefix_names(graph: onnx.GraphProto, prefix: str) -> None:
    """Add a prefix to all node outputs and initializer names in a graph (in-place).

    This prevents name collisions when merging two ONNX graphs.
    Inputs and outputs are NOT prefixed (handled separately).
    """
    # Build rename map for all internal names
    rename_map: dict[str, str] = {}

    # Prefix initializers
    for init in graph.initializer:
        new_name = prefix + init.name
        rename_map[init.name] = new_name
        init.name = new_name

    # Prefix node outputs (intermediate tensors)
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out:  # skip empty outputs
                new_name = prefix + out
                rename_map[out] = new_name
                node.output[i] = new_name

    # Update node inputs to use new names
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]

    # Update value_info
    for vi in graph.value_info:
        if vi.name in rename_map:
            vi.name = rename_map[vi.name]


def _fuse_graphs(
    base_onnx_path: str,
    residual_onnx_path: str,
    limits: torch.Tensor,
    output_path: str,
    obs_input_name: str | None,
    action_output_name: str,
) -> None:
    """Fuse two ONNX graphs: output = base_action + clamp(residual_delta, -limits, limits)."""
    base_model = onnx.load(base_onnx_path)
    residual_model = onnx.load(residual_onnx_path)

    base_graph = copy.deepcopy(base_model.graph)
    res_graph = copy.deepcopy(residual_model.graph)

    # Resolve base obs input name
    if obs_input_name is None:
        obs_input_name = base_graph.input[0].name

    # Get original output names before prefixing
    base_action_name = base_graph.output[0].name
    res_delta_name = res_graph.output[0].name
    res_obs_input_name = res_graph.input[0].name

    # Prefix all internal names to avoid collisions
    _prefix_names(base_graph, "base/")
    _prefix_names(res_graph, "res/")

    # Fix: update the graph input/output names to use prefixed internal names
    # but keep the external-facing names clean
    base_action_prefixed = "base/" + base_action_name
    res_delta_prefixed = "res/" + res_delta_name

    # _prefix_names does NOT rename graph input references in nodes
    # (since graph inputs aren't in the rename map). So residual nodes still
    # reference "residual_obs" directly. Rename it to share the base's obs input.
    for node in res_graph.node:
        for i, inp in enumerate(node.input):
            if inp == res_obs_input_name:
                node.input[i] = obs_input_name

    # Create clamp limits as initializers
    # Clip op requires scalar min/max, so use Min/Max for per-joint clamping
    limits_np = limits.numpy().astype(np.float32)
    neg_limits_np = (-limits).numpy().astype(np.float32)

    max_limit_init = numpy_helper.from_array(limits_np, name="fuse/max_limits")
    min_limit_init = numpy_helper.from_array(neg_limits_np, name="fuse/min_limits")

    # Per-joint clamp: clamped = max(min_limits, min(delta, max_limits))
    min_node = helper.make_node(
        "Min",
        inputs=[res_delta_prefixed, "fuse/max_limits"],
        outputs=["fuse/clipped_upper"],
    )
    max_node = helper.make_node(
        "Max",
        inputs=["fuse/clipped_upper", "fuse/min_limits"],
        outputs=["fuse/clamped_delta"],
    )
    add_node = helper.make_node(
        "Add",
        inputs=[base_action_prefixed, "fuse/clamped_delta"],
        outputs=[action_output_name],
    )

    # Determine output shape from base
    base_output_type = base_graph.output[0].type
    fused_output = helper.make_tensor_value_info(
        action_output_name, TensorProto.FLOAT,
        [d.dim_param or d.dim_value for d in base_output_type.tensor_type.shape.dim]
    )
    # Make batch dim dynamic
    fused_output.type.tensor_type.shape.dim[0].dim_param = "batch"
    fused_output.type.tensor_type.shape.dim[0].ClearField("dim_value")

    # Collect all inputs (base inputs, skip residual's obs since it shares with base)
    fused_inputs = list(base_graph.input)
    # Make batch dim dynamic on base inputs
    for inp in fused_inputs:
        if inp.type.tensor_type.shape.dim:
            inp.type.tensor_type.shape.dim[0].dim_param = "batch"
            inp.type.tensor_type.shape.dim[0].ClearField("dim_value")

    # Build fused graph
    all_nodes = list(base_graph.node) + list(res_graph.node) + [min_node, max_node, add_node]
    all_initializers = (
        list(base_graph.initializer)
        + list(res_graph.initializer)
        + [max_limit_init, min_limit_init]
    )

    fused_graph = helper.make_graph(
        all_nodes,
        "fused_residual",
        fused_inputs,
        [fused_output],
        initializer=all_initializers,
    )

    # Copy value_info for intermediate tensors
    for vi in list(base_graph.value_info) + list(res_graph.value_info):
        fused_graph.value_info.append(vi)

    # Determine opset version (take max of both models)
    base_opset = base_model.opset_import[0].version if base_model.opset_import else 17
    res_opset = residual_model.opset_import[0].version if residual_model.opset_import else 17
    opset = max(base_opset, res_opset)

    fused_model = helper.make_model(fused_graph, opset_imports=[helper.make_opsetid("", opset)])
    fused_model.ir_version = max(base_model.ir_version, residual_model.ir_version)

    # Validate
    onnx.checker.check_model(fused_model)
    onnx.save(fused_model, output_path)
