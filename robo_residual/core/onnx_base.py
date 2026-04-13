"""ONNX base policy wrapper.

Loads any ONNX policy file, makes the batch dimension dynamic, and provides
a simple forward() interface that accepts and returns PyTorch tensors.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def _make_batch_dynamic(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make the first dimension of all inputs/outputs dynamic (batch)."""
    model = copy.deepcopy(model)
    for tensor in list(model.graph.input) + list(model.graph.output):
        if tensor.type.tensor_type.shape.dim:
            dim = tensor.type.tensor_type.shape.dim[0]
            dim.ClearField("dim_value")
            dim.dim_param = "batch"
    return model


class OnnxBasePolicy:
    """Wraps an ONNX model as a frozen base policy.

    Auto-detects input/output names and dimensions from the ONNX graph.
    Makes batch dimension dynamic so any batch size works at inference.

    Args:
        onnx_path: Path to the ONNX model file.
        device: Device for output tensors ("cpu" or "cuda").
        obs_input: Which input is the main observation — index (int) or name (str).
        action_output: Which output is the action — index (int) or name (str).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cpu",
        obs_input: int | str = 0,
        action_output: int | str = 0,
    ) -> None:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Load and make batch dynamic
        raw_model = onnx.load(str(onnx_path))
        dynamic_model = _make_batch_dynamic(raw_model)

        # Create session from modified model bytes
        providers = ["CPUExecutionProvider"]
        if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            dynamic_model.SerializeToString(), sess_options=sess_opts, providers=providers
        )
        self._device = device

        # Parse input metadata
        self._inputs = self._session.get_inputs()
        self._outputs = self._session.get_outputs()
        self._input_names = [inp.name for inp in self._inputs]
        self._output_names = [out.name for out in self._outputs]

        # Resolve main obs input
        if isinstance(obs_input, int):
            self._obs_input_name = self._input_names[obs_input]
            self._obs_input_idx = obs_input
        else:
            self._obs_input_name = obs_input
            self._obs_input_idx = self._input_names.index(obs_input)

        # Resolve main action output
        if isinstance(action_output, int):
            self._action_output_name = self._output_names[action_output]
            self._action_output_idx = action_output
        else:
            self._action_output_name = action_output
            self._action_output_idx = self._output_names.index(action_output)

        # Extract dims (skip batch dim at index 0)
        obs_shape = self._inputs[self._obs_input_idx].shape
        action_shape = self._outputs[self._action_output_idx].shape
        self._num_obs = int(obs_shape[-1])
        self._num_actions = int(action_shape[-1])

    @property
    def num_obs(self) -> int:
        return self._num_obs

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def is_stateful(self) -> bool:
        """True if the model has more outputs than just the action (e.g. LSTM hidden states)."""
        return len(self._outputs) > 1

    def forward(self, obs: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Run the base policy.

        Args:
            obs: Observation tensor of shape ``(batch, num_obs)``.
            **extra_inputs: Additional named inputs (e.g. ``h_in``, ``c_in`` for LSTM).

        Returns:
            Action tensor of shape ``(batch, num_actions)`` on ``self._device``.
        """
        feeds = {self._obs_input_name: obs.detach().cpu().numpy().astype(np.float32)}
        for name, tensor in extra_inputs.items():
            feeds[name] = tensor.detach().cpu().numpy().astype(np.float32)

        outputs = self._session.run(None, feeds)
        action_np = outputs[self._action_output_idx]
        return torch.from_numpy(action_np).to(self._device)

    def forward_full(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run with all named inputs, return all named outputs.

        Useful for stateful models (LSTM) where you need hidden state outputs.

        Args:
            **inputs: All named inputs as tensors.

        Returns:
            Dict mapping output names to tensors on ``self._device``.
        """
        feeds = {}
        for name, tensor in inputs.items():
            feeds[name] = tensor.detach().cpu().numpy().astype(np.float32)

        outputs = self._session.run(None, feeds)
        return {
            name: torch.from_numpy(out).to(self._device)
            for name, out in zip(self._output_names, outputs)
        }
