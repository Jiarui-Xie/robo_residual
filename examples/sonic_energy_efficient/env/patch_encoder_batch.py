"""Patch model_encoder.onnx to support dynamic batch size.

The released SONIC encoder has hardcoded batch=1 in its input tensor and
in 23 Reshape node constants. This script replaces those with dynamic dims
so the encoder can process N observations in one call.

Usage:
    python patch_encoder_batch.py input.onnx output.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def patch_encoder(input_path: str | Path, output_path: str | Path) -> None:
    model = onnx.load(str(input_path))
    g = model.graph

    # ---------- 1) Make input batch dim dynamic ----------
    for inp in g.input:
        dim = inp.type.tensor_type.shape.dim[0]
        if dim.dim_value == 1:
            dim.dim_value = 0          # clear fixed value
            dim.dim_param = "batch"    # symbolic param
            print(f"[patch] input '{inp.name}': batch → 'batch'")

    # ---------- 2) Make output batch dim dynamic ----------
    for out in g.output:
        dim = out.type.tensor_type.shape.dim[0]
        if dim.dim_value == 1:
            dim.dim_value = 0
            dim.dim_param = "batch"
            print(f"[patch] output '{out.name}': batch → 'batch'")

    # ---------- 3) Patch Reshape shape constants ----------
    reshape_shape_inputs = set()
    for node in g.node:
        if node.op_type == "Reshape":
            reshape_shape_inputs.add(node.input[1])

    patched = 0

    # Patch Constant nodes
    for node in g.node:
        if node.op_type != "Constant" or not node.output:
            continue
        if node.output[0] not in reshape_shape_inputs:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = numpy_helper.to_array(attr.t)
            if arr.ndim != 1 or len(arr) == 0:
                continue
            if int(arr[0]) == 1 and len(arr) > 1 and not (arr[1:] < 0).any():
                new_arr = arr.copy()
                new_arr[0] = -1
                new_tensor = numpy_helper.from_array(new_arr, name=attr.t.name)
                attr.t.CopyFrom(new_tensor)
                print(f"[patch] Constant '{node.name}': {arr.tolist()} → {new_arr.tolist()}")
                patched += 1

    # Patch initializers
    for init in g.initializer:
        if init.name not in reshape_shape_inputs:
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim != 1 or len(arr) == 0:
            continue
        if int(arr[0]) == 1 and len(arr) > 1 and not (arr[1:] < 0).any():
            new_arr = arr.copy()
            new_arr[0] = -1
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_init)
            print(f"[patch] initializer '{init.name}': {arr.tolist()} → {new_arr.tolist()}")
            patched += 1

    print(f"[patch] Patched {patched} Reshape shape constants")

    # ---------- 4) Save ----------
    onnx.save(model, str(output_path))
    print(f"[patch] Saved to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="original encoder ONNX path")
    ap.add_argument("output", help="patched ONNX path")
    args = ap.parse_args()
    patch_encoder(args.input, args.output)


if __name__ == "__main__":
    main()
