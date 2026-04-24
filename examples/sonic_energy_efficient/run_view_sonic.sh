#!/bin/bash
OMNI_KIT_ACCEPT_EULA=yes python -u examples/sonic_energy_efficient/view_sonic.py \
    --decoder-onnx examples/sonic_energy_efficient/models/model_decoder.onnx \
    --encoder-onnx examples/sonic_energy_efficient/models/model_encoder_dyn.onnx \
    --planner-onnx examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx \
    --cmd-vel 2.0 \
    --steps 9999
