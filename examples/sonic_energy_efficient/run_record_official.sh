#!/bin/bash
# Record a reference trajectory from the official SONIC sim2sim stack,
# driving forward motion via the gamepad path over DDS (no keyboard faking).
#
# Architecture:
#   Terminal A: sim_with_gamepad.py — patched UnitreeSdk2Bridge reads
#               /tmp/wireless_remote_inject.bin into LowState.wireless_remote
#   Terminal B: this script — runs deploy.sh as a subprocess, writes gamepad
#               frames (F1 → start → ly=1.0 hold) into that same file,
#               records DDS into an npz.
#
# Before running this, start the patched sim in ANOTHER terminal:
#   cd /home/lumi/GR00T-WholeBodyControl
#   source .venv_sim/bin/activate
#   cd /home/lumi/robo_residual
#   python -m examples.sonic_energy_efficient.tools.sim_with_gamepad
#
# Then in this terminal (also .venv_sim):
#   bash examples/sonic_energy_efficient/run_record_official.sh
#
# After recording, fill tokens in the robo_residual GPU venv:
#   python examples/sonic_energy_efficient/tools/add_tokens_to_dataset.py \
#       --input /tmp/reference_dataset_300s_official.npz \
#       --output /tmp/reference_dataset_300s_official_tokens.npz

set -e

GEAR_SONIC_ROOT="${GEAR_SONIC_ROOT:-/home/lumi/GR00T-WholeBodyControl}"
DURATION="${DURATION:-300}"
TARGET_SPEED="${TARGET_SPEED:-1.5}"
OUTPUT="${OUTPUT:-/tmp/reference_dataset_${DURATION}s_official_v${TARGET_SPEED}.npz}"
TARGET_VEL="${TARGET_VEL:-$TARGET_SPEED}"
LY_FORWARD="${LY_FORWARD:-1.0}"
INIT_TIMEOUT="${INIT_TIMEOUT:-240}"

source "$GEAR_SONIC_ROOT/.venv_sim/bin/activate"

python -u /home/lumi/robo_residual/examples/sonic_energy_efficient/tools/record_official_sim.py \
    --duration "$DURATION" \
    --output "$OUTPUT" \
    --target-vel "$TARGET_VEL" \
    --target-speed "$TARGET_SPEED" \
    --ly-forward "$LY_FORWARD" \
    --init-timeout "$INIT_TIMEOUT"
