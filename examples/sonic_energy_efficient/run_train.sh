#!/bin/bash
# SONIC × robo_residual — canonical serial training pipeline
# Runs 3.0 m/s training then 1.7 m/s training (DO NOT PARALLELIZE — IsaacLab OOMs).
#
# Prerequisites:
#   1. Re-recorded reference datasets in models/ (run tools/generate_reference_dataset.py
#      if the bridge or physics alignment has changed since last recording)
#   2. `conda activate robo` reachable
#
# Parameters that matter (all defaults are v9-validated):
#   --rsi-min-height 0.55   : running at 3 m/s pitches the body lower than the default 0.65
#   --num-envs 1024         : uses ~7.5 GB VRAM on RTX 4090
#   --iterations 2000       : training plateaus and diverges ~iter 1000-1500; best checkpoint
#                             is saved on new-best mean reward, so final-iter may be worse.
#   energy_penalty          : set in configs/reward_config.py; keep at -0.002 (do NOT ratchet up)
#
# Output per run (runs/sonic_energy_<speed>_v9/):
#   checkpoint_best.pt           ← USE THIS (saved on new-best mean reward)
#   checkpoint_iter_{50..2000}.pt (every save-interval)
#   <log is streamed to ../sonic_energy_<speed>_v9.log via tee>
set -e

# Find and activate conda robo env (avoids `conda run` stdout buffering, see Pitfall #13)
for p in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
  if [ -f "$p/etc/profile.d/conda.sh" ]; then
    source "$p/etc/profile.d/conda.sh"
    break
  fi
done
conda activate robo

# Resolve repo root from this script's location (so it works from any cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DECODER="examples/sonic_energy_efficient/models/model_decoder.onnx"

export OMNI_KIT_ACCEPT_EULA=yes
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "[v9] START $(date)"
echo "  repo=$REPO_ROOT"
echo "============================================================"

# ── 3.0 m/s ────────────────────────────────────────────────────────────────
echo "[step1] Training 3.0 m/s → runs/sonic_energy_3p0_v9"
stdbuf -oL -eL python -u \
  examples/sonic_energy_efficient/train_isaac.py \
  --decoder-onnx "$DECODER" \
  --reference-dataset examples/sonic_energy_efficient/models/dataset_3p0_isaaclab.npz \
  --rsi-min-height 0.55 \
  --num-envs 1024 \
  --iterations 2000 \
  --num-steps 24 \
  --num-epochs 5 \
  --lr 3e-4 \
  --save-interval 50 \
  --eval-interval 25 \
  --eval-steps 200 \
  --output-dir runs/sonic_energy_3p0_v9 \
  2>&1 | tee runs/sonic_energy_3p0_v9.log
echo "[step1] DONE $(date)"

# ── 1.7 m/s ────────────────────────────────────────────────────────────────
echo "[step2] Training 1.7 m/s → runs/sonic_energy_1p7_v9"
stdbuf -oL -eL python -u \
  examples/sonic_energy_efficient/train_isaac.py \
  --decoder-onnx "$DECODER" \
  --reference-dataset examples/sonic_energy_efficient/models/dataset_1p7_isaaclab.npz \
  --rsi-min-height 0.55 \
  --num-envs 1024 \
  --iterations 2000 \
  --num-steps 24 \
  --num-epochs 5 \
  --lr 3e-4 \
  --save-interval 50 \
  --eval-interval 25 \
  --eval-steps 200 \
  --output-dir runs/sonic_energy_1p7_v9 \
  2>&1 | tee runs/sonic_energy_1p7_v9.log
echo "[step2] DONE $(date)"

echo "============================================================"
echo "[v9] ALL DONE $(date)"
echo "============================================================"
