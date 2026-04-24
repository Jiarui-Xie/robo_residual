"""Fill the `tokens` field of a reference dataset npz by replaying it through
SonicOnlinePlannerEncoder (GPU).

Runs in the robo_residual venv (not .venv_sim) because it needs onnxruntime-gpu
+ torch CUDA.

Usage (from /home/lumi/robo_residual):
    python -u examples/sonic_energy_efficient/tools/add_tokens_to_dataset.py \\
        --input /tmp/reference_dataset_300s_official.npz \\
        --output /tmp/reference_dataset_300s_official_tokens.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401

import numpy as np
import torch

from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder

ENCODER_ONNX = "examples/sonic_energy_efficient/models/model_encoder_dyn.onnx"
PLANNER_ONNX = "examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="reference_dataset.npz without tokens (or with zero tokens).")
    ap.add_argument("--output", required=True, help="Output path with tokens filled in.")
    ap.add_argument("--planner-speed", type=float, default=None,
                    help="Planner target speed for token generation. Defaults to cmd_buckets[0]. "
                         "Override when recorded planner speed differs from stored cmd_vel label "
                         "(e.g. recorded at 1.5 but labelled 1.7).")
    ap.add_argument("--replan-interval-steps", type=int, default=5)
    ap.add_argument("--motion-look-ahead-steps", type=int, default=5)
    args = ap.parse_args()

    data = dict(np.load(args.input))
    T = data["root_state_w"].shape[0]
    label_vel = float(data["cmd_buckets"][0])
    planner_vel = args.planner_speed if args.planner_speed is not None else label_vel
    print(f"[add_tokens] {T} frames, label_vel={label_vel:.2f} m/s, planner_vel={planner_vel:.2f} m/s")

    device = "cuda"
    token_provider = SonicOnlinePlannerEncoder(
        planner_onnx=PLANNER_ONNX,
        encoder_onnx_dyn=ENCODER_ONNX,
        num_envs=1,
        target_vel=planner_vel,
        device=device,
        replan_interval_steps=args.replan_interval_steps,
        motion_look_ahead_steps=args.motion_look_ahead_steps,
    )
    token_provider.set_target_vel(np.array([0]), np.array([planner_vel]))

    tokens = np.zeros((T, 64), dtype=np.float32)
    quats = data["root_state_w"][:, 3:7]  # wxyz

    with torch.no_grad():
        for t in range(T):
            quat_t = torch.tensor(quats[t], dtype=torch.float32, device=device).unsqueeze(0)
            vel_t = torch.tensor([planner_vel], dtype=torch.float32, device=device)
            tok = token_provider.get_tokens(velocity_cmd=vel_t, robot_base_quat=quat_t)
            tokens[t] = tok[0].cpu().numpy()
            token_provider.step_frame()
            if t % 500 == 0:
                print(f"  t={t}/{T}")

    data["tokens"] = tokens

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"[add_tokens] saved → {out}")


if __name__ == "__main__":
    main()
