"""Fuse trained residual into SONIC decoder ONNX for deployment.

Usage:
    python fuse.py --decoder-onnx models/model_decoder.onnx \
                   --checkpoint runs/sonic_energy/checkpoint_best.pt \
                   --output fused_decoder.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
import torch
import numpy as np

from robo_residual import ResidualActorCritic, fuse_residual_to_onnx

try:
    from .configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from .configs.sonic_residual_config import build_sonic_residual_config
except ImportError:
    from configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from configs.sonic_residual_config import build_sonic_residual_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse residual into SONIC decoder ONNX")
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="fused_decoder.onnx")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verify", action="store_true", default=True,
                   help="Verify fused output matches separate computation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Rebuild policy
    residual_cfg = build_sonic_residual_config()
    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=residual_cfg,
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)

    policy.residual.load_state_dict(ckpt["residual_state_dict"])
    print(f"[fuse] Loaded residual from iteration {ckpt['iteration']}")

    # Fuse
    output_path = Path(args.output)
    fuse_residual_to_onnx(
        base_onnx_path=args.decoder_onnx,
        residual_module=policy.residual,
        max_residual_limits=policy.max_residual_limits,
        output_path=str(output_path),
        num_obs=OBS_LAYOUT.total_dim,
    )
    print(f"[fuse] Fused model saved to: {output_path}")

    # Verify
    if args.verify:
        print("[fuse] Verifying fused output matches separate computation...")
        test_obs = torch.randn(4, OBS_LAYOUT.total_dim, device=device)

        # Separate: base + residual
        with torch.no_grad():
            expected = policy.act_inference(test_obs).cpu().numpy()

        # Fused ONNX
        sess = ort.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name
        fused_out = sess.run(None, {input_name: test_obs.cpu().numpy()})[0]

        max_diff = np.max(np.abs(expected - fused_out))
        print(f"[fuse] Max difference: {max_diff:.8f}")

        if max_diff < 1e-4:
            print("[fuse] PASSED — fused model matches separate computation")
        else:
            print(f"[fuse] WARNING — difference {max_diff:.6f} exceeds tolerance")

    print("\n[fuse] Deployment:")
    print(f"  Encoder:  model_encoder.onnx (unchanged)")
    print(f"  Decoder:  {output_path} (base + residual fused)")
    print(f"  Config:   observation_config.yaml (unchanged)")


if __name__ == "__main__":
    main()
