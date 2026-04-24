"""Velocity-bucketed cache of SONIC tokens for fast training-time lookup.

At setup, precomputes tokens for a grid of velocity commands by running
the full planner → encoder pipeline.  During training, each env looks up
its token by rounding its velocity command to the nearest bucket.

For a first training pass we use a SINGLE token per velocity bucket
(frame 0 of motion reference, robot assumed perfectly aligned).  This loses
per-step motion-tracking accuracy but makes encoder calls negligible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .sonic_pipeline import (
    SonicEncoder, SonicPlanner,
    build_encoder_input, calc_heading_quat,
    compute_joint_velocities_50hz, resample_30hz_to_50hz,
)
from ..configs.sonic_params import MODE_RUN, MODE_WALK


class VelocityBucketTokenCache:
    """Precomputes SONIC tokens at (velocity_bucket, frame) grid for fast lookup.

    Each env maintains a current_frame that advances by 1 per control step.
    Tokens are looked up per-env using (bucket_idx, current_frame).

    Args:
        planner_onnx: path to planner_sonic.onnx
        encoder_onnx: path to model_encoder.onnx
        velocity_buckets: list of target velocities to precompute (m/s)
        num_cached_frames: how many token frames to precompute per bucket.
            Beyond this, current_frame wraps (motion loops).
        default_height: robot standing height
        device: torch device
    """

    def __init__(
        self,
        planner_onnx: str | Path,
        encoder_onnx: str | Path,
        velocity_buckets: list[float] | None = None,
        num_cached_frames: int = 30,
        default_height: float = 0.72,
        device: str = "cuda",
    ) -> None:
        if velocity_buckets is None:
            velocity_buckets = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        self.velocity_buckets = np.array(velocity_buckets, dtype=np.float32)
        self.num_cached_frames = num_cached_frames
        self.device = device

        print(f"[token_cache] Loading planner ({Path(planner_onnx).stat().st_size/1e6:.0f}MB)...")
        planner = SonicPlanner(planner_onnx)
        print(f"[token_cache] Loading encoder...")
        encoder = SonicEncoder(encoder_onnx)

        print(f"[token_cache] Precomputing tokens: {len(velocity_buckets)} buckets × {num_cached_frames} frames...")

        # Tokens: (num_buckets, num_frames, 64)
        all_tokens = np.zeros((len(velocity_buckets), num_cached_frames, 64), dtype=np.float32)
        motions = []

        for b_idx, v in enumerate(velocity_buckets):
            mode = MODE_RUN  # 2-3 m/s is running gait
            qpos_30hz, num_valid = planner.run(target_vel=float(v), mode=mode, default_height=default_height)
            motion_50hz = resample_30hz_to_50hz(qpos_30hz, num_valid)
            motion_jvel = compute_joint_velocities_50hz(motion_50hz)

            # Measure achieved velocity
            n = min(50, len(motion_50hz) - 1)
            vx = (motion_50hz[n, 0] - motion_50hz[0, 0]) * 50.0 / n

            # Precompute token for each starting frame
            init_motion_quat = motion_50hz[0, 3:7].astype(np.float64)
            init_heading = calc_heading_quat(init_motion_quat)

            for f in range(num_cached_frames):
                current_frame = min(f, len(motion_50hz) - 1)
                # Assume robot perfectly tracks motion → base_quat = motion quat at current frame
                robot_quat = motion_50hz[current_frame, 3:7].astype(np.float64)
                enc_in = build_encoder_input(
                    motion_50hz=motion_50hz,
                    motion_joint_vel_50hz=motion_jvel,
                    current_frame=current_frame,
                    robot_base_quat=robot_quat,
                    init_motion_quat=init_motion_quat,
                    init_heading_quat=init_heading,
                )
                all_tokens[b_idx, f] = encoder.run(enc_in)

            motions.append(motion_50hz)
            print(f"  v={v:.2f} mode={mode}: {num_valid} 30Hz → {len(motion_50hz)} 50Hz frames, "
                  f"achieved vx={vx:.2f} m/s, {num_cached_frames} tokens cached")

        self.tokens = torch.tensor(all_tokens, device=device, dtype=torch.float32)  # (B, F, 64)
        self.motions = motions

        # Clear ONNX sessions
        del planner, encoder

    def get_tokens_for(self, velocity_cmd: Tensor, current_frame: Tensor) -> Tensor:
        """Lookup tokens per env by (velocity bucket, frame).

        Args:
            velocity_cmd: (N,) forward velocity command per env
            current_frame: (N,) int64 current motion frame (0-based), wrapped externally

        Returns:
            (N, 64) token tensor
        """
        buckets = torch.tensor(self.velocity_buckets, device=velocity_cmd.device)
        dists = (velocity_cmd.unsqueeze(-1) - buckets.unsqueeze(0)).abs()
        bucket_idx = dists.argmin(dim=-1)  # (N,)
        frame_clamped = current_frame.clamp(0, self.num_cached_frames - 1)
        return self.tokens[bucket_idx, frame_clamped]  # (N, 64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path,
            velocity_buckets=self.velocity_buckets,
            tokens=self.tokens.cpu().numpy(),
            num_cached_frames=np.array([self.num_cached_frames]),
            **{f"motion_{i}": m for i, m in enumerate(self.motions)},
        )
        print(f"[token_cache] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "VelocityBucketTokenCache":
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.velocity_buckets = data["velocity_buckets"]
        obj.tokens = torch.tensor(data["tokens"], device=device, dtype=torch.float32)
        # Handle old cache format (single token per bucket → shape (B, 64)) vs new (B, F, 64)
        if obj.tokens.ndim == 2:
            obj.tokens = obj.tokens.unsqueeze(1)  # (B, 1, 64)
            obj.num_cached_frames = 1
        else:
            obj.num_cached_frames = int(data["num_cached_frames"][0]) if "num_cached_frames" in data else obj.tokens.shape[1]
        obj.motions = [data[f"motion_{i}"] for i in range(len(obj.velocity_buckets))]
        obj.device = device
        return obj
