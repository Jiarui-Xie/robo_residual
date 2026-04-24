"""Live per-step per-env SONIC encoder provider.

Runs the patched (dynamic-batch) encoder at every control step, using:
- Per-env current_frame advancing through a precomputed motion reference
- Per-env robot base quaternion for accurate anchor_orientation
- Motion reference selected by rounding command velocity to nearest bucket

This is the "real SONIC" path — dynamic tokens, not cached approximations.

Architecture:
    1. At setup: run planner once per velocity bucket → motion_50hz references
    2. At each step:
       a. For each env: pick motion_ref by velocity, build encoder input with current_frame and actual base_quat
       b. Stack all envs → (N, 1762) batch
       c. Run patched encoder → (N, 64) tokens
       d. Return tokens to bridge
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from .sonic_pipeline import (
    SonicPlanner,
    build_encoder_input,
    calc_heading_quat,
    compute_joint_velocities_50hz,
    resample_30hz_to_50hz,
)
from ..configs.sonic_params import MODE_RUN


class SonicLiveEncoder:
    """Dynamic SONIC token provider: runs encoder each step with per-env data.

    Motion references are either (a) precomputed per velocity bucket via the
    SONIC planner, or (b) loaded from a pre-recorded .npz (longer, cyclic).
    anchor_orientation uses the real robot quaternion each step.
    """

    def __init__(
        self,
        planner_onnx: str | Path,
        encoder_onnx_dyn: str | Path,  # MUST be the batch-patched version
        num_envs: int,
        velocity_buckets: Optional[list[float]] = None,
        default_height: float = 0.72,
        device: str = "cuda",
        recorded_motion_path: Optional[str | Path] = None,
    ) -> None:
        if velocity_buckets is None:
            velocity_buckets = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        self.velocity_buckets = np.array(velocity_buckets, dtype=np.float32)
        self.num_envs = num_envs
        self.device = device

        motions = []  # list of [T, 36] arrays
        jvels = []    # list of [T, 29] arrays

        if recorded_motion_path is not None:
            # Use a single long recorded motion for ALL velocity buckets
            print(f"[live_encoder] Loading recorded motion: {recorded_motion_path}")
            rec = np.load(recorded_motion_path)
            rec_motion = rec["motion_50hz"].astype(np.float32)
            rec_jvel = rec["jvel_50hz"].astype(np.float32)
            for v in velocity_buckets:
                motions.append(rec_motion.copy())
                jvels.append(rec_jvel.copy())
            print(f"  Loaded {len(rec_motion)} frames ({len(rec_motion)/50.0:.1f}s) for all buckets")
        else:
            # Fallback: run planner once per bucket
            print(f"[live_encoder] Loading planner ({Path(planner_onnx).stat().st_size/1e6:.0f}MB)...")
            planner = SonicPlanner(planner_onnx)
            for v in velocity_buckets:
                qpos_30hz, num_valid = planner.run(
                    target_vel=float(v), mode=MODE_RUN, default_height=default_height,
                )
                m = resample_30hz_to_50hz(qpos_30hz, num_valid)
                jv = compute_joint_velocities_50hz(m)
                motions.append(m)
                jvels.append(jv)
                n = min(50, len(m) - 1)
                vx = (m[n, 0] - m[0, 0]) * 50.0 / n
                print(f"  v={v:.2f}: {num_valid} 30Hz → {len(m)} 50Hz frames, achieved vx={vx:.2f} m/s")
            del planner

        # Pad motions to same length for GPU indexing
        max_len = max(len(m) for m in motions)
        self.motion_len = max_len
        padded_motions = np.zeros((len(motions), max_len, 36), dtype=np.float32)
        padded_jvels = np.zeros((len(motions), max_len, 29), dtype=np.float32)
        self.motion_valid = np.zeros(len(motions), dtype=np.int64)
        for i, (m, jv) in enumerate(zip(motions, jvels)):
            T = len(m)
            padded_motions[i, :T] = m
            padded_jvels[i, :T] = jv
            # Pad with last frame (hold pose)
            padded_motions[i, T:] = m[-1]
            padded_jvels[i, T:] = jv[-1]
            self.motion_valid[i] = T

        # Keep on CPU — encoder input building happens on CPU before batching
        self.motions_padded = padded_motions  # (B, T, 36)
        self.jvels_padded = padded_jvels      # (B, T, 29)

        # Precompute init quats per bucket
        self.init_motion_quats = np.stack([m[0, 3:7].astype(np.float64) for m in motions])  # (B, 4)
        self.init_heading_quats = np.stack([calc_heading_quat(q) for q in self.init_motion_quats])  # (B, 4)

        # ---- 2) Load patched encoder ----
        print(f"[live_encoder] Loading patched encoder ({Path(encoder_onnx_dyn).stat().st_size/1e6:.0f}MB)...")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(encoder_onnx_dyn), providers=providers)

        # ---- 3) Per-env state: current_frame, bucket_idx ----
        self.current_frame = np.zeros(num_envs, dtype=np.int64)
        self.bucket_idx = np.zeros(num_envs, dtype=np.int64)

        print(f"[live_encoder] Ready: {len(velocity_buckets)} buckets × {max_len} max frames")

    def reset(self, env_ids: Optional[np.ndarray] = None) -> None:
        """Reset current_frame to 0 for specified envs (or all)."""
        if env_ids is None:
            self.current_frame.fill(0)
        else:
            self.current_frame[env_ids] = 0

    def step_frame(self) -> None:
        """Advance current_frame by 1 for all envs (wrap at motion end)."""
        valid_lens = self.motion_valid[self.bucket_idx]
        self.current_frame = (self.current_frame + 1) % valid_lens

    def update_buckets(self, velocity_cmd_np: np.ndarray) -> None:
        """Update each env's bucket_idx based on its velocity command."""
        dists = np.abs(velocity_cmd_np[:, None] - self.velocity_buckets[None, :])
        self.bucket_idx = dists.argmin(axis=1).astype(np.int64)

    def get_tokens(self, velocity_cmd: Tensor, robot_base_quat: Tensor) -> Tensor:
        """Build per-env encoder input and run batched encoder.

        Args:
            velocity_cmd: (N,) forward velocity command
            robot_base_quat: (N, 4) wxyz robot base quaternion

        Returns:
            (N, 64) token tensor on self.device
        """
        vel_np = velocity_cmd.detach().cpu().numpy()
        quat_np = robot_base_quat.detach().cpu().numpy().astype(np.float64)

        self.update_buckets(vel_np)

        # Build 1762D input for each env
        N = self.num_envs
        enc_in = np.zeros((N, 1762), dtype=np.float32)

        for i in range(N):
            b = int(self.bucket_idx[i])
            f = int(self.current_frame[i])
            enc_in[i] = build_encoder_input(
                motion_50hz=self.motions_padded[b],
                motion_joint_vel_50hz=self.jvels_padded[b],
                current_frame=f,
                robot_base_quat=quat_np[i],
                init_motion_quat=self.init_motion_quats[b],
                init_heading_quat=self.init_heading_quats[b],
            )

        # Batched encoder inference
        tokens = self.encoder.run(None, {"obs_dict": enc_in})[0]  # (N, 64)
        return torch.from_numpy(tokens).to(self.device)
