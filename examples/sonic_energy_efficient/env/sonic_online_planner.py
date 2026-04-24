"""Online planner-based token provider (deploy-faithful).

Mirrors the C++ deploy reference (`localmotion_kplanner.hpp::UpdateContextFromMotion`
and the splice in `g1_deploy_onnx_ref.cpp::2890-2960`):

  1. Context for the planner is sampled from the CURRENT planner motion at a
     look-ahead frame (`gen_frame = current_frame + motion_look_ahead_steps`),
     4 samples at 30 Hz spacing.  XY is NOT zeroed — the raw resampled pose
     (position, quat, joint pos) is fed in as-is.  This makes the planner
     generate a CONTINUATION of the already-running motion rather than a
     fresh stand→walk→run ramp.

  2. At replan, the new planner output is spliced into the running motion
     starting at `gen_frame` with an 8-frame linear blend (same as deploy's
     `blend_num_frames = 8`).  Old frames up to gen_frame are kept, then blend
     into new, then pure new afterwards.

  3. At cold start (no prior motion), context is a default standing pose
     (xy=0, z=default_height, quat=identity, joints=SONIC_DEFAULT_ANGLES) —
     matches deploy's `InitializeContext`.  The resulting motion has the
     stand→run ramp, which is expected ONCE on reset and never again.

Designed for SMALL env counts (1-8) since the planner is batch=1 on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from .sonic_pipeline import (
    SonicPlanner, build_encoder_input, calc_heading_quat,
    compute_joint_velocities_50hz, resample_30hz_to_50hz, slerp,
)
from ..configs.sonic_params import (
    MODE_RUN, MODE_WALK, SONIC_DEFAULT_ANGLES,
)


class SonicOnlinePlannerEncoder:
    """Online planner + encoder. Per-env motion reference updated via planner replays.

    Args:
        planner_onnx: SONIC planner model path.
        encoder_onnx_dyn: Patched dynamic-batch encoder.
        num_envs: Number of parallel envs.
        target_vel: Velocity command for planner (default 2.5).
        mode: Locomotion mode (MODE_WALK, MODE_RUN).
        replan_interval_steps: Replan every N control steps (50 Hz). Default 25
            (0.5s at 50Hz).  Smaller = tighter tracking but higher CPU cost.
        device: Torch device.
    """

    def __init__(
        self,
        planner_onnx: str | Path,
        encoder_onnx_dyn: str | Path,
        num_envs: int,
        target_vel: float = 2.5,
        mode: int = MODE_RUN,
        replan_interval_steps: int = 5,         # deploy: planner thread @ 10Hz → every 5 50Hz frames
        motion_look_ahead_steps: int = 5,       # deploy: gen_frame = current_frame + 5 (0.1s)
        default_height: float = 0.72,
        blend_frames: int = 8,                   # deploy: blend_num_frames = 8
        device: str = "cuda",
        **_ignored_kwargs,                       # swallow deprecated params like trim_startup_*
    ) -> None:
        self.num_envs = num_envs
        self.target_vel = np.full(num_envs, float(target_vel), dtype=np.float32)
        self.mode = mode
        self.replan_interval = replan_interval_steps
        self.motion_look_ahead = motion_look_ahead_steps
        self.blend_frames = blend_frames
        self.default_height = default_height
        self.device = device

        print(f"[online_planner] Loading planner ({Path(planner_onnx).stat().st_size/1e6:.0f}MB)...")
        self.planner = SonicPlanner(planner_onnx)
        print(f"[online_planner] Loading patched encoder...")
        self.encoder = ort.InferenceSession(
            str(encoder_onnx_dyn),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Per-env motion cache
        self._motions = [None] * num_envs      # list of [T, 36]
        self._jvels = [None] * num_envs        # list of [T, 29]
        self._current_frame = np.zeros(num_envs, dtype=np.int64)
        self._steps_since_replan = np.zeros(num_envs, dtype=np.int64)
        self._init_motion_quats = np.zeros((num_envs, 4), dtype=np.float64)
        self._init_heading_quats = np.zeros((num_envs, 4), dtype=np.float64)

        # Initial plan from default standing pose (deploy's InitializeContext)
        print(f"[online_planner] Initial plan for {num_envs} envs (cold start from stand)...")
        self._replan(np.arange(num_envs))
        print(f"[online_planner] Ready: replan every {replan_interval_steps} frames "
              f"({replan_interval_steps*0.02:.2f}s), look-ahead {motion_look_ahead_steps} frames")

    def _build_standing_context(self) -> np.ndarray:
        """Cold-start context: 4 identical frames of default standing pose."""
        ctx = np.zeros((4, 36), dtype=np.float32)
        ctx[:, 2] = self.default_height   # z
        ctx[:, 3] = 1.0                    # quat w = 1 (identity)
        ctx[:, 7:36] = SONIC_DEFAULT_ANGLES
        return ctx

    def _sample_context_from_motion(self, motion: np.ndarray, gen_frame: int) -> np.ndarray:
        """Mirror of deploy UpdateContextFromMotion: 4 frames @ 30Hz from motion[gen_frame:]."""
        ctx = np.zeros((4, 36), dtype=np.float32)
        gen_time = gen_frame / 50.0
        T = len(motion)
        for n in range(4):
            t = gen_time + n / 30.0
            f_50hz = t * 50.0
            f0 = min(int(np.floor(f_50hz)), T - 1)
            f1 = min(f0 + 1, T - 1)
            w1 = float(f_50hz - f0)
            w0 = 1.0 - w1
            # xyz lerp
            ctx[n, 0:3] = w0 * motion[f0, 0:3] + w1 * motion[f1, 0:3]
            # quat slerp
            q0 = motion[f0, 3:7].astype(np.float64)
            q1 = motion[f1, 3:7].astype(np.float64)
            ctx[n, 3:7] = slerp(q0, q1, w1).astype(np.float32)
            # joints lerp
            ctx[n, 7:36] = w0 * motion[f0, 7:36] + w1 * motion[f1, 7:36]
        return ctx

    def _splice_motion(self, prev: np.ndarray, prev_jvel: np.ndarray,
                       new: np.ndarray, new_jvel: np.ndarray,
                       current_frame: int, gen_frame: int) -> tuple:
        """Mirror of deploy splice in g1_deploy_onnx_ref.cpp:2890-2960.

        New timeline index f corresponds to:
          f_old = f + current_frame  (in prev motion)
          f_new = f_old - gen_frame   (in new motion)
        Blend over `blend_frames` starting at `blend_start = max(0, gen_frame - current_frame)`.
        """
        total = (gen_frame - current_frame) + len(new)
        total = max(total, 1)
        blend_start = max(0, gen_frame - current_frame)
        B = self.blend_frames
        out_m = np.zeros((total, 36), dtype=np.float32)
        out_jv = np.zeros((total, 29), dtype=np.float32)
        T_prev = len(prev)
        T_new = len(new)
        for f in range(total):
            f_old = max(0, min(f + current_frame, T_prev - 1))
            f_new = max(0, min(f + current_frame - gen_frame, T_new - 1))
            if f < blend_start:
                w_new = 0.0
            elif f >= blend_start + B:
                w_new = 1.0
            else:
                w_new = (f - blend_start) / float(B)
            w_old = 1.0 - w_new
            # xyz
            out_m[f, 0:3] = w_old * prev[f_old, 0:3] + w_new * new[f_new, 0:3]
            # quat slerp
            q0 = prev[f_old, 3:7].astype(np.float64)
            q1 = new[f_new, 3:7].astype(np.float64)
            out_m[f, 3:7] = slerp(q0, q1, w_new).astype(np.float32)
            # joints
            out_m[f, 7:36] = w_old * prev[f_old, 7:36] + w_new * new[f_new, 7:36]
            # joint velocities
            out_jv[f] = w_old * prev_jvel[f_old] + w_new * new_jvel[f_new]
        return out_m, out_jv

    def _replan(self, env_ids: np.ndarray) -> None:
        """Deploy-faithful replan: context from current motion look-ahead + splice."""
        for e in env_ids.tolist():
            prev_motion = self._motions[e]
            prev_jvel = self._jvels[e]

            if prev_motion is None:
                ctx_50hz = self._build_standing_context()
                gen_frame = 0
            else:
                gen_frame = int(self._current_frame[e] + self.motion_look_ahead)
                ctx_50hz = self._sample_context_from_motion(prev_motion, gen_frame)

            qpos_30hz, n_valid = self.planner.run(
                target_vel=float(self.target_vel[e]),
                mode=self.mode,
                context_qpos=ctx_50hz,
            )
            new_motion = resample_30hz_to_50hz(qpos_30hz[:n_valid], n_valid)
            new_jvel = compute_joint_velocities_50hz(new_motion)

            if prev_motion is None:
                self._motions[e] = new_motion
                self._jvels[e] = new_jvel
            else:
                spliced_m, spliced_jv = self._splice_motion(
                    prev_motion, prev_jvel, new_motion, new_jvel,
                    current_frame=int(self._current_frame[e]),
                    gen_frame=gen_frame,
                )
                self._motions[e] = spliced_m
                self._jvels[e] = spliced_jv

            # After splice, frame 0 of the new storage = where we are now
            self._current_frame[e] = 0
            self._steps_since_replan[e] = 0
            self._init_motion_quats[e] = self._motions[e][0, 3:7].astype(np.float64)
            self._init_heading_quats[e] = calc_heading_quat(self._init_motion_quats[e])

    def set_target_vel(self, env_ids: np.ndarray, vels: np.ndarray) -> None:
        """Update per-env target velocity (takes effect at next replan)."""
        self.target_vel[env_ids] = vels.astype(np.float32)

    def reset(self, env_ids: Optional[np.ndarray] = None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self._current_frame[env_ids] = 0
        self._steps_since_replan[env_ids] = 0
        # Clear prior motion so cold-start context is used
        for e in env_ids.tolist():
            self._motions[e] = None
            self._jvels[e] = None
        self._replan(env_ids)

    def step_frame(self) -> None:
        """Advance current_frame and replan any env that's hit the replan interval."""
        self._current_frame += 1
        self._steps_since_replan += 1

        # Also wrap current_frame if approaching motion end
        for e in range(self.num_envs):
            if self._current_frame[e] >= len(self._motions[e]) - 10:
                # Force replan early to avoid running out of motion
                self._steps_since_replan[e] = self.replan_interval

        # Find envs that need replanning
        need_replan = np.where(self._steps_since_replan >= self.replan_interval)[0]
        if len(need_replan) > 0:
            self._replan(need_replan)

    def get_tokens(
        self,
        velocity_cmd: Tensor,      # (N,) — reporting only; planner uses self.target_vel
        robot_base_quat: Tensor,   # (N, 4) wxyz — used for encoder anchor
        robot_base_pos: Optional[Tensor] = None,         # accepted for bridge compat (unused here)
        robot_joint_pos_grouped: Optional[Tensor] = None,  # accepted for bridge compat (unused here)
    ) -> Tensor:
        """Build per-env encoder input and run batched encoder.

        Planner context comes from motion look-ahead (not robot state), matching deploy.
        Robot state args are accepted but unused — kept for bridge interface compat.
        """
        quat_np = robot_base_quat.detach().cpu().numpy().astype(np.float64)

        # Build 1762D input for each env
        N = self.num_envs
        enc_in = np.zeros((N, 1762), dtype=np.float32)
        for e in range(N):
            motion = self._motions[e]
            jvel = self._jvels[e]
            cf = int(min(self._current_frame[e], len(motion) - 1))
            enc_in[e] = build_encoder_input(
                motion_50hz=motion,
                motion_joint_vel_50hz=jvel,
                current_frame=cf,
                robot_base_quat=quat_np[e],
                init_motion_quat=self._init_motion_quats[e],
                init_heading_quat=self._init_heading_quats[e],
            )

        # Batched encoder
        tokens = self.encoder.run(None, {"obs_dict": enc_in})[0]
        return torch.from_numpy(tokens).to(self.device)
