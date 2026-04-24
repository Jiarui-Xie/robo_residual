"""Reference dataset provider for RSI (Reference State Initialization) training.

Loads an offline dataset produced by tools/generate_reference_dataset.py and
exposes:

  1. Token lookup per env (get_tokens) — acts as a drop-in token provider so
     IsaacLabSonicBridge can get 64-D SONIC tokens without running the encoder.
  2. RSI state lookup (get_init_state, get_init_history) — used by the bridge
     to write robot state into IsaacLab and warm-start the obs history buffers
     when an episode resets.

Per-env state:
  start_frame[i]   : random frame index chosen at reset time
  step_offset[i]   : frames elapsed since reset (advances by 1 each step())

Lookup index at any step: (start_frame + step_offset) % total_frames.

The dataset contains multiple cmd buckets in sequence; start_frame is drawn
uniformly over the entire dataset, so wrap events (crossing a bucket boundary
mid-episode) are rare and statistically flat across training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ReferenceDatasetProvider:
    """Per-env pointer into an offline reference-motion dataset."""

    def __init__(
        self,
        dataset_path: str | list[str],
        num_envs: int,
        device: str = "cuda",
        history_frames: int = 10,
        episode_frames: int = 500,
        min_height_threshold: float = 0.65,
        sample_start_min: int | None = None,
        sample_start_max: int | None = None,
    ) -> None:
        """
        dataset_path: single path or list of paths — multiple files are concatenated.
        episode_frames: length of one training episode in frames (default 500 = 10s @ 50Hz).
          Ensures sampled start frames never cross a cmd-velocity bucket boundary.
        min_height_threshold: exclude start frames whose episode window contains any frame
          with base_h below this value (filters crouch/transition artifacts).
        """
        self.num_envs = num_envs
        self.device = device
        self.history_frames = history_frames
        self.episode_frames = episode_frames

        paths = [dataset_path] if isinstance(dataset_path, str) else dataset_path
        parts = [np.load(p) for p in paths]

        def _concat(name, dtype=torch.float32):
            return torch.tensor(
                np.concatenate([p[name] for p in parts], axis=0), dtype=dtype, device=device
            )

        data = parts[0]  # for scalar metadata

        self.jpos_abs     = _concat("jpos_abs")       # (T, 29)
        self.jpos_delta   = _concat("jpos_delta")     # (T, 29)
        self.jvel         = _concat("jvel")           # (T, 29)
        self.ang_vel_b    = _concat("ang_vel_b")      # (T, 3)
        self.gravity_b    = _concat("gravity_b")      # (T, 3)
        self.actions      = _concat("actions")        # (T, 29)
        self.root_state_w = _concat("root_state_w")   # (T, 13)  pos3+quat4+linvelW3+angvelW3
        self.tokens       = _concat("tokens")         # (T, 64)
        self.cmd_vel      = _concat("cmd_vel")        # (T, 3)

        self.total_frames = self.tokens.shape[0]

        # Build valid start frame pool:
        # 1. Entire episode [start, start+episode_frames) stays in the same cmd bucket.
        # 2. No frame in the episode has base_h below min_height_threshold (avoids
        #    crouch/transition artifacts from bucket transitions in the reference data).
        cmd_vx = self.cmd_vel[:, 0].cpu().numpy()
        heights = data["root_state_w"][:, 2]
        coarse_min = max(history_frames, sample_start_min if sample_start_min is not None else history_frames)
        coarse_max = min(self.total_frames - episode_frames, sample_start_max if sample_start_max is not None else self.total_frames)

        # Prefix sum for fast range-min query on bad_frame mask.
        bad_frame = (heights < min_height_threshold).astype(np.int32)
        prefix = np.zeros(self.total_frames + 1, dtype=np.int32)
        prefix[1:] = np.cumsum(bad_frame)

        valid = []
        for s in range(coarse_min, coarse_max):
            e = s + episode_frames
            if cmd_vx[s] != cmd_vx[e - 1]:
                continue
            if prefix[e] - prefix[s] > 0:
                continue
            valid.append(s)
        if not valid:
            raise ValueError(
                f"No valid start frames found (episode_frames={episode_frames}, "
                f"min_height={min_height_threshold}, window=[{coarse_min},{coarse_max})). "
                f"Try reducing episode_frames or lowering min_height_threshold."
            )
        self._valid_starts = torch.tensor(valid, dtype=torch.long, device=device)
        print(f"[ReferenceDatasetProvider] {len(valid)} valid start frames "
              f"across {len(set(cmd_vx[valid].tolist()))} cmd buckets "
              f"(episode={episode_frames} frames={episode_frames*0.02:.0f}s, "
              f"min_h={min_height_threshold})")

        # Per-env pointers
        self.start_frame = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_offset = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Compat attribute used by IsaacLabSonicBridge's velocity-cache branch
        # (we don't use that branch, but exposing this avoids issues).
        self.num_cached_frames = self.total_frames

    # ---- RSI sampling ----------------------------------------------------- #

    def sample_start_frames(self, n: int) -> Tensor:
        """Sample n start frames uniformly from the bucket-safe valid pool."""
        idx = torch.randint(len(self._valid_starts), (n,), device=self.device)
        return self._valid_starts[idx]

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Assign new random start_frames and zero step_offsets for env_ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        new_starts = self.sample_start_frames(env_ids.numel())
        self.start_frame[env_ids] = new_starts
        self.step_offset[env_ids] = 0

    def step_frame(self) -> None:
        """Advance per-env step_offset by 1."""
        self.step_offset += 1

    # ---- Lookups ---------------------------------------------------------- #

    def _cur_index(self, env_ids: Tensor | None = None) -> Tensor:
        if env_ids is None:
            idx = (self.start_frame + self.step_offset) % self.total_frames
        else:
            idx = (self.start_frame[env_ids] + self.step_offset[env_ids]) % self.total_frames
        return idx

    def get_tokens(self) -> Tensor:
        """(N, 64) tokens at current per-env pointer."""
        return self.tokens[self._cur_index()]

    def get_current_cmd(self) -> Tensor:
        """(N, 3) cmd_vel at current per-env pointer (for reward tracking)."""
        return self.cmd_vel[self._cur_index()]

    def get_init_state(self, env_ids: Tensor) -> dict:
        """Robot state at each env's start_frame, for IsaacLab writeback."""
        starts = self.start_frame[env_ids]
        return {
            "jpos_abs":     self.jpos_abs[starts],      # (k, 29)
            "jvel":         self.jvel[starts],          # (k, 29)
            "root_state_w": self.root_state_w[starts],  # (k, 13)
        }

    def get_init_history(self, env_ids: Tensor) -> dict:
        """Past `history_frames` frames ending at start_frame-1 for each env.

        Returns dict of (k, H, D) tensors suitable for warm-starting
        SonicObsBuilder's ring buffers.
        """
        starts = self.start_frame[env_ids]                                # (k,)
        H = self.history_frames
        # offsets [-H, -H+1, ..., -1] → index relative to start_frame
        rel = torch.arange(-H, 0, device=self.device)                      # (H,)
        idx = (starts.unsqueeze(1) + rel.unsqueeze(0)) % self.total_frames  # (k, H)
        idx_flat = idx.reshape(-1)

        def gather(tensor: Tensor) -> Tensor:
            flat = tensor[idx_flat]
            return flat.reshape(env_ids.numel(), H, tensor.shape[-1])

        return {
            "ang_vel_b":  gather(self.ang_vel_b),     # (k, H, 3)
            "jpos_delta": gather(self.jpos_delta),    # (k, H, 29)
            "jvel":       gather(self.jvel),          # (k, H, 29)
            "actions":    gather(self.actions),       # (k, H, 29)
            "gravity_b":  gather(self.gravity_b),     # (k, H, 3)
        }
