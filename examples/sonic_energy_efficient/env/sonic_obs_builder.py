"""Build the 994-D observation vector for SONIC's decoder.

Maintains a 10-frame ring buffer of proprioceptive history per environment
and concatenates it with the encoder token to form the decoder input.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
from ..configs.sonic_params import NATIVE_DEFAULT_ANGLES as DEFAULT_DOF_ANGLES


class SonicObsBuilder:
    """Constructs the 994-D decoder observation from raw sensor data.

    Layout: [token(64), ang_vel_hist(30), jpos_hist(290),
             jvel_hist(290), action_hist(290), gravity_hist(30)]

    History is stored oldest-first: [frame_{t-9}, ..., frame_{t-1}, frame_t].
    """

    def __init__(self, num_envs: int, device: str = "cuda") -> None:
        self.num_envs = num_envs
        self.device = device
        self.num_frames = OBS_LAYOUT.history_frames

        # Ring buffers: (num_envs, num_frames, dim_per_frame)
        self._ang_vel_buf = torch.zeros(num_envs, self.num_frames, 3, device=device)
        self._jpos_buf = torch.zeros(num_envs, self.num_frames, NUM_JOINTS, device=device)
        self._jvel_buf = torch.zeros(num_envs, self.num_frames, NUM_JOINTS, device=device)
        self._action_buf = torch.zeros(num_envs, self.num_frames, NUM_JOINTS, device=device)
        self._gravity_buf = torch.zeros(num_envs, self.num_frames, 3, device=device)

        # Default standing pose for initialization
        self._default_jpos = torch.tensor(
            DEFAULT_DOF_ANGLES, dtype=torch.float32, device=device
        )

        # Initialize gravity to [0, 0, -1] (standing upright)
        self._default_gravity = torch.tensor([0.0, 0.0, -1.0], device=device)

        self._frame_idx = 0

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset history buffers for specified environments (or all)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._ang_vel_buf[env_ids] = 0.0
        # joint_pos in decoder obs = joint_q - default_angles (delta), so
        # standing pose has delta = 0
        self._jpos_buf[env_ids] = 0.0
        self._jvel_buf[env_ids] = 0.0
        self._action_buf[env_ids] = 0.0
        self._gravity_buf[env_ids] = self._default_gravity.unsqueeze(0)

    def update(
        self,
        ang_vel: Tensor,     # (N, 3)
        joint_pos: Tensor,   # (N, 29)
        joint_vel: Tensor,   # (N, 29)
        actions: Tensor,     # (N, 29)
        gravity: Tensor,     # (N, 3)
    ) -> None:
        """Push a new frame into the history buffer (FIFO shift)."""
        # Shift left: drop oldest, append newest
        self._ang_vel_buf = torch.cat(
            [self._ang_vel_buf[:, 1:, :], ang_vel.unsqueeze(1)], dim=1
        )
        self._jpos_buf = torch.cat(
            [self._jpos_buf[:, 1:, :], joint_pos.unsqueeze(1)], dim=1
        )
        self._jvel_buf = torch.cat(
            [self._jvel_buf[:, 1:, :], joint_vel.unsqueeze(1)], dim=1
        )
        self._action_buf = torch.cat(
            [self._action_buf[:, 1:, :], actions.unsqueeze(1)], dim=1
        )
        self._gravity_buf = torch.cat(
            [self._gravity_buf[:, 1:, :], gravity.unsqueeze(1)], dim=1
        )

    def build(self, token: Tensor) -> Tensor:
        """Build the 436-D decoder input.

        Args:
            token: (N, 64) encoder token.

        Returns:
            (N, 436) observation tensor.
        """
        return torch.cat(
            [
                token,                                                    # 64
                self._ang_vel_buf.reshape(self.num_envs, -1),            # 12
                self._jpos_buf.reshape(self.num_envs, -1),               # 116
                self._jvel_buf.reshape(self.num_envs, -1),               # 116
                self._action_buf.reshape(self.num_envs, -1),             # 116
                self._gravity_buf.reshape(self.num_envs, -1),            # 12
            ],
            dim=-1,
        )
