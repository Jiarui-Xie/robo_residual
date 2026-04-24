"""Bridge between IsaacLab G1 env and SONIC's 994-D observation format.

Responsibilities:
1. Extract proprioceptive state from IsaacLab's simulation buffers.
2. Reorder joint indices from IsaacLab layout to SONIC NATIVE (ISAACLAB) layout.
3. Maintain 10-frame history, produce 994-D SONIC decoder input.
4. Pass decoder actions (NATIVE order) directly to IsaacLab (same order expected).
5. Compute extra reward terms (velocity tracking at 2-3 m/s, energy, stability).

Joint ordering:
  SONIC NATIVE (IsaacLab) order = what the decoder was trained with, what the
  env's JointPositionActionCfg expects.  The dataset from record_official_sim
  stores joint data in this order (body_q_grouped[MUJOCO_TO_ISAACLAB]).
  Do NOT use SONIC_GROUPED order for proprio or actions — that's MuJoCo ordering.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
from ..configs.reward_config import RewardConfig
from ..configs.sonic_params import (
    NATIVE_DEFAULT_ANGLES,
    NATIVE_LOCOMOTION_JOINT_NAMES,
)
from .isaaclab_env_cfg import SONIC_LOCOMOTION_JOINT_NAMES
from .reference_dataset_provider import ReferenceDatasetProvider
from .sonic_obs_builder import SonicObsBuilder
from .rewards import (
    action_smoothness,
    angular_velocity_penalty,
    base_height_tracking,
    energy_balance,
    energy_consumption,
    foot_slip,
    residual_magnitude,
    velocity_tracking,
)


class IsaacLabSonicBridge:
    """Translates between IsaacLab env step I/O and SONIC obs/action format.

    Assumes IsaacLab env has already been created and its robot's joint names
    are available. Builds a permutation tensor mapping SONIC indices to
    IsaacLab indices at construction.
    """

    def __init__(
        self,
        isaaclab_env,                 # ManagerBasedRLEnv
        token_provider,               # SonicEncoderWrapper | ZeroTokenProvider | VelocityBucketTokenCache
        reward_cfg: RewardConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self._env = isaaclab_env
        self._token_provider = token_provider
        self._reward_cfg = reward_cfg or RewardConfig()
        self.device = device

        # Detect token provider type (order matters — dataset also exposes
        # get_tokens + step_frame, so check it first)
        self._uses_reference_dataset = isinstance(token_provider, ReferenceDatasetProvider)
        self._uses_live_encoder = (
            (not self._uses_reference_dataset)
            and hasattr(token_provider, "get_tokens")
            and hasattr(token_provider, "step_frame")
        )
        self._uses_velocity_cache = (
            (not self._uses_reference_dataset)
            and (not self._uses_live_encoder)
            and hasattr(token_provider, "get_tokens_for")
        )

        robot = isaaclab_env.scene["robot"]
        isaac_joint_names = list(robot.data.joint_names)

        # Select joints in SONIC NATIVE (IsaacLab) order — same order decoder was trained with.
        native_idx = [isaac_joint_names.index(n) for n in NATIVE_LOCOMOTION_JOINT_NAMES]
        self._grouped_to_native = torch.tensor(native_idx, dtype=torch.long, device=device)

        # Delta = abs_pos - default; both in NATIVE order.
        self._default_angles = torch.tensor(
            NATIVE_DEFAULT_ANGLES, dtype=torch.float32, device=device
        )

        self._obs_builder = SonicObsBuilder(isaaclab_env.num_envs, device=device)
        self._prev_actions = torch.zeros(isaaclab_env.num_envs, NUM_JOINTS, device=device)
        self._prev_sonic_actions = torch.zeros(isaaclab_env.num_envs, NUM_JOINTS, device=device)
        # 1-step action delay: matches GR00T DDS async (action from t-1 applied at t)
        self._delayed_actions = torch.zeros(isaaclab_env.num_envs, NUM_JOINTS, device=device)

        # Per-joint energy weight: 2× for hip_pitch + ankle joints (indices in IsaacLab NATIVE order).
        # Verified mapping (SONIC_GROUPED → native via MUJOCO_TO_ISAACLAB):
        #   hip_pitch:   L=0,  R=1
        #   ankle_pitch: L=13, R=14
        #   ankle_roll:  L=17, R=18
        self._energy_weights = torch.ones(NUM_JOINTS, device=device)
        self._energy_weights[[0, 1, 13, 14, 17, 18]] = 2.0

        # Left/right leg joint indices (NATIVE order) for energy_balance reward
        #   L: hip_pitch=0,  hip_roll=3,  hip_yaw=6,  knee=9,  ankle_pitch=13, ankle_roll=17
        #   R: hip_pitch=1,  hip_roll=4,  hip_yaw=7,  knee=10, ankle_pitch=14, ankle_roll=18
        self._left_leg_indices = [0, 3, 6, 9, 13, 17]
        self._right_leg_indices = [1, 4, 7, 10, 14, 18]

        # Per-env motion frame counter for VelocityBucketTokenCache (wraps at num_cached_frames).
        self._current_frame = torch.zeros(isaaclab_env.num_envs, dtype=torch.long, device=device)
        self._num_cached_frames = getattr(token_provider, "num_cached_frames", None) if self._uses_velocity_cache else None

        self.num_envs = isaaclab_env.num_envs
        self.num_obs = OBS_LAYOUT.total_dim      # 994
        self.num_actions = NUM_JOINTS             # 29

    def _get_token(self) -> Tensor:
        """Get per-env tokens (N, 64)."""
        if self._uses_reference_dataset:
            return self._token_provider.get_tokens()
        if self._uses_live_encoder:
            cmd = self._get_command()
            vx = cmd[:, 0]
            robot = self._env.scene["robot"]
            base_quat = robot.data.root_quat_w  # (N, 4) wxyz
            # Online planner also needs current pos + joint state
            kwargs = {}
            import inspect
            sig = inspect.signature(self._token_provider.get_tokens)
            if "robot_base_pos" in sig.parameters:
                kwargs["robot_base_pos"] = robot.data.root_pos_w
            if "robot_joint_pos_grouped" in sig.parameters:
                # SONIC grouped proprio (absolute angles, NOT delta)
                jpos_all = robot.data.joint_pos
                # Need a separate permutation to SONIC grouped ordering
                if not hasattr(self, "_to_grouped"):
                    from ..configs.sonic_params import SONIC_GROUPED_JOINT_NAMES
                    isaac_names = list(robot.data.joint_names)
                    idx = [isaac_names.index(n) for n in SONIC_GROUPED_JOINT_NAMES]
                    self._to_grouped = torch.tensor(idx, dtype=torch.long, device=self.device)
                kwargs["robot_joint_pos_grouped"] = jpos_all[:, self._to_grouped]
            return self._token_provider.get_tokens(vx, base_quat, **kwargs)
        if self._uses_velocity_cache:
            cmd = self._get_command()
            vx = cmd[:, 0]
            return self._token_provider.get_tokens_for(vx, self._current_frame)
        return self._token_provider.get_token(self.num_envs)

    # --- sensor extraction ------------------------------------------------- #

    def _get_sonic_state(self):
        """Read current state from IsaacLab and reorder to SONIC joint layout."""
        robot = self._env.scene["robot"]
        # Ang vel (body frame)
        ang_vel = robot.data.root_ang_vel_b  # (N, 3)
        # Gravity direction (body frame)
        gravity = robot.data.projected_gravity_b  # (N, 3)
        # Joint positions/velocities — reorder to SONIC
        jpos_all = robot.data.joint_pos         # (N, 43)
        jvel_all = robot.data.joint_vel         # (N, 43)
        # NVIDIA native order; proprio is stored as DELTA from default_angles
        # (matching deploy: body_q[i] = joint_q[i] - default_angles[i])
        jpos_abs = jpos_all[:, self._grouped_to_native]  # (N, 29)
        jpos = jpos_abs - self._default_angles            # delta for decoder
        jvel = jvel_all[:, self._grouped_to_native]
        # Base linear velocity (for reward)
        lin_vel = robot.data.root_lin_vel_b  # (N, 3)
        # Base height
        base_h = robot.data.root_pos_w[:, 2]  # (N,)
        # Joint torques (applied)
        torques = robot.data.applied_torque[:, self._grouped_to_native]  # (N, 29) SONIC grouped order

        return {
            "ang_vel": ang_vel,
            "gravity": gravity,
            "jpos": jpos,
            "jvel": jvel,
            "lin_vel": lin_vel,
            "base_h": base_h,
            "torques": torques,
        }

    def _get_command(self) -> Tensor:
        """Get velocity command (N, 3) = (vx, vy, yaw_rate).

        When using the reference dataset, the cmd comes from the dataset frame
        (which drove the planner during offline recording), not from IsaacLab's
        command manager. This keeps token ↔ cmd consistent.
        """
        if self._uses_reference_dataset:
            return self._token_provider.get_current_cmd()
        return self._env.command_manager.get_command("base_velocity")

    # --- RSI (reference-state initialization) ----------------------------- #

    def _rsi_reset(self, env_ids: Tensor) -> None:
        """Reset env_ids from the reference dataset: sample start frames,
        write robot state into sim, warm-start obs history.
        """
        if env_ids.numel() == 0 or not self._uses_reference_dataset:
            return

        ds = self._token_provider
        ds.reset(env_ids)
        init = ds.get_init_state(env_ids)
        hist = ds.get_init_history(env_ids)

        robot = self._env.scene["robot"]

        # --- Joint state: dataset stores NATIVE locomotion-29 order;
        # write into IsaacLab at _grouped_to_native joint_ids ---
        jpos_native = init["jpos_abs"]  # (k, 29)
        jvel_native = init["jvel"]      # (k, 29)
        # write_joint_state_to_sim wants (k, J) matching joint_ids length
        robot.write_joint_state_to_sim(
            position=jpos_native,
            velocity=jvel_native,
            joint_ids=self._grouped_to_native.tolist(),
            env_ids=env_ids,
        )

        # --- Root state: add per-env origin offset so envs don't overlap ---
        root_state = init["root_state_w"].clone()   # (k, 13)
        env_origins = self._env.scene.env_origins[env_ids]  # (k, 3)
        root_state[:, :3] += env_origins
        robot.write_root_state_to_sim(root_state=root_state, env_ids=env_ids)

        # --- Warm-start obs history buffers from dataset's past H frames ---
        # hist[*] shapes: (k, H, D). Assign into the bridge's ring buffers.
        self._obs_builder._ang_vel_buf[env_ids] = hist["ang_vel_b"]
        self._obs_builder._jpos_buf[env_ids]    = hist["jpos_delta"]
        self._obs_builder._jvel_buf[env_ids]    = hist["jvel"]
        self._obs_builder._action_buf[env_ids]  = hist["actions"]
        self._obs_builder._gravity_buf[env_ids] = hist["gravity_b"]

        # prev_actions = last action from the dataset history (frame start-1)
        self._prev_sonic_actions[env_ids] = hist["actions"][:, -1, :]
        # delayed_actions: warm-start from last history action (v7-style; reverted
        # the "fix" to use actions[start] — empirically v9 with that fix collapsed).
        # The one-frame lag here is <20ms and doesn't hurt training stability.
        self._delayed_actions[env_ids] = hist["actions"][:, -1, :]

    # --- public API -------------------------------------------------------- #

    def reset(self):
        """Reset IsaacLab env, clear history, return 994-D obs."""
        obs_dict, _ = self._env.reset()
        self._obs_builder.reset()
        self._prev_actions.zero_()
        self._prev_sonic_actions.zero_()
        self._delayed_actions.zero_()
        self._current_frame.zero_()
        if self._uses_live_encoder:
            self._token_provider.reset()

        if self._uses_reference_dataset:
            # RSI path: draw start frames for every env, write state into sim,
            # warm-start history buffers from dataset. No priming update —
            # history is already filled with 10 frames of real proprio.
            all_ids = torch.arange(self.num_envs, device=self.device)
            self._rsi_reset(all_ids)
        else:
            # Fill all history_frames with the initial state so the decoder sees
            # a consistent "robot standing still" signal across the full window.
            # One update would leave 9 zero frames → OOD obs → violent initial actions.
            state = self._get_sonic_state()
            for _ in range(self._obs_builder.num_frames):
                self._obs_builder.update(
                    ang_vel=state["ang_vel"],
                    joint_pos=state["jpos"],
                    joint_vel=state["jvel"],
                    actions=self._prev_sonic_actions,
                    gravity=state["gravity"],
                )

        token = self._get_token()
        return self._obs_builder.build(token)

    def step(self, sonic_actions: Tensor, residual_delta: Tensor | None = None):
        """Step with NATIVE-order actions (N, 29); return (obs, reward, done, info).

        Args:
            sonic_actions: (N, 29) final actions (base + residual) in NATIVE order
            residual_delta: (N, 29) optional raw residual delta (clamped) for residual_magnitude reward
        """
        # 1-step action delay: apply action from previous step (matches GR00T DDS async)
        apply_actions = self._delayed_actions.clone()
        self._delayed_actions = sonic_actions.detach().clone()
        _, isaac_reward, terminated, truncated, info = self._env.step(apply_actions)
        done = terminated | truncated

        # Read new state
        state = self._get_sonic_state()
        self._obs_builder.update(
            ang_vel=state["ang_vel"],
            joint_pos=state["jpos"],
            joint_vel=state["jvel"],
            actions=sonic_actions,
            gravity=state["gravity"],
        )

        # Advance live encoder's frame counter (after state update, before token lookup)
        if self._uses_live_encoder:
            self._token_provider.step_frame()

        # Advance reference-dataset pointer (start_frame is fixed per episode;
        # only step_offset advances — wraps modulo total_frames).
        if self._uses_reference_dataset:
            self._token_provider.step_frame()

        # Advance velocity-cache frame counter; wrap at num_cached_frames; reset on done.
        if self._uses_velocity_cache:
            self._current_frame = self._current_frame + 1
            if self._num_cached_frames is not None:
                self._current_frame = self._current_frame % self._num_cached_frames
            self._current_frame = torch.where(done, torch.zeros_like(self._current_frame), self._current_frame)

        token = self._get_token()
        obs = self._obs_builder.build(token)

        # --- compute SONIC-specific reward components ---
        cmd = self._get_command()
        rcfg = self._reward_cfg
        components = {
            "velocity_tracking": velocity_tracking(
                state["lin_vel"], cmd, rcfg.velocity_tracking_sigma
            ),
            "energy_penalty": energy_consumption(state["torques"], state["jvel"], self._energy_weights),
            "energy_balance_penalty": energy_balance(
                state["torques"], state["jvel"],
                self._left_leg_indices, self._right_leg_indices
            ),
            "ang_vel_penalty": angular_velocity_penalty(state["ang_vel"]),
            "action_smoothness": action_smoothness(sonic_actions, self._prev_sonic_actions),
            "base_height_penalty": base_height_tracking(state["base_h"], rcfg.target_base_height),
            "alive_bonus": torch.full((self.num_envs,), 1.0, device=self.device),
        }
        if residual_delta is not None:
            components["residual_magnitude"] = residual_magnitude(residual_delta)
        weights = {
            "velocity_tracking": rcfg.velocity_tracking,
            "energy_penalty": rcfg.energy_penalty,
            "energy_balance_penalty": rcfg.energy_balance_penalty,
            "ang_vel_penalty": rcfg.ang_vel_penalty,
            "action_smoothness": rcfg.action_smoothness,
            "base_height_penalty": rcfg.base_height_penalty,
            "alive_bonus": rcfg.alive_bonus,
        }
        if residual_delta is not None:
            weights["residual_magnitude"] = rcfg.residual_magnitude

        # Apply terminations (already handled by IsaacLab done)
        fallen = state["base_h"] < rcfg.target_base_height * 0.5
        done = done | fallen
        components["alive_bonus"] = (~done).float()

        sonic_reward = sum(w * components[n] for n, w in weights.items())

        # Update prev actions first (dataset RSI may override for done envs).
        self._prev_sonic_actions = sonic_actions.detach().clone()

        # Reset obs history on done
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            if self._uses_reference_dataset:
                # RSI re-init for just these envs (overwrites their history
                # and prev_sonic_actions from the dataset).
                self._rsi_reset(done_ids)
                # Rebuild obs/token for these envs so the next rollout step
                # consumes the freshly-warm-started state.
                token = self._get_token()
                obs = self._obs_builder.build(token)
            else:
                self._obs_builder.reset(done_ids)
                self._delayed_actions[done_ids] = 0.0
                if self._uses_live_encoder:
                    self._token_provider.reset(done_ids.cpu().numpy())

        info["reward_components"] = components
        info["isaac_reward"] = isaac_reward
        info["commands"] = cmd
        info["lin_vel"] = state["lin_vel"]

        return obs, sonic_reward, done, info
