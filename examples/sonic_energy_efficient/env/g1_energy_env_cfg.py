"""IsaacLab environment config for G1 energy-efficient residual training.

This config defines the simulation scene, action/observation managers,
reward terms, and termination conditions.  It targets IsaacLab >= 2.3.0.

If IsaacLab is not installed, this module can still be imported — the
dataclasses are plain Python.  The actual env registration happens only
when IsaacLab is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rewards import (
    action_smoothness,
    angular_velocity_penalty,
    base_height_tracking,
    energy_consumption,
    foot_slip,
    velocity_tracking,
)
from ..configs.g1_joints import DEFAULT_DOF_ANGLES, MOTOR_KD, MOTOR_KP, NUM_JOINTS
from ..configs.reward_config import CMD_VEL_MAX, CMD_VEL_MIN, RewardConfig
from ..configs.train_config import TrainConfig


# --------------------------------------------------------------------------- #
#  Plain-Python config (no IsaacLab import required)
# --------------------------------------------------------------------------- #


@dataclass
class SimCfg:
    dt: float = 0.005             # 200 Hz physics
    decimation: int = 4           # policy at 50 Hz
    gravity: tuple[float, ...] = (0.0, 0.0, -9.81)


@dataclass
class RobotCfg:
    num_joints: int = NUM_JOINTS
    default_joint_pos: list[float] = field(default_factory=lambda: list(DEFAULT_DOF_ANGLES))
    kp: list[float] = field(default_factory=lambda: list(MOTOR_KP))
    kd: list[float] = field(default_factory=lambda: list(MOTOR_KD))
    action_scale: float = 0.25    # scale factor applied to policy output


@dataclass
class TerminationCfg:
    min_base_height: float = 0.3    # meters — below this = fallen
    max_body_tilt_deg: float = 60.0 # degrees — excessive lean


@dataclass
class CommandCfg:
    """Uniform velocity command sampling."""
    vel_x_range: tuple[float, float] = (CMD_VEL_MIN, CMD_VEL_MAX)
    vel_y_range: tuple[float, float] = (0.0, 0.0)
    yaw_rate_range: tuple[float, float] = (0.0, 0.0)
    resample_interval_s: float = 5.0  # resample command every N seconds


@dataclass
class G1EnergyEnvCfg:
    """Top-level config aggregating all sub-configs."""
    sim: SimCfg = field(default_factory=SimCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    command: CommandCfg = field(default_factory=CommandCfg)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
