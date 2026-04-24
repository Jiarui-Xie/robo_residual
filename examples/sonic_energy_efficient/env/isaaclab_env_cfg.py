"""IsaacLab environment config for G1 29-DOF residual training at 2-3 m/s.

Uses g1.usd (42 DOF including inspire hand) but only actuates 29 locomotion
joints — hand joints stay at default. SONIC was trained for 29-DOF rubber-hand
G1 but the physics and dynamics are identical for locomotion purposes.

Must be imported AFTER SimulationApp is instantiated.
"""

from __future__ import annotations

# Avoid circular import — isaaclab.envs must be imported before isaaclab.managers
import isaaclab.envs  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .mujoco_inertial import override_link_inertial_from_mujoco

from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from ..configs.sonic_params import (
    MUJOCO_DECIMATION, MUJOCO_GROUND_FRICTION, MUJOCO_JOINT_ARMATURE,
    MUJOCO_JOINT_DAMPING, MUJOCO_JOINT_FRICTIONLOSS, MUJOCO_TIMESTEP,
    NATIVE_ACTION_SCALE, NATIVE_DEFAULT_ANGLES, NATIVE_KD, NATIVE_KP,
    NATIVE_LOCOMOTION_JOINT_NAMES,
)


# 29 locomotion joint names in NVIDIA IsaacLab native G1 order.
# The decoder ONNX was trained with both proprio inputs and action outputs in
# this order. Evidence from deploy code: body_q_measured[mujoco_idx] =
# body_q[isaaclab_to_mujoco[mujoco_idx]] + default_angles[mujoco_idx] — which
# proves body_q (the proprio fed to decoder) is indexed by NVIDIA native idx.
SONIC_LOCOMOTION_JOINT_NAMES = NATIVE_LOCOMOTION_JOINT_NAMES


@configclass
class G1_29DOF_Sonic_FlatEnvCfg(G1RoughEnvCfg):
    """G1 29-DOF flat-terrain env with 2-3 m/s forward velocity command."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # ----- Robot asset: LOCAL g1_29dof USD (rubber hand, no inspire hand joints) -----
        robot_cfg = G1_29DOF_CFG.copy()
        robot_cfg.spawn.usd_path = (
            "/home/lumi/robo_residual/examples/sonic_energy_efficient/"
            "models/g1_rubber_hand/g1_29dof.usd"
        )
        robot_cfg.spawn.activate_contact_sensors = True

        # Remove 'hands' actuator group (G1_29DOF_CFG targets inspire hand joints
        # that don't exist in our 29-DOF rubber-hand URDF)
        robot_cfg.actuators = {
            k: v for k, v in robot_cfg.actuators.items() if k != "hands"
        }

        # ALIGN-9: PhysX max_depenetration_velocity 1.0 -> 1000.0
        # At running speed (foot impact >1 m/s vertical), PhysX caps the
        # depenetration velocity, weakening ground reaction force. MuJoCo's
        # Newton solver has no equivalent cap — contacts are fully resolved
        # within a timestep. Raise the cap so fast foot-strike push-off is
        # not artificially attenuated.
        robot_cfg.spawn.rigid_props.max_depenetration_velocity = 1000.0

        # Override init state to SONIC's default pose (deeper squat, NVIDIA native order)
        sonic_init_joint_pos = {
            NATIVE_LOCOMOTION_JOINT_NAMES[i]: float(NATIVE_DEFAULT_ANGLES[i])
            for i in range(29)
        }
        robot_cfg.init_state = robot_cfg.init_state.replace(joint_pos=sonic_init_joint_pos)

        # ---- Sim2Sim physics alignment with SONIC training MuJoCo XML ----
        # Source of truth: gear_sonic/data/robot_model/model_data/g1/
        #                  g1_29dof_with_hand.xml   (joint <default>)
        #                  gear_sonic_deploy/g1/scene_29dof.xml (ground friction)
        #
        # Each override below carries a CHANGELOG tag so it can be lifted
        # straight into the sim2sim alignment docs.
        #
        # ALIGN-1  stiffness/damping (Kp/Kd)      : NATIVE_KP, NATIVE_KD
        # ALIGN-2  armature                       : 0.03 -> 0.01
        # ALIGN-3  viscous_friction (joint damping): 0    -> 0.05
        # ALIGN-4  friction (static Coulomb)      : 0    -> 0.2
        # ALIGN-5  DCMotor torque-speed rolloff   : active -> disabled
        #           (SONIC trains with MuJoCo <motor> = ideal torque source,
        #            external C++ PD. DCMotor's (1 - |ω|/vel_limit) multiplier
        #            is an IsaacLab-only effect.)
        # ALIGN-6  ground μ : 0.5 (IsaacLab default matches SONIC, no override)
        _sonic_kp_by_name = {
            NATIVE_LOCOMOTION_JOINT_NAMES[i]: float(NATIVE_KP[i]) for i in range(29)
        }
        _sonic_kd_by_name = {
            NATIVE_LOCOMOTION_JOINT_NAMES[i]: float(NATIVE_KD[i]) for i in range(29)
        }

        def _patch_actuator_gains(actuator_cfg):
            """Apply ALIGN-1..5 to every actuator group on the robot."""
            import re
            from isaaclab.actuators import DCMotorCfg
            patched_stiff = {}
            patched_damp = {}
            patched_arm = {}
            patched_visc = {}
            patched_fric = {}
            for pattern in actuator_cfg.joint_names_expr:
                for jn, kp in _sonic_kp_by_name.items():
                    if re.fullmatch(pattern, jn):
                        patched_stiff[jn] = kp                              # ALIGN-1
                        patched_damp[jn] = _sonic_kd_by_name[jn]            # ALIGN-1
                        patched_arm[jn] = MUJOCO_JOINT_ARMATURE             # ALIGN-2
                        patched_visc[jn] = MUJOCO_JOINT_DAMPING             # ALIGN-3
                        patched_fric[jn] = MUJOCO_JOINT_FRICTIONLOSS        # ALIGN-4
            if not patched_stiff:
                return
            actuator_cfg.stiffness = patched_stiff
            actuator_cfg.damping = patched_damp
            actuator_cfg.armature = patched_arm
            actuator_cfg.viscous_friction = patched_visc
            actuator_cfg.friction = patched_fric
            if isinstance(actuator_cfg, DCMotorCfg):                        # ALIGN-5
                actuator_cfg.saturation_effort = 10000.0
                actuator_cfg.velocity_limit = 1000.0

        for name, act_cfg in robot_cfg.actuators.items():
            _patch_actuator_gains(act_cfg)

        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ----- Flat terrain -----
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ALIGN-6: ground static/dynamic μ -> 1.0 (SONIC training scene, not deploy)
        self.scene.terrain.physics_material.static_friction = MUJOCO_GROUND_FRICTION
        self.scene.terrain.physics_material.dynamic_friction = MUJOCO_GROUND_FRICTION

        # ALIGN-7: sim timestep 0.005 -> 0.002 (MuJoCo default),
        # decimation 4 -> 10 to preserve 50 Hz control rate.
        self.sim.dt = MUJOCO_TIMESTEP
        self.decimation = MUJOCO_DECIMATION
        self.sim.render_interval = self.decimation

        # ALIGN-10: PhysX contact solver tightening to match MuJoCo Newton solver.
        #   a) bounce_threshold_velocity 0.5 -> 1e9 : MuJoCo geom restitution=0
        #      (no bounce), but PhysX default lets any contact with |v_rel| > 0.5 m/s
        #      behave restitutively. At running foot-strike (>1 m/s vertical), this
        #      dissipates upward push-off as a pseudo-bounce. Set threshold above any
        #      expected contact velocity to force inelastic contact like MuJoCo.
        #   b) min_position_iteration_count 1 -> 8, min_velocity 0 -> 2 : MuJoCo's
        #      Newton solver runs up to 100 iterations per step with ls_iter=50 and
        #      resolves contact constraints fully within one dt. PhysX TGS defaults
        #      to actor-self-reported counts clamped only by min/max. Bumping the
        #      floor brings per-dt constraint convergence closer to MJ.
        self.sim.physx.bounce_threshold_velocity = 1e9
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 2

        # ----- Velocity command: 2-3 m/s forward only -----
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # ----- Actions: 29 locomotion joints in NVIDIA native order -----
        sonic_scale_by_name = {
            NATIVE_LOCOMOTION_JOINT_NAMES[i]: float(NATIVE_ACTION_SCALE[i]) for i in range(29)
        }
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=NATIVE_LOCOMOTION_JOINT_NAMES,
            preserve_order=True,
            scale=sonic_scale_by_name,
            use_default_offset=True,
        )

        # ----- Fix body references for 29-DOF model -----
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]

        # ALIGN-8: override per-link mass/inertia/COM with SONIC MuJoCo XML values.
        # USD-derived inertia from convex hulls differs sharply from the MuJoCo
        # <inertial> tags (foot Ixx -86%, torso mass -18%, hip_yaw Ixx -25%).
        # Fires before add_base_mass so any downstream randomization scales the
        # corrected baseline, not the USD one.
        self.events.align_link_inertial = EventTerm(
            func=override_link_inertial_from_mujoco,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ----- Rewards: replace G1Rewards with SONIC-focused ones -----
        # Drop problematic terms (reference elbow_pitch_joint which doesn't exist)
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        # Energy penalty (torque squared on locomotion joints)
        self.rewards.dof_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SONIC_LOCOMOTION_JOINT_NAMES)},
        )
        # Smooth actions
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        # Joint acceleration on legs (smooth gait)
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            )},
        )

        # ----- Episode length -----
        self.episode_length_s = 10.0

        # ----- Disable pushes (cleaner energy signal) -----
        self.events.push_robot = None


@configclass
class G1_29DOF_Sonic_FlatEnvCfg_PLAY(G1_29DOF_Sonic_FlatEnvCfg):
    """Small-scene variant for debugging."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
