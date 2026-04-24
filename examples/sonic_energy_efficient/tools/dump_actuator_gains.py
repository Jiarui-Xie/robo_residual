"""Dump the actual stiffness/damping/effort_limit IsaacLab applies at runtime."""

from __future__ import annotations

import sys


def main() -> None:
    sys.path.insert(0, "/home/lumi/robo_residual")
    import examples.sonic_energy_efficient._ort_cuda_setup  # noqa

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg

    cfg = G1_29DOF_Sonic_FlatEnvCfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=cfg)

    robot = env.scene["robot"]
    jnames = list(robot.data.joint_names)
    # IsaacLab stores effective PD in robot.data.joint_stiffness / joint_damping
    # after actuators are applied.
    stiff = robot.data.joint_stiffness[0].cpu().numpy()
    damp = robot.data.joint_damping[0].cpu().numpy()
    effort = robot.data.joint_effort_limits[0].cpu().numpy()

    print(f"{'idx':>3} {'joint':<32} {'Kp_runtime':>10} {'Kd_runtime':>10} {'effort':>8}")
    for i, jn in enumerate(jnames):
        print(f"{i:>3} {jn:<32} {stiff[i]:>10.3f} {damp[i]:>10.3f} {effort[i]:>8.2f}")

    # Also inspect the ArticulationCfg side to see what was REQUESTED
    print("\n-- G1_29DOF_Sonic_FlatEnvCfg.scene.robot.actuators (requested) --")
    for grp_name, act in cfg.scene.robot.actuators.items():
        print(f"[{grp_name}] joint_names_expr={act.joint_names_expr}  type={type(act).__name__}")
        print(f"  stiffness={act.stiffness}")
        print(f"  damping  ={act.damping}")
        if hasattr(act, "armature"):
            print(f"  armature ={act.armature}")
        if hasattr(act, "effort_limit"):
            print(f"  effort   ={act.effort_limit}")
        if hasattr(act, "velocity_limit"):
            print(f"  vel_lim  ={act.velocity_limit}")
        if hasattr(act, "saturation_effort"):
            print(f"  sat_eff  ={act.saturation_effort}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
