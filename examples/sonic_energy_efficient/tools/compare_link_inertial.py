"""Compare per-link mass (and inertia diagonal) between SONIC MuJoCo XML and
the USD IsaacLab loads at runtime.

Why: post-ALIGN-1..7 we still cap at vx~1.94 @ cmd=3.0. SONIC decoder reaches
3 m/s in its own MuJoCo — so the remaining gap is physics, not policy.
Link mass/inertia is the cheapest source-of-truth mismatch to verify.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


MUJOCO_XML = Path(
    "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/"
    "model_data/g1/g1_29dof_with_hand.xml"
)


def parse_mujoco_inertials(xml_path: Path) -> dict[str, dict]:
    """Return {body_name: {'mass': float, 'diaginertia': (x,y,z)}} from MuJoCo XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: dict[str, dict] = {}

    def walk(body):
        name = body.get("name")
        inertial = body.find("inertial")
        if name and inertial is not None:
            mass = float(inertial.get("mass", "0"))
            di = inertial.get("diaginertia")
            if di is not None:
                diag = tuple(float(x) for x in di.split())
            else:
                # MuJoCo sometimes uses fullinertia instead
                fi = inertial.get("fullinertia")
                diag = tuple(float(x) for x in fi.split()[:3]) if fi else (0.0, 0.0, 0.0)
            out[name] = {"mass": mass, "diaginertia": diag}
        for child in body.findall("body"):
            walk(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return out
    for body in worldbody.findall("body"):
        walk(body)
    return out


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

    from isaaclab.envs import ManagerBasedRLEnv
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg

    cfg = G1_29DOF_Sonic_FlatEnvCfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=cfg)

    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    mass_usd = robot.data.default_mass[0].cpu().numpy()  # (num_bodies,)
    inertia_usd = robot.data.default_inertia[0].cpu().numpy()  # (num_bodies, 9)

    mj = parse_mujoco_inertials(MUJOCO_XML)

    print(f"MuJoCo XML bodies parsed: {len(mj)}")
    print(f"USD bodies in IsaacLab : {len(body_names)}")
    print()

    total_usd = float(mass_usd.sum())
    total_mj = sum(v["mass"] for v in mj.values())
    print(f"Total mass — USD: {total_usd:.4f} kg   MuJoCo: {total_mj:.4f} kg   Δ: {total_usd-total_mj:+.4f} kg ({100*(total_usd/total_mj-1):+.2f}%)")
    print()

    print(f"{'link':<32} {'m_usd':>8} {'m_muj':>8} {'Δm':>7} {'Δ%':>6} | {'Ixx_usd':>9} {'Ixx_muj':>9} {'Δ%':>6}")
    print("-" * 100)

    missing_in_mj = []
    big_diffs = []
    for i, bn in enumerate(body_names):
        # IsaacLab uses same naming convention as the MuJoCo body names
        if bn not in mj:
            missing_in_mj.append(bn)
            print(f"{bn:<32} {mass_usd[i]:>8.4f}   [missing in MuJoCo XML]")
            continue
        m_u = float(mass_usd[i])
        m_m = mj[bn]["mass"]
        dm = m_u - m_m
        dp = 100 * (m_u / m_m - 1) if m_m > 0 else 0.0

        # USD inertia is 3x3 matrix flattened (9), MuJoCo diaginertia is already principal
        Ixx_u, Iyy_u, Izz_u = inertia_usd[i, 0], inertia_usd[i, 4], inertia_usd[i, 8]
        Ixx_m, Iyy_m, Izz_m = mj[bn]["diaginertia"]
        dip = 100 * (Ixx_u / Ixx_m - 1) if Ixx_m > 0 else 0.0

        print(f"{bn:<32} {m_u:>8.4f} {m_m:>8.4f} {dm:>+7.3f} {dp:>+5.1f}% | {Ixx_u:>9.5f} {Ixx_m:>9.5f} {dip:>+5.1f}%")

        if abs(dp) > 5.0 or abs(dip) > 10.0:
            big_diffs.append((bn, dp, dip))

    print()
    if missing_in_mj:
        print(f"USD bodies NOT in MuJoCo XML ({len(missing_in_mj)}):")
        for bn in missing_in_mj:
            print(f"  - {bn}")
    if big_diffs:
        print(f"\nLinks with |Δmass|>5% or |ΔIxx|>10%:")
        for bn, dp, dip in big_diffs:
            print(f"  - {bn}:  Δmass={dp:+.1f}%   ΔIxx={dip:+.1f}%")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
