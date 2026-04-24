"""ALIGN-8: override USD link mass / inertia / COM with MuJoCo XML values.

USD-derived inertia tensors (computed from per-link convex hulls) differ
significantly from SONIC's MuJoCo `<inertial>` tags:

- total mass: USD 33.34 kg vs MuJoCo 36.17 kg (-7.8%)
- torso_link: mass -18.6%
- ankle_roll_link: Ixx -86.7%  (foot rotational inertia way too low)
- hip_yaw_link: Ixx -25.6% (both legs)
- waist_yaw_link: Ixx -32.7%

Low foot/hip inertia changes swing-leg timing and lateral dynamics, which
plausibly caps the forward speed a MuJoCo-trained policy can achieve.

This module parses the MuJoCo XML once and exposes a startup event that
writes those values onto `root_physx_view` in place.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


MUJOCO_XML = Path(
    "/home/lumi/GR00T-WholeBodyControl/gear_sonic/data/robot_model/"
    "model_data/g1/g1_29dof_with_hand.xml"
)


def _parse_mujoco_inertials(xml_path: Path = MUJOCO_XML) -> dict[str, dict]:
    """Walk every `<body>` in the MuJoCo worldbody and read its `<inertial>`.

    Returns {body_name: {'mass': float,
                         'diaginertia': (Ixx, Iyy, Izz),
                         'com_pos': (x, y, z)}}
    We ignore the inertial `quat` (principal-axis rotation) — on G1 it's
    within a few × 1e-4 of identity for every link so diag is fine.
    """
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
                fi = inertial.get("fullinertia")
                diag = tuple(float(x) for x in fi.split()[:3]) if fi else (0.0, 0.0, 0.0)
            pos = inertial.get("pos", "0 0 0")
            com = tuple(float(x) for x in pos.split())
            out[name] = {"mass": mass, "diaginertia": diag, "com_pos": com}
        for child in body.findall("body"):
            walk(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            walk(body)
    return out


_CACHED: dict[str, dict] | None = None


def _mj_inertials() -> dict[str, dict]:
    global _CACHED
    if _CACHED is None:
        _CACHED = _parse_mujoco_inertials()
    return _CACHED


def override_link_inertial_from_mujoco(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Startup-time override: set each USD link's mass/inertia/COM to the
    MuJoCo XML values. Bodies not in the MuJoCo XML (e.g. hand fingers that
    the 29dof XML doesn't describe) keep their USD defaults.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    body_names = list(asset.data.body_names)
    mj = _mj_inertials()

    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    coms = asset.root_physx_view.get_coms().clone()

    applied = 0
    skipped: list[str] = []
    for i, bn in enumerate(body_names):
        if bn not in mj:
            skipped.append(bn)
            continue
        mass = mj[bn]["mass"]
        ix, iy, iz = mj[bn]["diaginertia"]
        cx, cy, cz = mj[bn]["com_pos"]

        masses[env_ids, i] = mass
        # inertia is (num_envs, num_bodies, 9) flattened 3x3, row-major:
        # [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
        inertia_row = torch.tensor(
            [ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz],
            dtype=inertias.dtype,
        )
        inertias[env_ids, i, :] = inertia_row
        # coms is (num_envs, num_bodies, 7) — [px, py, pz, qx, qy, qz, qw]
        # we only overwrite position; leave quat as USD default (identity-ish)
        coms[env_ids, i, 0] = cx
        coms[env_ids, i, 1] = cy
        coms[env_ids, i, 2] = cz
        applied += 1

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)
    asset.root_physx_view.set_coms(coms, env_ids)

    # also update cached default_* so downstream randomization sees the new baseline
    asset.data.default_mass[env_ids] = masses[env_ids].to(asset.data.default_mass.device)
    asset.data.default_inertia[env_ids] = inertias[env_ids].to(asset.data.default_inertia.device)

    print(
        f"[ALIGN-8] override_link_inertial_from_mujoco: "
        f"applied to {applied}/{len(body_names)} links "
        f"(kept USD defaults on: {skipped})"
    )
