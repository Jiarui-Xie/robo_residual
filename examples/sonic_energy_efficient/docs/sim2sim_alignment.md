# SONIC → IsaacLab Sim2Sim Physics Alignment

SONIC was trained under a specific MuJoCo physics configuration. Running the
same ONNX decoder in IsaacLab with IsaacLab's defaults produces a degraded
gait that saturates at ~2 m/s even when commanded 3 m/s. The residual policy
sits on top of that base, so until the base runs at its native capacity here
we cannot meaningfully measure or train residual improvements.

This doc records every physics knob we pin to SONIC's training values so the
base policy can reach its native behavior inside IsaacLab.

## Source of truth

All values come directly from SONIC's MuJoCo assets shipped with the trained
policy:

- **Joint physics** — `GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`
  ```xml
  <default>
    <default class="torso_motor">  <joint damping="0.05" armature="0.01" frictionloss="0.2"/></default>
    <default class="leg_motor">    <joint damping="0.05" armature="0.01" frictionloss="0.2"/></default>
    <default class="ankle_motor">  <joint damping="0.05" armature="0.01" frictionloss="0.2"/></default>
    <default class="arm_motor">    <joint damping="0.05" armature="0.01" frictionloss="0.2"/></default>
    <default class="wrist_motor">  <joint damping="0.05" armature="0.01" frictionloss="0.1"/></default>
    <default class="finger_motor"> <joint damping="0.05" armature="0.01" frictionloss="0.1"/></default>
  </default>
  ```

- **Actuator model** — `GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml`
  ```xml
  <actuator>
    <motor name="..." joint="..."/>   <!-- pure torque source, no speed-dependent rolloff -->
    ...
  </actuator>
  ```
  PD is computed in C++ outside MuJoCo (`gear_sonic_deploy/.../policy_parameters.hpp`)
  and fed in as motor torque.

- **Ground friction** — `GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_29dof.xml`
  ```xml
  <default><geom friction="0.5"/></default>
  ```

## Changes applied

All edits happen in `examples/sonic_energy_efficient/env/isaaclab_env_cfg.py`,
inside `G1_29DOF_Sonic_FlatEnvCfg._patch_actuator_gains()`. Each override is
tagged `ALIGN-N` in-code so it can be traced back here.

| Tag | Field                     | IsaacLab default                | SONIC training        | Rationale |
|-----|---------------------------|---------------------------------|-----------------------|-----------|
| 1   | actuator `stiffness` (Kp) | 100 / 200 / 20 (legs / knee / ankle) | `NATIVE_KP` (99.1 / 99.1 / 28.5) | Match external-PD gains from `policy_parameters.hpp` |
| 1   | actuator `damping` (Kd)   | 2.5 / 5 / 0.2                   | `NATIVE_KD` (6.3 / 6.3 / 1.8)    | Match external-PD damping |
| 2   | actuator `armature`       | 0.03                            | **0.01** (uniform)    | MuJoCo `<joint armature>` is the reflected rotor inertia added to the mass matrix. IsaacLab's 0.03 is 3× too high → gait feels over-damped and can't accelerate. |
| 3   | actuator `viscous_friction` | 0 (USD default)               | **0.05** (uniform)    | MuJoCo `<joint damping>` is a small per-joint viscous term, independent of PD. Without it the leg joints over-swing. |
| 4   | actuator `friction`         | 0 (USD default)               | **0.2** (uniform)     | MuJoCo `<joint frictionloss>` is static Coulomb friction. Without it joints respond to arbitrarily small torque → erratic tiny motions baked into SONIC's expectation aren't recreated. |
| 5   | DCMotor `saturation_effort` | 180 (legs) / 80 (feet)        | effectively `+∞`      | SONIC's MuJoCo uses `<motor>` — a pure torque source. IsaacLab's DCMotorCfg applies `τ_max = saturation_effort · (1 − |ω|/velocity_limit)`. At knee ω=20 rad/s that pushes torque toward zero. We raise the saturation and the velocity ceiling so the rolloff never kicks in during normal operation. |
| 5   | DCMotor `velocity_limit`    | 20 / 32 / 37 rad/s            | effectively `+∞`      | Same reason — prevent the DCMotor model from clipping torque based on joint velocity. Concrete values used: `saturation_effort=10000`, `velocity_limit=1000`. |
| 6   | Ground static/dynamic μ     | 0.5 / 0.5                     | **1.0** (training scene) | IsaacLab default matched the *deploy* scene XML (μ=0.5). SONIC was trained under `scene_43dof.xml` which uses `<geom friction="1.0">`. Lower friction in IsaacLab lets the foot slip at fast cadences → loss of forward propulsion. |
| 7   | `sim.dt`                    | 0.005 (200 Hz)                | **0.002** (MuJoCo default) | MuJoCo's `<option>` was absent in the training XML so it defaulted to 0.002. Stiff leg contacts at 200 Hz integrate poorly — toe slip/chatter differs from training. `decimation` is bumped 4 → 10 so the control rate stays 50 Hz. |
| 8   | per-link mass/inertia/COM    | USD convex-hull derived      | MuJoCo `<inertial>` tags  | Runtime startup event `override_link_inertial_from_mujoco` in `env/mujoco_inertial.py` writes every link's mass, diagonal inertia, and COM offset from the training XML onto `root_physx_view`. Notable gaps it closed: torso_link mass (−18.6 % → 0 %), ankle_roll_link Ixx (−86.7 % → 0 %), hip_yaw_link Ixx (−25.6 % → 0 %). **Null result on top speed** — see empirical note below — but keeps the physics faithful to the MuJoCo source of truth regardless. |
| 9   | PhysX `max_depenetration_velocity` | 1.0 m/s                | **1000.0** m/s            | PhysX caps how fast contact resolver can separate penetrating bodies; at running foot-strike (>1 m/s vertical) this attenuates ground reaction force. MuJoCo's Newton solver has no equivalent cap. **Null result on top speed** but is the right thing physically. |
| 10  | PhysX `bounce_threshold_velocity` | 0.5 m/s                 | **1e9** m/s (≡ disabled)  | Any contact with relative velocity above this becomes restitutive in PhysX; MuJoCo geoms use `restitution=0` → fully inelastic. Foot strikes at >0.5 m/s → PhysX was bleeding push-off energy into pseudo-bounce. Result: `cmd=2.5` peak jumps from ~2.1 to **2.69 m/s** (25 % higher). |
| 10  | PhysX `min_position_iteration_count` | 1                       | **8**                     | MuJoCo Newton runs up to 100 iterations per step; PhysX TGS defaults to actor-self-reported counts with floor 1. Bumping the floor brings per-dt constraint convergence closer to MJ. |
| 10  | PhysX `min_velocity_iteration_count` | 0                       | **2**                     | Same reasoning on velocity iterations. |

Note on wrist frictionloss: SONIC's training XML uses 0.1 (not 0.2) for
wrist/finger joints. We apply 0.2 uniformly because wrist contributions to
locomotion are negligible and keeping one value keeps the code simpler.

## What was tried and ruled out

- **Per-motor-type armature** (0.00361 for 5020, 0.0251 for 7520_22, etc.)
  came from `policy_parameters.hpp` and was initially conflated with the
  MuJoCo `<joint armature>` XML attribute. Those are reflected-motor
  inertias used to *derive* Kp/Kd — they are not the joint-level armature
  in the simulator. The MuJoCo XML uses a uniform 0.01. Using the motor
  inertias here led the gait OOD.

- **DCMotor rolloff alone** (without armature/viscous/friction alignment)
  made the gait unstable: momentary vx exceeded 2.34 m/s but yaw rate
  oscillated ±3.4 rad/s and vy drifted to ~−3 m/s. The robot "fell
  forward" because actuators became over-strong at low speed while the
  effective plant stayed stiff. We now change those parameters together.

## Empirical result after full alignment (2026-04-22, 40 s smoke)

Bucketed eval with pure SONIC base (no residual), 8 s hold per cmd.

| cmd (m/s) | ALIGN-1..7 `vx_mean` | + ALIGN-8 | + ALIGN-9 | + ALIGN-10 `vx_mean` | + ALIGN-10 `vx_peak` |
|----------:|---------------------:|----------:|----------:|---------------------:|---------------------:|
| 1.5       | 1.66                 | 1.67      | —         | 1.67                 | 2.14                 |
| 2.0       | 2.22                 | 2.28      | —         | **2.28**             | 2.50                 |
| 2.5       | 1.86                 | 1.81      | —         | 1.81                 | **2.69** ★           |
| 3.0       | 1.94                 | 1.97      | 1.97      | 1.97                 | 2.17                 |

ALIGN-8 (mass/inertia/COM) and ALIGN-9 (depen. velocity cap) were **null
results** on top speed despite closing large physical gaps. ALIGN-10 is a
**partial win**: `cmd=2.5` peak jumped from ~2.1 to **2.69 m/s** (a speed
never before reached) — confirming that the PhysX `bounce_threshold_velocity`
default was eating push-off impulse into pseudo-bounce. `cmd=3.0` mean remains
stuck at 1.97; still something else is capping sustained cadence.

### Diagnostic that pinned contact-solver as the culprit

`tools/mujoco_per_step_compare.py` resets MuJoCo's full state to IsaacLab's
pre-step state every frame in the cmd=3.0 window, then runs one decimation
with the same action through MuJoCo's plant. Findings:

- **Consistent Δvx = +0.058 m/s/step** (83.7 % sign-consistent) — MuJoCo
  converts the same action into more forward velocity than PhysX does.
- Per-joint `|Δq|` averages 0.02 rad but sign-consistency is only 50-55 %
  → not a DC offset; it's a **push-off-moment response difference**.
- Peak `|Δq|` (0.10–0.13 rad) occurs only on single swing→stance frames,
  on ankle_pitch and knee — exactly when foot meets ground.

Interpretation: at foot strike, MuJoCo's Newton solver fully resolves the
normal-force constraint within one dt; PhysX's default TGS setup loses
part of that impulse to per-actor-reported iteration counts and to the
`bounce_threshold_velocity=0.5 m/s` soft-bounce window.

### What remains (cmd=3.0 still capped at mean ≈ 1.97)

1. **Friction solver thresholds** (candidate ALIGN-11). `friction_offset_threshold=0.04` and `friction_correlation_distance=0.025` can delay the onset of tangential friction or over-cluster contact patches — both attenuate propulsion impulse.
2. **Terrain physics material restitution / compliance** may still be non-zero; verify against `scene.terrain.physics_material`.
3. **Foot collision geometry already matches**: the USD asset used here defines 4 spheres (`radius=5 mm`) on `left/right_ankle_roll_link/collisions`, positioned at the same four corners MuJoCo uses. This hypothesis is *not* live anymore.
4. Direct contact-force dump (normal + tangential) on a single push-off frame in both plants would be the definitive next diagnostic.

## Verification protocol

After each alignment change, run a bucketed-eval smoke:

```bash
OMNI_KIT_ACCEPT_EULA=yes python -u \
  examples/sonic_energy_efficient/tools/generate_reference_dataset.py \
  --decoder-onnx examples/sonic_energy_efficient/models/model_decoder.onnx \
  --planner-onnx examples/sonic_energy_efficient/models/planner_sonic_dyn.onnx \
  --encoder-onnx examples/sonic_energy_efficient/models/model_encoder_dyn.onnx \
  --output /tmp/align_check.npz \
  --duration 40 --cmd-buckets 1.5 2.0 2.5 3.0 --cmd-hold-seconds 8
```

Then inspect per-bucket `vx` means. Success criteria: `cmd=3.0 → vx ≥ 2.7` with
lateral drift `|vy| < 0.5` and yaw rate `|wz| < 1.0` sustained.
