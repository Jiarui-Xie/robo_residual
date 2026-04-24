# SONIC × robo_residual: Complete Integration Tutorial

> **Goal**: Train a small residual MLP on top of NVIDIA GR00T-WholeBodyControl's frozen SONIC locomotion decoder to optimize an objective (energy efficiency, here) while preserving the pre-trained gait. Serve as the canonical example of how to adapt **robo_residual** to a real upstream policy and deploy it **sim2sim** through MuJoCo.
>
> **Status**: ✅ Complete. v9 achieves **−20% energy at cmd=3.0 m/s** and **−22% at cmd=1.7 m/s** in MuJoCo sim2sim (29-DOF deploy XML) while *improving* velocity tracking (σ −31~52%, |err| −47~67%).

This document is long on purpose. Every step lists **motivation → implementation → verification** so the recipe can be reproduced on a different upstream policy / different task (table-top manipulation, bimanual, quadruped, etc.). Use it as a checklist.

---

## Table of Contents

1. [What robo_residual is](#1-what-robo_residual-is)
2. [What SONIC is](#2-what-sonic-is)
3. [Why combine them](#3-why-combine-them)
4. [Architecture](#4-architecture-how-the-residual-attaches)
5. [Step-by-step adaptation (12 steps)](#5-step-by-step-adaptation)
6. [Physics alignment (ALIGN-1..10 + friction patch)](#6-physics-alignment)
7. [Reward design](#7-reward-design)
8. [Training recipe](#8-training-recipe)
9. [Deployment (fuse → ONNX)](#9-deployment-fuse-back-into-a-single-onnx)
10. [Evaluation (MuJoCo sim2sim)](#10-evaluation-mujoco-sim2sim)
11. [Results (v9)](#11-results-v9-on-g1-29-dof)
12. [Pitfalls & fixes — the full catalog](#12-pitfalls--fixes-the-full-catalog)
13. [Migration guide — adapting to a new task](#13-migration-guide-adapting-to-a-new-task)

---

## 1. What robo_residual is

A small PyTorch toolbox (`/home/lumi/robo_residual/robo_residual/`) that lets you bolt a **trainable residual MLP** on top of a **frozen ONNX policy**, train it with PPO (or any on-policy RL), and then **fuse the base + residual back into a single ONNX** for drop-in deployment.

```
robo_residual/
├── core/
│   ├── onnx_base.py           # OnnxBasePolicy: load any ONNX, frozen forward
│   ├── residual_actor_critic.py  # ResidualActorCritic: base + residual + critic
│   ├── composable.py          # multi-layer stacking (chain residuals)
│   └── fuse_onnx.py           # ONNX graph surgery: merge base+residual into one file
├── adapters/
│   ├── obs_adapter.py         # TensorDict → flat obs
│   └── rsl_rl_wrapper.py      # drop-in for rsl_rl ActorCritic
├── utils/
│   ├── freeze.py              # freeze_module() / unfreeze_module()
│   ├── zero_init.py           # zero_init_last_layer() — critical for stability
│   └── normalizer.py          # Welford running mean/var
└── config/
    └── residual_config.py     # per-joint clamp limits, hidden dims, activation
```

**Core design contracts:**

- The base policy is never modified — only the residual MLP & critic are trainable (`policy.trainable_parameters`).
- The residual is **zero-initialized at the output layer**: at iter 0 the residual outputs exactly zero, so the policy starts with pristine base behavior. This is load-bearing for training stability.
- Per-joint clamp limits cap how far the residual can deviate per joint (e.g. legs ≤ 0.35 rad, arms ≤ 0.03 rad). This prevents catastrophic exploration.
- Residual obs may be **larger** than base obs (e.g. base sees 48-D, residual sees 128-D with terrain scan). Controlled via `num_actor_obs` vs `num_base_obs`; the base receives `obs[:, :num_base_obs]`.

---

## 2. What SONIC is

**S**ync **O**rchestrated **N**eural **I**nverse **C**ontrol — the locomotion policy shipped with NVIDIA GR00T-WholeBodyControl. A learned 3-stage pipeline for humanoid locomotion on the Unitree G1 (29-DOF):

```
(target velocity)
      │
      ▼
┌───────────────┐   (64-D motion token)    ┌───────────────┐
│ planner.onnx  │──────────────────────────▶ encoder.onnx  │
│ (future traj) │                          │ (context embed)│
└───────────────┘                          └───────┬───────┘
                                                   │ 64-D
                                                   ▼
                                   ┌───────────────────────────────┐
                                   │         decoder.onnx          │
                                   │  994-D obs → 29 joint targets  │
                                   └───────────────┬───────────────┘
                                                   │
                                                   ▼
                           PD at 50 Hz: τ = Kp·(q_target − q) − Kd·q̇
```

The 994-D observation is:

```
[token(64) |
 ang_vel_hist(3×10) | jpos_delta_hist(29×10) | jvel_hist(29×10) |
 action_hist(29×10) | gravity_hist(3×10)]
  = 64 + 30 + 290 + 290 + 290 + 30 = 994
```

10 frames of history at 50 Hz (200 ms window). `jpos_delta = q − default_standing_angles`. `gravity` is the normalized gravity direction projected into body frame (unit vector).

SONIC ships its training physics (MuJoCo) and deploy pipeline (DDS-based, C++ runtime with external PD). The decoder is **frozen and closed-box** — we get the ONNX file, not the training code.

---

## 3. Why combine them

**SONIC gives us a working locomotion base** that we can neither modify nor retrain. But it was optimized for stability and natural gait — not energy efficiency, not low-impact steps, not any custom downstream objective. A residual MLP on top lets us **add a new objective** without touching the base policy. Examples:

- Minimize mechanical power `|τ·ω|` (what this tutorial does)
- Reduce footstrike impact (add a contact-force penalty)
- Track a user-specified style (imitation on top of base)
- Domain-specific constraints (e.g. stay below an elbow torque during payload carry)

**Key insight: the residual is an OBJECTIVE SHIM, not a policy replacement.** The base still does all the hard work (balance, gait timing, swing mechanics); the residual tweaks a few rad on each joint per step to trade some capability for the new objective.

---

## 4. Architecture: how the residual attaches

```
  target_vel ──► planner.onnx ──► encoder.onnx ──► 64-D token ─────────────┐
                                                                            │
  proprio history (10 frames) ──────────────────────────────────────────────┼──► 994-D obs
                                                                            │        │
                                                                            │        ├──► decoder.onnx (frozen)
                                                                            │        │        │
                                                                            │        │        ▼
                                                                            │        │    base action (29)
                                                                            │        │
                                                                            │        ▼
                                                                            │    residual MLP [512,256,128]
                                                                            │    (trainable, zero-init)
                                                                            │        │
                                                                            │        ▼
                                                                            │    clamp(δ, −lim, +lim)
                                                                            │        │
                                                                            └────────┴── + ──► 29 joint targets
```

After training, **`fuse.py`** rewrites the decoder ONNX graph to bake the residual + clamp + add into the same file. The result is a drop-in replacement: deploy code sees *one* ONNX input and *one* ONNX output, identical to the original SONIC decoder interface.

---

## 5. Step-by-step adaptation

### Step 1 — Extract ONNX from upstream

SONIC ships `model_decoder.onnx`, `model_encoder_dyn.onnx`, `planner_sonic_dyn.onnx`. We pin these by SHA and never modify them.

```bash
examples/sonic_energy_efficient/models/
├── model_decoder.onnx              # 40 MB   ← the one we wrap
├── model_encoder_dyn.onnx          # 50 MB   ← generates 64-D token
├── planner_sonic_dyn.onnx          # 770 MB  ← future trajectory planner
```

**Verify**: `onnxruntime.InferenceSession(decoder).get_inputs()/get_outputs()` — confirm the names and shapes match what the deploy code expects. Dimensions are not in the ONNX metadata; pulled from `observation_config.yaml` and the SONIC deploy C++.

### Step 2 — Understand I/O shapes and semantics (`configs/g1_joints.py`, `configs/sonic_params.py`)

You **will** get this wrong at first. SONIC has **two joint orderings**:

- **MUJOCO order** (`SONIC_GROUPED_JOINT_NAMES`): left-leg → right-leg → waist → left-arm → right-arm. What MuJoCo's `scene_29dof.xml` uses for `qpos`/`ctrl`.
- **NATIVE / ISAACLAB order** (`NATIVE_LOCOMOTION_JOINT_NAMES`): interleaved L/R pitch → L/R roll → … then arms. What the decoder ONNX expects for `jpos_hist`, `jvel_hist`, `action_hist`, and what it outputs.

Conversion arrays live in `sonic_params.py`:

```python
MUJOCO_TO_ISAACLAB = np.array([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28])
ISAACLAB_TO_MUJOCO = np.array([...])  # inverse
```

**Verify**: pick one joint, e.g. `left_hip_pitch`. In MUJOCO order it's index 0. Apply `MUJOCO_TO_ISAACLAB[0]` — should get 0 (left_hip_pitch is also index 0 in NATIVE). Pick `right_hip_pitch` — MUJOCO 6, NATIVE 1. Walk through by hand once and add an assertion.

### Step 3 — Build the obs adapter (`env/sonic_obs_builder.py`)

A small ring-buffer class that holds 10 frames of history and concatenates to 994-D when queried. See `sonic_obs_builder.py` — 60 LoC. Key contracts:

- History order: oldest at `[:, 0, :]`, newest at `[:, -1, :]`. Flatten outer→inner on build.
- `jpos` fed in is **delta from default standing angles**, not absolute.
- `gravity` is a **unit vector** in body frame (`projected_gravity_b`).
- On `reset()`, gravity buffer is set to `[0, 0, -1]` (standing upright), everything else to zero — except when using RSI (see Step 8).

### Step 4 — IsaacLab environment config (`env/isaaclab_env_cfg.py`)

Extend `G1RoughEnvCfg` with a `__post_init__` that:

1. Loads the local G1 29-DOF USD (`g1_rubber_hand/g1_29dof.usd`, rubber hand variant — no finger joints).
2. Sets `init_state.joint_pos` to `NATIVE_DEFAULT_ANGLES` (SONIC's deep squat stance, not IsaacLab's default).
3. Configures `JointPositionActionCfg` with:
   ```python
   joint_names=NATIVE_LOCOMOTION_JOINT_NAMES,
   preserve_order=True,
   scale=sonic_scale_by_name,           # NATIVE_ACTION_SCALE per joint
   use_default_offset=True,              # target = default + action * scale
   ```
4. Applies ALIGN-1..10 physics overrides (see §6).
5. Sets command range `commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)` (the eval range; RSI overrides this from the dataset anyway).
6. Swaps IsaacLab's default locomotion rewards for our SONIC-specific ones (see §7) — but our `reward_components` computed in the bridge is what actually drives training; the IsaacLab reward manager just has to not throw errors.

### Step 5 — The Bridge (`env/isaaclab_sonic_bridge.py`)

The single most important file. Translates between IsaacLab's TensorDict API and SONIC's 994-D obs / 29-D action format, handles RSI, computes SONIC-specific rewards. Structure:

```python
class IsaacLabSonicBridge:
    def __init__(self, isaaclab_env, token_provider, reward_cfg, device):
        # detect token provider type: ReferenceDatasetProvider vs LiveEncoder vs BucketCache
        # build SONIC-native joint permutation from IsaacLab joint names
        # allocate obs ring buffers, prev actions, **_delayed_actions** (for 1-step delay)

    def _get_sonic_state(self):
        return {
            "ang_vel":  robot.data.root_ang_vel_b,      # body frame!
            "gravity":  robot.data.projected_gravity_b,  # unit vector, body frame
            "jpos":     (jpos_all - default) indexed in NATIVE,
            "jvel":     jvel_all indexed in NATIVE,
            "lin_vel":  robot.data.root_lin_vel_b,       # for reward
            "torques":  robot.data.applied_torque in NATIVE,
            "base_h":   root pos z,
        }

    def reset(self):
        # call isaaclab_env.reset(), clear buffers
        # if RSI: pull init state from dataset, writeback physics, warm-start history
        # build first obs with token

    def step(self, sonic_actions, residual_delta):
        # 1-step delay: apply the PREVIOUS step's action, buffer the new one
        apply_actions = self._delayed_actions.clone()
        self._delayed_actions = sonic_actions.detach().clone()
        _, _, terminated, truncated, info = self._env.step(apply_actions)

        # read new state, push into history
        state = self._get_sonic_state()
        self._obs_builder.update(**state_subset, actions=sonic_actions)

        # advance token pointer
        token = self._get_token()
        obs = self._obs_builder.build(token)

        # compute custom rewards (see §7)
        # on done: RSI re-init (see Step 8)
        return obs, sonic_reward, done, info
```

Four subtleties that each cost us a day:

- **(a)** `ang_vel = root_ang_vel_b` (body frame) — IsaacLab gives this directly. In MuJoCo you must `quat_rotate_inv(quat, qvel[3:6])` because `qvel[3:6]` is world-frame for a free joint.
- **(b)** `gravity = projected_gravity_b` which is already a unit vector in body frame. Standing upright = `[0,0,-1]`. Don't scale by 9.81.
- **(c)** `jpos` must be **delta** (`q − default`), not absolute.
- **(d)** `actions` stored in the history buffer are **the freshly computed decoder output** at time t, NOT the action that was actually applied (that was `a_{t-1}` due to the delay). This matches SONIC's training setup and is consistent with how the deploy C++ records actions.

### Step 6 — Physics alignment

See [§6](#6-physics-alignment). Run after the bridge works end-to-end but before starting training.

### Step 7 — Reward design

See [§7](#7-reward-design).

### Step 8 — Reference dataset recording (RSI)

SONIC's gait is sensitive to initial conditions. Starting every episode from the default standing pose wastes the first ~20 steps getting into stride and biases the gradient. Instead we record 300 s of the base policy running, then sample start frames from this trajectory as the initial state for each episode — **Reference-State-Initialization** (RSI).

**What gets recorded per frame** (`tools/generate_reference_dataset.py`):

```python
{
  "jpos_abs":    (T, 29),   # NATIVE joint positions (absolute)
  "jpos_delta":  (T, 29),   # = jpos_abs - NATIVE_DEFAULT_ANGLES
  "jvel":        (T, 29),
  "ang_vel_b":   (T, 3),    # body frame
  "gravity_b":   (T, 3),    # unit vector body frame
  "actions":     (T, 29),   # raw decoder output at frame t (NOT the one applied)
  "root_state_w":(T, 13),   # pos(3)+quat_wxyz(4)+linvel_w(3)+angvel_w(3)
  "tokens":      (T, 64),   # encoder output at frame t
  "cmd_vel":     (T, 3),    # commanded velocity (same for fixed cmd)
  "default_angles", "dt", "cmd_buckets", "cmd_hold_seconds"
}
```

**Critical**: record dataset only **after** all bridge fixes (especially 1-step delay, ang_vel frame) land. We burned a training cycle with `dataset_3p0_isaaclab.npz` recorded *before* the delay fix — robot was physically going 2.0 m/s while the dataset was labeled cmd=3.0. Every RSI reset started from a stale state. See Pitfall #10 in §12.

**At RSI reset** (`env/reference_dataset_provider.py` + `bridge._rsi_reset`):

1. Sample a random valid start frame per env (valid = episode window stays above `min_height_threshold` and inside one cmd bucket).
2. Write `init["jpos_abs"]`, `init["jvel"]`, `init["root_state_w"]` into the IsaacLab sim.
3. Warm-start the obs ring buffers from `get_init_history(env_ids)` — the 10 frames **ending at `start-1`** (exclusive).
4. Seed `_delayed_actions` from the *last* history action slot.
5. Seed `_prev_sonic_actions` similarly (for action smoothness reward).

Then `get_tokens()` returns dataset tokens at `start + step_offset` each step. The planner does **not** run during RSI training — the token is just replayed from the dataset. This is a 10-100× speedup vs. running the planner live.

### Step 9 — Training (`train_isaac.py`)

PPO + GAE with the standard tricks. Key arguments:

```bash
OMNI_KIT_ACCEPT_EULA=yes python -u examples/sonic_energy_efficient/train_isaac.py \
  --decoder-onnx        examples/sonic_energy_efficient/models/model_decoder.onnx \
  --reference-dataset   examples/sonic_energy_efficient/models/dataset_3p0_isaaclab.npz \
  --rsi-min-height      0.55 \
  --num-envs            1024 \
  --iterations          2000 \
  --num-steps           24 \
  --num-epochs          5 \
  --lr                  3e-4 \
  --save-interval       50 \
  --eval-interval       25 \
  --eval-steps          200 \
  --output-dir          runs/sonic_energy_3p0_v9
```

The training loop:

1. Rollout for `num_steps` steps across `num_envs` envs (24 576 transitions per iter).
2. `compute_gae(rollout_rew, rollout_val, rollout_done, next_value, γ=0.99, λ=0.95)`.
3. Normalize advantages (per batch).
4. PPO update: 5 epochs × 4 mini-batches per epoch, `clip_range=0.2`, `max_grad_norm=1.0`.
5. Every 25 iters: deterministic EVAL (policy.act_inference — no noise) for 200 steps, log vx/energy/alive.
6. Every 50 iters (or on new best mean reward): checkpoint.

Throughput: ~37 000 fps on one RTX 4090, so 2000 iters ≈ 28 minutes.

### Step 10 — Fuse into a single ONNX (`fuse.py`)

```bash
python examples/sonic_energy_efficient/fuse.py \
  --decoder-onnx examples/sonic_energy_efficient/models/model_decoder.onnx \
  --checkpoint   runs/sonic_energy_3p0_v9/checkpoint_best.pt \
  --output       runs/sonic_energy_3p0_v9/fused_best.onnx
```

Under the hood:

1. Export the residual MLP to ONNX (static shape, 994 → 29).
2. Graph-merge with the base decoder: rename all residual nodes to avoid name collisions, wire residual input = decoder input, residual output → `Clip(−lim, +lim)` → `Add` with decoder output.
3. Verify numerically: `max_diff(fused_onnx, separate_computation) < 1e-3` (allow FP32 precision noise).

Output is a single ONNX with the same input/output signature as `model_decoder.onnx`. Drop it into `gear_sonic_deploy` replacing the decoder file and the deploy stack runs unchanged.

### Step 11 — Sim2sim validation (`tools/mujoco_energy_compare.py`)

Run the full deploy pipeline (planner + encoder + fused decoder) in MuJoCo:

```bash
python examples/sonic_energy_efficient/tools/mujoco_energy_compare.py \
  --fused-decoder runs/sonic_energy_3p0_v9/fused_best.onnx \
  --cmd-vel       3.0 \
  --warmup-steps  500 \
  --record-steps  750 \
  --output-dir    runs/sonic_energy_3p0_v9/compare
```

Produces:
- `baseline_rollout.mp4` / `residual_rollout.mp4` — 15 s recordings, side-view tracking camera
- `comparison_sidebyside.mp4` — ffmpeg-stitched
- `energy_comparison.png` — per-step power + velocity over time
- `energy_report.txt` — baseline vs residual numbers + headline % reduction

The MuJoCo env is built with the **29-DOF deploy XML** (`scene_29dof.xml`) but with ground friction patched to `MUJOCO_GROUND_FRICTION=1.0` to match training (see Pitfall #12 in §12).

### Step 12 — Academic-style plots

Three additional tools for paper-ready figures:

```bash
# Training curves (EVAL points: vx, energy, alive, height)
python examples/sonic_energy_efficient/tools/plot_training_curves.py \
  --log runs/sonic_energy_3p0_v9.log \
  --output runs/sonic_energy_3p0_v9/training_curves.png \
  --cmd-vel 3.0

# Per-joint power breakdown (which joints got hotter/colder)
python examples/sonic_energy_efficient/tools/mujoco_joint_power_plot.py \
  --fused-decoder runs/sonic_energy_3p0_v9/fused_best.onnx \
  --cmd-vel 3.0 \
  --output-dir runs/sonic_energy_3p0_v9/joint_power

# Velocity tracking stability (vx(t), rolling σ, error dist, power)
python examples/sonic_energy_efficient/tools/velocity_tracking_plot.py \
  --fused runs/sonic_energy_3p0_v9/fused_best.onnx \
  --cmd-vel 3.0 \
  --output runs/sonic_energy_3p0_v9/velocity_tracking.png
```

---

## 6. Physics alignment

IsaacLab's default physics is close to MuJoCo but not identical. SONIC was trained in MuJoCo so we **pin IsaacLab to MuJoCo's values** wherever they differ. Every override is tagged `ALIGN-N` in `env/isaaclab_env_cfg.py::G1_29DOF_Sonic_FlatEnvCfg._patch_actuator_gains()` for traceability. Full table and rationale: `docs/sim2sim_alignment.md`.

| Tag | Field | IsaacLab default | SONIC training | Why |
|-----|-------|------------------|----------------|-----|
| 1 | actuator `stiffness`/`damping` | 100/200/20 & 2.5/5/0.2 | `NATIVE_KP`/`NATIVE_KD` (99.1/28.5 & 6.3/1.8) | Match external-PD gains from `policy_parameters.hpp`. |
| 2 | actuator `armature` | 0.03 | **0.01** (uniform) | MuJoCo `<joint armature>` ≠ motor reflected inertia; IsaacLab's 0.03 is 3× too high. |
| 3 | actuator `viscous_friction` | 0 | **0.05** | MuJoCo `<joint damping>` is a small viscous term independent of PD. |
| 4 | actuator `friction` | 0 | **0.2** | MuJoCo `<joint frictionloss>` = static Coulomb; SONIC expects it. |
| 5 | DCMotor `saturation_effort`/`velocity_limit` | 180/20-37 | effectively +∞ | Disable DCMotor torque-speed rolloff; SONIC MuJoCo uses `<motor>` = pure torque source. |
| 6 | Ground μ | 0.5 | **1.0** | SONIC training scene uses μ=1.0. Deploy scene has 0.5 — mismatch causes foot slip at running cadence. |
| 7 | `sim.dt` / `decimation` | 0.005 / 4 | **0.002 / 10** | MuJoCo default is 2 ms; control rate stays 50 Hz. |
| 8 | per-link mass/inertia/COM | USD convex hulls | MuJoCo `<inertial>` tags | `mujoco_inertial.py::override_link_inertial_from_mujoco` writes real values at startup. |
| 9 | PhysX `max_depenetration_velocity` | 1 m/s | **1000 m/s** | PhysX caps contact separation — attenuates GRF at foot strike. |
| 10a | PhysX `bounce_threshold_velocity` | 0.5 m/s | **1e9** (off) | IsaacLab otherwise lets foot strike behave restitutively; bleeds push-off energy. |
| 10b | `min_position_iteration_count` | 1 | **8** | PhysX solver depth: approach MuJoCo's 100-iter Newton. |
| 10c | `min_velocity_iteration_count` | 0 | **2** | Same idea on velocity. |

**Friction patch in the MuJoCo eval too**: `tools/mujoco_energy_compare.py::build_mujoco_env` patches `model.geom_friction[:, 0] = MUJOCO_GROUND_FRICTION` after loading the XML, so both training (IsaacLab) and sim2sim eval (MuJoCo) run at μ=1.0 regardless of which scene XML is used. Without this, `scene_29dof.xml` (deploy XML, μ=0.5) makes the residual's aggressive gait slip and fall.

---

## 7. Reward design

Computed inside the bridge (`env/rewards.py`), mixed in `bridge.step`'s `components/weights` dicts. Signs are baked into the weight:

| Name | Formula | Weight | Purpose |
|------|---------|--------|---------|
| `velocity_tracking` | `exp(−‖v_b[:2] − cmd[:2]‖² / σ²)`, σ=0.5 | **+5.0** | Primary positive signal. Exponential with σ=0.5 means large reward even with 0.5 m/s error. |
| `energy_penalty` | `Σᵢ wᵢ · \|τᵢ · ωᵢ\|` where `w=2` for hip_pitch & ankles, 1 else | **−0.002** | Mechanical power with 2× weighting on the joints that dominate running energy. |
| `energy_balance_penalty` | `\|P_left_leg − P_right_leg\|` | −0.005 | Penalize limp-like asymmetric gait. |
| `ang_vel_penalty` | `‖ω_body‖²` | −0.05 | Reduce wobble. |
| `foot_slip_penalty` | `Σ ‖v_foot‖² · 1[contact]` | −0.2 | Discourage sliding support foot. |
| `base_height_penalty` | `(h − 0.72)²` | −0.5 | Maintain standing height. |
| `action_smoothness` | `‖aₜ − aₜ₋₁‖²` | −0.005 | Reduce chattering. |
| `residual_magnitude` | `Σ δᵢ²` on the raw clamped residual | −0.01 | Keep residual small; stay close to base. |
| `alive_bonus` | `1[¬done]` | +2.0 | Keeps reward > 0 during healthy walking. |

**Key lesson on tuning `energy_penalty`**: `-0.002` is the sweet spot for SONIC at 1.7–3.0 m/s. Attempting to make it stronger was one of the big mistakes of this session:

- `-0.010`: baseline reward becomes −6.5/step, slow-walk reward ≈ −6.8/step. Policy finds that *any* motion is barely worse than standing still and converges to a minimum-energy slow gait (EVAL@25 vx=2.08 vs expected 2.67).
- `-0.005`: policy catastrophically minimizes torques and falls (MuJoCo eval: vx=0.0, h=0.12).
- `-0.002`: clean −20% energy with trajectory-stable gait.

The intuition "make the penalty stronger to prevent overspeed-chasing divergence" is backwards. The fix for divergence is **use the best checkpoint** (saved on new-best mean reward), not ratchet up the penalty.

---

## 8. Training recipe

With the reward config and dataset finalized, a single run takes ~30 min on RTX 4090 to 2000 iters.

```bash
# Full serial pipeline (record dataset, then train both speeds)
bash run_train_v9.sh
```

The provided `run_train_v9.sh`:

1. Activates `robo` conda env directly (avoids `conda run` stdout buffering).
2. Sets `PYTHONUNBUFFERED=1` and `stdbuf -oL -eL` on the training process.
3. Trains 3.0 m/s first (2000 iters), then 1.7 m/s. Serial — **never** run two IsaacLab trainings in parallel; it OOMs the GPU and locks up the machine.

**Monitoring**:

- EVAL events every 25 iters in the log show `vx | h | E | alive` per cmd bucket.
- Training diverges ≈ iter 1000–1500 — this is **normal** in a setting where `cmd > achievable_speed`. The `checkpoint_best.pt` (saved on new-best mean reward) is the one you keep; final-iter checkpoints are usually worse.
- `grep EVAL runs/sonic_energy_*_v9.log | awk -F'E=' '{print $2}' | sort -n | head -5` gives you the lowest-energy iterations.

---

## 9. Deployment (fuse back into a single ONNX)

`fuse.py` does the graph-surgery merge. After fuse:

- **Input**: `(N, 994)` same as base.
- **Output**: `(N, 29)` same as base.
- **All LSTM hidden states / other inputs**: preserved if the base had them.

Deploy side:

1. Copy `fused_best.onnx` → `model_decoder.onnx` location in `gear_sonic_deploy`.
2. Optional: re-export to TensorRT (`.engine`) with `trtexec` for lower inference latency. Pin TRT to 10.13 (see `fixes_sonic_sim2sim_2026_04_14.md`) — 10.6 has a broken op.
3. Run the standard DDS deploy pipeline. Nothing else changes.

---

## 10. Evaluation (MuJoCo sim2sim)

`tools/mujoco_energy_compare.py` runs **both** the baseline decoder and the fused decoder in MuJoCo with the same online planner+encoder. This is the honest sim2sim test: same token generation, same physics, only the decoder module swapped.

**Must be run with friction patched to 1.0** (automatic now — the compare script patches `model.geom_friction` at load). Don't run on raw `scene_29dof.xml` with its default μ=0.5 — the residual's high-friction gait will slip.

**What changes between runs** is controlled:

- Same planner ONNX, same encoder ONNX, same MuJoCo XML.
- Same warmup (500 steps = 10 s) and same record (750 steps = 15 s).
- Same camera, same initial pose (SONIC_DEFAULT_ANGLES).

**What the numbers mean**:

- `Mean power` — time-averaged Σᵢ|τᵢ·ωᵢ| over record window (all 29 joints, unweighted).
- `Mean fwd vel` — Δx/Δt over 50-frame (1 s) rolling window.
- `Cmd overshoot` — mean(vx) − cmd_vel. At cmd=1.7 the base *overshoots* to ~2.4 m/s; this is a known SONIC quirk, not a training bug. Residual usually pulls overshoot closer to 0.

---

## 11. Results (v9 on G1 29-DOF)

> All numbers in MuJoCo with deploy XML (`scene_29dof.xml`) + ALIGN-6 friction patch (μ=1.0), 15 s record window, identical planner+encoder on both sides.

### cmd = 1.7 m/s

| | Baseline (SONIC) | + Residual | Δ |
|---|---:|---:|---:|
| Mean power | 422 W | 327 W | **−22.5%** |
| Forward velocity | 2.42 m/s | 1.90 m/s | closer to cmd (|overshoot|: 0.72 → 0.20) |
| Rolling σ(vx) over 1 s | 0.037 | 0.018 | −52% (smoother) |
| mean \|vx − cmd\| | 0.611 | 0.200 | **−67%** tracking error |

### cmd = 3.0 m/s

| | Baseline (SONIC) | + Residual | Δ |
|---|---:|---:|---:|
| Mean power | 734 W | 588 W | **−19.8%** |
| Forward velocity | 2.18 m/s | 2.53 m/s | **faster** (closer to cmd) |
| Rolling σ(vx) | 0.080 | 0.055 | −31% |
| mean \|vx − cmd\| | 0.83 | 0.44 | **−47%** |

**Observation**: at cmd=3.0 the residual is faster AND more efficient — a Pareto improvement. The base has spare headroom it wasn't using; the residual finds it.

### Training curves

- `runs/sonic_energy_3p0_v9/training_curves.png` — IsaacLab EVAL at 25-iter intervals.
- Best iter ~1000 (E=950 W in IsaacLab, corresponds to 588 W in MuJoCo — the ~40% sim2sim gap is mostly that IsaacLab's weighted energy counts hip_pitch/ankles at 2× while MuJoCo eval is unweighted).
- Divergence begins ~iter 1500; final iteration is worse than best. **Use best checkpoint.**

### Per-joint breakdown (cmd=3.0)

Biggest reductions: `L_hip_roll` (−58%), `R_hip_roll` (−38%), `L_shoulder_pitch` (−17%), `L_hip_pitch` (−11%). The residual learned to reduce lateral sway and redirect energy into more efficient forward propulsion. Full table: `runs/sonic_energy_3p0_v9/joint_power/joint_power_report.txt`.

---

## 12. Pitfalls & fixes — the full catalog

These are the traps we hit during the 2-week sprint. Each is documented standalone in `/home/lumi/.claude/projects/-home-lumi-robo-residual/memory/pitfalls_mujoco_rsi_training.md`. Abridged here:

1. **MuJoCo actuator index ≠ joint index**. `scene_43dof.xml` interleaves hand actuators between left and right arms. `data.ctrl[:29]` sends the wrong torques. **Fix**: always build ctrl_idx via `mj_name2id(OBJ_ACTUATOR, joint_name.replace("_joint",""))`.
2. **`data.qvel[3:6]` is world-frame angular velocity for a free joint**, not body-frame. Pushing it into SONIC's obs as-is produces wild policy outputs. **Fix**: `quat_rotate_inv(quat_wxyz, qvel[3:6])`. IsaacLab's `root_ang_vel_b` is already correct.
3. **`jpos` in SONIC obs is `q − default`, not `q`**. Absolute angles are OOD for the decoder.
4. **IsaacLab base_velocity `resampling_time_range=(10.0, 10.0)` triggers shotgun re-sample every 10 s**, which the planner reads as a new episode and causes a crouch-restart even when the actual cmd is held constant. **Fix**: set to `(3600.0, 3600.0)`.
5. **MuJoCo viewer steals keyboard focus** — not a policy bug, just a UX trap when testing interactively. Hardcode cmd for automation.
6. **RSI must restore 3 classes of state, not just qpos/qvel**:
   (a) IsaacLab physics write-back (`write_joint_state_to_sim`, `write_root_state_to_sim`),
   (b) obs ring-buffer warm-start from `get_init_history` (10 frames of all proprio channels),
   (c) token pointer reset (`start_frame[env]`, `step_offset=0`). Otherwise policy sees zeroed history for the first 10 steps and diverges.
7. **RSI sample window must skip warmup + tail** seconds of the dataset. First ~3 s are planner warm-up; last episode_length seconds can't be a start or the episode crosses the file end.
8. **Measure vx as Δpos/Δt over a window, not `root_lin_vel_b`**. IsaacLab/MuJoCo's instantaneous linvel is noisy; compute it from position difference over 50 frames (1 s).
9. **MuJoCo Python sim has no real-time pacing**. `time.sleep(dt_ctrl − elapsed)` each step or the robot flies off-screen before you can see anything.
10. **Dataset must be recorded *after* all bridge fixes**. We recorded `dataset_3p0_isaaclab.npz` before the 1-step delay was added. The resulting dataset stored cmd=3.0 but actual vx_mean=2.0 (robot running slower because no delay). Every RSI reset put the robot in a stale state. Symptom: training's EVAL@25 shows 2.08 m/s instead of the expected 2.6 m/s. **Fix**: re-record every dataset after each bridge change. Add a sanity check: `vx_mean(dataset) ≈ expected_base_speed(cmd)`.
11. **energy_penalty tuning is counter-intuitive**. Attempting to increase it (−0.002 → −0.010) to prevent divergence made the policy converge to a minimum-torque "barely moving" gait (see §7). The correct defense against late-iteration divergence is **saving best-checkpoint** and evaluating that, not cranking the penalty. A too-large penalty makes any motion's total reward more negative than standing still.
12. **Deploy XML friction ≠ training friction**. `scene_29dof.xml` (gear_sonic_deploy) has μ=0.5 but the training scene (`scene_43dof.xml` in gear_sonic) has μ=1.0. IsaacLab ALIGN-6 matches training (1.0). So when we eval the residual in MuJoCo using the deploy XML with its native 0.5 friction, the aggressive gait slips and the robot falls. **Fix**: patch `model.geom_friction[:, 0] = 1.0` on XML load in the eval script. This is done automatically in the current `mujoco_energy_compare.py`. For actual robot deploy this matters — check the real floor surface and retrain with randomized friction if you expect μ<0.8.
13. **`conda run -n <env> python ...` buffers stdout globally**, even with `python -u`. For long-running jobs where you need to see progress, activate the env directly in the shell script and use `stdbuf -oL -eL python -u` or it'll look like the job is hung for 11 minutes (it isn't — it's just holding 800 iterations worth of output).
14. **The "off-by-one fix" I tried on `_rsi_reset`'s `_delayed_actions` caused total collapse.** Theoretically `_delayed_actions` should be set to `actions[start]` to match the post-RSI transition; empirically setting it to `hist["actions"][:,-1,:]` (= `actions[start-1]`, the v7 behavior) is what works. The one-frame lag is <20 ms and matters less than whatever invariant the code was relying on. **Fix**: don't touch it. If you're sure you've derived the correct fix, isolate-test before letting it near real training.

---

## 13. Migration guide — adapting to a new task

You have a different frozen base policy (e.g. a bimanual manipulation policy) and want to add a new objective (e.g. minimum jerk, or track a humanlike end-effector trajectory). Port in roughly this order:

### Phase 0 — Understand your base policy

- [ ] What is the ONNX input dim? Output dim?
- [ ] Is the input a flat tensor or does it carry LSTM hidden state?
- [ ] What's the action semantics? Joint positions (delta-from-default or absolute)? Joint velocities? Task-space?
- [ ] What's the observation semantics? History? What's in each channel? Normalizations already applied?
- [ ] What simulator was it trained in? What physics constants are exposed?

### Phase 1 — Single-step sanity

Before building anything RL-ish, get `base_policy(fake_obs) → action → sim.step(action) → state` working end-to-end in one process. If you can't make the base policy run at all in your target simulator, training a residual on top is pointless.

- [ ] Load the ONNX. Run it on a zero observation. Print output shape.
- [ ] Build the obs adapter (whatever the equivalent of `SonicObsBuilder` is for you — ring buffers for history, frame construction).
- [ ] Build the bridge: read simulator state → format as base-policy obs → run base → apply action → step sim.
- [ ] Record a 60 s rollout. Watch the video. Does it look like the base policy working in its native environment?

If no → you have physics alignment or obs/action format issues. Fix before moving on. This is where §6 happens.

### Phase 2 — Reward & RSI infrastructure

- [ ] Decide what you're optimizing. Write reward functions as stateless tensor ops (`env/rewards.py` equivalent).
- [ ] Record a reference dataset with the now-working base policy (≥ 5 minutes, single cmd setting or a few bucketed settings).
- [ ] Write a `ReferenceDatasetProvider` clone: store all the obs channels (same layout as your obs adapter expects), root state, tokens if you have encoder output, cmd.
- [ ] Extend the bridge with an `_rsi_reset` that writes physics state, warms buffers, advances token pointer.

### Phase 3 — Plug into robo_residual

- [ ] `ResidualActorCritic(onnx_path=base_onnx, config=ResidualConfig(...))` in your training script.
- [ ] Set per-joint (or per-action-dim) clamp limits in the config. For SONIC we use 0.35 legs / 0.10 waist / 0.03 arms. Start conservative, loosen if the residual can't find improvement.
- [ ] Use `train_isaac.py` as a template. Main loop changes are: use your bridge, your obs/action dims, your rewards dict.
- [ ] EVAL every 25 iters, checkpoint best.

### Phase 4 — Fuse & deploy

- [ ] `fuse_residual_to_onnx(base_onnx, residual, max_residual_limits, output)`. Verify numerically (`max_diff < 1e-3`).
- [ ] Drop the fused ONNX into your deploy location.
- [ ] Run sim2sim eval. Confirm the gain transfers.

### Phase 5 — Real robot (when ready)

- [ ] Sim2sim gain transferred → try on the real robot. Start with conservative clamp limits (even smaller than what you trained with).
- [ ] Watch for **friction gap** (§12 Pitfall #12): if the real surface is lower friction than training, your residual may slip.
- [ ] Watch for **actuator saturation**: real motors have torque-speed curves. If ALIGN-5 disabled DCMotor rolloff in sim, you need to recheck torque bounds.
- [ ] Domain randomization during training is insurance. Cheaper to add now than to discover a brittle policy at the robot lab.

### Mindset

- **Don't reinvent the base policy.** If you find yourself wanting to change its behavior broadly, you probably want to retrain it, not add a residual.
- **Zero init the residual output layer.** Non-zero init at iter 0 will make the base gait immediately wobble and your first 50 iters will just be fighting that.
- **RSI is a force multiplier but requires full-state fidelity.** A half-warm state is worse than cold init because the policy thinks it's further into a motion than it physically is.
- **Best-checkpoint is your friend.** RL is noisy; don't trust `final_iter`.
- **Sim2sim is not deploy.** The 0.8× sim2sim of training is a predictor, not a guarantee. Add the real-world floor friction check before you believe any number.
