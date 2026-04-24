# Project History: robo_residual × SONIC

Chronological record of the robo_residual → SONIC integration, from an empty `examples/` dir to v9 sim2sim-validated residual. Intended as institutional memory: **why** each decision was made, what we **tried and ruled out**, and what the final working state is.

---

## Timeline

### Sprint 1 — **robo_residual toolbox v0.1.0** (repo init)

Initial commit: `304d5e2 Initial commit: robo_residual v0.1.0`. The toolbox was built as a generic residual-on-ONNX utility with no specific robot task in mind. Core API landed:

- `OnnxBasePolicy` — load any ONNX via onnxruntime, auto-detect input/output dims, make batch dim dynamic.
- `ResidualActorCritic` — base + residual MLP + critic. Standard PPO interface.
- `ComposableResidual` — multi-phase stacking.
- `fuse_residual_to_onnx` — ONNX graph surgery.
- Adapters for rsl_rl.

Tested against synthetic ONNX models built in `conftest.py`. 68 tests, no real robot env.

**Decisions carried forward**:
- Zero-init last layer of the residual. At iter 0 the residual outputs zero → policy = base policy.
- Per-joint clamp limits, stored as a tensor in `policy.max_residual_limits`, baked into the fused ONNX via a `Clip` node.
- Residual obs can be larger than base obs (`num_actor_obs > num_base_obs`). Base gets `obs[:, :num_base_obs]`.

### Sprint 2 — SONIC example scaffold (`examples/sonic_energy_efficient/`)

First real target: NVIDIA GR00T-WholeBodyControl's SONIC policy on the Unitree G1. Files created in this phase:

```
configs/g1_joints.py          # 29-joint naming + OBS_LAYOUT dataclass
configs/sonic_params.py       # SONIC_DEFAULT_ANGLES, NATIVE_*, MUJOCO_TO_ISAACLAB, KP/KD, action_scale
configs/sonic_residual_config.py
configs/reward_config.py
env/sonic_obs_builder.py      # 10-frame ring buffer → 994-D concat
env/rewards.py                # velocity_tracking, energy_consumption, etc.
env/sonic_online_planner.py   # planner + encoder_dyn wrapper (live mode)
env/sonic_encoder_wrapper.py  # zero-token provider (no planner)
tools/record_official_sim.py  # capture gear_sonic_deploy rollout for reference
tools/mujoco_sonic_rollout.py # baseline SONIC rollout in MuJoCo
```

Early reward config leaned heavily on standard locomotion reward shaping: velocity tracking as the primary positive, small penalties for energy/smoothness/height. This became the template for v1 and everything after.

### Sprint 3 — The "why is IsaacLab slow" saga (phase6 sim2sim alignment)

**Problem**: same SONIC ONNX that runs at ~2.74 m/s at cmd=3.0 in MuJoCo topped out at ~1.97 m/s in IsaacLab. Gait was visibly different — more lateral sway, less push-off.

**Investigation** (recorded in `phase6_sim2sim_align_2026_04_22.md`):
- Dumped every physics parameter IsaacLab was using and compared against SONIC's training MuJoCo XMLs (`gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`, `scene_43dof.xml`).
- Identified 10+ differences, tagged as ALIGN-1 through ALIGN-10 in `env/isaaclab_env_cfg.py`.
- Applied them one at a time with a 40 s smoke test at cmd ∈ {1.5, 2.0, 2.5, 3.0} after each.

**Outcomes**:
- ALIGN-1..7 (PD gains, armature, viscous/static joint friction, DCMotor rolloff, ground μ, sim.dt): cmd=2.0 `vx_mean` 1.80 → 2.22 m/s.
- ALIGN-8 (per-link mass/inertia/COM override): null result on top speed — but the right physics regardless.
- ALIGN-9 (PhysX max_depenetration_velocity 1 → 1000): null result.
- ALIGN-10 (PhysX bounce_threshold + solver iterations): cmd=2.5 peak jumped to **2.69 m/s** — first time this speed was reached in IsaacLab.

Full alignment table: `docs/sim2sim_alignment.md`.

**Remaining gap** at end of this sprint: cmd=3.0 `vx_mean` stuck at ~1.97 m/s. Something else was wrong.

### Sprint 4 — Two physics bugs that fixed sim2sim parity (this session, early hours)

**Bug A — angular velocity frame in MuJoCo** (`tools/mujoco_energy_compare.py::get_state`):

```python
# WRONG (was in the code):
ang_vel = data.qvel[3:6].copy().astype(np.float32)

# CORRECT:
ang_vel = quat_rotate_inv(quat, data.qvel[3:6].copy()).astype(np.float32)
```

MuJoCo's `qvel[3:6]` for a free joint is the **world-frame** angular velocity, not body-frame. IsaacLab's `root_ang_vel_b` IS body-frame. SONIC's obs expects body-frame. Running without the rotation silently worked (obs values were in the right range) but the policy got wrong information.

**Bug B — 1-step action delay** (`isaaclab_sonic_bridge.py::step`, `tools/mujoco_energy_compare.py`):

GR00T's deploy stack is async: the decoder at step t outputs action a_t, but the DDS bus takes one tick to deliver it, so the robot actually applies a_{t-1} at step t. SONIC was trained with this delay — it's not a deploy artifact, it's part of the system the policy learned to work with. Neither IsaacLab nor the MuJoCo Python bridge had this delay.

```python
# Added in bridge.step():
apply_actions = self._delayed_actions.clone()
self._delayed_actions = sonic_actions.detach().clone()
_, *_ = self._env.step(apply_actions)

# And initialize self._delayed_actions in __init__
# And zero it on env reset / warm-start it on RSI reset
```

**Result**: IsaacLab EVAL@25 at cmd=3.0 jumped from ~2.04 m/s to ~2.67 m/s. MuJoCo max_vx at cmd=3.0 jumped from ~2.0 to 2.74 m/s ≈ GR00T DDS's measured 2.743 m/s ✓.

### Sprint 5 — RSI training infrastructure (pre-this-session)

`env/reference_dataset_provider.py` + `bridge._rsi_reset` + `tools/generate_reference_dataset.py`. Captures a 5-minute base-policy rollout, stores (jpos, jvel, ang_vel_b, gravity_b, actions, root_state_w, tokens, cmd) per frame, and at training time every reset pulls a random valid start frame:

1. Write physics state to the sim (`write_joint_state_to_sim`, `write_root_state_to_sim`).
2. Warm-start the obs ring buffer with the 10 frames before the start (`get_init_history`).
3. Seed `_delayed_actions`, `_prev_sonic_actions` from the history.
4. Set dataset pointer to the start frame; each step advances it by 1 (tokens replayed, planner not re-run).

Decision log:
- **Valid-start filter**: reject windows that cross a cmd-bucket boundary OR contain any frame below `min_height_threshold` (initially 0.65, later 0.55 — see below).
- **No online planner during training**: 10-100× speedup vs. running the planner live, since the planner is 770 MB and the encoder is 50 MB of ONNX.
- **Training and validation rollouts both use RSI**: online-planner code exists in the bridge but is deploy-only. Mixing the two is an invitation for bugs.

### Sprint 6 — First residual training (v1, v2 — pre-this-session)

**v1** (`sonic_energy_1p7_v1`): cmd=1.7 m/s, energy_penalty=-0.0005, clamp=0.05 rad on legs. Result: 3.1% MuJoCo energy reduction, saturated at iter ~500.

**v2** (`sonic_energy_1p7_v2`): raised `energy_penalty=-0.005`, clamp=0.10. Result: **−5.3% MuJoCo energy** at cmd=1.7 m/s, mean_vx 2.00 → 1.89. Side-by-side video (`comparison_sidebyside.mp4`) clearly showed reduced arm swing and less lateral hip motion. First publishable result.

Issue at end of Sprint 6: vx at cmd=1.7 in MuJoCo was **2.00 m/s** (base overshoot), not the 2.4 m/s we see now. Reason: MuJoCo eval did not have the 1-step delay yet (that was Sprint 4's fix, but only landed for the IsaacLab bridge; the MuJoCo Python script was patched separately and later).

### Sprint 7 — 3.0 m/s failure mode and the dataset bug (this session, morning)

Ran `sonic_energy_3p0_v6`/`v7` to get the higher-speed result. At iter ~275 training **diverged**: EVAL energy jumped from ~1050 W (−10% vs baseline 1176 W) to 1400-1600 W while vx dropped from 2.58 to 2.2. Final policy was *worse* than the base.

**Initial hypothesis**: `energy_penalty=-0.002` too weak. Policy found that using +300 W more torque gets vx from 2.67 to 2.96 m/s → large velocity_tracking reward increase (+3.13/step) vs. small energy cost (−0.6/step).

Interventions tried:
1. `energy_penalty = -0.010`: training collapsed to a minimum-energy slow gait (EVAL@25 vx=2.08). Total reward became dominated by energy, and the policy discovered that standing still has reward ≈ 0 while walking has reward ≈ −6. Reverted.
2. `energy_penalty = -0.005`: policy **fell** (iter_850 in MuJoCo eval: vx=0.0, h=0.12). Reverted.
3. "RSI off-by-one fix": theoretically `_delayed_actions` should be `actions[start]` not `hist["actions"][:,-1,:]` (= `actions[start-1]`). Empirically applying this "fix" caused total collapse. Reverted (kept `actions[start-1]` — the v7 behavior).

**Root-cause that actually stuck** (found by `stat`-ing the dataset npz): `dataset_3p0_isaaclab.npz` was recorded *before* Sprint 4's 1-step delay landed in the bridge. At recording time, the robot was only running at **vx_mean = 1.99 m/s** despite being commanded 3.0. Every RSI reset put the robot in a slow-gait state, then asked it to track 3.0 m/s — a recipe for bad training.

**Fix**: re-record both `dataset_3p0_isaaclab.npz` and `dataset_1p7_isaaclab.npz` with the delay-fixed bridge. New vx_mean: 2.71 (3p0) and 2.28 (1p7). Also lowered `rsi_min_height_threshold` from 0.65 to 0.55 because a robot running at 3 m/s naturally pitches forward and the pelvis is often at 0.67 m.

### Sprint 8 — v9 (this session, afternoon): clean win

Configuration that worked:
- Re-recorded datasets
- `energy_penalty = -0.002` (same as v7, the baseline that was working)
- RSI `_delayed_actions` reverted to `hist["actions"][:,-1,:]` (v7 behavior)
- `rsi_min_height=0.55`
- All other bridge/env settings unchanged

Training ran 2000 iters in ~30 min on RTX 4090. EVAL curve:

| iter | cmd=3.0 EVAL | | cmd=1.7 EVAL | |
|-----:|-----:|:-----|-----:|:-----|
| 25   | vx=2.61 E=1282 | baseline | vx=2.25 E=803 | baseline |
| 150  | vx=2.64 E=1162 | **−9%** | vx=2.11 E=664 | **−17%** |
| 500  | vx=2.89 E=1061 | **−17%** | vx=1.93 E=574 | **−28%** |
| 625  | vx=2.97 E=1030 | **−20%** | **BEST iter=625** | **−29%** |
| 1000 | vx=2.93 E=**950** | **−26% BEST** | — | — |
| 1500 | vx=2.84 E=1490 | divergence | — | — |
| 2000 | vx=2.84 E=1493 | — | vx=2.32 E=849 | diverged |

Best checkpoints saved at iter ~625 (1.7 m/s) and iter ~1000 (3.0 m/s). Final iters diverged but we keep best.

### Sprint 9 — Sim2sim validation & the friction patch (this session, evening)

MuJoCo eval with default `scene_29dof.xml` (the deploy target): **residual falls** at cmd=3.0 (vx=0.0, h=0.12). Base decoder: works fine. So the residual exploits something IsaacLab allows that MuJoCo(deploy-XML) doesn't.

Investigation showed the deploy XML has `<geom friction="0.5"/>` while the training XML (`scene_43dof.xml`) has `friction="1.0"`. IsaacLab's ALIGN-6 matched training (1.0). So the residual's gait is calibrated for μ=1.0 and slips on μ=0.5.

**Fix**: `tools/mujoco_energy_compare.py::build_mujoco_env` now patches `model.geom_friction[:, 0] = MUJOCO_GROUND_FRICTION` (=1.0) regardless of XML. Final sim2sim numbers:

- cmd=1.7: 422 W → 327 W (**−22.5%**), vx 2.42 → 1.90 (closer to cmd).
- cmd=3.0: 734 W → 588 W (**−19.8%**), vx 2.18 → 2.53 (faster AND more efficient).
- Velocity tracking σ: −31% at 3.0 m/s, −52% at 1.7 m/s.
- Tracking error: −47% at 3.0 m/s, −67% at 1.7 m/s.

**Real-robot implication**: deploy target is the 29-DOF deploy XML which represents whatever floor the real robot will face. If the real surface is closer to μ=0.5 than μ=1.0, the current residual will likely slip. Before a physical deploy, either retrain with domain-randomized friction OR verify the real-floor μ.

### Sprint 10 — Archival & tutorial (this session, end)

Three documents produced:
- `docs/TUTORIAL.md` — step-by-step walkthrough with migration guide.
- `docs/PROJECT_HISTORY.md` — this file.
- Updated `README.md` with v9 results.

Tools added this session:
- `play_residual.py` — IsaacSim window with a residual checkpoint loaded (visual inspection).
- `tools/velocity_tracking_plot.py` — 4-panel vx stability figure.

---

## Things we tried and ruled out

- **Per-motor-type armature** (0.00361/0.0251 per motor family): came from `policy_parameters.hpp`, was initially conflated with MuJoCo `<joint armature>`. These are *reflected motor inertias* used to derive Kp/Kd, not the simulator armature. The XML uses a uniform 0.01 → that's what ALIGN-2 applies.
- **DCMotor rolloff alone**: without ALIGN-2/3/4 in place, disabling DCMotor saturation just made actuators over-strong at low speed. Yaw rate oscillated ±3.4 rad/s, vy drifted to −3 m/s. Must apply physics alignments together.
- **`energy_penalty` > -0.002**: collapses policy to minimum-torque slow gait (§11 Pitfall).
- **`_delayed_actions = ds.actions[starts]` in RSI**: theoretically better, empirically catastrophic. Leaving as `hist["actions"][:,-1,:]`.
- **pty-simulated keyboard input to deploy's stdin**: the deploy C++ reads keyboard only from active terminal focus; a pty workaround collided with momentum-decay logic.
- **Running two IsaacLab trainings in parallel**: GPU OOMs and the whole machine locks up, requiring a forced reboot.

---

## Final directory structure

```
examples/sonic_energy_efficient/
├── README.md                           # High-level example description
├── docs/
│   ├── TUTORIAL.md                     # This tutorial (complete process)
│   ├── PROJECT_HISTORY.md              # This file (timeline)
│   ├── sim2sim_alignment.md            # ALIGN-1..10 rationale
│   └── results/                        # v2 MuJoCo comparison artifacts
├── configs/
│   ├── g1_joints.py                    # Joint naming, OBS_LAYOUT
│   ├── reward_config.py                # Reward weights
│   ├── sonic_params.py                 # Physics constants, joint-order permutations
│   ├── sonic_residual_config.py        # Residual MLP + per-joint clamp
│   └── train_config.py                 # Training hyperparameters
├── env/
│   ├── isaaclab_env_cfg.py             # G1_29DOF_Sonic_FlatEnvCfg (ALIGN-1..10)
│   ├── isaaclab_sonic_bridge.py        # ★ The bridge: IsaacLab ↔ SONIC format
│   ├── reference_dataset_provider.py   # RSI: valid-starts, init state/history lookup
│   ├── rewards.py                      # Per-term reward functions
│   ├── sonic_obs_builder.py            # 994-D obs ring buffer
│   ├── sonic_online_planner.py         # Deploy-faithful planner (not used in RSI training)
│   ├── sonic_encoder_wrapper.py        # Zero-token provider (degraded fallback)
│   ├── mujoco_inertial.py              # ALIGN-8 runtime inertia override
│   └── ...
├── models/
│   ├── model_decoder.onnx              # Frozen SONIC decoder (base)
│   ├── model_encoder_dyn.onnx          # Frozen encoder (64-D token)
│   ├── planner_sonic_dyn.onnx          # Frozen planner
│   ├── dataset_3p0_isaaclab.npz        # Reference trajectory at 3.0 m/s
│   ├── dataset_1p7_isaaclab.npz        # Reference trajectory at 1.7 m/s
│   └── g1_rubber_hand/g1_29dof.usd     # Robot USD (rubber hand, no finger joints)
├── tools/
│   ├── generate_reference_dataset.py   # Record RSI dataset in IsaacLab
│   ├── mujoco_energy_compare.py        # ★ Sim2sim: baseline vs residual, video + report
│   ├── mujoco_joint_power_plot.py      # Per-joint power breakdown
│   ├── velocity_tracking_plot.py       # vx(t), rolling σ, error dist (added this session)
│   ├── plot_training_curves.py         # Parse log → EVAL curves
│   └── ...
├── train_isaac.py                      # Main PPO training script
├── fuse.py                             # Bake residual into single ONNX
├── play_residual.py                    # IsaacSim viewport with residual (added this session)
├── view_sonic.py                       # IsaacSim viewport with base only
├── eval.py
├── eval_sonic_mujoco.py
└── eval_decoder_only.py
```

---

## Versions & artifacts

| Version | Date | Cmd | Δ Energy (MuJoCo) | Checkpoint | Notes |
|---------|------|----:|------------------:|------------|-------|
| v1 | 2026-04-16 | 1.7 | −3.1% | iter~500 | First working, clamp=0.05 |
| v2 | 2026-04-17 | 1.7 | **−5.3%** | iter~1000 | clamp=0.10, published |
| v5 | 2026-04-19 | 1.7 | −3% | — | Reward experiments |
| v6 | 2026-04-22 | 3.0 | — | diverged | First 3 m/s attempt |
| v7 | 2026-04-24 | 3.0 | −3.4% | iter 225 | Best pre-dataset-fix |
| v7 | 2026-04-24 | 1.7 | −5.9% | best | |
| v8 | 2026-04-24 | 1.7 | **−22.1%** | iter 300 | Still using old dataset |
| **v9** | **2026-04-24** | **3.0** | **−19.8%** | **iter ~1000** | **New dataset, all fixes** |
| **v9** | **2026-04-24** | **1.7** | **−22.5%** | **iter ~625** | |

Current canonical artifacts:

- `runs/sonic_energy_3p0_v9/fused_best_iter1000.onnx` — deployable 3.0 m/s residual
- `runs/sonic_energy_1p7_v9/fused_best.onnx` — deployable 1.7 m/s residual
- `runs/sonic_energy_3p0_v9/compare/*` — side-by-side video + energy plot
- `runs/sonic_energy_1p7_v9/compare/*`
- `runs/sonic_energy_*_v9/joint_power/*` — per-joint breakdown
- `runs/sonic_energy_*_v9/velocity_tracking.png` — stability analysis
- `runs/sonic_energy_*_v9/training_curves.png` — IsaacLab EVAL curves

---

## Remaining open questions (not blockers for v9)

- **Why does training diverge around iter 1000–1500?** Hypothesis: as the policy finds lower-energy gaits, the value function loses calibration (advantages get noisier), and eventually PPO updates become too large for the narrow "good region". Mitigations not yet tried: LR cosine decay, KL-adaptive LR, early stopping, or simply more conservative clip_range.
- **Does the residual transfer to the real robot?** Unknown. Depends on the real floor friction (see Sprint 9). Should be the next empirical test.
- **Can we train with randomized friction to get μ-robustness?** Yes, IsaacLab's event system supports per-episode friction randomization; we just haven't added it. For the next push this is the obvious extension.
- **Does a smaller residual (fewer params) work just as well?** Haven't ablated. Current MLP is [512, 256, 128] = 1.35M params. Might get the same result with [128, 64] (~0.1M). Would simplify deployment.
