# Residual Locomotion Toolbox

> The LoRA of locomotion policy training.

Take any trained policy (ONNX), freeze it, learn a small clamped residual on top — without retraining from scratch. After training, fuse everything into a single ONNX for deployment. Works with any robot (biped, quadruped, humanoid) and any base architecture (MLP, LSTM, CNN).

```
base.onnx (your trained policy)
    + ResidualConfig (per-joint clamp limits)
    → Train residual MLP via PPO
    → fused.onnx (single file, deploy to robot)
```

## Example: SONIC × Energy Efficiency

The first end-to-end example trains on top of NVIDIA's [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) humanoid locomotion policy on the Unitree G1 (29 DOF). Validated in MuJoCo sim2sim, ~1000 PPO iterations:

| | cmd = 1.7 m/s | cmd = 3.0 m/s |
|---|---|---|
| Energy reduction | **−22.5%** (422 W → 327 W) | **−19.8%** (734 W → 588 W) |
| Velocity tracking error | −67% | −47% |

| 1.7 m/s | 3.0 m/s |
|---|---|
| ![1.7 m/s baseline vs residual](https://github.com/user-attachments/assets/0fccf5af-3122-4a45-8f24-cdc7e1da9b36) | ![3.0 m/s baseline vs residual](https://github.com/user-attachments/assets/a22c8d1f-821b-475e-8087-a4a155126c92) |
| ![](examples/sonic_energy_efficient/docs/results/1p7_v9/energy_comparison.png) | ![](examples/sonic_energy_efficient/docs/results/3p0_v9/energy_comparison.png) |

→ **[Full example, quickstart, and adaptation guide](examples/sonic_energy_efficient/README.md)**

---

## Install

```bash
cd robo_residual
pip install -e .
```

Dependencies: `torch`, `onnx`, `onnxruntime`.

## Quick Start

### 1. Define your residual config

Decide which joints get how much residual budget. Like LoRA rank — different body parts get different budgets.

```python
from robo_residual import ResidualConfig, JointGroupConfig

config = ResidualConfig(
    joint_groups=[
        # Legs: small residual (preserve gait)
        JointGroupConfig("legs", indices=list(range(12)), max_residual=0.05),
        # Arms: larger residual (fix arm waggling)
        JointGroupConfig("arms", indices=list(range(12, 29)), max_residual=0.15),
    ],
    residual_hidden_dims=[256, 128],   # residual MLP size
    residual_activation="elu",
    init_noise_std=0.1,                # exploration noise
    default_max_residual=0.1,          # for joints not in any group
)
```

### 2. Create the residual actor-critic

Point it at your base ONNX and the config. That's it.

```python
from robo_residual import ResidualActorCritic

policy = ResidualActorCritic(
    onnx_path="path/to/base_policy.onnx",
    config=config,
    device="cuda",
)
```

What happens under the hood:
- The ONNX model is loaded via `onnxruntime` (frozen, no gradients)
- Batch dimension is made dynamic automatically (your ONNX can be exported with batch=1)
- A residual MLP is created and **zero-initialized** (initial behavior = base policy exactly)
- A fresh critic is created for value estimation
- Input/output dimensions are auto-detected from the ONNX graph

### 3. Train with PPO

The module exposes the standard PPO interface. Wire it into your training loop:

```python
import torch

optimizer = torch.optim.Adam(policy.trainable_parameters, lr=5e-5)

for iteration in range(num_iterations):
    # --- Rollout ---
    obs = env.get_observations()          # (num_envs, obs_dim)
    actions = policy.act(obs)             # sample from distribution
    next_obs, rewards, dones, infos = env.step(actions)

    # --- PPO update ---
    # These work exactly like rsl_rl's ActorCritic:
    log_prob = policy.get_actions_log_prob(actions)
    values = policy.evaluate(critic_obs)
    entropy = policy.entropy
    action_mean = policy.action_mean
    action_std = policy.action_std

    # ... your PPO loss computation here ...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

For inference (no noise):

```python
actions = policy.act_inference(obs)  # deterministic: base + clamped residual
```

### 4. Export fused ONNX for deployment

After training, merge base + residual into a single ONNX file:

```python
from robo_residual import fuse_residual_to_onnx

fuse_residual_to_onnx(
    base_onnx_path="path/to/base_policy.onnx",
    residual_module=policy.residual,            # the trained residual MLP
    max_residual_limits=policy.max_residual_limits,
    output_path="path/to/fused_policy.onnx",
    num_obs=policy.num_actor_obs,
)
```

The fused model computes `base(obs) + clamp(residual(obs), -limits, +limits)` in a single forward pass. The fusion uses **ONNX graph surgery** — it merges both ONNX graphs at the graph level, so it works with any base architecture (MLP, LSTM, CNN, multi-input). Deploy the output the same way you deploy any ONNX policy — `onnxruntime`, TensorRT, etc.

## Different Observations for Residual vs Base

The residual MLP can take larger observations than the base policy. For example, the base might only see proprioception (48 dims), while the residual additionally sees terrain heightmaps (128 dims total).

```python
policy = ResidualActorCritic(
    onnx_path="base_policy.onnx",
    config=config,
    num_actor_obs=128,    # residual sees 128-dim obs
    num_base_obs=48,      # base ONNX only needs first 48 dims
    num_critic_obs=200,   # critic can have its own obs dim
    device="cuda",
)

# During training:
actor_obs = env.get_observations()  # (N, 128) — full observation
actions = policy.act(actor_obs)     # base gets actor_obs[:, :48], residual gets all 128

# Critic uses separate obs:
critic_obs = env.get_privileged_obs()  # (N, 200)
values = policy.evaluate(critic_obs)
```

The base automatically receives `actor_obs[:, :num_base_obs]`. Override `_get_base_obs()` for custom slicing logic.

## Observation Normalization

Enable running-mean normalization (Welford's algorithm) for actor and/or critic observations:

```python
policy = ResidualActorCritic(
    onnx_path="base_policy.onnx",
    config=config,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    device="cuda",
)

# Update running stats each step during rollout:
policy.update_normalization(actor_obs, critic_obs)
```

The normalizer zero-centers and scales observations to unit variance, updated online from training data.

## rsl_rl Integration

For projects using [rsl_rl](https://github.com/leggedrobotics/rsl_rl) with TensorDict observations, use the drop-in wrapper:

```python
from robo_residual import RslRlResidualActorCritic

# rsl_rl passes TensorDict obs with named groups
obs = env.get_observations()  # {"policy": tensor, "critic": tensor, ...}

policy = RslRlResidualActorCritic(
    obs=obs,
    obs_groups={
        "policy": ["policy"],           # actor uses "policy" group
        "critic": ["critic"],           # critic uses "critic" group
    },
    onnx_path="base_policy.onnx",
    config=config,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    device="cuda",
)

# Use with rsl_rl's OnPolicyRunner as normal:
actions = policy.act(obs)                   # accepts TensorDict
values = policy.evaluate(obs)               # accepts TensorDict
policy.update_normalization(obs)            # accepts TensorDict
log_prob = policy.get_actions_log_prob(actions)
```

### Multiple observation groups

If your actor concatenates multiple observation sources:

```python
obs_groups = {
    "policy": ["proprio", "terrain"],       # actor = proprio + terrain
    "critic": ["proprio", "terrain", "privileged"],  # critic gets more
}
obs = {
    "proprio": torch.randn(N, 48, device="cuda"),
    "terrain": torch.randn(N, 100, device="cuda"),
    "privileged": torch.randn(N, 20, device="cuda"),
}
# actor_obs = cat(proprio, terrain) → (N, 148)
# critic_obs = cat(proprio, terrain, privileged) → (N, 168)
```

### Using ObsAdapter standalone

```python
from robo_residual import ObsAdapter

adapter = ObsAdapter({"policy": ["proprio", "terrain"], "critic": ["critic"]})
actor_obs = adapter.get_actor_obs(obs)   # flat tensor
critic_obs = adapter.get_critic_obs(obs) # flat tensor
```

## Multi-Phase Training (Composable Residuals)

Stack multiple residual phases, each targeting a different objective:

```
Phase 0: base.onnx (e.g. AMP natural gait)
Phase 1: + residual for energy optimization   → fuse → phase1.onnx
Phase 2: + residual for terrain adaptation     → fuse → phase2.onnx
Phase 3: + residual for arm task               → training...
```

### Step 1: Train phase 1, fuse it

```python
# Train phase 1 as shown above, then fuse
fuse_residual_to_onnx(base_onnx, policy.residual, policy.max_residual_limits,
                      "phase1_fused.onnx", num_obs=...)
```

### Step 2: Use fused phase 1 as new base, OR stack explicitly

**Option A — Sequential fuse (simpler):**

Just use `phase1_fused.onnx` as the base for phase 2:

```python
policy_phase2 = ResidualActorCritic(
    onnx_path="phase1_fused.onnx",   # already includes base + phase1
    config=phase2_config,
    device="cuda",
)
# Train, then fuse again → phase2_fused.onnx
```

**Option B — Explicit stacking (more control):**

```python
from robo_residual.core.composable import ComposableResidual

policy = ComposableResidual(
    base_onnx_path="base.onnx",
    frozen_residual_onnx_paths=["phase1_residual.onnx"],  # exported residual MLP only
    frozen_residual_limits=[phase1_limits],
    active_config=phase2_config,    # config for the new phase
    device="cuda",
)
# Train phase 2...
```

This gives independent clamp limits per phase and keeps the base ONNX untouched.

## Working with Different ONNX Models

The toolbox auto-detects input/output from the ONNX graph. Examples of supported models:

| Model Type | Inputs | Outputs | Works? |
|-----------|--------|---------|--------|
| Simple MLP | `obs [B, D]` | `actions [B, A]` | Yes |
| LSTM/Recurrent | `obs [B, D]` + `h_in` + `c_in` | `actions [B, A]` + `h_out` + `c_out` | Yes (use `forward_full` for hidden states) |
| Multi-input | `motion [B, 58]` + `command [B, 3]` | `motion [B, 58]` | Yes (pass extra inputs as kwargs) |

### Specifying which input/output to use

If your ONNX has multiple inputs or outputs, tell the config which one is the main observation and action:

```python
config = ResidualConfig(
    joint_groups=[...],
    base_obs_input=0,          # index of the obs input (default: 0)
    base_action_output=0,      # index of the action output (default: 0)
    # or use names:
    # base_obs_input="obs",
    # base_action_output="actions",
)
```

### Inspecting your ONNX model

```python
from robo_residual import OnnxBasePolicy

base = OnnxBasePolicy("your_model.onnx", device="cuda")
print(f"Inputs:  {base.input_names}")     # e.g. ['obs']
print(f"Outputs: {base.output_names}")    # e.g. ['actions']
print(f"Obs dim: {base.num_obs}")         # e.g. 495
print(f"Act dim: {base.num_actions}")     # e.g. 29
print(f"Stateful: {base.is_stateful}")   # True if LSTM
```

### Using OnnxBasePolicy standalone

You can also use `OnnxBasePolicy` on its own to run any ONNX model:

```python
base = OnnxBasePolicy("policy.onnx", device="cuda")
obs = torch.randn(128, base.num_obs, device="cuda")
actions = base.forward(obs)  # (128, 29)

# For LSTM models, pass hidden states:
actions = base.forward(obs, h_in=h, c_in=c)

# Get all outputs (including hidden states):
outputs = base.forward_full(obs=obs, h_in=h, c_in=c)
# outputs = {"actions": ..., "h_out": ..., "c_out": ...}
```

## API Reference

### `ResidualConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `joint_groups` | `list[JointGroupConfig]` | `[]` | Per-group clamp limits |
| `residual_hidden_dims` | `list[int]` | `[256, 128]` | Residual MLP hidden layers |
| `residual_activation` | `str` | `"elu"` | Activation function |
| `init_noise_std` | `float` | `0.1` | Initial exploration noise |
| `noise_std_type` | `str` | `"log"` | `"log"` or `"scalar"` |
| `default_max_residual` | `float` | `0.1` | Limit for joints not in any group |
| `base_obs_input` | `int \| str` | `0` | Which ONNX input is the observation |
| `base_action_output` | `int \| str` | `0` | Which ONNX output is the action |

### `JointGroupConfig`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Group name (e.g. `"legs"`, `"arms"`) |
| `indices` | `list[int]` | Joint indices in the action vector |
| `max_residual` | `float` | Max correction in radians |

### `ResidualActorCritic`

| Method / Property | Description |
|-------------------|-------------|
| `act(obs)` | Sample action (training, with noise) |
| `act_inference(obs)` | Deterministic action (deployment) |
| `evaluate(critic_obs)` | Critic value estimate |
| `get_actions_log_prob(actions)` | Log probability under current distribution |
| `update_normalization(actor_obs, critic_obs)` | Update running normalization stats |
| `entropy` | Distribution entropy |
| `action_mean` / `action_std` | Distribution statistics |
| `trainable_parameters` | Parameters for the optimizer |
| `reset(dones)` | Reset internal state (no-op for MLP) |
| `num_actor_obs` / `num_base_obs` / `num_actions` | Dimension properties |
| `residual` / `critic` / `base` | Sub-module access |
| `max_residual_limits` | Per-joint clamp limits tensor |

### `RslRlResidualActorCritic`

Drop-in replacement for rsl_rl's `ActorCritic`. Same interface as `ResidualActorCritic` but accepts TensorDict observations and handles obs group concatenation automatically.

| Constructor arg | Type | Description |
|----------------|------|-------------|
| `obs` | `dict[str, Tensor]` | Initial observation (for dim inference) |
| `obs_groups` | `dict[str, list[str]]` | Maps `"policy"` / `"critic"` to obs group names |
| `onnx_path` | `str` | Path to base ONNX |
| `config` | `ResidualConfig` | Residual config |
| `num_base_obs` | `int \| None` | Base obs dim (auto-detected if None) |
| `actor_obs_normalization` | `bool` | Enable actor obs normalization |
| `critic_obs_normalization` | `bool` | Enable critic obs normalization |

### `fuse_residual_to_onnx`

```python
fuse_residual_to_onnx(
    base_onnx_path,          # path to base ONNX
    residual_module,         # trained nn.Module (policy.residual)
    max_residual_limits,     # per-joint limits (policy.max_residual_limits)
    output_path,             # where to save fused ONNX
    num_obs,                 # observation dimension
    obs_input_name=None,     # base obs input name (auto-detected if None)
    action_output_name="actions",  # output name in fused ONNX
)
```

Uses ONNX graph surgery to merge both models at the graph level. All base inputs are preserved (including LSTM hidden states). The residual shares the main obs input with the base.

## Project Structure

```
robo_residual/
├── pyproject.toml
├── robo_residual/
│   ├── __init__.py
│   ├── core/
│   │   ├── onnx_base.py              # Load ONNX, make batch dynamic, run inference
│   │   ├── residual_actor_critic.py   # ONNX base + PyTorch residual + critic
│   │   ├── composable.py             # Multi-phase residual stacking
│   │   └── fuse_onnx.py              # ONNX graph surgery: merge base + residual
│   ├── config/
│   │   └── residual_config.py         # JointGroupConfig, ResidualConfig
│   ├── adapters/
│   │   ├── obs_adapter.py            # TensorDict obs_groups → flat tensors
│   │   └── rsl_rl_wrapper.py         # rsl_rl-compatible ActorCritic wrapper
│   └── utils/
│       ├── freeze.py                 # freeze_module / unfreeze_module
│       ├── zero_init.py              # Zero-init residual output layer
│       └── normalizer.py            # EmpiricalNormalization (Welford's)
└── tests/                            # 68 tests, all run on GPU when available
```

## Saving / Loading During Training

The toolbox doesn't prescribe a checkpoint format. Save what you need:

```python
# Save
torch.save({
    "residual_state_dict": policy.residual.state_dict(),
    "critic_state_dict": policy.critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "log_std": policy.log_std.data,
    "iteration": iteration,
    "base_onnx_path": "path/to/base.onnx",  # for reproducibility
}, "checkpoint.pt")

# Load
ckpt = torch.load("checkpoint.pt")
policy.residual.load_state_dict(ckpt["residual_state_dict"])
policy.critic.load_state_dict(ckpt["critic_state_dict"])
policy.log_std.data = ckpt["log_std"]
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

## Tests

```bash
cd robo_residual
pytest tests/ -v
```

68 tests covering: ONNX loading, residual clamping, gradient flow, ONNX graph surgery fusion, config validation, multi-phase stacking, obs override, normalization, rsl_rl wrapper, obs adapter. All run on GPU when available.

## Roadmap

- [x] End-to-end example on [SONIC / GR00T-WBC](https://github.com/NVlabs/GR00T-WholeBodyControl) — −20% energy, −47–67% tracking error ([see example](examples/sonic_energy_efficient/README.md))
- [ ] LSTM / recurrent residual MLP — for tasks requiring memory in the residual
- [ ] Support PyTorch `.pt` checkpoints as base policy — fuse residual weights directly into the original model without ONNX conversion
- [ ] Training visualization — log per-joint residual delta magnitude, clamp saturation rate (wandb / tensorboard)
- [ ] Pre-built joint group configs for common robots (Unitree G1, H1, Go2, etc.)
