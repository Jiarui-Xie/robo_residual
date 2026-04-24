"""Real training entry point: SONIC decoder + residual MLP on IsaacLab G1 29DOF.

Usage:
    cd /home/lumi/robo_residual/examples/sonic_energy_efficient
    OMNI_KIT_ACCEPT_EULA=yes python train_isaac.py \
        --decoder-onnx models/model_decoder.onnx \
        --num-envs 1024 --iterations 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Preload nvidia libs BEFORE any onnxruntime import so CUDAExecutionProvider loads.
sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401  (side effect: preload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--planner-onnx", type=str, default=None,
                   help="SONIC planner ONNX path")
    p.add_argument("--encoder-onnx", type=str, default=None,
                   help="SONIC encoder ONNX. For --live-encoder, must be the batch-patched version.")
    p.add_argument("--live-encoder", action="store_true",
                   help="Run encoder every step with per-env dynamic tokens (real SONIC behavior)")
    p.add_argument("--recorded-motion", type=str, default=None,
                   help="Path to recorded .npz motion (motion_50hz, jvel_50hz). "
                        "Use instead of planner offline generation to avoid run-stop wrap artifacts.")
    p.add_argument("--online-planner", action="store_true",
                   help="Deploy-faithful online planner (UpdateContextFromMotion + splice). "
                        "Small num_envs only — planner is batch=1 CPU. Takes precedence over --live-encoder.")
    p.add_argument("--reference-dataset", type=str, default=None,
                   help="Offline reference dataset (.npz) for RSI training. "
                        "Takes precedence over all other token providers.")
    p.add_argument("--rsi-warmup-seconds", type=float, default=10.0,
                   help="Skip this many seconds from the start of the dataset when sampling RSI start frames. "
                        "Avoids planner warm-up artifacts (default: 10s).")
    p.add_argument("--rsi-tail-seconds", type=float, default=10.0,
                   help="Skip this many seconds from the end of the dataset when sampling RSI start frames. "
                        "Ensures every episode has at least this many frames remaining (default: 10s).")
    p.add_argument("--rsi-min-height", type=float, default=0.55,
                   help="Reject RSI start windows that contain any frame with base_h below this. "
                        "For 3 m/s running (robot pitches forward), use 0.55. Default 0.55.")
    p.add_argument("--token-cache", type=str, default=None,
                   help="Path to saved token cache .npz (load instead of recompute)")
    p.add_argument("--output-dir", type=str, default="runs/isaac_sonic_energy")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--num-steps", type=int, default=24)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=25,
                   help="Run deterministic eval every N iterations (0=disable)")
    p.add_argument("--eval-steps", type=int, default=200,
                   help="Steps per deterministic eval rollout (4s at 50Hz)")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint .pt to resume from (loads residual, critic, log_std)")
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="PPO entropy bonus coefficient (0=off, prevents log_std explosion)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 1. Start IsaacSim FIRST (before importing isaaclab / torch-CUDA etc.) ----
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    # ---- 2. IsaacLab imports (require app running) ----
    sys.path.insert(0, "/home/lumi/robo_residual")

    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from robo_residual import ResidualActorCritic

    from examples.sonic_energy_efficient.configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from examples.sonic_energy_efficient.configs.reward_config import RewardConfig
    from examples.sonic_energy_efficient.configs.sonic_residual_config import build_sonic_residual_config
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg
    from examples.sonic_energy_efficient.env.isaaclab_sonic_bridge import IsaacLabSonicBridge
    from examples.sonic_energy_efficient.env.sonic_encoder_wrapper import ZeroTokenProvider
    from examples.sonic_energy_efficient.env.sonic_live_encoder import SonicLiveEncoder
    from examples.sonic_energy_efficient.env.sonic_online_planner import SonicOnlinePlannerEncoder
    from examples.sonic_energy_efficient.env.sonic_token_cache import VelocityBucketTokenCache
    from examples.sonic_energy_efficient.env.reference_dataset_provider import ReferenceDatasetProvider
    from examples.sonic_energy_efficient.train import compute_gae, ppo_update

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Create IsaacLab env ----
    env_cfg = G1_29DOF_Sonic_FlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 10.0
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[train_isaac] IsaacLab env: {args.num_envs} envs, device={isaac_env.device}")

    # ---- 4. SONIC token provider ----
    if args.reference_dataset:
        print(f"[train_isaac] RSI mode: loading reference dataset from {args.reference_dataset}")
        # Compute sampling window from seconds → frames (dataset dt saved in npz if available)
        _tmp_data = np.load(args.reference_dataset)
        _dt = float(_tmp_data["dt"][0]) if "dt" in _tmp_data else 0.02
        _total = int(_tmp_data["tokens"].shape[0])
        _warmup_frames = max(10, int(args.rsi_warmup_seconds / _dt))
        _tail_frames   = max(0,  int(args.rsi_tail_seconds   / _dt))
        _max_start = _total - _tail_frames
        print(f"[train_isaac] RSI window: frames [{_warmup_frames}, {_max_start}) "
              f"of {_total}  ({args.rsi_warmup_seconds:.0f}s warmup, {args.rsi_tail_seconds:.0f}s tail skipped)")
        del _tmp_data
        _episode_frames = int(env_cfg.episode_length_s / _dt)
        token_provider = ReferenceDatasetProvider(
            dataset_path=args.reference_dataset,
            num_envs=args.num_envs,
            device=device,
            history_frames=10,
            episode_frames=_episode_frames,
            min_height_threshold=args.rsi_min_height,
            sample_start_min=_warmup_frames,
            sample_start_max=_max_start,
        )
        print(f"[train_isaac] Dataset: {token_provider.total_frames} frames, "
              f"cmds={sorted(set(token_provider.cmd_vel[:, 0].cpu().numpy().round(2).tolist()))}")
    elif args.online_planner and args.planner_onnx and args.encoder_onnx:
        print("[train_isaac] Using deploy-faithful online planner (continuous run)")
        token_provider = SonicOnlinePlannerEncoder(
            planner_onnx=args.planner_onnx,
            encoder_onnx_dyn=args.encoder_onnx,
            num_envs=args.num_envs,
            target_vel=2.5,                 # overridden per step via set_target_vel if needed
            replan_interval_steps=25,       # 0.5s
            motion_look_ahead_steps=5,
            device=device,
        )
    elif args.live_encoder and args.planner_onnx and args.encoder_onnx:
        if args.recorded_motion:
            print(f"[train_isaac] Live encoder + recorded motion (avoids wrap artifacts): {args.recorded_motion}")
        else:
            print("[train_isaac] Live encoder + planner offline (WARNING: 73-frame wrap causes run-stop)")
        token_provider = SonicLiveEncoder(
            planner_onnx=args.planner_onnx,
            encoder_onnx_dyn=args.encoder_onnx,
            num_envs=args.num_envs,
            velocity_buckets=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            device=device,
            recorded_motion_path=args.recorded_motion,
        )
    elif args.token_cache:
        print(f"[train_isaac] Loading token cache from {args.token_cache}")
        token_provider = VelocityBucketTokenCache.load(args.token_cache, device=device)
    elif args.planner_onnx and args.encoder_onnx:
        token_provider = VelocityBucketTokenCache(
            planner_onnx=args.planner_onnx,
            encoder_onnx=args.encoder_onnx,
            velocity_buckets=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            device=device,
        )
        token_provider.save(output_dir / "token_cache.npz")
    else:
        print("[train_isaac] WARNING: no planner/encoder — using zero tokens (decoder will NOT walk)")
        token_provider = ZeroTokenProvider(device=device)

    bridge = IsaacLabSonicBridge(
        isaac_env, token_provider, RewardConfig(), device=device
    )

    # ---- 5. Residual actor-critic ----
    residual_cfg = build_sonic_residual_config()
    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=residual_cfg,
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(policy.trainable_parameters, lr=args.lr)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        policy.residual.load_state_dict(ckpt["residual_state_dict"])
        policy.critic.load_state_dict(ckpt["critic_state_dict"])
        with torch.no_grad():
            policy.log_std.copy_(ckpt["log_std"])
        print(f"[train_isaac] Resumed from {args.resume_from} "
              f"(iter={ckpt.get('iteration','?')} rew={ckpt.get('mean_reward',float('nan')):+.3f})")

    n_params = sum(p.numel() for p in policy.trainable_parameters)
    print(f"[train_isaac] Residual params: {n_params:,}")
    print(f"[train_isaac] Output: {output_dir}")

    # ---- 6. Training loop ----
    num_envs = args.num_envs
    num_steps = args.num_steps
    best_reward = float("-inf")

    obs = bridge.reset()
    print(f"[train_isaac] Initial obs: {obs.shape}")

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()

        # Rollout
        all_obs, all_act, all_lp, all_rew, all_done, all_val = [], [], [], [], [], []
        all_comps: list[dict] = []
        with torch.no_grad():
            for _ in range(num_steps):
                actions = policy.act(obs)
                log_prob = policy.get_actions_log_prob(actions)
                value = policy.evaluate(obs).squeeze(-1)
                next_obs, rewards, dones, info = bridge.step(actions, policy.last_residual_delta)

                all_obs.append(obs)
                all_act.append(actions)
                all_lp.append(log_prob)
                all_rew.append(rewards)
                all_done.append(dones)
                all_val.append(value)
                all_comps.append({k: v.detach() for k, v in info["reward_components"].items()})
                obs = next_obs
            next_value = policy.evaluate(obs).squeeze(-1)

        # Mean reward components over entire rollout (not just last step)
        comp_keys = list(all_comps[0].keys())
        comp_mean = {k: torch.stack([c[k] for c in all_comps]).mean().item() for k in comp_keys}

        rollout_obs = torch.stack(all_obs)
        rollout_act = torch.stack(all_act)
        rollout_lp = torch.stack(all_lp)
        rollout_rew = torch.stack(all_rew)
        rollout_done = torch.stack(all_done)
        rollout_val = torch.stack(all_val)

        # GAE
        advantages, returns = compute_gae(
            rollout_rew, rollout_val, rollout_done, next_value, 0.99, 0.95
        )
        adv_std = advantages.std().item()

        # Flatten
        flat_obs = rollout_obs.reshape(-1, OBS_LAYOUT.total_dim)
        flat_act = rollout_act.reshape(-1, NUM_JOINTS)
        flat_lp = rollout_lp.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_ret = returns.reshape(-1)

        # PPO update
        batch_size = num_steps * num_envs
        mini_batch_size = batch_size // 4
        ep_loss = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_up = 0
        for _ in range(args.num_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                losses = ppo_update(
                    policy, optimizer,
                    flat_obs[idx], flat_act[idx], flat_lp[idx],
                    flat_adv[idx], flat_ret[idx],
                    0.2, args.entropy_coef, 0.5, 1.0,
                )
                for k in ep_loss:
                    ep_loss[k] += losses[k]
                n_up += 1
        for k in ep_loss:
            ep_loss[k] /= max(n_up, 1)

        mean_rew = rollout_rew.mean().item()
        elapsed = time.time() - t0
        fps = (num_steps * num_envs) / elapsed
        ep_done_rate = rollout_done.float().mean().item()

        if iteration % args.log_interval == 0 or iteration == 1:
            print(
                f"[iter {iteration:>4d}/{args.iterations}] "
                f"rew={mean_rew:+.3f} "
                f"vel={comp_mean['velocity_tracking']:+.3f} "
                f"E={comp_mean['energy_penalty']:.3f} "
                f"ω²={comp_mean['ang_vel_penalty']:.3f} "
                f"h={comp_mean['base_height_penalty']:.3f} "
                f"ploss={ep_loss['policy_loss']:+.4f} "
                f"vloss={ep_loss['value_loss']:.3f} "
                f"ent={ep_loss['entropy']:.2f} "
                f"adv_std={adv_std:.3f} "
                f"done%={ep_done_rate*100:.1f} "
                f"fps={fps:.0f}"
            )

        # Deterministic eval (no noise, fixed residual).
        # Per-env cmd varies (dataset has 4 buckets), so bucket vx by cmd to
        # see tracking quality at each speed. Averaging across cmds is misleading.
        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            eval_vx_per_step: list[torch.Tensor] = []     # (num_envs,) each
            eval_cmd_per_step: list[torch.Tensor] = []    # (num_envs,) each
            eval_alive: list[float] = []
            eval_h: list[float] = []
            eval_energy: list[float] = []
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    eval_act = policy.act_inference(obs)
                    obs, _, _, eval_info = bridge.step(eval_act, policy.last_residual_delta)
                    eval_vx_per_step.append(eval_info["lin_vel"][:, 0].detach())
                    eval_cmd_per_step.append(eval_info["commands"][:, 0].detach())
                    eval_alive.append(eval_info["reward_components"]["alive_bonus"].mean().item())
                    eval_h.append(eval_info["reward_components"]["base_height_penalty"].mean().item())
                    eval_energy.append(eval_info["reward_components"]["energy_penalty"].mean().item())
            vx_flat = torch.stack(eval_vx_per_step).reshape(-1)        # (steps * N,)
            cmd_flat = torch.stack(eval_cmd_per_step).reshape(-1)
            # Bucket by unique cmd values actually seen in eval
            bucket_edges = torch.unique(cmd_flat.round(decimals=1))
            bucket_str_parts = []
            for c in bucket_edges:
                mask = (cmd_flat - c).abs() < 0.05
                if mask.any():
                    mean_vx = vx_flat[mask].mean().item()
                    n = mask.sum().item()
                    bucket_str_parts.append(f"cmd{c.item():.1f}→vx{mean_vx:+.2f}(n={n})")
            buckets_str = " ".join(bucket_str_parts) if bucket_str_parts else "(none)"
            mean_eval_energy = sum(eval_energy) / len(eval_energy)
            print(
                f"  [EVAL@{iteration}] {buckets_str} "
                f"alive={sum(eval_alive)/len(eval_alive):.2f} "
                f"h={sum(eval_h)/len(eval_h):+.4f} "
                f"E={mean_eval_energy:.1f}W"
            )

        # Checkpoint
        if iteration % args.save_interval == 0 or mean_rew > best_reward:
            if mean_rew > best_reward:
                best_reward = mean_rew
                tag = "best"
            else:
                tag = f"iter_{iteration}"
            ckpt_path = output_dir / f"checkpoint_{tag}.pt"
            torch.save({
                "iteration": iteration,
                "residual_state_dict": policy.residual.state_dict(),
                "critic_state_dict": policy.critic.state_dict(),
                "log_std": policy.log_std.data,
                "mean_reward": mean_rew,
            }, ckpt_path)

    print(f"\n[train_isaac] Done. Best reward: {best_reward:+.3f}")
    isaac_env.close()
    app.close()


if __name__ == "__main__":
    main()
