"""Rollout check: SONIC decoder with RSI from reference dataset (same as training).

Runs the base decoder (residual=0 or loaded from checkpoint) in IsaacLab using
ReferenceDatasetProvider — the exact same setup as training.

Usage:
    cd /home/lumi/robo_residual/examples

    # Baseline (zero residual):
    OMNI_KIT_ACCEPT_EULA=yes python -u -m sonic_energy_efficient.rollout_check \
        --decoder-onnx sonic_energy_efficient/models/model_decoder.onnx \
        --dataset sonic_energy_efficient/models/dataset_1p7.npz \
        --steps 1000 --headless false

    # Trained residual:
    OMNI_KIT_ACCEPT_EULA=yes python -u -m sonic_energy_efficient.rollout_check \
        --decoder-onnx sonic_energy_efficient/models/model_decoder.onnx \
        --dataset sonic_energy_efficient/models/dataset_1p7.npz \
        --checkpoint ../runs/sonic_energy_1p7_v2/checkpoint_best.pt \
        --steps 1000 --headless false
"""
from __future__ import annotations
import argparse, sys

sys.path.insert(0, "/home/lumi/robo_residual")
import examples.sonic_energy_efficient._ort_cuda_setup  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-onnx", type=str, required=True)
    p.add_argument("--dataset", type=str,
                   default="sonic_energy_efficient/models/dataset_1p7.npz")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint_best.pt to load residual weights. "
                        "If omitted, residual=0 (baseline).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--headless", type=lambda s: s.lower() in ("true", "1", "yes"), default=True)
    args = p.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import carb
    carb.settings.get_settings().set_string(
        "/persistent/isaac/asset_root/cloud",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

    sys.path.insert(0, "/home/lumi/robo_residual")
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from robo_residual import ResidualActorCritic

    from examples.sonic_energy_efficient.configs.g1_joints import NUM_JOINTS, OBS_LAYOUT
    from examples.sonic_energy_efficient.configs.reward_config import RewardConfig
    from examples.sonic_energy_efficient.configs.sonic_residual_config import build_sonic_residual_config
    from examples.sonic_energy_efficient.env.isaaclab_env_cfg import G1_29DOF_Sonic_FlatEnvCfg
    from examples.sonic_energy_efficient.env.isaaclab_sonic_bridge import IsaacLabSonicBridge
    from examples.sonic_energy_efficient.env.reference_dataset_provider import ReferenceDatasetProvider

    device = "cuda"

    env_cfg = G1_29DOF_Sonic_FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 10.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    _data = np.load(args.dataset)
    _dt = float(_data["dt"][0]) if "dt" in _data else 0.02
    _total = int(_data["tokens"].shape[0])
    _warmup = max(10, int(10.0 / _dt))
    _max_start = _total - max(0, int(10.0 / _dt))
    print(f"[rollout] dataset: {_total} frames ({_total*_dt:.0f}s), "
          f"RSI window [{_warmup}, {_max_start})")
    del _data

    token_provider = ReferenceDatasetProvider(
        dataset_path=args.dataset,
        num_envs=1,
        device=device,
        history_frames=10,
        episode_frames=500,
        sample_start_min=_warmup,
        sample_start_max=_max_start,
    )

    bridge = IsaacLabSonicBridge(env, token_provider, RewardConfig(), device=device)

    policy = ResidualActorCritic(
        onnx_path=args.decoder_onnx,
        config=build_sonic_residual_config(),
        num_actor_obs=OBS_LAYOUT.total_dim,
        num_critic_obs=OBS_LAYOUT.total_dim,
        device=device,
    ).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.residual.load_state_dict(ckpt["residual_state_dict"])
        print(f"[rollout] Loaded residual from {args.checkpoint} (iter {ckpt['iteration']}, rew={ckpt['mean_reward']:.3f})")
        label = f"RESIDUAL (iter={ckpt['iteration']})"
    else:
        print("[rollout] No checkpoint — running baseline (residual=0)")
        label = "BASELINE"

    policy.eval()

    obs = bridge.reset()
    print(f"[rollout] [{label}] obs shape: {obs.shape}, starting {args.steps}-step rollout")

    robot = env.scene["robot"]
    dt = env.step_dt
    print(f"[rollout] step_dt={dt:.4f}s  ({1/dt:.0f}Hz)")

    energy_log, vx_log = [], []

    with torch.no_grad():
        for t in range(args.steps):
            actions = policy.act_inference(obs)
            obs, rewards, dones, info = bridge.step(actions)

            # Track energy and velocity
            rc = info["reward_components"]
            energy_log.append(rc["energy_penalty"][0].item())
            vx_log.append(info["lin_vel"][0, 0].item())

            # Follow robot with camera every 5 steps
            if t % 5 == 0:
                pos = robot.data.root_pos_w[0].cpu().numpy()
                env.sim.set_camera_view(
                    eye=(pos[0] - 2.5, pos[1] - 2.5, pos[2] + 1.5),
                    target=(pos[0], pos[1], pos[2] + 0.5),
                )

            if t % 50 == 0 or t == args.steps - 1:
                lin_vel = robot.data.root_lin_vel_b[0]
                base_h = robot.data.root_pos_w[0, 2].item()
                cmd = info["commands"][0]
                E_mean = float(sum(energy_log[-50:]) / max(1, len(energy_log[-50:])))
                print(
                    f"[t={t:3d} / {t*dt:5.2f}s] "
                    f"cmd_vx={cmd[0].item():+.2f} "
                    f"actual_vx={lin_vel[0].item():+.3f} "
                    f"base_h={base_h:.3f} "
                    f"E={E_mean:.1f}W "
                    f"rew={rewards[0].item():+.3f}"
                )

            if dones.any():
                print(f"[rollout] episode done at step {t}, resetting")
                obs = bridge.reset()

    # Summary
    import statistics
    mean_E  = statistics.mean(energy_log)
    mean_vx = statistics.mean(vx_log)
    print(f"\n{'='*55}")
    print(f"  [{label}] Summary over {args.steps} steps ({args.steps*dt:.0f}s)")
    print(f"  Mean energy : {mean_E:.1f} W")
    print(f"  Mean vx     : {mean_vx:.3f} m/s")
    print(f"{'='*55}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
