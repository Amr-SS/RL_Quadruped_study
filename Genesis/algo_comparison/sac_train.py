"""
SAC training wrapper for Go2 push-hardening using SB3 + Genesis.
Mirrors sb3_train.py structure with default SB3 SAC hyperparameters.
"""

import sys
import os
import argparse
import subprocess
import numpy as np
import torch
import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# ── CPU thread config ─────────────────────────────────────────────────────────
torch.set_num_threads(16)
torch.set_num_interop_threads(4)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "locomotion")))
from go2_env import Go2Env
from go2_train import get_cfgs

# Import shared components from sb3_train
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sb3_train import GenesisVecEnv, RewardLoggerCallback, unpack_obs


def parse_args():
    parser = argparse.ArgumentParser(description="Go2 SAC Training with SB3 + Genesis")

    # Environment
    parser.add_argument("--num-envs", type=int, default=64)

    # Push perturbation
    parser.add_argument("--push-min", type=float, default=5.0)
    parser.add_argument("--push-max", type=float, default=15.0)
    parser.add_argument("--push-interval", type=int, default=150)
    parser.add_argument("--push-duration", type=int, default=10)

    # Training — using SB3 SAC defaults where possible
    parser.add_argument("--total-steps", type=int, default=14_000_000,
                        help="Total environment steps (SAC needs more exploration)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="Entropy coefficient ('auto' for learned temperature)")
    parser.add_argument("--target-entropy", type=str, default="auto",
                        help="Target entropy for auto ent_coef tuning")

    # Logging / saving
    parser.add_argument("--run-name", type=str, default="sac_push_run")
    parser.add_argument("--save-name", type=str, default="go2_sac_push_hardened")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip auto-generating comparison plots on finish")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Go2 SAC Push-Hardening Training")
    print(f"  num_envs        : {args.num_envs}")
    print(f"  push force      : {args.push_min}-{args.push_max} N")
    print(f"  total steps     : {args.total_steps:,}")
    print(f"  lr              : {args.lr}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  buffer_size     : {args.buffer_size:,}")
    print(f"  ent_coef        : {args.ent_coef}")
    print(f"  run name        : {args.run_name}")
    print("=" * 60)

    gs.init(backend=gs.gpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    raw_env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    env = GenesisVecEnv(
        raw_env,
        num_envs=args.num_envs,
        push_interval=args.push_interval,
        push_force_min=args.push_min,
        push_force_max=args.push_max,
        push_duration=args.push_duration,
    )

    log_dir = os.path.join(os.path.dirname(__file__), "logs", "sac")
    model_dir = os.path.join(os.path.dirname(__file__), "models", "sac")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Parse ent_coef: "auto" stays as string, numeric becomes float
    ent_coef = args.ent_coef
    if ent_coef != "auto":
        ent_coef = float(ent_coef)

    # Parse target_entropy similarly
    target_entropy = args.target_entropy
    if target_entropy != "auto":
        target_entropy = float(target_entropy)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        gamma=args.gamma,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        tensorboard_log=log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=model_dir,
        name_prefix=args.save_name,
        verbose=1,
    )
    reward_cb = RewardLoggerCallback()
    callbacks = CallbackList([checkpoint_cb, reward_cb])

    print("\nStarting SAC training...\n")
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, args.save_name)
    model.save(final_path)
    print(f"\nSAC training complete!")
    print(f"Final model saved to : {final_path}.zip")
    print(f"Checkpoints saved in : {model_dir}/")

    if not args.skip_plots:
        print("\nGenerating comparison plots...")
        plot_script = os.path.join(os.path.dirname(__file__), "generate_plots.py")
        subprocess.run([sys.executable, plot_script], check=False)


if __name__ == "__main__":
    main()
