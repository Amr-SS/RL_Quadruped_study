"""
TD3 training wrapper for Go2 push-hardening using SB3 + Genesis.
Mirrors sb3_train.py structure with TD3-specific hyperparameters.
"""

import sys
import os
import argparse
import subprocess
import numpy as np
import torch
import genesis as gs
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
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
    parser = argparse.ArgumentParser(description="Go2 TD3 Training with SB3 + Genesis")

    # Environment
    parser.add_argument("--num-envs", type=int, default=64)

    # Push perturbation
    parser.add_argument("--push-min", type=float, default=5.0)
    parser.add_argument("--push-max", type=float, default=15.0)
    parser.add_argument("--push-interval", type=int, default=150)
    parser.add_argument("--push-duration", type=int, default=10)

    # Training
    parser.add_argument("--total-steps", type=int, default=9_000_000,
                        help="Total environment steps (TD3 needs more than PPO)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                        help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Steps of random exploration before learning")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient for target networks")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Std dev for NormalActionNoise on each action dim")
    parser.add_argument("--policy-delay", type=int, default=2,
                        help="Delay policy updates (TD3 trick)")

    # Logging / saving
    parser.add_argument("--run-name", type=str, default="td3_push_run")
    parser.add_argument("--save-name", type=str, default="go2_td3_push_hardened")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip auto-generating comparison plots on finish")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Go2 TD3 Push-Hardening Training")
    print(f"  num_envs        : {args.num_envs}")
    print(f"  push force      : {args.push_min}-{args.push_max} N")
    print(f"  total steps     : {args.total_steps:,}")
    print(f"  lr              : {args.lr}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  buffer_size     : {args.buffer_size:,}")
    print(f"  noise_std       : {args.noise_std}")
    print(f"  policy_delay    : {args.policy_delay}")
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

    # TD3 exploration noise — independent Gaussian per action dimension
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_std * np.ones(n_actions),
    )

    log_dir = os.path.join(os.path.dirname(__file__), "logs", "td3")
    model_dir = os.path.join(os.path.dirname(__file__), "models", "td3")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = TD3(
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
        action_noise=action_noise,
        policy_delay=args.policy_delay,
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

    print("\nStarting TD3 training...\n")
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, args.save_name)
    model.save(final_path)
    print(f"\nTD3 training complete!")
    print(f"Final model saved to : {final_path}.zip")
    print(f"Checkpoints saved in : {model_dir}/")

    if not args.skip_plots:
        print("\nGenerating comparison plots...")
        plot_script = os.path.join(os.path.dirname(__file__), "generate_plots.py")
        subprocess.run([sys.executable, plot_script], check=False)


if __name__ == "__main__":
    main()
