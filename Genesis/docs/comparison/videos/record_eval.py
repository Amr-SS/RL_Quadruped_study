"""
Record a fixed-camera, fixed-duration evaluation video for ANY of the three
algorithms (PPO / SAC / TD3) from a saved Stable-Baselines3 checkpoint.

Designed for an apples-to-apples side-by-side comparison:
  * identical environment (Go2Env, flat plane, same config)
  * identical forward command (0.5 m/s)
  * identical camera (chase camera locked to a fixed offset from the base)
  * identical duration and frame rate

The robot is NOT artificially reset on falls, so the video honestly shows each
final policy's stability (a collapsed policy will fall and stay down).

Usage:
  python docs/comparison/videos/record_eval.py --algo ppo \
      --model go2_ppo_comparison --steps 500 \
      --out docs/comparison/videos/raw_ppo.mp4
"""

import os
import sys
import argparse
import numpy as np
import torch

# Resolve repo paths regardless of where the script is called from
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # Genesis/
sys.path.append(os.path.join(ROOT, "examples", "locomotion"))

import genesis as gs
from go2_env import Go2Env
from go2_train import get_cfgs

ALGOS = {"ppo": "PPO", "sac": "SAC", "td3": "TD3"}


def load_model(algo, path):
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path)
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC.load(path)
    if algo == "td3":
        from stable_baselines3 import TD3
        return TD3.load(path)
    raise ValueError(f"unknown algo {algo!r}")


def unpack(obs):
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    if isinstance(obs, torch.Tensor):
        return obs.cpu().numpy().astype(np.float32)
    return obs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=list(ALGOS))
    ap.add_argument("--model", required=True, help="checkpoint path (no .zip needed)")
    ap.add_argument("--steps", type=int, default=500, help="control steps to record (50 = 1s)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--allow-reset", action="store_true",
                    help="reset the robot on fall (default: let it stay down — honest stability view)")
    args = ap.parse_args()

    gs.init(backend=gs.gpu)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]

    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)

    print(f"[{ALGOS[args.algo]}] loading {args.model}")
    model = load_model(args.algo, args.model)

    cam = env.cam
    frames = []

    reset_mask = torch.ones(1, dtype=torch.bool, device="cuda")
    env._reset_idx(reset_mask)
    env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = 0.5, 0.0, 0.0
    env.dof_pos[:] = env.default_dof_pos
    env._update_observation()
    obs = unpack(env.get_observations())

    for i in range(args.steps):
        action, _ = model.predict(obs[0], deterministic=True)
        action_t = torch.tensor(action, device="cuda", dtype=torch.float32).unsqueeze(0)
        result = env.step(action_t)
        obs = unpack(result[0])
        dones = result[2]
        env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = 0.5, 0.0, 0.0

        robot_pos = env.base_pos[0].cpu().numpy()
        cam.set_pose(
            pos=(robot_pos[0] - 3.5, robot_pos[1], robot_pos[2] + 2.5),
            lookat=(robot_pos[0], robot_pos[1], robot_pos[2] + 0.5),
        )
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        frames.append(np.asarray(rgb))

        if dones[0] and args.allow_reset:
            env._reset_idx(reset_mask)
            env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = 0.5, 0.0, 0.0
            env.dof_pos[:] = env.default_dof_pos
            env._update_observation()
            obs = unpack(env.get_observations())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import imageio
    imageio.mimwrite(args.out, frames, fps=50, quality=8)
    print(f"[{ALGOS[args.algo]}] wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
