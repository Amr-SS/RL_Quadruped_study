import sys
import os
import argparse
import genesis as gs
import torch
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath("examples/locomotion"))
from go2_env import Go2Env
from go2_train import get_cfgs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Go2 SB3 model and record video")
    parser.add_argument("--model",      type=str, default="go2_push_hardened_64env",
                        help="Path to model zip (no extension)")
    parser.add_argument("--video",      type=str, default=None,
                        help="Output video path (default: <model>_eval.mp4)")
    parser.add_argument("--max-steps",  type=int, default=2000,
                        help="Number of steps to run")
    parser.add_argument("--no-video",   action="store_true",
                        help="Disable video recording, show viewer instead")
    return parser.parse_args()


def unpack_obs(obs):
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    # Always return CPU numpy
    if isinstance(obs, torch.Tensor):
        return obs.cpu().numpy().astype(np.float32)
    return obs.astype(np.float32)


def set_forward_command(env):
    """Force forward velocity command — prevents _resample_commands overwriting it."""
    env.commands[:, 0] = 0.5  # forward x
    env.commands[:, 1] = 0.0  # lateral y
    env.commands[:, 2] = 0.0  # yaw


def main():
    args = parse_args()

    MODEL_PATH   = args.model
    RECORD_VIDEO = not args.no_video
    VIDEO_PATH   = args.video if args.video else f"{args.model}_eval.mp4"
    MAX_STEPS    = args.max_steps

    gs.init(backend=gs.gpu)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Override command range to match training (0.5 m/s forward)
    command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"]   = [0.0, 0.0]

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.no_video,
    )

    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("Model loaded!")

    # ── Camera setup ───────────────────────────────────────────────────────────
    if RECORD_VIDEO:
        cam = env.cam
        frames = []
        print(f"Recording to: {VIDEO_PATH}")

    # ── Reset ──────────────────────────────────────────────────────────────────
    reset_mask = torch.ones(1, dtype=torch.bool, device="cuda")
    env._reset_idx(reset_mask)
    set_forward_command(env)
    # FIX: _reset_idx stores dof_pos in URDF joint order, but observations
    # expect joint_names order. Robot is at default pose after reset.
    env.dof_pos[:] = env.default_dof_pos
    env._update_observation()
    obs = unpack_obs(env.get_observations())

    print(f"Running for {MAX_STEPS} steps...")
    print(f"Forward command: {env.commands[0].cpu().numpy()}")

    for i in range(MAX_STEPS):
        # Policy inference
        action, _ = model.predict(obs[0], deterministic=True)

        # Send action to GPU
        action_tensor = torch.tensor(action, device="cuda", dtype=torch.float32).unsqueeze(0)

        # Step physics
        result = env.step(action_tensor)
        obs    = unpack_obs(result[0])
        dones  = result[2]

        # Force command every step so resample can't overwrite it
        set_forward_command(env)

        # Capture frame
        if RECORD_VIDEO:
            robot_pos = env.base_pos[0].cpu().numpy()
            cam.set_pose(
                pos=(robot_pos[0] - 3.5, robot_pos[1], robot_pos[2] + 2.5),
                lookat=(robot_pos[0], robot_pos[1], robot_pos[2] + 0.5),
            )
            rgb, depth, seg, normal = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
            frames.append(rgb)

        if dones[0]:
            print(f"Robot fell at step {i} — resetting")
            env._reset_idx(reset_mask)
            set_forward_command(env)
            env.dof_pos[:] = env.default_dof_pos  # FIX: match joint_names order
            env._update_observation()
            obs = unpack_obs(env.get_observations())

    # ── Save video ─────────────────────────────────────────────────────────────
    if RECORD_VIDEO and frames:
        try:
            import imageio
            print(f"\nSaving {len(frames)} frames to {VIDEO_PATH}...")
            imageio.mimwrite(VIDEO_PATH, frames, fps=50, quality=8)
            print(f"Video saved to: {VIDEO_PATH}")
        except ImportError:
            import cv2
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(
                VIDEO_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                50, (w, h)
            )
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Video saved to: {VIDEO_PATH}")
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()
