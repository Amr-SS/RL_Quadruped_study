import sys
import os
import genesis as gs
import torch
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath("examples/locomotion"))
from go2_env import Go2Env
from go2_train import get_cfgs

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "go2_push_hardened_64env"   # change to any checkpoint path
RECORD_VIDEO = True                        # set False to just view without saving
VIDEO_PATH   = "go2_eval.mp4"
MAX_STEPS    = 2000


def unpack_obs(obs):
    """go2_env.get_observations() returns a tuple — safely unpack."""
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    return obs


def main():
    gs.init(backend=gs.gpu)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    scene_kwargs = dict(show_viewer=not RECORD_VIDEO)

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not RECORD_VIDEO,
    )

    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("Model loaded!")

    # ── Start recording ────────────────────────────────────────────────────────
    if RECORD_VIDEO:
        print(f"Recording to: {VIDEO_PATH}")
        env.scene.record_video(VIDEO_PATH, fps=50)

    # ── Reset ──────────────────────────────────────────────────────────────────
    reset_mask = torch.ones(1, dtype=torch.bool, device="cuda")
    env._reset_idx(reset_mask)
    obs = unpack_obs(env.get_observations())

    print(f"Running for {MAX_STEPS} steps...")

    for i in range(MAX_STEPS):
        # Policy inference on CPU
        action, _ = model.predict(obs[0].cpu().numpy(), deterministic=True)

        # Send action to GPU
        action_tensor = torch.tensor(action, device="cuda", dtype=torch.float32).unsqueeze(0)

        # Step physics
        result = env.step(action_tensor)
        obs    = unpack_obs(result[0])
        dones  = result[2]

        if dones[0]:
            print(f"Robot fell at step {i} — resetting")
            env._reset_idx(reset_mask)
            obs = unpack_obs(env.get_observations())

    # ── Save video ─────────────────────────────────────────────────────────────
    if RECORD_VIDEO:
        env.scene.stop_recording()
        print(f"\nVideo saved to: {VIDEO_PATH}")
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()
