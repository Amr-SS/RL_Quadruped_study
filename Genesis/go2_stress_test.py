import sys
import os
import torch
import genesis as gs
import numpy as np

sys.path.append(os.path.abspath("examples/locomotion"))
from go2_env import Go2Env
from go2_train import get_cfgs

# ── Config ─────────────────────────────────────────────────────────────────────
# Switch between RSL-RL and SB3 models by changing MODE
MODE = "sb3"       # "rsl" or "sb3"

RSL_CKPT  = "logs/go2-walking/model_100.pt"
SB3_CKPT  = "train_4_walk"   # .zip added automatically by SB3


def unpack_obs(obs):
    """go2_env.get_observations() returns a tuple — safely unpack."""
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    return obs


def apply_lateral_force(env, magnitude):
    """Apply a lateral Y-axis force to the robot base link."""
    solver         = env.robot._solver
    base_link_idx  = env.robot.links[0].idx

    force = torch.zeros(1, 1, 3, device="cuda", dtype=torch.float32)
    force[0, 0, 1] = magnitude

    try:
        solver.apply_links_external_force(
            force=force,
            links_idx=[base_link_idx],
            envs_idx=None,
        )
    except AttributeError:
        solver.links_state[base_link_idx, 0].cfrc_ext_vel[1] -= magnitude


def main():
    gs.init(backend=gs.gpu)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # ── Load policy ────────────────────────────────────────────────────────────
    if MODE == "rsl":
        from rsl_rl.modules import ActorCritic
        actor_critic = ActorCritic(
            num_actor_obs=45, num_critic_obs=45, num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
        ).to("cuda")
        actor_critic.load_state_dict(torch.load(RSL_CKPT)["model_state_dict"])
        actor_critic.eval()
        print(f"Loaded RSL-RL model: {RSL_CKPT}")

        def get_action(obs_tensor):
            with torch.no_grad():
                return actor_critic.act(obs_tensor)

    else:
        from stable_baselines3 import PPO
        model = PPO.load(SB3_CKPT)
        print(f"Loaded SB3 model: {SB3_CKPT}")

        def get_action(obs_tensor):
            action, _ = model.predict(obs_tensor.cpu().numpy(), deterministic=True)
            return torch.tensor(action, device="cuda", dtype=torch.float32)

    # ── Reset ──────────────────────────────────────────────────────────────────
    reset_mask = torch.ones(1, dtype=torch.bool, device="cuda")
    env._reset_idx(reset_mask)
    obs = unpack_obs(env.get_observations())

    force_magnitude = 0.0
    print(f"\n--- STRESS TEST STARTED ({MODE.upper()} mode) ---")
    print(f"Baseline to beat: 25N (RSL-RL model_100.pt)\n")

    for i in range(20000):
        # Get action from whichever policy is loaded
        current_obs = obs[0] if obs.ndim > 1 else obs
        action = get_action(current_obs)

        # Step physics
        if action.ndim == 1:
            action = action.unsqueeze(0)
        step_output = env.step(action)
        obs   = unpack_obs(step_output[0])
        dones = step_output[2]

        # Ramp up force every 150 steps
        if i % 150 == 0 and i > 150:
            force_magnitude += 5.0
            print(f">>> Applying Side Push: {force_magnitude} N")
            apply_lateral_force(env, force_magnitude)

        if dones[0]:
            print(f"\n{'='*40}")
            print(f"FAILURE AT {force_magnitude} N")
            if force_magnitude > 25.0:
                print("✅ BEAT THE RSL-RL BASELINE (25N)!")
            elif force_magnitude == 25.0:
                print("➖ MATCHED THE RSL-RL BASELINE (25N)")
            else:
                print("❌ BELOW RSL-RL BASELINE (25N)")
            print(f"{'='*40}")
            break

    else:
        print(f"\n✅ SURVIVED ALL 20000 STEPS — last push: {force_magnitude} N")


if __name__ == "__main__":
    main()
