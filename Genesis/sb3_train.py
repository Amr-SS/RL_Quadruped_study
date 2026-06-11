import sys
import os
import argparse
import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ── CPU thread config ──────────────────────────────────────────────────────────
# Intel Ultra 9 185H: 16 P-cores handle float math, skip E-cores
try:
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass  # already set by caller

sys.path.append(os.path.abspath("examples/locomotion"))
from go2_env import Go2Env
from go2_train import get_cfgs


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Go2 Push-Hardening Training with SB3 + Genesis")

    # Environment
    parser.add_argument("--num-envs",      type=int,   default=64,
                        help="Number of parallel robots on GPU (start 64, drop if OOM)")

    # Push perturbation
    parser.add_argument("--push-min",      type=float, default=5.0,
                        help="Minimum push force in Newtons")
    parser.add_argument("--push-max",      type=float, default=15.0,
                        help="Maximum push force in Newtons (keep below failure threshold)")
    parser.add_argument("--push-interval", type=int,   default=150,
                        help="Steps between pushes per env")
    parser.add_argument("--push-duration", type=int,   default=10,
                        help="How many steps each push is held")

    # Training
    parser.add_argument("--total-steps",   type=int,   default=5_000_000,
                        help="Total environment steps to train for")
    parser.add_argument("--lr",            type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--n-steps",       type=int,   default=2048,
                        help="Rollout steps per env before each PPO update")
    parser.add_argument("--batch-size",    type=int,   default=512,
                        help="Minibatch size for PPO gradient steps")
    parser.add_argument("--n-epochs",      type=int,   default=10,
                        help="PPO epochs per update")
    parser.add_argument("--clip-range",    type=float, default=0.1,
                        help="PPO clip range (tighter than default 0.2)")
    parser.add_argument("--ent-coef",      type=float, default=0.01,
                        help="Entropy bonus coefficient")

    # Logging / saving
    parser.add_argument("--run-name",        type=str, default="ppo_push_run",
                        help="TensorBoard run name")
    parser.add_argument("--save-name",       type=str, default="go2_sb3_push_hardened",
                        help="Final model save name (no extension)")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000,
                        help="Save a checkpoint every N total env-steps")
    parser.add_argument("--tensorboard-log", type=str, default="./push_training_logs/",
                        help="TensorBoard log directory")

    return parser.parse_args()


# ── Helper: safely unpack get_observations() ──────────────────────────────────
def unpack_obs(obs):
    """
    Go2Env.get_observations() returns a tuple (obs_tensor, ...) or just a tensor.
    Always returns a plain (num_envs, 45) float32 CPU numpy array.
    """
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    return obs.cpu().numpy().astype(np.float32)


# ── Vectorized Genesis environment ─────────────────────────────────────────────
class GenesisVecEnv(VecEnv):
    """
    Wraps a Genesis Go2Env as an SB3-compatible VecEnv.
    Runs num_envs robots in parallel on the GPU.
    Applies randomized lateral push perturbations per env.
    """

    def __init__(self, genesis_env, num_envs,
                 push_interval=150,
                 push_force_min=5.0,
                 push_force_max=15.0,
                 push_duration=10):

        self.env      = genesis_env
        self.num_envs = num_envs

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        super().__init__(num_envs, observation_space, action_space)

        # Push config
        self.push_interval  = push_interval
        self.push_force_min = push_force_min
        self.push_force_max = push_force_max
        self.push_duration  = push_duration

        # Per-env push state vectors
        self._step_counts          = np.zeros(num_envs, dtype=np.int32)
        self._push_steps_remaining = np.zeros(num_envs, dtype=np.int32)
        self._current_push_forces  = np.zeros(num_envs, dtype=np.float32)

        # Cache solver references — computed once, reused every step
        self._base_link_idx = self.env.robot.links[0].idx
        self._solver        = self.env.robot._solver

        # Episode tracking for TensorBoard logging
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Pending actions buffer (set in step_async, used in step_wait)
        self._pending_actions = None

        print(f"\n[GenesisVecEnv] Initialized")
        print(f"  robots (num_envs)   : {num_envs}")
        print(f"  base link global idx: {self._base_link_idx}")
        print(f"  push force range    : {push_force_min}–{push_force_max} N")
        print(f"  push interval       : every {push_interval} steps")
        print(f"  push duration       : {push_duration} steps per push\n")

        # Full reset on startup
        self._last_obs = self._do_reset(np.arange(num_envs))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _do_reset(self, env_indices):
        """Reset a subset of environments by their index array."""
        if len(env_indices) == 0:
            return self._last_obs

        # Build boolean mask for Genesis
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda")
        mask[env_indices] = True
        self.env._reset_idx(mask)

        # Clear push state for these envs
        self._step_counts[env_indices]          = 0
        self._push_steps_remaining[env_indices] = 0
        self._current_push_forces[env_indices]  = 0.0
        self._episode_rewards[env_indices]      = 0.0
        self._episode_lengths[env_indices]      = 0

        # FIX: get_observations() returns a tuple — unpack safely
        obs = self.env.get_observations()
        return unpack_obs(obs)

    def _apply_pushes(self):
        """
        Build per-env force tensor and submit to Genesis solver.
        Forces are applied on the Y axis (lateral push).
        Direction is randomized per push so the robot learns both sides.
        """
        # Shape: (num_envs, 1, 3)  —  1 link, 3 force components (x, y, z)
        force = torch.zeros(self.num_envs, 1, 3, device="cuda", dtype=torch.float32)

        for i in range(self.num_envs):

            if self._push_steps_remaining[i] > 0:
                # Continue an active push this step
                # FIX: cast numpy.float32 → Python float before assigning to CUDA tensor
                force[i, 0, 1]                = float(self._current_push_forces[i])
                self._push_steps_remaining[i] -= 1

            elif (self._step_counts[i] > 0 and
                  self._step_counts[i] % self.push_interval == 0):
                # Schedule a new push for this env
                magnitude = np.random.uniform(self.push_force_min, self.push_force_max)
                direction = np.random.choice([-1.0, 1.0])   # random left or right
                self._current_push_forces[i]  = magnitude * direction
                # FIX: cast numpy.float32 → Python float before assigning to CUDA tensor
                force[i, 0, 1]                = float(self._current_push_forces[i])
                self._push_steps_remaining[i] = self.push_duration - 1

        # Skip the solver call entirely if no env has an active push this step
        if force.abs().sum().item() == 0.0:
            return

        try:
            # High-level API — available in Genesis post Dec-20-2024 / 0.4.1+
            self._solver.apply_links_external_force(
                force=force,
                links_idx=[self._base_link_idx],
                envs_idx=None,
            )
        except AttributeError:
            # Fallback: direct Taichi buffer write for older Genesis builds
            # Note: uses -= to match Genesis internal sign convention
            for i in range(self.num_envs):
                val = force[i, 0, 1].item()
                if val != 0.0:
                    self._solver.links_state[
                        self._base_link_idx, i
                    ].cfrc_ext_vel[1] -= val

    # ── SB3 VecEnv interface ───────────────────────────────────────────────────

    def reset(self):
        """Full reset of all environments."""
        obs = self._do_reset(np.arange(self.num_envs))
        self._last_obs = obs
        return obs

    def step_async(self, actions):
        """
        SB3 calls this first with the policy's actions.
        We just store them — physics runs in step_wait.
        """
        self._pending_actions = actions

    def step_wait(self):
        """
        SB3 calls this to collect results.
        Order: push → physics → pull results → auto-reset done envs.
        """
        # 1. Apply push forces BEFORE physics so solver sees them this tick
        self._apply_pushes()

        # 2. Push actions to GPU and step all robots simultaneously
        action_tensor = torch.tensor(
            self._pending_actions, device="cuda", dtype=torch.float32
        )
        obs, rewards, dones, infos = self.env.step(action_tensor)

        # 3. Pull everything back to CPU for SB3
        # FIX: unpack obs tuple safely before .cpu()
        obs_cpu     = unpack_obs(obs)
        rewards_cpu = rewards.cpu().numpy().astype(np.float32)
        dones_cpu   = dones.cpu().numpy().astype(bool)

        # 4. Increment step counters and episode tracking
        self._step_counts += 1
        self._episode_rewards += rewards_cpu
        self._episode_lengths += 1

        # 5. Genesis auto-resets internally — we just track episodes here
        done_indices = np.where(dones_cpu)[0]

        info_list = [{} for _ in range(self.num_envs)]
        for i in done_indices:
            info_list[i]["episode"] = {
                "r": float(self._episode_rewards[i]),
                "l": int(self._episode_lengths[i]),
            }
            # Reset episode tracking for completed episodes
            self._episode_rewards[i] = 0.0
            self._episode_lengths[i] = 0
            self._push_steps_remaining[i] = 0
            self._current_push_forces[i] = 0.0

        self._last_obs = obs_cpu

        return obs_cpu, rewards_cpu, dones_cpu, info_list

    # ── Required VecEnv abstract methods (SB3 boilerplate) ────────────────────

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        pass


# ── Reward Logger Callback ─────────────────────────────────────────────────────
class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward and length to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_rewards = []
        self._ep_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])
        if len(self._ep_rewards) >= 1:
            self.logger.record("rollout/ep_rew_mean", np.mean(self._ep_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(self._ep_lengths))
            self._ep_rewards = []
            self._ep_lengths = []
        return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("  Go2 Push-Hardening Training")
    print(f"  num_envs      : {args.num_envs}")
    print(f"  push force    : {args.push_min}–{args.push_max} N")
    print(f"  push interval : every {args.push_interval} steps")
    print(f"  push duration : {args.push_duration} steps")
    print(f"  total steps   : {args.total_steps:,}")
    print(f"  lr            : {args.lr}")
    print(f"  n_steps×envs  : {args.n_steps} × {args.num_envs} = "
          f"{args.n_steps * args.num_envs:,} samples/update")
    print(f"  run name      : {args.run_name}")
    print(f"  save name     : {args.save_name}")
    print("=" * 60)

    # Boot Genesis physics engine on RTX 4070
    gs.init(backend=gs.gpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    raw_env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,      # no viewer during training — saves VRAM and FPS
    )

    env = GenesisVecEnv(
        raw_env,
        num_envs=args.num_envs,
        push_interval=args.push_interval,
        push_force_min=args.push_min,
        push_force_max=args.push_max,
        push_duration=args.push_duration,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",                   # PPO on CPU, Genesis on GPU — no conflict
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,                    # value function loss weight
        max_grad_norm=0.5,              # gradient clipping
        gamma=0.99,                     # discount factor
        gae_lambda=0.95,                # GAE smoothing
        tensorboard_log=args.tensorboard_log,
    )

    os.makedirs("./checkpoints", exist_ok=True)

    from stable_baselines3.common.callbacks import CallbackList
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path="./checkpoints/",
        name_prefix=args.save_name,
        verbose=1,
    )
    reward_cb = RewardLoggerCallback()
    callbacks = CallbackList([checkpoint_cb, reward_cb])

    print("\nStarting training...\n")
    print("Watch these metrics climb over time:")
    print("  ep_len_mean  → robot surviving longer   (target: 1000+)")
    print("  ep_rew_mean  → reward accumulating      (target: 2.0+)")
    print("  entropy_loss → policy getting confident (should decrease from -17)\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    model.save(args.save_name)
    print(f"\nTraining complete!")
    print(f"Final model saved to : {args.save_name}.zip")
    print(f"Checkpoints saved in : ./checkpoints/")
    print(f"\nNext step — run the stress test against your best checkpoint:")
    print(f"  python go2_stress_test.py")
    print(f"  (edit ckpt_path to point at a checkpoint .zip)")


if __name__ == "__main__":
    main()
