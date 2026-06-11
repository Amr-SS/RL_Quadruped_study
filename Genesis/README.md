# RL Quadruped Study — Unitree Go2 Push Hardening

Training a **Unitree Go2** quadruped to walk and resist lateral push perturbations using **SB3 PPO/SAC/TD3** + **Genesis physics engine** — all on a single RTX 4070.

---

## What This Is

A complete RL locomotion study on the Go2 robot that covers:
- **Baseline PPO** walking policy (Genesis native RSL-RL)
- **Push-hardened PPO** — robot trained to survive 5–15 N lateral impulses
- **64-env and 512-env** parallel training on GPU
- **PPO vs SAC vs TD3** head-to-head comparison on the push task
- **Stress test** script that ramps force until the robot falls

---

## Tech Stack

| Component | Details |
|---|---|
| Physics | [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU-accelerated rigid-body sim |
| RL | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO / SAC / TD3 |
| Robot | Unitree Go2 — 12 DoF, URDF loaded in Genesis |
| Hardware | RTX 4070, Intel Ultra 9 185H |
| Parallel envs | 64–512 robots simulated simultaneously on GPU |

---

## Results

### Training Progression

![Reward Comparison](for_github_and_linkedin/training_graphs/comparison_ep_rew.png)

![Ep Len Comparison](for_github_and_linkedin/training_graphs/comparison_ep_len.png)

### Push-Hardened 64-Env Run

![Push Hardened Metrics](for_github_and_linkedin/training_graphs/go2_64env_2_metrics.png)

### Walk Policy — 512 Parallel Envs (Best Run)

![512env Walk Metrics](for_github_and_linkedin/training_graphs/v2_walk_512env_1_metrics.png)

---

## Algorithm Comparison — PPO vs SAC vs TD3

Same task: 5–15 N lateral push, 64 parallel envs, 5M training steps each.

### Combined Dashboard

![Algo Comparison](for_github_and_linkedin/algo_comparison/combined_dashboard.png)

### Reward vs Wall-Clock Time

![Reward vs Time](for_github_and_linkedin/algo_comparison/reward_vs_time.png)

| Algorithm | Training Time (5M steps) | Winner |
|---|---|---|
| PPO | ~20 min | Fastest |
| TD3 | 28.3 min | Best balance |
| SAC | 35.6 min | Smoothest gait |

### Convergence & Stability

| | |
|---|---|
| ![Convergence](for_github_and_linkedin/algo_comparison/convergence_bar.png) | ![Stability](for_github_and_linkedin/algo_comparison/stability_std.png) |

---

## Stress Test Results

`go2_stress_test.py` ramps lateral force every 150 steps until the robot falls.

| Model | Failure Threshold |
|---|---|
| RSL-RL `model_100.pt` (baseline) | 25 N |
| SB3 PPO push-hardened (this repo) | Run `go2_stress_test.py` to measure |

---

## Files

| File | Purpose |
|---|---|
| `sb3_train.py` | Main training script — PPO with push perturbations, configurable envs/force |
| `sb3_eval.py` | Load a checkpoint and render/record it in Genesis viewer |
| `sb3_train_v2.py` | V2 walk training — higher env count, tuned reward shaping |
| `go2_stress_test.py` | Ramp lateral force until robot falls, report failure Newton value |
| `algo_comparison/` | PPO vs SAC vs TD3 comparison scripts + plots |
| `examples/locomotion/go2_env.py` | Go2 gym environment — rewards, observations, push mechanics |
| `examples/locomotion/go2_train.py` | Original Genesis RSL-RL baseline config |
| `for_github_and_linkedin/` | Key plots and results for showcase |

---

## Setup

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .
pip install stable-baselines3 gymnasium torch

# Clone this repo into your Genesis folder
git clone https://github.com/Amr-SS/RL_Quadruped_study.git .
```

---

## Train

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Push-hardened PPO (64 envs, ~15-20 min on RTX 4070)
python sb3_train.py \
    --num-envs 64 \
    --push-min 5.0 \
    --push-max 15.0 \
    --total-steps 5000000 \
    --run-name "go2_64env" \
    --save-name "go2_push_hardened_64env"
```

Checkpoints saved every 500K steps to `./checkpoints/`

---

## Evaluate

```bash
python sb3_eval.py
# Edit MODEL_PATH at top of file to point to a checkpoint
# Set RECORD_VIDEO = True to save mp4
```

---

## Stress Test

```bash
python go2_stress_test.py
# Edit MODE = "sb3" or "rsl" at top of file
# Baseline to beat: 25N (RSL-RL model_100.pt)
```

---

## Run Algo Comparison

```bash
cd algo_comparison
python train_all.py          # trains PPO, SAC, TD3 sequentially
python generate_plots.py     # generates comparison plots
```

---

## Monitor Training

```bash
tensorboard --logdir ./push_training_logs/
# open http://localhost:6006
```

Key metrics:
- `ep_len_mean` — survival time (target: 1000+)
- `ep_rew_mean` — reward accumulation (target: 2.0+)
- `explained_variance` — value function quality (target: 0.7+)

---

## Key Findings

1. **PPO trains fastest** on this task — 64 parallel GPU envs gives ~300k steps/min on RTX 4070
2. **Push hardening works** — robot learns active balance recovery against lateral impulses
3. **Reward shaping is critical** — reducing `base_height` penalty from -50 → -5 unlocked proper walking
4. **512 parallel envs** dramatically improves sample efficiency vs 64 envs
5. **SAC produces the smoothest gait** but is ~75% slower to train than PPO
6. **TD3 is the best balance** — faster than SAC, more stable reward curve than PPO

---

*All training on a single RTX 4070 — no cloud compute.*
