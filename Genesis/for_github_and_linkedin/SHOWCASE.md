# RL Quadruped Study — Results Showcase

Unitree Go2 trained to walk and resist lateral push perturbations using **Genesis physics engine** + **Stable-Baselines3 PPO/SAC/TD3** on an RTX 4070.

---

## Project At-a-Glance

| | |
|---|---|
| **Robot** | Unitree Go2 (12 DoF quadruped) |
| **Simulator** | Genesis (GPU-accelerated) |
| **Hardware** | RTX 4070 Laptop GPU (8 GB), Intel Core Ultra 9 185H |
| **Parallel envs** | 64–512 robots on a single GPU |
| **Steps trained** | 5M–15M per run |
| **Algorithms tested** | PPO, SAC, TD3 |
| **Push hardening** | 5–15 N lateral forces during training |

---

## Training Progression

### Reward Over Training Runs

![Reward Comparison](training_graphs/comparison_ep_rew.png)

### Episode Length (Survival Time)

![Ep Len Comparison](training_graphs/comparison_ep_len.png)

### Push-Hardened 64-Env Run (5M steps)

![64env Push Metrics](training_graphs/go2_64env_2_metrics.png)

### Walk Training V2 — 512 Parallel Envs

![v2 Walk 512env](training_graphs/v2_walk_512env_1_metrics.png)

### Walk Training — Extended Run

![Walk Long](training_graphs/Train_4_walk_long_1_metrics.png)

---

## PPO vs SAC vs TD3 — Algorithm Comparison (5M steps each)

All three algorithms trained on the same push-hardened task (5–15 N lateral forces, 64 parallel envs).

### Combined Dashboard

![Algo Comparison Dashboard](algo_comparison/combined_dashboard.png)

### Reward vs Steps

![Reward vs Steps](algo_comparison/reward_vs_steps.png)

### Reward vs Wall-Clock Time

![Reward vs Time](algo_comparison/reward_vs_time.png)

| Algorithm | Time (~5M steps) | Final reward | Final survival | Notes |
|---|---|---|---|---|
| **PPO** | **12.6 min** | **34.3** | **914 / 1000** | Most stable; retains 98% of peak |
| TD3 | 28.3 min | 6.7 | 147 / 1000 | Peaks at 4.5M then **collapses** |
| SAC | 35.6 min | 18.8 | 547 / 1000 | Fastest to peak (2M steps) but **degrades** |

*All values measured from TensorBoard logs — see [`../docs/report/algorithm_comparison.md`](../docs/report/algorithm_comparison.md).*

### Convergence Speed

![Convergence Bar](algo_comparison/convergence_bar.png)

### Policy Stability (Std Dev)

![Stability](algo_comparison/stability_std.png)

### Episode Length vs Steps

![Ep Len vs Steps](algo_comparison/ep_len_vs_steps.png)

### Training Time Comparison

![Time Bar](algo_comparison/time_bar.png)

---

## Stress Test

The `go2_stress_test.py` script ramps lateral force every 150 steps to find the model's failure threshold.

**Baseline:** RSL-RL `model_100.pt` fails at **25 N**

Run with:
```bash
python go2_stress_test.py  # MODE = "sb3" at top of file
```

---

## How to Reproduce

```bash
# Train push-hardened model (64 envs, 5M steps, RTX 4070 ~13 min)
python sb3_train.py \
    --num-envs 64 \
    --push-min 5.0 \
    --push-max 15.0 \
    --total-steps 5000000 \
    --run-name "go2_64env" \
    --save-name "go2_push_hardened_64env"

# Evaluate — opens Genesis viewer or records mp4
python sb3_eval.py

# Stress test
python go2_stress_test.py

# Run PPO vs SAC vs TD3 comparison
cd algo_comparison && python train_all.py
```

---

## Key Findings (measured)

1. **PPO is the best overall policy** — highest final reward (34.3), longest survival
   (914/1000 steps under active pushes), and fastest to train (12.6 min, ~6.7k fps).
2. **PPO is the most stable** — it retains **98%** of its peak performance to the end of
   training, versus 53% (SAC) and 25% (TD3).
3. **SAC is the most sample-efficient but fragile** — reaches full-episode survival
   (1001 steps) at just 2.0M steps (½ of PPO's budget), then **degrades**.
4. **TD3 is the least stable** — peaks near 4.5M steps then **collapses** to 147-step survival.
5. **Push hardening works** — the trained policy maintains balance under continual 5–15 N
   lateral impulses (qualitative; no persisted stress-test metric — see audit).
6. **Reward shaping was decisive** — raising `tracking_lin_vel` (1.0 → 5.0) and softening
   `base_height` (−50 → −5) unlocked walking.

> ⚠️ An earlier version of this file claimed "TD3 is the best balance" and "SAC produces
> the smoothest gait." Re-extracting the raw TensorBoard logs showed the opposite — both
> off-policy methods degrade while PPO stays stable. The numbers above are the corrected,
> measured findings. Full analysis: [`../docs/report/algorithm_comparison.md`](../docs/report/algorithm_comparison.md).

---

*All training done locally on a single RTX 4070 GPU (no cloud compute). Every number is
traceable to a TensorBoard event file via `../extract_all_data.py`.*
