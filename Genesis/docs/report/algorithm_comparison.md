# Algorithm Comparison Study — PPO vs SAC vs TD3

**Task:** Go2 push-hardening — forward walk at 0.5 m/s under randomized lateral
perturbations (5–15 N, every 150 steps, 10-step duration, random direction).
**Matched conditions:** ~5 M environment steps, 64 parallel environments,
identical `go2_env` dynamics, observation space, reward function, and termination.
**Hardware:** single RTX 4070 + Intel Ultra 9 185H.
**Data provenance:** all values measured from `algo_comparison/logs/{ppo,sac,td3}/`
TensorBoard event files (`extract_all_data.py`); wall-clock cross-checked against
`wall_clock.json`. Figures: `docs/comparison/` (real-data-only generator).

> **Caveat — N = 1.** Each algorithm was run once (no seed repeats). Results are
> a strong, internally consistent indication of behavior on this task, **not** a
> statistically significant ranking. Treat magnitudes as descriptive.

---

## 1. Headline Result

> **PPO is the best overall policy on this task: highest final reward (34.3),
> longest final survival (914/1000 steps), most stable, and ~2.3× faster to
> train. SAC is the most sample-efficient — reaching full-episode survival at
> 2.0 M steps — but degrades badly in the second half of training. TD3 is the
> least stable and collapses after ~4.5 M steps.**

This is consistent with the broader continuous-control literature: on-policy PPO
tends to be more robust and lower-variance for locomotion, while off-policy
SAC/TD3 are more sample-efficient but more sensitive to instability without
careful tuning.

---

## 2. Master Comparison Table

| Metric | PPO | SAC | TD3 | Best |
|---|---|---|---|---|
| Environment steps | 5.11 M | 4.99 M | 5.00 M | — |
| **Final reward** (last 10 %) | **34.3** | 18.8 | 6.7 | PPO |
| **Peak reward** | 35.0 | **35.5** | 27.0 | SAC |
| Step of peak reward | 4.72 M | **1.26 M** | 4.49 M | SAC (earliest) |
| **Final episode length** | **914** | 547 | 147 | PPO |
| **Peak episode length** | 942 | **1001** | 785 | SAC |
| Step of peak survival | 4.59 M | **2.00 M** | 4.49 M | SAC (earliest) |
| **Reward retention** (final/peak) | **98 %** | 53 % | 25 % | PPO |
| **Survival retention** (final/peak) | **97 %** | 55 % | 19 % | PPO |
| Wall-clock (≈5 M steps) | **12.6 min** | 35.6 min | 28.3 min | PPO |
| Mean throughput | **6,754 fps** | 2,544 fps | 2,974 fps | PPO |
| Policy class | on-policy | off-policy | off-policy | — |

---

## 3. Dimension-by-Dimension Analysis

### 3.1 Training Stability — **Winner: PPO**
The defining difference between the three runs.
- **PPO** retains 98 % of its peak reward and 97 % of its peak survival at the
  end of training — a near-monotonic curve (minimum reward 2.18, rising steadily).
- **SAC** oscillates heavily: its reward repeatedly swings between ~5 and ~35
  across training, and it ends at 53 % of its peak reward / 55 % of peak survival.
- **TD3** collapses: after peaking at 785-step survival near 4.5 M steps, final
  survival falls to 147 (19 % retention).

See `fig_peak_vs_final.png` — the peak-vs-final gap is the clearest single view
of this result.

### 3.2 Convergence — **Winner: PPO (to a stable optimum)**
PPO reaches its stable plateau and stays there. SAC and TD3 reach high values
*transiently* but do not hold them, so they have no stable convergence point on
this task within 5 M steps.

### 3.3 Sample Efficiency — **Winner: SAC**
SAC is the standout here: peak reward at **1.26 M steps** and full-episode
survival (1001) at **2.0 M steps**, versus PPO reaching its peak near 4.6–4.7 M.
Off-policy replay makes SAC ~2× more sample-efficient *to first reach* high
performance — the problem is retention, not learning speed.

### 3.4 Final Locomotion Quality — **Winner: PPO**
At the end of training, only PPO sustains near-full-episode walking under pushes
(914/1000). SAC (547) and TD3 (147) fall well before the episode horizon,
indicating frequent loss of balance in their final policies.

### 3.5 Velocity Tracking — **Winner: PPO (inferred)**
The reward is dominated by `tracking_lin_vel` (weight 5.0, max ≈ 35). PPO's final
reward of 34.3 implies it tracks the 0.5 m/s command closely for most of the
episode. SAC/TD3's lower final reward reflects both shorter episodes and poorer
tracking. *Note:* tracking error is inferred from reward, not logged directly.

### 3.6 Push Recovery / Robustness — **Not directly measured**
Every run trains *under* pushes, so survival under perturbation is the implicit
robustness signal: PPO 914, SAC 547, TD3 147. **However, no independent
stress-test threshold (failure Newtons) was persisted for any algorithm** — the
ranking here is by in-training survival, not by an isolated robustness test. This
is a documented gap (see `repository_audit.md` §6).

### 3.7 Computational Cost — **Winner: PPO**
PPO trains ~5 M steps in 12.6 min at 6,754 fps. SAC (35.6 min) and TD3 (28.3 min)
are ~2.3–2.8× slower because they run a gradient update at (nearly) every
environment step, while PPO batches 2048-step rollouts. See `fig_training_cost.png`.

### 3.8 Final Performance — **Winner: PPO**
Combining final reward, final survival, stability, and cost, PPO is the
unambiguous best *deployable* policy from this study.

---

## 4. Scorecard

Qualitative ranking per dimension (◆ = best, ○ = weakest), grounded in §3:

| Dimension | PPO | SAC | TD3 |
|---|:---:|:---:|:---:|
| Training stability | ◆ | ◑ | ○ |
| Convergence (to stable optimum) | ◆ | ◑ | ○ |
| Sample efficiency (to first peak) | ◑ | ◆ | ○ |
| Final locomotion quality | ◆ | ◑ | ○ |
| Velocity tracking (inferred) | ◆ | ◑ | ○ |
| Push recovery (in-training survival) | ◆ | ◑ | ○ |
| Computational cost | ◆ | ○ | ◑ |
| Final performance | ◆ | ◑ | ○ |

---

## 5. Why These Results Make Sense

- **PPO's clipped on-policy update** trades sample efficiency for stability — it
  never moves far from the current policy, which suppresses the catastrophic
  policy shifts seen in SAC/TD3 here.
- **SAC's entropy-regularized off-policy learning** explores aggressively and
  learns fast from replay, explaining both its early peak and its late-training
  oscillation/instability on this perturbed task.
- **TD3's deterministic policy with delayed updates** is the most brittle of the
  three under continual perturbation; once its critic degrades, the actor follows
  and survival collapses.

---

## 6. Figures

| File | Shows |
|---|---|
| `docs/comparison/fig_reward_vs_steps.png` | Reward learning curves, all three |
| `docs/comparison/fig_eplen_vs_steps.png` | Episode survival vs steps |
| `docs/comparison/fig_peak_vs_final.png` | Peak-vs-final retention (stability) |
| `docs/comparison/fig_training_cost.png` | Wall-clock + throughput |
| `docs/comparison/fig_dashboard.png` | Combined 4-panel summary |

---

## 7. Recommendation

For a **deployable Go2 push-hardened walking policy**, use **PPO** — it is the
most stable, highest final-performing, and cheapest to train on this task. If
**sample budget is the binding constraint** and a checkpoint-selection strategy
is in place (early-stop at peak), **SAC** is attractive given its 2.0 M-step
full-survival result — but it requires saving and selecting the peak checkpoint,
not the final one. **TD3** is not recommended for this task without substantial
stabilization work.

**To make this study conclusive:** repeat each algorithm with ≥3 seeds, persist
an isolated stress-test threshold per checkpoint, and report mean ± std.
