# Experiment Summary — Extracted Results

**Source of every number:** TensorBoard event files parsed by
`extract_all_data.py` → `docs/report/extracted_metrics.json`.
Wall-clock cross-checked against `algo_comparison/logs/wall_clock.json`.
**Nothing below is estimated or synthetic.**

Conventions:
- *Final* = mean of the last 10 % of logged points (robust to single-point noise).
- *Peak* = maximum logged value, with the step at which it occurred.
- *Max episode length* = 1000 control steps (20 s @ 50 Hz).

---

## 1. Complete Run Inventory

| Run | Group | Steps | Final reward | Peak reward | Final ep-len | Peak ep-len | Wall-clock | Status |
|---|---|---|---|---|---|---|---|---|
| `ppo_comparison_1` | algo/PPO | 5.11 M | 34.3 | 35.0 | 914 | 942 | 12.6 min | ✅ complete |
| `sac_push_run_1` | algo/SAC | 4.99 M | 18.8 | 35.5 | 547 | 1001 | 35.6 min | ✅ complete |
| `td3_push_run_1` | algo/TD3 | 5.00 M | 6.7 | 27.0 | 147 | 785 | 28.3 min | ✅ complete |
| `Train_4_walk_long_1` | walk | 15.7 M | 33.7 | 33.7 | 933 | 933 | 17.7 min | ✅ best walk |
| `v2_walk_512env_1` | walk | 15.7 M | 33.7 | 33.7 | 921 | 921 | 57.8 min | ✅ complete |
| `v2_walk2_fixed_1` | walk | 0.52 M | 21.2 | 21.2 | 601 | 605 | 0.5 min | ✅ short |
| `Train_1_2` | push (old scale†) | 5.11 M | 6.9 | 6.9 | 955 | 981 | 13.1 min | ✅ complete |
| `Train_2_1` | push (old scale†) | 5.11 M | 6.8 | 6.9 | 959 | 978 | 12.8 min | ✅ complete |
| `Train_3_1` | push (old scale†) | 5.11 M | 6.7 | 6.8 | 958 | 997 | 12.2 min | ✅ complete |
| `SB3_v2_test_1` | test | — | 0.3 | 0.3 | 122 | 128 | — | ⚠ 5 points |
| `Train_1_1` | push | — | 0.0 | 0.0 | 0 | 0 | — | ⚠ degenerate |
| `Train_4_1` | walk | — | −0.1 | −0.1 | 64 | 64 | — | ⚠ 1 point |
| `go2_64env_2` | push | — | — | — | — | — | — | ⚠ no rollout tags |
| `Train_4_walk_1` | walk | — | — | — | — | — | — | ❌ empty |
| `go2_64env_1` | push | — | — | — | — | — | — | ❌ empty |
| `v2_walk_512env_2` | walk | — | — | — | — | — | — | ❌ empty |

† Old-scale runs used `tracking_lin_vel = 1.0` (max reward ≈ 7). Reward is not
comparable to tuned-scale runs; **episode length is the cross-run metric.**

---

## 2. Best Runs by Objective

**Best forward-walk policy — `Train_4_walk_long_1`**
- 15.7 M steps, final reward **33.7**, episode length **933 / 1000** (93 % survival).
- Trained in **17.7 min** at a mean **19,334 env-steps/s** (peak 68,865 fps).
- Interpretation: the robot sustains a commanded 0.5 m/s forward walk for
  essentially the full episode.

**Best push-hardening policy (tuned scale) — `ppo_comparison_1`**
- 5.11 M steps under 5–15 N lateral pushes, final reward **34.3**,
  episode length **914 / 1000** (91 % survival).
- Maintains near-full survival *despite* periodic lateral perturbations — the
  strongest in-repo evidence that push-hardening trained successfully.

**Most sample-efficient (but unstable) — `sac_push_run_1`**
- Reached peak reward **35.5 at 1.26 M steps** and **full-episode survival
  (1001) at 2.0 M steps** — roughly **2× faster to peak than PPO**.
- Then degraded: final survival fell to 547 (see Algorithm Comparison).

---

## 3. Locomotion-Quality Indicators (what the numbers imply)

The environment does not log a direct "gait quality" scalar, so quality is
inferred from measured proxies:

| Indicator | Source metric | Reading |
|---|---|---|
| Stays upright | episode length → 914–933 / 1000 | Rarely triggers the 10° roll/pitch termination |
| Tracks commanded speed | `tracking_lin_vel` dominates reward; reward → ~34 of ~35 max | Closely follows the 0.5 m/s command |
| Smooth control | `action_rate` penalty active throughout | No reward collapse from jitter in the best runs |
| Recovers from pushes | survival ~914 under active 5–15 N pushes | Balance recovery learned |

These are **proxy** indicators. Direct gait metrics (foot-contact schedule,
center-of-mass deviation, cost of transport) are **not measured** — listed as a
gap in the audit.

---

## 4. Throughput / Hardware Findings

| Run type | Mean fps | Note |
|---|---|---|
| Walk, 512 envs (`Train_4_walk_long_1`) | 19,334 | Peak 68,865 fps |
| PPO push, 64 envs | 6,754 | On-policy, batched |
| SAC push, 64 envs | 2,544 | Off-policy, per-step updates |
| TD3 push, 64 envs | 2,974 | Off-policy |

On-policy PPO runs ~2.3× faster in wall-clock than the off-policy methods at
equal step budgets, because SAC/TD3 perform a gradient update at (nearly) every
environment step while PPO updates once per 2048-step rollout.

---

## 5. Reproducing This Summary

```bash
# From Genesis/ with the project venv active:
python extract_all_data.py
# → prints headline metrics, writes docs/report/extracted_metrics.json

python docs/comparison/make_figures.py
# → regenerates all figures in docs/comparison/ from the same event files
```

Both scripts read only measured TensorBoard logs and will error (not fabricate)
if a log is missing.
