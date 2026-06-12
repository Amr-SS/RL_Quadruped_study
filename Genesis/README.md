# RL Quadruped Study — Unitree Go2 (project files)

> This folder contains the project code, logs, and results. For the full
> research-style write-up with results and figures, see the **[root README](../README.md)**.

Training a Unitree Go2 to walk and resist lateral push perturbations using
**Stable-Baselines3 (PPO / SAC / TD3)** on the **Genesis** GPU physics engine.

---

## Documentation

| Document | Contents |
|---|---|
| [`../README.md`](../README.md) | Full research-style README (overview → results → comparison) |
| [`docs/report/repository_audit.md`](docs/report/repository_audit.md) | Complete repository audit |
| [`docs/report/experiment_summary.md`](docs/report/experiment_summary.md) | Extracted results for every run |
| [`docs/report/algorithm_comparison.md`](docs/report/algorithm_comparison.md) | Detailed PPO vs SAC vs TD3 study |
| [`docs/report/admissions_highlights.md`](docs/report/admissions_highlights.md) | Skills & relevance summary |
| [`docs/comparison/`](docs/comparison/) | Real-data-only publication figures + generator |

---

## Key Files

| File | Purpose |
|---|---|
| `examples/locomotion/go2_env.py` | Go2 environment — dynamics, observation, reward, termination |
| `examples/locomotion/go2_train.py` | Config source-of-truth + RSL-RL baseline |
| `sb3_train.py` | PPO push-hardening trainer (`GenesisVecEnv` SB3 bridge) |
| `sb3_eval.py` | Load checkpoint → render or record mp4 |
| `go2_stress_test.py` | Ramp lateral force until fall |
| `algo_comparison/` | PPO / SAC / TD3 trainers, logs, plots |
| `extract_all_data.py` | Ground-truth metric extraction → JSON |
| `docs/comparison/make_figures.py` | Real-data-only figure generator (no synthetic fallback) |

---

## Quick Start

```bash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# Train push-hardened PPO (64 envs, 5M steps, ~13 min on RTX 4070)
python sb3_train.py --num-envs 64 --push-min 5.0 --push-max 15.0 \
    --total-steps 5000000 --run-name go2_64env --save-name go2_push_hardened_64env

# Evaluate / stress-test
python sb3_eval.py
python go2_stress_test.py

# Reproduce reported numbers & figures (no training needed)
python extract_all_data.py
python docs/comparison/make_figures.py
```

---

## Headline Result

On the push-hardening task (~5M steps, 64 envs, identical config), **PPO** is the
best overall policy — **914 / 1000**-step survival under active 5–15 N pushes,
**98 %** peak-performance retention, and **~2.3× faster** training than the
off-policy methods. **SAC** is ~2× more sample-efficient but unstable; **TD3**
collapses late. Full data and figures in the [root README](../README.md) and
[`docs/report/`](docs/report/).

*All numbers are extracted directly from the TensorBoard logs in this repo — see
`extract_all_data.py`.*
