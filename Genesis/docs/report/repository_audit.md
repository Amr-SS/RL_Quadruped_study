# Repository Audit — RL Quadruped Locomotion Study (Unitree Go2)

**Audit date:** 2026-06-12
**Auditor scope:** full repository — code, logs, checkpoints, media, documentation
**Method:** every quantitative claim below is extracted directly from TensorBoard
event files via `extract_all_data.py` (output: `docs/report/extracted_metrics.json`).
No values are estimated or synthesised.

---

## 1. Project Summary

A reinforcement-learning study of quadruped locomotion for the Unitree Go2,
built on the **Genesis** GPU physics engine with **Stable-Baselines3** policies.
The project covers two coupled problems:

1. **Forward locomotion** — learning a stable forward walk from a 45-dimensional
   proprioceptive observation to 12 joint-position targets.
2. **Push-hardening** — training the policy under randomized lateral force
   perturbations (5–15 N) so it learns active balance recovery.

On top of these, a **PPO vs SAC vs TD3** algorithm comparison was run on the
push-hardening task under matched conditions (~5M steps, 64 parallel
environments, identical environment and reward configuration).

---

## 2. Repository Structure

```
Genesis/
├── examples/locomotion/
│   ├── go2_env.py             # Go2 environment: dynamics, observation, reward, termination
│   └── go2_train.py           # Config source-of-truth (env/obs/reward/command cfgs) + RSL-RL baseline
├── sb3_train.py               # PPO push-hardening trainer (SB3 VecEnv wrapper around Genesis)
├── sb3_train_v2.py            # V2 trainer variant (tuned reward shaping / higher env count)
├── sb3_eval.py                # Load checkpoint, render or record mp4
├── go2_stress_test.py         # Ramp lateral force until fall; prints failure threshold
├── plot_training_graphs.py    # Per-run + comparison plots from push_training_logs/
├── extract_all_data.py        # [added] Ground-truth metric extraction → JSON
├── algo_comparison/
│   ├── train_all.py           # Orchestrates PPO/SAC/TD3 runs
│   ├── sac_train.py           # SAC trainer
│   ├── td3_train.py           # TD3 trainer
│   ├── generate_plots.py      # Comparison plots (⚠ has synthetic fallback — see §7)
│   ├── logs/{ppo,sac,td3}/     # TensorBoard event files (real)
│   │   └── wall_clock.json     # Recorded training time: TD3 28.3 min, SAC 35.6 min
│   ├── models/{sac,td3}/       # Checkpoints (~112 MB, git-ignored)
│   └── plots/                  # 7 generated comparison PNGs
├── checkpoints/               # PPO / walk checkpoints (~166 MB, git-ignored)
├── push_training_logs/        # 14 TensorBoard runs (PPO push + walk experiments)
├── training_graphs/           # 13 per-run metric PNGs + 2 comparison PNGs
├── docs/
│   ├── report/                # [added] Audit, experiment summary, algo comparison, admissions
│   └── comparison/            # [added] Real-data-only publication figures + generator
├── *.mp4                      # 6 evaluation videos (git-ignored by pattern)
└── for_github_and_linkedin/   # Curated showcase subset
```

---

## 3. Training Pipeline

**Environment (`go2_env.py`)**
- Control frequency 50 Hz (`dt = 0.02`), 2 physics substeps.
- Episode length 20 s = **1000 control steps** (`episode_length_s / dt`).
- PD joint control: `kp = 20.0`, `kd = 0.5`, `action_scale = 0.25`.
- Termination on roll > 10° or pitch > 10°.

**Observation space — 45 dimensions**
| Block | Dim | Scaling |
|---|---|---|
| Base angular velocity | 3 | 0.25 |
| Projected gravity | 3 | — |
| Velocity command (x, y, yaw) | 3 | [2.0, 2.0, 0.25] |
| Joint positions (− default) | 12 | 1.0 |
| Joint velocities | 12 | 0.05 |
| Previous action | 12 | — |

**Action space — 12 dimensions:** per-joint position offsets, scaled by 0.25 and
added to the default stance, tracked by the PD controller.

**Reward terms (`get_cfgs`)**
| Term | Weight | Purpose |
|---|---|---|
| `tracking_lin_vel` | +5.0 | Follow commanded forward velocity |
| `tracking_ang_vel` | +0.2 | Follow commanded yaw rate |
| `lin_vel_z` | −1.0 | Penalize vertical bounce |
| `base_height` | −5.0 | Maintain target body height |
| `action_rate` | −0.005 | Penalize jerky actions |
| `similar_to_default` | −0.1 | Stay near nominal stance |

**Push perturbation (`sb3_train.py::GenesisVecEnv`)**
- Lateral (Y-axis) force, magnitude sampled U(5, 15) N, random ± direction.
- Applied every 150 steps, held for 10 steps, per environment independently.

**PPO hyperparameters:** `n_steps=2048`, `batch_size=512`, `n_epochs=10`,
`lr=3e-4`, `clip_range=0.1`, `ent_coef=0.01`, `gamma=0.99`, `gae_lambda=0.95`,
`MlpPolicy`. Policy optimization on CPU; Genesis simulation on GPU.

---

## 4. Evaluation Pipeline

- **`sb3_eval.py`** — loads a checkpoint, runs the policy in the Genesis viewer
  or records an mp4. Six evaluation videos exist (`*_eval.mp4`).
- **`go2_stress_test.py`** — ramps lateral force by 5 N every 150 steps and
  prints the Newton value at which the robot falls. **Important:** results are
  printed to stdout only; **no stress-test result is persisted anywhere in the
  repository** (see §6).

---

## 5. Available Experimental Data (measured)

### 5.1 Algorithm comparison — push-hardening task (the primary study)

All three trained ~5M steps, 64 envs, identical env/reward config.

| Metric | PPO | SAC | TD3 |
|---|---|---|---|
| Steps trained | 5.11 M | 4.99 M | 5.00 M |
| Wall-clock (event-file span) | 12.6 min | 35.3 min | 28.1 min |
| Wall-clock (`wall_clock.json`) | not logged | 35.6 min | 28.3 min |
| Mean throughput | 6,754 fps | 2,544 fps | 2,974 fps |
| Final reward (last 10 %) | **34.3** | 18.8 | 6.7 |
| Peak reward | 35.0 @ 4.72 M | 35.5 @ 1.26 M | 27.0 @ 4.49 M |
| Final episode length | **914** | 547 | 147 |
| Peak episode length | 942 @ 4.59 M | 1001 @ 2.00 M | 785 @ 4.49 M |
| Reward retention (final / peak) | 98 % | 53 % | 25 % |
| Logged data points | 39 | 2,976 | 26,357 |

(`logged data points` differ because PPO logs once per rollout update while the
off-policy methods log per episode/gradient step — this does not affect the
underlying step counts.)

### 5.2 Forward-walk experiments (`push_training_logs/`)

| Run | Steps | Final reward | Final ep-len | Wall-clock | Notes |
|---|---|---|---|---|---|
| `Train_4_walk_long_1` | 15.7 M | 33.7 | 933 | 17.7 min | Best walk policy |
| `v2_walk_512env_1` | 15.7 M | 33.7 | 921 | 57.8 min | 512 envs, render overhead |
| `v2_walk2_fixed_1` | 0.52 M | 21.2 | 605 | 0.5 min | Short config-fix run |
| `Train_1_2` | 5.11 M | 6.9 † | 955 | 13.1 min | Early push run |
| `Train_2_1` | 5.11 M | 6.8 † | 959 | 12.8 min | Early push run |
| `Train_3_1` | 5.11 M | 6.8 † | 958 | 12.2 min | Early push run |

† The early push runs used the **original reward scaling** (`tracking_lin_vel = 1.0`,
max reward ≈ 7), whereas the algorithm-comparison and walk runs used the **tuned
scaling** (`tracking_lin_vel = 5.0`, max reward ≈ 35). **Raw reward is therefore
not comparable across reward configurations** — episode length (capped at 1000)
is the scale-invariant performance metric and shows these early runs already
reached near-full survival (~955/1000).

---

## 6. Missing / Incomplete Data (explicitly documented)

| Gap | Evidence | Impact |
|---|---|---|
| **No persisted push-recovery metric** | `go2_stress_test.py` prints to stdout, writes no file; no stress-test artifact in repo | The "25 N baseline" and any model's failure threshold are **unmeasured in-repo**; push robustness is shown only qualitatively (video). |
| **No evaluation metrics** | Only `*_eval.mp4` videos; no CSV of tracking error, mean velocity, or success rate | Final locomotion quality is not quantified beyond training-time reward/ep-len. |
| **Single run per algorithm (N = 1)** | One event file each for PPO/SAC/TD3 | No seeds, confidence intervals, or significance tests — comparison is indicative, not statistically conclusive. |
| **`monitor.csv` absent** | `extra_plots.py` reads `logs/*/monitor.csv` which do not exist | `extra_plots.py` is non-functional as written. |
| **RSL-RL baseline absent** | `go2_stress_test.py` references `logs/go2-walking/model_100.pt`; not present | Baseline comparison cannot be reproduced from this repo alone. |
| **Empty/partial runs** | `Train_4_walk_1`, `go2_64env_1`, `v2_walk_512env_2` have no scalars; `go2_64env_2` logs only `train/*`; `Train_4_1` has 1 point | These runs carry no usable learning curve. |

---

## 7. Data-Integrity Note (important)

`algo_comparison/generate_plots.py` contains a `generate_synthetic_curves()`
fallback with hard-coded reference statistics (PPO 5M/19 min, TD3 9M/34 min,
SAC 14M/53 min). If a real log fails to load, it **silently substitutes
fabricated curves**. The hard-coded step counts (5M/9M/14M) do **not** match the
measured reality (all three ran ~5M steps). The real logs do load, so the
existing plots are believed to use real data — but the presence of a silent
synthetic fallback is a scientific-integrity risk.

**Mitigation (added in this audit):** `docs/comparison/make_figures.py`
regenerates every comparison figure from measured event files with **no
synthetic fallback** — it raises an error if any log or tag is missing. The
figures in `docs/comparison/` are therefore guaranteed real.

---

## 8. Technical Strengths

- **GPU-parallel simulation** — 64–512 robots stepped simultaneously in Genesis;
  measured throughput up to ~19k env-steps/s on a single RTX 4070.
- **Clean SB3 ↔ Genesis integration** — a custom `VecEnv` bridges a GPU tensor
  simulator to CPU-side PPO, including per-environment perturbation scheduling.
- **Reward engineering with measurable effect** — re-balancing `tracking_lin_vel`
  (1.0 → 5.0) and `base_height` (−50 → −5) is documented as the change that
  unlocked walking.
- **Genuine multi-algorithm study** — PPO, SAC, and TD3 under matched conditions,
  with a clear and reproducible methodology.
- **Reproducible configuration** — all environment/reward/PPO settings live in
  versioned code, not notebooks.

## 9. Weaknesses / Risks

- **Unquantified robustness** — the project's headline claim (push hardening) has
  no persisted numerical result.
- **N = 1 experiments** — no statistical rigor on the algorithm comparison.
- **Synthetic fallback in legacy plotting** — addressed but worth removing.
- **Heterogeneous reward scaling** across runs makes some cross-run reward
  comparisons invalid if not careful (documented in §5.2).
- **Dead code** — `extra_plots.py` depends on non-existent CSVs.

---

## 10. Recommended Next Steps (engineering)

1. Persist stress-test output to CSV/JSON and plot failure threshold per
   checkpoint — converts the headline claim from qualitative to quantitative.
2. Add a deterministic evaluation harness (N episodes → mean velocity, tracking
   error, fall rate) and log to CSV.
3. Re-run PPO/SAC/TD3 with ≥3 seeds each; report mean ± std.
4. Remove the synthetic fallback from `generate_plots.py`; fix or delete
   `extra_plots.py`.
5. Add a curriculum that raises push force as survival improves, and log the
   schedule.
