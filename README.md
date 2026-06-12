# Reinforcement Learning for Robust Quadruped Locomotion
### Push-Hardened Walking on the Unitree Go2 — and a Controlled PPO vs SAC vs TD3 Study

A reinforcement-learning study that trains a **Unitree Go2** quadruped to walk and
to **recover from lateral push perturbations**, built on the **Genesis** GPU
physics engine and **Stable-Baselines3**. The project implements the full pipeline
— environment, reward design, GPU-parallel training, evaluation — and runs a
**matched-condition comparison of three RL algorithms (PPO, SAC, TD3)** on the
perturbation task.

> **Headline result (measured, not estimated):** On the push-hardening task,
> **PPO** learns the most stable and highest-performing policy — **914 / 1000**
> control steps of survival under active 5–15 N pushes, retaining **98 %** of its
> peak performance — and trains **~2.3× faster** than the off-policy methods.
> **SAC** is ~2× more *sample-efficient* (full-episode survival at 2.0 M steps)
> but **degrades** in late training; **TD3 collapses** after ~4.5 M steps.
> Every number in this README is extracted directly from the TensorBoard logs in
> this repository (see [Reproducibility](#reproducibility)).

<p align="center">
<img src="Genesis/docs/comparison/fig_dashboard.png" width="100%">
</p>

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Problem Statement](#3-problem-statement)
4. [System Architecture](#4-system-architecture)
5. [Reinforcement Learning Framework](#5-reinforcement-learning-framework)
6. [Simulation Environment](#6-simulation-environment)
7. [Observation Space](#7-observation-space)
8. [Action Space](#8-action-space)
9. [Reward Design](#9-reward-design)
10. [Training Pipeline](#10-training-pipeline)
11. [Experimental Setup](#11-experimental-setup)
12. [Results](#12-results)
13. [PPO vs SAC vs TD3 Comparison](#13-ppo-vs-sac-vs-td3-comparison)
14. [Key Engineering Contributions](#14-key-engineering-contributions)
15. [Reproducibility](#15-reproducibility)
16. [Limitations](#16-limitations)
17. [Future Work](#17-future-work)
18. [Citation](#18-citation)
19. [Acknowledgements](#19-acknowledgements)

---

## 1. Project Overview

Quadruped robots must keep their balance while walking over uneven terrain and
absorbing disturbances. This project studies that problem in simulation: a Unitree
Go2 (12 actuated joints) learns a forward walking gait and is then **hardened
against lateral pushes** by training under randomized perturbation forces. The
work is implemented end-to-end and concludes with a controlled study of how three
standard deep-RL algorithms behave on the same robust-locomotion task.

**Scope at a glance**

| | |
|---|---|
| Robot | Unitree Go2 — 12-DoF quadruped (URDF in Genesis) |
| Simulator | Genesis — GPU-accelerated rigid-body physics |
| RL library | Stable-Baselines3 (PPO, SAC, TD3) |
| Parallelism | 64–512 robots stepped simultaneously on one GPU |
| Hardware | NVIDIA RTX 4070, Intel Core Ultra 9 185H |
| Control | 50 Hz PD position control |
| Disturbance | Lateral 5–15 N impulses, randomized in time and direction |
| Best walk policy | 33.7 reward, **933 / 1000** step survival (15.7 M steps, 17.7 min) |

**Documentation**
- [`docs/report/repository_audit.md`](Genesis/docs/report/repository_audit.md) — full repository audit
- [`docs/report/experiment_summary.md`](Genesis/docs/report/experiment_summary.md) — extracted results, every run
- [`docs/report/algorithm_comparison.md`](Genesis/docs/report/algorithm_comparison.md) — detailed PPO/SAC/TD3 study
- [`docs/report/admissions_highlights.md`](Genesis/docs/report/admissions_highlights.md) — skills & relevance summary

---

## 2. Motivation

Legged locomotion under disturbance is a core robotics problem: a robot that walks
well on a treadmill is useless if a shove or an uneven step puts it on the ground.
Two questions drive this project:

1. **Can a model-free RL policy learn active balance recovery** from lateral
   pushes, using only proprioceptive observations (no vision, no external state)?
2. **Which RL algorithm family is best suited** to this robust-locomotion task —
   stable on-policy (PPO) or sample-efficient off-policy (SAC, TD3)?

Answering (2) with real, matched experiments — rather than folklore — is the
project's distinctive contribution.

---

## 3. Problem Statement

Learn a control policy **π(a | o)** mapping a 45-dimensional proprioceptive
observation **o** to 12 joint-position targets **a**, such that the Go2:

- tracks a commanded forward velocity of **0.5 m/s**,
- maintains body height and an upright orientation (|roll|, |pitch| < 10°),
- and **survives randomized lateral pushes of 5–15 N** for as much of a 20-second
  (1000-step) episode as possible.

Performance is measured by **mean episode reward** (dominated by velocity
tracking) and, as a scale-invariant robustness proxy, **mean episode length**
(how long the robot stays upright before a termination).

---

## 4. System Architecture

```
                ┌─────────────────────────────────────────────┐
                │  Stable-Baselines3  (PPO / SAC / TD3)  [CPU] │
                │  policy + value/critic networks, optimizer   │
                └───────────────▲───────────────┬─────────────┘
                    obs (45-D)   │               │  actions (12-D)
                 reward, done     │               ▼
                ┌────────────────┴───────────────────────────┐
                │   GenesisVecEnv  (custom SB3 VecEnv bridge) │
                │   • marshals CPU actions → GPU tensors      │
                │   • schedules per-env lateral push forces   │
                │   • per-env reset masks, episode bookkeeping │
                └───────────────▲───────────────┬─────────────┘
                                │               │
                ┌───────────────┴───────────────▼─────────────┐
                │   Go2Env  →  Genesis Scene  (RTX 4070) [GPU] │
                │   64–512 robots stepped in parallel @ 50 Hz  │
                └─────────────────────────────────────────────┘
```

The key systems-integration piece is **`GenesisVecEnv`** (`Genesis/sb3_train.py`):
it presents a GPU tensor simulator to SB3's CPU-side `VecEnv` API, handling action
marshaling, per-environment perturbation scheduling, reset masks, and episode
tracking.

---

## 5. Reinforcement Learning Framework

| Component | Choice |
|---|---|
| Policy / value network | MLP (`MlpPolicy`) |
| On-policy algorithm | **PPO** — clipped surrogate objective |
| Off-policy algorithms | **SAC** (entropy-regularized), **TD3** (twin critics, delayed updates) |
| Discount γ | 0.99 |
| GAE λ (PPO) | 0.95 |
| Rollout (PPO) | `n_steps = 2048`, `batch_size = 512`, `n_epochs = 10` |
| Learning rate | 3 × 10⁻⁴ |
| PPO clip range | 0.1 |
| Entropy coefficient | 0.01 |
| Compute split | Policy optimization on CPU; physics on GPU |

---

## 6. Simulation Environment

Defined in [`Genesis/examples/locomotion/go2_env.py`](Genesis/examples/locomotion/go2_env.py).

| Property | Value |
|---|---|
| Control frequency | 50 Hz (`dt = 0.02 s`), 2 physics substeps |
| Episode length | 20 s = **1000 control steps** |
| Joint controller | PD position, `kp = 20.0`, `kd = 0.5` |
| Action scaling | 0.25 (action → joint offset from default stance) |
| Termination | |roll| > 10° or |pitch| > 10° |
| Command | forward velocity fixed at 0.5 m/s |
| Push perturbation | lateral (Y-axis) 5–15 N, every 150 steps, 10-step hold, random ± direction |
| Parallel envs | 64 (comparison) to 512 (walk training) |

---

## 7. Observation Space

A single 45-dimensional proprioceptive vector (no vision, no privileged state):

| Block | Dim | Scale |
|---|---|---|
| Base angular velocity | 3 | 0.25 |
| Projected gravity vector | 3 | — |
| Velocity command (x, y, yaw) | 3 | [2.0, 2.0, 0.25] |
| Joint positions (− default stance) | 12 | 1.0 |
| Joint velocities | 12 | 0.05 |
| Previous action | 12 | — |
| **Total** | **45** | |

---

## 8. Action Space

A 12-dimensional continuous vector — one **target position offset per joint**.
Each action is multiplied by `action_scale = 0.25` and added to the default
stance; the PD controller (`kp = 20`, `kd = 0.5`) tracks the resulting target.
A one-step action latency is simulated to better match real hardware.

---

## 9. Reward Design

Six-term reward (`get_cfgs` in `go2_train.py`), summed per step:

| Term | Weight | Role |
|---|---|---|
| `tracking_lin_vel` | **+5.0** | Follow commanded forward velocity (dominant) |
| `tracking_ang_vel` | +0.2 | Follow commanded yaw rate |
| `lin_vel_z` | −1.0 | Penalize vertical bounce |
| `base_height` | −5.0 | Maintain target body height |
| `action_rate` | −0.005 | Penalize jerky action changes |
| `similar_to_default` | −0.1 | Stay near nominal stance |

**Reward-engineering finding.** Early runs failed to walk. Raising
`tracking_lin_vel` from 1.0 → 5.0 (making forward motion the dominant objective)
and softening `base_height` from −50 → −5 (which had been pinning the robot in
place) was the change that **unlocked walking**. Because of this re-scaling, raw
reward is **not comparable across reward configurations** — episode length (capped
at 1000) is used as the scale-invariant cross-run metric.

---

## 10. Training Pipeline

```bash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# Push-hardened PPO — 64 parallel robots, 5M steps (~13 min on RTX 4070)
python Genesis/sb3_train.py \
    --num-envs 64 --push-min 5.0 --push-max 15.0 \
    --total-steps 5000000 \
    --run-name go2_64env --save-name go2_push_hardened_64env
```

Checkpoints are written every 500 K steps to `Genesis/checkpoints/`. Training is
monitored in TensorBoard (`--tensorboard-log`).

---

## 11. Experimental Setup

**Algorithm comparison** (the primary study): PPO, SAC, and TD3 were each trained
for **~5 M environment steps** on the **identical** push-hardening environment
(64 envs, same dynamics / observation / reward / termination), via
`Genesis/algo_comparison/train_all.py`. Wall-clock time was recorded to
`wall_clock.json`; all scalars logged to TensorBoard.

**Forward-walk experiments**: longer runs (up to 15.7 M steps, 512 envs) producing
the best deployable walking policy.

> **Methodological caveat:** each algorithm was run **once (N = 1)**. Results are a
> clear, internally consistent indication of behavior on this task — not a
> statistically significant ranking. Multi-seed runs are listed under
> [Future Work](#17-future-work).

---

## 12. Results

### 12.1 Best Forward-Walking Policy

The strongest locomotion policy (`Train_4_walk_long_1`, 15.7 M steps) reaches a
final reward of **33.7** and an episode length of **933 / 1000** (93 % survival),
trained in **17.7 minutes** at a mean **19,334 env-steps/s**.

<p align="center">
<img src="Genesis/docs/comparison/fig_walk_curve.png" width="80%">
</p>

### 12.2 Push-Recovery (qualitative)

Trained policies maintain near-full-episode survival *while being pushed*
(PPO: 914 / 1000 under continual 5–15 N lateral impulses). Recovery behavior is
shown in the evaluation videos (`Genesis/*_eval.mp4`).

> **Honest gap:** the `go2_stress_test.py` ramp test prints a failure threshold to
> stdout but **does not persist it**, so no per-model robustness number is stored
> in the repo. Push robustness is therefore reported here via **in-training
> survival** and **video**, not an isolated stress-test metric. See
> [Limitations](#16-limitations).

---

## 13. PPO vs SAC vs TD3 Comparison

All three trained ~5 M steps under identical conditions. **Every value below is
measured** from `Genesis/algo_comparison/logs/`.

| Metric | PPO | SAC | TD3 | Best |
|---|---|---|---|---|
| Environment steps | 5.11 M | 4.99 M | 5.00 M | — |
| **Final reward** (last 10 %) | **34.3** | 18.8 | 6.7 | PPO |
| Peak reward | 35.0 | **35.5** | 27.0 | SAC |
| Step of peak reward | 4.72 M | **1.26 M** | 4.49 M | SAC |
| **Final episode length** | **914** | 547 | 147 | PPO |
| Peak episode length | 942 | **1001** | 785 | SAC |
| **Reward retention** (final/peak) | **98 %** | 53 % | 25 % | PPO |
| Survival retention (final/peak) | **97 %** | 55 % | 19 % | PPO |
| Wall-clock (~5 M steps) | **12.6 min** | 35.6 min | 28.3 min | PPO |
| Mean throughput | **6,754 fps** | 2,544 fps | 2,974 fps | PPO |

**Reward vs steps** — PPO climbs to a stable plateau; SAC oscillates; TD3 stays
low then spikes and collapses.

<p align="center">
<img src="Genesis/docs/comparison/fig_reward_vs_steps.png" width="80%">
</p>

**Policy retention (the core stability finding)** — PPO keeps ~98 % of peak
performance to the end; SAC and TD3 lose most of theirs.

<p align="center">
<img src="Genesis/docs/comparison/fig_peak_vs_final.png" width="92%">
</p>

**Training cost** — on-policy PPO is ~2.3× faster in wall-clock and ~2.3× higher
throughput than the off-policy methods, which update at nearly every step.

<p align="center">
<img src="Genesis/docs/comparison/fig_training_cost.png" width="80%">
</p>

**Interpretation**
- **PPO — best overall & most deployable.** Highest final reward and survival,
  near-monotonic, cheapest to train. The clipped on-policy update suppresses the
  large policy shifts that destabilize the others.
- **SAC — most sample-efficient, but fragile.** Full-episode survival at 2.0 M
  steps (½ of PPO's budget), then degrades — usable only with peak-checkpoint
  selection, not the final policy.
- **TD3 — least stable.** Peaks late then collapses to 147-step survival;
  not recommended on this task without stabilization work.

Full analysis: [`docs/report/algorithm_comparison.md`](Genesis/docs/report/algorithm_comparison.md).

---

## 14. Key Engineering Contributions

- **Simulator–RL integration.** Implemented `GenesisVecEnv`, a custom SB3 `VecEnv`
  that bridges a GPU-tensor physics simulator to CPU-side PPO/SAC/TD3 — including
  action marshaling, per-environment reset masking, and episode bookkeeping.
- **Perturbation subsystem.** Built a per-environment lateral-push scheduler
  (randomized magnitude, timing, and direction) applied through the Genesis
  external-force API, enabling robustness training at scale.
- **GPU-parallel training.** Drove 64–512 simulated robots concurrently on a
  single RTX 4070, with measured throughput up to **~19,000 env-steps/s**.
- **Reward engineering.** Diagnosed a non-walking failure mode and resolved it by
  re-balancing the velocity-tracking and body-height reward terms — a documented,
  reproducible fix.
- **Controlled algorithm study.** Ran PPO, SAC, and TD3 under matched conditions
  and analyzed them from raw event files, surfacing a clear stability /
  sample-efficiency trade-off.
- **Scientific integrity tooling.** Identified a silent *synthetic-data fallback*
  in the legacy plotting script and replaced it with a **real-data-only** figure
  generator (`docs/comparison/make_figures.py`) that errors rather than fabricates
  when a log is missing.

---

## 15. Reproducibility

**Environment**
```bash
# 1. Genesis physics engine (this repo is built around it)
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis && pip install -e .

# 2. RL dependencies
pip install stable-baselines3 gymnasium torch tensorboard matplotlib
```

**Hardware used** — NVIDIA RTX 4070 (12 GB), Intel Core Ultra 9 185H. A CUDA GPU
is required for Genesis. Expected timings on an RTX 4070: ~13 min for 5 M PPO
steps (64 envs), ~18 min for 15.7 M walk steps (512 envs).

**Train / evaluate / stress-test**
```bash
python Genesis/sb3_train.py --num-envs 64 --total-steps 5000000   # train
python Genesis/sb3_eval.py                                        # render / record mp4
python Genesis/go2_stress_test.py                                 # ramp force to failure
```

**Algorithm comparison**
```bash
cd Genesis/algo_comparison
python train_all.py          # PPO, SAC, TD3 sequentially
```

**Reproduce the reported numbers and figures (read-only, no training needed)**
```bash
cd Genesis
python extract_all_data.py            # → docs/report/extracted_metrics.json
python docs/comparison/make_figures.py  # → docs/comparison/*.png
```
Both scripts read only measured TensorBoard logs and **raise an error if a log is
missing** — they never substitute synthetic data.

---

## 16. Limitations

These are stated plainly; they bound the strength of the claims above.

1. **Single seed per algorithm (N = 1).** No confidence intervals or significance
   tests — the PPO/SAC/TD3 ranking is indicative, not conclusive.
2. **No persisted robustness metric.** The stress test prints to stdout only; no
   per-checkpoint failure-force number is stored. Robustness is shown via
   in-training survival and video.
3. **No standalone evaluation harness.** Gait quality (tracking error, fall rate,
   cost of transport) is inferred from training reward/episode-length, not
   measured in dedicated eval episodes.
4. **Simulation only.** No sim-to-real transfer or domain randomization beyond the
   push perturbation.
5. **Heterogeneous reward scaling across runs** — handled by using episode length
   as the cross-run metric, but a caveat to be aware of.

---

## 17. Future Work

- **Multi-seed runs (≥ 3)** for each algorithm; report mean ± std and significance.
- **Persisted stress-test curve** — log failure force per checkpoint and plot
  robustness vs training progress.
- **Deterministic evaluation harness** — velocity-tracking error, fall rate, cost
  of transport over N held-out episodes.
- **Push curriculum** — raise perturbation force as survival improves.
- **Omnidirectional & terrain robustness** — X-axis pushes, uneven ground.
- **Sim-to-real** — domain randomization toward Go2 hardware deployment.

---

## 18. Citation

```bibtex
@misc{go2_rl_quadruped_study,
  title        = {Reinforcement Learning for Robust Quadruped Locomotion:
                  Push-Hardened Walking on the Unitree Go2 and a PPO/SAC/TD3 Study},
  author       = {Amrit},
  year         = {2026},
  howpublished = {\url{https://github.com/Amr-SS/RL_Quadruped_study}},
  note         = {Built on the Genesis physics engine and Stable-Baselines3}
}
```

---

## 19. Acknowledgements

- **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** — GPU-accelerated
  physics engine and the Go2 locomotion example this project builds upon.
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — PPO, SAC,
  and TD3 implementations.
- **Unitree Robotics** — the Go2 platform and URDF model.

---

*All experiments were run locally on a single RTX 4070 — no cloud compute. Every
quantitative claim in this document is traceable to a measured TensorBoard event
file via the extraction scripts in this repository.*
