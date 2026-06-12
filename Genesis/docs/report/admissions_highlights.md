# Project Highlights — For Admissions & Recruiting Reviewers

A concise, evidence-backed summary of this project for reviewers evaluating
applications to **Robotics / Automation / Mechatronics / AI / Autonomous Systems**
graduate programs, research-assistant positions, and robotics engineering
internships.

---

## One-Paragraph Summary

This project trains a **Unitree Go2 quadruped** to walk and to resist lateral
push perturbations using deep reinforcement learning in the **Genesis** GPU
physics simulator. It implements a complete RL pipeline — environment design,
reward engineering, GPU-parallel training (64–512 robots simultaneously), and
evaluation — and culminates in a **controlled comparison of three RL algorithms
(PPO, SAC, TD3)** on the same perturbation task. The headline empirical finding,
extracted directly from training logs, is that **on-policy PPO produces the most
stable, highest-performing, and cheapest-to-train policy (914/1000-step survival
under active pushes), while off-policy SAC is roughly twice as sample-efficient
but unstable, and TD3 collapses late in training.**

---

## Skills Demonstrated (mapped to evidence)

| Skill area | What was done | Evidence in repo |
|---|---|---|
| **Reinforcement learning** | Trained and compared PPO, SAC, TD3 on a continuous-control task | `sb3_train.py`, `algo_comparison/`, TensorBoard logs |
| **Robot control** | 50 Hz PD position control of a 12-DoF quadruped | `go2_env.py` (kp/kd, action scaling) |
| **Physics-based simulation** | GPU-parallel rigid-body sim, up to ~19k steps/s on one GPU | Genesis integration, measured fps |
| **Environment design** | 45-D observation, 12-D action, termination logic, command sampling | `go2_env.py` |
| **Reward engineering** | Six-term reward; documented re-balancing that unlocked walking | `get_cfgs`, README reward section |
| **Sim infrastructure** | Custom SB3 `VecEnv` bridging a GPU tensor sim to CPU PPO, with per-env perturbation scheduling | `GenesisVecEnv` in `sb3_train.py` |
| **Experimental methodology** | Matched-condition algorithm comparison, fixed step/env budget | `algo_comparison/train_all.py` |
| **Data analysis & visualization** | Metric extraction from event files; publication-quality figures | `extract_all_data.py`, `docs/comparison/` |
| **Scientific integrity** | Identified and corrected a silent synthetic-data fallback in legacy plotting | `repository_audit.md` §7 |
| **Reproducibility** | All configs versioned in code; runnable extraction + figure scripts | repo-wide |

---

## Why This Is Research-Relevant

- **Legged locomotion under perturbation** is an active robotics research area
  (sim-to-real, robust control, whole-body control). This project engages it
  directly with a standard platform (Go2) and standard methods.
- **Algorithm robustness vs sample efficiency** is a central RL question. The
  project produces a clean, reproducible instance of the classic trade-off:
  PPO stable, SAC efficient-but-fragile, TD3 brittle — observed on a real task,
  not a toy benchmark.
- **GPU-parallel simulation** (Genesis) is at the frontier of how modern robot
  learning scales; demonstrating fluency with it is current and relevant.

## Why This Is Engineering-Relevant

- End-to-end ownership: environment, training loop, perturbation system,
  evaluation, analysis, and documentation.
- Systems integration across a GPU simulator and a CPU RL library, with explicit
  attention to throughput (measured 6.7k–19k env-steps/s).
- Honest engineering judgment: the author-facing audit lists exactly what is and
  is not measured, which is the kind of rigor production ML teams value.

---

## Technical Complexity (honest assessment)

**Substantial:**
- Bridging a GPU-tensor simulator to SB3's CPU `VecEnv` API correctly (action
  marshaling, per-env reset masks, perturbation scheduling) is non-trivial
  systems work.
- Running three algorithm families under matched conditions and analyzing the
  results from raw event files is real experimental work.

**Bounded (stated plainly):**
- Single-seed experiments (N = 1) — indicative, not statistically conclusive.
- Robustness is shown via in-training survival and video, not a persisted
  stress-test metric.
- No sim-to-real transfer (simulation only).

These bounds are normal for an individual portfolio project and are documented
rather than hidden — which itself signals research maturity.

---

## Suggested Talking Points (interview / SoP)

1. *"I found the real story only by parsing the raw TensorBoard logs — the
   convenience plots had a synthetic-data fallback that could have masked the
   actual result. PPO retained 98 % of peak performance; SAC and TD3 did not."*
2. *"SAC reached full-episode survival in 2 M steps — half of PPO's — but it
   couldn't hold it. That's the sample-efficiency vs stability trade-off, on a
   real legged-locomotion task."*
3. *"The single change that unlocked walking was reward re-balancing — raising
   velocity-tracking weight and softening the height penalty."*
4. *"I can quantify exactly what's missing — a persisted push-recovery metric and
   multi-seed runs — and I've scoped how to add them."*

---

## How to Strengthen Toward a Top-Tier Submission

1. Multi-seed runs (≥3) with mean ± std and significance tests.
2. Persisted, plotted stress-test curve (failure force per checkpoint).
3. A deterministic evaluation harness (velocity tracking error, fall rate, cost
   of transport).
4. A short methods write-up framed as a mini-paper (the README already approaches
   this).
5. Optional: a sim-to-real or domain-randomization section.
