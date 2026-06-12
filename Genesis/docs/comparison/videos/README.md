# Side-by-Side Evaluation Video — PPO | SAC | TD3

![comparison](ppo_sac_td3_comparison.gif)

`ppo_sac_td3_comparison.mp4` (full quality) and `.gif` (inline preview) show the
three **final push-hardened policies** evaluated under **identical conditions**:

| Condition | Value |
|---|---|
| Environment | `Go2Env`, flat plane (same for all three) |
| Command | forward 0.5 m/s, deterministic policy |
| Perturbation | **none** during this clip (open-loop eval) |
| Camera | chase camera locked to a fixed offset from the base |
| Duration | 500 control steps = 10 s @ 50 fps |
| Labels | embedded header band (algorithm + policy family) |

## What this clip does and does not show — read this

This is an **honest, open-loop, no-push evaluation**. Objective measurements over
the 500 steps (no rendering, `record_eval.py` logic):

| Policy | Falls in 500 steps | Mean body height | Forward translation |
|---|---|---|---|
| PPO | 0 | 0.28 m | ~0 m |
| SAC | 0 | 0.25 m | ~0 m |
| TD3 | 0 | 0.25 m | ~0 m |

- **All three final policies hold balance** in this deterministic, no-push
  setting — so the clip is deliberately **not** labelled "stable/unstable/collapse".
  That distinction is a **training-time** result and lives in the quantitative
  curves (`../fig_peak_vs_final.png`), where PPO retains 98 % of peak performance
  while SAC/TD3 degrade under the perturbed, stochastic training distribution.
- **Minimal forward translation.** In this standalone eval harness the policies
  balance roughly in place rather than translating at the commanded 0.5 m/s. The
  same behavior appears for the dedicated walk checkpoints in this harness, which
  points to an **observation-ordering quirk** in the open-loop evaluator (see the
  `dof_pos` joint-order `FIX` notes in `sb3_eval.py` and `go2_stress_test.py`).
  This is documented rather than patched, to avoid altering any measured training
  conclusion. Fixing it is listed as future work.

In short: the video is a faithful artifact of the trained policies under matched
conditions; the **headline quantitative comparison remains the training-log
analysis**, not this clip.

## Reproduce / regenerate

```bash
# 1. Record each algorithm under identical conditions (needs GPU + display)
python docs/comparison/videos/record_eval.py --algo ppo \
    --model go2_ppo_comparison --steps 500 --out docs/comparison/videos/raw_ppo.mp4
python docs/comparison/videos/record_eval.py --algo sac \
    --model algo_comparison/models/sac/go2_sac_push_hardened --steps 500 \
    --out docs/comparison/videos/raw_sac.mp4
python docs/comparison/videos/record_eval.py --algo td3 \
    --model algo_comparison/models/td3/go2_td3_push_hardened --steps 500 \
    --out docs/comparison/videos/raw_td3.mp4

# 2. Stitch into the labelled side-by-side (pure Python, no system ffmpeg)
python docs/comparison/videos/stitch_sidebyside.py
```

To make the **robustness difference visible**, extend `record_eval.py` to apply
an identical scripted lateral push schedule to all three policies (the push
mechanism already exists in `sb3_train.py::GenesisVecEnv._apply_pushes`). That
turns this clip from a balance demo into a true push-recovery comparison.
