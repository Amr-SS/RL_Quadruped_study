# RL Quadruped Study — Go2 Push Hardening

Training a Unitree Go2 quadruped to resist lateral push perturbations using SB3 PPO + Genesis physics engine.

---

## Setup

```bash
# Clone Genesis and install
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .

# Install dependencies
pip install stable-baselines3 gymnasium torch
```

Clone this repo into your Genesis folder:
```bash
cd ~/Genesis
git clone https://github.com/Amr-SS/RL_Quadruped_study.git .
```

---

## Files

| File | Purpose |
|---|---|
| `sb3_train.py` | Main training script — PPO with push perturbations, 64 parallel envs |
| `sb3_eval.py` | Load a saved model and watch/record it in the Genesis viewer |
| `go2_stress_test.py` | Stress test any checkpoint — ramps up force until robot falls |

---

## Train

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python sb3_train.py \
    --num-envs 64 \
    --push-min 5.0 \
    --push-max 15.0 \
    --total-steps 5000000 \
    --run-name "go2_64env" \
    --save-name "go2_push_hardened_64env"
```

Runs 64 parallel robots on GPU. ~15-20 minutes for 5M steps on RTX 4070.

Checkpoints saved every 500K steps to `./checkpoints/`

---

## Evaluate — Watch the robot

```bash
python sb3_eval.py
```

Edit `MODEL_PATH` at the top of `sb3_eval.py` to load a specific checkpoint.
Set `RECORD_VIDEO = True` to save an mp4 instead of opening the viewer.

---

## Stress Test — Find the failure threshold

```bash
python go2_stress_test.py
```

Edit `MODE` at the top:
- `MODE = "sb3"` — test your SB3 trained model
- `MODE = "rsl"` — test the RSL-RL baseline (needs `logs/go2-walking/model_100.pt`)

The script ramps up lateral force every 150 steps and reports the Newton value when the robot falls.

**Baseline to beat:** RSL-RL `model_100.pt` fails at **25N**.

---

## Monitor Training

```bash
tensorboard --logdir ./push_training_logs/
# open http://localhost:6006
```

Key metrics to watch:
- `ep_len_mean` → how long robot survives (target: 1000+)
- `ep_rew_mean` → reward accumulation (target: 2.0+)
- `explained_variance` → value function quality (target: 0.7+)

---

## Current Results

| Model | Failure Threshold |
|---|---|
| RSL-RL `model_100.pt` (baseline) | 25N |
| SB3 PPO 5M steps (this repo) | TBD — run stress test |

---

## Next Steps

- [ ] Fine-tune RSL-RL weights inside SB3 wrapper (Option B)
- [ ] Increase `push_force_max` as policy improves (curriculum)
- [ ] Test with X-axis pushes (omnidirectional robustness)
- [ ] Deploy on real Go2 hardware
