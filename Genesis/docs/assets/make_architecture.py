"""
Publication-quality system-architecture diagram for the Go2 RL pipeline.
Renders identical SVG (vector) and PNG (raster) via matplotlib.

Pipeline (closed RL loop):
  Observation -> Policy Network -> Action -> Quadruped Controller
   -> Genesis Physics -> Reward -> Policy Update -> (back to Policy Network)

Run:  python docs/assets/make_architecture.py
Out:  docs/assets/architecture.svg, docs/assets/architecture.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette
AGENT = "#1f77b4"      # learner side (blue)
AGENT_BG = "#e8f1fb"
ENV = "#2ca02c"        # environment side (green)
ENV_BG = "#eaf6ea"
EDGE = "#33404d"
TEXT = "#16202b"

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(11, 9.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")

# title
ax.text(5, 10.6, "System Architecture — Go2 Reinforcement-Learning Locomotion",
        ha="center", va="center", fontsize=15, fontweight="bold", color=TEXT)
ax.text(5, 10.18, "Genesis (GPU physics)  +  Stable-Baselines3 (PPO / SAC / TD3)",
        ha="center", va="center", fontsize=10.5, color="#5a6b7b", style="italic")

# Box geometry
BW, BH = 4.4, 1.02
CX = 5.0
ys = [9.0, 7.6, 6.2, 4.8, 3.4, 2.0, 0.6]

stages = [
    ("1 · Observation Space", "45-D proprioception: base ang-vel, gravity,\nvelocity command, joint pos/vel, prev action", AGENT, AGENT_BG),
    ("2 · Policy Network", "MLP actor (Stable-Baselines3)\nPPO clipped · SAC entropy · TD3 twin-critic", AGENT, AGENT_BG),
    ("3 · Action Generation", "12-D joint position targets\na · 0.25 + default stance,  1-step latency", AGENT, AGENT_BG),
    ("4 · Quadruped Controller", "PD position control  (kp = 20, kd = 0.5)\n12 actuated joints @ 50 Hz", ENV, ENV_BG),
    ("5 · Genesis Physics Simulation", "GPU rigid-body, 2 substeps, dt = 0.02 s\n64–512 robots in parallel · + lateral push 5–15 N", ENV, ENV_BG),
    ("6 · Reward Computation", "6 terms: vel-tracking (+5), yaw (+0.2),\nheight (−5), z-vel (−1), action-rate, stance", ENV, ENV_BG),
    ("7 · Policy Update", "GAE-λ advantage · Adam · γ = 0.99\non-policy rollout (PPO) / replay (SAC, TD3)", AGENT, AGENT_BG),
]

centers = {}
for (title, sub, edge, bg), y in zip(stages, ys):
    box = FancyBboxPatch((CX - BW / 2, y - BH / 2), BW, BH,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=2, edgecolor=edge, facecolor=bg, zorder=3)
    ax.add_patch(box)
    ax.text(CX, y + 0.21, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=edge, zorder=4)
    ax.text(CX, y - 0.21, sub, ha="center", va="center",
            fontsize=8.3, color=TEXT, zorder=4)
    centers[title] = y

# downward arrows between consecutive boxes
for y_top, y_bot in zip(ys[:-1], ys[1:]):
    ax.add_patch(FancyArrowPatch((CX, y_top - BH / 2), (CX, y_bot + BH / 2),
                 arrowstyle="-|>", mutation_scale=16, linewidth=2,
                 color=EDGE, zorder=2))

# ---- Feedback loop: Policy Update (7) -> Policy Network (2), left side ----
x_left = CX - BW / 2
lx = 1.1
ax.add_patch(FancyArrowPatch(
    (x_left, ys[6]), (x_left, ys[1]),
    connectionstyle=f"arc3,rad=0", arrowstyle="-|>", mutation_scale=18,
    linewidth=2.2, color=AGENT, zorder=1,
    patchA=None, shrinkA=0, shrinkB=0))
# route it out to the left as an elbow
ax.add_patch(FancyArrowPatch((x_left, ys[6]), (lx, ys[6]), arrowstyle="-",
             linewidth=2.2, color=AGENT, zorder=1))
ax.add_patch(FancyArrowPatch((lx, ys[6]), (lx, ys[1]), arrowstyle="-",
             linewidth=2.2, color=AGENT, zorder=1))
ax.add_patch(FancyArrowPatch((lx, ys[1]), (x_left, ys[1]), arrowstyle="-|>",
             mutation_scale=18, linewidth=2.2, color=AGENT, zorder=1))
ax.text(lx - 0.12, (ys[6] + ys[1]) / 2, "gradient update\n(learning loop)",
        ha="center", va="center", rotation=90, fontsize=8.6,
        color=AGENT, fontweight="bold")

# ---- Rollout loop: Genesis Physics (5) -> Observation (1), right side ----
x_right = CX + BW / 2
rx = 8.9
ax.add_patch(FancyArrowPatch((x_right, ys[4]), (rx, ys[4]), arrowstyle="-",
             linewidth=2.2, color=ENV, zorder=1))
ax.add_patch(FancyArrowPatch((rx, ys[4]), (rx, ys[0]), arrowstyle="-",
             linewidth=2.2, color=ENV, zorder=1))
ax.add_patch(FancyArrowPatch((rx, ys[0]), (x_right, ys[0]), arrowstyle="-|>",
             mutation_scale=18, linewidth=2.2, color=ENV, zorder=1))
ax.text(rx + 0.12, (ys[4] + ys[0]) / 2, "next state\n(env rollout)",
        ha="center", va="center", rotation=90, fontsize=8.6,
        color=ENV, fontweight="bold")

# ---- Compute-placement side bands ----
ax.text(1.05, 9.9, "AGENT", ha="center", fontsize=9, fontweight="bold", color=AGENT)
ax.text(1.05, 9.6, "CPU policy", ha="center", fontsize=7.6, color="#5a6b7b")
ax.text(8.95, 9.9, "ENVIRONMENT", ha="center", fontsize=9, fontweight="bold", color=ENV)
ax.text(8.95, 9.6, "GPU physics", ha="center", fontsize=7.6, color="#5a6b7b")

fig.tight_layout()
for ext in ("svg", "png"):
    out = os.path.join(HERE, f"architecture.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)
plt.close(fig)
