"""
Plot training metrics from TensorBoard event files.

Reads all runs from push_training_logs/ and generates:
  training_graphs/<run_name>_metrics.png  — individual run with all metrics
  training_graphs/comparison_ep_len.png   — all runs overlaid (ep_len_mean)
  training_graphs/comparison_ep_rew.png   — all runs overlaid (ep_rew_mean)
"""

import os
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = "push_training_logs"
OUT_DIR = "training_graphs"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "train/explained_variance",
    "train/entropy_loss",
    "train/approx_kl",
    "train/loss",
]

# Short display names for subplot titles
DISPLAY_NAMES = {
    "rollout/ep_len_mean":      "Episode Length (mean)",
    "rollout/ep_rew_mean":      "Episode Reward (mean)",
    "train/explained_variance":  "Explained Variance",
    "train/entropy_loss":        "Entropy Loss",
    "train/approx_kl":           "Approx KL",
    "train/loss":                "Loss",
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  8,
})


def load_run(run_dir):
    """Load scalar metrics from a single TensorBoard run directory."""
    ea = EventAccumulator(run_dir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    data = {}
    for metric in METRICS:
        if metric not in available:
            continue
        events = ea.Scalars(metric)
        steps  = [e.step for e in events]
        values = [e.value for e in events]
        if steps:
            data[metric] = (steps, values)
    return data


def plot_individual(run_name, data):
    """Plot all metrics for a single run in a 2x3 grid."""
    present = [m for m in METRICS if m in data]
    if not present:
        print(f"  [{run_name}] No metrics found, skipping.")
        return

    n = len(present)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(run_name, fontsize=14, fontweight="bold", y=1.01)

    for idx, metric in enumerate(present):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        steps, values = data[metric]
        ax.plot(steps, values, linewidth=1.2)
        ax.set_title(DISPLAY_NAMES.get(metric, metric))
        ax.set_xlabel("Steps")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{run_name}_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(all_data, metric, filename):
    """Overlay one metric across all runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(DISPLAY_NAMES.get(metric, metric), fontsize=14, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel(DISPLAY_NAMES.get(metric, metric))

    for run_name in sorted(all_data.keys()):
        data = all_data[run_name]
        if metric not in data:
            continue
        steps, values = data[metric]
        ax.plot(steps, values, linewidth=1.2, label=run_name, alpha=0.85)

    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    # Discover runs
    run_dirs = sorted(glob.glob(os.path.join(LOG_DIR, "*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    if not run_dirs:
        print(f"No runs found in {LOG_DIR}/")
        return

    print(f"Found {len(run_dirs)} runs in {LOG_DIR}/\n")

    # Load all runs
    all_data = {}
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"Loading {run_name}...")
        data = load_run(run_dir)
        if data:
            all_data[run_name] = data
        else:
            print(f"  [{run_name}] No scalar data found.")

    # Individual plots
    print(f"\nGenerating individual plots...")
    for run_name, data in all_data.items():
        plot_individual(run_name, data)

    # Comparison plots
    print(f"\nGenerating comparison plots...")
    plot_comparison(all_data, "rollout/ep_len_mean", "comparison_ep_len.png")
    plot_comparison(all_data, "rollout/ep_rew_mean", "comparison_ep_rew.png")

    print(f"\nDone. All graphs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
