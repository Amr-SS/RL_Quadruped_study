"""
Single entry point: train PPO, TD3, and SAC sequentially, then generate comparison plots.

Usage:
    python algo_comparison/train_all.py --timesteps 5000000
"""

import sys
import os
import json
import argparse
import subprocess
import time


def run_training(label, cmd, cwd):
    """Run a training subprocess and report wall-clock time."""
    print(f"\n{'=' * 60}")
    print(f"  Starting {label} training...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0

    minutes = elapsed / 60
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[train_all] {label} finished in {minutes:.1f} min — {status}")
    return result.returncode, minutes


def main():
    parser = argparse.ArgumentParser(
        description="Run PPO, TD3, and SAC training sequentially, then generate comparison plots."
    )
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total env steps for each algorithm")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--skip-ppo", action="store_true",
                        help="Skip PPO training (use existing logs)")
    parser.add_argument("--skip-td3", action="store_true",
                        help="Skip TD3 training (use existing logs)")
    parser.add_argument("--skip-sac", action="store_true",
                        help="Skip SAC training (use existing logs)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    algo_dir = os.path.dirname(os.path.abspath(__file__))
    log_base = os.path.join(algo_dir, "logs")
    py = sys.executable

    results = {}
    total_t0 = time.time()

    # ── PPO ────────────────────────────────────────────────────────────────────
    if not args.skip_ppo:
        rc, mins = run_training("PPO", [
            py, os.path.join(project_root, "sb3_train.py"),
            "--total-steps", str(args.timesteps),
            "--num-envs", str(args.num_envs),
            "--run-name", "ppo_comparison",
            "--save-name", "go2_ppo_comparison",
            "--tensorboard-log", os.path.join(log_base, "ppo"),
        ], cwd=project_root)
        results["PPO"] = {"rc": rc, "minutes": mins}
    else:
        print("[train_all] Skipping PPO (--skip-ppo)")

    # ── TD3 ────────────────────────────────────────────────────────────────────
    if not args.skip_td3:
        rc, mins = run_training("TD3", [
            py, os.path.join(algo_dir, "td3_train.py"),
            "--total-steps", str(args.timesteps),
            "--num-envs", str(args.num_envs),
            "--skip-plots",
        ], cwd=project_root)
        results["TD3"] = {"rc": rc, "minutes": mins}
    else:
        print("[train_all] Skipping TD3 (--skip-td3)")

    # ── SAC ────────────────────────────────────────────────────────────────────
    if not args.skip_sac:
        rc, mins = run_training("SAC", [
            py, os.path.join(algo_dir, "sac_train.py"),
            "--total-steps", str(args.timesteps),
            "--num-envs", str(args.num_envs),
            "--skip-plots",
        ], cwd=project_root)
        results["SAC"] = {"rc": rc, "minutes": mins}
    else:
        print("[train_all] Skipping SAC (--skip-sac)")

    total_mins = (time.time() - total_t0) / 60

    # ── Save wall-clock timing ─────────────────────────────────────────────────
    os.makedirs(log_base, exist_ok=True)
    timing_path = os.path.join(log_base, "wall_clock.json")
    with open(timing_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train_all] Wall-clock timing saved to {timing_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Training Summary")
    print(f"{'=' * 60}")
    for algo, info in results.items():
        status = "OK" if info["rc"] == 0 else "FAILED"
        print(f"  {algo:4s} : {info['minutes']:6.1f} min  [{status}]")
    print(f"  {'Total':4s} : {total_mins:6.1f} min")
    print(f"{'=' * 60}")

    # ── Generate comparison plots ──────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    plot_script = os.path.join(algo_dir, "generate_plots.py")
    subprocess.run([py, plot_script, "--log-base", log_base], cwd=project_root, check=False)

    print("\nAll done! Check algo_comparison/plots/ for comparison charts.")


if __name__ == "__main__":
    main()
