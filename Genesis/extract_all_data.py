"""
Ground-truth extraction of ALL real scalar data from every TensorBoard run in
this repository. Writes a single JSON summary so downstream documentation uses
ONLY measured values — never synthetic fallbacks.

Outputs: docs/report/extracted_metrics.json
"""

import os
import glob
import json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "report", "extracted_metrics.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Every directory that may hold event files
LOG_GROUPS = {
    "push_training_logs": os.path.join(ROOT, "push_training_logs"),
    "algo_ppo": os.path.join(ROOT, "algo_comparison", "logs", "ppo"),
    "algo_sac": os.path.join(ROOT, "algo_comparison", "logs", "sac"),
    "algo_td3": os.path.join(ROOT, "algo_comparison", "logs", "td3"),
}


def find_event_dirs(base):
    """Return every directory under `base` that directly contains an event file."""
    if not os.path.isdir(base):
        return []
    dirs = set()
    for path in glob.glob(os.path.join(base, "**", "events.out.tfevents.*"), recursive=True):
        dirs.add(os.path.dirname(path))
    return sorted(dirs)


def load_run(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for tag in tags:
        events = ea.Scalars(tag)
        out[tag] = {
            "steps": [int(e.step) for e in events],
            "values": [float(e.value) for e in events],
            "wall_time": [float(e.wall_time) for e in events],
        }
    return tags, out


def summarize_scalar(series):
    """Compute summary stats for a (steps, values) series."""
    v = np.array(series["values"], dtype=np.float64)
    s = np.array(series["steps"], dtype=np.float64)
    wt = np.array(series["wall_time"], dtype=np.float64)
    if len(v) == 0:
        return None
    # final = mean of last 10% of points (robust final value)
    tail = max(1, len(v) // 10)
    return {
        "n_points": int(len(v)),
        "step_min": float(s.min()),
        "step_max": float(s.max()),
        "value_min": float(v.min()),
        "value_max": float(v.max()),
        "value_mean": float(v.mean()),
        "final_value": float(v[-1]),
        "final_value_tail_mean": float(v[-tail:].mean()),
        "peak_value": float(v.max()),
        "peak_at_step": float(s[int(v.argmax())]),
        "wall_clock_minutes": float((wt.max() - wt.min()) / 60.0),
    }


def main():
    full = {}
    summary = {}

    for group, base in LOG_GROUPS.items():
        event_dirs = find_event_dirs(base)
        print(f"\n=== {group} ({base}) ===")
        if not event_dirs:
            print("  (no event files found)")
            continue
        for run_dir in event_dirs:
            run_name = os.path.relpath(run_dir, base)
            if run_name == ".":
                run_name = os.path.basename(base)
            key = f"{group}/{run_name}"
            try:
                tags, data = load_run(run_dir)
            except Exception as e:
                print(f"  [{run_name}] FAILED: {e}")
                continue
            print(f"  [{run_name}] tags: {tags}")
            full[key] = data
            summary[key] = {}
            for tag, series in data.items():
                stat = summarize_scalar(series)
                if stat:
                    summary[key][tag] = stat

    # Write the compact summary (committed provenance artifact).
    with open(OUT, "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(f"\nWrote {OUT} (summary only)")

    # Optionally dump the full raw curves alongside (git-ignored; large).
    if os.environ.get("DUMP_RAW"):
        raw_path = OUT.replace(".json", "_raw.json")
        with open(raw_path, "w") as f:
            json.dump({"summary": summary, "raw": full}, f, indent=2)
        print(f"Wrote {raw_path} (full raw curves)")

    # Print a compact human-readable summary of the headline metrics
    print("\n" + "=" * 70)
    print("HEADLINE METRICS (reward + ep_len, final tail-mean / peak)")
    print("=" * 70)
    for key in sorted(summary.keys()):
        rew = summary[key].get("rollout/ep_rew_mean")
        epl = summary[key].get("rollout/ep_len_mean")
        line = f"{key:45s}"
        if rew:
            line += f" | rew final={rew['final_value_tail_mean']:.3f} peak={rew['peak_value']:.3f} pts={rew['n_points']}"
        if epl:
            line += f" | eplen final={epl['final_value_tail_mean']:.0f} peak={epl['peak_value']:.0f}"
        print(line)


if __name__ == "__main__":
    main()
