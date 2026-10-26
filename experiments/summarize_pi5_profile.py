#!/usr/bin/env python3
"""Summarize per-phase nanosecond profiles from a verified Pi5 benchmark JSON."""
import argparse
import json
from pathlib import Path

PHASES = ("preparation_ns", "zero_grad_ns", "forward_ns", "loss_metrics_ns",
          "backward_ns", "update_ns", "other_ns")


def summarize(report: dict) -> list[dict]:
    summaries = []
    for run in report["runs"]:
        epochs = run["learning_curve"]
        profiles = [e.get("training_profile") for e in epochs]
        if not profiles or any(not isinstance(p, dict) for p in profiles):
            raise ValueError("missing profiled epoch; use --profile-training image")
        wall = sum(p["training_wall_ns"] for p in profiles)
        if wall <= 0:
            raise ValueError("invalid measured training duration")
        elapsed = {key: sum(p[key] for p in profiles) for key in PHASES}
        if sum(elapsed.values()) > wall:
            raise ValueError("reported phase times exceed wall time")
        summaries.append({
            "backend": run["backend"],
            "samples": sum(p["samples"] for p in profiles),
            "batches": sum(p["batches"] for p in profiles),
            "wall_s": wall / 1e9,
            "phase_s": {key: elapsed[key] / 1e9 for key in PHASES},
            "percent": {key: 100 * elapsed[key] / wall for key in PHASES},
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="SHA256-verified benchmark-result.json")
    args = parser.parse_args()
    profiles = summarize(json.loads(args.report.read_text()))
    print("Training path profile (wall-clock Instant; instrumentation enabled)")
    for profile in profiles:
        print(f"\\n{profile['backend']}: {profile['wall_s']:.3f}s; "
              f"{profile['samples']} samples, {profile['batches']} batches")
        for key in PHASES:
            print(f"  {key:<19} {profile['phase_s'][key]:>9.3f}s"
                  f" {profile['percent'][key]:>6.2f}%")
    print("\\nNote: per-sample FP32 calls vs per-batch integer calls; "
          "this is high-level attribution, not CPU-cycle or energy profiling.")


if __name__ == "__main__":
    main()
