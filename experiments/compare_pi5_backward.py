#!/usr/bin/env python3
"""Compare paired physical Pi 5 per-phase training reports, without assuming a win."""
import argparse
import json
from pathlib import Path

FIELDS = ("dataset_manifest_sha256",)
RUN_FIELDS = ("backend", "architecture", "architecture_kind", "seed",
              "split_seed", "train_size", "validation_size", "test_size",
              "batch_size", "epochs", "lr_shift", "quantization_shift",
              "momentum_shift", "grad_clip_mantissa", "validation_only",
              "test_evaluated", "build_options", "rustc")


def compare(baseline: dict, candidate: dict) -> list[dict]:
    for key in FIELDS:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"mismatched report protocol: {key}")
    first = baseline["runs"]
    second = candidate["runs"]
    if [v["backend"] for v in first] != ["integer", "fp32"] or [
            v["backend"] for v in second] != ["integer", "fp32"]:
        raise ValueError("expected integer followed by scalar FP32 in both reports")
    output = []
    for before, after in zip(first, second):
        for key in RUN_FIELDS:
            if before.get(key) != after.get(key):
                raise ValueError(f"mismatched run protocol: {key}")
        if not before["validation_only"] or before["test_evaluated"]:
            raise ValueError("refusing test-set evaluation comparisons")
        for key in ("architecture", "hardware_model", "kernel", "cpu_limit",
                    "memory_limit_bytes"):
            machine_key = "architecture" if key == "architecture" else key
            if before["machine"].get(machine_key) != after["machine"].get(machine_key):
                raise ValueError(f"mismatched host configuration: {key}")
        phases = ("training_wall_ns", "backward_ns", "forward_ns",
                  "update_ns", "preparation_ns")
        def totals(record):
            profiles = [e.get("training_profile") for e in record["learning_curve"]]
            if any(not isinstance(p, dict) for p in profiles):
                raise ValueError("need --profile-training JSON for both builds")
            return {key: sum(p[key] for p in profiles) for key in phases}
        old = totals(before)
        new = totals(after)
        metrics = {}
        for key in phases:
            if old[key] <= 0:
                raise ValueError(f"non-positive reference timing {key}")
            metrics[key] = {
                "baseline_s": old[key] / 1e9,
                "candidate_s": new[key] / 1e9,
                "change_pct": 100 * (new[key] - old[key]) / old[key],
            }
        output.append({
            "backend": before["backend"],
            "timings": metrics,
            "validation_before_pct": before["learning_curve"][-1]["validation_accuracy_pct"],
            "validation_after_pct": after["learning_curve"][-1]["validation_accuracy_pct"],
            "peak_process_rss_before_kb": before["peak_rss_kb"],
            "peak_process_rss_after_kb": after["peak_rss_kb"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text())
    after = json.loads(args.candidate.read_text())
    for result in compare(before, after):
        print(f"\\n{result['backend']}:")
        for phase, values in result["timings"].items():
            print(f"  {phase:<20}: {values['baseline_s']:>8.3f} -> "
                  f"{values['candidate_s']:>8.3f} s ({values['change_pct']:+.2f}%)")
        print(f"  Final validation: {result['validation_before_pct']:.3f}% -> "
              f"{result['validation_after_pct']:.3f}%")
        print(f"  Process RSS: {result['peak_process_rss_before_kb']} -> "
              f"{result['peak_process_rss_after_kb']} kB")
    print("\\nNegative time change is faster. CPU throttling, frequency and host "
          "contention require independent measurement; one pair does not "
          "establish an optimization win.")


if __name__ == "__main__":
    main()
