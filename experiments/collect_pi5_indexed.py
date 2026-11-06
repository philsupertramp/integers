#!/usr/bin/env python3
"""Collect all indexed Pi 5 paired results from completed Kubernetes Pod logs.

Usage: python3 experiments/collect_pi5_indexed.py --job integers-pi5-full-validation
"""
import argparse
import json
import subprocess
from pathlib import Path

from pi5_benchmark import extract_result_archive, select_indexed_seed


INDEX_ANNOTATION = "batch.kubernetes.io/job-completion-index"


def parse_seeds(value: str) -> list[int]:
    # Reuse strict seed validation used by the benchmark container.
    seeds = value.split(",")
    for i in range(len(seeds)):
        select_indexed_seed(value, str(i))
    return [int(s) for s in seeds]


def indexed_pods(items: list[dict], seeds: list[int]) -> dict[int, str]:
    pods = {}
    for pod in items:
        metadata = pod.get("metadata", {})
        index = metadata.get("annotations", {}).get(INDEX_ANNOTATION)
        if index is None:
            raise ValueError(f"missing index annotation: {metadata.get('name')}")
        i = int(index)
        if i < 0 or i >= len(seeds):
            raise ValueError(f"index outside seed list: {i}")
        if i in pods:
            raise ValueError(f"duplicate Pod for index {i}; examine Job status")
        if pod.get("status", {}).get("phase") != "Succeeded":
            raise ValueError(f"index {i} has not successfully completed")
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"missing Pod name for index {i}")
        pods[i] = name
    if set(pods) != set(range(len(seeds))):
        raise ValueError("missing successful indexed Job Pods")
    return pods


def collect(job: str, namespace: str, seeds: list[int], out_dir: Path,
            kubectl: str = "kubectl") -> list[Path]:
    listing = subprocess.run(
        [kubectl, "-n", namespace, "get", "pods",
         "-l", f"batch.kubernetes.io/job-name={job}", "-o", "json"],
        check=True, capture_output=True, text=True)
    items = json.loads(listing.stdout)["items"]
    names = indexed_pods(items, seeds)
    extracted = []
    # Decode and verify everything before writing ANY result files.
    for i, seed in enumerate(seeds):
        logs = subprocess.run(
            [kubectl, "-n", namespace, "logs", f"pod/{names[i]}", "-c", "benchmark"],
            check=True, capture_output=True, text=True).stdout
        raw = extract_result_archive(logs)
        result = json.loads(raw)
        records = result.get("runs", [])
        if len(records) != 2 or [r.get("backend") for r in records] != ["integer", "fp32"]:
            raise ValueError(f"invalid backend pair for seed {seed}")
        for record in records:
            if (record.get("seed") != seed or
                record.get("job_completion_index") != i or
                record.get("indexed_job_seeds") != seeds or
                not record.get("validation_only") or
                record.get("test_evaluated") is not False or
                record.get("test_accuracy_pct") is not None or
                record.get("train_size") != 55000 or
                record.get("validation_size") != 5000):
                raise ValueError(f"invalid indexed validation report for seed {seed}")
        extracted.append((seed, raw, logs))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed, raw, logs in extracted:
        path = out_dir / f"seed-{seed}.json"
        path.write_bytes(raw)
        (out_dir / f"seed-{seed}.log").write_text(logs)
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="integers-pi5-full-validation")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--seeds", default="1337,2027,4242")
    parser.add_argument("--out-dir", type=Path, default=Path("pi5-results/full-development"))
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    for path in collect(args.job, args.namespace, seeds, args.out_dir):
        print(f"Verified and saved {path}")


if __name__ == "__main__":
    main()
