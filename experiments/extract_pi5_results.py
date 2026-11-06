#!/usr/bin/env python3
"""Restore and verify complete benchmark-result.json from a Kubernetes Job log.

Usage:
    kubectl logs job/integers-pi5-benchmark > pi5.log
    python3 experiments/extract_pi5_results.py pi5.log --output result.json
"""
import argparse
from pathlib import Path
from pi5_benchmark import extract_result_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="kubectl logs output (timestamps accepted)")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination benchmark-result.json")
    args = parser.parse_args()
    raw = extract_result_archive(args.log.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"Restored and SHA256-verified {len(raw)} bytes to {args.output}")


if __name__ == "__main__":
    main()
