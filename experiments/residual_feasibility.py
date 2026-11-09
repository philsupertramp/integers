#!/usr/bin/env python3
"""Bounded four-arm validation-only MNIST residual-path feasibility for #39.

This one-seed exploratory control checks that independently trained plain and
residual models (integer and FP32) execute and produce valid learning curves.
It is NOT a speed comparison, convergence gate, or ResNet-20 benchmark.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ARCHS = (
    ("plain", "cnn-plain-control", "mnist_cnn_plain_control_v1"),
    ("residual", "cnn-residual-control", "mnist_cnn_residual_control_v1"),
)
BACKENDS = ("integer", "fp32")
REQUIRED_DATA = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(record, backend, arch_kind, seed, epochs, train, validation):
    assert record["schema"] == 1
    assert record["dataset"] == "mnist"
    assert record["backend"] == backend
    assert record["architecture_kind"] == arch_kind
    assert record["seed"] == seed
    assert record["epochs"] == epochs and record["batch_size"] == 32
    assert record["lr_shift"] == 7 and record["quantization_shift"] == 7
    assert record["train_size"] == record["train_limit"] == train
    assert record["validation_size"] == record["validation_limit"] == validation
    assert record["test_size"] == 0 and record["validation_only"] is True
    assert record["test_evaluated"] is False and record["test_correct"] is None
    assert record["test_accuracy_pct"] is None
    assert len(record["learning_curve"]) == epochs
    for epoch_index, epoch in enumerate(record["learning_curve"], 1):
        assert epoch["epoch"] == epoch_index
        assert epoch["training_samples"] == train
        assert 0 <= epoch["validation_accuracy_pct"] <= 100
        assert 0 <= epoch["train_accuracy_pct"] <= 100
        assert len(epoch["validation_confusion"]) == 10
        assert all(len(row) == 10 for row in epoch["validation_confusion"])
        assert sum(map(sum, epoch["validation_confusion"])) == validation
        assert sum(epoch["validation_predictions"]) == validation
        assert epoch["nonfinite_outputs"] == 0 and epoch["extreme_outputs"] == 0
        if backend == "fp32":
            assert epoch["nonfinite_fp32_parameters"] == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("target/release/examples/held_out"))
    p.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=1024)
    p.add_argument("--validation-limit", type=int, default=256)
    args = p.parse_args()
    if not (args.seed > 0 and 1 <= args.epochs <= 10
            and 32 <= args.train_limit <= 55000
            and 10 <= args.validation_limit <= 5000):
        p.error("invalid fixed-batch, development-only feasibility budget")
    if not args.binary.is_file():
        p.error("build the held_out release example before running")
    if any(not (args.data_dir / name).is_file() for name in REQUIRED_DATA):
        p.error("missing train-only verified MNIST IDX; run fetch_mnist.py --train-only")
    if any((args.data_dir / name).exists() for name in
           ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")):
        p.error("official MNIST test files must not exist for this study")
    # Refuse an existing output directory. Never replace original evidence.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "design": "one-seed four-arm validation-only residual feasibility",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "binary_sha256": sha256(args.binary),
        "data_sha256": {name: sha256(args.data_dir / name) for name in REQUIRED_DATA},
        "seed": args.seed, "epochs": args.epochs,
        "train": args.train_limit, "validation": args.validation_limit,
        "batch_size": 32, "lr_shift": 7,
        "runs": [
            {"architecture": label, "backend": backend, "report": f"{label}-{backend}.json"}
            for label, _, _ in ARCHS for backend in BACKENDS
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    for label, cli_arch, arch_kind in ARCHS:
        for backend in BACKENDS:
            name = f"{label}-{backend}"
            report = args.out_dir / f"{name}.json"
            cmd = [
                str(args.binary.resolve()), "--dataset", "mnist",
                "--architecture", cli_arch, "--backend", backend,
                "--data-dir", str(args.data_dir.resolve()), "--validation-only",
                "--seed", str(args.seed), "--epochs", str(args.epochs),
                "--train-limit", str(args.train_limit),
                "--validation-limit", str(args.validation_limit),
                "--batch-size", "32", "--lr-shift", "7",
                "--report", str(report.resolve()),
            ]
            print(f"Running {name}", flush=True)
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, check=False)
            (args.out_dir / f"{name}.log").write_text(proc.stdout)
            if proc.returncode:
                raise RuntimeError(f"{name} failed (exit {proc.returncode}); log preserved")
            record = json.loads(report.read_text())
            verify(record, backend, arch_kind, args.seed, args.epochs,
                   args.train_limit, args.validation_limit)
            summaries.append({
                "arm": name,
                "validation_accuracy_pct": record["learning_curve"][-1]["validation_accuracy_pct"],
                "training_accuracy_pct": record["learning_curve"][-1]["train_accuracy_pct"],
                "training_ms": sum(e["training_ms"] for e in record["learning_curve"]),
                "report_sha256": sha256(report),
                "log_sha256": sha256(args.out_dir / f"{name}.log"),
            })
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Residual path: one-seed development feasibility (#39)", "",
        f"Source: \`{manifest['source_commit']}\`; seed: {args.seed}; "
        f"epochs: {args.epochs}; development train/validation: "
        f"{args.train_limit}/{args.validation_limit}; batch 32; LR shift 7.", "",
        "| Architecture | Backend | Final train accuracy | Final validation accuracy | Training time (diagnostic only) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        architecture, backend = row["arm"].split("-", 1)
        lines.append(
            f"| {architecture} | {backend} | {row['training_accuracy_pct']:.2f}% | "
            f"{row['validation_accuracy_pct']:.2f}% | {row['training_ms']} ms |")
    lines += [
        "", "**Interpretation:** this one-seed, small-sample run proves only that "
        "the four training/reporting paths execute without reported instability. "
        "It does not establish residual superiority, depth recovery or a valid "
        "CPU speed comparison. The official test set was never accessed.",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, RuntimeError, KeyError) as exc:
        sys.exit(f"Residual feasibility failed closed: {exc}")
