#!/usr/bin/env python3
"""Run issue #15 paired integer/FP32 experiments and preserve their provenance.

No external Python packages required. Each backend/seed runs in a NEW process,
so Linux VmHWM measures its own peak RSS. No test-driven tuning is performed.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def command_output(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(dataset, data_dir, validation_only=False):
    if dataset == "iris":
        # Iris validation HPO never reads or even hashes historical test data.
        return ([ROOT / "data/iris_train.tsv"] if validation_only else
                [ROOT / "data/iris_train.tsv", ROOT / "data/iris_test.tsv"])
    # Validation-only MNIST requires precisely the two official TRAINING IDX
    # files: do not check existence of or hash the test inputs.
    names = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    if not validation_only:
        names += ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    return [data_dir / name for name in names]


def summarize(records, source, output_dir, metadata):
    grouped = {}
    for row in records:
        grouped.setdefault(row["backend"], []).append(row)
    lines = [
        "# Held-out experiment report",
        "",
        f"Dataset: **{records[0]['dataset']}**; smoke mode: **{records[0]['smoke']}**; "
        f"CI sample: **{records[0].get('ci_sample', False)}**.",
        f"Commit: `{metadata['git_commit']}`; Rust: `{metadata['rustc']}`.",
        f"Target: `{metadata['platform']}`; generated at `{metadata['timestamp_utc']}`.",
        "",
        "Source SHA-256 (each exact input file):",
        "",
    ]
    for path, digest in source.items():
        lines.append(f"- `{path}`: `{digest}`")
    ci_sample = records[0].get("ci_sample", False)
    validation_only = records[0].get("validation_only", False)
    validation_metric = ci_sample or validation_only
    lines.extend([
        "",
        ("## Validation-only results (test NOT loaded or evaluated)" if validation_only
         else "## CI sampled training diagnostics (official test NOT evaluated)" if ci_sample
         else "## Results (held-out test evaluated once per run, after fixed epochs)"),
        "",
        ("| Backend | Seeds | Final validation accuracy mean [min, max] | Wall time mean [min, max], s | Peak RSS mean [min, max], MiB |" if validation_metric
         else "| Backend | Seeds | Test accuracy mean [min, max] | Wall time mean [min, max], s | Peak RSS mean [min, max], MiB |"),
        "| --- | --- | --- | --- | --- |",
    ])
    for backend, runs in sorted(grouped.items()):
        def triplet(values, factor=1.0):
            if any(value is None for value in values):
                return "not available"
            values = [value / factor for value in values]
            return f"{statistics.mean(values):.2f} [{min(values):.2f}, {max(values):.2f}]"
        measured_accuracy = ([row["learning_curve"][-1]["validation_accuracy_pct"] for row in runs]
                             if validation_metric else [row["test_accuracy_pct"] for row in runs])
        lines.append(
            f"| {backend} | {', '.join(str(row['seed']) for row in runs)} | "
            f"{triplet(measured_accuracy)}% | "
            f"{triplet([row['wall_ms'] for row in runs], 1000)} | "
            f"{triplet([row['peak_rss_kb'] for row in runs], 1024)} |"
        )
    lines += [
        "",
        "Each seed/backend JSON contains its **full per-epoch learning curve**, split sizes,",
        ("fixed hyperparameters, an explicit unevaluated-test marker, and process-local wall time / peak RSS."
         if validation_metric else "fixed hyperparameters, final test result, and process-local wall time / peak RSS."),
        "Training accuracy is measured online (before parameter updates), while",
        ("validation is measured after each epoch; no test evaluation was performed."
         if validation_metric else "validation is measured after each epoch. Final test is evaluated only once."),
        "Validation is used for diagnostics; epochs and hyperparameters are specified",
        "in advance and never selected using test performance.",
        "",
        "**Comparison limits:** FP32 is a transparent scalar Rust reference, not an",
        "optimized BLAS/PyTorch implementation. It uses the identical dense architecture,",
        "starting weights, sample order, target and squared-error objective, power-of-two",
        "learning rate and momentum convention, but continuous arithmetic differs from",
        "dyadic quantization, stochastic rounding, and STE. Latency reflects kernels and",
        "implementation maturity, not intrinsic integer-versus-float hardware efficiency.",
    ]
    if validation_only:
        lines.extend(["", "**VALIDATION ONLY:** the test partition was not "
                      "opened, loaded, hashed, or evaluated. This score is used "
                      "for selection, not an independent generalization estimate."])
    if records[0]["dataset"] == "iris" and not validation_only:
        lines.extend([
            "",
            "**Iris caveat:** the repository's 20-row reserved test split has appeared in",
            "historical experiments. It is excluded from *this* run's training,",
            "preprocessing and selection, but cannot be called an unseen pristine test set.",
        ])
    if ci_sample:
        lines.extend(["", "**CI SAMPLE ONLY:** 640 train examples (64/class), 200 validation "
                      "examples (20/class), and NO test evaluation. This is a "
                      "bounded runtime/regression diagnostic, not evidence of full-MNIST "
                      "generalization, convergence, or accepted pass/fail thresholds."])
    if records[0]["smoke"]:
        lines.extend(["", "**SMOKE ONLY:** fixture-scale execution is not benchmark evidence."])
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n")
    output_dir.joinpath("manifest.json").write_text(json.dumps({
        "schema": 1, "environment": metadata, "sources_sha256": source,
        "runs": [f"{r['backend']}-seed-{r['seed']}.json" for r in records],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("iris", "mnist"), default="iris")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mnist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027, 4242])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-shift", type=int, default=None,
                        help="effective SGD learning rate = 2**(-lr_shift)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/held_out")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true",
                        help="Train/validate without ever opening or hashing test data (Iris/MNIST)")
    parser.add_argument("--ci-sample", action="store_true",
                        help="MNIST 640-train/200-validation CI diagnostic, never evaluate test")
    args = parser.parse_args()
    if args.lr_shift is None:
        args.lr_shift = 6 if args.dataset == "iris" else 7
    if args.validation_only and args.ci_sample:
        parser.error("--validation-only and --ci-sample cannot be combined")
    if args.ci_sample and (args.dataset != "mnist" or args.smoke):
        parser.error("--ci-sample requires --dataset mnist and excludes --smoke")
    if len(set(args.seeds)) != len(args.seeds) or any(s <= 0 for s in args.seeds):
        parser.error("seeds must be unique positive nonzero integers")
    if not 0 <= args.lr_shift <= 30:
        parser.error("--lr-shift must be between 0 and 30")
    if args.batch_size <= 0 or (args.epochs is not None and args.epochs <= 0):
        parser.error("batch size and epochs must be positive")
    if args.dataset == "mnist" and args.data_dir != ROOT / "data/mnist":
        args.data_dir = args.data_dir.resolve()
    paths = source_paths(args.dataset, args.data_dir, args.validation_only)
    for path in paths:
        if not path.is_file():
            parser.error(f"missing source: {path} (get MNIST IDX files first)")
    source = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
              file_sha256(path) for path in paths}
    profile = "validation-only" if args.validation_only else "ci-sample" if args.ci_sample else "smoke" if args.smoke else "full"
    output_dir = args.output_dir.resolve() / args.dataset / profile
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "rustc": command_output("rustc", "--version"),
        "cargo": command_output("cargo", "--version"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lr_shift": args.lr_shift,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    records = []
    for seed in args.seeds:
        for backend in ("integer", "fp32"):
            name = f"{backend}-seed-{seed}.json"
            path = output_dir / name
            cmd = [
                "cargo", "run", "--locked", "--release" if not args.smoke else "--quiet",
                "--example", "held_out", "--",
                "--dataset", args.dataset, "--backend", backend,
                "--seed", str(seed), "--batch-size", str(args.batch_size),
                "--lr-shift", str(args.lr_shift),
                "--data-dir", str(args.data_dir), "--report", str(path),
            ]
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.smoke:
                cmd.append("--smoke")
            if args.ci_sample:
                cmd.append("--ci-sample")
            if args.validation_only:
                cmd.append("--validation-only")
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            row = json.loads(path.read_text())
            assert row["seed"] == seed and row["backend"] == backend
            assert row["lr_shift"] == args.lr_shift
            assert row["test_size"] == (0 if args.validation_only else 20 if args.dataset == "iris" else 10000)
            assert row["ci_sample"] == args.ci_sample
            assert row["validation_only"] == args.validation_only
            assert row["test_evaluated"] == (not args.ci_sample and not args.validation_only)
            if args.validation_only:
                assert row["test_correct"] is None and row["test_accuracy_pct"] is None
                assert row["test_size"] == 0
                if args.dataset == "iris":
                    assert row["train_size"] == (12 if args.smoke else 106)
                    assert row["validation_size"] == (3 if args.smoke else 24)
                else:
                    assert row["train_size"] == (40 if args.smoke else 55000)
                    assert row["validation_size"] == (10 if args.smoke else 5000)
            if args.ci_sample:
                assert row["train_size"] == 640 and row["validation_size"] == 200
                assert row["test_correct"] is None and row["test_accuracy_pct"] is None
            assert len(row["learning_curve"]) == row["epochs"]
            assert all(sum(epoch["validation_predictions"]) == row["validation_size"]
                       for epoch in row["learning_curve"])
            records.append(row)
    summarize(records, source, output_dir, metadata)
    print(f"Complete: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
