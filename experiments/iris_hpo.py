#!/usr/bin/env python3
"""Iris HPO using only the 130 development examples (never read the test TSV).

A fixed deterministic grid is evaluated over multiple stratified validation
splits, independent model seeds, and both arithmetic backends. One training
run provides all predeclared epoch checkpoints. Candidate selection uses only
integer validation accuracy; FP32 is reported as a paired reference.
No extra Python libraries required.
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
SOURCE = ROOT / "data/iris_train.tsv"
DEFAULT_SPLITS = (0x0123456789ABCDEF, 0x0F1E2D3C4B5A6978, 0x5B8D9F1A2C3E4701)


def output(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-shifts", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[100, 200, 400])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 4242])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLITS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/iris_hpo")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/held_out")
    args = parser.parse_args()
    for name in ("lr_shifts", "batch_sizes", "checkpoints", "seeds", "split_seeds"):
        seq = getattr(args, name)
        if not seq or len(set(seq)) != len(seq) or any(value <= 0 for value in seq):
            parser.error(f"--{name.replace('_', '-')} must contain unique positive values")
    if any(value > 30 for value in args.lr_shifts):
        parser.error("lr shifts must fit the experiment's 0..30 contract")
    if max(args.split_seeds) >= 1 << 64:
        parser.error("split seeds must fit u64")
    return args


def run():
    args = parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"Missing {binary}; build with cargo build --locked --release --example held_out")
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    # This program NEVER opens, checksums, names as an input, or evaluates
    # the historical data/iris_test.tsv. The Rust validation-only mode
    # independently enforces the same boundary.
    root = args.output_dir.resolve()
    curves = root / "curves"
    curves.mkdir(parents=True, exist_ok=True)

    max_epoch = max(args.checkpoints)
    runs = []
    for batch in args.batch_sizes:
        for lr_shift in args.lr_shifts:
            for split_seed in args.split_seeds:
                for seed in args.seeds:
                    for backend in ("integer", "fp32"):
                        run_id = f"batch{batch}-lr{lr_shift}-split{split_seed}-seed{seed}-{backend}"
                        relative = f"curves/{run_id}.json"
                        path = root / relative
                        cmd = [
                            str(binary), "--dataset", "iris", "--validation-only",
                            "--backend", backend, "--seed", str(seed),
                            "--split-seed", str(split_seed),
                            "--epochs", str(max_epoch), "--batch-size", str(batch),
                            "--lr-shift", str(lr_shift), "--report", str(path),
                        ]
                        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
                        row = json.loads(path.read_text())
                        assert row["dataset"] == "iris" and row["backend"] == backend
                        assert row["validation_only"] is True
                        assert row["test_evaluated"] is False
                        assert row["test_correct"] is None and row["test_accuracy_pct"] is None
                        assert row["test_size"] == 0, "test file must not be loaded"
                        assert row["train_size"] == 106 and row["validation_size"] == 24
                        assert row["seed"] == seed and row["split_seed"] == split_seed
                        assert row["lr_shift"] == lr_shift and row["batch_size"] == batch
                        assert row["epochs"] == max_epoch
                        assert len(row["learning_curve"]) == max_epoch
                        assert all(epoch["training_samples"] == 106 for epoch in row["learning_curve"])
                        assert all(sum(epoch["validation_predictions"]) == 24 for epoch in row["learning_curve"])
                        for checkpoint in args.checkpoints:
                            score = row["learning_curve"][checkpoint - 1]["validation_accuracy_pct"]
                            runs.append({
                                "batch_size": batch, "lr_shift": lr_shift,
                                "epochs": checkpoint, "seed": seed,
                                "split_seed": split_seed, "backend": backend,
                                "validation_accuracy_pct": score, "curve": relative,
                            })
            print(f"Completed batch={batch}, lr_shift={lr_shift}", flush=True)

    candidates = []
    for batch in args.batch_sizes:
        for lr_shift in args.lr_shifts:
            for checkpoint in args.checkpoints:
                subset = [r for r in runs if (r["batch_size"], r["lr_shift"], r["epochs"])
                          == (batch, lr_shift, checkpoint)]
                groups = {
                    backend: [r["validation_accuracy_pct"] for r in subset if r["backend"] == backend]
                    for backend in ("integer", "fp32")
                }
                n = len(args.seeds) * len(args.split_seeds)
                assert len(groups["integer"]) == n and len(groups["fp32"]) == n
                candidates.append({
                    "batch_size": batch, "lr_shift": lr_shift, "epochs": checkpoint,
                    "integer_mean": statistics.mean(groups["integer"]),
                    "integer_min": min(groups["integer"]),
                    "integer_max": max(groups["integer"]),
                    "fp32_mean": statistics.mean(groups["fp32"]),
                    "fp32_min": min(groups["fp32"]),
                    "fp32_max": max(groups["fp32"]),
                    "observations_per_backend": n,
                })
    # Primary research target is the integer backend. Stable, predeclared
    # tie-breaks favor the more robust minimum, fewer epochs, fewer samples
    # per batch, and higher shift (lower learning rate). FP32 never chooses.
    ranked = sorted(candidates, key=lambda c: (
        -c["integer_mean"], -c["integer_min"],
        c["epochs"], c["batch_size"], -c["lr_shift"],
    ))
    selected = ranked[0]
    env = {
        "git_commit": output("git", "rev-parse", "HEAD"),
        "rustc": output("rustc", "--version"),
        "cargo": output("cargo", "--version"),
        "platform": platform.platform(), "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "development_source": "data/iris_train.tsv",
        "development_source_sha256": source_digest,
        "test_source_read": False,
    }
    result = {
        "schema": 1, "dataset": "iris", "objective": "mean integer validation accuracy",
        "selection_key": ["integer_mean DESC", "integer_min DESC",
                          "epochs ASC", "batch_size ASC", "lr_shift DESC"],
        "environment": env,
        "protocol": {"validation_per_class": 8, "train_size": 106, "validation_size": 24,
                     "split_seeds": args.split_seeds, "model_seeds": args.seeds,
                     "lr_shifts": args.lr_shifts, "batch_sizes": args.batch_sizes,
                     "epoch_checkpoints": args.checkpoints, "test_evaluated": False},
        "selected": selected, "ranking": ranked, "observations": runs,
    }
    (root / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Iris hyperparameter search — validation only",
        "",
        "**The historical test file was not read or evaluated.** No test accuracy is available.",
        f"Commit: \`{env['git_commit']}\` | Rust: \`{env['rustc']}\`",
        f"Source SHA-256: \`{source_digest}\` (130 development rows only).",
        f"{len(args.split_seeds)} fixed stratified splits × {len(args.seeds)} model seeds × "
        f"2 backends, 106 train / 24 validation per split.",
        "",
        "## Predeclared winner by mean integer validation accuracy",
        "",
        f"Batch **{selected['batch_size']}**, LR shift **{selected['lr_shift']}**, "
        f"epochs **{selected['epochs']}**.",
        f"Integer **{selected['integer_mean']:.2f}%** "
        f"[{selected['integer_min']:.2f}, {selected['integer_max']:.2f}]; "
        f"paired FP32 **{selected['fp32_mean']:.2f}%** "
        f"[{selected['fp32_min']:.2f}, {selected['fp32_max']:.2f}].",
        "",
        "## Ranked grid (all candidates)",
        "",
        "| Rank | Batch | LR shift | Epochs | Integer mean [min, max] | FP32 mean [min, max] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, c in enumerate(ranked, 1):
        lines.append(
            f"| {index} | {c['batch_size']} | {c['lr_shift']} | {c['epochs']} | "
            f"{c['integer_mean']:.2f}% [{c['integer_min']:.2f}, {c['integer_max']:.2f}] | "
            f"{c['fp32_mean']:.2f}% [{c['fp32_min']:.2f}, {c['fp32_max']:.2f}] |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The same small development pool is reused across candidates and across three",
        "predeclared stratified holdouts. Validation scores are selection evidence,",
        "not an independent estimate of future generalization. Individual model seeds",
        "and splits can overlap; do not treat the nine observations as independent",
        "datasets. The 20-row historical test set was reused in prior experiments,",
        "so a future one-off evaluation cannot make it pristine. This search does",
        "not authorize a new ADR-004 success threshold.",
        "",
        "Only the integer validation mean determined the winner; FP32 was reported",
        "in parallel and never used as an optimization objective or test selector.",
        "No tests, raw datasets, or checkpoints are included in these artifacts.",
        "Each per-run JSON contains the complete epoch-by-epoch learning curve.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:17]), flush=True)


if __name__ == "__main__":
    run()
