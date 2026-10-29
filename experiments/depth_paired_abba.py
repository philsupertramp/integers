#!/usr/bin/env python3
"""Validation-only, paired integer/FP32 CNN depth study (issue #38).

Local x86_64 feasibility comes first; the ABBA/BAAB scheduling primitive and
checksum-verified log envelope are shared with Pi benchmark #55, but its
speedup acceptance thresholds are deliberately NOT reused. Each source binary
contains both numerical backends and runs each arm in an independent process.
"""
import argparse
import hashlib
import io
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from pi5_benchmark import emit_result_archive, extract_result_archive
from pi5_interleaved_ab import mirrored_arms

DEFAULT_SEEDS = (1337, 2027, 4242)
DEPTHS = (4, 8)
DATA_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
BACKENDS = {"A": "integer", "B": "fp32"}
# First proof of learning, not a final accuracy benchmark or arithmetic oracle.
# Freeze before interpreting the 10-epoch exploratory run.
MIN_MEAN_VALIDATION_PCT = 80.0
MAX_MEAN_INTEGER_GAP_PP = 5.0


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_numbers(text: str, allowed=None) -> list[int]:
    values = [int(x) for x in text.split(",")]
    if not values or len(set(values)) != len(values) or any(v <= 0 for v in values):
        raise ValueError("unique positive integer values required")
    if allowed is not None and any(v not in allowed for v in values):
        raise ValueError(f"only {allowed} are supported")
    return values


def trial_order(block: int, single_pass: bool) -> str:
    """Use independent AB training runs for exploration; full mirrored ABBA for timing."""
    return "AB" if single_pass else mirrored_arms(block)


def build_plan(binary: Path, data_dir: Path, seeds: list[int],
               depths: list[int], epochs: int, train: int, validation: int,
               batch: int, lr_shift: int, single_pass: bool = False) -> dict:
    binary = binary.resolve(strict=True)
    data_dir = data_dir.resolve(strict=True)
    if not binary.is_file() or not all((data_dir / p).is_file() for p in DATA_FILES):
        raise ValueError("release held_out binary and canonical MNIST training IDX files required")
    if not seeds or len(seeds) > 16 or len(set(seeds)) != len(seeds) or any(
            not 0 < s < 1 << 64 for s in seeds):
        raise ValueError("use 1..16 unique positive u64 seeds")
    if not depths or len(set(depths)) != len(depths) or any(d not in DEPTHS for d in depths):
        raise ValueError("only distinct 4/8 depths allowed")
    if not (1 <= epochs <= 100 and 1 <= train <= 55000 and
            1 <= validation <= 5000 and 1 <= batch <= 1024 and 0 <= lr_shift <= 30):
        raise ValueError("invalid bounded validation-only workload")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(source_sha) != 40:
        raise ValueError("full source commit required")
    trials = []
    for depth in depths:
        for block, seed in enumerate(seeds):
            for arm in trial_order(block, single_pass):
                trials.append({
                    "index": len(trials), "depth": depth, "seed": seed,
                    "block": block, "arm": arm, "backend": BACKENDS[arm],
                })
    plan = {
        "schema": 1, "design": ("paired-depth-AB" if single_pass
                                 else "paired-depth-ABBA-BAAB"),
        "source_sha": source_sha,
        "binary": str(binary), "binary_sha256": file_digest(binary),
        "data_dir": str(data_dir),
        "dataset_sha256": {p: file_digest(data_dir / p) for p in DATA_FILES},
        "seeds": seeds, "depths": depths,
        "workload": {"epochs": epochs, "train": train, "validation": validation,
                     "batch": batch, "lr_shift": lr_shift},
        "trials": trials,
        "test_set": "never_loaded",
        "interpretation": "three or more seeds for learning; duplicated runs are timing controls only",
    }
    plan["integrity_sha256"] = digest(canonical(plan))
    return plan


def verify_plan(plan: dict):
    copy = dict(plan)
    checksum = copy.pop("integrity_sha256")
    if digest(canonical(copy)) != checksum or plan.get("schema") != 1 or (
            plan.get("design") not in ("paired-depth-AB", "paired-depth-ABBA-BAAB")):
        raise ValueError("experiment plan was altered")
    expected = []
    single_pass = plan["design"] == "paired-depth-AB"
    for depth in plan["depths"]:
        if depth not in DEPTHS:
            raise ValueError("unsupported depth")
        for block, seed in enumerate(plan["seeds"]):
            for arm in trial_order(block, single_pass):
                expected.append({"index": len(expected), "depth": depth,
                                 "seed": seed, "block": block,
                                 "arm": arm, "backend": BACKENDS[arm]})
    if plan["trials"] != expected or plan["test_set"] != "never_loaded":
        raise ValueError("modified experiment order or test policy")
    return plan


def load_plan(root: Path) -> dict:
    return verify_plan(json.loads((root / "plan.json").read_text()))


def check_environment(plan: dict):
    binary = Path(plan["binary"])
    if not binary.is_file() or file_digest(binary) != plan["binary_sha256"]:
        raise ValueError("executable changed since experiment was frozen")
    data = Path(plan["data_dir"])
    for file, sha in plan["dataset_sha256"].items():
        if file not in DATA_FILES or file_digest(data / file) != sha:
            raise ValueError("MNIST development data changed since plan creation")


def trial_name(trial: dict) -> str:
    return f"{trial['index']:03d}-d{trial['depth']}-{trial['arm']}-s{trial['seed']}"


def validate_record(record: dict, trial: dict, plan: dict):
    w = plan["workload"]
    if (record.get("schema") != 1 or record.get("dataset") != "mnist"
            or record.get("seed") != trial["seed"]
            or record.get("backend") != trial["backend"]
            or record.get("architecture_kind") != f"mnist_cnn_{trial['depth']}conv_he_v2"
            or record.get("validation_only") is not True
            or record.get("test_evaluated") is not False
            or record.get("test_size") != 0
            or record.get("test_accuracy_pct") is not None
            or record.get("train_size") != w["train"]
            or record.get("validation_size") != w["validation"]
            or record.get("epochs") != w["epochs"]
            or record.get("batch_size") != w["batch"]
            or record.get("lr_shift") != w["lr_shift"]):
        raise ValueError(f"unmatched model, data, or test isolation for {trial_name(trial)}")
    curve = record.get("learning_curve")
    if not isinstance(curve, list) or len(curve) != w["epochs"]:
        raise ValueError("incomplete learning curve")
    for epoch_number, epoch in enumerate(curve, 1):
        confusion = epoch.get("validation_confusion")
        if (epoch.get("epoch") != epoch_number or
                epoch.get("training_samples") != w["train"] or
                not isinstance(confusion, list) or len(confusion) != 10 or
                any(not isinstance(row, list) or len(row) != 10 or
                    any(not isinstance(n, int) or n < 0 for n in row)
                    for row in confusion)):
            raise ValueError("invalid epoch/confusion shape")
        if sum(map(sum, confusion)) != w["validation"]:
            raise ValueError("confusion does not cover validation cohort")
        correct = sum(confusion[i][i] for i in range(10))
        if abs(epoch["validation_accuracy_pct"] - 100 * correct / w["validation"]) > 0.000002:
            raise ValueError("reported accuracy inconsistent with confusion")
        if epoch.get("validation_predictions") != [
                sum(confusion[i][j] for i in range(10)) for j in range(10)]:
            raise ValueError("predicted class totals disagree with confusion")
        if epoch.get("nonfinite_outputs") != 0 or epoch.get("extreme_outputs") != 0:
            raise ValueError("output numerical instability")
        if trial["backend"] == "fp32" and epoch.get("nonfinite_fp32_parameters") != 0:
            raise ValueError("FP32 parameter numerical instability")
        if not isinstance(epoch.get("training_ms"), (int, float)) or epoch["training_ms"] <= 0:
            raise ValueError("invalid training time")
    return record


def artifact_paths(root: Path, trial: dict):
    name = trial_name(trial)
    path = root / "trials"
    return path / f"{name}.json", path / f"{name}.log", path / f"{name}.meta.json"


def read_trial(root: Path, plan: dict, trial: dict):
    report, log, meta_file = artifact_paths(root, trial)
    if not all(path.is_file() for path in (report, log, meta_file)):
        raise FileNotFoundError(f"trial {trial_name(trial)} is missing or incomplete")
    raw = report.read_bytes()
    meta = json.loads(meta_file.read_text())
    if (meta.get("sha256") != digest(raw) or
            meta.get("index") != trial["index"] or
            meta.get("binary_sha256") != plan["binary_sha256"] or
            extract_result_archive(log.read_text()) != raw):
        raise ValueError(f"archived trial {trial_name(trial)} failed integrity checks")
    return validate_record(json.loads(raw), trial, plan)


def run(root: Path):
    plan = load_plan(root)
    check_environment(plan)
    (root / "trials").mkdir(exist_ok=True)
    for trial in plan["trials"]:
        paths = artifact_paths(root, trial)
        if any(path.exists() for path in paths):
            read_trial(root, plan, trial)  # existing complete result or fail closed
            print(f"reused verified {trial_name(trial)}", flush=True)
            continue
        w = plan["workload"]
        with tempfile.TemporaryDirectory(prefix="depth-trial-", dir=root) as tmp:
            report_path = Path(tmp) / "result.json"
            cmd = [
                plan["binary"], "--dataset", "mnist", "--backend", trial["backend"],
                "--architecture", f"cnn{trial['depth']}", "--validation-only",
                "--data-dir", plan["data_dir"], "--report", str(report_path),
                "--seed", str(trial["seed"]), "--epochs", str(w["epochs"]),
                "--batch-size", str(w["batch"]), "--lr-shift", str(w["lr_shift"]),
                "--train-limit", str(w["train"]), "--validation-limit", str(w["validation"]),
            ]
            print(f"[{trial['index'] + 1}/{len(plan['trials'])}] "
                  f"{trial_name(trial)}", flush=True)
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            raw = report_path.read_bytes()
            validate_record(json.loads(raw), trial, plan)
            stream = io.StringIO()
            stream.write(result.stdout)
            stream.write("\n")
            emit_result_archive(raw, stream=stream)
            metadata = {"index": trial["index"], "sha256": digest(raw),
                        "binary_sha256": plan["binary_sha256"],
                        "stdout_sha256": digest(result.stdout.encode())}
            # Immutable results: publish only after all three files are prepared.
            paths[0].write_bytes(raw)
            paths[1].write_text(stream.getvalue())
            paths[2].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            read_trial(root, plan, trial)


def analyze(root: Path):
    plan = load_plan(root)
    by_depth = {}
    for depth in plan["depths"]:
        per_seed = []
        for seed in plan["seeds"]:
            trials = [t for t in plan["trials"] if t["depth"] == depth and t["seed"] == seed]
            records = [(t, read_trial(root, plan, t)) for t in trials]
            by_backend = {
                backend: [r for t, r in records if t["backend"] == backend]
                for backend in BACKENDS.values()
            }
            for backend, values in by_backend.items():
                required = 1 if plan["design"] == "paired-depth-AB" else 2
                if len(values) != required:
                    raise ValueError(f"expected {required} run(s) for {backend}")
                # Re-running the same seeded algorithm must NOT change outcomes.
                curve = values[0]["learning_curve"]
                if any(
                    [(e["validation_accuracy_pct"], e["validation_confusion"],
                      e["train_accuracy_pct"]) for e in value["learning_curve"]]
                    != [(e["validation_accuracy_pct"], e["validation_confusion"],
                         e["train_accuracy_pct"]) for e in curve]
                    for value in values[1:]):
                    raise ValueError(f"non-reproducible training for {backend} seed {seed}")
            def last_acc(backend):
                return by_backend[backend][0]["learning_curve"][-1]["validation_accuracy_pct"]
            def mean_train_ms(backend):
                return statistics.mean(sum(e["training_ms"] for e in value["learning_curve"])
                                       for value in by_backend[backend])
            a, b = last_acc("integer"), last_acc("fp32")
            t_a, t_b = mean_train_ms("integer"), mean_train_ms("fp32")
            per_seed.append({
                "seed": seed, "integer_final_validation_pct": a,
                "fp32_final_validation_pct": b, "integer_minus_fp32_pp": a - b,
                "integer_training_ms": t_a, "fp32_training_ms": t_b,
                "integer_over_fp32_time": t_a / t_b,
                "integer_peak_rss_kb": max(
                    (r.get("peak_rss_kb") or 0) for r in by_backend["integer"]),
                "fp32_peak_rss_kb": max(
                    (r.get("peak_rss_kb") or 0) for r in by_backend["fp32"]),
                "integer_curve": by_backend["integer"][0]["learning_curve"],
                "fp32_curve": by_backend["fp32"][0]["learning_curve"],
            })
        mean_integer = statistics.mean(
            r["integer_final_validation_pct"] for r in per_seed)
        mean_fp32 = statistics.mean(
            r["fp32_final_validation_pct"] for r in per_seed)
        if mean_fp32 < MIN_MEAN_VALIDATION_PCT:
            quality_status = "inconclusive_fp32_not_learned"
        elif mean_integer < MIN_MEAN_VALIDATION_PCT:
            quality_status = "integer_below_learning_threshold"
        elif mean_fp32 - mean_integer > MAX_MEAN_INTEGER_GAP_PP:
            quality_status = "integer_accuracy_gap_exceeds_threshold"
        else:
            quality_status = "initial_quality_gate_pass"
        by_depth[str(depth)] = {
            "seeds": per_seed,
            "quality_status": quality_status,
            "learning_min_validation_pct": MIN_MEAN_VALIDATION_PCT,
            "max_mean_integer_gap_pp": MAX_MEAN_INTEGER_GAP_PP,
            "mean_integer_minus_fp32_pp": statistics.mean(
                r["integer_minus_fp32_pp"] for r in per_seed),
            "worst_integer_minus_fp32_pp": min(
                r["integer_minus_fp32_pp"] for r in per_seed),
            "mean_integer_validation_pct": statistics.mean(
                r["integer_final_validation_pct"] for r in per_seed),
            "mean_fp32_validation_pct": statistics.mean(
                r["fp32_final_validation_pct"] for r in per_seed),
            "mean_integer_over_fp32_time": statistics.mean(
                r["integer_over_fp32_time"] for r in per_seed),
        }
    output = {
        "schema": 1, "plan_sha256": plan["integrity_sha256"],
        "binary_sha256": plan["binary_sha256"],
        "source_sha": plan["source_sha"], "dataset_sha256": plan["dataset_sha256"],
        "workload": plan["workload"], "design": plan["design"], "depths": by_depth,
        "caveat": "ABBA repetitions control timing drift; only distinct seeds inform learning. "
                  "No Pi speed/energy claim; no official test evaluation; no library-wide "
                  "correctness claim; numerical equivalence not assumed.",
    }
    lines = ["# Paired 4/8-convolution-layer integer vs FP32 feasibility",
             "", "Validation-only development data; no official test evaluation.",
             ("Single-pass AB exploration: one run per backend per depth/seed."
              if plan["design"] == "paired-depth-AB" else
              "ABBA/BAAB repeated processes are timing controls, not additional seeds."), ""]
    for depth, summary in by_depth.items():
        lines += [f"## {depth} convolutions", "",
                  "| Seed | Integer validation | FP32 validation | Difference | Integer / FP32 training time |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for x in summary["seeds"]:
            lines.append(f"| {x['seed']} | {x['integer_final_validation_pct']:.3f}% | "
                         f"{x['fp32_final_validation_pct']:.3f}% | "
                         f"{x['integer_minus_fp32_pp']:+.3f} pp | "
                         f"{x['integer_over_fp32_time']:.2f}x |")
        lines += ["", f"**Exploratory learning verdict: {summary['quality_status']}** "
                  f"(both means ≥{MIN_MEAN_VALIDATION_PCT:g}%; "
                  f"integer deficit ≤{MAX_MEAN_INTEGER_GAP_PP:g} pp)",
                  f"Mean validation gap (integer - FP32): "
                  f"{summary['mean_integer_minus_fp32_pp']:+.3f} pp",
                  f"Mean integer/FP32 training-time ratio: "
                  f"{summary['mean_integer_over_fp32_time']:.2f}x", ""]
    lines += ["A passed initial accuracy gate only establishes learning on this development"
              " cohort. It does not prove gradient equivalence, statistical noninferiority,"
              " generalization on unseen test data or library-wide correctness.", "",
              "All per-epoch curves, predictions, confusion matrices, checksums and original "
              "reports are in summary.json and trials/. Detailed numerical diagnosis is "
              "reserved for #37 after reviewing these results.", ""]
    (root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (root / "summary.md").write_text("\n".join(lines))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("target/release/examples/held_out"))
    p.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--depths", default="4,8")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-limit", type=int, default=2048)
    p.add_argument("--validation-limit", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-shift", type=int, default=7)
    p.add_argument("--single-pass", action="store_true",
                   help="one matched integer/FP32 run per depth/seed for learning exploration; "
                        "full mirrored ABBA/BAAB remains the default for timing")
    for action in ("run", "analyze"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            plan = build_plan(args.binary, args.data_dir,
                              parse_numbers(args.seeds),
                              parse_numbers(args.depths, DEPTHS),
                              args.epochs, args.train_limit,
                              args.validation_limit, args.batch_size, args.lr_shift,
                              args.single_pass)
            args.dir.mkdir(parents=True, exist_ok=False)
            (args.dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            print(f"Planned {len(plan['trials'])} validation-only trials "
                  f"({plan['design']})")
        elif args.action == "run":
            run(args.dir)
        else:
            result = analyze(args.dir)
            print((args.dir / "summary.md").read_text())
            print(f"Verified {len(result['depths'])} depth comparisons")
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Depth study failed closed: {exc}\n")


if __name__ == "__main__":
    main()
