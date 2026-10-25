#!/usr/bin/env python3
"""Offline, container-ready launcher for the existing paired held_out executable.

One independent process per backend ensures VmHWM remains process-local.
No network, training code, dataset download, or test-set tuning lives here.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SPLIT_SEED = 81985529216486895


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_dataset(data_dir: Path, validation_only: bool = False) -> tuple[dict, str]:
    manifest_path = data_dir / "dataset-manifest.json"
    expected = (data_dir / "dataset-manifest.sha256").read_text().strip()
    actual = digest(manifest_path)
    if len(expected) != 64 or actual != expected:
        raise ValueError("dataset manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported dataset manifest schema")
    if manifest["splits"]["seed"] != SPLIT_SEED or manifest["splits"]["validation_per_class"] != 500:
        raise ValueError("unexpected MNIST split definition")
    if manifest["preprocessing"]["shift"] != 7:
        raise ValueError("unexpected MNIST input quantization")
    for name, meta in manifest["artifacts"].items():
        if name not in ("train-inputs.i8", "train-labels.u8", "test-inputs.i8", "test-labels.u8"):
            raise ValueError(f"unrecognized artifact: {name}")
        # Validation-only mode MUST NOT open or even hash official test files.
        if validation_only and name.startswith("test-"):
            continue
        path = data_dir / name
        if path.stat().st_size != meta["bytes"] or digest(path) != meta["sha256"]:
            raise ValueError(f"dataset checksum/size mismatch: {name}")
    if set(manifest["artifacts"]) != {"train-inputs.i8", "train-labels.u8", "test-inputs.i8", "test-labels.u8"}:
        raise ValueError("incomplete prepared MNIST")
    return manifest, actual


def try_text(path: str):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def machine_metadata() -> dict:
    model = try_text("/proc/device-tree/model")
    if model is not None:
        model = model.rstrip("\0")
    if not model:
        cpuinfo = try_text("/proc/cpuinfo") or ""
        model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                      if line.startswith(("Model\t", "model name", "Hardware\t"))), "not_measured")
    cgroup_cpu = try_text("/sys/fs/cgroup/cpu.max")
    cgroup_mem = try_text("/sys/fs/cgroup/memory.max")
    thermal = []
    for sensor in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            thermal.append({"sensor": sensor.parent.name,
                            "type": try_text(str(sensor.parent / "type")) or "unknown",
                            "celsius": int(sensor.read_text().strip()) / 1000.0})
        except (OSError, ValueError):
            continue
    return {
        "architecture": platform.machine(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "hardware_model": model,
        "cpu_limit": cgroup_cpu if cgroup_cpu is not None else "not_measured",
        "memory_limit_bytes": (
            int(cgroup_mem) if cgroup_mem is not None and cgroup_mem.isdigit()
            else "unlimited" if cgroup_mem == "max" else "not_measured"
        ),
        "thermal": thermal if thermal else "not_measured",
        "throttling": "not_measured",
        "cpu_frequency": "not_measured",
        "energy": "not_measured",
        "container_memory_peak": "not_measured",
        "process_rss_method": "Linux /proc/self/status VmHWM (per isolated training process; kB)",
    }


def run(args) -> dict:
    start = time.monotonic()
    manifest, manifest_hash = verify_dataset(args.data_dir, validation_only=args.validation_only)
    setup_ms = round((time.monotonic() - start) * 1000, 3)
    machine = machine_metadata()
    if args.require_arm64 and machine["architecture"] not in ("aarch64", "arm64"):
        raise RuntimeError(f"ARM64 required; found {machine['architecture']}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for backend in ("integer", "fp32"):
        out = args.results_dir / f"{backend}-seed-{args.seed}.json"
        command = [
            str(args.executable), "--dataset", "mnist", "--architecture", args.architecture,
            "--backend", backend, "--seed", str(args.seed), "--split-seed", str(SPLIT_SEED),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--lr-shift", str(args.lr_shift), "--data-dir", str(args.data_dir),
            "--prepared-data", "--report", str(out),
        ]
        if args.validation_only:
            command.append("--validation-only")
            command.extend(["--train-limit", str(args.train_limit),
                            "--validation-limit", str(args.validation_limit)])
        else:
            command.append("--final-test")
        print("$", " ".join(command), flush=True)
        run_start = time.monotonic()
        # Stream the existing runner's epoch-level progress through kubectl logs.
        completed = subprocess.run(command, check=True)
        process_ms = round((time.monotonic() - run_start) * 1000, 3)
        record = json.loads(out.read_text())
        if record["backend"] != backend or record["seed"] != args.seed:
            raise ValueError("runner output does not match request")
        record["benchmark_schema"] = 1
        record["git_commit_sha"] = os.environ.get("BENCHMARK_GIT_SHA", "unknown")
        record["image_reference"] = os.environ.get("BENCHMARK_IMAGE", "not_provided")
        record["rustc"] = os.environ.get("BENCHMARK_RUSTC") or try_text("/opt/build-info/rustc.txt") or "unknown"
        record["build_options"] = os.environ.get("BENCHMARK_BUILD_OPTIONS", "release; locked")
        record["dataset_manifest_sha256"] = manifest_hash
        record["dataset_identity"] = manifest["dataset"]
        record["dataset_splits"] = manifest["splits"]
        record["prepared_artifact_sha256"] = {k: v["sha256"] for k, v in manifest["artifacts"].items()}
        record["machine"] = machine
        record["machine_after_run"] = machine_metadata()
        record["entrypoint_setup_ms"] = setup_ms
        record["process_elapsed_ms"] = process_ms
        record["subprocess_exit_code"] = completed.returncode
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)
    first, second = records
    for key in ("seed", "split_seed", "epochs", "train_size", "validation_size", "test_size",
                "batch_size", "architecture_kind", "validation_only", "test_evaluated"):
        if first[key] != second[key]:
            raise ValueError(f"backends used different {key}")
    if not args.validation_only and (first["test_accuracy_pct"] is None or second["test_accuracy_pct"] is None):
        raise ValueError("explicit final test missing")
    if args.validation_only and (first["test_evaluated"] or second["test_evaluated"]):
        raise ValueError("test was evaluated in validation-only mode")
    summary = {
        "schema_version": 1,
        "dataset_manifest_sha256": manifest_hash,
        "entrypoint_setup_ms": setup_ms,
        "total_wall_ms_excluding_image_pull": round((time.monotonic() - start) * 1000, 3),
        "runs": records,
    }
    (args.results_dir / "benchmark-result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("BENCHMARK_RESULT=" + json.dumps({
        "dataset_manifest_sha256": manifest_hash,
        "result_file": str(args.results_dir / "benchmark-result.json"),
        "backends": [{ "backend": r["backend"], "validation_accuracy_pct": r["learning_curve"][-1]["validation_accuracy_pct"],
                       "test_accuracy_pct": r["test_accuracy_pct"], "wall_ms": r["wall_ms"],
                       "dataset_loading_ms": r["dataset_loading_ms"], "peak_rss_kb": r["peak_rss_kb"]}
                     for r in records],
    }, sort_keys=True), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("/usr/local/bin/held_out"))
    parser.add_argument("--data-dir", type=Path, default=Path("/opt/dataset"))
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    parser.add_argument("--architecture", choices=("dense", "cnn"), default="dense")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-shift", type=int, default=7)
    parser.add_argument("--validation-only", action="store_true", default=False)
    parser.add_argument("--final-test", action="store_true", default=False)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--require-arm64", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.validation_only == args.final_test:
        parser.error("choose exactly one of --validation-only or --final-test")
    if args.final_test and (args.train_limit is not None or args.validation_limit is not None):
        parser.error("sample limits require --validation-only")
    if args.validation_only:
        args.train_limit = 128 if args.train_limit is None else args.train_limit
        args.validation_limit = 128 if args.validation_limit is None else args.validation_limit
    if not (0 < args.epochs <= 1000 and args.seed > 0 and 0 < args.batch_size <= 65536
            and 0 <= args.lr_shift <= 30):
        parser.error("invalid budget/seed/batch/learning rate")
    if args.validation_only and (args.train_limit <= 0 or args.validation_limit <= 0):
        parser.error("sample limits must be positive")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
