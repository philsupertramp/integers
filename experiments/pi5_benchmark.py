#!/usr/bin/env python3
"""Offline, container-ready launcher for the existing paired held_out executable.

One independent process per backend ensures VmHWM remains process-local.
No network, training code, dataset download, or test-set tuning lives here.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SPLIT_SEED = 81985529216486895

# CRI logging implementations can split very long lines. Bound each chunk
# below 2 KiB instead of printing an arbitrarily large JSON record in one line.
ARCHIVE_PREFIX = "BENCHMARK_ARCHIVE_V1_"
ARCHIVE_CHUNK_LENGTH = 1800
ARCHIVE_MAX_BYTES = 64 * 1024 * 1024


def emit_result_archive(json_bytes: bytes, stream=sys.stdout) -> None:
    """Write verified, independently recoverable, chunked JSON to pod logs.

    The Job's single container exits, so kubectl exec cannot read files from
    its ephemeral emptyDir after completion. Kubernetes logs remain readable
    while the Job Pod is retained. Each chunk is short enough for CRI logs.
    """
    if len(json_bytes) > ARCHIVE_MAX_BYTES:
        raise ValueError("benchmark JSON exceeds log archive size limit")
    # Assert we only archive a complete JSON document.
    json.loads(json_bytes)
    digest = hashlib.sha256(json_bytes).hexdigest()
    encoded = base64.b64encode(gzip.compress(json_bytes, mtime=0)).decode("ascii")
    chunks = [encoded[i:i + ARCHIVE_CHUNK_LENGTH]
              for i in range(0, len(encoded), ARCHIVE_CHUNK_LENGTH)]
    print(f"{ARCHIVE_PREFIX}BEGIN={len(json_bytes)}:{digest}:{len(chunks)}",
          file=stream, flush=True)
    for index, chunk in enumerate(chunks):
        print(f"{ARCHIVE_PREFIX}CHUNK={index}:{chunk}", file=stream)
    print(f"{ARCHIVE_PREFIX}END={digest}", file=stream, flush=True)


def extract_result_archive(log_lines: str) -> bytes:
    """Decode a complete archive from ordinary or --timestamps kubectl logs.

    Strictly verify chunk order/count, uncompressed size, SHA256, and JSON
    validity, rather than silently persisting incomplete/truncated results.
    """
    expected = None
    chunks = []
    recovered = None
    for line in log_lines.splitlines():
        offset = line.find(ARCHIVE_PREFIX)
        if offset < 0:
            continue
        record = line[offset + len(ARCHIVE_PREFIX):].strip()
        if record.startswith("BEGIN="):
            if expected is not None:
                raise ValueError("previous benchmark JSON archive is incomplete")
            fields = record.removeprefix("BEGIN=").split(":")
            if len(fields) != 3:
                raise ValueError("invalid JSON archive header")
            nbytes, sha256, count = fields
            expected = (int(nbytes), sha256, int(count))
            if not 0 <= expected[0] <= ARCHIVE_MAX_BYTES or expected[2] <= 0:
                raise ValueError("invalid JSON archive length/chunk count")
            chunks = []
        elif record.startswith("CHUNK="):
            if expected is None:
                raise ValueError("archive chunk without header")
            index, sep, data = record.removeprefix("CHUNK=").partition(":")
            if not sep or int(index) != len(chunks):
                raise ValueError("out-of-order or missing archive chunk")
            chunks.append(data)
            if len(chunks) > expected[2]:
                raise ValueError("unexpected extra archive chunk")
        elif record.startswith("END="):
            if expected is None:
                raise ValueError("archive trailer without header")
            size, sha256, count = expected
            if record.removeprefix("END=") != sha256 or len(chunks) != count:
                raise ValueError("archive trailer/checksum or chunk count mismatch")
            try:
                compressed = base64.b64decode("".join(chunks), validate=True)
                # Bound decompression output before accepting untrusted pod logs.
                import zlib
                decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
                content = decompressor.decompress(compressed, size + 1)
                content += decompressor.flush()
            except (ValueError, zlib.error) as exc:
                raise ValueError("invalid compressed JSON archive") from exc
            if len(content) != size or hashlib.sha256(content).hexdigest() != sha256:
                raise ValueError("JSON archive size or SHA256 mismatch")
            json.loads(content)
            recovered = content
            expected = None
            chunks = []
    if expected is not None:
        raise ValueError("incomplete benchmark JSON archive in logs")
    if recovered is None:
        raise ValueError("no complete benchmark JSON archive in logs")
    return recovered


def select_indexed_seed(seed_list: str, index: str | None) -> int:
    """Map a Kubernetes completion index to a unique predeclared seed.

    Fail closed if the controller-provided index is absent or invalid.
    """
    parts = seed_list.split(",")
    if not parts or any(not s.isascii() or not s.isdecimal() for s in parts):
        raise ValueError("--seed-list must contain positive decimal seeds separated by commas")
    seeds = [int(s) for s in parts]
    if (len(seeds) != len(set(seeds)) or
            any(not 0 < seed < (1 << 64) for seed in seeds)):
        raise ValueError("--seed-list must contain unique positive u64 seeds")
    if index is None or not index.isascii() or not index.isdecimal():
        raise ValueError("JOB_COMPLETION_INDEX must be a nonnegative decimal number")
    i = int(index)
    if i >= len(seeds):
        raise ValueError("JOB_COMPLETION_INDEX exceeds available seeds")
    return seeds[i]


def indexed_seed_metadata(args) -> dict:
    if args.seed_list is None:
        return {}
    return {
        "job_completion_index": int(os.environ["JOB_COMPLETION_INDEX"]),
        "indexed_job_seeds": [int(x) for x in args.seed_list.split(",")],
    }


PROFILE_TIMINGS = ("preparation_ns", "zero_grad_ns", "forward_ns",
                   "loss_metrics_ns", "backward_ns", "update_ns", "other_ns")


def verify_training_profile(record: dict, enabled: bool) -> None:
    """Reject partial/mislabeled profiles; preserve older unprofiled reports."""
    if enabled and record.get("profile_training") is not True:
        raise ValueError("Rust runner did not enable requested training profiling")
    if not enabled and record.get("profile_training", False):
        raise ValueError("unexpected training profiling in ordinary benchmark")
    for epoch in record["learning_curve"]:
        profile = epoch.get("training_profile")
        if not enabled:
            if profile is not None:
                raise ValueError("unexpected per-epoch training profile")
            continue
        if not isinstance(profile, dict):
            raise ValueError("training profile missing for epoch")
        if (profile.get("schema") != 1 or profile.get("unit") != "ns" or
                profile.get("clock") != "Instant" or
                profile.get("samples") != record["train_size"] or
                profile.get("batches") !=
                (record["train_size"] + record["batch_size"] - 1) // record["batch_size"]):
            raise ValueError("training profile metadata mismatch")
        layers = profile.get("backward_layers", [])
        if not isinstance(layers, list):
            raise ValueError("invalid backward layer profile array")
        if record["backend"] == "integer" and not layers:
            raise ValueError("missing integer backward layer attribution")
        seen = set()
        layer_total = 0
        for layer in layers:
            if not isinstance(layer, dict):
                raise ValueError("invalid backward layer entry")
            index = layer.get("index")
            nanoseconds = layer.get("elapsed_ns")
            if (not isinstance(index, int) or index in seen or index < 0 or
                    not isinstance(layer.get("name"), str) or not layer["name"] or
                    not isinstance(nanoseconds, int) or nanoseconds < 0):
                raise ValueError("invalid backward layer data")
            seen.add(index)
            layer_total += nanoseconds
        if seen != set(range(len(layers))) or layer_total > profile["backward_ns"]:
            raise ValueError("backward layer times do not reconcile")
        values = [profile.get(k) for k in PROFILE_TIMINGS]
        wall = profile.get("training_wall_ns")
        if (not isinstance(wall, int) or wall < 0 or
                any(not isinstance(v, int) or v < 0 for v in values) or
                sum(values) > wall or
                abs(wall // 1_000_000 - epoch["training_ms"]) > 1):
            raise ValueError("training profile timing totals inconsistent")


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
        if args.profile_training:
            command.append("--profile-training")
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
        verify_training_profile(record, args.profile_training)
        record["benchmark_schema"] = 1
        record.update(indexed_seed_metadata(args))
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
    # Emit the complete verified result before the process terminates. This is
    # the default durable retrieval path for a Job using ephemeral emptyDir.
    emit_result_archive((args.results_dir / "benchmark-result.json").read_bytes())
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("/usr/local/bin/held_out"))
    parser.add_argument("--data-dir", type=Path, default=Path("/opt/dataset"))
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    parser.add_argument("--architecture", choices=("dense", "cnn"), default="dense")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-list", type=str, default=None,
                        help="indexed Kubernetes Job seed list (JOB_COMPLETION_INDEX)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-shift", type=int, default=7)
    parser.add_argument("--validation-only", action="store_true", default=False)
    parser.add_argument("--final-test", action="store_true", default=False)
    parser.add_argument("--profile-training", action="store_true", default=False)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--require-arm64", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.seed_list is not None:
        if args.seed is not None:
            parser.error("choose --seed or --seed-list, not both")
        try:
            args.seed = select_indexed_seed(args.seed_list,
                                            os.environ.get("JOB_COMPLETION_INDEX"))
        except ValueError as exc:
            parser.error(str(exc))
    elif args.seed is None:
        args.seed = 1337
    if args.validation_only == args.final_test:
        parser.error("choose exactly one of --validation-only or --final-test")
    if args.profile_training and (not args.validation_only or args.architecture != "dense"):
        parser.error("--profile-training requires dense --validation-only")
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
