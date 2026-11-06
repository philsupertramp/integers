#!/usr/bin/env python3
"""Submit, collect, and summarize reproducible multi-seed Pi 5 training profiles.

One Indexed Kubernetes Job schedules all seeds, serially by default (no
competing benchmark Pods on the same Pi). The collector verifies every Pod,
archive, and numerical/protocol record before saving any results. No official
MNIST test data is read, and existing Jobs are never overwritten.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from collect_pi5_indexed import indexed_pods, parse_seeds
from compare_pi5_backward import compare
from pi5_benchmark import extract_result_archive
from summarize_pi5_profile import summarize

DEFAULT_SEEDS = "1337,2026,2027,4242,42069"
DEFAULT_IMAGE = "ghcr.io/philsupertramp/integers-benchmark"
DIGEST = re.compile(r"^ghcr\.io/philsupertramp/integers-benchmark@sha256:[a-f0-9]{64}$")
JOB_NAME = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
PHASES = ("training_wall_ns", "backward_ns", "forward_ns", "update_ns",
          "preparation_ns", "zero_grad_ns", "loss_metrics_ns")
SAME_ACROSS_SEEDS = (
    "git_commit_sha", "image_reference", "build_options", "rustc",
    "dataset_manifest_sha256", "dataset_splits", "prepared_artifact_sha256",
    "split_seed", "epochs", "batch_size", "lr_shift", "architecture_kind",
    "architecture", "quantization_shift", "output_bits", "momentum_shift",
    "grad_clip_mantissa", "train_size", "validation_size", "validation_only",
    "test_evaluated", "profile_training",
)
MACHINE_FIELDS = ("architecture", "hardware_model", "kernel", "cpu_limit",
                  "memory_limit_bytes")


def valid_name(value: str) -> str:
    if len(value) > 63 or not JOB_NAME.fullmatch(value):
        raise ValueError("Job name must be DNS-label compatible (lowercase, <=63 chars)")
    return value


def parse_args_map(args: list[str]) -> dict[str, str | bool]:
    flags = {"--require-arm64", "--validation-only", "--profile-training"}
    keys = {"--architecture", "--seed-list", "--epochs", "--batch-size",
            "--lr-shift", "--train-limit", "--validation-limit"}
    values = {}
    pos = 0
    while pos < len(args):
        item = args[pos]
        if item in values or item not in flags | keys:
            raise ValueError(f"unexpected or duplicate benchmark argument: {item}")
        if item in flags:
            values[item] = True
            pos += 1
        else:
            if pos + 1 >= len(args):
                raise ValueError(f"missing benchmark argument value for {item}")
            values[item] = args[pos + 1]
            pos += 2
    if set(values) != flags | keys:
        raise ValueError("missing indexed profiled validation arguments")
    if values["--architecture"] != "dense":
        raise ValueError("only dense integer profiles are supported")
    return values


def new_job(job: str, image: str, seeds: list[int], epochs: int,
            train: int, validation: int, batch: int, lr_shift: int) -> dict:
    valid_name(job)
    if not DIGEST.fullmatch(image):
        raise ValueError("pin the published benchmark image by immutable sha256 digest")
    if not 1 <= len(seeds) <= 32 or len(set(seeds)) != len(seeds) or any(
            seed <= 0 or seed >= 1 << 64 for seed in seeds):
        raise ValueError("use 1-32 distinct positive u64 seeds")
    if not (1 <= epochs <= 1000 and 1 <= train <= 55000 and
            1 <= validation <= 5000 and 1 <= batch <= 65536 and
            0 <= lr_shift <= 30):
        raise ValueError("invalid bounded validation/profile configuration")
    seed_csv = ",".join(str(x) for x in seeds)
    args = ["--require-arm64", "--validation-only", "--profile-training",
            "--architecture", "dense", "--seed-list", seed_csv,
            "--epochs", str(epochs), "--batch-size", str(batch),
            "--lr-shift", str(lr_shift), "--train-limit", str(train),
            "--validation-limit", str(validation)]
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job, "labels": {
            "app.kubernetes.io/name": job,
            "integers.dev/benchmark-kind": "profile-seed-sweep"}},
        "spec": {
            "completionMode": "Indexed", "completions": len(seeds),
            "parallelism": 1, "backoffLimit": 0,
            # This deadline covers all sequential completions, not each Pod.
            "activeDeadlineSeconds": 3600 * len(seeds),
            "template": {"metadata": {"labels": {"app.kubernetes.io/name": job}},
                         "spec": {
                "restartPolicy": "Never",
                "nodeSelector": {
                    "kubernetes.io/arch": "arm64",
                    "benchmark.integers.dev/pi5": "true"},
                "securityContext": {
                    "runAsNonRoot": True, "runAsUser": 65532,
                    "runAsGroup": 65532, "fsGroup": 65532,
                    "seccompProfile": {"type": "RuntimeDefault"}},
                "containers": [{
                    "name": "benchmark", "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "env": [{"name": "BENCHMARK_IMAGE", "value": image}],
                    "args": args,
                    "resources": {
                        "requests": {"cpu": "1", "memory": "1Gi"},
                        "limits": {"cpu": "2", "memory": "3Gi"}},
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]}},
                    "volumeMounts": [{"name": "results", "mountPath": "/results"}]
                }],
                "volumes": [{"name": "results", "emptyDir": {}}]}}
        }
    }


def protocol(job: dict) -> dict:
    if job.get("kind") != "Job" or job.get("apiVersion") != "batch/v1":
        raise ValueError("not a Kubernetes Job")
    valid_name(job["metadata"]["name"])
    if job["metadata"].get("labels", {}).get("integers.dev/benchmark-kind") != "profile-seed-sweep":
        raise ValueError("Job is not a managed profiled seed sweep")
    spec = job["spec"]
    pods = spec["template"]["spec"]
    if (spec.get("completionMode") != "Indexed" or spec.get("parallelism") != 1
            or spec.get("backoffLimit") != 0):
        raise ValueError("refusing non-serial/non-Indexed benchmark Job")
    if pods.get("nodeSelector") != {
            "kubernetes.io/arch": "arm64", "benchmark.integers.dev/pi5": "true"}:
        raise ValueError("Job is not pinned to the labelled Pi 5")
    if len(pods["containers"]) != 1 or pods["containers"][0]["name"] != "benchmark":
        raise ValueError("unexpected benchmark container")
    c = pods["containers"][0]
    image = c["image"]
    if not DIGEST.fullmatch(image) or c.get("env") != [
            {"name": "BENCHMARK_IMAGE", "value": image}]:
        raise ValueError("benchmark image and recorded image digest differ")
    args = parse_args_map(c["args"])
    seeds = parse_seeds(args["--seed-list"])
    if spec["completions"] != len(seeds) or not 1 <= len(seeds) <= 32:
        raise ValueError("seed count does not match indexed completions")
    for key in ("--epochs", "--batch-size", "--lr-shift",
                "--train-limit", "--validation-limit"):
        if not re.fullmatch(r"[0-9]+", args[key]):
            raise ValueError(f"invalid integer argument {key}")
    config = {
        "seeds": seeds, "image": image,
        "epochs": int(args["--epochs"]),
        "batch_size": int(args["--batch-size"]),
        "lr_shift": int(args["--lr-shift"]),
        "train_size": int(args["--train-limit"]),
        "validation_size": int(args["--validation-limit"]),
    }
    # Reject invalid/tampered budgets by reusing submission validation.
    new_job(job["metadata"]["name"], image, seeds, config["epochs"],
            config["train_size"], config["validation_size"],
            config["batch_size"], config["lr_shift"])
    return config


def kubectl_json(kubectl: str, namespace: str, kind: str, name: str) -> dict:
    return json.loads(subprocess.run(
        [kubectl, "-n", namespace, "get", kind, name, "-o", "json"],
        check=True, capture_output=True, text=True).stdout)


def verify_report(raw: bytes, seed: int, index: int,
                  config: dict) -> dict:
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"seed {seed}: unsupported benchmark schema")
    runs = report.get("runs")
    if not isinstance(runs, list) or [r.get("backend") for r in runs] != ["integer", "fp32"]:
        raise ValueError(f"seed {seed}: expected integer then FP32 pair")
    for run in runs:
        for key, expected in {
            "seed": seed, "indexed_job_seeds": config["seeds"],
            "job_completion_index": index,
            "train_size": config["train_size"],
            "validation_size": config["validation_size"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "lr_shift": config["lr_shift"],
            "image_reference": config["image"],
            "validation_only": True, "profile_training": True,
            "test_evaluated": False, "test_size": 0,
            "test_accuracy_pct": None,
            "architecture_kind": "dense_mlp",
        }.items():
            if run.get(key) != expected:
                raise ValueError(f"seed {seed} {run['backend']}: wrong {key}")
        if report.get("dataset_manifest_sha256") != run.get("dataset_manifest_sha256"):
            raise ValueError(f"seed {seed}: inconsistent dataset manifest")
        machine = run.get("machine", {})
        if (machine.get("architecture") not in ("aarch64", "arm64") or
                "Raspberry Pi 5" not in str(machine.get("hardware_model", ""))):
            raise ValueError(f"seed {seed}: device was not verified as Pi 5")
        if len(run.get("learning_curve", [])) != config["epochs"]:
            raise ValueError(f"seed {seed}: missing training epochs")
    # Reuse existing phase/per-layer invariants; not merely checking existence.
    summarize(report)
    return report


def ensure_same_protocol(reports: list[dict]) -> None:
    reference = reports[0]["runs"][0]
    for report in reports:
        for run in report["runs"]:
            for key in SAME_ACROSS_SEEDS:
                if run.get(key) != reference.get(key):
                    raise ValueError(f"cross-seed protocol mismatch: {key}")
            for key in MACHINE_FIELDS:
                if run["machine"].get(key) != reference["machine"].get(key):
                    raise ValueError(f"cross-seed machine mismatch: {key}")


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
            "n": len(values)}


def phase_totals(run: dict) -> dict:
    return {key: sum(epoch["training_profile"][key]
                     for epoch in run["learning_curve"]) / 1e9
            for key in PHASES}


def summarize_sweep(reports: list[dict], seeds: list[int],
                    baselines: list[dict] | None = None) -> dict:
    ensure_same_protocol(reports)
    rows = []
    for index, (seed, report) in enumerate(zip(seeds, reports)):
        if report["runs"][0]["seed"] != seed:
            raise ValueError("incorrect report ordering for summary")
        row = {"seed": seed, "backends": {}}
        for run in report["runs"]:
            row["backends"][run["backend"]] = {
                "validation_accuracy_pct": run["learning_curve"][-1]["validation_accuracy_pct"],
                "peak_process_rss_kb": run["peak_rss_kb"],
                "phase_s": phase_totals(run),
            }
        row["layers"] = summarize(report)[0]["backward_layers"]
        if baselines is not None:
            matched = compare(baselines[index], report)
            row["compared"] = {entry["backend"]: entry for entry in matched}
        rows.append(row)
    grouped = {}
    for backend in ("integer", "fp32"):
        grouped[backend] = {
            "validation_accuracy_pct": stats([
                row["backends"][backend]["validation_accuracy_pct"] for row in rows]),
            "peak_process_rss_kb": stats([
                row["backends"][backend]["peak_process_rss_kb"] for row in rows]),
            "phase_s": {key: stats([
                row["backends"][backend]["phase_s"][key] for row in rows])
                for key in PHASES},
        }
    layer_keys = [(x["index"], x["name"]) for x in rows[0]["layers"]]
    if any([(x["index"], x["name"]) for x in row["layers"]] != layer_keys
           for row in rows):
        raise ValueError("different layer layouts across seeds")
    layers = [{
        "index": index, "name": name,
        "seconds": stats([row["layers"][j]["seconds"] for row in rows])
    } for j, (index, name) in enumerate(layer_keys)]
    summary = {
        "schema_version": 1, "seeds": seeds, "per_seed": rows,
        "aggregate": grouped, "backward_layers": layers,
        "image_reference": reports[0]["runs"][0]["image_reference"],
        "git_commit_sha": reports[0]["runs"][0]["git_commit_sha"],
        "rustc": reports[0]["runs"][0]["rustc"],
        "build_options": reports[0]["runs"][0]["build_options"],
        "machine": {key: reports[0]["runs"][0]["machine"].get(key)
                    for key in MACHINE_FIELDS},
        "dataset_manifest_sha256": reports[0]["dataset_manifest_sha256"],
    }
    if baselines is not None:
        summary["paired_baseline"] = {}
        for backend in ("integer", "fp32"):
            matched = [row["compared"][backend] for row in rows]
            summary["paired_baseline"][backend] = {
                "timing_change_pct": {
                    phase: stats([entry["timings"][phase]["change_pct"]
                                  for entry in matched])
                    for phase in ("training_wall_ns", "backward_ns",
                                  "forward_ns", "update_ns", "preparation_ns")
                },
                "validation_change_pp": stats([
                    entry["validation_after_pct"] - entry["validation_before_pct"]
                    for entry in matched]),
                "peak_rss_change_kb": stats([
                    entry["peak_process_rss_after_kb"]
                    - entry["peak_process_rss_before_kb"] for entry in matched]),
            }
    return summary


def summary_markdown(summary: dict) -> str:
    lines = [
        "# Pi 5 profile seed sweep",
        "",
        f"Image: {summary['image_reference']}",
        f"Git commit: {summary['git_commit_sha']}",
        f"Compiler: {summary['rustc']}",
        f"Build: {summary['build_options']}",
        f"Device: {summary['machine'].get('hardware_model')}",
        f"CPU quota: {summary['machine'].get('cpu_limit')}",
        f"Dataset manifest SHA256: {summary['dataset_manifest_sha256']}",
        f"Seeds: {', '.join(map(str, summary['seeds']))}",
        "",
        "| Seed | Integer train (s) | Integer backward (s) | FP32 train (s) | Integer val (%) | FP32 val (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_seed"]:
        i = row["backends"]["integer"]
        f = row["backends"]["fp32"]
        lines.append(
            f"| {row['seed']} | {i['phase_s']['training_wall_ns']:.3f} "
            f"| {i['phase_s']['backward_ns']:.3f} "
            f"| {f['phase_s']['training_wall_ns']:.3f} "
            f"| {i['validation_accuracy_pct']:.3f} "
            f"| {f['validation_accuracy_pct']:.3f} |")
    lines.extend(["", "## Mean ± sample SD across distinct initializations", ""])
    def mean_sd(metric, unit="", signed=False):
        format_code = "+.3f" if signed else ".3f"
        mean = format(metric["mean"], format_code)
        deviation = (f" ± {metric['sample_sd']:.3f}" if
                     metric["sample_sd"] is not None else " (SD n/a)")
        return f"{mean}{deviation}{unit}"

    for backend, group in summary["aggregate"].items():
        lines.append(
            f"- {backend}: train {mean_sd(group['phase_s']['training_wall_ns'], 's')}"
            f", backward {mean_sd(group['phase_s']['backward_ns'], 's')}"
            f", validation {mean_sd(group['validation_accuracy_pct'], '%')}"
            f", process peak RSS {mean_sd(group['peak_process_rss_kb'], ' kB')}"
            f" (n={len(summary['seeds'])})")
    lines.extend(["", "## Integer backward layers (mean time per run)", ""])
    for layer in summary["backward_layers"]:
        lines.append(f"- {layer['index']} {layer['name']}: "
                     f"{mean_sd(layer['seconds'], 's')}")
    if "paired_baseline" in summary:
        lines.extend(["", "## Same-seed paired changes vs baseline", ""])
        for backend, comparison in summary["paired_baseline"].items():
            for phase, values in comparison["timing_change_pct"].items():
                lines.append(f"- {backend} {phase}: {mean_sd(values, '%', signed=True)}"
                             " (negative means faster)")
            lines.append(f"- {backend} validation change: "
                         f"{mean_sd(comparison['validation_change_pp'], ' pp', signed=True)}")
            lines.append(f"- {backend} peak process RSS change: "
                         f"{mean_sd(comparison['peak_rss_change_kb'], ' kB', signed=True)}")
    lines.extend([
        "",
        "Timing is instrumented wall time, not CPU cycles or energy.",
        "Distinct seeds characterize initialization variability, not repeated A/B"
        " measurements of thermal or contention noise. Use interleaved repeated"
        " physical-Pi runs before claiming a statistically established speedup.",
        "Both backends use the same validation cohort; official test is untouched.",
        "",
    ])
    return "\n".join(lines)


def collect_sweep(job_name: str, namespace: str, out_dir: Path, kubectl: str,
                  baseline_dir: Path | None = None) -> dict:
    job = kubectl_json(kubectl, namespace, "job", job_name)
    config = protocol(job)
    if job["metadata"]["name"] != job_name:
        raise ValueError("returned a different Job")
    if job.get("status", {}).get("succeeded") != len(config["seeds"]):
        raise ValueError("Job is not fully completed: do not collect partial seed results")
    listing = subprocess.run(
        [kubectl, "-n", namespace, "get", "pods", "-l",
         f"batch.kubernetes.io/job-name={job_name}", "-o", "json"],
        check=True, capture_output=True, text=True)
    items = json.loads(listing.stdout)["items"]
    uid = job["metadata"].get("uid")
    if not uid or any(not any(
            owner.get("kind") == "Job" and owner.get("uid") == uid
            for owner in pod["metadata"].get("ownerReferences", []))
            for pod in items):
        raise ValueError("Pod ownership does not match requested Job")
    pod_names = indexed_pods(items, config["seeds"])
    archives = []
    for index, seed in enumerate(config["seeds"]):
        logs = subprocess.run(
            [kubectl, "-n", namespace, "logs", f"pod/{pod_names[index]}",
             "-c", "benchmark"],
            check=True, capture_output=True, text=True).stdout
        raw = extract_result_archive(logs)  # verifies chunks, length, SHA256
        report = verify_report(raw, seed, index, config)
        archives.append((seed, raw, logs, report))
    reports = [item[3] for item in archives]
    ensure_same_protocol(reports)
    baselines = None
    if baseline_dir is not None:
        baselines = []
        for seed in config["seeds"]:
            path = baseline_dir / f"seed-{seed}.json"
            baselines.append(json.loads(path.read_text()))
    summary = summarize_sweep(reports, config["seeds"], baselines)
    summary["job_name"] = job_name
    summary["namespace"] = namespace
    summary["configuration"] = config
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing results: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # No partial result sets: stage everything, then atomically rename directory.
    with tempfile.TemporaryDirectory(prefix=".pi5-sweep-", dir=out_dir.parent) as temp:
        stage = Path(temp)
        for seed, raw, logs, _ in archives:
            (stage / f"seed-{seed}.json").write_bytes(raw)
            (stage / f"seed-{seed}.log").write_text(logs)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (stage / "summary.md").write_text(summary_markdown(summary))
        stage.rename(out_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit", help="create serial Indexed Job; no overwrite")
    submit.add_argument("--job", required=True, type=valid_name)
    submit.add_argument("--namespace", default="default")
    submit.add_argument("--image", required=True, help="published immutable GHCR sha256 digest")
    submit.add_argument("--seeds", default=DEFAULT_SEEDS)
    submit.add_argument("--epochs", type=int, default=2)
    submit.add_argument("--train-limit", type=int, default=8192)
    submit.add_argument("--validation-limit", type=int, default=1042)
    submit.add_argument("--batch-size", type=int, default=32)
    submit.add_argument("--lr-shift", type=int, default=7)
    submit.add_argument("--dry-run", action="store_true", help="print Job JSON without kubectl")
    submit.add_argument("--kubectl", default="kubectl")
    fetch = sub.add_parser("collect", help="verify every Pod and summarize all seeds")
    fetch.add_argument("--job", required=True, type=valid_name)
    fetch.add_argument("--namespace", default="default")
    fetch.add_argument("--out-dir", type=Path, default=None)
    fetch.add_argument("--baseline-dir", type=Path, default=None)
    fetch.add_argument("--kubectl", default="kubectl")
    args = parser.parse_args(argv)
    try:
        if args.command == "submit":
            seeds = parse_seeds(args.seeds)
            manifest = new_job(args.job, args.image, seeds, args.epochs,
                               args.train_limit, args.validation_limit,
                               args.batch_size, args.lr_shift)
            serialized = json.dumps(manifest, indent=2) + "\n"
            if args.dry_run:
                print(serialized, end="")
            else:
                subprocess.run(
                    [args.kubectl, "-n", args.namespace, "create", "-f", "-"],
                    input=serialized, text=True, check=True)
                print(f"Scheduled {len(seeds)} seeds serially as Indexed Job {args.job}")
                print(f"After completion: python3 experiments/pi5_seed_sweep.py collect "
                      f"--job {args.job} --namespace {args.namespace}")
        else:
            directory = args.out_dir or Path("pi5-results") / args.job
            result = collect_sweep(args.job, args.namespace, directory,
                                   args.kubectl, args.baseline_dir)
            print(summary_markdown(result))
            print(f"Saved verified archives and summaries to {directory}")
    except (ValueError, KeyError, IndexError, TypeError, OSError,
            subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Pi 5 seed sweep failed: {exc}\n")


if __name__ == "__main__":
    main()
