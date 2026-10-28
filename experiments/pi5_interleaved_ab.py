#!/usr/bin/env python3
"""Resumable, serial ABBA/BAAB experiment on one physical Raspberry Pi 5.

Uses the existing pinned-image single-seed Job and verified archive collector.
Does not create workloads during plan/analyze. Never overwrites a Kubernetes Job
or raw trial result, never evaluates official MNIST test data.
"""
import argparse
import datetime as dt
import json
import math
import statistics
import subprocess
from pathlib import Path

from collect_pi5_indexed import parse_seeds
from pi5_seed_sweep import (
    DEFAULT_SEEDS, DIGEST, MACHINE_FIELDS, PHASES, SAME_ACROSS_SEEDS,
    collect_sweep, kubectl_json, new_job, phase_totals, protocol, valid_name,
    verify_report,
)

IMAGE_A = ("ghcr.io/philsupertramp/integers-benchmark@sha256:"
           "2c8f32322dcb4916c38b09f3123fb4e0043a009f094881be99d395aa33c8adf9")
IMAGE_B = ("ghcr.io/philsupertramp/integers-benchmark@sha256:"
           "1345a53c6976736524894cf7bb07f7ff5cb8392e8afdcb14a374b4db8a699bcc")
COMMIT_A = "b0ce98372e500a6b621abc0007e46ca2861fc4ab"
COMMIT_B = "cb36399f7bbacb6239490a0ff8fad50c06347c1f"
DATASET_SHA = "8042b6e40973be57d479b8eb3673f39b89f1cdb11e05db3ab74fecd814023a5d"
T95_DF4 = 2.7764451051977987
# This experiment uses five independent seed blocks, each with 2 A and 2 B runs.
STANDARD_SEEDS = [1337, 2026, 2027, 4242, 42069]
METRICS = ("training_wall_ns", "forward_ns", "backward_ns", "update_ns")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_plan(run_id, image_a=IMAGE_A, image_b=IMAGE_B, seeds=STANDARD_SEEDS,
               namespace="default"):
    valid_name(run_id)
    if len(run_id) > 24:
        raise ValueError("run ID must be <=24 chars to keep Job names short")
    if not DIGEST.fullmatch(image_a) or not DIGEST.fullmatch(image_b):
        raise ValueError("both images must be immutable pinned benchmark digests")
    if image_a == image_b:
        raise ValueError("A and B images must differ")
    seeds = list(seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(not 0 < s < 1 << 64 for s in seeds):
        raise ValueError("experiment requires five distinct positive u64 seeds")
    if not namespace or not all(ch.isalnum() or ch == "-" for ch in namespace):
        raise ValueError("invalid namespace")
    trials = []
    for block, seed in enumerate(seeds):
        # ABBA and BAAB both put treatment observations symmetrically in time.
        # Reversing by seed protects against systematic first-arm bias.
        for letter in ("ABBA" if block % 2 == 0 else "BAAB"):
            number = len(trials)
            job = valid_name(f"pi5-{run_id}-{number:02d}")
            trials.append({"index": number, "block": block, "seed": seed,
                           "arm": letter, "job": job})
    return {
        "schema_version": 1, "design": "mirrored-ABBA-BAAB", "run_id": run_id,
        "namespace": namespace, "images": {"A": image_a, "B": image_b},
        "commits": {"A": COMMIT_A, "B": COMMIT_B},
        "dataset_manifest_sha256": DATASET_SHA, "seeds": seeds,
        "workload": {"epochs": 2, "train_limit": 8192,
                     "validation_limit": 1042, "batch_size": 32,
                     "lr_shift": 7},
        "trials": trials,
        "acceptance": {
            "max_training_ratio_point": 0.95,
            "max_training_ratio_ci_upper": 1.0,
            "min_mean_validation_change_pp": -1.0,
            "min_seed_validation_change_pp": -3.0,
            "max_fp32_control_drift_fraction": 0.05,
            "max_rss_growth_fraction": 0.10,
        },
    }


def validate_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("design") != "mirrored-ABBA-BAAB":
        raise ValueError("unsupported experiment plan")
    expected = build_plan(plan["run_id"], plan["images"]["A"],
                          plan["images"]["B"], plan["seeds"], plan["namespace"])
    if expected != plan:
        raise ValueError("experiment plan modified or acceptance gates changed")
    return plan


def save_plan(root, plan):
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def load_plan(root):
    return validate_plan(json.loads((root / "plan.json").read_text()))


def trial_path(root, trial):
    return root / "trials" / f"{trial['index']:02d}-{trial['arm']}-{trial['seed']}"


def desired_job(plan, trial):
    cfg = plan["workload"]
    return new_job(trial["job"], plan["images"][trial["arm"]],
                   [trial["seed"]], cfg["epochs"], cfg["train_limit"],
                   cfg["validation_limit"], cfg["batch_size"], cfg["lr_shift"])


def check_existing_job(live, desired):
    if live.get("metadata", {}).get("name") != desired["metadata"]["name"]:
        raise ValueError("wrong Kubernetes Job identity")
    # After Kubernetes admission, defaulted fields vary. Validate every
    # experimental field through the strict existing protocol parser.
    a, b = protocol(live), protocol(desired)
    if a != b:
        raise ValueError("existing Job differs from pinned experimental config")
    expected_labels = desired["metadata"]["labels"]
    if any(live["metadata"].get("labels", {}).get(k) != v
           for k, v in expected_labels.items()):
        raise ValueError("existing Job has different ownership labels")
    for field in ("resources", "securityContext", "volumeMounts", "imagePullPolicy"):
        if live["spec"]["template"]["spec"]["containers"][0].get(field) != (
                desired["spec"]["template"]["spec"]["containers"][0].get(field)):
            raise ValueError(f"existing Job changed container {field}")
    if live["spec"]["template"]["spec"].get("nodeSelector") != (
            desired["spec"]["template"]["spec"]["nodeSelector"]):
        raise ValueError("existing Job changed node placement")


def command_json(cmd):
    return json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                     check=True).stdout)


def node_identity(kubectl, namespace, job, uid):
    pods = command_json([kubectl, "-n", namespace, "get", "pods", "-l",
                         f"batch.kubernetes.io/job-name={job}", "-o", "json"])["items"]
    if len(pods) != 1:
        raise ValueError(f"expected exactly one Pod for Job {job}")
    pod = pods[0]
    if not any(ref.get("uid") == uid and ref.get("kind") == "Job"
               for ref in pod["metadata"].get("ownerReferences", [])):
        raise ValueError("benchmark Pod not owned by expected Job UID")
    if pod.get("status", {}).get("phase") != "Succeeded":
        raise ValueError("benchmark Pod was not successful")
    node = pod.get("spec", {}).get("nodeName")
    if not node:
        raise ValueError("missing physical node identity")
    node_info = command_json([kubectl, "get", "node", node, "-o", "json"])
    labels = node_info["metadata"].get("labels", {})
    if labels.get("benchmark.integers.dev/pi5") != "true":
        raise ValueError("Pod ran on node without explicit Pi5 label")
    return {"node": node, "node_uid": node_info["metadata"]["uid"],
            "pod": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "job_uid": uid}


def collect_one(root, plan, trial, kubectl, live):
    path = trial_path(root, trial)
    cfg = protocol(desired_job(plan, trial))
    if path.exists():
        raw = (path / f"seed-{trial['seed']}.json").read_bytes()
        verify_report(raw, trial["seed"], 0, cfg)
        meta = json.loads((path / "trial.json").read_text())
        if any(meta.get(k) != expected for k, expected in {
            "job": trial["job"], "arm": trial["arm"],
            "seed": trial["seed"], "node": meta.get("node"),
            "job_uid": live["metadata"]["uid"],
        }.items()):
            raise ValueError("existing local trial metadata mismatch")
        return
    # collect_sweep stages and atomically publishes its verified archives.
    collect_sweep(trial["job"], plan["namespace"], path, kubectl)
    node = node_identity(kubectl, plan["namespace"], trial["job"],
                         live["metadata"]["uid"])
    (path / "trial.json").write_text(json.dumps(
        {**node, "job": trial["job"], "arm": trial["arm"],
         "seed": trial["seed"], "collected_at_utc": utc_now()},
        indent=2, sort_keys=True) + "\n")


def run_plan(root, kubectl="kubectl", timeout_seconds=900):
    plan = load_plan(root)
    if timeout_seconds < 60 or timeout_seconds > 86400:
        raise ValueError("timeout must be 60..86400 seconds")
    for trial in plan["trials"]:
        desired = desired_job(plan, trial)
        name = trial["job"]
        try:
            live = kubectl_json(kubectl, plan["namespace"], "job", name)
        except subprocess.CalledProcessError as exc:
            # Only 404 is an acceptable reason to create; don't hide a
            # permission, connection, or transient API failure.
            if "NotFound" not in (exc.stderr or "") and "not found" not in (
                    exc.stderr or "").lower():
                raise
            subprocess.run(
                [kubectl, "-n", plan["namespace"], "create", "-f", "-"],
                input=json.dumps(desired), text=True, check=True)
            live = kubectl_json(kubectl, plan["namespace"], "job", name)
        check_existing_job(live, desired)
        # Refuse an unvalidated local result even if the K8s Job succeeded.
        path = trial_path(root, trial)
        if path.exists() and (path / "trial.json").exists():
            collect_one(root, plan, trial, kubectl, live)
            continue
        print(f"[{trial['index']+1}/20] {trial['arm']} seed={trial['seed']} job={name}", flush=True)
        if live.get("status", {}).get("succeeded") != 1:
            subprocess.run([kubectl, "-n", plan["namespace"], "wait",
                            "--for=condition=complete", f"job/{name}",
                            f"--timeout={timeout_seconds}s"], check=True)
            live = kubectl_json(kubectl, plan["namespace"], "job", name)
        if live.get("status", {}).get("succeeded") != 1:
            raise ValueError(f"Job {name} did not complete successfully")
        if path.exists():
            # Crash between atomic archive collection and metadata writing:
            # recover node metadata while retaining complete verified archives.
            cfg = protocol(desired)
            verify_report((path / f"seed-{trial['seed']}.json").read_bytes(),
                          trial["seed"], 0, cfg)
            node = node_identity(kubectl, plan["namespace"], name,
                                 live["metadata"]["uid"])
            (path / "trial.json").write_text(json.dumps(
                {**node, "job": name, "arm": trial["arm"],
                 "seed": trial["seed"], "collected_at_utc": utc_now()},
                indent=2, sort_keys=True) + "\n")
        else:
            collect_one(root, plan, trial, kubectl, live)
    return analyze(root)


def log_interval(rows):
    # One paired unit per independent seed (each arm averages 2 runs).
    if len(rows) != 5 or any(x <= 0 for x in rows):
        raise ValueError("five positive paired measurements required")
    logs = [math.log(x) for x in rows]
    center = statistics.mean(logs)
    half = T95_DF4 * statistics.stdev(logs) / math.sqrt(len(logs))
    return {"geometric_ratio": math.exp(center),
            "ci95_lower": math.exp(center - half),
            "ci95_upper": math.exp(center + half),
            "ratios_by_seed": rows, "blocks": len(rows),
            "method": "paired seed-block mean log ratio; two-sided t95 df=4"}


def mean_temperatures(run):
    values = []
    for field in ("machine", "machine_after_run"):
        sensors = run.get(field, {}).get("thermal", [])
        if isinstance(sensors, list):
            values.extend(s["celsius"] for s in sensors
                          if isinstance(s.get("celsius"), (int, float)))
    return statistics.mean(values) if values else None


def analyze(root):
    plan = load_plan(root)
    blocks = {seed: {"A": [], "B": []} for seed in plan["seeds"]}
    all_nodes = set()
    source_commits = {arm: set() for arm in "AB"}
    temperatures = {"A": [], "B": []}
    for trial in plan["trials"]:
        path = trial_path(root, trial)
        if not (path / "trial.json").exists():
            raise ValueError(f"incomplete trial: {trial['job']}")
        meta = json.loads((path / "trial.json").read_text())
        if any(meta.get(k) != v for k, v in {
            "arm": trial["arm"], "seed": trial["seed"],
            "job": trial["job"]}.items()):
            raise ValueError("trial metadata does not match immutable plan")
        if not meta.get("pod_uid") or not meta.get("job_uid") or not meta.get("node_uid"):
            raise ValueError("trial lacks verified Kubernetes identifiers")
        all_nodes.add((meta["node"], meta["node_uid"]))
        cfg = protocol(desired_job(plan, trial))
        report = verify_report(
            (path / f"seed-{trial['seed']}.json").read_bytes(),
            trial["seed"], 0, cfg)
        if report["dataset_manifest_sha256"] != plan["dataset_manifest_sha256"]:
            raise ValueError("wrong pinned dataset manifest")
        for run in report["runs"]:
            source_commits[trial["arm"]].add(run.get("git_commit_sha"))
            for epoch in run["learning_curve"]:
                if epoch.get("nonfinite_outputs") != 0 or epoch.get("extreme_outputs") != 0:
                    raise ValueError("nonfinite or extreme outputs detected")
                if run["backend"] == "fp32" and epoch.get("nonfinite_fp32_parameters") not in (0,):
                    raise ValueError("nonfinite FP32 parameters")
            # Record all available temperatures; frequency/energy are explicitly
            # unavailable in the immutable historical benchmark images.
            t = mean_temperatures(run)
            if t is not None and run["backend"] == "integer":
                temperatures[trial["arm"]].append(t)
        blocks[trial["seed"]][trial["arm"]].append(report)
    if len(all_nodes) != 1:
        raise ValueError("experiment ran on multiple physical Pi nodes")
    if source_commits != {"A": {plan["commits"]["A"]},
                          "B": {plan["commits"]["B"]}}:
        raise ValueError("image source commit differs from preregistered commit")
    rows = []
    ref = None
    for seed in plan["seeds"]:
        byarm = blocks[seed]
        if len(byarm["A"]) != 2 or len(byarm["B"]) != 2:
            raise ValueError(f"seed {seed}: expected two trials per arm")
        measurements = {}
        for arm in "AB":
            trials = byarm[arm]
            runs = [r for report in trials for r in report["runs"]]
            for r in runs:
                # Ignore differences by source/image/seed/index, but require
                # identical architecture, pinned input artifacts and machine.
                protocol_key = {k: r.get(k) for k in SAME_ACROSS_SEEDS
                                if k not in ("git_commit_sha", "image_reference",
                                             "indexed_job_seeds", "job_completion_index")}
                protocol_key.update({k: r["machine"].get(k) for k in MACHINE_FIELDS})
                if ref is None:
                    ref = protocol_key
                elif protocol_key != ref:
                    raise ValueError("incomparable runtime, build or dataset protocol")
            measurements[arm] = {}
            for backend in ("integer", "fp32"):
                matches = [r for r in runs if r["backend"] == backend]
                if len(matches) != 2:
                    raise ValueError("missing backend trial")
                measurements[arm][backend] = {
                    "phase_s": {k: statistics.mean(phase_totals(r)[k]
                                                 for r in matches) for k in PHASES},
                    "validation_pct": statistics.mean(
                        r["learning_curve"][-1]["validation_accuracy_pct"] for r in matches),
                    "peak_rss_kb": statistics.mean(r["peak_rss_kb"] for r in matches),
                }
                if backend == "integer":
                    measurements[arm][backend]["first_linear_backward_s"] = statistics.mean(
                        sum(epoch["training_profile"]["backward_layers"][0]["elapsed_ns"]
                            for epoch in r["learning_curve"]) / 1e9 for r in matches)
        rows.append({"seed": seed, "A": measurements["A"], "B": measurements["B"]})
    result = {
        "schema_version": 1, "design": plan["design"],
        "images": plan["images"], "node": list(all_nodes)[0][0],
        "node_uid": list(all_nodes)[0][1], "dataset_manifest_sha256": DATASET_SHA,
        "trials": len(plan["trials"]), "seed_blocks": len(rows),
        "per_seed": rows, "paired_runtime_ratio_B_over_A": {},
        "environment": {
            "temperature_c_mean": {k: statistics.mean(v) if v else None
                                   for k, v in temperatures.items()},
            "cpu_frequency": "not_measured", "cpu_throttling": "not_measured",
            "energy": "not_measured",
        },
    }
    for backend in ("integer", "fp32"):
        keys = list(METRICS)
        if backend == "integer":
            keys.append("first_linear_backward_s")
        for key in keys:
            ratios = []
            for row in rows:
                a, b = row["A"][backend], row["B"][backend]
                left = a["phase_s"][key] if key in METRICS else a[key]
                right = b["phase_s"][key] if key in METRICS else b[key]
                ratios.append(right / left)
            result["paired_runtime_ratio_B_over_A"][f"{backend}.{key}"] = log_interval(ratios)
    validation = [row["B"]["integer"]["validation_pct"] -
                  row["A"]["integer"]["validation_pct"] for row in rows]
    result["validation_change_pp"] = {
        "mean": statistics.mean(validation), "worst_seed": min(validation),
        "per_seed": validation}
    growth = max(row["B"]["integer"]["peak_rss_kb"] /
                 row["A"]["integer"]["peak_rss_kb"] - 1 for row in rows)
    result["max_integer_rss_growth_fraction"] = growth
    int_ratio = result["paired_runtime_ratio_B_over_A"]["integer.training_wall_ns"]
    fp_ratio = result["paired_runtime_ratio_B_over_A"]["fp32.training_wall_ns"]
    a = plan["acceptance"]
    reasons = []
    if int_ratio["geometric_ratio"] > a["max_training_ratio_point"]:
        reasons.append("integer training point estimate is below 5% improvement")
    if int_ratio["ci95_upper"] >= a["max_training_ratio_ci_upper"]:
        reasons.append("95% paired interval does not exclude zero speedup")
    if result["validation_change_pp"]["mean"] < a["min_mean_validation_change_pp"]:
        reasons.append("mean validation loss exceeds allowed 1pp")
    if result["validation_change_pp"]["worst_seed"] < a["min_seed_validation_change_pp"]:
        reasons.append("at least one seed loses more than 3pp validation")
    if abs(fp_ratio["geometric_ratio"] - 1) > a["max_fp32_control_drift_fraction"]:
        reasons.append("FP32 negative-control drift exceeds 5%")
    if growth > a["max_rss_growth_fraction"]:
        reasons.append("peak RSS growth exceeds 10%")
    result["gate"] = {"status": "pass" if not reasons else "inconclusive_or_regressed",
                      "reasons": reasons, "criteria": a}
    (root / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md = ["# Interleaved physical Pi 5 A/B experiment", "",
          f"Design: {plan['design']}; five seeds × ABBA/BAAB = 20 serial Jobs.",
          f"Hardware: {result['node']} (UID {result['node_uid']}); 2-core quota.",
          f"A: {plan['images']['A']}",
          f"B: {plan['images']['B']}", "",
          "| Metric | B/A geometric ratio | 95% paired seed-block CI | Reduction |",
          "| --- | ---: | ---: | ---: |"]
    for key, item in result["paired_runtime_ratio_B_over_A"].items():
        md.append(f"| {key} | {item['geometric_ratio']:.4f} | "
                  f"[{item['ci95_lower']:.4f}, {item['ci95_upper']:.4f}] | "
                  f"{(1-item['geometric_ratio'])*100:.2f}% |")
    md += ["", f"Validation mean change: {result['validation_change_pp']['mean']:+.3f}pp; "
           f"worst seed {result['validation_change_pp']['worst_seed']:+.3f}pp.",
           f"Max integer peak-RSS growth: {100*growth:+.2f}%.",
           f"CPU temperatures (captured at report boundaries): {result['environment']['temperature_c_mean']}.",
           f"**Prespecified gate: {result['gate']['status']}**.",
           f"Reasons: {'; '.join(reasons) if reasons else 'none'}.", "",
           "The 95% t interval summarizes five seed blocks; it is **not** "
           "a hardware-randomization guarantee. Residual scheduling drift, "
           "frequency scaling and throttling are unmeasured. "
           "FP32 is a drift control, not a substitute for device telemetry.",
           "Official MNIST test set was not evaluated.", ""]
    (root / "comparison.md").write_text("\n".join(md))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan_cmd = sub.add_parser("plan", help="write immutable 20-trial plan, no kubectl")
    plan_cmd.add_argument("--dir", required=True, type=Path)
    plan_cmd.add_argument("--run-id", required=True)
    plan_cmd.add_argument("--image-a", default=IMAGE_A)
    plan_cmd.add_argument("--image-b", default=IMAGE_B)
    plan_cmd.add_argument("--seeds", default=DEFAULT_SEEDS)
    plan_cmd.add_argument("--namespace", default="default")
    execute = sub.add_parser("run", help="sequentially create/wait/collect 20 Jobs")
    execute.add_argument("--dir", required=True, type=Path)
    execute.add_argument("--kubectl", default="kubectl")
    execute.add_argument("--timeout-seconds", type=int, default=900)
    analyze_cmd = sub.add_parser("analyze", help="offline verification + paired report")
    analyze_cmd.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            p = build_plan(args.run_id, args.image_a, args.image_b,
                           parse_seeds(args.seeds), args.namespace)
            save_plan(args.dir, p)
            print(f"Wrote {len(p['trials'])} serial pinned-image trials to {args.dir / 'plan.json'}")
        else:
            result = (run_plan(args.dir, args.kubectl, args.timeout_seconds)
                      if args.action == "run" else analyze(args.dir))
            print((args.dir / "comparison.md").read_text())
            print(f"Gate: {result['gate']['status']}")
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Pi 5 A/B experiment failed closed: {exc}\n")


if __name__ == "__main__":
    main()
