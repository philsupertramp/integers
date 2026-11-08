#!/usr/bin/env python3
"""Seed-matched numerical convergence gate for experimental first-layer RNG elision.

Run *baseline and candidate immutable images* in a hosted ARM64 CI runner
to compare training behavior, not Pi performance. Independently evaluate per-epoch
validation accuracy and class recall with no official test set access.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

SEEDS = (1337, 2026, 2027, 4242, 42069)
MIN_MEAN_CHANGE_PP = -1.0
MIN_WORST_SEED_CHANGE_PP = -3.0


def verify_record(value, seed):
    if (value.get("schema") != 1 or value.get("backend") != "integer"
            or value.get("seed") != seed
            or value.get("validation_only") is not True
            or value.get("test_evaluated") is not False
            or value.get("test_size") != 0
            or value.get("architecture_kind") != "dense_mlp"):
        raise ValueError(f"seed {seed}: wrong training policy or test-set isolation")
    if value.get("test_accuracy_pct") is not None:
        raise ValueError("official test accuracy must be null")
    curve = value.get("learning_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("missing learning curve")
    if len(curve) != value.get("epochs"):
        raise ValueError("missing training epochs")
    for epoch_num, epoch in enumerate(curve, 1):
        if epoch.get("epoch") != epoch_num:
            raise ValueError("missing / out-of-order epoch")
        if epoch.get("nonfinite_outputs") != 0 or epoch.get("extreme_outputs") != 0:
            raise ValueError("numerical instability")
        confusion = epoch.get("validation_confusion")
        if (not isinstance(confusion, list) or len(confusion) != 10 or
                any(len(row) != 10 or any(not isinstance(v, int) or v < 0 for v in row)
                    for row in confusion)):
            raise ValueError("invalid confusion matrix")
        size = value["validation_size"]
        if sum(map(sum, confusion)) != size:
            raise ValueError("confusion matrix does not cover validation cohort")
        correct = sum(confusion[i][i] for i in range(10))
        if abs(epoch["validation_accuracy_pct"] - 100 * correct / size) > .000002:
            raise ValueError("validation accuracy and confusion disagree")
        expected_recall = [confusion[i][i] / sum(confusion[i])
                           if sum(confusion[i]) else None for i in range(10)]
        recalls = epoch.get("validation_per_class_recall")
        if (not isinstance(recalls, list) or len(recalls) != 10 or
                any((r is None) != (e is None) or
                    (r is not None and abs(r - e) > .000001)
                    for r, e in zip(recalls, expected_recall))):
            raise ValueError("per-class recall disagrees with confusion")
        predictions = epoch.get("validation_predictions")
        if predictions != [sum(confusion[i][j] for i in range(10)) for j in range(10)]:
            raise ValueError("predictions disagree with confusion")
        if epoch.get("training_samples") != value.get("train_size"):
            raise ValueError("training sample count mismatch")
    return value


def analyze(root, seeds=SEEDS):
    compared = []
    contract = None
    provenance_file = root / "provenance.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else None
    if provenance is not None:
        images = provenance.get("images", {})
        if not all(images.get(k) for k in ("A", "B")) or images["A"] == images["B"]:
            raise ValueError("A/B must use distinct recorded images")
    for seed in seeds:
        records = []
        for arm in "ab":
            path = root / f"seed-{seed}-{arm}.json"
            raw = path.read_bytes()
            if provenance is not None:
                expected = provenance.get("sha256_by_file", {}).get(path.name)
                if expected != hashlib.sha256(raw).hexdigest():
                    raise ValueError("A/B result artifact differs from recorded checksum")
            records.append(verify_record(json.loads(raw), seed))
        left, right = records
        immutable = ("dataset", "split_seed", "architecture", "train_size",
                     "validation_size", "epochs", "batch_size", "lr_shift",
                     "quantization_shift", "momentum_shift", "output_bits",
                     "grad_clip_mantissa", "profile_training")
        for record in (left, right):
            row = {key: record.get(key) for key in immutable}
            if contract is None:
                contract = row
            elif row != contract:
                raise ValueError("A/B training protocol mismatch")
        curves = []
        for a, b in zip(left["learning_curve"], right["learning_curve"]):
            curves.append({
                "epoch": a["epoch"],
                "a_train_pct": a["train_accuracy_pct"],
                "b_train_pct": b["train_accuracy_pct"],
                "a_val_pct": a["validation_accuracy_pct"],
                "b_val_pct": b["validation_accuracy_pct"],
                "validation_change_pp": b["validation_accuracy_pct"] - a["validation_accuracy_pct"],
                "recall_change_pp_by_class": [
                    None if p is None or q is None else 100 * (q - p)
                    for p, q in zip(a["validation_per_class_recall"],
                                    b["validation_per_class_recall"])
                ],
                "a_predictions": a["validation_predictions"],
                "b_predictions": b["validation_predictions"],
                "a_confusion": a["validation_confusion"],
                "b_confusion": b["validation_confusion"],
            })
        compared.append({"seed": seed, "epochs": curves,
                         "final_change_pp": curves[-1]["validation_change_pp"]})
    diffs = [row["final_change_pp"] for row in compared]
    result = {
        "schema_version": 1,
        "seeds": list(seeds),
        "protocol": contract,
        "per_seed": compared,
        "final_validation_change_pp": {
            "mean": statistics.mean(diffs),
            "worst": min(diffs),
            "per_seed": diffs,
        },
        "gate": {
            "mean_minimum_pp": MIN_MEAN_CHANGE_PP,
            "worst_seed_minimum_pp": MIN_WORST_SEED_CHANGE_PP,
            "status": ("pass" if statistics.mean(diffs) >= MIN_MEAN_CHANGE_PP
                       and min(diffs) >= MIN_WORST_SEED_CHANGE_PP else "regressed"),
        },
        "image_provenance": provenance["images"] if provenance else "unverified_offline_fixture",
        "note": "Two-image quality comparison, not physical Pi performance.",
    }
    (root / "quality.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Experimental no-draw RNG: matched-seed convergence", "",
             "No official MNIST test data evaluated. Distinct baseline/candidate binaries; integer backend only.",
             "", "| Seed | A final validation | B final validation | Change |",
             "| --- | ---: | ---: | ---: |"]
    for row in compared:
        e = row["epochs"][-1]
        lines.append(f"| {row['seed']} | {e['a_val_pct']:.3f}% | "
                     f"{e['b_val_pct']:.3f}% | {e['validation_change_pp']:+.3f}pp |")
    lines += ["", f"Mean change: {statistics.mean(diffs):+.3f}pp",
              f"Worst seed: {min(diffs):+.3f}pp",
              f"**Numerical quality gate: {result['gate']['status']}**.",
              "Detailed per-epoch convergence, per-class recall and confusion matrices "
              "are in quality.json.", ""]
    (root / "quality.md").write_text("\n".join(lines))
    return result


def run(image_a, image_b, root, docker="docker", train_size=8192,
        validation_size=1042, epochs=2):
    if image_a == image_b:
        raise ValueError("baseline and candidate images must differ")
    for image in (image_a, image_b):
        if not image or not image.isascii() or any(ch.isspace() for ch in image):
            raise ValueError("explicit, distinct baseline/candidate images required")
    if root.exists() and list(root.iterdir()):
        raise FileExistsError("refusing to overwrite existing convergence data")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o777)  # Docker image uses non-root benchmark user.
    checksums = {}
    for seed in SEEDS:
        for arm, image in (("a", image_a), ("b", image_b)):
            output = root / f"seed-{seed}-{arm}.json"
            command = [
                docker, "run", "--rm", "--network", "none", "--read-only",
                "--mount", f"type=bind,src={root.resolve()},dst=/results",
                "--entrypoint", "/usr/local/bin/held_out", image,
                "--dataset", "mnist", "--backend", "integer",
                "--architecture", "dense", "--seed", str(seed),
                "--epochs", str(epochs), "--batch-size", "32",
                "--lr-shift", "7", "--prepared-data",
                "--data-dir", "/opt/dataset",
                "--validation-only", "--profile-training",
                "--train-limit", str(train_size),
                "--validation-limit", str(validation_size),
                "--report", f"/results/{output.name}",
            ]
            print(f"seed {seed} image {arm}: running", flush=True)
            subprocess.run(command, check=True)
            verify_record(json.loads(output.read_text()), seed)
            checksums[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps(
        {"images": {"A": image_a, "B": image_b},
         "sha256_by_file": checksums,
         "mode": "two-image arm64 numerical-quality comparison",
         "seeds": SEEDS, "source": "Docker benchmark release images"},
        indent=2) + "\n")
    return analyze(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--image-a", required=True)
    run_cmd.add_argument("--image-b", required=True)
    run_cmd.add_argument("--out-dir", required=True, type=Path)
    run_cmd.add_argument("--docker", default="docker")
    run_cmd.add_argument("--train-size", type=int, default=8192)
    run_cmd.add_argument("--validation-size", type=int, default=1042)
    run_cmd.add_argument("--epochs", type=int, default=2)
    analyze_cmd = sub.add_parser("analyze")
    analyze_cmd.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = (run(args.image_a, args.image_b, args.out_dir, args.docker,
                      args.train_size, args.validation_size, args.epochs)
                  if args.command == "run" else analyze(args.out_dir))
        print((args.out_dir / "quality.md").read_text())
        if result["gate"]["status"] != "pass":
            parser.exit(2, "Experimental convergence quality gate FAILED\n")
    except (OSError, ValueError, KeyError, TypeError,
            subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Experimental convergence failed closed: {exc}\n")


if __name__ == "__main__":
    main()
