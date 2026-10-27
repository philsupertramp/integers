"""Offline tests for serial Pi 5 profiled seed sweeps (no Kubernetes needed)."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import pi5_benchmark as runner
import pi5_seed_sweep as sweep


IMAGE = "ghcr.io/philsupertramp/integers-benchmark@sha256:" + "a" * 64
SEEDS = [1337, 2027, 4242]


def record(backend, seed, index, image=IMAGE):
    phases = {
        "preparation_ns": 20, "zero_grad_ns": 10, "forward_ns": 150,
        "loss_metrics_ns": 10, "backward_ns": 500, "update_ns": 100,
        "other_ns": 210,
    }
    profile = {
        "training_wall_ns": 1000, "samples": 8, "batches": 1,
        "backward_layers": (
            [{"index": 0, "name": "Linear", "elapsed_ns": 450}]
            if backend == "integer" else []),
        **phases,
    }
    return {
        "backend": backend, "seed": seed, "job_completion_index": index,
        "indexed_job_seeds": SEEDS, "epochs": 1, "batch_size": 32,
        "lr_shift": 7, "train_size": 8, "validation_size": 10,
        "test_size": 0, "validation_only": True, "test_evaluated": False,
        "test_accuracy_pct": None, "profile_training": True,
        "image_reference": image, "git_commit_sha": "deadbeef",
        "build_options": "release; locked", "rustc": "rustc 1.95",
        "architecture_kind": "dense_mlp", "architecture": [784, 128, 128, 10],
        "dataset_manifest_sha256": "manifest", "dataset_splits": {"seed": 7},
        "prepared_artifact_sha256": {"train": "hash"}, "split_seed": 7,
        "quantization_shift": 7, "output_bits": 31,
        "momentum_shift": 1, "grad_clip_mantissa": 8192,
        "machine": {"architecture": "aarch64", "hardware_model": "Raspberry Pi 5 Model B",
                    "kernel": "6.12", "cpu_limit": "200000 100000",
                    "memory_limit_bytes": 3221225472},
        "learning_curve": [{"validation_accuracy_pct": 75 + index,
                            "training_profile": profile}],
        "peak_rss_kb": 415000,
    }


def report(seed, index):
    return {"schema_version": 1, "dataset_manifest_sha256": "manifest",
            "runs": [record("integer", seed, index),
                     record("fp32", seed, index)]}


def logs_for(value):
    out = io.StringIO()
    runner.emit_result_archive((json.dumps(value) + "\n").encode(), stream=out)
    return out.getvalue()


def fake_cluster(monkey_reports=None, status=3, pod_phase="Succeeded"):
    job = sweep.new_job("integers-perf-sweep", IMAGE, SEEDS, 1, 8, 10, 32, 7)
    job["metadata"]["uid"] = "job-uid"
    job["status"] = {"succeeded": status}
    pods = [
        {"metadata": {
            "name": f"sweep-{i}", "annotations": {
                "batch.kubernetes.io/job-completion-index": str(i)},
            "ownerReferences": [{"kind": "Job", "uid": "job-uid"}]},
         "status": {"phase": pod_phase}}
        for i in range(3)]
    reports = monkey_reports if monkey_reports is not None else [
        report(seed, i) for i, seed in enumerate(SEEDS)]
    mapping = {f"pod/sweep-{i}": logs_for(r) for i, r in enumerate(reports)}

    def run(command, **kwargs):
        if "get" in command and "job" in command:
            output = json.dumps(job)
        elif "get" in command and "pods" in command:
            output = json.dumps({"items": pods})
        elif "logs" in command:
            output = mapping[command[4]]
        else:
            raise AssertionError(f"unexpected kubectl command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output)

    return run, job, pods


class Pi5SeedSweepTests(unittest.TestCase):
    def test_submit_strict_pinning_and_indexed_seriality(self):
        job = sweep.new_job("integers-perf-sweep", IMAGE, SEEDS, 2, 8192, 1042, 32, 7)
        spec = job["spec"]
        self.assertEqual((spec["completionMode"], spec["completions"],
                          spec["parallelism"], spec["backoffLimit"]),
                         ("Indexed", 3, 1, 0))
        self.assertNotIn("ttlSecondsAfterFinished", spec)
        self.assertEqual(sweep.protocol(job), {
            "seeds": SEEDS, "image": IMAGE, "epochs": 2,
            "batch_size": 32, "lr_shift": 7, "train_size": 8192,
            "validation_size": 1042,
        })
        self.assertEqual(
            spec["template"]["spec"]["nodeSelector"]["benchmark.integers.dev/pi5"],
            "true")
        self.assertEqual(spec["template"]["spec"]["containers"][0]["image"], IMAGE)
        for invalid in ("abc", "ghcr.io/philsupertramp/integers-benchmark:latest"):
            with self.assertRaisesRegex(ValueError, "digest"):
                sweep.new_job("integers-perf-sweep", invalid, SEEDS, 2,
                              8192, 1042, 32, 7)
        for invalid in ("A", "some/unsafe-name", "x" * 64):
            with self.assertRaises(ValueError):
                sweep.valid_name(invalid)
        for invalid in ([1337, 1337], [0], [1 << 64]):
            with self.assertRaises(ValueError):
                sweep.new_job("integers-perf-sweep", IMAGE, invalid, 2,
                              8192, 1042, 32, 7)

    def test_collect_rejects_modified_workload_and_incomplete_jobs(self):
        job = sweep.new_job("integers-perf-sweep", IMAGE, SEEDS, 1, 8, 10, 32, 7)
        job["spec"]["parallelism"] = 2
        with self.assertRaisesRegex(ValueError, "non-serial"):
            sweep.protocol(job)
        job["spec"]["parallelism"] = 1
        job["spec"]["template"]["spec"]["containers"][0]["args"].extend(
            ["--final-test"])
        with self.assertRaisesRegex(ValueError, "unexpected"):
            sweep.protocol(job)
        for state in (2, 0):
            command, _, _ = fake_cluster(status=state)
            with tempfile.TemporaryDirectory() as root:
                output = Path(root) / "results"
                with mock.patch.object(sweep.subprocess, "run", command):
                    with self.assertRaisesRegex(ValueError, "not fully completed"):
                        sweep.collect_sweep("integers-perf-sweep", "default", output,
                                            "kubectl")
                self.assertFalse(output.exists())

    def test_complete_collection_logs_and_summaries(self):
        command, _, _ = fake_cluster()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "results"
            with mock.patch.object(sweep.subprocess, "run", command):
                summary = sweep.collect_sweep("integers-perf-sweep", "default",
                                              output, "kubectl")
            self.assertEqual(summary["seeds"], SEEDS)
            self.assertEqual(len(summary["per_seed"]), 3)
            self.assertEqual(summary["backward_layers"][0]["seconds"]["mean"],
                             450 / 1e9)
            self.assertAlmostEqual(
                summary["aggregate"]["integer"]["validation_accuracy_pct"]["mean"],
                76)
            self.assertTrue((output / "summary.md").exists())
            self.assertTrue((output / "summary.json").exists())
            for seed in SEEDS:
                log = (output / f"seed-{seed}.log").read_text()
                restored = runner.extract_result_archive(log)
                self.assertEqual(restored, (output / f"seed-{seed}.json").read_bytes())
            with mock.patch.object(sweep.subprocess, "run", command):
                with self.assertRaises(FileExistsError):
                    sweep.collect_sweep("integers-perf-sweep", "default", output,
                                        "kubectl")

    def test_fail_closed_on_corruption_and_mismatched_seed(self):
        for changed in ("seed", "test_evaluated", "image_reference", "machine"):
            reports = [report(seed, i) for i, seed in enumerate(SEEDS)]
            if changed == "seed":
                reports[1]["runs"][0]["seed"] = 12
            elif changed == "test_evaluated":
                reports[1]["runs"][0]["test_evaluated"] = True
            elif changed == "image_reference":
                reports[1]["runs"][0]["image_reference"] = "fake"
            else:
                reports[1]["runs"][0]["machine"]["hardware_model"] = "hosted ARM64"
            command, _, _ = fake_cluster(reports)
            with tempfile.TemporaryDirectory() as root:
                output = Path(root) / "results"
                with mock.patch.object(sweep.subprocess, "run", command):
                    with self.assertRaises(ValueError, msg=changed):
                        sweep.collect_sweep("integers-perf-sweep", "default",
                                            output, "kubectl")
                self.assertFalse(output.exists())

    def test_paired_baseline_fails_on_missing_or_wrong_seed(self):
        reports = [report(seed, i) for i, seed in enumerate(SEEDS)]
        with tempfile.TemporaryDirectory() as root:
            baseline_dir = Path(root) / "baseline"
            baseline_dir.mkdir()
            for seed, data in zip(SEEDS, reports):
                (baseline_dir / f"seed-{seed}.json").write_text(json.dumps(data))
            command, _, _ = fake_cluster(reports)
            with mock.patch.object(sweep.subprocess, "run", command):
                result = sweep.collect_sweep("integers-perf-sweep", "default",
                                             Path(root) / "paired", "kubectl",
                                             baseline_dir)
            self.assertEqual(result["paired_baseline"]["integer"]
                             ["backward_ns"]["mean"], 0)
            (baseline_dir / "seed-2027.json").unlink()
            with mock.patch.object(sweep.subprocess, "run", command):
                with self.assertRaises(FileNotFoundError):
                    sweep.collect_sweep("integers-perf-sweep", "default",
                                        Path(root) / "failed", "kubectl",
                                        baseline_dir)
            self.assertFalse((Path(root) / "failed").exists())

    def test_dry_run_is_offline_and_submit_uses_create(self):
        buffer = io.StringIO()
        with mock.patch.object(sweep.subprocess, "run") as run:
            with redirect_stdout(buffer):
                sweep.main(["submit", "--job", "integers-perf-sweep",
                            "--image", IMAGE, "--dry-run", "--seeds",
                            "1337,2027,4242"])
            run.assert_not_called()
        self.assertEqual(json.loads(buffer.getvalue())["spec"]["completions"], 3)
        with mock.patch.object(sweep.subprocess, "run") as run:
            with redirect_stdout(io.StringIO()):
                sweep.main(["submit", "--job", "integers-perf-sweep",
                            "--image", IMAGE])
            self.assertEqual(run.call_args.args[0][-3:], ["create", "-f", "-"])


if __name__ == "__main__":
    unittest.main()
