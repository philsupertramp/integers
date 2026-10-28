"""Offline tests for mirrored, physically pinned Pi5 A/B experiment."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import pi5_interleaved_ab as ab
from pi5_benchmark import emit_result_archive
from test_pi5_seed_sweep import record


def write_report(path, report):
    raw = (json.dumps(report) + "\n").encode()
    path.write_bytes(raw)
    stream = io.StringIO()
    emit_result_archive(raw, stream=stream)
    path.with_suffix(".log").write_text(stream.getvalue())


def fixture(plan, trial, version=None):
    arm = trial["arm"]
    seed = trial["seed"]
    image = plan["images"][arm]
    runs = []
    for backend in ("integer", "fp32"):
        run = copy.deepcopy(record(backend, seed, 0, image))
        run["indexed_job_seeds"] = [seed]
        run["git_commit_sha"] = plan["commits"][arm]
        run["dataset_manifest_sha256"] = ab.DATASET_SHA
        run["epochs"] = 2
        run["train_size"] = 8192
        run["validation_size"] = 1042
        epochs = []
        for _ in range(2):
            epoch = copy.deepcopy(run["learning_curve"][0])
            epoch["validation_accuracy_pct"] = 79.0 if backend == "integer" else 80.0
            epoch["nonfinite_outputs"] = 0
            epoch["extreme_outputs"] = 0
            epoch["nonfinite_fp32_parameters"] = 0 if backend == "fp32" else None
            profile = epoch["training_profile"]
            profile["samples"] = 8192
            profile["batches"] = 256
            if backend == "integer":
                # Forward unchanged, first Linear backward 20% faster.
                if arm == "B":
                    profile["backward_ns"] = 400
                    profile["training_wall_ns"] = 900
                    profile["backward_layers"][0]["elapsed_ns"] = 350
            epochs.append(epoch)
        run["learning_curve"] = epochs
        runs.append(run)
    return {"schema_version": 1, "dataset_manifest_sha256": ab.DATASET_SHA,
            "runs": runs}


def populated(root):
    plan = ab.build_plan("abba-test")
    ab.save_plan(root, plan)
    for trial in plan["trials"]:
        path = ab.trial_path(root, trial)
        path.mkdir(parents=True)
        write_report(path / f"seed-{trial['seed']}.json", fixture(plan, trial))
        (path / "trial.json").write_text(json.dumps({
            "job": trial["job"], "seed": trial["seed"], "arm": trial["arm"],
            "node": "pi5-local", "node_uid": "node-uid",
            "pod_uid": f"pod-{trial['index']}", "job_uid": f"job-{trial['index']}",
        }))
    return plan


class InterleavedTests(unittest.TestCase):
    def test_plan_is_deterministic_mirrored_and_never_overwritten(self):
        plan = ab.build_plan("abba-01")
        self.assertEqual(len(plan["trials"]), 20)
        for block in range(5):
            trials = plan["trials"][block * 4:(block+1) * 4]
            self.assertEqual("".join(t["arm"] for t in trials),
                             "ABBA" if block % 2 == 0 else "BAAB")
            self.assertEqual(len(set(t["job"] for t in trials)), 4)
            self.assertTrue(all(t["seed"] == plan["seeds"][block] for t in trials))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ab"
            ab.save_plan(root, plan)
            self.assertEqual(ab.load_plan(root), plan)
            with self.assertRaises(FileExistsError):
                ab.save_plan(root, plan)
            damaged = copy.deepcopy(plan)
            damaged["acceptance"]["max_training_ratio_point"] = 1
            with self.assertRaisesRegex(ValueError, "modified"):
                ab.validate_plan(damaged)

    def test_rejects_invalid_plan_or_reused_image(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            ab.build_plan("ab", ab.IMAGE_A, ab.IMAGE_A)
        with self.assertRaisesRegex(ValueError, "five"):
            ab.build_plan("ab", seeds=[1337, 1337, 2027, 4242, 42069])
        with self.assertRaises(ValueError):
            ab.build_plan("A" * 60)
        with self.assertRaisesRegex(ValueError, "digest"):
            ab.build_plan("ab", image_a="latest")

    def test_existing_job_strictly_matches_pinned_fields(self):
        plan = ab.build_plan("ab")
        desired = ab.desired_job(plan, plan["trials"][0])
        ab.check_existing_job(copy.deepcopy(desired), desired)
        for modification in ("image", "cpu", "pod_labels"):
            tampered = copy.deepcopy(desired)
            if modification == "image":
                tampered["spec"]["template"]["spec"]["containers"][0]["image"] = ab.IMAGE_B
            elif modification == "cpu":
                tampered["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"] = "4"
            else:
                tampered["metadata"]["labels"]["integers.dev/benchmark-kind"] = "fake"
            with self.assertRaises(ValueError, msg=modification):
                ab.check_existing_job(tampered, desired)

    def test_paired_log_ratio_interval_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            populated(root)
            result = ab.analyze(root)
            self.assertEqual(result["trials"], 20)
            self.assertEqual(result["gate"]["status"], "pass")
            metric = result["paired_runtime_ratio_B_over_A"]["integer.training_wall_ns"]
            self.assertAlmostEqual(metric["geometric_ratio"], 0.9)
            self.assertAlmostEqual(metric["ci95_upper"], 0.9)
            self.assertAlmostEqual(
                result["paired_runtime_ratio_B_over_A"]["fp32.training_wall_ns"]["geometric_ratio"],
                1)
            self.assertEqual(result["validation_change_pp"]["mean"], 0)
            self.assertTrue((root / "comparison.md").exists())

    def test_missing_or_corrupted_trials_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            plan = populated(root)
            t = plan["trials"][0]
            path = ab.trial_path(root, t) / f"seed-{t['seed']}.json"
            original = path.read_text()
            data = json.loads(original)
            data["runs"][0]["test_evaluated"] = True
            write_report(path, data)
            with self.assertRaisesRegex(ValueError, "test_evaluated"):
                ab.analyze(root)
            self.assertFalse((root / "comparison.json").exists())
            write_report(path, json.loads(original))
            (ab.trial_path(root, t) / "trial.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                ab.analyze(root)

    def test_hardware_or_coverage_breaks_and_accuracy_noninferiority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            plan = populated(root)
            trial = plan["trials"][1]
            path = ab.trial_path(root, trial) / "trial.json"
            meta = json.loads(path.read_text())
            meta["node_uid"] = "different-machine"
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "multiple physical"):
                ab.analyze(root)
            meta["node_uid"] = "node-uid"
            path.write_text(json.dumps(meta))
            # Both B repetitions of seed 1337 become 5pp worse: speed remains
            # excellent, but quality gate must refuse the numerical regression.
            for trial in plan["trials"][:4]:
                if trial["arm"] != "B":
                    continue
                file = ab.trial_path(root, trial) / f"seed-{trial['seed']}.json"
                report = json.loads(file.read_text())
                report["runs"][0]["learning_curve"][-1]["validation_accuracy_pct"] = 74
                write_report(file, report)
            outcome = ab.analyze(root)
            self.assertEqual(outcome["gate"]["status"], "inconclusive_or_regressed")
            self.assertLess(outcome["validation_change_pp"]["worst_seed"], -3)

    def test_serial_orchestrator_resumes_successful_jobs_without_resubmitting(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            plan = ab.build_plan("resume-test")
            ab.save_plan(root, plan)
            observed = []

            def existing_job(_kubectl, _namespace, _kind, name):
                trial = next(t for t in plan["trials"] if t["job"] == name)
                job = ab.desired_job(plan, trial)
                job["metadata"]["uid"] = f"uid-{trial['index']}"
                job["status"] = {"succeeded": 1}
                return job

            def collect_stub(_root, _plan, trial, _kubectl, _live):
                observed.append((trial["arm"], trial["seed"]))

            with mock.patch.object(ab, "kubectl_json", side_effect=existing_job), \
                    mock.patch.object(ab, "collect_one", side_effect=collect_stub), \
                    mock.patch.object(ab, "analyze", return_value={"gate": {"status": "pass"}}), \
                    mock.patch.object(ab.subprocess, "run") as runner:
                ab.run_plan(root)
                runner.assert_not_called()
            self.assertEqual(observed, [(t["arm"], t["seed"]) for t in plan["trials"]])
            self.assertEqual(len(observed), 20)

    def test_new_job_fail_closed_when_cluster_wait_fails(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            plan = ab.build_plan("abort-test")
            ab.save_plan(root, plan)
            calls = []

            def get_job(_kubectl, _namespace, _kind, name):
                calls.append(name)
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(
                        1, ["kubectl", "get"], stderr="Error from server (NotFound)")
                job = ab.desired_job(plan, plan["trials"][0])
                job["metadata"]["uid"] = "uid-1"
                job["status"] = {"succeeded": 0}
                return job

            def invoke(cmd, **kwargs):
                if "create" in cmd:
                    return subprocess.CompletedProcess(cmd, 0)
                raise subprocess.CalledProcessError(1, cmd, stderr="deadline exceeded")

            with mock.patch.object(ab, "kubectl_json", side_effect=get_job), \
                    mock.patch.object(ab.subprocess, "run", side_effect=invoke) as runner, \
                    mock.patch.object(ab, "collect_one") as collect:
                with self.assertRaises(subprocess.CalledProcessError):
                    ab.run_plan(root)
                collect.assert_not_called()
                self.assertEqual(runner.call_count, 2)
                self.assertEqual(len(calls), 2)

    def test_independent_paired_t_interval_not_raw_seed_sd(self):
        result = ab.log_interval([0.75, 0.78, 0.80, 0.82, 0.85])
        self.assertLess(result["ci95_lower"], result["geometric_ratio"])
        self.assertGreater(result["ci95_upper"], result["geometric_ratio"])
        self.assertEqual(result["blocks"], 5)
        with self.assertRaisesRegex(ValueError, "five"):
            ab.log_interval([0.8] * 2)


if __name__ == "__main__":
    unittest.main()
