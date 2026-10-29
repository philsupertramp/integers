"""Offline depth-study contract and ABBA/BAAB archive regression tests."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import depth_paired_abba as depth
from pi5_benchmark import emit_result_archive


def fixture(trial, plan, val=50.0):
    n = plan["workload"]["validation"]
    correct = int(n * val / 100)
    matrix = [[0] * 10 for _ in range(10)]
    matrix[0][0] = correct
    matrix[0][1] = n - correct
    epoch = {
        "epoch": 1, "training_samples": plan["workload"]["train"],
        "train_accuracy_pct": val, "validation_accuracy_pct": 100 * correct / n,
        "validation_confusion": matrix,
        "validation_predictions": [correct, n - correct] + [0] * 8,
        "nonfinite_outputs": 0, "extreme_outputs": 0,
        "nonfinite_fp32_parameters": 0 if trial["backend"] == "fp32" else None,
        "training_ms": 100 if trial["backend"] == "integer" else 80,
    }
    return {
        "schema": 1, "dataset": "mnist", "seed": trial["seed"],
        "backend": trial["backend"],
        "architecture_kind": f"mnist_cnn_{trial['depth']}conv_he_v2",
        "validation_only": True, "test_evaluated": False,
        "test_size": 0, "test_accuracy_pct": None,
        "train_size": plan["workload"]["train"], "validation_size": n,
        "epochs": 1, "batch_size": plan["workload"]["batch"],
        "lr_shift": plan["workload"]["lr_shift"], "learning_curve": [epoch],
    }


def archive(root, plan, trial, val=50.0):
    result, log, metadata = depth.artifact_paths(root, trial)
    result.parent.mkdir(parents=True, exist_ok=True)
    raw = depth.canonical(fixture(trial, plan, val))
    stream = io.StringIO()
    emit_result_archive(raw, stream=stream)
    result.write_bytes(raw)
    log.write_text(stream.getvalue())
    metadata.write_text(json.dumps({
        "index": trial["index"], "sha256": depth.digest(raw),
        "binary_sha256": plan["binary_sha256"]
    }))


class DepthPairedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binary = self.root / "held_out"
        self.binary.write_bytes(b"dummy executable")
        self.data = self.root / "data"
        self.data.mkdir()
        for name in depth.DATA_FILES:
            (self.data / name).write_bytes(name.encode())

    def plan(self, seeds=None, depths=None):
        with mock.patch.object(depth.subprocess, "check_output",
                               return_value="a" * 40 + "\n"):
            return depth.build_plan(self.binary, self.data,
                                    seeds or [1337, 2027, 4242],
                                    depths or [4, 8], 1, 20, 10, 2, 7)

    def test_mirrored_order_and_distinct_backends(self):
        plan = self.plan()
        self.assertEqual(len(plan["trials"]), 24)
        for d in [4, 8]:
            blocks = [t for t in plan["trials"] if t["depth"] == d]
            self.assertEqual(
                ["".join(t["arm"] for t in blocks[i:i+4]) for i in (0, 4, 8)],
                ["ABBA", "BAAB", "ABBA"])
            self.assertTrue(all(t["backend"] == depth.BACKENDS[t["arm"]] for t in blocks))
        depth.verify_plan(plan)
        altered = json.loads(json.dumps(plan))
        altered["trials"][0]["backend"] = "fp32"
        with self.assertRaisesRegex(ValueError, "altered"):
            depth.verify_plan(altered)

    def test_single_pass_pairs_preserve_provenance_and_quality_verdict(self):
        plan = self.plan(seeds=[1337, 2027, 4242], depths=[4, 8])
        # Build a separate, immutable single-pass plan with unchanged workload.
        with mock.patch.object(depth.subprocess, "check_output",
                               return_value="a" * 40):
            plan = depth.build_plan(self.binary, self.data,
                                    [1337, 2027, 4242], [4, 8],
                                    1, 20, 10, 2, 7, single_pass=True)
        self.assertEqual(plan["design"], "paired-depth-AB")
        self.assertEqual(len(plan["trials"]), 12)
        self.assertEqual([t["arm"] for t in plan["trials"][:2]], ["A", "B"])
        self.assertIs(depth.verify_plan(plan), plan)
        (self.root / "plan.json").write_text(json.dumps(plan))
        for trial in plan["trials"]:
            archive(self.root, plan, trial,
                    60.0 if trial["backend"] == "fp32" else 50.0)
        outcome = depth.analyze(self.root)
        self.assertEqual(
            outcome["depths"]["4"]["quality_status"],
            "inconclusive_fp32_not_learned")
        self.assertEqual(len(outcome["depths"]["8"]["seeds"]), 3)
        self.assertEqual(outcome["design"], "paired-depth-AB")
        # Duplicating a run is forbidden, not considered an extra seed.
        bad = json.loads(json.dumps(plan))
        bad["trials"].append(bad["trials"][0])
        with self.assertRaisesRegex(ValueError, "altered"):
            depth.verify_plan(bad)

    def test_deeper_mnist_depths_have_distinct_archives_and_are_fail_closed(self):
        plan = self.plan(seeds=[1337], depths=[12, 16, 20])
        self.assertEqual(len(plan["trials"]), 12)
        self.assertEqual([trial["depth"] for trial in plan["trials"][::4]],
                         [12, 16, 20])
        self.assertEqual([trial["arm"] for trial in plan["trials"][:4]],
                         list("ABBA"))
        (self.root / "plan.json").write_text(json.dumps(plan))
        for trial in plan["trials"]:
            archive(self.root, plan, trial, 50.0)
        outcome = depth.analyze(self.root)
        self.assertEqual(set(outcome["depths"]), {"12", "16", "20"})
        for d in ("12", "16", "20"):
            self.assertEqual(outcome["depths"][d]["quality_status"],
                             "inconclusive_fp32_not_learned")
        altered = json.loads(json.dumps(plan))
        altered["depths"].append(24)
        with self.assertRaisesRegex(ValueError, "altered"):
            depth.verify_plan(altered)

    def test_deep_single_pass_and_depth_whitelist(self):
        with mock.patch.object(depth.subprocess, "check_output",
                               return_value="a" * 40):
            plan = depth.build_plan(self.binary, self.data, [1337], [12, 16, 20],
                                    1, 20, 10, 2, 7, single_pass=True)
        self.assertEqual(len(plan["trials"]), 6)
        self.assertEqual([t["backend"] for t in plan["trials"]],
                         ["integer", "fp32"] * 3)
        self.assertEqual(depth.parse_numbers("12,16,20", depth.DEPTHS),
                         [12, 16, 20])
        with self.assertRaisesRegex(ValueError, "only"):
            depth.parse_numbers("12,24", depth.DEPTHS)

    def test_rejects_changed_binary_and_dataset(self):
        plan = self.plan()
        depth.check_environment(plan)
        self.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "executable changed"):
            depth.check_environment(plan)
        self.binary.write_bytes(b"dummy executable")
        (self.data / depth.DATA_FILES[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "development data changed"):
            depth.check_environment(plan)

    def test_archives_and_numeric_comparison(self):
        plan = self.plan()
        (self.root / "plan.json").write_text(json.dumps(plan))
        for trial in plan["trials"]:
            archive(self.root, plan, trial, 60.0 if trial["backend"] == "fp32" else 50.0)
        output = depth.analyze(self.root)
        self.assertEqual(set(output["depths"]), {"4", "8"})
        self.assertEqual(len(output["depths"]["4"]["seeds"]), 3)
        self.assertLess(output["depths"]["4"]["mean_integer_minus_fp32_pp"], 0)
        self.assertTrue((self.root / "summary.md").exists())
        record_path, _, _ = depth.artifact_paths(self.root, plan["trials"][0])
        record_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            depth.analyze(self.root)

    def test_no_official_test_or_modified_architecture(self):
        plan = self.plan(seeds=[1337], depths=[8])
        trial = plan["trials"][0]
        record = fixture(trial, plan)
        record["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "unmatched"):
            depth.validate_record(record, trial, plan)
        record = fixture(trial, plan)
        record["architecture_kind"] = "mnist_cnn_v1"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            depth.validate_record(record, trial, plan)

    def test_repetition_mismatch_not_counted_as_seed(self):
        plan = self.plan(seeds=[1337], depths=[4])
        (self.root / "plan.json").write_text(json.dumps(plan))
        for index, trial in enumerate(plan["trials"]):
            archive(self.root, plan, trial, 60.0 if index == 3 else 50.0)
        with self.assertRaisesRegex(ValueError, "non-reproducible"):
            depth.analyze(self.root)

    def test_existing_partial_archive_is_not_overwritten(self):
        plan = self.plan(seeds=[1337], depths=[4])
        (self.root / "plan.json").write_text(json.dumps(plan))
        path, _, _ = depth.artifact_paths(self.root, plan["trials"][0])
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
        with mock.patch.object(depth.subprocess, "run") as called:
            with self.assertRaises(FileNotFoundError):
                depth.run(self.root)
            called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
