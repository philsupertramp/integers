"""Offline validity and quality-gate tests for research RNG-elision evaluation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import evaluate_rng_elision as eva


def sample(seed, relaxed, accuracy=79):
    # 1,042 examples, 100 per class except class 0 with 142 examples.
    mat = []
    for j in range(10):
        n = 142 if j == 0 else 100
        correct = 112 if j == 0 else 79
        if j == 0 and accuracy == 75:
            correct = 70
        row = [0] * 10
        row[j] = correct
        row[(j + 1) % 10] = n - correct
        mat.append(row)
    total_correct = sum(mat[j][j] for j in range(10))
    val_pct = 100 * total_correct / 1042
    epochs = [
        {
            "epoch": epoch, "train_accuracy_pct": 70.0,
            "validation_accuracy_pct": val_pct,
            "validation_confusion": mat,
            "validation_per_class_recall": [
                mat[i][i] / sum(mat[i]) for i in range(10)
            ],
            "validation_predictions": [sum(mat[i][j] for i in range(10))
                                       for j in range(10)],
            "nonfinite_outputs": 0,
            "extreme_outputs": 0,
            "training_samples": 8192,
        } for epoch in (1, 2)]
    return {
        "schema": 1, "dataset": "mnist", "backend": "integer",
        "seed": seed, "experimental_skip_unused_rng": relaxed,
        "validation_only": True, "test_evaluated": False,
        "test_size": 0, "test_accuracy_pct": None, "epochs": 2,
        "split_seed": 1, "architecture_kind": "dense_mlp",
        "architecture": [784, 128, 128, 10],
        "train_size": 8192, "validation_size": 1042,
        "batch_size": 32, "lr_shift": 7, "quantization_shift": 7,
        "momentum_shift": 1, "output_bits": 31,
        "grad_clip_mantissa": 8192, "profile_training": True,
        "learning_curve": epochs,
    }


def populated(root):
    root.mkdir(parents=True)
    for seed in eva.SEEDS:
        for arm, relaxed in (("a", False), ("b", True)):
            (root / f"seed-{seed}-{arm}.json").write_text(json.dumps(sample(seed, relaxed)))


class ExperimentalQualityTests(unittest.TestCase):
    def test_seed_matched_gate_pass_and_epoch_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "quality"
            populated(root)
            result = eva.analyze(root)
            self.assertEqual(result["gate"]["status"], "pass")
            self.assertEqual(result["final_validation_change_pp"]["mean"], 0)
            self.assertEqual(len(result["per_seed"]), 5)
            self.assertEqual(len(result["per_seed"][0]["epochs"]), 2)
            self.assertEqual(len(result["per_seed"][0]["epochs"][0]
                                 ["recall_change_pp_by_class"]), 10)
            self.assertTrue((root / "quality.md").exists())

    def test_gate_blocks_real_numerical_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "quality"
            populated(root)
            path = root / "seed-1337-b.json"
            candidate = json.loads(path.read_text())
            for epoch in candidate["learning_curve"]:
                mat = epoch["validation_confusion"]
                mat[0][1] += mat[0][0] - 70
                mat[0][0] = 70
                epoch["validation_accuracy_pct"] = (
                    100 * sum(mat[j][j] for j in range(10)) / 1042)
                epoch["validation_per_class_recall"][0] = 70 / 142
                epoch["validation_predictions"] = [
                    sum(mat[i][j] for i in range(10)) for j in range(10)]
            path.write_text(json.dumps(candidate))
            result = eva.analyze(root)
            self.assertEqual(result["gate"]["status"], "regressed")
            self.assertLess(result["final_validation_change_pp"]["worst"], -3)

    def test_wrong_policy_or_test_access_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "quality"
            populated(root)
            path = root / "seed-1337-b.json"
            original = json.loads(path.read_text())
            for field, value in (("experimental_skip_unused_rng", False),
                                 ("test_evaluated", True),
                                 ("test_size", 10000)):
                broken = copy.deepcopy(original)
                broken[field] = value
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    eva.analyze(root)
            path.write_text(json.dumps(original))
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                eva.analyze(root)

    def test_inconsistent_accuracy_or_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "quality"
            populated(root)
            file = root / "seed-2027-b.json"
            original = json.loads(file.read_text())
            broken = copy.deepcopy(original)
            broken["learning_curve"][0]["validation_accuracy_pct"] = 99
            file.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "accuracy"):
                eva.analyze(root)
            broken = copy.deepcopy(original)
            broken["batch_size"] = 128
            file.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                eva.analyze(root)


if __name__ == "__main__":
    unittest.main()
