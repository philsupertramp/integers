"""Dataset integrity/provenance tests; no MNIST download required."""
import hashlib
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import prepare_pi5_dataset as prep
import pi5_benchmark as runner


class Pi5DatasetTests(unittest.TestCase):
    def test_shared_input_normalization(self):
        self.assertEqual(prep.pixel_to_signed(0), -127)
        self.assertEqual(prep.pixel_to_signed(255), 127)
        self.assertEqual(prep.normalized_bytes(bytes(range(256))),
                         bytes((prep.pixel_to_signed(p) & 255) for p in range(256)))
        self.assertTrue(all(-127 <= prep.pixel_to_signed(p) <= 127 for p in range(256)))

    def test_manifest_and_artifact_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in ("train-inputs.i8", "train-labels.u8", "test-inputs.i8", "test-labels.u8"):
                content = (name + "\n").encode()
                (root / name).write_bytes(content)
                artifacts[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            manifest = {
                "schema_version": 1,
                "splits": {"seed": runner.SPLIT_SEED, "validation_per_class": 500},
                "preprocessing": {"shift": 7},
                "artifacts": artifacts,
            }
            raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
            (root / "dataset-manifest.json").write_bytes(raw)
            (root / "dataset-manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
            _, digest = runner.verify_dataset(root)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            (root / "test-labels.u8").write_bytes(b"corrupted")
            # Strict validation-only isolation: test bytes must never be opened.
            runner.verify_dataset(root, validation_only=True)
            with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                runner.verify_dataset(root)
            (root / "test-labels.u8").write_bytes(("test-labels.u8\n").encode())
            (root / "dataset-manifest.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                runner.verify_dataset(root)

    def test_log_archive_roundtrip_and_corruption_detection(self):
        # A large fixture forces multiple chunks, exercising CRI-safe framing.
        payload = (json.dumps({"schema_version": 1, "runs": [
            {"epoch": i, "confusion": [[i + j] * 10 for j in range(10)]}
            for i in range(100)]}, indent=2) + "\n").encode()
        output = io.StringIO()
        runner.emit_result_archive(payload, stream=output)
        logs = output.getvalue()
        self.assertGreater(logs.count("BENCHMARK_ARCHIVE_V1_CHUNK="), 1)
        self.assertTrue(all(len(line) < 2048 for line in logs.splitlines()))
        self.assertEqual(runner.extract_result_archive(logs), payload)
        timed_logs = "".join("2026-10-25T20:00:00Z " + line + "\n"
                             for line in logs.splitlines())
        self.assertEqual(runner.extract_result_archive(timed_logs), payload)
        broken = logs.replace("BENCHMARK_ARCHIVE_V1_CHUNK=0:",
                              "BENCHMARK_ARCHIVE_V1_CHUNK=9:", 1)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            runner.extract_result_archive(broken)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.extract_result_archive("\n".join(logs.splitlines()[:-1]))

    def test_cli_does_not_implicitly_evaluate_official_test(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--epochs", "1"])
        with self.assertRaises(SystemExit):
            runner.parse_args(["--validation-only", "--final-test"])
        args = runner.parse_args(["--validation-only"])
        self.assertEqual((args.train_limit, args.validation_limit), (128, 128))


if __name__ == "__main__":
    unittest.main()
