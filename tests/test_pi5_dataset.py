"""Dataset integrity/provenance tests; no MNIST download required."""
import hashlib
import json
import io
import sys
import tempfile
from unittest import mock
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import prepare_pi5_dataset as prep
import pi5_benchmark as runner
import collect_pi5_indexed as collector
import summarize_pi5_profile as profiler


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

    def test_indexed_job_seeds(self):
        seeds = "1337,2027,4242"
        for i, expected in enumerate((1337, 2027, 4242)):
            with mock.patch.dict("os.environ", {"JOB_COMPLETION_INDEX": str(i)}):
                args = runner.parse_args(["--validation-only", "--seed-list", seeds])
                self.assertEqual(args.seed, expected)
                self.assertEqual(runner.indexed_seed_metadata(args), {
                    "job_completion_index": i,
                    "indexed_job_seeds": [1337, 2027, 4242],
                })
        with mock.patch.dict("os.environ", {"JOB_COMPLETION_INDEX": "2"}):
            self.assertEqual(runner.parse_args(["--validation-only"]).seed, 1337)
        for index in ("-1", "3", "invalid", ""):
            with mock.patch.dict("os.environ", {"JOB_COMPLETION_INDEX": index}):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--validation-only", "--seed-list", seeds])
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--validation-only", "--seed-list", seeds])
        for bad_seeds in ("1337,1337", "1337,0", "1337,-1", "1337,", "x,2027"):
            with mock.patch.dict("os.environ", {"JOB_COMPLETION_INDEX": "0"}):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--validation-only", "--seed-list", bad_seeds])
        with mock.patch.dict("os.environ", {"JOB_COMPLETION_INDEX": "0"}):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--validation-only", "--seed", "1337",
                                   "--seed-list", seeds])

    def test_indexed_collector_pod_validation(self):
        def pod(index, phase="Succeeded"):
            return {"metadata": {"name": f"pi-benchmark-{index}-pod",
                    "annotations": {collector.INDEX_ANNOTATION: str(index)}},
                    "status": {"phase": phase}}
        self.assertEqual(collector.indexed_pods(
            [pod(2), pod(0), pod(1)], [1337, 2027, 4242]),
            {0: "pi-benchmark-0-pod", 1: "pi-benchmark-1-pod",
             2: "pi-benchmark-2-pod"})
        for pods in ([pod(0), pod(1)], [pod(0), pod(0), pod(2)],
                     [pod(0), pod(1, "Pending"), pod(2)]):
            with self.assertRaises(ValueError):
                collector.indexed_pods(pods, [1337, 2027, 4242])

    def test_training_profiling_opt_in_and_fail_closed(self):
        args = runner.parse_args(["--validation-only", "--profile-training",
                                  "--train-limit", "8", "--validation-limit", "10"])
        self.assertTrue(args.profile_training)
        for options in (["--final-test", "--profile-training"],
                        ["--validation-only", "--architecture", "cnn",
                         "--profile-training"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(options)
        epoch = {"training_ms": 10, "training_profile": {
            "schema": 1, "clock": "Instant", "unit": "ns",
            "batches": 1, "samples": 8, "training_wall_ns": 10_000_000,
            "preparation_ns": 1_000_000, "zero_grad_ns": 1_000_000,
            "forward_ns": 2_000_000, "loss_metrics_ns": 1_000_000,
            "backward_ns": 3_000_000, "update_ns": 1_000_000,
            "other_ns": 1_000_000,
        }}
        record = {"profile_training": True, "train_size": 8,
                  "batch_size": 32, "learning_curve": [epoch]}
        runner.verify_training_profile(record, True)
        with self.assertRaises(ValueError):
            runner.verify_training_profile(record, False)
        epoch["training_profile"]["samples"] = 7
        with self.assertRaises(ValueError):
            runner.verify_training_profile(record, True)
        record["profile_training"] = False
        epoch["training_profile"] = None
        runner.verify_training_profile(record, False)

    def test_profile_summary_rejects_missing_or_bad_timing(self):
        profile = {"training_wall_ns": 100, "samples": 8, "batches": 1}
        profile.update({name: 0 for name in profiler.PHASES})
        profile["forward_ns"] = 60
        profile["other_ns"] = 40
        report = {"runs": [{"backend": "integer",
                            "learning_curve": [{"training_profile": profile}]}]}
        summary = profiler.summarize(report)[0]
        self.assertEqual(summary["percent"]["forward_ns"], 60.0)
        self.assertEqual(summary["samples"], 8)
        profile["other_ns"] = 50
        with self.assertRaises(ValueError):
            profiler.summarize(report)
        report["runs"][0]["learning_curve"][0]["training_profile"] = None
        with self.assertRaises(ValueError):
            profiler.summarize(report)

    def test_cli_does_not_implicitly_evaluate_official_test(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--epochs", "1"])
        with self.assertRaises(SystemExit):
            runner.parse_args(["--validation-only", "--final-test"])
        args = runner.parse_args(["--validation-only"])
        self.assertEqual((args.train_limit, args.validation_limit), (128, 128))


if __name__ == "__main__":
    unittest.main()
