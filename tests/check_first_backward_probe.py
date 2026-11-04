#!/usr/bin/env python3
"""An initial paired backprop observer must not affect baseline integer training."""
import json
import subprocess
import tempfile
from pathlib import Path

BINARY = Path("target/release/examples/held_out").resolve()
DATA = Path("data/mnist").resolve()

for depth in (16, 20):
    with tempfile.TemporaryDirectory(prefix="backprop-first-batch-") as work:
        reports = {}
        for observe in (False, True):
            path = Path(work) / ("observed.json" if observe else "unobserved.json")
            cmd = [
                str(BINARY), "--dataset", "mnist", "--backend", "integer",
                "--architecture", f"cnn{depth}", "--validation-only",
                "--data-dir", str(DATA), "--report", str(path),
                "--seed", "4242", "--epochs", "1", "--batch-size", "16",
                "--train-limit", "64", "--validation-limit", "64",
                "--lr-shift", "7", "--trace-layers",
            ]
            if observe:
                cmd.append("--probe-first-backward")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            reports[observe] = json.loads(path.read_text())

        plain, observed = reports[False], reports[True]
        assert plain["first_backward_probe"] is None
        probe = observed["first_backward_probe"]
        assert probe["schema"] == 1
        assert probe["scope"] == "first_training_minibatch_before_any_optimizer_update"
        assert probe["samples"] == 16
        assert probe["same_indices"] and probe["same_decoded_initial_weights"]
        assert len(probe["parameter_gradient_zeros"]) == depth + 2
        assert len(probe["gate_agreements"]) == depth + 1
        boundaries = probe["layer_gradient_boundaries"]
        assert len(boundaries) == 2 * depth + 6  # conv/relu + pools + flatten + dense/relu/head
        assert boundaries[0]["integer_propagated"] is None
        assert boundaries[0]["fp32_propagated"] is None
        for layer in boundaries:
            for side in ("received", "propagated"):
                for backend in ("integer", "fp32"):
                    row = layer[f"{backend}_{side}"]
                    if row is not None:
                        assert 0 <= row["zero"] <= row["count"]
                        if backend == "fp32":
                            assert row["nonfinite"] == 0
        for row in probe["gate_agreements"]:
            assert row["elements"] == sum(row[k] for k in
                ("both_active", "integer_only", "fp32_only", "both_inactive"))
            assert row["elements"] > 0
        for a, b in zip(plain["learning_curve"], observed["learning_curve"]):
            for key in ("train_accuracy_pct", "validation_accuracy_pct",
                        "validation_confusion", "validation_predictions",
                        "nonfinite_outputs", "extreme_outputs"):
                assert a[key] == b[key], (depth, key, a[key], b[key])
        assert plain["layer_diagnostics"] == observed["layer_diagnostics"]
        print(f"OK first-backprop paired trace non-invasive depth={depth}", flush=True)
