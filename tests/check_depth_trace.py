#!/usr/bin/env python3
"""End-to-end: instrumentation must leave training and RNG outcomes unchanged."""
import json
import subprocess
import tempfile
from pathlib import Path

BINARY = Path("target/release/examples/held_out").resolve()
DATA = Path("data/mnist").resolve()
for depth in (16, 20):
    for backend in ("integer", "fp32"):
        with tempfile.TemporaryDirectory(prefix="layer-trace-contract-") as tmp:
            outcomes = {}
            for traced in (False, True):
                path = Path(tmp) / ("traced.json" if traced else "plain.json")
                args = [
                    str(BINARY), "--dataset", "mnist", "--backend", backend,
                    "--architecture", f"cnn{depth}", "--validation-only",
                    "--data-dir", str(DATA), "--report", str(path),
                    "--seed", "4242", "--epochs", "2", "--batch-size", "16",
                    "--train-limit", "64", "--validation-limit", "64", "--lr-shift", "7",
                ]
                if traced:
                    args.append("--trace-layers")
                subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
                outcomes[traced] = json.loads(path.read_text())
            plain, traced = outcomes[False], outcomes[True]
            assert plain.get("layer_diagnostics") is None
            records = traced["layer_diagnostics"]["records"]
            assert [r["epoch"] for r in records] == [1, 2]
            for record in records:
                layers = record["layers"]
                assert len(layers) == depth + 2, (depth, backend, len(layers))
                assert [layer["index"] for layer in layers] == sorted(layer["index"] for layer in layers)
                for layer in layers:
                    assert layer["gradient_count"] == layer["parameter_count"]
                    assert layer["changed_parameters"] + layer["unchanged_parameters"] == layer["parameter_count"]
                    assert layer["quantized_to_zero_updates"] is None
                    assert 0 <= layer["zero_gradient_count"] <= layer["gradient_count"]
                    if backend == "integer":
                        assert 0 <= layer["forward_saturated_count"] <= layer["forward_count"]
                        assert 0 <= layer["forward_zero_count"] <= layer["forward_count"]
                    else:
                        assert layer["forward_saturated_count"] is None
            for a, b in zip(plain["learning_curve"], traced["learning_curve"]):
                for key in ("train_accuracy_pct", "validation_accuracy_pct",
                            "validation_confusion", "validation_predictions",
                            "nonfinite_outputs", "extreme_outputs",
                            "nonfinite_fp32_parameters"):
                    assert a[key] == b[key], (depth, backend, key, a[key], b[key])
            print(f"OK unchanged learning and valid trace depth={depth} backend={backend}",
                  flush=True)
