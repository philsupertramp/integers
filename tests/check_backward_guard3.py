#!/usr/bin/env python3
"""Fixed initial sample: forward gates unchanged, backward-only arm observable."""
import json
import subprocess
import tempfile
from pathlib import Path

BINARY = Path("target/release/examples/held_out").resolve()
DATA = Path("data/mnist").resolve()

with tempfile.TemporaryDirectory(prefix="backward-guard3-smoke-") as td:
    records = {}
    for guard in (False, True):
        path = Path(td) / ("guard.json" if guard else "baseline.json")
        args = [
            str(BINARY), "--dataset", "mnist", "--backend", "integer",
            "--architecture", "cnn16", "--validation-only",
            "--data-dir", str(DATA), "--report", str(path),
            "--seed", "4242", "--epochs", "1", "--batch-size", "16",
            "--train-limit", "64", "--validation-limit", "64",
            "--lr-shift", "7", "--trace-layers", "--probe-first-backward"
        ]
        if guard:
            args.append("--backward-guard3")
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
        records[guard] = json.loads(path.read_text())
    original, treatment = records[False], records[True]
    assert original["backward_precision"] == "baseline"
    assert treatment["backward_precision"] == "guard3"
    before = original["first_backward_probe"]
    after = treatment["first_backward_probe"]
    assert before["schema"] == after["schema"] == 2
    assert before["gate_agreements"] == after["gate_agreements"], (
        "first-batch forward gates changed before any backward calculation"
    )
    assert before["pool_winner_agreements"] == after["pool_winner_agreements"], (
        "first-batch pooling routing changed before any backward calculation"
    )
    assert before["samples"] == after["samples"] == 16
    assert len(before["layer_gradient_boundaries"]) == (
        len(after["layer_gradient_boundaries"])
    )
    assert all(r["index"] == s["index"] for r, s in zip(
        before["layer_gradient_boundaries"],
        after["layer_gradient_boundaries"]
    ))
    # Gradient magnitudes and training outcomes may change in the treatment.
    # The invariant here is that only the backward-propagation path differs.
    print("OK: identical initial forward gates/pool winners; both backward paths recorded")
