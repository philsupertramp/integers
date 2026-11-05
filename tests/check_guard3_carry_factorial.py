#!/usr/bin/env python3
"""Research invariant: all four treatment arms begin from identical forward gates."""
import json
import subprocess
import tempfile
from pathlib import Path

BINARY = Path("target/release/examples/held_out").resolve()
DATA = Path("data/mnist").resolve()
arms = {
    "baseline": [],
    "carry24": ["--subunit-carry"],
    "guard3": ["--backward-guard3"],
    "guard3_carry24": ["--backward-guard3", "--subunit-carry"],
}
with tempfile.TemporaryDirectory(prefix="guard3-carry-factorial-") as tmp:
    records = {}
    for arm, flags in arms.items():
        report = Path(tmp) / f"{arm}.json"
        cmd = [
            str(BINARY), "--dataset", "mnist", "--backend", "integer",
            "--architecture", "cnn16", "--validation-only",
            "--data-dir", str(DATA), "--report", str(report),
            "--seed", "4242", "--epochs", "1", "--batch-size", "16",
            "--train-limit", "64", "--validation-limit", "64",
            "--lr-shift", "7", "--trace-layers", "--probe-first-backward",
            *flags,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        records[arm] = json.loads(report.read_text())
    probe = records["baseline"]["first_backward_probe"]
    assert probe["schema"] == 2 and probe["samples"] == 16
    for name, record in records.items():
        candidate = record["first_backward_probe"]
        assert candidate["gate_agreements"] == probe["gate_agreements"], name
        assert candidate["pool_winner_agreements"] == probe["pool_winner_agreements"], name
        assert candidate["samples"] == probe["samples"] == 16
        assert record["optimizer_mode"] == (
            "subunit_carry24" if "carry" in name else "baseline"
        )
        assert record["backward_precision"] == (
            "guard3" if "guard3" in name else "baseline"
        )
    print("OK four arms start from identical forward gates and MaxPool winners")
