#!/usr/bin/env python3
"""Conservative, zero-dependency source audit of the ADR-002 integer backend.

This does NOT prove arbitrary third-party module implementations or transitive
code safe. It prevents accidental *direct* floating-point arithmetic in the
explicitly reviewed Rust functions and rejects unknown built-in Module impls.
Sequential additionally enforces a fail-closed runtime allowlist.
Run from the repository root: python3 scripts/check_integer_boundary.py
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Per-source reviewed runtime functions, including helpers called by layers.
# Initialization and describe/reporting functions are intentionally excluded.
CHECKS = {
    "src/nn/linear.rs": {
        "impl Linear": ["w", "flat", "forward_one", "backward_one"],
        "impl Module for Linear": ["forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad"],
    },
    "src/nn/conv.rs": {
        "impl Conv2D": ["output_len", "in_idx", "out_idx", "k_idx", "forward_one", "backward_one"],
        "impl Module for Conv2D": ["forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad"],
    },
    "src/nn/activations.rs": {
        "impl Module for ReLU": ["forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad"],
    },
    "src/nn/regularization.rs": {
        "impl Dropout": ["sample_mask", "apply_mask"],
        "impl Module for Dropout": ["forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad", "set_training"],
        "fn dyadic_greater": [],
        "impl MaxPool2D": ["in_idx", "out_idx", "pool_forward", "pool_backward", "output_len"],
        "impl Module for MaxPool2D": ["forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad"],
    },
    "src/nn/mod.rs": {
        "impl Sequential": ["add", "check_integer_backend", "forward", "backward", "forward_batch", "backward_batch", "update", "zero_grad"],
        "impl Module for Flatten": ["forward", "backward", "update", "zero_grad"],
        "fn sgd_step": [], "fn momentum_step": [], "fn apply_updates": [],
    },
    "src/lib/dyadic.rs": {
        "fn stochastic_round": [], "fn stochastic_round_i64": [],
        "fn rescale_mantissa": [], "fn align": [], "fn add": [],
        "fn sub": [], "fn mul": [], "fn div": [], "fn clip": [],
        "fn requantize": [], "fn ste_clip": [], "fn ste_requantize": [],
    },
    "src/lib/rng.rs": {
        "impl XorShift64": ["next_u64", "gen_range"],
        "fn rng_next": [], "fn rng_range": [],
    },
    "src/lib.rs": {
        "fn mse_grad": [], "fn cross_entropy_grad": [],
    },
}
# Every current built-in module must be explicitly reviewed. New implementations
# require an audit+manifest update before they can claim IntegerOnly.
INTEGER_BUILTINS = {"Linear", "Conv2D", "ReLU", "MaxPool2D", "Flatten", "Dropout"}
FLOAT_BUILTINS = {"Softmax", "BatchNorm1D", "BatchNorm2D"}
MODULE_FILES = ("src/nn/linear.rs", "src/nn/conv.rs",
                "src/nn/activations.rs", "src/nn/regularization.rs", "src/nn/mod.rs")
BAD = re.compile(r"\b(?:f32|f64)\b|\bto_f64\s*\(|\.(?:sqrt|exp|log2|powi|powf|round|floor|ceil)\s*\(")


def block(text: str, needle: str) -> str:
    """Extract a balanced Rust braced block from a known symbol's declaration."""
    if needle.startswith("fn "):
        pattern = r"(?<![\w])" + re.escape(needle) + r"\s*\("
    else:
        pattern = r"(?<![\w])" + re.escape(needle) + r"\s*\{"
    m = re.search(pattern, text)
    if m is None:
        raise AssertionError(f"Audited symbol missing: {needle}")
    start = text.find("{", m.end() if needle.startswith("fn ") else m.start())
    if start == -1:
        raise AssertionError(f"Missing Rust body: {needle}")
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos]
    raise AssertionError(f"Unbalanced Rust body: {needle}")


def main() -> None:
    tested = 0
    for filename, scopes in CHECKS.items():
        contents = (ROOT / filename).read_text()
        for scope, functions in scopes.items():
            region = block(contents, scope) if functions else block(contents, scope)
            targets = [(name, block(region, "fn " + name)) for name in functions] if functions else [(scope, region)]
            for name, body in targets:
                forbidden = BAD.search(body)
                if forbidden:
                    raise AssertionError(f"{filename}: {scope}/{name} contains float-like runtime code: {forbidden.group()}")
                tested += 1

    actual = set()
    for filename in MODULE_FILES:
        source = (ROOT / filename).read_text()
        actual.update(re.findall(r"impl\s+Module\s+for\s+(\w+)\s*\{", source))
    if actual != INTEGER_BUILTINS | FLOAT_BUILTINS:
        raise AssertionError(f"Module inventory changed: actual={actual}, expected={INTEGER_BUILTINS | FLOAT_BUILTINS}")
    for filename in MODULE_FILES:
        source = (ROOT / filename).read_text()
        for name in actual:
            if f"impl Module for {name} {{" not in source:
                continue
            implementation = block(source, f"impl Module for {name}")
            contract = "IntegerOnly" if name in INTEGER_BUILTINS else "FloatingPoint"
            fn_body = block(implementation, "fn training_arithmetic")
            if not re.search(r"TrainingArithmetic::" + contract + r"\b", fn_body):
                raise AssertionError(f"{name} must explicitly declare {contract}")

    # Sentinel verifies that our forbidden-expression matcher really rejects
    # an FP64 forward path rather than silently ignoring it.
    assert BAD.search("let logits: Vec<f64> = input.iter().map(|v| v.to_f64()).collect();")
    print(f"ADR-002: audited {tested} reviewed function bodies and {len(actual)} built-in Module contracts")


if __name__ == "__main__":
    main()
