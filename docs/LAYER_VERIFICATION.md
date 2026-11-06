# Independent layer verification (draft issue #14)

This is unit-level evidence for the candidate branch, **not** approval of PR #9, proof of full network convergence, or proof of integer-only computation. Tests live in tests/layer_oracles.rs and require no downloaded data.

## Numerical methodology

Reference FP64 helpers decode each Dyadic value as mantissa multiplied by 2 raised to the negative scale, without calling production dyadic arithmetic. The test suite independently computes Linear matrix/vector affine products, Conv2D valid cross-correlations, and their corresponding transposed-stencil / outer-product backpropagation results. A central-difference FP64 test (step 1e-5) verifies the continuous, unsaturated Linear input derivative, *not* a derivative through stochastic quantization. Analytical tests compare value and signed scale.

Seeded stochastic test paths use 14, 16–20, 0x14 and 0x140; deterministic update cases use exactly divisible shifts to avoid stochastic ties.

For exact fixtures: require matching integer mantissas/scales or decoded FP64 difference below 1e-12. The comparison helper allows an explicit multiple of the output LSB (2 raised to negative output shift), plus 1e-12. Softmax output permits half an output LSB. Per-operation rounding may accumulate; these narrow tolerances are specific to the fixtures, not global error guarantees. Saturation is tested against signed output-bit bounds and never evaluated with ordinary finite differences. Linear and Conv2D keep a clipped/nonclipped mask, so the STE suppresses gradients only for *genuinely* saturated results; exactly-on-boundary but unclipped values pass.

## Coverage matrix

| Component | Forward | Backward, updates and validation | Limits |
|---|---|---|---|
| Linear | FP64 affine; batch/per-sample | Analytical input, weight and bias gradient, FP64 finite difference, accumulation, zero_grad, momentum/no-momentum, cache replacement, clipping STE and mismatched shapes | Large shifts and stochastic distributions not proven (#12) |
| ReLU | Negative/zero/positive and mixed scales, batch | Masked gradient and cache/shape validation | Derivative at zero conventionally zero |
| Conv2D | FP64 2x2 kernel on 3x3 one-channel input; batch equivalence | Transposed-stencil input gradient, analytical kernel/bias accumulation and updates, zero_grad, clipped STE, batch caches and dimensions | Multi-channel and broader shapes need expanded coverage |
| MaxPool2D | Signed mixed-scale selection and ties, batch | First-index tie routing, gradient scattering, dimension validation | Overlapping-window accumulation not independently property-tested |
| Flatten | Preserves data and shape through sample/batch | Pass-through backward | Current implementation is identity, not a general reshape |
| Loss gradient helpers | MSE y-t, fused Softmax+CE p-t | Independent decoded FP64 differences, empty matched arrays, mismatch rejection | CE gradient is fused pre-softmax, not standalone probability CE |
| Softmax | FP64 exponentials and output quantization | Explicit identity STE versus true Jacobian | FP64 runtime forward violates integer-only boundary (#26) |
| Dropout | Seeded mask and quantized inverted keep-scale, eval identity | Same mask applied to gradients, batch and per-sample consistency | RNG distribution not statistically verified |

BatchNorm1D/2D have **no integer-backend compliance claim**: current runtime forward and backward use FP64. Their replacement or exclusion is issue #26. The Softmax forward FP64 violation is also reported to #26. Dropout inverse-scale FP32 is construction-time setup, not runtime training computation.

## Fixes and remaining boundaries

In-scope corrections reject loss length/scale mismatches (no silent zip truncation), validate Linear and Conv2D input and batch gradient lengths, preserve actual clipping decisions for backward STE, compare MaxPool dyadics by represented value using integer-only comparisons, and validate ReLU/Dropout caches and masks.

Known remaining gaps: comprehensive dyadic extremes (#12), loss-gradient arithmetic saturation (#12), noncompliant Softmax/BatchNorm runtime (#26), general Tensor::from_vec release-mode shape validation, full-network convergence (#15), public shape/precision policy (#16/#19), and broader Clippy/rustfmt cleanup (#20). The tests do not establish distributional rounding unbiasedness for arbitrary layer configurations or Pi 5 performance.

GitHub Actions runs the added no-dataset test target from dyadic-correctness.yml on pull requests into big-refactor, alongside existing unit, dyadic, public API, doctest and all-target checks. Final-run outcomes and links are recorded on the PR.
