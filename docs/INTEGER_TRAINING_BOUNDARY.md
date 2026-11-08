# ADR-002 — Integer-only training boundary

Status: draft PR #29 / issue #26, candidate branch `big-refactor`. The representation remains `Dyadic { v, s }` = `v × 2^(-s)`. No claim is made that the *whole executable* runs without floating-point instructions.

## Enforced execution boundary

- `nn::Sequential::new()` is the **default integer backend**. `add()` rejects modules not explicitly classified `TrainingArithmetic::IntegerOnly`. The same gate runs **before** `forward`, `backward`, `forward_batch`, `backward_batch`, `update`, `zero_grad` and `set_training` so public direct manipulation of `Sequential.layers` cannot silently evade it. An unverified third-party `Module` defaults to `Unverified`, not automatically trusted. This is a code review declaration, not a proof about custom implementations.
- `Sequential::new_experimental_float()` is an explicit compatibility/experimental opt-in that permits the historically documented FP64 Softmax and BatchNorm forward/backward. **Do not describe models constructed this way as integer-only training.** Directly calling their `Module` methods is likewise an experimental operation outside the enforced `Sequential` integer backend. The public library does not attempt to prohibit all independent experimental/reference use of those types.
- Existing public dyadic helpers, initializers, data processing and training reporters remain available. Their existence does **not** make the compliant training path floating point.

## Audited operation/function inventory

| Execution operation | Implementation/function(s) | Runtime arithmetic | Integer-backend status |
|---|---|---|---|
| Linear affine | `Linear::forward_one`, `nn::linear::forward(_batch)`, `dyadic::{mul,add,requantize}` | Integer mantissas, integer scale arithmetic, stochastic integer rounding | Allowed; numerical fixtures in #14 |
| Linear gradients | `Linear::backward_one`, `ste_requantize`, dyadic `mul/add` | Integer | Allowed; actual clipping flags cached |
| Conv2D | `Conv2D::forward_one/backward_one`, index helpers | Integer | Allowed; tiny oracle fixtures in #14 |
| ReLU | `ReLU::forward/backward` and batch variants | Integer compares and masks | Allowed |
| MaxPool2D | `dyadic_greater`, `MaxPool2D::pool_forward/pool_backward` | Integer mixed-scale comparisons and integer gradient scatter | Allowed |
| Flatten | `Flatten::forward/backward` | Memory-copy/identity only | Allowed |
| Dropout | `Dropout::sample_mask/apply_mask`, forward/backward variants, `rng_next` | Integer RNG mask, integer quantized dyadic scale multiply | Allowed; **constructor** uses f32 to precompute scale and threshold |
| Optimizer | `sgd_step`, `momentum_step`, `apply_updates`, Linear/Conv2D `update` | Integer shifts, stochastic rounding, saturation | Allowed; extreme-scale contract remains #12 |
| Loss gradients | `mse_grad`, `cross_entropy_grad` | Integer saturating subtraction (scale validated) | Allowed; CE is fused pre-softmax gradient |
| Dyadic arithmetic | `stochastic_round(_i64)`, `rescale_mantissa`, `align/add/sub/mul/div/clip/requantize/ste_clip/ste_requantize` | Integer | Used by approved path; broad overflow/domain proof remains #12 |
| Softmax output | `Softmax::softmax_forward` | **FP64** decode, exponential, sum, division, requantize | **Excluded** from default integer backend; experimental only |
| BatchNorm1D/2D | `forward/backward` (sample and batch), running-stat updates and parameter gradients | **FP64** means, variance, inverse standard deviation and derivatives | **Excluded** from default integer backend; experimental only |
| Constructors | `Linear::new`, `Conv2D::new`, `Dropout::new`, BatchNorm constructors | FP64/FP32 parameter initialization and setup | **Permitted before training**, not a runtime kernel |
| Descriptions/reports | `Linear::describe`, `Conv2D::describe`, `Dropout::describe`, `TrainingReporter`, `Dyadic::to_f64` | Floating-point reporting/inspection | Permitted diagnostics only; no call from audited integer math |
| Data pipeline | `DatasetBuilder`, quantization setup, sample conversion, optional parquet/image loaders | May use float and external libraries | Preprocessing; not the integer training backend |

The explicitly known BatchNorm/Softmax floating-point paths were already documented in the project's blog series; this PR enforces the previously approved boundary rather than claiming to have discovered them.

### Runtime enforcement and review

The enforcement mechanism is the Rust `TrainingArithmetic` classification and the fail-closed checks in `Sequential` itself, rather than a second source-text scanner. `tests/integer_boundary.rs` exercises the default rejection behavior, including mutations of the public `layers` collection, and a real integer-only Linear → ReLU → Dropout → Linear minibatch forward/backward/update with seed `0x260`. Newly added or custom `Module` implementations default to `Unverified` and require an explicit declaration and code review before admission to the standard integer backend. The per-operation inventory above is maintained through code review.

A module that falsely declares `IntegerOnly` cannot be made safe by this runtime classification alone; no whole-program/compiler-level proof of float-free execution is claimed.

## Independent numerical diagnostics

`tests/integer_boundary.rs` exercises experimental FP64 Softmax and BatchNorm **only in explicit experimental mode** and compares decoded dyadic outputs to independent mathematical reference expressions. Softmax's identity backward STE is *not* treated as a true softmax derivative. BatchNorm reference covers zero variance (constant input), signed activations, small multifeature/multichannel batches and extreme input mantissas; expected values are quantized at shift 7. Individual unsaturated activation comparisons allow 0.5–0.51 output LSB (or 1.5 LSB where accumulated rounding can contribute). A separate saturation regression confirms that an extreme Softmax probability saturates to the largest signed Q7 value (127/128 instead of exactly 1), **one Q7 LSB** below the ideal result; oversized BatchNorm affine products saturate at **i32::MIN/i32::MAX**, not wrap. FP64 computation in these tests is reference-only. The experimental BatchNorm fix corrects batch cardinality and channel indexing while retaining its FP64 arithmetic; it does **not** make BatchNorm integer compliant.

## Remaining gaps

- Full numerical coverage for extreme shifts, stochastic rounding statistics, saturating/overflow cases and domain semantics belongs to #12. Numerical unit results do not prove network convergence (#15).
- The experimental BatchNorm/Softmax implementations remain noncompliant by design; an actual **integer-valued normalization/softmax backend** would require separate numerical approximation design, reference validation and performance evaluation. This PR deliberately does not implement one or silently route it to FP.
- Entire-binary or third-party verified absence of floating-point instructions, user-provided Module implementations, cross-architecture/device reproducibility and Raspberry Pi 5 AArch64 benchmarks are not established.
- Repository-wide rustfmt/Clippy debt is tracked in #20; dataset-dependent examples are not CI training acceptance evidence (#13).
