# Dyadic numerical contract — `big-refactor` candidate

**Status:** experimental; documents the current candidate behavior for [#12](https://github.com/philsupertramp/integers/issues/12). It is not an approval to merge [PR #9](https://github.com/philsupertramp/integers/pull/9), freeze a stable public API, or select new storage widths under ADR-006. The maintainer-approved runtime arithmetic boundary remains ADR-002.

## Representation and domain

`Dyadic { v: i32, s: u32 }` represents the exact rational `v / 2^s`. Larger `s` means **finer** resolution. The representation is not unique; `(3, 2)` and `(6, 3)` both encode 3/4. `s` is always nonnegative, and a requested arithmetic output exponent outside `0..=u32::MAX` is invalid.

`Dyadic::equivalent` compares canonical odd mantissas and residual nonnegative scales after removing factors of two. It does not align mantissas by potentially overflowing shifts. Zero is equivalent to zero regardless of shift. The derived `PartialEq` still compares the **stored pair**, not the decoded value; use `equivalent` for numeric equality.

`to_f64` is an **approximate diagnostic decode**, not part of the integer-only training backend. It scales in bounded steps to avoid exponent casts wrapping for very large `s`. Subnormal rounding or underflow to zero remains possible in f64 and does not change the stored rational.

## Scale conversion and integer bounds

- **Finer target** (`s_target > s_source`): the exact mantissa is `v * 2^(s_target - s_source)`. If it exceeds signed i32, saturate to `i32::MIN` or `i32::MAX` (zero stays zero). No wraparound is allowed.
- **Equal target:** preserve `v`.
- **Coarser target** (`s_target < s_source`): stochastic-round the mantissa by `k = s_source - s_target`. The result can differ from the exact rational by **less than one target-scale LSB** before any storage saturation; large shifts can still yield a very rare signed unit rather than always zero.
- `align` exposes the resulting Dyadic but **not** an overflow flag. `requantize` additionally clamps into inclusive, valid `[q_min, q_max]` and reports `clipped=true` if i32 rescale overflow or that output-range clamp occurred. Discarding fractional bits during stochastic rounding is *not* recorded as clipping; a value exactly at a bound and not truncated past it is not automatically clipped.
- `signed_bounds(bits)` returns the signed interval for `1..=31` bits; `bits>=32` returns the full i32 range, while `bits=0` panics. `clip` expects ordered `q_min<=q_max` bounds.

All overflow policies concern **mantissa storage**. A saturated Dyadic no longer exactly represents the input rational; there is no general small-error guarantee after saturation. Layer STE consumers must use a forward clipping/saturation record rather than infer clipping from the final bounded mantissa ([#14](https://github.com/philsupertramp/integers/issues/14)).

## Arithmetic

| Operation | Output shift | Mantissa computation | Overflow/domain |
| --- | --- | --- | --- |
| `add(a,b)` | `max(a.s,b.s)` | Align each operand (possibly saturating), then saturating i32 addition | Intermediate alignment and sum saturate separately |
| `sub(a,b)` | `max(a.s,b.s)` | Align each operand, then saturating i32 subtraction | As with add |
| `mul(a,b,q)` | `a.s + b.s - q` | Exact i64 product of i32 mantissas, stochastic-round by `q`, saturate to i32 | Panics if output exponent is negative or exceeds `u32::MAX` |
| `div(a,b,p)` | `a.s + p - b.s` | Truncate `(a.v * 2^p) / b.v` **toward zero**, then saturate to i32 | Panics on zero denominator or invalid output exponent; never wraps on large `p` |

The output exponent calculations use widened unsigned intermediates; they must not be silently clamped to zero or `u32::MAX`. **Invalid exponents deliberately panic** rather than silently reinterpret numeric values. These are current candidate preconditions; they do not introduce a new enum or public error-handling API.

For division with nonzero denominator, integer truncation gives strictly less than one output mantissa unit of rational error **only if the quotient did not saturate**. Division is *not* stochastic or rounded-to-nearest. For large `p`, a nonzero quotient known to exceed i32 bounds is saturated without performing an overflowing left shift. `mul` can saturate or stochastically round after exact i64 multiplication; `q` does not authorize a negative exponent.

## Stochastic rounding and reproducibility

For positive integer shift `k`, ideal stochastic rounding is `floor(v / 2^k) + Bernoulli((v mod 2^k) / 2^k)`; the remainder uses Euclidean/nonnegative semantics, so it works for negative mantissas as well. Under an **ideal uniform-random-bit assumption** its expectation equals `v / 2^k`. The `i64` intermediate path uses the same rule before the final saturating conversion to i32. At and beyond the word width (`k>=32` for i32, `k>=64` for i64), rare signed nonzero outcomes are sampled rather than replaced by deterministic zero. For very large shifts, leading-zero 32-bit chunk rejection keeps expected computation bounded.

The implementation uses a deterministic **thread-local XorShift64 PRNG**, not a cryptographic RNG or perfectly independent uniform bits. Therefore the mathematical expectation is a theoretical target, **not a formal guarantee that a finite pseudorandom stream is strictly unbiased**. Statistical tests cover representative signed large-shift distributions; they do not prove global statistical independence or error guarantees after saturation. `seed_rng(nonzero_seed)` resets only the calling thread; use a fixed seed and identical call ordering per thread for reproducible draws. Do not assume ordering across threads.

## Verification and limitations

`tests/dyadic_regression.rs` provides independent small exact-integer/`i128` rational oracles for value-preserving conversion, addition, subtraction and division (including residual error less than one output LSB for unsaturated division), alongside saturation, extremes, invalid exponents, and seeded stochastic statistics. `cargo test --locked --test dyadic_regression` is the targeted command; `cargo test --locked --lib` and the broader CI jobs should run for integration.

Passing tests supports the **specified finite tested domains**, not full convergence, accuracy parity, energy savings, or correctness of every layer. Acceptance of numerical risk and future width/SIMD policy remains an architectural decision under [#10](https://github.com/philsupertramp/integers/issues/10), [#15](https://github.com/philsupertramp/integers/issues/15), and [#16](https://github.com/philsupertramp/integers/issues/16).
