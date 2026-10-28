//! Dyadic rational arithmetic for integer neural network training.
//!
//! Every value is a pair `(v, s)` that encodes the rational number `v · 2^{-s}`.
//! Operations preserve the scale interpretation, subject to explicitly
//! documented rounding, saturating storage, and invalid-scale errors.
//! See docs/DYADIC_NUMERICS.md for the exact per-operation contract.
//!
//! # Stochastic rounding performance
//!
//! `stochastic_round` is called on every single multiply, alignment, and
//! requantization.  A forward pass through a 784-neuron layer calls it ~100 000
//! times per sample.  Performance is therefore critical.
//!
//! This module uses a **thread-local XorShift64 RNG** instead of `rand::thread_rng()`:
//! - XorShift64 = 3 bitwise ops vs ChaCha8's 20+ operations
//! - No dynamic dispatch, no atomics, no initialization overhead per call
//! - Bitmask instead of modulo — no integer division
//! - `rand` is not imported here at all; weight initialization in `nn.rs` still uses it
///
/// Reference: https://blog.godesteem.de/posts/integers-improving-mnist/
use crate::rng::{XorShift64, rng_next};

// ─── Core type ────────────────────────────────────────────────────────────────

/// A dyadic rational: the pair `(v, s)` encodes `v · 2^{-s}`.
///
/// The representation is non-unique: `(3, 2)`, `(6, 3)` and `(12, 4)` all
/// encode `3/4`.  `s` is restricted to `u32` (>= 0), so all values satisfy
/// `|decoded| <= |v|`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Dyadic {
    /// Raw integer mantissa.
    pub v: i32,
    /// Scale exponent: larger == finer granularity (smaller unit step).
    pub s: u32,
}

impl Dyadic {
    #[inline]
    pub fn new(v: i32, s: u32) -> Self {
        Self { v, s }
    }

    /// Decode to `f64`: `[[x]] = v · 2^{-s}`.
    /// Decode for diagnostics; very small values may underflow to zero in f64.
    ///
    /// Scaling in bounded chunks avoids wrapping exponents > i32::MAX and
    /// avoids underflowing 2^(-s) before multiplying by the mantissa.
    #[inline]
    pub fn to_f64(self) -> f64 {
        let mut value = self.v as f64;
        let mut remaining = self.s;
        while remaining > 0 && value != 0.0 {
            let step = remaining.min(512);
            value *= 2f64.powi(-(step as i32));
            remaining -= step;
        }
        value
    }

    /// Two representations are equivalent if they encode the same rational.
    ///
    /// Reduce powers of two from each mantissa rather than shifting by the
    /// other operand's exponent. The latter can overflow even an i64.
    pub fn equivalent(self, rhs: Self) -> bool {
        fn canonical(mut v: i32, mut s: u32) -> (i32, u32) {
            if v == 0 {
                return (0, 0);
            }
            let factors = v.trailing_zeros().min(s);
            v >>= factors;
            s -= factors;
            (v, s)
        }
        canonical(self.v, self.s) == canonical(rhs.v, rhs.s)
    }
}

impl std::fmt::Display for Dyadic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}) ≈ {:.4}", self.v, self.s, self.to_f64())
    }
}

// ─── Stochastic rounding ─────────────────────────────────────────────────────

/// `SR(v, k) = ⌊v · 2^{-k}⌋ + 𝕀((v mod 2^{k}) > U)`,  `U ~ 𝒰{0, 2^{k} − 1}`.
///
/// An unbiased estimator of `v · 2^{-k}` before any storage saturation:
/// `E[SR(v, k)] = v / 2^{k}` under ideal uniform PRNG draws.
/// For `k >= 32`, use the wide path rather than discarding rare outcomes.
#[inline]
pub fn stochastic_round(v: i32, k: u32) -> i32 {
    if k == 0 {
        return v;
    }
    if k >= 32 {
        return stochastic_round_i64(v as i64, k);
    }
    // Use upper 32 bits of the 64-bit RNG output — better quality than lower.
    let mask = (1u32 << k) - 1;
    let threshold = (rng_next() >> 32) as u32 & mask; // U ~ Uniform{0, 2^k − 1}
    let floor = v >> k;
    let remainder = (v as u32) & mask; // v mod 2^k  (unsigned)
    floor + i32::from(remainder > threshold)
}

/// Draw a Bernoulli event with ideal probability `magnitude / 2^bits`.
///
/// Sample in 32-bit chunks so even a full-zero chunk is possible with the
/// non-zero-state XorShift64 generator. Early rejection makes huge exponents
/// cheap on average without deterministically eliminating rare outcomes.
/// As with all pseudorandom rounding, unbiasedness assumes uniform draws.
#[inline]
fn bernoulli_power_of_two_using(magnitude: u64, bits: u32, next: &mut impl FnMut() -> u64) -> bool {
    debug_assert!(bits >= 64);
    let mut extra_bits = bits - 64;
    while extra_bits >= 32 {
        if (next() >> 32) != 0 {
            return false;
        }
        extra_bits -= 32;
    }
    if extra_bits > 0 && ((next() >> 32) & ((1u64 << extra_bits) - 1)) != 0 {
        return false;
    }
    // Compose two 32-bit draws rather than one XorShift64 result: the
    // XorShift64 state/output never equals zero, but a composed draw can.
    let draw = ((next() >> 32) << 32) | (next() >> 32);
    draw < magnitude
}

/// Stochastic rounding for `i64` intermediates (used in [`mul`]).
///
/// The output is saturated to i32; an overflowed result cannot be unbiased
/// relative to the original i64 value. The rounding itself is unbiased,
/// including negative values and shifts at/above the word width.
#[inline]
fn stochastic_round_i64_using(v: i64, k: u32, next: &mut impl FnMut() -> u64) -> i32 {
    let rounded = if k == 0 {
        v
    } else if k < 64 {
        let mask = (1u64 << k) - 1;
        let threshold = next() & mask;
        let floor = v >> k;
        let remainder = (v as u64) & mask;
        floor + i64::from(remainder > threshold)
    } else if v == 0 {
        0
    } else if bernoulli_power_of_two_using(v.unsigned_abs(), k, next) {
        v.signum()
    } else {
        0
    };
    rounded.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline]
pub fn stochastic_round_i64(v: i64, k: u32) -> i32 {
    stochastic_round_i64_using(v, k, &mut rng_next)
}

// ─── Scale alignment ─────────────────────────────────────────────────────────

/// Rescale mantissa so `v * 2^(-s)` stays invariant when representable.
///
/// Increasing the exponent makes the resolution *finer*: the mantissa must be
/// left-shifted. If it exceeds i32, saturate rather than wrap. Decreasing the
/// exponent is a lossy coarsening operation and uses stochastic rounding.
/// Returns `(mantissa, overflowed_i32)`.
fn rescale_mantissa(x: Dyadic, target_shift: u32) -> (i32, bool) {
    if target_shift == x.s {
        return (x.v, false);
    }
    if target_shift < x.s {
        let k = x.s - target_shift;
        return (
            if k < 32 {
                stochastic_round(x.v, k)
            } else {
                stochastic_round_i64(x.v as i64, k)
            },
            false,
        );
    }

    let k = target_shift - x.s;
    if k >= 32 {
        return if x.v == 0 {
            (0, false)
        } else if x.v < 0 {
            (i32::MIN, true)
        } else {
            (i32::MAX, true)
        };
    }
    // i32 << 31 fits i64; a wider intermediate detects i32 saturation.
    let wide = (x.v as i64) << k;
    let narrowed = wide.clamp(i32::MIN as i64, i32::MAX as i64);
    (narrowed as i32, wide != narrowed)
}

/// Align a Dyadic to a target exponent, preserving decoded value when possible.
///
/// A larger exponent means a *finer*, not coarser, scale. Saturates on overflow;
/// reducing the exponent uses stochastic rounding.
pub fn align(x: Dyadic, s_star: u32) -> Dyadic {
    Dyadic {
        v: rescale_mantissa(x, s_star).0,
        s: s_star,
    }
}

// ─── Four arithmetic operations ───────────────────────────────────────────────

/// `x_1 \oplus x_2 = (\hat{v}_1 + \hat{v}_2, s^*)`,  `s^* = max(s_1, s_2)`.
pub fn add(x1: Dyadic, x2: Dyadic) -> Dyadic {
    let s = x1.s.max(x2.s);
    Dyadic {
        v: align(x1, s).v.saturating_add(align(x2, s).v),
        s,
    }
}

/// `x_1 \ominus x_2 = (\hat{v}_1 − \hat{v}_2, s^*)`,  `s^* = max(s_1, s_2)`.
pub fn sub(x1: Dyadic, x2: Dyadic) -> Dyadic {
    let s = x1.s.max(x2.s);
    Dyadic {
        v: align(x1, s).v.saturating_sub(align(x2, s).v),
        s,
    }
}

/// `x_1 \otimes x_2 = (SR(v_1 * v_2, q), s_1 + s_2 - q)`.
///
/// The exact exponent is calculated in u64 and must be representable as u32.
/// Reject an unrepresentable negative exponent or exponent overflow rather
/// than silently changing the decoded numeric value. Mantissa narrowing
/// saturates to i32; stochastic rounding applies before narrowing.
#[inline]
fn mul_using(x1: Dyadic, x2: Dyadic, q: u32, next: &mut impl FnMut() -> u64) -> Dyadic {
    let output_shift = (x1.s as u64 + x2.s as u64)
        .checked_sub(q as u64)
        .expect("mul: negative output scale");
    let s = u32::try_from(output_shift).expect("mul: output scale overflow");
    let product = (x1.v as i64) * (x2.v as i64);
    Dyadic {
        v: stochastic_round_i64_using(product, q, next),
        s,
    }
}

pub fn mul(x1: Dyadic, x2: Dyadic, q: u32) -> Dyadic {
    mul_using(x1, x2, q, &mut rng_next)
}

/// Same multiplication and stochastic-rounding contract, drawing from the
/// caller's scoped XorShift64 state instead of acquiring TLS per product.
/// Must only be called from a with_rng_state scope.
#[inline(always)]
pub(crate) fn mul_with_rng(x1: Dyadic, x2: Dyadic, q: u32, rng: &mut XorShift64) -> Dyadic {
    mul_using(x1, x2, q, &mut || rng.next_u64())
}

/// Consume precisely the stochastic draws that an unused dyadic product
/// would have consumed, without calculating its mantissa for 0 <= q < 64.
///
/// This is only for a caller that intentionally discards the **value** of
/// the multiplication (e.g. the input gradients of the first training layer).
/// We still validate its output scale just as mul_using does, even for q=0.
/// For q >= 64 the number of draws depends on the product and PRNG state,
/// so execute the original operation and discard its value instead.
/// This must be called inside a with_rng_state scope.
#[inline(always)]
pub(crate) fn discard_mul_with_rng(x1: Dyadic, x2: Dyadic, q: u32, rng: &mut XorShift64) {
    if q >= 64 {
        let _ = mul_with_rng(x1, x2, q, rng);
        return;
    }

    let output_shift = (x1.s as u64 + x2.s as u64)
        .checked_sub(q as u64)
        .expect("mul: negative output scale");
    let _ = u32::try_from(output_shift).expect("mul: output scale overflow");
    // stochastic_round_i64_using consumes exactly one draw for 1..64
    // regardless of the product (including zero and saturation).
    // q=0 is exact and consumes none.
    if q != 0 {
        rng.next_u64();
    }
}

/// `x_1 / x_2 = (trunc(v_1 * 2^p / v_2), s_1 + p - s_2)`.
///
/// Division truncates toward zero (not stochastic or nearest rounding).
/// The result saturates to i32 on quotient overflow. `p` is allowed up to
/// u32::MAX provided the resulting nonnegative scale fits u32. Invalid
/// scales and division by zero panic explicitly.
pub fn div(x1: Dyadic, x2: Dyadic, p: u32) -> Dyadic {
    assert!(x2.v != 0, "div: division by zero");
    let output_shift = (x1.s as u64 + p as u64)
        .checked_sub(x2.s as u64)
        .expect("div: negative output scale");
    let s = u32::try_from(output_shift).expect("div: output scale overflow");

    // |numerator| >= 2^63 for non-zero inputs at p >= 63, and
    // |denominator| <= 2^31; the quotient cannot fit i32.
    // Avoid a huge shift or overflowing a fixed-width intermediate.
    let quotient = if x1.v == 0 {
        0
    } else if p >= 63 {
        if (x1.v < 0) ^ (x2.v < 0) {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        let shifted = (x1.v as i128) << p;
        (shifted / x2.v as i128).clamp(i32::MIN as i128, i32::MAX as i128) as i32
    };
    Dyadic { v: quotient, s }
}

// ─── Clipping ────────────────────────────────────────────────────────────────

/// Representable bounds for a signed `b`-bit integer.
/// Returns `(−2^(b−1), 2^(b−1) − 1)`.
pub fn signed_bounds(bits: u32) -> (i32, i32) {
    assert!(bits > 0, "signed_bounds: bits must be positive");
    if bits >= 32 {
        return (i32::MIN, i32::MAX);
    }
    (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
}

/// Standard saturation clamp.
#[inline]
pub fn clip(v: i32, q_min: i32, q_max: i32) -> i32 {
    v.clamp(q_min, q_max)
}

// ─── Requantisation ───────────────────────────────────────────────────────────

/// Rescale to `s_target` then clamp into the supplied signed storage bounds.
///
/// Rescaling to a larger exponent requires multiplying the mantissa, while
/// rescaling to a smaller exponent requires stochastic rounding. The clipping
/// flag also records overflow of the intermediate i32 mantissa itself.
pub fn requantize(x: Dyadic, s_target: u32, q_min: i32, q_max: i32) -> (Dyadic, bool) {
    let (scaled, overflowed) = rescale_mantissa(x, s_target);
    let was_clipped = overflowed || scaled < q_min || scaled > q_max;
    (
        Dyadic {
            v: clip(scaled, q_min, q_max),
            s: s_target,
        },
        was_clipped,
    )
}

// ─── Straight-Through Estimator ───────────────────────────────────────────────

/// STE for `clip`: gradient passes through iff `v_y` was within bounds.
pub fn ste_clip(grad: Dyadic, v_y: i32, q_min: i32, q_max: i32) -> Dyadic {
    if v_y >= q_min && v_y <= q_max {
        grad
    } else {
        Dyadic::new(0, grad.s)
    }
}

/// Combined STE for `ℛ`: SR is treated as identity, clip gates the gradient.
pub fn ste_requantize(grad_y: Dyadic, v_y: i32, q_min: i32, q_max: i32) -> Dyadic {
    ste_clip(grad_y, v_y, q_min, q_max)
}

// ─── Tensor ───────────────────────────────────────────────────────────────────

/// A row-major n-dimensional array of `Dyadic` elements.
///
/// Indexing convention: `shape = [d₀, d₁, …, dₙ]`, element `[i₀, i₁, …, iₙ]`
/// lives at `data[i₀·(d₁·…·dₙ) + i₁·(d₂·…·dₙ) + … + iₙ]`.
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub data: Vec<Dyadic>,
    pub shape: Vec<usize>,
}

/// A View into a Tensor. It BORROWS the memory.
/// This is what an iterator yields
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [Dyadic],
    pub shape: &'a [usize],
}

/// A Mutable View into a Tensor.
/// Allows modifying the original Tensor's data in-place.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub data: &'a mut [Dyadic],
    pub shape: &'a [usize],
}

impl<'a> TensorView<'a> {
    /// Materializes the borrowed view into a fully owned Tensor.
    /// This allocates memory and copies the underlying data.
    pub fn to_tensor(&self) -> Tensor {
        Tensor {
            data: self.data.to_vec(),
            shape: self.shape.to_vec(),
        }
    }
}

impl<'a> From<TensorView<'a>> for Tensor {
    fn from(view: TensorView<'a>) -> Self {
        view.to_tensor()
    }
}

/// Iterator yielding read-only views (TensorView)
pub struct TensorIter<'a> {
    chunks: std::slice::Chunks<'a, Dyadic>,
    sub_shape: &'a [usize],
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = TensorView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|chunk| TensorView {
            data: chunk,
            shape: self.sub_shape,
        })
    }
}

/// Iterator yielding mutable views (TensorViewMut)
pub struct TensorIterMut<'a> {
    // ChunksMut guarantees we don't yield overlapping mutable references
    chunks: std::slice::ChunksMut<'a, Dyadic>,
    sub_shape: &'a [usize],
}

impl<'a> Iterator for TensorIterMut<'a> {
    type Item = TensorViewMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|chunk| TensorViewMut {
            data: chunk,
            shape: self.sub_shape,
        })
    }
}

impl Tensor {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: vec![],
        }
    }
    /// Construct from a flat buffer and a shape.  Panics in debug mode if the
    /// buffer length does not match the product of the shape dimensions.
    pub fn from_vec(data: Vec<Dyadic>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor::from_vec: buffer length {} ≠ shape product {}",
            data.len(),
            shape.iter().product::<usize>(),
        );
        Self { data, shape }
    }

    /// Allocate an all-zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            data: vec![Dyadic::new(0, 0); n],
            shape,
        }
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` if the tensor has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrows the ENTIRE Tensor as a read-only View.
    /// This is O(1) and does no allocation.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data, // Borrows the whole Vec as a slice
            shape: &self.shape,
        }
    }

    /// Borrows the ENTIRE Tensor as a mutable View.
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            data: &mut self.data,
            shape: &self.shape,
        }
    }
    /// Returns an iterator over the first dimension (the batch dimension)
    pub fn iter(&self) -> TensorIter<'_> {
        // Calculate how many elements are in one "item" of the batch
        let item_size = if self.shape.len() <= 1 {
            1
        } else {
            self.shape[1..].iter().product()
        };

        // The shape of the yielded views (everything except the batch dimension)
        let sub_shape = if self.shape.is_empty() {
            &[]
        } else {
            &self.shape[1..]
        };

        TensorIter {
            chunks: self.data.chunks(item_size.max(1)),
            sub_shape,
        }
    }

    /// Returns a mutable iterator over the first dimension
    pub fn iter_mut(&mut self) -> TensorIterMut<'_> {
        let item_size = if self.shape.len() <= 1 {
            1
        } else {
            self.shape[1..].iter().product()
        };

        let sub_shape = if self.shape.is_empty() {
            &[]
        } else {
            &self.shape[1..]
        };

        TensorIterMut {
            chunks: self.data.chunks_mut(item_size.max(1)),
            sub_shape,
        }
    }
}
impl FromIterator<Tensor> for Tensor {
    fn from_iter<I: IntoIterator<Item = Tensor>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let mut all_data = Vec::new();
        let mut base_shape = None;
        let mut batch_size = 0;

        // Peel off the first item to get the shape and item size
        if let Some(first) = iter.next() {
            base_shape = Some(first.shape.clone());
            let item_size = first.data.len();

            // THE MAGIC TRICK: Ask the iterator how many items are left,
            // and pre-allocate the EXACT memory needed for the whole batch!
            let (lower_bound_remaining, _) = iter.size_hint();
            all_data.reserve(item_size * (lower_bound_remaining + 1));

            all_data.extend(first.data);
            batch_size += 1;

            for tensor in iter {
                all_data.extend(tensor.data); // No more reallocations!
                batch_size += 1;
            }
        }

        let mut final_shape = vec![batch_size];
        if let Some(shape) = base_shape {
            final_shape.extend(shape);
        }

        Tensor {
            data: all_data,
            shape: final_shape,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;
    const EPS: f64 = 1e-5;

    #[test]
    fn interpretation_map() {
        assert!((Dyadic::new(3, 2).to_f64() - 0.75).abs() < EPS);
    }

    #[test]
    fn non_uniqueness() {
        let (a, b, c) = (Dyadic::new(3, 2), Dyadic::new(6, 3), Dyadic::new(12, 4));
        assert!(a.equivalent(b) && b.equivalent(c) && a.equivalent(c));
    }

    #[test]
    fn align_exact_power_of_two() {
        let a = align(Dyadic::new(4, 0), 2);
        assert_eq!(a.s, 2);
        assert_eq!(a.v, 16);
        assert_eq!(a.to_f64(), 4.0);
    }

    #[test]
    fn add_same_scale() {
        assert_eq!(add(Dyadic::new(3, 4), Dyadic::new(5, 4)), Dyadic::new(8, 4));
    }

    #[test]
    fn mul_scale_arithmetic() {
        let r = mul(Dyadic::new(4, 2), Dyadic::new(3, 2), 2);
        assert_eq!(r.s, 2);
        let step = 2f64.powi(-(r.s as i32));
        assert!((r.to_f64() - 0.75).abs() <= step + EPS);
    }

    #[test]
    fn stochastic_round_unbiased() {
        // SR(1, 1) should average to 0.5 over many samples.
        seed_rng(42);
        let n = 100_000;
        let sum: i32 = (0..n).map(|_| stochastic_round(1, 1)).sum();
        let mean = sum as f64 / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean = {mean:.4}, expected 0.5");
    }

    #[test]
    fn stochastic_round_identity_at_zero() {
        for v in [-100, -1, 0, 1, 100] {
            assert_eq!(stochastic_round(v, 0), v);
        }
    }

    #[test]
    fn requantize_clips() {
        let (lo, hi) = signed_bounds(8);
        let (r, clipped) = requantize(Dyadic::new(1000, 0), 0, lo, hi);
        assert!(clipped);
        assert_eq!(r.v, 127);
    }

    #[test]
    fn ste_passes_in_bounds() {
        let (lo, hi) = signed_bounds(8);
        assert_eq!(ste_requantize(Dyadic::new(42, 4), 50, lo, hi).v, 42);
    }

    #[test]
    fn ste_zeros_when_clipped() {
        let (lo, hi) = signed_bounds(8);
        let out = ste_requantize(Dyadic::new(42, 4), 200, lo, hi);
        assert_eq!(out.v, 0);
        assert_eq!(out.s, 4);
    }
}

#[cfg(test)]
mod scoped_mul_tests {
    use super::*;
    use crate::rng::{rng_next, seed_rng, with_rng_state};

    #[test]
    fn scoped_multiplication_matches_public_mul_and_rng_stream() {
        let values = [i32::MIN, -8193, -127, -1, 0, 1, 127, 8193, i32::MAX];
        let shifts = [0u32, 1, 7, 31, 32, 63, 64, 65, 96, 128];
        let mut cases = Vec::new();
        for &q in &shifts {
            let left_shift = q / 2;
            let right_shift = q - left_shift;
            for &a in &values {
                for &b in &values {
                    cases.push((Dyadic::new(a, left_shift), Dyadic::new(b, right_shift), q));
                }
            }
        }
        // A nonzero result exponent, including mixed-scale operands.
        cases.extend([
            (Dyadic::new(-127, 7), Dyadic::new(19, 9), 7),
            (Dyadic::new(i32::MAX, 14), Dyadic::new(i32::MIN, 21), 31),
            (Dyadic::new(0, 70), Dyadic::new(-1, 61), 64),
        ]);

        for seed in [1, 0x1234_5678, u64::MAX] {
            seed_rng(seed);
            let expected: Vec<_> = cases.iter().map(|&(a, b, q)| mul(a, b, q)).collect();
            let next_expected = rng_next();
            seed_rng(seed);
            let actual: Vec<_> = with_rng_state(|rng| {
                cases
                    .iter()
                    .map(|&(a, b, q)| mul_with_rng(a, b, q, rng))
                    .collect()
            });
            assert_eq!(
                actual, expected,
                "different multiplication with seed {seed}"
            );
            assert_eq!(rng_next(), next_expected, "RNG diverged with seed {seed}");
        }
    }
}
