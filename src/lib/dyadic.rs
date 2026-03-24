//! Dyadic rational arithmetic for integer neural network training.
//!
//! Every value is a pair `(v, s)` that encodes the rational number `v · 2⁻ˢ`.
//! All arithmetic preserves this invariant exactly (add/sub) or approximately
//! (mul/div/requantize), with stochastic rounding used to keep results unbiased.
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
use crate::rng::rng_next;

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
    #[inline] pub fn new(v: i32, s: u32) -> Self { Self { v, s } }

    /// Decode to `f64`: `[[x]] = v · 2^{-s}`.
    #[inline] pub fn to_f64(self) -> f64 { (self.v as f64) * 2f64.powi(-(self.s as i32)) }

    /// Two representations are equivalent if they encode the same rational.
    pub fn equivalent(self, rhs: Self) -> bool {
        (self.v as i64) << rhs.s == (rhs.v as i64) << self.s
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
/// An unbiased estimator of `v · 2^{-k}`: `E[SR(v, k)] = v / 2^{k}`.
/// When `k = 0` it is the identity; when `k >= 32` all bits are shifted out.
#[inline]
pub fn stochastic_round(v: i32, k: u32) -> i32 {
    if k == 0  { return v; }
    if k >= 32 { return 0; }
    // Use upper 32 bits of the 64-bit RNG output — better quality than lower.
    let mask      = (1u32 << k) - 1;
    let threshold = (rng_next() >> 32) as u32 & mask; // U ~ Uniform{0, 2^k − 1}
    let floor     = v >> k;
    let remainder = (v as u32) & mask;         // v mod 2^k  (unsigned)
    floor + i32::from(remainder > threshold)
}

/// Stochastic rounding for `i64` intermediates (used in [`mul`]).
#[inline]
pub fn stochastic_round_i64(v: i64, k: u32) -> i32 {
    if k == 0  { return v as i32; }
    if k >= 64 { return 0; }

    let mask      = (1u64 << k) - 1;
    let threshold = rng_next() & mask;
    let floor     = v >> k;
    let remainder = (v as u64) & mask;
    (floor + i64::from(remainder > threshold)) as i32
}

// ─── Scale alignment ─────────────────────────────────────────────────────────

/// Align `x_i = (v_i, s_i)` to a coarser target scale `s^* >= s_i`.
///
/// `align(x_i, s^*) = (⌊v_i / 2^(s^*−s_i)⌉, s^*)` using stochastic rounding.
/// Lossy: bits shifted out are irrecoverable.
pub fn align(x: Dyadic, s_star: u32) -> Dyadic {
    debug_assert!(s_star >= x.s, "align: target scale must be ≥ source scale");
    Dyadic { v: stochastic_round(x.v, s_star - x.s), s: s_star }
}

// ─── Four arithmetic operations ───────────────────────────────────────────────

/// `x_1 \oplus x_2 = (\hat{v}_1 + \hat{v}_2, s^*)`,  `s^* = max(s_1, s_2)`.
pub fn add(x1: Dyadic, x2: Dyadic) -> Dyadic {
    let s = x1.s.max(x2.s);
    Dyadic { v: align(x1, s).v.saturating_add(align(x2, s).v), s }
}

/// `x_1 \ominus x_2 = (\hat{v}_1 − \hat{v}_2, s^*)`,  `s^* = max(s_1, s_2)`.
pub fn sub(x1: Dyadic, x2: Dyadic) -> Dyadic {
    let s = x1.s.max(x2.s);
    Dyadic { v: align(x1, s).v.saturating_sub(align(x2, s).v), s }
}

/// `x_1 \otimes x_2 = (⌊v_1 \cdot v_2 / 2^q⌉, s_1 + s_2 − q)`.
///
/// Scales add automatically so no alignment is needed, but the raw product
/// `v_1 \cdot v_2` is twice as wide.  `q` rescales it back.
pub fn mul(x1: Dyadic, x2: Dyadic, q: u32) -> Dyadic {
    let product = (x1.v as i64) * (x2.v as i64);
    let s = x1.s.saturating_add(x2.s).saturating_sub(q);
    Dyadic { v: stochastic_round_i64(product, q), s }
}

/// `x_1 \oslash x_2 = (⌊v_1·2^p / v_2⌉, s_1 − s_2 + p)`.
pub fn div(x1: Dyadic, x2: Dyadic, p: u32) -> Dyadic {
    assert!(x2.v != 0, "div: division by zero");
    let shifted = (x1.v as i64) << p;
    let s = (x1.s + p).saturating_sub(x2.s);
    Dyadic { v: (shifted / x2.v as i64) as i32, s }
}

// ─── Clipping ────────────────────────────────────────────────────────────────

/// Representable bounds for a signed `b`-bit integer.
/// Returns `(−2^(b−1), 2^(b−1) − 1)`.
pub fn signed_bounds(bits: u32) -> (i32, i32) {
    if bits >= 32 { return (i32::MIN, i32::MAX); }
    (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
}

/// Standard saturation clamp.
#[inline]
pub fn clip(v: i32, q_min: i32, q_max: i32) -> i32 { v.clamp(q_min, q_max) }

// ─── Requantisation ───────────────────────────────────────────────────────────

/// `ℛ(x, s_t) = (clip(SR(v_x, s_t-s_x), Q_{min}, Q_{max}), s_t)`.
///
/// Combines stochastic rescaling with bit-width enforcement.
/// Returns the quantised value and a flag indicating whether clipping occurred.
pub fn requantize(x: Dyadic, s_target: u32, q_min: i32, q_max: i32) -> (Dyadic, bool) {
    debug_assert!(s_target >= x.s, "requantize: s_target must be >= x.s");
    let rounded     = stochastic_round(x.v, s_target - x.s);
    let was_clipped = rounded < q_min || rounded > q_max;
    (Dyadic { v: clip(rounded, q_min, q_max), s: s_target }, was_clipped)
}

// ─── Straight-Through Estimator ───────────────────────────────────────────────

/// STE for `clip`: gradient passes through iff `v_y` was within bounds.
pub fn ste_clip(grad: Dyadic, v_y: i32, q_min: i32, q_max: i32) -> Dyadic {
    if v_y >= q_min && v_y <= q_max { grad } else { Dyadic::new(0, grad.s) }
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
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data:  Vec<Dyadic>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Construct from a flat buffer and a shape.  Panics in debug mode if the
    /// buffer length does not match the product of the shape dimensions.
    pub fn from_vec(data: Vec<Dyadic>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor::from_vec: buffer length {} ≠ shape product {}",
            data.len(), shape.iter().product::<usize>(),
        );
        Self { data, shape }
    }

    /// Allocate an all-zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { data: vec![Dyadic::new(0, 0); n], shape }
    }

    /// Total number of elements.
    #[inline] pub fn len(&self) -> usize { self.data.len() }

    /// `true` if the tensor has no elements.
    #[inline] pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-5;

    #[test] fn interpretation_map() {
        assert!((Dyadic::new(3, 2).to_f64() - 0.75).abs() < EPS);
    }

    #[test] fn non_uniqueness() {
        let (a, b, c) = (Dyadic::new(3, 2), Dyadic::new(6, 3), Dyadic::new(12, 4));
        assert!(a.equivalent(b) && b.equivalent(c) && a.equivalent(c));
    }

    #[test] fn align_exact_power_of_two() {
        let a = align(Dyadic::new(4, 0), 2);
        assert_eq!(a.s, 2);
        assert_eq!(a.v, 1);
    }

    #[test] fn add_same_scale() {
        assert_eq!(add(Dyadic::new(3, 4), Dyadic::new(5, 4)), Dyadic::new(8, 4));
    }

    #[test] fn mul_scale_arithmetic() {
        let r = mul(Dyadic::new(4, 2), Dyadic::new(3, 2), 2);
        assert_eq!(r.s, 2);
        let step = 2f64.powi(-(r.s as i32));
        assert!((r.to_f64() - 0.75).abs() <= step + EPS);
    }

    #[test] fn stochastic_round_unbiased() {
        // SR(1, 1) should average to 0.5 over many samples.
        seed_rng(42);
        let n = 100_000;
        let sum: i32 = (0..n).map(|_| stochastic_round(1, 1)).sum();
        let mean = sum as f64 / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean = {mean:.4}, expected 0.5");
    }

    #[test] fn stochastic_round_identity_at_zero() {
        for v in [-100, -1, 0, 1, 100] {
            assert_eq!(stochastic_round(v, 0), v);
        }
    }

    #[test] fn requantize_clips() {
        let (lo, hi) = signed_bounds(8);
        let (r, clipped) = requantize(Dyadic::new(1000, 0), 0, lo, hi);
        assert!(clipped);
        assert_eq!(r.v, 127);
    }

    #[test] fn ste_passes_in_bounds() {
        let (lo, hi) = signed_bounds(8);
        assert_eq!(ste_requantize(Dyadic::new(42, 4), 50, lo, hi).v, 42);
    }

    #[test] fn ste_zeros_when_clipped() {
        let (lo, hi) = signed_bounds(8);
        let out = ste_requantize(Dyadic::new(42, 4), 200, lo, hi);
        assert_eq!(out.v, 0);
        assert_eq!(out.s, 4);
    }
}
