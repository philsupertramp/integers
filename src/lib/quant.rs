//! Per-column quantisation strategies.
//!
//! Every function takes a column of raw `f32` values and returns
//! `(quantised_i32_values, shift)` where `shift` is the scale exponent `s`
//! such that the decoded value is `v · 2⁻ˢ`.
//!
//! All strategies that map to the range `[-127, 127]` return `shift = 7`,
//! since `127 · 2⁻⁷ ≈ 0.992 ≈ 1.0`.

// ─── Identity (no quantisation) ───────────────────────────────────────────────

/// Cast each `f32` to `i32` without any rescaling.  `shift = 0`.
///
/// Use when the data is already in a suitable integer range (e.g. raw pixel
/// bytes `[0, 255]`), or when you want to inspect the raw floating-point bits.
pub fn none_quantize(col: &[f32]) -> (Vec<i32>, i32) {
    (col.iter().map(|&x| x as i32).collect(), 0)
}

// ─── Min-max to [-127, 127] ───────────────────────────────────────────────────

/// Scale each column linearly so that `min → -127` and `max → +127`.
///
/// `shift = 7`:  decoded value `= v · 2⁻⁷ ∈ [-0.992, 0.992]`.
///
/// Constant columns (range = 0) are mapped to 0.
pub fn minmax_quantize(col: &[f32]) -> (Vec<i32>, i32) {
    let min = col.iter().cloned().fold(f32::INFINITY,     f32::min);
    let max = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    let values = if range == 0.0 {
        vec![0i32; col.len()]
    } else {
        col.iter()
            .map(|&x| {
                // [0, 1] → [-127, 127]
                let norm   = (x - min) / range;
                let scaled = norm * 254.0 - 127.0;
                scaled.round().clamp(-127.0, 127.0) as i32
            })
            .collect()
    };
    (values, 7)
}

// ─── Z-score to [-127, 127] ───────────────────────────────────────────────────

/// Standardise each column (z = (x − μ) / σ) and then scale so that ±3σ
/// maps approximately to ±127.
///
/// `shift = 7`:  decoded value `= v · 2⁻⁷ ∈ [-0.992, 0.992]` (for ±3σ inputs).
///
/// Constant columns (std = 0) are mapped to 0.
pub fn standard_score_quantize(col: &[f32]) -> (Vec<i32>, i32) {
    let n    = col.len() as f32;
    let mean = col.iter().sum::<f32>() / n;
    let var  = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std  = var.sqrt();

    let values = if std == 0.0 {
        vec![0i32; col.len()]
    } else {
        col.iter()
            .map(|&x| {
                let z      = (x - mean) / std;
                let scaled = z * (127.0 / 3.0); // ±3σ → ±127
                scaled.round().clamp(-127.0, 127.0) as i32
            })
            .collect()
    };
    (values, 7)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_endpoints() {
        let (q, s) = minmax_quantize(&[0.0, 0.5, 1.0]);
        assert_eq!(s, 7);
        assert_eq!(q[0], -127);
        assert_eq!(q[2],  127);
    }

    #[test]
    fn minmax_constant_column() {
        let (q, _) = minmax_quantize(&[3.0, 3.0, 3.0]);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn zscore_within_bounds() {
        let col: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (q, s) = standard_score_quantize(&col);
        assert_eq!(s, 7);
        assert!(q.iter().all(|&v| v >= -127 && v <= 127));
    }

    #[test]
    fn none_is_identity_cast() {
        let (q, s) = none_quantize(&[1.7, -3.2, 0.0]);
        assert_eq!(s, 0);
        assert_eq!(q, vec![1, -3, 0]);
    }
}
