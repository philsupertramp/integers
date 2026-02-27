//! Demonstrates how the gradient signal dies at depth when shifts are misconfigured.
//!
//! # Network
//!
//! ```text
//!   input(all 2) → L1(4→8) → L2(8→8) → L3(8→4) → grad_out(all 100)
//! ```
//!
//! All storage weights are 20 in both scenarios.
//!
//! # Exact arithmetic (overflow scenario, scale_shift=0, grad_shift=0)
//!
//! ## Forward pass — activations saturate immediately
//!
//! ```text
//! L1 output:  dot([2;4], [20;4]) = 160  →  clamp_i8(160) = 127  (h1)
//! L2 output:  dot([127;8], [20;8]) = 20320  →  127               (h2)
//! ```
//!
//! ## Backward pass — two distinct failure modes
//!
//! ### L3 (8→4): grad_input back to L2
//!
//! ```text
//! dx = 100 × 20 = 2000         fits in i16
//! sum over 4 outputs × 2 batch = 8 × 2000  (not yet overflowing)
//! g2 ≈ 8000  ← signal intact here
//! ```
//!
//! ### L2 (8→8): weight grad (failure mode 1 — wrapping cast → saturation)
//!
//! ```text
//! cached input = h1 = 127 (saturated forward activation!)
//! dw  = 8000 × 127 = 1,016,000
//! (1,016,000 >> 0) as i16  →  1,016,000 mod 65,536 = 32,960
//!                           →  32,960 as i16 = -32,576  (wraps negative!)
//! batch=2: saturating_add(-32,576, -32,576) = -32,768
//! L2 weight grads: SATURATED at -32,768 (wrong sign, max magnitude)
//! ```
//!
//! ### L2 (8→8): grad_input back to L1 (cascade)
//!
//! ```text
//! dx = 8000 × 20 = 160,000
//! (160,000 >> 0) as i16  →  160,000 mod 65,536 = 28,928  (positive)
//! sum over 8 outputs → saturating_add saturates at 32,767
//! g1 = 32,767  ← max i16, correct sign but wrong magnitude
//! ```
//!
//! ### L1 (4→8): weight grad (failure mode 2 — wrapping to near-zero)
//!
//! ```text
//! cached input = original input = 2
//! dw  = 32,767 × 2 = 65,534
//! (65,534 >> 0) as i16  →  65,534 - 65,536 = -2  (wraps to -2!)
//! batch=2: -2 + -2 = -4
//! L1 weight grads: mean |g| = 4  — near zero, CORRUPTED (wrong sign too)
//! ```
//!
//! The cascade: forward saturation (h1=127) amplifies dw in L2 backward →
//! truncating `as i16` cast wraps it negative → that flows into L1 → another
//! wrapping cast leaves L1 with near-zero and wrong-signed gradients.
//!
//! # Healthy scenario (scale_shift=4, grad_shift=8)
//!
//! Activations are kept in a sane range by the shifts, so no truncation wrapping
//! occurs. All layers receive directional gradient signal with comparable magnitude.

#[cfg(debug_assertions)]
use integers::debug::{get_overflow_stats, reset_overflow_stats};
use integers::nn::{Linear, Module};
use integers::nn::optim::{SGDConfig};
use integers::{Tensor, XorShift64};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Mean absolute value of weight gradients.
/// Near-zero means no meaningful update signal at that layer.
fn mean_abs_grad(layer: &Linear) -> f64 {
    match &layer.weights.grads {
        None => 0.0,
        Some(g) if g.data.is_empty() => 0.0,
        Some(g) => {
            g.data.iter().map(|&x| x.unsigned_abs() as f64).sum::<f64>()
                / g.data.len() as f64
        }
    }
}

/// Fraction of weight gradients at the i16 saturation boundary (|value| ≥ 32,767).
/// Saturated gradients carry maximum magnitude but the sign may be wrong.
fn sat_frac(layer: &Linear) -> f64 {
    match &layer.weights.grads {
        None => 0.0,
        Some(g) if g.data.is_empty() => 0.0,
        Some(g) => {
            let sat = g.data.iter().filter(|&&x| x.unsigned_abs() >= 32_767).count();
            sat as f64 / g.data.len() as f64
        }
    }
}

// ─── scenario runner ─────────────────────────────────────────────────────────

/// Runs one forward + backward pass and returns per-layer diagnostics:
/// `[(sat_frac, mean_abs_grad); 3]` ordered L3 → L2 → L1 (deepest).
///
/// Storage weights are forced to exactly 20 by setting `master = 20 × 2^scale_shift`.
/// No fractional bits means the stochastic downcast is fully deterministic.
fn run_scenario(
    scale_shift: u32,
    grad_shift: u32,
    input: &Tensor<i8>,
    grad_out: &Tensor<i16>,
) -> [(f64, f64); 3] {
    const STORAGE_WEIGHT: i32 = 20;
    let master_weight = STORAGE_WEIGHT * (1i32 << scale_shift);

    let mut rng = XorShift64::new(0xDEAD_BEEF);

    let mut l1 = Linear::new(4, 8, scale_shift);
    let mut l2 = Linear::new(8, 8, scale_shift);
    let mut l3 = Linear::new(8, 4, scale_shift);

    for w in l1.weights.master.data.iter_mut() { *w = master_weight; }
    for w in l2.weights.master.data.iter_mut() { *w = master_weight; }
    for w in l3.weights.master.data.iter_mut() { *w = master_weight; }

    l1.sync_weights(&mut rng);
    l2.sync_weights(&mut rng);
    l3.sync_weights(&mut rng);

    let h1 = l1.forward(&input, scale_shift, &mut rng);
    let h2 = l2.forward(&h1, l1.weights.output_shift.expect("No forward called."), &mut rng);
    let _  = l3.forward(&h2, l2.weights.output_shift.expect("No forward called."), &mut rng);

    let g2 = l3.backward(grad_out, Some(grad_shift));
    let g1 = l2.backward(&g2,      Some(grad_shift));
    let _  = l1.backward(&g1,      Some(grad_shift));

    [
        (sat_frac(&l3), mean_abs_grad(&l3)),
        (sat_frac(&l2), mean_abs_grad(&l2)),
        (sat_frac(&l1), mean_abs_grad(&l1)),
    ]
}

// ─── tests ───────────────────────────────────────────────────────────────────

/// Verifies the two distinct failure modes and measures the signal collapse.
///
/// The signal ratio L3/L1 is the headline metric:
///   overflow  → > 1000× (L1 gets essentially nothing)
///   healthy   → < 5×    (all layers receive proportional signal)
#[test]
#[cfg(debug_assertions)]
fn test_overflow_kills_deep_gradient_signal() {
    let input    = Tensor::from_vec(vec![2i8;    2 * 4], vec![2, 4]);
    let grad_out = Tensor::from_vec(vec![100i16; 2 * 4], vec![2, 4]);

    reset_overflow_stats();
    let overflow = run_scenario(0, 0, &input, &grad_out);
    let overflow_clamps = get_overflow_stats().downcast_clamps;

    reset_overflow_stats();
    let healthy  = run_scenario(4, 8, &input, &grad_out);
    let healthy_clamps = get_overflow_stats().downcast_clamps;

    let overflow_ratio = overflow[0].1 / overflow[2].1.max(0.1);
    let healthy_ratio  = healthy[0].1  / healthy[2].1.max(0.1);

    println!();
    println!("┌{:─<66}┐", "");
    println!("│{:^66}│", " GRADIENT SIGNAL CASCADE — OVERFLOW vs HEALTHY ");
    println!("├{:─<66}┤", "");
    println!("│  {:<24} {:>18} {:>18}  │", "metric", "overflow", "healthy");
    println!("│  {:<24} {:>18} {:>18}  │", "downcast clamps", overflow_clamps, healthy_clamps);
    println!("├{:─<66}┤", "");
    println!("│  {:<10} {:<14} {:>7} {:>8}   {:>7} {:>8}  │",
        "layer", "failure mode", "sat%", "mean|g|", "sat%", "mean|g|");
    println!("├{:─<66}┤", "");
    for (name, mode, ov, he) in [
        ("L3 (loss)",  "—",            overflow[0], healthy[0]),
        ("L2",         "wrap→sat",     overflow[1], healthy[1]),
        ("L1 (deep)",  "wrap→zero",    overflow[2], healthy[2]),
    ] {
        println!("│  {:<10} {:<14} {:>6.0}% {:>8.0}   {:>6.0}% {:>8.0}  │",
            name, mode,
            ov.0 * 100.0, ov.1,
            he.0 * 100.0, he.1,
        );
    }
    println!("├{:─<66}┤", "");
    println!("│  {:<24} {:>17.0}×  {:>17.1}×  │",
        "signal ratio L3/L1", overflow_ratio, healthy_ratio);
    println!("└{:─<66}┘", "");
    println!();

    // ── assertions ────────────────────────────────────────────────────────────

    // [1] Overflow produces more forward clamping — confirms the forward
    //     pass is already saturating before the backward even starts.
    assert!(
        overflow_clamps > healthy_clamps,
        "Overflow should produce more downcast clamps ({overflow_clamps} vs {healthy_clamps})",
    );

    // [2] L3 retains strong signal — the cascade hasn't reached it.
    //     Analytical: mean|g| = 4 outputs × batch=2 × (100 × input_cached) / 8 params ≈ 25,400.
    assert!(
        overflow[0].1 > 10_000.0,
        "L3 should retain strong signal in overflow (mean|g|={:.0})",
        overflow[0].1,
    );

    // [3] L2 weight grads are SATURATED — failure mode 1.
    //     dw = 8000 × 127 = 1,016,000 → as i16 wraps to -32,576 → saturates.
    assert!(
        overflow[1].0 > 0.9,
        "L2 should be saturated in overflow (sat={:.0}%)",
        overflow[1].0 * 100.0,
    );

    // [4] L1 weight grads are NEAR ZERO — failure mode 2.
    //     NOT saturated. The wrapping cast gives 65,534 as i16 = -2,
    //     then batch=2 → mean|g| ≈ 4. True gradient should be large and positive.
    assert!(
        overflow[2].1 < 10.0,
        "L1 grads should be near-zero (corrupted by wrapping cast), got mean|g|={:.1}",
        overflow[2].1,
    );

    // [5] Healthy has no saturation anywhere.
    for (sat, name) in healthy.iter().map(|s| s.0).zip(["L3", "L2", "L1"]) {
        assert!(
            sat < 0.1,
            "Healthy {name} should have no saturation (sat={:.0}%)",
            sat * 100.0,
        );
    }

    // [6] The headline metric: signal ratio L3/L1 exceeds 1000× in overflow.
    //     L1 is not just receiving less signal — it receives near-zero signal.
    assert!(
        overflow_ratio > 1_000.0,
        "Overflow signal ratio L3/L1 should exceed 1000× (got {:.0}×)",
        overflow_ratio,
    );

    // [7] Healthy ratio stays below 5× — all layers are proportionally updated.
    assert!(
        healthy_ratio < 5.0,
        "Healthy signal ratio L3/L1 should be proportional (got {:.1}×)",
        healthy_ratio,
    );
}

/// Confirms the diagnostic is reflected in actual weight updates, not just
/// gradient statistics — after one backward+step, L1 weights barely change
/// in the overflow scenario while the healthy scenario produces real movement.
#[test]
#[cfg(debug_assertions)]
fn test_overflow_weight_update_is_wrong_at_depth() {
    let input    = Tensor::from_vec(vec![2i8;    2 * 4], vec![2, 4]);
    let grad_out = Tensor::from_vec(vec![100i16; 2 * 4], vec![2, 4]);

    const STORAGE_WEIGHT: i32 = 20;

    let run_and_measure = |scale_shift: u32, grad_shift: u32| -> f64 {
        let master_weight = STORAGE_WEIGHT * (1i32 << scale_shift);
        let mut rng = XorShift64::new(0xDEAD_BEEF);
        // lr_shift=0 means lr=1 — maximises the visible weight delta so
        // even tiny gradients produce measurable (or not) updates.
        let optim = SGDConfig::new().with_learn_rate(0.5);

        let mut l1 = Linear::new(4, 8, scale_shift);
        let mut l2 = Linear::new(8, 8, scale_shift);
        let mut l3 = Linear::new(8, 4, scale_shift);

        for w in l1.weights.master.data.iter_mut() { *w = master_weight; }
        for w in l2.weights.master.data.iter_mut() { *w = master_weight; }
        for w in l3.weights.master.data.iter_mut() { *w = master_weight; }

        let before: Vec<i32> = l1.weights.master.data.clone();

        l1.sync_weights(&mut rng);
        l2.sync_weights(&mut rng);
        l3.sync_weights(&mut rng);

        let h1 = l1.forward(&input, scale_shift, &mut rng);
        let h2 = l2.forward(&h1, l1.weights.output_shift.expect("No forward called."), &mut rng);
        let _  = l3.forward(&h2, l2.weights.output_shift.expect("No forward called."), &mut rng);

        let g2 = l3.backward(&grad_out, Some(grad_shift));
        let g1 = l2.backward(&g2,       Some(grad_shift));
        let _  = l1.backward(&g1,       Some(grad_shift));

        l1.step(&optim);
        l2.step(&optim);
        l3.step(&optim);

        l1.weights.master.data.iter()
            .zip(before.iter())
            .map(|(a, b)| (a - b).unsigned_abs() as f64)
            .sum::<f64>() / l1.weights.master.data.len() as f64
    };

    let overflow_change = run_and_measure(0, 0);
    let healthy_change  = run_and_measure(4, 8);

    println!();
    println!("L1 mean weight Δ — overflow: {overflow_change:.1}  healthy: {healthy_change:.1}");
    println!();

    // overflow: L1 grad ≈ -4 → weight moves by 4 (tiny, wrong direction)
    assert!(
        overflow_change < 10.0,
        "Overflow L1 weight change should be near-zero (got {overflow_change:.1})",
    );

    // healthy: proper gradient flows → weight moves substantially
    assert!(
        healthy_change > overflow_change * 5.0,
        "Healthy L1 should update much more than overflow \
         ({healthy_change:.1} vs {overflow_change:.1})",
    );
}
