//! Experimental optimizer A/B: integer-only error feedback below one stored
//! weight LSB. This does NOT increase weight/activation representation precision.
//! A 2^-24 LSB residue accumulates the same decoded momentum/learning-rate
//! update until a representable integer weight step can be applied.
//! The standard stochastic rounding is still executed exactly ONCE for each
//! parameter to preserve the existing seeded RNG schedule; its sampled delta
//! is deliberately discarded in this opt-in carry arm.

use super::{Dyadic, OptimizerQuantizationStats, Tensor, add, mean_gradient, sgd_step};
use crate::dyadic::stochastic_round;

const CARRY_BITS: u32 = 24;
const DENOMINATOR: i128 = 1i128 << CARRY_BITS;

fn units_from_effective_step(weight: Dyadic, velocity: Dyadic, lr_shift: u32) -> i128 {
    let eff = velocity.s as i64 + lr_shift as i64 - weight.s as i64;
    if eff < 0 {
        // The legacy non-fractional step saturates the integer delta after a
        // capped shift. Preserve that policy for extremely coarse scales.
        let shift = (-eff).min(30) as u32;
        let delta = ((velocity.v as i64) << shift)
            .clamp(i32::MIN as i64, i32::MAX as i64);
        return delta as i128 * DENOMINATOR;
    }
    if eff <= CARRY_BITS as i64 {
        (velocity.v as i128) << (CARRY_BITS as i64 - eff) as u32
    } else {
        let shift = eff - CARRY_BITS as i64;
        // At this point even the largest i32 mantissa is less than 2^-24
        // weight-LSB, so truncation yields no representable fine units.
        if shift >= 32 { 0 } else {
            (velocity.v as i128) / (1i128 << shift as u32)
        }
    }
}

fn update_one(
    weight: &mut Dyadic,
    effective_step: Dyadic,
    carry: &mut i128,
    lr_shift: u32,
    stats: Option<&mut OptimizerQuantizationStats>,
) {
    // Preserve the baseline algorithm's RNG draw at its normal location.
    let _standard_delta_discarded = sgd_step(*weight, effective_step, lr_shift);
    let pending = *carry + units_from_effective_step(*weight, effective_step, lr_shift);
    let proposed = pending / DENOMINATOR; // toward zero, symmetric by sign
    let next = (*weight).v as i128 - proposed;
    let next = next.clamp(i32::MIN as i128, i32::MAX as i128) as i32;
    let applied = (*weight).v as i128 - next as i128;
    // Do not accumulate unbounded carry when storage is saturated.
    *carry = if applied == proposed {
        pending - applied * DENOMINATOR
    } else { 0 };
    weight.v = next;
    if let Some(stats) = stats {
        stats.parameters_seen += 1;
        if effective_step.v == 0 {
            stats.zero_effective_steps += 1;
        } else {
            stats.nonzero_effective_steps += 1;
            if applied == 0 {
                stats.subunit_deferred_steps += 1;
            }
        }
        if proposed != 0 { stats.nonzero_rounded_deltas += 1; }
        if proposed != applied { stats.saturation_blocked_deltas += 1; }
    }
}

/// Same averaging, momentum update, and RNG draw count/order as the existing
/// optimizer. Only the mapping of an effective step to stored weight changes.
pub(super) fn apply_updates_mean_carry(
    weights: &mut [Dyadic],
    grads: &Tensor,
    velocities: &mut [Dyadic],
    carry: &mut [i128],
    lr_shift: u32,
    momentum_shift: Option<u32>,
    samples: usize,
    mut stats: Option<&mut OptimizerQuantizationStats>,
) {
    assert!(samples > 0, "carry optimizer requires accumulated samples");
    assert_eq!(weights.len(), grads.data.len(), "weight / gradient mismatch");
    assert_eq!(weights.len(), velocities.len(), "weight / velocity mismatch");
    assert_eq!(weights.len(), carry.len(), "weight / carry mismatch");
    for (((w, &gradient), v), residue) in weights.iter_mut()
        .zip(&grads.data).zip(velocities.iter_mut()).zip(carry.iter_mut())
    {
        let averaged = mean_gradient(gradient, samples);
        let effective_step = if let Some(m) = momentum_shift {
            let decayed = Dyadic::new(stochastic_round(v.v, m), v.s);
            *v = add(decayed, averaged);
            *v
        } else { averaged };
        update_one(w, effective_step, residue, lr_shift, stats.as_deref_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::apply_updates_mean;
    use crate::rng::{rng_next, seed_rng};

    #[test]
    fn accumulates_fractional_steps_without_changing_learning_rate_or_rng_schedule() {
        let grad = Tensor::from_vec(vec![Dyadic::new(1, 7)], vec![1]);
        let mut base = vec![Dyadic::new(100, 7)];
        let mut fixed = base.clone();
        let mut v_base = vec![Dyadic::new(0, 7)];
        let mut v_fixed = v_base.clone();
        let mut carry = vec![0i128];
        seed_rng(1337);
        for _ in 0..128 {
            apply_updates_mean(&mut base, &grad, &mut v_base, 7, None, 1);
        }
        let base_rng = rng_next();
        seed_rng(1337);
        for _ in 0..128 {
            apply_updates_mean_carry(&mut fixed, &grad, &mut v_fixed,
                &mut carry, 7, None, 1, None);
        }
        let fixed_rng = rng_next();
        assert_eq!(base_rng, fixed_rng, "same stochastic round draws");
        assert_eq!(v_base, v_fixed, "unchanged momentum");
        assert_eq!(fixed[0].v, 99, "128 steps of 1/128 LSB produce one LSB");
        assert!(carry[0].abs() < DENOMINATOR);
    }

    #[test]
    fn observed_and_unobserved_carry_produce_exact_same_weights_rng_and_residue() {
        for momentum in [None, Some(1)] {
            let g = Tensor::from_vec(vec![
                Dyadic::new(3, 7), Dyadic::new(0, 7),
                Dyadic::new(-70, 7),
            ], vec![3]);
            let initial = vec![
                Dyadic::new(19, 7), Dyadic::new(-24, 7),
                Dyadic::new(10, 7)
            ];
            let mut p = initial.clone();
            let mut q = initial;
            let mut pv = vec![Dyadic::new(0, 7); 3];
            let mut qv = pv.clone();
            let mut pc = vec![0i128; 3];
            let mut qc = pc.clone();
            seed_rng(2027);
            for _ in 0..40 {
                apply_updates_mean_carry(&mut p, &g, &mut pv,
                    &mut pc, 7, momentum, 1, None);
            }
            let next_plain = rng_next();
            seed_rng(2027);
            for _ in 0..40 {
                let mut observed = OptimizerQuantizationStats::default();
                apply_updates_mean_carry(&mut q, &g, &mut qv,
                    &mut qc, 7, momentum, 1, Some(&mut observed));
                assert_eq!(observed.parameters_seen, 3);
                assert_eq!(observed.zero_effective_steps + observed.nonzero_effective_steps, 3);
            }
            let next_trace = rng_next();
            assert_eq!(p, q);
            assert_eq!(pv, qv);
            assert_eq!(pc, qc);
            assert_eq!(next_plain, next_trace);
        }
    }

    #[test]
    fn saturating_carry_does_not_build_unbounded_residual() {
        let mut weight = Dyadic::new(i32::MIN, 7);
        let mut carry = 0i128;
        for _ in 0..100 {
            update_one(&mut weight, Dyadic::new(128, 7),
                &mut carry, 0, None);
        }
        assert_eq!(weight.v, i32::MIN);
        assert_eq!(carry, 0);
    }
}
