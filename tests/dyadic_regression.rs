//! Regression coverage for the documented Dyadic(v, s) = v * 2^(-s)
//! representation. Reference expectations are calculated independently of
//! the production align/requantize/add implementation.
//! See https://github.com/philsupertramp/integers/issues/12

use integers::dyadic::{
    Dyadic, add, align, div, mul, requantize, signed_bounds, stochastic_round,
    stochastic_round_i64, sub,
};
use integers::rng::seed_rng;

fn exact_mantissa_at(value: Dyadic, target_shift: u32) -> Option<i32> {
    // i128 is an independent exact oracle for the small test cases.
    assert!(target_shift >= value.s);
    let wide = (value.v as i128) << (target_shift - value.s);
    i32::try_from(wide).ok()
}

#[test]
fn finer_scale_conversion_preserves_real_value() {
    for source_shift in 0..=6 {
        for target_shift in source_shift..=7 {
            for mantissa in [-127, -13, -1, 0, 1, 7, 127] {
                let input = Dyadic::new(mantissa, source_shift);
                let expected = exact_mantissa_at(input, target_shift).unwrap();
                let actual = align(input, target_shift);
                assert_eq!(
                    actual,
                    Dyadic::new(expected, target_shift),
                    "value={mantissa}/2^{source_shift}, target shift={target_shift}"
                );
                assert_eq!(actual.to_f64(), input.to_f64());
            }
        }
    }
}

#[test]
fn mixed_scale_add_and_sub_match_exact_rational_oracle() {
    for (a, b) in [
        (Dyadic::new(3, 1), Dyadic::new(1, 3)),
        (Dyadic::new(-7, 2), Dyadic::new(3, 4)),
        (Dyadic::new(-2, 0), Dyadic::new(-1, 3)),
        (Dyadic::new(0, 0), Dyadic::new(9, 5)),
    ] {
        let shift = a.s.max(b.s);
        let av = exact_mantissa_at(a, shift).unwrap() as i64;
        let bv = exact_mantissa_at(b, shift).unwrap() as i64;
        assert_eq!(add(a, b), Dyadic::new((av + bv) as i32, shift));
        assert_eq!(sub(a, b), Dyadic::new((av - bv) as i32, shift));
    }
}

#[test]
fn requantize_supports_exact_finer_and_coarser_scales() {
    let (fine, clipped) = requantize(Dyadic::new(-3, 1), 4, i32::MIN, i32::MAX);
    assert_eq!(fine, Dyadic::new(-24, 4));
    assert!(!clipped);

    let (coarse, clipped) = requantize(Dyadic::new(24, 4), 1, i32::MIN, i32::MAX);
    assert_eq!(coarse, Dyadic::new(3, 1));
    assert!(!clipped);
}

#[test]
fn requantize_clips_after_rescaling_and_reports_it() {
    let (lo, hi) = signed_bounds(8);
    let (positive, clipped) = requantize(Dyadic::new(120, 0), 2, lo, hi);
    assert_eq!(positive, Dyadic::new(127, 2));
    assert!(clipped);

    let (negative, clipped) = requantize(Dyadic::new(-120, 0), 2, lo, hi);
    assert_eq!(negative, Dyadic::new(-128, 2));
    assert!(clipped);

    let (within, clipped) = requantize(Dyadic::new(-12, 3), 3, lo, hi);
    assert_eq!(within, Dyadic::new(-12, 3));
    assert!(!clipped);
}

#[test]
fn equivalent_handles_large_shifts_without_overflow() {
    assert!(Dyadic::new(1, 100).equivalent(Dyadic::new(2, 101)));
    assert!(Dyadic::new(-3, 70).equivalent(Dyadic::new(-6, 71)));
    assert!(Dyadic::new(0, 150).equivalent(Dyadic::new(0, 0)));
    assert!(!Dyadic::new(1, 100).equivalent(Dyadic::new(3, 101)));
    assert!(!Dyadic::new(0, 150).equivalent(Dyadic::new(1, 150)));
}

#[test]
fn saturating_scale_conversion_never_wraps() {
    // The exact mantissa exceeds i32; saturation must be observable and
    // never wrap or change sign.
    assert_eq!(align(Dyadic::new(1, 0), 31), Dyadic::new(i32::MAX, 31));
    assert_eq!(align(Dyadic::new(-1, 0), 31), Dyadic::new(i32::MIN, 31));
    assert_eq!(align(Dyadic::new(0, 0), 100), Dyadic::new(0, 100));
    assert_eq!(add(Dyadic::new(i32::MAX, 2), Dyadic::new(1, 2)).v, i32::MAX);
}

#[test]
fn multiplication_and_division_retain_simple_exact_values() {
    // 1 * 3/4 = 3/4, q=2 on the product mantissa.
    assert_eq!(
        mul(Dyadic::new(4, 2), Dyadic::new(3, 2), 2),
        Dyadic::new(3, 2)
    );
    assert_eq!(
        div(Dyadic::new(3, 2), Dyadic::new(3, 2), 4),
        Dyadic::new(16, 4)
    );
    // Intermediate is i64; converting it to an i32 result must not wrap.
    assert_eq!(
        mul(Dyadic::new(i32::MAX, 0), Dyadic::new(2, 0), 0).v,
        i32::MAX
    );
}

#[test]
fn stochastic_rounding_is_seeded_bounded_and_unbiased_for_signed_halves() {
    seed_rng(0x1234_5678);
    let first: Vec<i32> = (0..100).map(|_| stochastic_round(-3, 1)).collect();
    assert!(first.iter().all(|&v| v == -2 || v == -1));
    seed_rng(0x1234_5678);
    let again: Vec<i32> = (0..100).map(|_| stochastic_round(-3, 1)).collect();
    assert_eq!(first, again);

    seed_rng(0x0bad_f00d);
    let draws = 20_000;
    let mean = (0..draws)
        .map(|_| stochastic_round(-3, 1) as i64)
        .sum::<i64>() as f64
        / draws as f64;
    assert!((mean + 1.5).abs() < 0.035, "mean {mean}, expected -1.5");
}

// Issue #12: independent i128 reference for truncating division. The source
// implementation is deliberately not used to form either expected value.
fn reference_div(a: Dyadic, b: Dyadic, precision: u32) -> Dyadic {
    assert_ne!(b.v, 0);
    assert!(precision <= 62);
    let scale = (a.s as u64 + precision as u64)
        .checked_sub(b.s as u64)
        .expect("negative output scale");
    let numerator = (a.v as i128) * (1i128 << precision);
    let quotient = numerator / b.v as i128;
    Dyadic::new(
        quotient.clamp(i32::MIN as i128, i32::MAX as i128) as i32,
        scale as u32,
    )
}

#[test]
fn division_matches_independent_rational_oracle_and_lsb_bound() {
    for a_shift in 0..=5 {
        for b_shift in 0..=5 {
            for precision in 0..=12 {
                if a_shift + precision < b_shift {
                    continue;
                }
                for numerator in [-127, -9, -1, 0, 1, 11, 120] {
                    for denominator in [-13, -7, -1, 1, 3, 17] {
                        let a = Dyadic::new(numerator, a_shift);
                        let b = Dyadic::new(denominator, b_shift);
                        let actual = div(a, b, precision);
                        let expected = reference_div(a, b, precision);
                        assert_eq!(actual, expected, "{a:?} / {b:?}, p={precision}");
                        // For non-saturated quotients, truncation error must
                        // be strictly less than one output LSB. This is an
                        // exact integer/rational comparison (no f64 oracle).
                        let scaled = (numerator as i128) << precision;
                        let residual = scaled - (actual.v as i128) * denominator as i128;
                        if actual.v != i32::MIN && actual.v != i32::MAX {
                            assert!(residual.abs() < (denominator as i128).abs());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn division_extreme_precision_never_wraps_or_panics_for_valid_scales() {
    for precision in [31, 32, 63, 64, 100] {
        assert_eq!(
            div(Dyadic::new(1, 0), Dyadic::new(1, 0), precision),
            Dyadic::new(i32::MAX, precision)
        );
        assert_eq!(
            div(Dyadic::new(-1, 0), Dyadic::new(1, 0), precision),
            Dyadic::new(i32::MIN, precision)
        );
        assert_eq!(
            div(Dyadic::new(1, 0), Dyadic::new(-1, 0), precision),
            Dyadic::new(i32::MIN, precision)
        );
        assert_eq!(
            div(Dyadic::new(0, 0), Dyadic::new(1, 0), precision),
            Dyadic::new(0, precision)
        );
    }
    assert_eq!(
        div(Dyadic::new(i32::MIN, 0), Dyadic::new(-1, 0), 0),
        Dyadic::new(i32::MAX, 0)
    );
}

#[test]
fn arithmetic_rejects_unrepresentable_unsigned_output_scales() {
    assert!(
        std::panic::catch_unwind(|| { div(Dyadic::new(1, 0), Dyadic::new(1, 1), 0) }).is_err(),
        "division must not silently reinterpret negative exponent as zero"
    );

    assert!(
        std::panic::catch_unwind(|| { div(Dyadic::new(1, u32::MAX), Dyadic::new(1, 0), 1) })
            .is_err(),
        "division output exponent overflow must be explicit"
    );

    assert!(
        std::panic::catch_unwind(|| { mul(Dyadic::new(1, u32::MAX), Dyadic::new(1, 1), 0) })
            .is_err(),
        "product exponent overflow must be explicit"
    );

    assert!(
        std::panic::catch_unwind(|| { mul(Dyadic::new(1, 0), Dyadic::new(1, 0), 1) }).is_err(),
        "product output cannot use an unrepresentable negative exponent"
    );
}

#[test]
fn huge_exponents_remain_canonical_and_decode_without_wraparound() {
    assert!(Dyadic::new(i32::MIN, u32::MAX).equivalent(Dyadic::new(i32::MIN / 2, u32::MAX - 1)));
    assert!(!Dyadic::new(1, u32::MAX).equivalent(Dyadic::new(1, 0)));
    assert_eq!(Dyadic::new(1, u32::MAX).to_f64(), 0.0);
    assert_eq!(Dyadic::new(-1, u32::MAX).to_f64(), 0.0);
    assert_eq!(align(Dyadic::new(0, 0), u32::MAX), Dyadic::new(0, u32::MAX));
}

#[test]
fn extreme_signed_stochastic_rounding_is_seeded_and_statistically_unbiased() {
    fn check_i32(v: i32, k: u32, expected: f64) {
        seed_rng(0x8abc_def0_1234_5678);
        let n = 50_000;
        let first: Vec<i32> = (0..64).map(|_| stochastic_round(v, k)).collect();
        seed_rng(0x8abc_def0_1234_5678);
        assert_eq!(
            first,
            (0..64).map(|_| stochastic_round(v, k)).collect::<Vec<_>>()
        );
        seed_rng(0x8abc_def0_1234_5678);
        let mean = (0..n).map(|_| stochastic_round(v, k) as i64).sum::<i64>() as f64 / n as f64;
        assert!(
            (mean - expected).abs() < 0.025,
            "stochastic_round({v}, {k}) mean={mean}, expected={expected}"
        );
    }
    // Old k>=32 shortcut returned zero, biasing both of these by 0.5.
    check_i32(i32::MIN, 32, -0.5);
    check_i32(i32::MAX, 32, 0.5 - 1.0 / 4_294_967_296.0);
    check_i32(i32::MIN, 33, -0.25);

    fn check_i64(v: i64, k: u32, expected: f64) {
        seed_rng(0xfeed_face_1234_5678);
        let n = 50_000;
        let mean = (0..n)
            .map(|_| stochastic_round_i64(v, k) as i64)
            .sum::<i64>() as f64
            / n as f64;
        assert!(
            (mean - expected).abs() < 0.025,
            "stochastic_round_i64({v}, {k}) mean={mean}, expected={expected}"
        );
    }
    check_i64(i64::MIN, 64, -0.5);
    check_i64(i64::MAX, 64, 0.5);
    check_i64(i64::MIN, 65, -0.25);
    assert_eq!(stochastic_round_i64(i64::MAX, 0), i32::MAX);
    assert_eq!(stochastic_round_i64(i64::MIN, 0), i32::MIN);
}

#[test]
fn signed_bounds_rejects_zero_bit_storage() {
    assert!(std::panic::catch_unwind(|| signed_bounds(0)).is_err());
}

#[test]
fn mixed_scale_multiplication_is_within_one_output_lsb_of_exact_product() {
    // Integer rational oracle: the requested post-shift mantissa represents
    // (a.v * b.v) / 2^q, irrespective of the original input exponents.
    seed_rng(0x5678_abcd);
    for a_shift in 0..=6 {
        for b_shift in 0..=6 {
            for q in 0..=(a_shift + b_shift) {
                for a in [-91, -7, -1, 0, 1, 19, 97] {
                    for b in [-31, -3, -1, 0, 1, 7, 57] {
                        let actual = mul(Dyadic::new(a, a_shift), Dyadic::new(b, b_shift), q);
                        assert_eq!(actual.s, a_shift + b_shift - q);
                        let product = a as i128 * b as i128;
                        let scale = 1i128 << q;
                        let scaled = actual.v as i128 * scale;
                        assert!(
                            (scaled - product).abs() < scale || (q == 0 && scaled == product),
                            "{a}/2^{a_shift} * {b}/2^{b_shift}, q={q}: {actual:?}"
                        );
                        if product % scale == 0 {
                            assert_eq!(scaled, product, "exactly representable product");
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        mul(Dyadic::new(i32::MIN, 0), Dyadic::new(-1, 0), 0).v,
        i32::MAX
    );
    assert_eq!(
        mul(Dyadic::new(i32::MIN, 0), Dyadic::new(2, 0), 0).v,
        i32::MIN
    );
}
