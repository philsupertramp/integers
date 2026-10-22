//! Regression coverage for the documented Dyadic(v, s) = v * 2^(-s)
//! representation. Reference expectations are calculated independently of
//! the production align/requantize/add implementation.
//! See https://github.com/philsupertramp/integers/issues/12

use integers::dyadic::{
    add, align, div, mul, requantize, signed_bounds, stochastic_round, sub, Dyadic,
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
    assert_eq!(mul(Dyadic::new(4, 2), Dyadic::new(3, 2), 2), Dyadic::new(3, 2));
    assert_eq!(div(Dyadic::new(3, 2), Dyadic::new(3, 2), 4), Dyadic::new(16, 4));
    // Intermediate is i64; converting it to an i32 result must not wrap.
    assert_eq!(mul(Dyadic::new(i32::MAX, 0), Dyadic::new(2, 0), 0).v, i32::MAX);
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
    let mean = (0..draws).map(|_| stochastic_round(-3, 1) as i64).sum::<i64>()
        as f64 / draws as f64;
    assert!((mean + 1.5).abs() < 0.035, "mean {mean}, expected -1.5");
}
