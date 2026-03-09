use std::sync::OnceLock;
use crate::nn::{XorShift64};
use crate::{Scalar, Numeric};
use crate::debug::{increase_clamp_downcast};

/// A global reference to our dynamically generated lookup table.
static TANH_LUT: OnceLock<[i8; 256]> = OnceLock::new();

/// Generates the LUT based on a specific input scale.
fn generate_tanh_lut_i8(input_scale: f32) -> [i8; 256] {
    let mut lut = [0i8; 256];

    for i in -128..=127 {
        // De-quantize, calculate, and re-quantize
        let float_x = (i as f32) / input_scale;
        let float_y = float_x.tanh();

        // Scale to [-127, 127] and clamp
        let int_y = (float_y * 127.0).round() as i32;
        lut[(i + 128) as usize] = int_y.clamp(-128, 127) as i8;
    }

    lut
}

fn clamp_i8(shifted: i32) -> i8 {
    let mut val = shifted as i32;
    let mut clamp = true;
    if shifted > 127 {
        val = 127;
    } else if shifted < -128 {
        val = -128
    } else { clamp = false; }

    #[cfg(debug_assertions)]
    {
        if clamp {
            increase_clamp_downcast();
        }
    }
    val as i8
}

pub fn mul_mixed_scalar(a: i32, b: i32) -> i32 {
    let result = (a as i64) * (b as i64);
    result.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

pub fn mul_mixed_scalar_scaled(a: i32, b: i32, shift: u32) -> i32 {
    let result = (a as i64) * (b as i64) >> shift;
    result.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}


pub fn dot_product_scalar<S: Scalar>(a: &[S], b: &[S]) -> S::Acc {
    let mut sum: S::Acc = S::Acc::zero();
    for (x, y) in a.iter().zip(b.iter()){
        sum = sum.add(x.mul(*y));
    }
    sum
}

pub fn dot_product_scalar_mixed<S: Scalar>(a: &[S], b: &[S::Acc]) -> S::Acc {
    let mut sum: S::Acc = S::Acc::zero();
    for (x, y) in a.iter().zip(b.iter()){
        sum = sum.add(x.into_acc().mul(*y));
    }
    sum
}

pub fn dot_product_scalar_scaled(a: &[i32], b: &[i32], shift: u32) -> i32 {
    // TODO: implement dot product that computes
    //       sum((a[i] as i64 * b[i] as i64) >> shift)
    let mut sum: i64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum = sum.saturating_add(((*x as i64) * (*y as i64)) >> shift);
    }
    sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Newton-Raphson Method
pub fn isqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}
pub fn isqrt_64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Tanh via piecewise linear approximation over i32 domain.
/// Real tanh saturates at ~ +/- 3, so we map
/// |x| >= 64 (i32 scale) -> +/- 127
/// and linear in between
pub fn tanh_i8(x: i8) -> i8 {
    // Get the table, initializing it on the very first call if necessary.
    let lut = TANH_LUT.get_or_init(|| generate_tanh_lut_i8(127.0));

    let index = (x as i16 + 128) as usize;
    lut[index]
}

pub fn stochastic_downcast(val: i32, shift: u32, rng: &mut XorShift64) -> i32 {
    if shift == 0 { return val; }

    let mask = (1 << shift) - 1;
    let frac = val & mask;

    let thresh = rng.gen_range(1 << shift) as i32;
    let round_bit = if frac.abs() > thresh { 1 } else { 0 };

    let shifted = (val >> shift) + round_bit;

    //clamp_i32(shifted)
    shifted
}

#[cfg(target_arch = "aarch64")]
pub mod arm_neon {
    use std::arch::aarch64::*;

    pub unsafe fn dot_product_neon_raw(a: &[i32], b: &[i32]) -> i32 {
        // 1. Initialize accumulator to zero
        let mut sum_vec = vdupq_n_s32(0);

        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();
        let len = a.len();

        // 2. Main Loop: Process 16 elements (128 bits) per iteration
        for _ in (0..len).step_by(16) {
            let va = vld1q_s8(ptr_a);
            let vb = vld1q_s8(ptr_b);

            // Widening Multiply (The stable workaround for SDOT)
            // Split 128-bit regs into low/high halves
            let va_lo = vget_low_s8(va);
            let vb_lo = vget_low_s8(vb);
            let va_hi = vget_high_s8(va);
            let vb_hi = vget_high_s8(vb);

            // Multiply i32 -> i32
            let prod_lo = vmull_s8(va_lo, vb_lo);
            let prod_hi = vmull_s8(va_hi, vb_hi);

            // Accumulate i32 -> i32
            sum_vec = vpadalq_s16(sum_vec, prod_lo);
            sum_vec = vpadalq_s16(sum_vec, prod_hi);

            ptr_a = ptr_a.add(16);
            ptr_b = ptr_b.add(16);
        }

        // 3. Horizontal Sum (Reduce vector to scalar)
        vaddvq_s32(sum_vec)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    // Kernel tests
    #[test]
    fn test_kernel_clamp_i8(){
        assert_eq!(clamp_i8(10000), 127);
        assert_eq!(clamp_i8(128), 127);
        assert_eq!(clamp_i8(127), 127);
        assert_eq!(clamp_i8(0), 0);
        assert_eq!(clamp_i8(-127), -127);
        assert_eq!(clamp_i8(-128), -128);
        assert_eq!(clamp_i8(-129), -128);
        assert_eq!(clamp_i8(-10000), -128);
    }

    #[test]
    fn test_kernel_mul_mixed_scalar(){
        assert_eq!(mul_mixed_scalar(127, 128), 127 * 128);
    }

    #[test]
    fn test_kernel_isqrt_64(){
        assert_eq!(isqrt_64(4), 2);
        assert_eq!(isqrt_64(10000), 100);
        assert_eq!(isqrt_64(1000000), 1000);
    }

    #[test]
    fn test_generate_tanh_lut_i8() {
        let lut = generate_tanh_lut_i8(127.0);
        let inputs: Vec<i8> = (-128..=127).collect();

        for inp in inputs {
            let expected = ((inp as f32 / 127.0).tanh() * 127.0).round().clamp(-128.0, 127.0) as i8;
            let index = (inp as i16 + 128) as usize;
            assert_eq!(
                lut[index],
                expected, 
                "mismatch at input {inp}: got {}, expected {expected}", tanh_i8(inp)
            );
        }
    }

    #[test]
    fn test_kernel_tanh_i8() {
        let inputs: Vec<i8> = (-128..=127).collect();

        for inp in inputs {
            let expected = ((inp as f32 / 127.0).tanh() * 127.0).round().clamp(-128.0, 127.0) as i8;
            assert_eq!(
                tanh_i8(inp), 
                expected, 
                "mismatch at input {inp}: got {}, expected {expected}", tanh_i8(inp)
            );
        }
    }

    #[test]
    fn test_kernel_dot_product_scalar() {
        let vec1 = [1, 1, 1];
        let vec2 = [1, 1, 1];

        let res = dot_product_scalar(&vec1, &vec2);

        assert_eq!(res, 3);
    }

    #[test]
    fn test_kernel_isqrt() {
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(8), 2);
        assert_eq!(isqrt(7), 2);
        assert_eq!(isqrt(6), 2);
        assert_eq!(isqrt(5), 2);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(1), 1);
    }

    #[test]
    fn test_kernel_stochastic_downcast_deterministic() {
        let mut rng = XorShift64::new(999);

        // Case 1: Exact integer (no fraction)
        // 10.0 (shifted by 1 -> 5)
        let val = 10;
        let shift = 1;
        let res = stochastic_downcast(val, shift, &mut rng);
        assert_eq!(res, 5, "10 >> 1 should always be 5");

        // Case 2: Zero
        let res = stochastic_downcast(0, 5, &mut rng);
        assert_eq!(res, 0);
    }

    #[test]
    fn test_kernel_stochastic_downcast_saturation() {
        let mut rng = XorShift64::new(111);

        // Value: 1000. Shift: 0. Should clamp to 127.
        let res = stochastic_downcast(2147483647, 0, &mut rng);
        assert_eq!(res, 2147483647);

        // Value: -1000. Shift: 0. Should clamp to -128.
        let res = stochastic_downcast(-2147483648, 0, &mut rng);
        assert_eq!(res, -2147483648);
    }

    #[test]
    fn test_kernel_stochastic_downcast_statistics() {
        let mut rng = XorShift64::new(777);
        let n_runs = 100_000;

        // Scenario: Value 5 (binary 101). Shift 1.
        // Real value is 2.5.
        // We expect ~50% 2s and ~50% 3s.
        // Average should be ~2.5.
        let val = 5;
        let shift = 1;

        let mut sum: i64 = 0;
        for _ in 0..n_runs {
            sum += stochastic_downcast(val, shift, &mut rng) as i64;
        }

        let avg = sum as f64 / n_runs as f64;
        println!("Average for 2.5 was: {}", avg);

        // Allow a small margin of error (Standard Error ~ 1/sqrt(N))
        assert!(
            avg > 2.49 && avg < 2.51,
            "Average {} was too far from 2.5",
            avg
        );
    }

    #[test]
    fn test_kernel_stochastic_downcast_negative_statistics() {
        let mut rng = XorShift64::new(888);
        let n_runs = 100_000;

        // Scenario: Value -5 (binary ...11111011). Shift 1.
        // Integer division -5 >> 1 is -3.
        // Real value is -2.5.
        // We expect -3 and -2 mixed.
        let val = -5;
        let shift = 1;

        let mut sum: i64 = 0;
        for _ in 0..n_runs {
            sum += stochastic_downcast(val, shift, &mut rng) as i64;
        }

        let avg = sum as f64 / n_runs as f64;
        println!("Average for -2.5 was: {}", avg);

        assert!(
            avg > -2.51 && avg < -2.49,
            "Average {} was too far from -2.5",
            avg
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_kernel_dot_product_neon() {
        let vec1 = [1i32; 16];
        let vec2 = [1i32; 16];

        let res = unsafe { arm_neon::dot_product_neon_raw(&vec1, &vec2) };

        assert_eq!(res, 16);

        let vec3 = [-1i32; 16];
        let res_neg = unsafe { arm_neon::dot_product_neon_raw(&vec1, &vec3) };
        assert_eq!(res_neg, -16);
    }
}
