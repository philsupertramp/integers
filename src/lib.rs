use std::fmt;


#[derive(Debug)]
#[derive(Clone)]
pub struct Tensor<T>
where T: Clone + Copy + fmt::Debug + Default
{
    /// Flattened data storage (Row-Major contiguous layout)
    pub data: Vec<T>,
    /// Dimension of the tensor, e.g. [batch, input_dim]
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where T: Clone + Copy + fmt::Debug + Default
{
    pub fn new(shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        Self {
            data: vec![T::default(); total_elements],
            shape,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Shape {:?} does not match data len {}",
            shape,
            data.len()
        );
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct XorShift64 {
    pub state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // edge case handleing for state = 0
        let state = if seed == 0 { 0xCAFEBABE } else { seed };
        Self { state }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Random value generator for value in range [0, range)
    #[inline(always)]
    pub fn gen_range(&mut self, range: u32) -> u32 {
        (self.next() as u32) % range
    }
}

pub mod kernels {
    use super::*;

    pub fn dot_product_scalar(a: &[i8], b: &[i8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum()
    }

    pub fn stochastic_downcast(val: i32, shift: u32, rng: &mut XorShift64) -> i8 {
        let mask = (1 << shift) - 1;
        let frac = val & mask;

        let thresh = rng.gen_range(1 << shift) as i32;
        let round_bit = if frac.abs() > thresh { 1 } else { 0 };

        let shifted = (val >> shift) + round_bit;

        if shifted > 127 {
            127
        } else if shifted < -128 {
            -128
        } else {
            shifted as i8
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod arm_neon {
        use std::arch::aarch64::*;

        pub unsafe fn dot_product_neon_raw(a: &[i8], b: &[i8]) -> i32 {
            let len = a.len();
            // In a raw kernel, we trust the caller to pad, but debug_assert helps dev.
            debug_assert_eq!(len % 16, 0, "NEON raw kernel len must be multiple of 16");

            unsafe {
                // 1. Initialize the accumulator vector to [0, 0, 0, 0]
                let mut sum_vec = vdupq_n_s32(0);

                // 2. Iterate via pointers for maximum efficiency
                let mut ptr_a = a.as_ptr();
                let mut ptr_b = b.as_ptr();

                // Unrolling is handled well by LLVM.
                // WE USE A STABLE FALLBACK HERE instead of SDOT (vdotq_s32)
                // to ensure this compiles on stable Rust without 'dotprod' feature gates.
                for _ in (0..len).step_by(16) {
                    // Load 16 x i8 elements (128 bits)
                    let va = vld1q_s8(ptr_a);
                    let vb = vld1q_s8(ptr_b);

                    // Strategy: Widening Multiply (vmull) -> Pairwise Add (vpadal)
                    
                    // Split 128-bit vectors into Low/High 64-bit halves (8 elements each)
                    let va_lo = vget_low_s8(va);
                    let vb_lo = vget_low_s8(vb);
                    let va_hi = vget_high_s8(va);
                    let vb_hi = vget_high_s8(vb);

                    // Multiply: i8 -> i16 results
                    // vmull_s8: Multiplies two int8x8 vectors into an int16x8 vector
                    let prod_lo = vmull_s8(va_lo, vb_lo); 
                    let prod_hi = vmull_s8(va_hi, vb_hi);

                    // Accumulate: i16 -> i32
                    // vpadalq_s16: Pairwise adds int16s and accumulates into int32x4
                    // [x0+x1, x2+x3, x4+x5, x6+x7]
                    sum_vec = vpadalq_s16(sum_vec, prod_lo);
                    sum_vec = vpadalq_s16(sum_vec, prod_hi);

                    ptr_a = ptr_a.add(16);
                    ptr_b = ptr_b.add(16);
                }

                // 3. Horizontal Sum: Sum the 4 x i32 lanes into a single scalar
                vaddvq_s32(sum_vec)
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_determinism() {
        let mut rng1 = XorShift64::new(12345);
        let mut rng2 = XorShift64::new(12345);
        
        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
    }

    #[test]
    fn test_dot_product_scalar() {
        let vec1 = [1, 1, 1];
        let vec2 = [1, 1, 1];

        let res = kernels::dot_product_scalar(&vec1, &vec2);

        assert_eq!(res, 3);
    }

    #[test]
    fn test_stochastic_downcast_deterministic() {
        let mut rng = XorShift64::new(999);
        
        // Case 1: Exact integer (no fraction)
        // 10.0 (shifted by 1 -> 5)
        let val = 10; 
        let shift = 1; 
        let res = kernels::stochastic_downcast(val, shift, &mut rng);
        assert_eq!(res, 5, "10 >> 1 should always be 5");

        // Case 2: Zero
        let res = kernels::stochastic_downcast(0, 5, &mut rng);
        assert_eq!(res, 0);
    }

    #[test]
    fn test_stochastic_downcast_saturation() {
        let mut rng = XorShift64::new(111);
        
        // Value: 1000. Shift: 0. Should clamp to 127.
        let res = kernels::stochastic_downcast(1000, 0, &mut rng);
        assert_eq!(res, 127);

        // Value: -1000. Shift: 0. Should clamp to -128.
        let res = kernels::stochastic_downcast(-1000, 0, &mut rng);
        assert_eq!(res, -128);
    }

    #[test]
    fn test_stochastic_downcast_statistics() {
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
            sum += kernels::stochastic_downcast(val, shift, &mut rng) as i64;
        }

        let avg = sum as f64 / n_runs as f64;
        println!("Average for 2.5 was: {}", avg);
        
        // Allow a small margin of error (Standard Error ~ 1/sqrt(N))
        assert!(avg > 2.49 && avg < 2.51, "Average {} was too far from 2.5", avg);
    }

    #[test]
    fn test_stochastic_downcast_negative_statistics() {
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
            sum += kernels::stochastic_downcast(val, shift, &mut rng) as i64;
        }

        let avg = sum as f64 / n_runs as f64;
        println!("Average for -2.5 was: {}", avg);

        assert!(avg > -2.51 && avg < -2.49, "Average {} was too far from -2.5", avg);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_neon() {
        let vec1 = [1i8; 16];
        let vec2 = [1i8; 16];

        let res = unsafe { kernels::arm_neon::dot_product_neon_raw(&vec1, &vec2 ) };

        assert_eq!(res, 16);

        let vec3 = [-1i8; 16];
        let res_neg = unsafe { kernels::arm_neon::dot_product_neon_raw(&vec1, &vec3) };
        assert_eq!(res_neg, -16);
    }
}


