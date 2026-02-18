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

    pub fn dot_product_mixed_scalar(a: &[i8], b: &[i16]) -> i32 {
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

        if shifted > 127 { 127 }
        else if shifted < -128 { -128 }
        else { shifted as i8 }
    }

    pub fn update_weights(weights: &mut [i32], grads: &[i16], lr_shift: u32) {
        assert_eq!(weights.len(), grads.len(), "weights and grads must match length!" );

        for (w, g) in weights.iter_mut().zip(grads.iter()) {
            let update = (*g as i32) >> lr_shift;

            *w = w.wrapping_sub(update);
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod arm_neon {
        use std::arch::aarch64::*;

        pub unsafe fn dot_product_neon_raw(a: &[i8], b: &[i8]) -> i32 {
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

                // Multiply i8 -> i16
                let prod_lo = vmull_s8(va_lo, vb_lo); 
                let prod_hi = vmull_s8(va_hi, vb_hi);

                // Accumulate i16 -> i32
                sum_vec = vpadalq_s16(sum_vec, prod_lo);
                sum_vec = vpadalq_s16(sum_vec, prod_hi);

                ptr_a = ptr_a.add(16);
                ptr_b = ptr_b.add(16);
            }

            // 3. Horizontal Sum (Reduce vector to scalar)
            vaddvq_s32(sum_vec)
        }
    }
}

pub struct Linear {
    /// all weights are of Transposed Form [out, in] for memory reasons

    /// Source of truth
    weights_master: Tensor<i32>,
    /// Shadow copy used for dot-product
    weights_storage: Tensor<i8>,
    /// Biases before quantization
    bias: Tensor<i32>,
    /// "hyperparameter" that determines how much we right-shift
    scale_shift: u32,
}

impl Linear {
    pub fn sync_weights(&mut self, rng: &mut XorShift64){
        for idx in 0..self.weights_master.data.len() {
            self.weights_storage.data[idx] = kernels::stochastic_downcast(
                self.weights_master.data[idx],
                self.scale_shift,
                rng
            )
        }
    }

    pub fn forward(&self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        assert_eq!(input.shape[1], self.weights_storage.shape[1], "Input in wrong dimension for weights! {} vs {}", input.shape[1], self.weights_storage.shape[1]);

        let batch_size = input.shape[0];
        let input_dim = input.shape[1];
        let output_size = self.weights_storage.shape[0];
        let mut out = Tensor::new([batch_size, output_size].to_vec());
        for b in 0..batch_size {
            let in_start = b * input_dim;
            let in_end = in_start + input_dim;

            let input_row = &input.data[in_start..in_end];

            for o in 0..output_size {
                let w_start = o * input_dim;
                let w_end = w_start + input_dim;

                let weight_row = &self.weights_storage.data[w_start..w_end];
                let raw_val = kernels::dot_product_scalar(input_row, weight_row);
                let acc = raw_val + self.bias.data[o];
                out.data[b * output_size + o] += kernels::stochastic_downcast(
                    acc,
                    self.scale_shift,
                    rng
                );
                
            }
        }
        return out;
    }

    pub fn backward(&self, input: &Tensor<i8>, grad_output: &Tensor<i16>) -> (Tensor<i16>, Tensor<i16>) {
        let output_size = self.weights_storage.shape[1];
        let input_dim = input.shape[1];
        let batch_size = input.shape[0];

        let mut grad_input = Tensor::new(self.weights_storage.shape);
        let mut grad_weights = Tensor::new(input.shape);
        for o in 0..output_size {
            let in_start = b * input_dim;
            let in_end = in_start + input_dim;
            let grad_output_row = &grad_output.data[in_start..in_end];

            for b in 0..batch_size {
                let w_start = o * input_dim;
                let w_end = w_start + input_dim;

                let w_row = &input.data[w_start..w_end];
                let val = kernels::dot_product_mixed_scalar(
                    &w_row,
                    &grad_output_row
                );
                grad_weights.data[i * input.shape[0] + o] += ((val >> 8) as i16);
            }
        }
        for b in 0..batch_size {
            let in_start = b * input_dim;
            let in_end = in_start + input_dim;
            let grad_output_row = &grad_output.data[in_start..in_end];

            for o in 0..output_size {
                let w_start = o * input_dim;
                let w_end = w_start + input_dim;
                let weight_row = &self.weights_storage.data[w_start..w_end];
                let val = kernels::dot_product_mixed_scalar(
                    &weight_row,
                    &grad_output_row
                );
                for i in 0..input_dim {
                    grad_input.data[b * input.shape[1] + i] += (val >> 8) as i16;
                }
            }
        }

        [grad_input, grad_weights].into()
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
    fn test_optimizer_shift() {
        let mut weights = [1000, 1000];
        let grads = [100, -100];

        kernels::update_weights(&mut weights, &grads, 1);

        assert_eq!(weights[0], 950);
        assert_eq!(weights[1], 1050);
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

    #[test]
    fn test_linear_forward() {
        let mut lin = Linear{
            weights_master: Tensor::new([2, 2].to_vec()),
            weights_storage: Tensor::new([2, 2].to_vec()),
            bias: Tensor::new([1, 2].to_vec()),
            scale_shift: 0
        };
        lin.weights_master.data[0] = 1;
        lin.weights_master.data[3] = 1;
        let mut rng = XorShift64{state: 420};
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new([1, 2].to_vec());
        input.data[0] = 10;
        input.data[1] = 20;
        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.data[0], 10);
        assert_eq!(out.data[1], 20);
    }

    #[test]
    fn test_linear_forward_saturation() {
        let mut lin = Linear{
            weights_master: Tensor::new([2, 2].to_vec()),
            weights_storage: Tensor::new([2, 2].to_vec()),
            bias: Tensor::new([1, 2].to_vec()),
            scale_shift: 0
        };
        lin.weights_master.data[0] = 126;
        lin.weights_master.data[3] = 126;
        let mut rng = XorShift64{state: 420};
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new([1, 2].to_vec());
        input.data[0] = 126;
        input.data[1] = 126;
        let out = lin.forward(&input, &mut rng);

        // clamping around 127
        assert_eq!(out.data[0], 127);
        assert_eq!(out.data[1], 127);
    }

    #[test]
    fn test_linear_forward_shape() {
        let mut lin = Linear{
            weights_master: Tensor::new([3, 2].to_vec()),
            weights_storage: Tensor::new([3, 2].to_vec()),
            bias: Tensor::new([1, 3].to_vec()),
            scale_shift: 0
        };
        let mut rng = XorShift64{state: 420};
        lin.sync_weights(&mut rng);

        let input = Tensor::new([10, 2].to_vec());
        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
    }
}


