use std::fmt;


#[derive(Clone, Debug, PartialEq)]
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

    pub fn mul_mixed_scalar(a: i16, b: i8) -> i32 {
        (a as i32) * (b as i32)
    }

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

    /// Newton-Raphson Method
    pub fn isqrt(n: u32) -> u32 {
        if n == 0 { return 0; }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x
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
        let divisor = 1 << lr_shift;
        for (w, g) in weights.iter_mut().zip(grads.iter()) {
            let update = (*g as i32) / divisor;

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

    /// Source of truth (High Res)
    pub weights_master: Tensor<i32>,
    /// Shadow copy used for dot-product (Low Res)
    pub weights_storage: Tensor<i8>,
    /// Biases before quantization (High Res)
    pub bias: Tensor<i32>,
    /// "hyperparameter" that determines how much we right-shift
    pub scale_shift: u32,

    ///
    pub forward_cache: Option<Tensor<i8>>,
    pub grad_cache: Option<Tensor<i16>>,
    pub bias_grad_cache: Option<Tensor<i16>>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, scale_shift: u32) -> Self {
        Self {
            weights_master: Tensor::new(vec![output_dim, input_dim]),
            weights_storage: Tensor::new(vec![output_dim, input_dim]),
            bias: Tensor::new(vec![output_dim]),
            scale_shift,
            forward_cache: None,
            grad_cache: None,
            bias_grad_cache: None,
        }
    }
    pub fn sync_weights(&mut self, rng: &mut XorShift64){
        for idx in 0..self.weights_master.data.len() {
            self.weights_storage.data[idx] = kernels::stochastic_downcast(
                self.weights_master.data[idx],
                self.scale_shift,
                rng
            )
        }
    }

    pub fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        assert_eq!(input.shape[1], self.weights_storage.shape[1], "Input in wrong dimension for weights! {} vs {}", input.shape[1], self.weights_storage.shape[1]);

        // store input for backward pass
        self.forward_cache = Some(input.clone());

        let batch_size = input.shape[0];
        let input_dim = input.shape[1];
        let output_size = self.weights_storage.shape[0];
        let mut out = Tensor::new(vec![batch_size, output_size]);
        for b in 0..batch_size {
            let in_start = b * input_dim;
            let in_end = in_start + input_dim;

            let input_row = &input.data[in_start..in_end];

            for o in 0..output_size {
                let w_start = o * input_dim;
                let w_end = w_start + input_dim;

                let weight_row = &self.weights_storage.data[w_start..w_end];
                //
                #[cfg(target_arch = "aarch64")]
                let raw_val = unsafe {
                    if input_dim % 16 == 0 { kernels::arm_neon::dot_product_neon_raw(input_row, weight_row)}
                    else { kernels::dot_product_scalar(input_row, weight_row) }
                };
                #[cfg(not(target_arch = "aarch64"))]
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

    pub fn backward(&mut self, grad_output: &Tensor<i16>, gradient_shift: Option<u32>) -> Tensor<i16> {
        let input = self.forward_cache.as_ref().expect("Backward called without forward call!");

        let output_dim = self.weights_storage.shape[0];
        let input_dim = input.shape[1];
        let batch_size = input.shape[0];
        let grad_shift = gradient_shift.unwrap_or(8);

        assert_eq!(grad_output.shape[0], batch_size);
        assert_eq!(grad_output.shape[1], output_dim);

        let mut grad_input = Tensor::<i16>::new(vec![batch_size, input_dim]);
        let mut grad_weights = Tensor::<i16>::new(vec![output_dim, input_dim]);
        let mut grad_bias = Tensor::<i16>::new(vec![output_dim]);

        // compute dW (grad w.r.t Weights)
        for b in 0..batch_size {
            let in_start = b * input_dim;
            let grad_start = b * output_dim;

            for o in 0..output_dim {
                let g_val = grad_output.data[grad_start + o];
                grad_bias.data[o] = grad_bias.data[o].wrapping_add(g_val);
                if g_val == 0 { continue; }

                for i in 0..input_dim {
                    let x_val = input.data[in_start + i];
                    let val = kernels::mul_mixed_scalar(g_val, x_val);

                    let idx = o * input_dim + i;
                    grad_weights.data[idx] = grad_weights.data[idx].wrapping_add(
                        (val >> grad_shift) as i16
                    );
                }
            }
        }

        // compute dX (grad w.r.t Input)
        for b in 0..batch_size {
            let grad_start = b * output_dim;

            for o in 0..output_dim {
                let g_val = grad_output.data[grad_start + o];
                if g_val == 0 { continue; }

                let w_start = o * input_dim;

                for i in 0..input_dim {
                    let w_val = self.weights_storage.data[w_start + i];
                    let val = kernels::mul_mixed_scalar(g_val, w_val);

                    let idx = b * input_dim + i;
                    grad_input.data[idx] = grad_input.data[idx].wrapping_add(
                        (val >> grad_shift) as i16
                    )
                }
            }
        }
        self.grad_cache = Some(grad_weights);
        self.bias_grad_cache = Some(grad_bias);
        grad_input
    }
}

pub struct ReLU {
    forward_cache: Option<Tensor<i8>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            forward_cache: None
        }
    }
    pub fn forward(&mut self, input: &Tensor<i8>, _rng: &mut XorShift64) -> Tensor<i8> {
        self.forward_cache = Some(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for idx in 0..input.data.len() {
            output.data[idx] = if input.data[idx] > 0 { input.data[idx] } else { 0 };
        }
        output
    }
    pub fn backward(&self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self.forward_cache.as_ref().expect("No state registered. Perform forward pass first!");
        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            output.data[o] = if input.data[o] > 0 { grad_output.data[o] } else { 0 };
        }
        output
    }
}

pub enum Layer {
    Linear(Linear),
    ReLU(ReLU),
}

impl Layer {
    pub fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        match self {
            Layer::Linear(l) => l.forward(input, rng),
            Layer::ReLU(r) => r.forward(input, rng),
        }
    }

    pub fn backward(&mut self, grad_output: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16> {
        match self {
            Layer::Linear(l) => l.backward(grad_output, grad_shift),
            Layer::ReLU(r) => r.backward(grad_output, grad_shift),
        }
    }

    pub fn sync_weights(&mut self, rng: &mut XorShift64) {
        if let Layer::Linear(l) = self {
            l.sync_weights(rng);
        }
    }
}

pub trait Optimizer {
    fn step(&mut self, param_idx: usize, weights: &mut [i32], grads: &[i16]);
}

pub struct SGD {
    pub lr_shift: u32,
    pub momentum_shift: Option<u32>,
    pub velocity: Vec<Vec<i32>>,
}

impl SGD {
    pub fn new(lr_shift: u32, momentum_shift: Option<u32>) -> Self {
        Self { lr_shift, momentum_shift, velocity: Vec::new() }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param_idx: usize, weights: &mut [i32], grads: &[i16]) {
        let lr_div = 1 << self.lr_shift;

        if let Some(m_shift) = self.momentum_shift {
            // Lazy init momentum tensor if it doesn't exist
            if self.velocity.len() <= param_idx {
                self.velocity.push(vec![0; weights.len()]);
            }
            
            let m_div = 1 << m_shift;
            let m_vec = &mut self.velocity[param_idx];

            for ((w, m), g) in weights.iter_mut().zip(m_vec.iter_mut()).zip(grads.iter()) {
                let mut decay = *m / m_div;
                if decay == 0 && *m != 0 { decay = m.signum(); }
                *m = m.wrapping_sub(decay).wrapping_add(*g as i32);
                *w = w.wrapping_sub(*m / lr_div);
            }
        } else {
            // Vanilla SGD
            for (w, g) in weights.iter_mut().zip(grads.iter()) {
                *w = w.wrapping_sub((*g as i32) / lr_div);
            }
        }
    }
}

pub struct Adam {
    pub lr_mult: i32,
    pub b1_shift: u32,
    pub b2_shift: u32,
    pub eps: i32,
    pub m: Vec<Vec<i32>>,
    pub v: Vec<Vec<i32>>,
}

impl Adam {
    pub fn new(lr_mult: i32) -> Self {
        Self {
            lr_mult,
            b1_shift: 3, // /approx 0.875
            b2_shift: 4, // /approx 0.9375
            eps: 1,       // to prevent division by 0
            m: Vec::new(),
            v: Vec::new()
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param_idx: usize, weights: &mut [i32], grads: &[i16]){
        if self.m.len() <= param_idx {
            self.m.push(vec![0; weights.len()]);
            self.v.push(vec![0; weights.len()]);
        }

        let m_vec = &mut self.m[param_idx];
        let v_vec = &mut self.v[param_idx];
        let b1_div = 1 << self.b1_shift;
        let b2_div = 1 << self.b2_shift;

        for i in 0..weights.len() {
            let g = grads[i] as i32;
            let g_sq = g.wrapping_mul(g); // delta^2
                                          //
            let m_decay = m_vec[i] / b1_div;
            m_vec[i] = m_vec[i].wrapping_sub(m_decay).wrapping_add(g);

            let v_decay = v_vec[i] / b2_div;
            v_vec[i] = v_vec[i].wrapping_sub(v_decay).wrapping_add(g_sq);

            let denom = kernels::isqrt(v_vec[i].max(0) as u32) as i32 + self.eps;

            let step_size = (m_vec[i] * self.lr_mult) / denom;
            weights[i] = weights[i].wrapping_sub(step_size);
        }
    }
}

pub struct Sequential {
    pub layers: Vec<Layer>,
}

impl Sequential {
    pub fn new() -> Self{
        Self {
            layers: vec![]
        }
    }
    pub fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output, rng);
        }
        output
    }
    pub fn backward(&mut self, grad_output: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16> {
        let mut output = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            output = layer.backward(&output, grad_shift);
        }
        output
    }
    pub fn sync_weights(&mut self, rng: &mut XorShift64) {
        for layer in self.layers.iter_mut() {
            layer.sync_weights(rng)
        }

    }
    pub fn step<O: Optimizer>(&mut self, optim: &mut O) {
        let mut param_idx = 0;
        for layer in self.layers.iter_mut() {
            if let Layer::Linear(l) = layer {
                if let Some(cache) = l.grad_cache.take() {
                    optim.step(param_idx, &mut l.weights_master.data, &cache.data);
                    param_idx += 1;
                }
                if let Some(b_cache) = l.bias_grad_cache.take() {
                    optim.step(param_idx, &mut l.bias.data, &b_cache.data);
                    param_idx += 1;
                }
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
            scale_shift: 0,
            forward_cache: None,
            grad_cache: None,
            bias_grad_cache: None
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
            scale_shift: 0,
            forward_cache: None,
            grad_cache: None,
            bias_grad_cache: None,
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
            scale_shift: 0,
            forward_cache: None,
            grad_cache: None,
            bias_grad_cache: None,
        };
        let mut rng = XorShift64{state: 420};
        lin.sync_weights(&mut rng);

        let input = Tensor::new([10, 2].to_vec());
        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
    }

    #[test]
    fn test_linear_forward_identity() {
        // Linear 2->2, Scale 0 (Identity mapping potential)
        let mut lin = Linear::new(2, 2, 0);
        
        // Identity Matrix in Master Weights [1, 0] / [0, 1]
        lin.weights_master.data[0] = 1; 
        lin.weights_master.data[3] = 1; 
        
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new(vec![1, 2]);
        input.data[0] = 10;
        input.data[1] = 20;

        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.data[0], 10);
        assert_eq!(out.data[1], 20);
    }

    #[test]
    fn test_linear_forward_shape_batch() {
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::new(vec![10, 2]);
        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
        assert_eq!(out.data.len(), 30);
    }

    #[test]
    fn test_linear_backward_shapes() {
        // Batch 2, Input 4 -> Output 3
        let mut lin = Linear::new(4, 3, 0);
        
        let input = Tensor::new(vec![2, 4]); // [Batch, In]
        let grad_out = Tensor::new(vec![2, 3]); // [Batch, Out]

        lin.forward_cache = Some(input.clone());

        let dX = lin.backward(&grad_out, None);
        let dW = lin.grad_cache.unwrap().clone();

        // Check shapes
        assert_eq!(dX.shape, vec![2, 4]); // [Batch, In]
        assert_eq!(dW.shape, vec![3, 4]); // [Out, In]
    }

    #[test]
    fn test_train_linear_regression() {
        // We want to learn the function: y = 3x
        // 1 Input, 1 Output. Scale shift = 0.
        let mut layer = Linear::new(1, 1, 0);
        
        // Initialize Master Weight poorly (start at 10)
        layer.weights_master.data[0] = 10;
        let mut rng = XorShift64::new(42);

        // Dataset
        let x = Tensor::from_vec(vec![1, 2, 3, 4], vec![4, 1]); // Inputs
        let y_target = vec![3, 6, 9, 12];                       // Targets
        
        // Hyperparameters
        let epochs = 20;
        let lr_shift = 4; // Shift right by 4 (approx learning rate of 1/16 = 0.0625)
        let grad_shift = 0; // Don't shrink the gradients here, numbers are small

        println!("--- Starting Integer Training ---");
        for epoch in 0..epochs {
            // 1. Sync weights from i32 -> i8
            layer.sync_weights(&mut rng);

            // 2. Forward Pass
            let preds = layer.forward(&x, &mut rng);

            // 3. Compute Loss & Gradients (Error = Pred - Target)
            let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
            let mut loss = 0;

            for i in 0..4 {
                let error = preds.data[i] as i16 - y_target[i] as i16;
                grad_out.data[i] = error; // The "Gradient" is just the error
                loss += (error as i32) * (error as i32); // MSE
            }

            // 4. Backward Pass
            let _dX = layer.backward(&grad_out, Some(grad_shift));
            let dW = layer.grad_cache.clone().unwrap();

            // 5. Optimizer Step: w = w - (dW >> lr_shift)
            kernels::update_weights(&mut layer.weights_master.data, &dW.data, lr_shift);

            println!("Epoch {:02}: Loss = {:04}, Master Weight = {}, Storage Weight = {}", 
                     epoch, loss, layer.weights_master.data[0], layer.weights_storage.data[0]);
                     
            if loss == 0 {
                println!("Converged early! Epoch {}", epoch);
                break;
            }
        }

        layer.sync_weights(&mut rng);
        assert_eq!(layer.weights_storage.data[0], 3, "Model failed to learn y=3x");
    }

    #[test]
    fn test_relu_forward(){
        let mut relu = ReLU::new();
        let mut input = Tensor::<i8>::new(vec![4, 1]);
        let mut rng = XorShift64::new(42);

        input.data[0] = -5;
        input.data[1] = 0;
        input.data[2] = 5;
        input.data[3] = 127;

        let res = relu.forward(&input, &mut rng);
        assert_eq!(res.data[0], 0);
        assert_eq!(res.data[1], 0);
        assert_eq!(res.data[2], 5);
        assert_eq!(res.data[3], 127);
    }

    #[test]
    fn test_relu_backward(){
        let mut relu = ReLU::new();
        let mut input = Tensor::<i8>::new(vec![4, 1]);
        let mut rng = XorShift64::new(42);

        input.data[0] = -5;
        input.data[1] = 0;
        input.data[2] = 5;
        input.data[3] = 127;

        let _res = relu.forward(&input, &mut rng);

        let mut grad = Tensor::<i16>::new(vec![4, 1]);
        grad.data[0] = 10;
        grad.data[1] = 10;
        grad.data[2] = 10;
        grad.data[3] = 10;

        let res = relu.backward(&grad, Some(8));
        assert_eq!(res.data[0], 0);
        assert_eq!(res.data[1], 0);
        assert_eq!(res.data[2], 10);
        assert_eq!(res.data[3], 10);
    }

    #[test]
    fn test_train_xor_sgd_momentum() {
        let mut rng = XorShift64::new(777);

        let mut l1 = Linear::new(2, 8, 2);
        let mut l2 = Linear::new(8, 1, 2);

        for w in l1.weights_master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30; 
        }
        for w in l2.weights_master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30; 
        }
        for b in l1.bias.data.iter_mut() {
            *b = 5;
        }

        let mut model = Sequential {
            layers: vec![
                Layer::Linear(l1),
                Layer::ReLU(ReLU::new()),
                Layer::Linear(l2),
            ],
        };

        let mut optim = SGD::new(4, Some(2));

        let x = Tensor::from_vec(vec![
            0, 0,
            0, 1,
            1, 0,
            1, 1,
        ], vec![4, 2]);

        let y_target = vec![0, 20, 20, 0];
        let epochs = 8000; 
        
        println!("--- Starting Integer XOR Training with SGD + Momentum ---");

        for epoch in 0..epochs {
            model.sync_weights(&mut rng);
            let preds = model.forward(&x, &mut rng);

            let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
            let mut loss = 0;

            for i in 0..4 {
                let error = preds.data[i] as i16 - y_target[i] as i16;
                grad_out.data[i] = error; 
                loss += (error as i32) * (error as i32); 
            }

            if loss == 0 {
                println!("Converged early at epoch {}!", epoch);
                break;
            }

            model.backward(&grad_out, Some(0)); 
            model.step(&mut optim);
        }

        model.sync_weights(&mut rng);
        let final_preds = model.forward(&x, &mut rng);

        assert!(final_preds.data[0] < 8, "{}", final_preds.data[0]);
        assert!(final_preds.data[3] < 8, "{}", final_preds.data[3]);
        assert!(final_preds.data[1] > 12, "{}", final_preds.data[1]);
        assert!(final_preds.data[2] > 12, "{}", final_preds.data[2]);
    }

    #[test]
    fn test_train_xor_adam() {
        let mut rng = XorShift64::new(777);

        let mut l1 = Linear::new(2, 8, 2);
        let mut l2 = Linear::new(8, 1, 2);

        for w in l1.weights_master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30; 
        }
        for w in l2.weights_master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30; 
        }
        for b in l1.bias.data.iter_mut() {
            *b = 5;
        }

        let mut model = Sequential {
            layers: vec![
                Layer::Linear(l1),
                Layer::ReLU(ReLU::new()),
                Layer::Linear(l2),
            ],
        };

        // NEW: Instantiate our Integer Adam!
        // We set the learning rate multiplier to 2.
        let mut optim = Adam::new(2);

        let x = Tensor::from_vec(vec![
            0, 0,
            0, 1,
            1, 0,
            1, 1,
        ], vec![4, 2]);

        let y_target = vec![0, 20, 20, 0];
        let epochs = 800; 
        
        println!("--- Starting Integer XOR Training with ADAM ---");

        for epoch in 0..epochs {
            model.sync_weights(&mut rng);
            let preds = model.forward(&x, &mut rng);

            let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
            let mut loss = 0;

            for i in 0..4 {
                let error = preds.data[i] as i16 - y_target[i] as i16;
                grad_out.data[i] = error; 
                loss += (error as i32) * (error as i32); 
            }

            if epoch % 50 == 0 {
                println!("Epoch {:03}: Loss = {:05}, Preds: [{}, {}, {}, {}]", 
                         epoch, loss, preds.data[0], preds.data[1], preds.data[2], preds.data[3]);
            }

            if loss == 0 {
                println!("Converged early at epoch {}!", epoch);
                break;
            }

            model.backward(&grad_out, Some(0)); 
            
            // NEW: Pass the optimizer to the model step
            model.step(&mut optim);
        }

        model.sync_weights(&mut rng);
        let final_preds = model.forward(&x, &mut rng);
        let p00 = final_preds.data[0]; let p01 = final_preds.data[1];
        let p10 = final_preds.data[2]; let p11 = final_preds.data[3];

        println!("Final XOR Evaluation: 0,0->{} | 0,1->{} | 1,0->{} | 1,1->{}", p00, p01, p10, p11);

        assert!(p00 < 8, "0,0 failed: expected low, got {}", p00);
        assert!(p11 < 8, "1,1 failed: expected low, got {}", p11);
        assert!(p01 > 12, "0,1 failed: expected high, got {}", p01);
        assert!(p10 > 12, "1,0 failed: expected high, got {}", p10);
    }}


