use std::fmt;
use std::sync::OnceLock;

/// A global reference to our dynamically generated lookup table.
static TANH_LUT: OnceLock<[i8; 256]> = OnceLock::new();

/// Generates the LUT based on a specific input scale.
fn generate_tanh_lut(input_scale: f32) -> [i8; 256] {
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

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default,
{
    /// Flattened data storage (Row-Major contiguous layout)
    pub data: Vec<T>,
    /// Dimension of the tensor, e.g. [batch, input_dim]
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default,
{
    pub fn new(shape: Vec<usize>) -> Self {
        let mut total_elements: usize = 0;

        if !shape.is_empty() {
            total_elements = shape.iter().product();
        }
        Self {
            data: vec![T::default(); total_elements],
            shape,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let mut expected = 0;
        if !shape.is_empty() {
            expected = shape.iter().product();
        }
        assert_eq!(
            data.len(),
            expected,
            "Tensor::from_vec: Shape {:?} does not match data len {}",
            shape,
            data.len()
        );
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
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

    /// Tanh via piecewise linear approximation over i8 domain.
    /// Real tanh saturates at ~ +/- 3, so we map
    /// |x| >= 64 (i8 scale) -> +/- 127
    /// and linear in between
    pub fn tanh_i8(x: i8) -> i8 {
        // Get the table, initializing it on the very first call if necessary.
        let lut = TANH_LUT.get_or_init(|| generate_tanh_lut(127.0));

        let index = (x as i16 + 128) as usize;
        lut[index]
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

pub trait OptimizerConfig {
    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState);
    fn init_state(&self, len: usize) -> OptimizerState;
}

pub enum OptimizerState {
    None,
    SGD { velocity: Vec<i32> },
    Adam { m: Vec<i32>, v: Vec<i64> },
}

// SGD
pub struct SGDConfig {
    pub lr_shift: u32,
    pub momentum_shift: Option<u32>,
}

impl SGDConfig {
    pub fn new(lr_shift: u32, momentum_shift: Option<u32>) -> Self {
        Self {
            lr_shift,
            momentum_shift,
        }
    }
}

impl OptimizerConfig for SGDConfig {
    fn init_state(&self, len: usize) -> OptimizerState {
        match self.momentum_shift {
            Some(_) => OptimizerState::SGD {
                velocity: vec![0; len],
            },
            None => OptimizerState::None,
        }
    }

    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState) {
        assert_eq!(
            weights.len(),
            grads.len(),
            "Weights and Gradients must match length! Got {} vs. {}",
            weights.len(),
            grads.len(),
        );
        let lr_div = 1 << self.lr_shift;

        match (self.momentum_shift, state) {
            (Some(m_shift), OptimizerState::SGD { velocity }) => {
                let m_div = 1 << m_shift;
                for ((w, m), g) in weights
                    .iter_mut()
                    .zip(velocity.iter_mut())
                    .zip(grads.iter())
                {
                    let mut decay = *m / m_div;
                    if decay == 0 && *m != 0 {
                        decay = m.signum();
                    }
                    *m = m.wrapping_sub(decay).wrapping_add(*g);
                    *w = w.wrapping_sub(*m / lr_div);
                }
            }
            _ => {
                for (w, g) in weights.iter_mut().zip(grads) {
                    *w = w.wrapping_sub(*g / lr_div);
                }
            }
        }
    }
}

// Adam
pub struct AdamConfig {
    pub lr_mult: i32,
    pub b1_shift: u32,
    pub b2_shift: u32,
    pub eps: i32,
}

impl AdamConfig {
    pub fn new(lr_mult: i32) -> Self {
        Self {
            lr_mult,
            b1_shift: 3,
            b2_shift: 4,
            eps: 1,
        }
    }
}

impl OptimizerConfig for AdamConfig {
    fn init_state(&self, len: usize) -> OptimizerState {
        OptimizerState::Adam {
            m: vec![0; len],
            v: vec![0i64; len],
        }
    }

    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState) {
        if let OptimizerState::Adam { m, v } = state {
            let b1_div = 1 << self.b1_shift;
            let b2_div = 1 << self.b2_shift;

            for i in 0..weights.len() {
                let g = grads[i];
                let g_64 = grads[i] as i64;
                m[i] = m[i].wrapping_sub(m[i] / b1_div).wrapping_add(g);
                v[i] = v[i]
                    .wrapping_sub(v[i] / (b2_div as i64))
                    .wrapping_add(g_64 * g_64);
                let denom = kernels::isqrt_64(v[i].max(0) as u64) as i32 + self.eps;
                weights[i] = weights[i].wrapping_sub((m[i] * self.lr_mult) / denom);
            }
        }
    }
}

// Module trait
// Every building block needs to implement this.
// Containers delegate; primitives do work.
//
pub struct ModuleInfo {
    pub name: &'static str,
    pub params: usize,
    pub static_bytes: usize,
    pub children: Vec<ModuleInfo>,
}

impl ModuleInfo {
    pub fn total(&self) -> (usize, usize) {
        let child_totals = self
            .children
            .iter()
            .map(|c| c.total())
            .fold((0, 0), |(p, b), (cp, cb)| (p + cp, b + cb));
        (
            self.params + child_totals.0,
            self.static_bytes + child_totals.1,
        )
    }
}
pub trait Module {
    fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8>;
    fn backward(&mut self, grad: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16>;

    /// quantize i32 master weights -> i8 storage. No-op for non-parameterized modules.
    fn sync_weights(&mut self, _rng: &mut XorShift64) {}

    /// Apply one optimizer step using internal grad cache. No-op if no grads cached.
    fn step(&mut self, _optim: &dyn OptimizerConfig) {}

    fn memory_report(&self) -> (usize, usize) {
        (0, 0)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "unknown",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }

    fn print_summary(&self, info: &ModuleInfo, depth: usize) {
        let indent = "  ".repeat(depth);
        let (total_params, total_bytes) = info.total();
        println!(
            "{}{}  params={} static={}B",
            indent, info.name, total_params, total_bytes
        );
        for child in &info.children {
            self.print_summary(child, depth + 1);
        }
    }
}

pub struct Params {
    pub master: Tensor<i32>,
    pub storage: Tensor<i8>,
    pub grads: Option<Tensor<i32>>,
    pub state: Option<OptimizerState>,
    pub shift: u32,
}

impl Params {
    pub fn new(shape: Vec<usize>, shift: u32) -> Self {
        Self {
            master: Tensor::new(shape.clone()),
            storage: Tensor::new(shape),
            grads: None,
            state: None,
            shift,
        }
    }

    pub fn init_uniform(&mut self, rng: &mut XorShift64, range: i32) {
        let range = range.max(1);
        let spread = (2 * range) as u32;
        for w in self.master.data.iter_mut() {
            *w = (rng.gen_range(spread) as i32) - range;
        }
    }

    pub fn init_xavier_uniform(&mut self, rng: &mut XorShift64, fan_in: usize, fan_out: usize) {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let limit_i8 = (limit * 127.0).round() as i32;
        let limit_master = limit_i8 * (1 << self.shift);
        self.init_uniform(rng, limit_master);
    }

    pub fn sync(&mut self, rng: &mut XorShift64) {
        for (m, s) in self.master.data.iter().zip(self.storage.data.iter_mut()) {
            *s = kernels::stochastic_downcast(*m, self.shift, rng);
        }
    }

    pub fn accumulate_grads(&mut self, new_grads: Tensor<i16>) {
        match self.grads.as_mut() {
            Some(existing) => {
                // Saturating add to avoid wrapping on large accumulated signals
                for (e, n) in existing.data.iter_mut().zip(new_grads.data.iter()) {
                    *e = e.saturating_add(*n as i32);
                }
            }
            None => {
                let mut i32_grads = Tensor::<i32>::new(new_grads.shape.clone());
                for (i, &n) in new_grads.data.iter().enumerate() {
                    i32_grads.data[i] = n as i32;
                }
                self.grads = Some(i32_grads);
            }
        }
    }

    pub fn zero_grads(&mut self) {
        self.grads = None;
    }

    pub fn step(&mut self, optim: &dyn OptimizerConfig) {
        if let Some(grads) = self.grads.take() {
            let state = self
                .state
                .get_or_insert_with(|| optim.init_state(self.master.len()));
            optim.update(&mut self.master.data, &grads.data, state);
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.master.memory_bytes() + self.storage.memory_bytes()
    }
}

pub struct Linear {
    /// all weights are of Transposed Form [out, in] for memory reasons
    pub weights: Params,
    pub bias: Params,
    pub cache: Vec<Tensor<i8>>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, scale_shift: u32) -> Self {
        Self {
            weights: Params::new(vec![output_dim, input_dim], scale_shift),
            bias: Params::new(vec![output_dim], scale_shift),
            cache: Vec::new(),
        }
    }

    pub fn init_xavier(&mut self, rng: &mut XorShift64) {
        let fan_in = self.weights.master.shape[1];
        let fan_out = self.weights.master.shape[0];
        self.weights.init_xavier_uniform(rng, fan_in, fan_out);
        for b in self.bias.master.data.iter_mut() {
            *b = 0;
        }
    }
}

impl Module for Linear {
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        self.weights.sync(rng);
        self.bias.sync(rng);
    }

    fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        assert_eq!(
            input.shape[1], self.weights.storage.shape[1],
            "Linear::forward: Input in wrong dimension for weights! {} vs {}",
            input.shape[1], self.weights.storage.shape[1]
        );
        let batch = input.shape[0];
        let input_dim = input.shape[1];
        let output_dim = self.weights.storage.shape[0];

        self.cache.push(input.clone());
        let mut out = Tensor::new(vec![batch, output_dim]);

        for b in 0..batch {
            let in_row = &input.data[b * input_dim..(b + 1) * input_dim];
            for o in 0..output_dim {
                let w_row = &self.weights.storage.data[o * input_dim..(o + 1) * input_dim];
                #[cfg(target_arch = "aarch64")]
                let raw_val = unsafe {
                    if input_dim % 16 == 0 {
                        kernels::arm_neon::dot_product_neon_raw(in_row, w_row)
                    } else {
                        kernels::dot_product_scalar(in_row, w_row)
                    }
                };
                #[cfg(not(target_arch = "aarch64"))]
                let raw_val = kernels::dot_product_scalar(in_row, w_row);
                let acc = raw_val + self.bias.storage.data[o] as i32;
                out.data[b * output_dim + o] =
                    kernels::stochastic_downcast(acc, self.weights.shift, rng);
            }
        }
        out
    }

    fn backward(&mut self, grad_output: &Tensor<i16>, gradient_shift: Option<u32>) -> Tensor<i16> {
        let input = self.cache.pop().expect("Backward called without forward");
        let batch = input.shape[0];
        let input_dim = input.shape[1];
        let output_dim = self.weights.storage.shape[0];
        let gshift = gradient_shift.unwrap_or(8);

        // Mathematically required for the chain rule:
        let wshift = self.weights.shift;

        let mut grad_input = Tensor::<i16>::new(vec![batch, input_dim]);
        let mut grad_weights = Tensor::<i16>::new(vec![output_dim, input_dim]);
        let mut grad_bias = Tensor::<i16>::new(vec![output_dim]);

        for b in 0..batch {
            for o in 0..output_dim {
                let g = grad_output.data[b * output_dim + o];
                if g == 0 {
                    continue;
                }

                // Bias update is an accumulator, so apply gshift
                let g_shifted = if gshift > 0 { g >> gshift } else { g };
                grad_bias.data[o] = grad_bias.data[o].saturating_add(g_shifted);

                for i in 0..input_dim {
                    let x = input.data[b * input_dim + i];
                    let dw = kernels::mul_mixed_scalar(g, x);
                    // Weight update is an accumulator, so apply gshift
                    grad_weights.data[o * input_dim + i] =
                        grad_weights.data[o * input_dim + i].saturating_add((dw >> gshift) as i16);

                    let w = self.weights.storage.data[o * input_dim + i];
                    let dx = kernels::mul_mixed_scalar(g, w);

                    // Input gradient flows back through the network, apply wshift!
                    grad_input.data[b * input_dim + i] =
                        grad_input.data[b * input_dim + i].saturating_add((dx >> wshift) as i16);
                }
            }
        }

        self.weights.accumulate_grads(grad_weights);
        self.bias.accumulate_grads(grad_bias);
        grad_input
    }

    fn step(&mut self, optim: &dyn OptimizerConfig) {
        self.weights.step(optim);
        self.bias.step(optim);
    }

    fn memory_report(&self) -> (usize, usize) {
        let stat = self.weights.memory_bytes() + self.bias.memory_bytes();
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (stat, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        let in_dim = self.weights.storage.shape[1];
        let out_dim = self.weights.storage.shape[0];
        ModuleInfo {
            name: "Linear",
            params: (in_dim * out_dim) + out_dim, // weights + bias
            static_bytes: self.weights.memory_bytes() + self.bias.memory_bytes(),
            children: vec![],
        }
    }
}

pub struct ReLU {
    pub cache: Vec<Tensor<i8>>,
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self { cache: Vec::new() }
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: &Tensor<i8>, _rng: &mut XorShift64) -> Tensor<i8> {
        self.cache.push(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for idx in 0..input.data.len() {
            output.data[idx] = if input.data[idx] > 0 {
                input.data[idx]
            } else {
                0
            };
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self
            .cache
            .pop()
            .expect("ReLU::backward: No state registered. Perform forward pass first!");
        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            output.data[o] = if input.data[o] > 0 {
                grad_output.data[o]
            } else {
                0
            };
        }
        output
    }
    fn memory_report(&self) -> (usize, usize) {
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (0, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "ReLU",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }
}

pub struct Tanh {
    cache: Vec<Tensor<i8>>,
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    pub fn new() -> Self {
        Self { cache: Vec::new() }
    }
}

impl Module for Tanh {
    fn forward(&mut self, input: &Tensor<i8>, _rng: &mut XorShift64) -> Tensor<i8> {
        self.cache.push(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for (o, &x) in output.data.iter_mut().zip(&input.data) {
            *o = kernels::tanh_i8(x);
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self
            .cache
            .pop()
            .expect("Tanh::backward: No state registered. Perform forward pass first!");
        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            let t = kernels::tanh_i8(input.data[o]) as i32;
            let dtanh = (127 * 127 - t * t) / 127;
            output.data[o] =
                ((grad_output.data[o] as i32 * dtanh) / 128).clamp(-32768, 32767) as i16;
        }
        output
    }
    fn memory_report(&self) -> (usize, usize) {
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (0, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Tanh",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }
}

pub struct Sequential {
    pub modules: Vec<Box<dyn Module>>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { modules: vec![] }
    }

    pub fn add(&mut self, m: impl Module + 'static) -> &mut Self {
        self.modules.push(Box::new(m));
        self
    }
}

impl Module for Sequential {
    fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        let mut output = input.clone();
        for m in self.modules.iter_mut() {
            output = m.forward(&output, rng);
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16> {
        let mut output = grad_output.clone();
        for m in self.modules.iter_mut().rev() {
            output = m.backward(&output, grad_shift);
        }
        output
    }
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        for m in self.modules.iter_mut() {
            m.sync_weights(rng)
        }
    }
    fn step(&mut self, optim: &dyn OptimizerConfig) {
        for m in self.modules.iter_mut() {
            m.step(optim);
        }
    }

    fn memory_report(&self) -> (usize, usize) {
        self.modules
            .iter()
            .map(|m| m.memory_report())
            .fold((0, 0), |(s, d), (ms, md)| (s + ms, d + md))
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Sequential",
            params: 0,
            static_bytes: 0,
            children: self.modules.iter().map(|m| m.describe()).collect(),
        }
    }
}

pub struct RNNCell {
    pub w_ih: Linear,
    pub w_hh: Linear,
    pub act: Tanh,
    pub h_prev: Option<Tensor<i8>>,
    pub hidden_dim: usize,

    d_h_next: Option<Tensor<i16>>,
}

impl RNNCell {
    pub fn new(input_dim: usize, hidden_dim: usize, scale_shift: u32) -> Self {
        Self {
            w_ih: Linear::new(input_dim, hidden_dim, scale_shift),
            w_hh: Linear::new(hidden_dim, hidden_dim, scale_shift),
            act: Tanh::new(),
            h_prev: None,
            hidden_dim,
            d_h_next: None,
        }
    }

    pub fn reset_state(&mut self) {
        self.h_prev = None;
        self.d_h_next = None;

        self.w_ih.cache.clear();
        self.w_hh.cache.clear();
        self.act.cache.clear();
    }

    pub fn init_weights(&mut self, rng: &mut XorShift64) {
        self.w_ih.init_xavier(rng);

        let hidden_dim = self.w_hh.weights.master.shape[0];
        let spectral_cap_i8 = kernels::isqrt(16129 / (hidden_dim as u32)) as i32;
        let spectral_cap_master = spectral_cap_i8 * (1 << self.w_hh.weights.shift);

        let fan_in = self.w_hh.weights.master.shape[1];
        let fan_out = self.w_hh.weights.master.shape[0];
        let xavier_limit_i8 = kernels::isqrt(96774 / (fan_in + fan_out) as u32) as i32;

        let xavier_limit_master = xavier_limit_i8 * (1 << self.w_hh.weights.shift);
        let range = xavier_limit_master.min(spectral_cap_master);
        self.w_hh.weights.init_uniform(rng, range);
    }
}

impl Module for RNNCell {
    fn forward(&mut self, input: &Tensor<i8>, rng: &mut XorShift64) -> Tensor<i8> {
        let batch = input.shape[0];

        let h = self
            .h_prev
            .get_or_insert_with(|| Tensor::new(vec![batch, self.hidden_dim]));

        // h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        // We do both linear transforms, add elementwise, then tanh.
        // Simplest: pass through w_ih, add w_hh output, apply tanh.
        //
        // Note: adding two i8 tensors before tanh risks overflow.
        // Accumulate in i16, then downcast before tanh.
        let ih = self.w_ih.forward(input, rng);
        let hh = self.w_hh.forward(h, rng);

        let mut comb = Tensor::<i8>::new(vec![batch, self.hidden_dim]);
        for i in 0..comb.data.len() {
            let sum = (ih.data[i] as i16).wrapping_add(hh.data[i] as i16);
            comb.data[i] = sum.clamp(-128, 127) as i8;
        }

        let h_next = self.act.forward(&comb, rng);
        self.h_prev = Some(h_next.clone());
        h_next
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16> {
        let combined_grad = match self.d_h_next.take() {
            Some(carry) => {
                let mut combined = grad_output.clone();
                for (c, k) in combined.data.iter_mut().zip(carry.data.iter()) {
                    *c = c.saturating_add(*k);
                }
                combined
            }
            None => grad_output.clone(),
        };

        let d_comb = self.act.backward(&combined_grad, grad_shift);

        let d_ih = self.w_ih.backward(&d_comb, grad_shift);
        let d_hh = self.w_hh.backward(&d_comb, grad_shift); // compute it...
        // d_hh should be fed back as the grad for h_prev in the next BPTT step
        self.d_h_next = Some(d_hh);

        d_ih
    }
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        self.w_ih.sync_weights(rng);
        self.w_hh.sync_weights(rng);
    }

    fn step(&mut self, optim: &dyn OptimizerConfig) {
        self.w_ih.step(optim);
        self.w_hh.step(optim);
    }

    fn memory_report(&self) -> (usize, usize) {
        let (s1, d1) = self.w_ih.memory_report();
        let (s2, d2) = self.w_hh.memory_report();
        let h_mem = self.h_prev.as_ref().map_or(0, |t| t.memory_bytes());
        (s1 + s2, d1 + d2 + h_mem)
    }

    fn describe(&self) -> ModuleInfo {
        let children = vec![
            self.w_ih.describe(),
            self.w_hh.describe(),
            self.act.describe(),
        ];
        ModuleInfo {
            name: "RNNCell",
            params: children.iter().map(|e| e.params).sum(),
            static_bytes: 0,
            children,
        }
    }
}

pub struct RNN {
    pub cell: RNNCell,
    pub bptt_steps: usize,
}

impl RNN {
    pub fn new(input_dim: usize, hidden_dim: usize, scale_shift: u32, bptt_steps: usize) -> Self {
        Self {
            cell: RNNCell::new(input_dim, hidden_dim, scale_shift),
            bptt_steps,
        }
    }

    pub fn reset_state(&mut self) {
        self.cell.reset_state();
    }

    pub fn forward_seq(
        &mut self,
        input_seq: &[Tensor<i8>],
        rng: &mut XorShift64,
    ) -> Vec<Tensor<i8>> {
        input_seq
            .iter()
            .map(|x| self.cell.forward(x, rng))
            .collect()
    }

    pub fn backward_seq(
        &mut self,
        grad_seq: &[Tensor<i16>],
        grad_shift: Option<u32>,
    ) -> Vec<Tensor<i16>> {
        let start = grad_seq.len().saturating_sub(self.bptt_steps);
        grad_seq[start..]
            .iter()
            .rev()
            .map(|g| self.cell.backward(g, grad_shift))
            .collect()
    }
}

pub trait Loss {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>);
}

pub struct MSE;
pub struct MAE;

impl Loss for MSE {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MSE::forward: vector sizes don't match."
        );
        let mut loss: i32 = 0;
        let mut grad = Tensor::<i16>::new(preds.shape.clone());

        for i in 0..preds.data.len() {
            let error = preds.data[i] as i16 - targets.data[i] as i16;
            loss += (error * error) as i32;
            // dL/dy = 2*(y - t), dropping the 2 it's absorbed by lr
            grad.data[i] = error;
        }
        (loss, grad)
    }
}

impl Loss for MAE {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MAE::forward: vector sizes don't match."
        );
        let mut loss: i32 = 0;
        let mut grad = Tensor::<i16>::new(preds.shape.clone());

        for i in 0..preds.data.len() {
            let error = preds.data[i] as i16 - targets.data[i] as i16;
            loss += error.abs() as i32;
            // dL/dy = 2*(y - t), dropping the 2 it's absorbed by lr
            grad.data[i] = error;
        }
        (loss / (preds.data.len() as i32), grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tensor<T> test suite
    #[test]
    fn test_tensor_new() {
        let t: Tensor<i32> = Tensor::new(vec![2, 4]);

        assert_eq!(t.shape[0], 2);
        assert_eq!(t.shape[1], 4);
        assert_eq!(t.data.len(), 8);
    }

    #[test]
    fn test_tensor_new_empty_vec() {
        let t: Tensor<i32> = Tensor::new(vec![]);

        assert_eq!(t.shape.len(), 0);
        assert_eq!(t.data.len(), 0);
    }

    #[test]
    fn test_tensor_from_vec() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2, 1], vec![2, 1]);
        assert_eq!(t.data.len(), t.shape[0] * t.shape[1]);
    }

    #[test]
    fn test_tensor_from_vec_empty_vec() {
        let t: Tensor<i32> = Tensor::from_vec(vec![], vec![]);
        assert_eq!(t.shape.len(), 0);
        assert_eq!(t.data.len(), 0);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Tensor::from_vec: Shape [2, 1] does not match data len 3\n  left: 3\n right: 2"
    )]
    fn test_tensor_from_vec_wrong_shape_for_data() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2, 1, 3], vec![2, 1]);
    }

    #[test]
    fn test_tensor_len() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2, 1], vec![2, 1]);

        assert_eq!(t.len(), 2);
    }
    #[test]
    fn test_tensor_len_empty() {
        let t: Tensor<i32> = Tensor::new(vec![]);

        assert_eq!(t.len(), 0, "{:?}", t);
    }

    #[test]
    fn test_tensor_memory_bytes() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2], vec![1, 1]);

        assert_eq!(t.len(), 1);
        assert_eq!(t.memory_bytes(), 4);
    }

    // XorShift64 tests
    #[test]
    fn test_xorshift_new() {
        let rng1 = XorShift64::new(12345);
        assert_eq!(rng1.state, 12345);
        let rng2 = XorShift64::new(0);
        assert_eq!(rng2.state, 0xCAFEBABE);
    }

    #[test]
    fn test_xorshift_next() {
        let mut rng = XorShift64::new(1);

        assert_eq!(rng.next(), 1082269761);
        assert_eq!(rng.state, 1082269761);
    }

    #[test]
    fn test_xorshift_determinism() {
        let mut rng1 = XorShift64::new(12345);
        let mut rng2 = XorShift64::new(12345);

        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
    }

    #[test]
    fn test_xorshift_gen_range() {
        let mut rng = XorShift64::new(1);

        for i in 0..1000 {
            let val = rng.gen_range(12);
            assert!(val <= 12);
            assert!(val >= 0);
        }
    }

    // Kernel tests
    #[test]
    fn test_kernel_mul_mixed_scalar() {
        assert_eq!(kernels::mul_mixed_scalar(5 as i16, 3 as i8), 15 as i32);
    }

    #[test]
    fn test_kernel_dot_product_scalar() {
        let vec1 = [1, 1, 1];
        let vec2 = [1, 1, 1];

        let res = kernels::dot_product_scalar(&vec1, &vec2);

        assert_eq!(res, 3);
    }

    #[test]
    fn test_kernel_dot_product_mixed_scalar() {
        let vec1: &[i8] = &[1, 1, 1];
        let vec2: &[i16] = &[1, 1, 1];

        let res = kernels::dot_product_mixed_scalar(&vec1, &vec2);

        assert_eq!(res, 3);
    }

    #[test]
    fn test_kernel_isqrt() {
        assert_eq!(kernels::isqrt(9), 3);
        assert_eq!(kernels::isqrt(8), 2);
        assert_eq!(kernels::isqrt(7), 2);
        assert_eq!(kernels::isqrt(6), 2);
        assert_eq!(kernels::isqrt(5), 2);
        assert_eq!(kernels::isqrt(4), 2);
        assert_eq!(kernels::isqrt(3), 1);
        assert_eq!(kernels::isqrt(2), 1);
        assert_eq!(kernels::isqrt(1), 1);
    }

    #[test]
    fn test_kernel_stochastic_downcast_deterministic() {
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
    fn test_kernel_stochastic_downcast_saturation() {
        let mut rng = XorShift64::new(111);

        // Value: 1000. Shift: 0. Should clamp to 127.
        let res = kernels::stochastic_downcast(1000, 0, &mut rng);
        assert_eq!(res, 127);

        // Value: -1000. Shift: 0. Should clamp to -128.
        let res = kernels::stochastic_downcast(-1000, 0, &mut rng);
        assert_eq!(res, -128);
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
            sum += kernels::stochastic_downcast(val, shift, &mut rng) as i64;
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
            sum += kernels::stochastic_downcast(val, shift, &mut rng) as i64;
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
        let vec1 = [1i8; 16];
        let vec2 = [1i8; 16];

        let res = unsafe { kernels::arm_neon::dot_product_neon_raw(&vec1, &vec2) };

        assert_eq!(res, 16);

        let vec3 = [-1i8; 16];
        let res_neg = unsafe { kernels::arm_neon::dot_product_neon_raw(&vec1, &vec3) };
        assert_eq!(res_neg, -16);
    }

    // Linear tests
    #[test]
    fn test_linear_new() {
        let lin = Linear::new(4, 8, 2);

        assert_eq!(lin.weights.master.len(), 4 * 8);
        assert_eq!(lin.weights.storage.len(), 4 * 8);
        assert_eq!(lin.bias.master.len(), 8);
        assert_eq!(lin.weights.shift, 2);
        assert_eq!(lin.cache, Vec::new());
        assert_eq!(lin.weights.grads, None);
        assert_eq!(lin.bias.grads, None);
    }

    #[test]
    fn test_linear_sync_weights() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 10;
        lin.weights.master.data[3] = 10;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 10);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 10);

        lin.weights.shift = 1;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 5);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 5);

        lin.weights.shift = 2;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 2);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 2);

        lin.weights.shift = 3;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 2);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 1);
    }

    #[test]
    fn test_linear_forward() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new(vec![1, 2]);
        input.data[0] = 10;
        input.data[1] = 20;
        let out = lin.forward(&input, &mut rng);

        assert_eq!(out.data[0], 10);
        assert_eq!(out.data[1], 20);
    }

    #[test]
    fn test_linear_forward_saturation() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 126;
        lin.weights.master.data[3] = 126;
        let mut rng = XorShift64 { state: 420 };
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
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64 { state: 420 };
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
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;

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

        lin.cache = vec![input.clone()];

        let dX = lin.backward(&grad_out, None);
        let dW = lin.weights.grads.unwrap().clone();

        // Check shapes
        assert_eq!(dX.shape, vec![2, 4]); // [Batch, In]
        assert_eq!(dW.shape, vec![3, 4]); // [Out, In]
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Linear::forward: Input in wrong dimension for weights! 1 vs 2\n  left: 1\n right: 2"
    )]
    fn test_linear_forward_wrong_dimensional_input() {
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::from_vec(vec![2; 10], vec![10, 1]);
        lin.forward(&input, &mut rng);
    }

    #[test]
    fn test_linear_memory_report() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1, 2);
        let x = Tensor::from_vec(vec![0, 0], vec![1, 2]);
        l1.forward(&x, &mut rng);

        let (a, b) = l1.memory_report();
        assert_eq!(a, 15);
        assert_eq!(b, 2);

        let mut l2 = Linear::new(2, 10, 2);
        l2.forward(&x, &mut rng);

        let (a2, b2) = l2.memory_report();
        assert_eq!(a2, 150);
        assert_eq!(b2, 2);
    }

    #[test]
    fn test_linear_step() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1, 1);
        let mut optim = SGDConfig::new(0, None);
        l1.weights.master.data[0] = 111;
        l1.weights.master.data[1] = 222;
        l1.sync_weights(&mut rng);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        // no gradient -> no change
        l1.step(&mut optim);
        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        let x = Tensor::from_vec(vec![1, 1], vec![1, 2]);

        l1.forward(&x, &mut rng);
        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        let x2 = Tensor::from_vec(vec![2, 2], vec![1, 2]);

        l1.sync_weights(&mut rng);
        let preds = l1.forward(&x2, &mut rng);

        // 3. Compute Loss & Gradients (Error = Pred - Target)
        let mut grad_out = Tensor::<i32>::from_vec(vec![126, 2], vec![2, 1]);

        l1.weights.grads = Some(grad_out);

        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], -15);
        assert_eq!(l1.weights.master.data[1], 220);
        assert_eq!(l1.weights.storage.data[0], 56);
        assert_eq!(l1.weights.storage.data[1], 111);
    }

    #[test]
    fn test_linear_step_with_shifting() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1, 2);
        let mut optim = SGDConfig::new(0, None);
        l1.weights.master.data[0] = 111;
        l1.weights.master.data[1] = 222;

        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 0);
        assert_eq!(l1.weights.storage.data[1], 0);
    }

    #[test]
    fn test_relu_forward() {
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
    fn test_relu_backward() {
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
    #[should_panic(expected = "ReLU::backward: No state registered. Perform forward pass first!")]
    fn test_relu_backward_without_forward_call() {
        let mut relu = ReLU::new();
        let mut input = Tensor::<i16>::new(vec![4, 1]);
        let mut rng = XorShift64::new(42);
        let _res = relu.backward(&input, Some(2));
    }

    #[test]
    fn test_relu_memory_report() {
        let mut relu = ReLU::new();

        let (mut stat, mut dyna) = relu.memory_report();

        assert_eq!(stat, 0);
        assert_eq!(dyna, 0);

        let mut input = Tensor::<i8>::new(vec![4, 1]);
        let mut rng = XorShift64::new(42);
        relu.forward(&input, &mut rng);

        (stat, dyna) = relu.memory_report();

        assert_eq!(stat, 0);
        assert_eq!(dyna, 4);
    }

    // Examples

    #[test]
    fn test_summary() {
        let model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 8, 2)),
                Box::new(ReLU::new()),
                Box::new(Linear::new(8, 1, 2)),
            ],
        };

        model.print_summary(&model.describe(), 1);
    }

    #[test]
    fn test_train_xor_sgd_momentum() {
        let mut rng = XorShift64::new(777);

        let mut l1 = Linear::new(2, 8, 2);
        let mut l2 = Linear::new(8, 1, 2);

        for w in l1.weights.master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30;
        }
        for w in l2.weights.master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30;
        }
        for b in l1.bias.master.data.iter_mut() {
            *b = 5;
        }

        let mut model = Sequential {
            modules: vec![Box::new(l1), Box::new(ReLU::new()), Box::new(l2)],
        };

        let mut optim = SGDConfig::new(4, Some(2));

        let x = Tensor::from_vec(vec![0, 0, 0, 1, 1, 0, 1, 1], vec![4, 2]);

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

        for w in l1.weights.master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30;
        }
        for w in l2.weights.master.data.iter_mut() {
            *w = (rng.gen_range(60) as i32) - 30;
        }
        for b in l1.bias.master.data.iter_mut() {
            *b = 5;
        }

        let mut model = Sequential {
            modules: vec![Box::new(l1), Box::new(ReLU::new()), Box::new(l2)],
        };

        // NEW: Instantiate our Integer Adam!
        // We set the learning rate multiplier to 2.
        let mut optim = AdamConfig::new(2);

        let x = Tensor::from_vec(vec![0, 0, 0, 1, 1, 0, 1, 1], vec![4, 2]);

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
                println!(
                    "Epoch {:03}: Loss = {:05}, Preds: [{}, {}, {}, {}]",
                    epoch, loss, preds.data[0], preds.data[1], preds.data[2], preds.data[3]
                );
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
        let p00 = final_preds.data[0];
        let p01 = final_preds.data[1];
        let p10 = final_preds.data[2];
        let p11 = final_preds.data[3];

        println!(
            "Final XOR Evaluation: 0,0->{} | 0,1->{} | 1,0->{} | 1,1->{}",
            p00, p01, p10, p11
        );

        assert!(p00 < 8, "0,0 failed: expected low, got {}", p00);
        assert!(p11 < 8, "1,1 failed: expected low, got {}", p11);
        assert!(p01 > 12, "0,1 failed: expected high, got {}", p01);
        assert!(p10 > 12, "1,0 failed: expected high, got {}", p10);
    }
}
