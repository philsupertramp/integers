use crate::nn::{Linear, ModuleInfo, Module, HasWeights};
use crate::nn::activations::{Tanh};
use crate::nn::optim::{OptimizerConfig};
use crate::{Tensor, XorShift64, checked_add_i16};
use crate::nn::kernels;

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

    pub fn init_weights_auto(&mut self, rng: &mut XorShift64) {
        self.init_weights(rng);

        let inferred_shift = self.infer_scale_shift();
        self.w_ih.weights.shift = inferred_shift;
        self.w_ih.bias.shift = inferred_shift;
        self.w_hh.weights.shift = inferred_shift;
        self.w_hh.bias.shift = inferred_shift;
    }
}

impl Module for RNNCell {
    fn get_output_shift(&self) -> u32 {
        self.w_hh.weights.output_shift.unwrap_or(0)
    }
    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
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
        let ih = self.w_ih.forward(input, input_shift, rng);
        let hh = self.w_hh.forward(h, self.w_ih.weights.output_shift.expect("Expected output shift"), rng);

        let mut comb = Tensor::<i8>::new(vec![batch, self.hidden_dim]);
        for i in 0..comb.data.len() {
            let sum = checked_add_i16!(ih.data[i] as i16, hh.data[i] as i16, forward_wraps);
            comb.data[i] = sum.clamp(-128, 127) as i8;
        }

        let h_next = self.act.forward(&comb, self.w_hh.weights.output_shift.expect("Expected output shift"), rng);
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

    fn init(&mut self, rng: &mut XorShift64) {
        self.init_weights(rng);
    }
}
impl HasWeights for RNNCell {
    fn get_all_weights(&self) -> Vec<&Tensor<i32>> {
        vec![
            &self.w_ih.weights.master,
            &self.w_ih.bias.master,
            &self.w_hh.weights.master,
            &self.w_hh.bias.master,
        ]
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
        input_shift: u32,
        rng: &mut XorShift64,
    ) -> Vec<Tensor<i8>> {
        input_seq
            .iter()
            .map(|x| self.cell.forward(x, input_shift, rng))
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

