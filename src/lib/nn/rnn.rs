use crate::nn::{Linear, ModuleInfo, Module, HasWeights, compute_shift_for_max};
use crate::nn::activations::{Tanh};
use crate::nn::optim::{OptimizerConfig};
use crate::{Tensor, XorShift64, checked_add_counting, Scalar, Numeric};
use crate::nn::kernels;

use std::any::Any;

pub struct RNNCell<S: Scalar> {
    pub w_ih: Linear<S>,
    pub w_hh: Linear<S>,
    pub act: Tanh<S>,
    pub h_prev: Option<Tensor<S>>,
    pub hidden_dim: usize,

    // caches
    s_x_cache: Vec<u32>,
    s_h_cache: Vec<u32>,
    d_h_next: Option<(Tensor<S::Acc>, u32)>,

}

impl<S: Scalar + 'static> RNNCell<S> {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_ih: Linear::new(input_dim, hidden_dim),
            w_hh: Linear::new(hidden_dim, hidden_dim),
            act: Tanh::new(),
            s_x_cache: Vec::new(),
            s_h_cache: Vec::new(),
            h_prev: None,
            hidden_dim,
            d_h_next: None,
        }
    }

    pub fn reset_state(&mut self) {
        self.h_prev = None;
        self.d_h_next = None;
        //self.output_shift = None;
        //self.input_shift = None;

        self.w_ih.cache.clear();
        self.w_hh.cache.clear();
        self.act.cache.clear();
        self.s_x_cache.clear();
        self.s_h_cache.clear();
    }

    pub fn init_weights(&mut self, rng: &mut XorShift64) {
        self.w_ih.init_xavier(rng);

        let hidden_dim = self.w_hh.weights.master.shape[0];
        let spectral_cap_i32 = kernels::isqrt(16129 / (hidden_dim as u32)) as i32;
        let spectral_cap_master = spectral_cap_i32 * (1 << self.w_hh.weights.quant_shift);

        let fan_in = self.w_hh.weights.master.shape[1];
        let fan_out = self.w_hh.weights.master.shape[0];
        let xavier_limit_i32 = kernels::isqrt(96774 / (fan_in + fan_out) as u32) as i32;

        let xavier_limit_master = xavier_limit_i32 * (1 << self.w_hh.weights.quant_shift);
        let range = xavier_limit_master.min(spectral_cap_master);
        self.w_hh.weights.init_uniform(rng, range);
    }

    pub fn init_weights_auto(&mut self, rng: &mut XorShift64) {
        self.init_weights(rng);

        let inferred_shift = self.infer_scale_shift();
        self.w_ih.weights.quant_shift = inferred_shift;
        self.w_ih.bias.quant_shift = inferred_shift;
        self.w_hh.weights.quant_shift = inferred_shift;
        self.w_hh.bias.quant_shift = inferred_shift;
    }
}

impl<S: Scalar + 'static> Module<S> for RNNCell<S> {
    fn get_output_shift(&self) -> u32 {
        0
    }
    fn forward(&mut self, input: &Tensor<S>, s_x: u32, rng: &mut XorShift64) -> (Tensor<S>, u32) {
        let batch = input.shape[0];

        let h = self
            .h_prev
            .get_or_insert_with(|| Tensor::new(vec![batch, self.hidden_dim]));

        let s_h = self.s_h_cache.last().copied().unwrap_or(0);

        let (ih, s_ih) = self.w_ih.forward(input, s_x, rng);
        let (hh, s_hh) = self.w_hh.forward(h, s_h, rng);

        let mut comb = Tensor::<S>::new(vec![batch, self.hidden_dim]);
        for i in 0..comb.data.len() {
            let sum = ih.data[i].into_acc().add(hh.data[i].into_acc());
            // TODO: we might need to tune shifting here, for integer spaces
            comb.data[i] = S::downcast(sum, 1, rng);
        }

        let s_comb = s_ih.min(s_hh).saturating_sub(1);

        let (h_next, s_out) = self.act.forward(&comb, s_comb, rng);

        self.s_x_cache.push(s_x);
        self.s_h_cache.push(s_out);
        self.h_prev = Some(h_next.clone());

        (h_next, s_out)
    }

    fn backward(&mut self, grad_output: &Tensor<S::Acc>, s_g: u32) -> (Tensor<S::Acc>, u32) {
        let (combined_grad, combined_s_g) = match self.d_h_next.take() {
            Some((carry, carry_s_g)) => {
                let mut combined = grad_output.clone();
                for (c, k) in combined.data.iter_mut().zip(carry.data.iter()) {
                    *c = c.add(*k);
                }
                (combined, s_g.min(carry_s_g))
            }
            None => (grad_output.clone(), s_g),
        };

        let (d_comb, s_g_act) = self.act.backward(&combined_grad, combined_s_g);

        let (d_ih, s_g_ih) = self.w_ih.backward(&d_comb, s_g_act);
        let (d_hh, s_g_hh) = self.w_hh.backward(&d_comb, s_g_act); // compute it...
        // d_hh should be fed back as the grad for h_prev in the next BPTT step
        self.d_h_next = Some((d_hh, s_g_hh));

        (d_ih, s_g_ih)
    }
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        self.w_ih.sync_weights(rng);
        self.w_hh.sync_weights(rng);
    }

    fn step(&mut self, optim: &dyn OptimizerConfig<S>) {
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

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

impl<S: Scalar> HasWeights<S> for RNNCell<S> {
    fn get_all_weights(&self) -> Vec<&Tensor<S::Acc>> {
        vec![
            &self.w_ih.weights.master,
            &self.w_ih.bias.master,
            &self.w_hh.weights.master,
            &self.w_hh.bias.master,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnncell_new(){
        let cell = RNNCell::new(2, 4);

        assert_eq!(cell.w_ih.weights.master.shape, vec![4, 2]);
        assert_eq!(cell.w_hh.weights.master.shape, vec![4, 4]);
        assert_eq!(cell.hidden_dim, 4);
        assert_eq!(cell.h_prev, None);
        assert_eq!(cell.d_h_next, None);
        assert_eq!(cell.output_shift, None);
    }

    #[test]
    fn test_rnncell_reset_state(){
        let mut cell = RNNCell::new(2, 4);

        let t1 = Tensor::from_vec(vec![1, 2], vec![1, 2]);

        cell.h_prev = Some(t1.clone());
        cell.d_h_next = Some(t1.clone());
        cell.output_shift = Some(2);
        cell.w_ih.cache.push(t1.clone());
        cell.w_hh.cache.push(t1.clone());
        cell.act.cache.push(t1.clone());

        cell.reset_state();
        
        assert_eq!(cell.h_prev, None);
        assert_eq!(cell.d_h_next, None);
        assert_eq!(cell.output_shift, None);
        assert!(cell.w_ih.cache.is_empty());
        assert!(cell.w_hh.cache.is_empty());
        assert!(cell.act.cache.is_empty());
    }

    #[test]
    fn test_rnncell_init_weights(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 4);

        cell.init_weights(&mut rng);

        assert_eq!(cell.w_ih.weights.master.data, vec![124, -105, 44, -53, 59, -11, 5, 35]);
        assert_eq!(cell.w_hh.weights.master.data, vec![20, -43, 36, 22, 2, -14, 15, 29, 13, 36, -49, -42, -7, 8, 12, 0]);
    }

    #[test]
    fn test_init_weights_auto(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 4);

        // sets weights and determines quant_shift
        cell.init_weights_auto(&mut rng);

        assert_eq!(cell.w_ih.weights.master.data, vec![124, -105, 44, -53, 59, -11, 5, 35]);
        assert_eq!(cell.w_hh.weights.master.data, vec![20, -43, 36, 22, 2, -14, 15, 29, 13, 36, -49, -42, -7, 8, 12, 0]);
        assert_eq!(cell.w_ih.weights.quant_shift, 0);
        assert_eq!(cell.w_hh.weights.quant_shift, 0);
    }

    #[test]
    fn test_rnn_cell_get_output_shift(){
        let mut cell = RNNCell::new(2, 4);

        assert_eq!(cell.get_output_shift(), 0);

        cell.output_shift = Some(4);

        assert_eq!(cell.get_output_shift(), 4);
    }

    #[test]
    fn test_rnncell_forward(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 2);

        cell.w_ih.weights.storage.data = vec![1, 0, 0, 1];
        cell.w_hh.weights.storage.data = vec![1, 0, 0, 1];

        // first value stays 0, second and third aren't fully saturated,
        // value 4 will clamp to the max value (97)
        let input = Tensor::from_vec(vec![0, 100, 126, 127], vec![2, 2]);

        // first call, cell.h_prev is zero-vector
        let val1 = cell.forward(&input, 0, &mut rng);

        assert_eq!(val1.data, vec![0, 83, 96, 97]);
        assert_eq!(
            cell.h_prev,
            Some(Tensor::from_vec(vec![0, 83, 96, 97], vec![2, 2]))
        );

        // sequential call, now we have cell.h_prev available
        // the three last elements are fully saturated
        let val2 = cell.forward(&input, 0, &mut rng);

        assert_eq!(val2.data, vec![0, 97, 97, 97]);
    }

    #[test]
    fn test_rnncell_backward(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 2);

        cell.w_ih.weights.storage.data = vec![1, 0, 0, 1];
        cell.w_hh.weights.storage.data = vec![1, 0, 0, 1];

        let input = Tensor::from_vec(vec![0, 100, 200, 300], vec![2, 2]);

        let _ = cell.forward(&input, 0, &mut rng);
        let _ = cell.forward(&input, 0, &mut rng);

        // with round values we can calculate the outgoing gradients,
        // otherwise we get random behavior
        let grad = Tensor::from_vec(vec![0, -1, -1, 100], vec![2, 2]);

        assert_eq!(cell.d_h_next, None);

        let out = cell.backward(&grad);

        assert_eq!(out.data, vec![0, -1, -1, 47]);
        assert_eq!(cell.d_h_next, Some(Tensor::from_vec(vec![0, -1, -1, 47], vec![2, 2])));

        let out2 = cell.backward(&grad);
        
        assert_eq!(out2.data, vec![0, -2, -2, 128]);
        assert_eq!(cell.d_h_next, Some(Tensor::from_vec(vec![0, -2, -2, 128], vec![2, 2])));
    }

    #[test]
    fn test_rnncell_step(){
        use crate::nn::optim::{SGDConfig};

        // step updates master weights.
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 2);

        cell.w_ih.weights.master.data = vec![1, 0, 0, 1];
        cell.w_hh.weights.master.data = vec![1, 0, 0, 1];
        cell.sync_weights(&mut rng);

        let input = Tensor::from_vec(vec![0, 100, 200, 300], vec![2, 2]);

        let _ = cell.forward(&input, 0, &mut rng);
        let _ = cell.forward(&input, 0, &mut rng);

        // with round values we can calculate the outgoing gradients,
        // otherwise we get random behavior
        let grad = Tensor::from_vec(vec![0, -1, -1, 100], vec![2, 2]);

        assert_eq!(cell.d_h_next, None);

        let _ = cell.backward(&grad);
        let _ = cell.backward(&grad);

        let mut optim = SGDConfig::new()
            .with_learn_rate(1.0);

        assert_eq!(cell.w_ih.weights.master.data, cell.w_ih.weights.storage.data);
        assert_eq!(cell.w_hh.weights.master.data, cell.w_hh.weights.storage.data);

        cell.step(&mut optim);

        assert_ne!(cell.w_ih.weights.master.data, cell.w_ih.weights.storage.data);
        assert_ne!(cell.w_hh.weights.master.data, cell.w_hh.weights.storage.data);
    }

    #[test]
    fn test_rnncell_sync_weights(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 4);

        // sets weights and determines quant_shift
        cell.init_weights_auto(&mut rng);

        assert_eq!(cell.w_ih.weights.master.data, vec![124, -105, 44, -53, 59, -11, 5, 35]);
        assert_eq!(cell.w_hh.weights.master.data, vec![20, -43, 36, 22, 2, -14, 15, 29, 13, 36, -49, -42, -7, 8, 12, 0]);

        // no shifting
        cell.sync_weights(&mut rng);

        assert_eq!(cell.w_ih.weights.storage.data, vec![124, -105, 44, -53, 59, -11, 5, 35]);
        assert_eq!(cell.w_hh.weights.storage.data, vec![20, -43, 36, 22, 2, -14, 15, 29, 13, 36, -49, -42, -7, 8, 12, 0]);
    }

    #[test]
    fn test_rnncell_memory_report(){
        let cell = RNNCell::new(2, 4);

        let (s, d) = cell.memory_report();

        assert_eq!(d, 0);
        assert_eq!(s, 256);
    }

    #[test]
    fn test_rnncell_describe(){
        let cell = RNNCell::new(2, 4);

        assert_eq!(cell.describe(), ModuleInfo{
            name: "RNNCell",
            params: 32,
            static_bytes: 0,
            children: vec![
                ModuleInfo{
                    name: "Linear",
                    params: 12,
                    static_bytes: 96,
                    children: vec![],
                },
                ModuleInfo{
                    name: "Linear",
                    params: 20,
                    static_bytes: 160,
                    children: vec![],
                },
                ModuleInfo{
                    name: "Tanh",
                    params: 0,
                    static_bytes: 0,
                    children: vec![],
                }
            ]
        });
    }

    #[test]
    fn test_rnncell_init(){
        let mut rng = XorShift64::new(420);
        let mut cell = RNNCell::new(2, 4);

        // sets weights
        cell.init(&mut rng);

        assert_eq!(cell.w_ih.weights.master.data, vec![124, -105, 44, -53, 59, -11, 5, 35]);
        assert_eq!(cell.w_hh.weights.master.data, vec![20, -43, 36, 22, 2, -14, 15, 29, 13, 36, -49, -42, -7, 8, 12, 0]);
    }

    #[test]
    fn test_rnncell_get_all_weights(){
        let cell = RNNCell::new(2, 4);

        assert_eq!(cell.get_all_weights(), vec![
            &cell.w_ih.weights.master,
            &cell.w_ih.bias.master,
            &cell.w_hh.weights.master,
            &cell.w_hh.bias.master,
        ]);

    }
}
