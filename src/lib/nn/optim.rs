use crate::nn::kernels;
use crate::{checked_sub_counting, checked_add_counting, Scalar, Numeric};

use std::fmt;

pub trait OptimizerConfig<S: Scalar> {
    fn update(&self, weights: &mut [S::Acc], grads: &[S::Acc], state: &mut OptimizerState<S>, quant_shift: u32);
    fn init_state(&self, len: usize) -> OptimizerState<S>;
}

#[derive(Debug, PartialEq)]
pub enum OptimizerState<S: Scalar> {
    None,
    SGD { velocity: Vec<S::Acc> },
    Adam { m: Vec<S::Acc>, v: Vec<S::Acc> },
}

// SGD
pub struct SGDConfig {
    pub lr_shift: u32,
    pub momentum_shift: Option<u32>,
}

impl SGDConfig {
    pub fn new() -> Self {
        Self {
            lr_shift: 2,
            momentum_shift: None,
        }
    }

    fn learn_rate_to_shift(&self, learn_rate: f32) -> u32 {
        let inv = 1.0 / learn_rate;
        let shift = inv.log2().round() as u32;
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Learning rate {} gives shift {}, should be 0-8", learn_rate, shift);
        shift
    }

    pub fn with_learn_rate(mut self, learn_rate: f32) -> Self {
        let lr_shift = self.learn_rate_to_shift(learn_rate);
        self.lr_shift = lr_shift;
        self
    }

    fn momentum_to_shift(&self, momentum: f32) -> u32 {
        assert!(momentum > 0.0 && momentum < 1.0, "Momentum must be in [0, 1], got {}", momentum);

        let inv = 1.0 / (1.0 - momentum);
        let shift = inv.log2().round() as u32;
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Momentum {} gives shift {}, should be 0-8", momentum, shift);
        shift
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        let momentum_shift = self.momentum_to_shift(momentum);
        self.momentum_shift = Some(momentum_shift);
        self
    }
}

impl<S: Scalar> OptimizerConfig<S> for SGDConfig {
    fn init_state(&self, len: usize) -> OptimizerState<S> {
        match self.momentum_shift {
            Some(_) => OptimizerState::SGD {
                velocity: vec![S::Acc::zero(); len],
            },
            None => OptimizerState::None,
        }
    }

    fn update(&self, weights: &mut [S::Acc], grads: &[S::Acc], state: &mut OptimizerState<S>, quant_shift: u32) {
        assert_eq!(
            weights.len(),
            grads.len(),
            "Weights and Gradients must match length! Got {} vs. {}",
            weights.len(),
            grads.len(),
        );
        let lr_div = S::Acc::from_i32(1 << self.lr_shift);

        match (self.momentum_shift, state) {
            (Some(m_shift), OptimizerState::SGD { velocity }) => {
                let m_div = S::Acc::from_i32(1 << m_shift);
                for ((w, m), g) in weights
                    .iter_mut()
                    .zip(velocity.iter_mut())
                    .zip(grads.iter())
                {
                    *m = m.sub(m.div(m_div)).add(g.div(m_div));
                    *w = w.sub(m.div(lr_div));
                }
            }
            _ => {
                for (w, g) in weights.iter_mut().zip(grads) {
                    *w = w.sub(g.div(lr_div));
                }
            }
        }
    }
}

// Adam
pub struct AdamConfig {
    pub lr_shift: u32,
    pub b1_shift: u32,
    pub b2_shift: u32,
    pub eps: i32,
}

impl AdamConfig {
    pub fn new() -> Self {
        Self {
            lr_shift: 2,
            b1_shift: 3,
            b2_shift: 4,
            eps: 1,
        }
    }

    fn learn_rate_to_shift(&self, learn_rate: f32) -> u32 {
        let inv = 1.0 / learn_rate;
        let shift = inv.log2().round() as u32;
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Learning rate {} gives shift {}, should be 0-8", learn_rate, shift);
        shift
    }

    fn beta_to_shift(&self, beta: f32) -> u32 {
        // beta = 1 - 1/2^shift
        // shift = log2(1 / (1 - beta))
        let inv = 1.0 / (1.0 - beta);
        let shift = inv.log2().round() as u32;
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Beta {} gives shift {}, should be 0-8", beta, shift);
        shift
    }
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.b1_shift = self.beta_to_shift(beta1);
        self.b2_shift = self.beta_to_shift(beta2);
        self
    }

    pub fn with_learn_rate(mut self, learn_rate: f32) -> Self {
        self.lr_shift = self.learn_rate_to_shift(learn_rate);
        self
    }

    pub fn with_eps(mut self, epsilon: f32) -> Self {
        self.eps = epsilon as i32;
        self
    }
}

impl fmt::Display for AdamConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Adam(lr={}, β1={}, β2={})", self.lr_shift, self.b1_shift, self.b2_shift)
    }
}
impl<S: Scalar> OptimizerConfig<S> for AdamConfig {
    fn init_state(&self, len: usize) -> OptimizerState<S> {
        OptimizerState::Adam {
            m: vec![S::Acc::zero(); len],
            v: vec![S::Acc::zero(); len],
        }
    }

    fn update(&self, weights: &mut [S::Acc], grads: &[S::Acc], state: &mut OptimizerState<S>, quant_shift: u32) {
        if let OptimizerState::Adam { m, v } = state {
            let b1_div = S::Acc::from_i32(1 << self.b1_shift);
            let b2_div = S::Acc::from_i32(1 << self.b2_shift);
            let lr_div = S::Acc::from_i32(1 << self.lr_shift);

            let eps = S::Acc::from_i32(self.eps);

            for i in 0..weights.len() {
                let g = grads[i];
                m[i] = m[i].sub(m[i].div(b1_div)).add(g.div(b1_div));
                v[i] = v[i].sub(v[i].div(b2_div)).add(g.mul(g).div(b2_div));

                let denom = v[i].sqrt().add(eps);
                weights[i] = weights[i].sub(m[i].div(lr_div).div(denom));
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_init() {
        let optim = SGDConfig::new();

        assert_eq!(optim.lr_shift, 2);
    }

    #[test]
    fn test_sgd_with_learn_rate() {
        let mut optim = SGDConfig::new();

        assert_eq!(optim.lr_shift, 2);

        let samples = vec![
            2.0,
            1.0, 0.5, 0.25,
            0.125, 0.0625, 0.03125,
            0.015625, 0.0078125, 0.00390625,
        ];
        let expected_values = vec![
            0,
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ];

        for (val, expected_f32) in samples.iter().zip(expected_values) {
            optim = optim.with_learn_rate(*val);
            assert_eq!(optim.lr_shift, expected_f32);
        }
    }

    #[test]
    #[should_panic(
        expected="Momentum must be in [0, 1], got 1.1"
    )]
    fn test_sgd_with_momentum_wrong_value(){
        let optim = SGDConfig::new();

        optim.momentum_to_shift(1.1);
    }

    #[test]
    fn test_sgd_with_momentum(){
        let mut optim = SGDConfig::new();

        assert_eq!(optim.momentum_shift, None);

        let samples = vec![
            0.997, 0.99, 0.98,
            0.97, 0.95, 0.88,
            0.82, 0.50, 0.20,
        ];
        let expected_values = vec![
            8, 7, 6, 5, 4, 3, 2, 1, 0
        ];

        for (val, expected_f32) in samples.iter().zip(expected_values) {
            optim = optim.with_momentum(*val);
            assert_eq!(optim.momentum_shift, Some(expected_f32));
        }
    }

    #[test]
    fn test_sgd_init_state(){
        let optim = SGDConfig::new();

        let state = optim.init_state(420);
        assert_eq!(state, OptimizerState::None);

        let optim = SGDConfig::new()
            .with_momentum(0.8);


        let state = optim.init_state(420);
        assert_eq!(state, OptimizerState::SGD{ velocity: vec![0; 420] });
    }

    #[test]
    #[should_panic(
        expected="assertion `left == right` failed: Weights and Gradients must match length! Got 2 vs. 3\n  left: 2\n right: 3"
    )]
    fn test_sgd_update_wrong_sizes(){
        let optim = SGDConfig::new();

        let mut state = optim.init_state(5);

        let mut input = vec![2; 2];
        let grad = vec![2; 3];
        optim.update(&mut input, &grad, &mut state, 0);
    }

    #[test]
    fn test_sgd_init_state_behavior() {
        let no_momentum = SGDConfig::new();
        assert!(matches!(no_momentum.init_state(5), OptimizerState::None));

        let with_momentum = SGDConfig::new().with_momentum(0.8);
        assert!(matches!(
            with_momentum.init_state(5),
            OptimizerState::SGD { .. }
        ));
    }

    #[test]
    fn test_sgd_update_sets_velocities(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0)
            .with_momentum(0.8);

        let mut state = optim.init_state(2);

        let grad = vec![2; 2];
        // will be [0, 0] -> [-2, -2]
        let mut weights = vec![0; 2];

        let og_velocity = match &state {
            OptimizerState::SGD { velocity } => velocity.clone(),
            _ => panic!("Expected SGD state with velocity!"),
        };
        optim.update(&mut weights, &grad, &mut state, 0);

        let new_velocity = match &state {
            OptimizerState::SGD { velocity } => velocity.clone(),
            _ => panic!("Expected SGD state with velocity!"),
        };

        for (og_vel, new_vel) in og_velocity.into_iter().zip(new_velocity.into_iter()) {
            assert_ne!(og_vel, new_vel);
        }

    }

    #[test]
    fn test_sgd_no_momentum_update(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0);

        let mut state = optim.init_state(0);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-2, -2]
        let mut weights = vec![0; 2];

        optim.update(&mut weights, &grad, &mut state, 0);

        // will be [1, 1] -> [-1, -1]
        let mut bias = vec![1; 2];

        optim.update(&mut bias, &grad, &mut state, 0);

        assert_eq!(weights, vec![-2, -2]);
        assert_eq!(bias, vec![-1, -1]);
    }

    #[test]
    fn test_sgd_no_momentum_update_check_lr_influence(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(0.5);

        let mut state = optim.init_state(0);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-1, -1]
        let mut weights = vec![0; 2];
        // will be [1, 1] -> [0, 0]
        let mut bias = vec![1; 2];

        optim.update(&mut weights, &grad, &mut state, 0);
        optim.update(&mut bias, &grad, &mut state, 0);

        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }

    #[test]
    fn test_sgd_update_decay_too_small(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0)
            // ~ shift by 1
            .with_momentum(0.88);

        let mut state = optim.init_state(2);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];

        // will be [0, 0] -> [-2, -2]
        let mut weights = vec![0; 2];

        optim.update(&mut weights, &grad, &mut state, 0);
        
        let og_velocity = match &state {
            OptimizerState::SGD { velocity } => velocity.clone(),
            _ => panic!("Expected SGD state with velocity!"),
        };

        assert_eq!(weights, vec![-2, -2]);
        assert_eq!(og_velocity[0], 2);

        // next decay value would be 0, but we handle this case explicitly
        // Hence there will be an update to weights once we run optim.update again
        assert_eq!(og_velocity[0] / (1 << optim.momentum_shift.unwrap_or(0)), 0);
        
        optim.update(&mut weights, &grad, &mut state, 0);
        assert_ne!(weights, vec![-2, -2]);
        assert_eq!(weights, vec![-5, -5]);
    }

    #[test]
    fn test_sgd_with_momentum_update(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0)
            // ~ shift by 1
            .with_momentum(0.88);

        let mut state = optim.init_state(2);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        let grad_bias = vec![2; 1];

        // will be [0, 0] -> [-2, -2]
        let mut weights = vec![0; 2];

        // will be [1, 1] -> [-2, -2]
        let mut bias = vec![1; 1];

        optim.update(&mut weights, &grad, &mut state, 0);
        optim.update(&mut bias, &grad_bias, &mut state, 0);

        assert_eq!(weights, vec![-2, -2]);
        assert_eq!(bias, vec![-2]);
    }

    #[test]
    fn test_sgd_with_momentum_update_check_lr_influence(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(0.5)
            // momentum shift ~ 2
            .with_momentum(0.5);

        let mut state = optim.init_state(2);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-1, -1]
        let mut weights = vec![0; 2];
        // will be [1, 1] -> [0, 0]
        let mut bias = vec![1; 2];

        optim.update(&mut weights, &grad, &mut state, 0);
        optim.update(&mut bias, &grad, &mut state, 0);

        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }
    
    #[test]
    fn test_sgd_update_too_small_state_size(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(0.5)
            // momentum shift ~ 2
            .with_momentum(0.5);

        let mut state = optim.init_state(1);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-1, -1]
        let mut weights = vec![0; 2];
        // will be [1, 1] -> [0, 0]
        let mut bias = vec![1; 2];

        optim.update(&mut weights, &grad, &mut state, 0);
        optim.update(&mut bias, &grad, &mut state, 0);

        // weights[1] and bias[1] are not updated
        assert_eq!(weights, vec![-1, 0]);
        assert_eq!(bias, vec![0, 1]);
    }

    #[test]
    fn test_sgd_update_too_big_state_size_no_issue_except_velocity(){
        let optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(0.5)
            // momentum shift ~ 2
            .with_momentum(0.5);

        let mut state = optim.init_state(3);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-1, -1]
        let mut weights = vec![0; 2];
        // will be [1, 1] -> [0, 0]
        let mut bias = vec![1; 2];

        optim.update(&mut weights, &grad, &mut state, 0);
        optim.update(&mut bias, &grad, &mut state, 0);


        // the full matrix was updated
        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }

    #[test]
    fn test_adam_new(){
        let optim = AdamConfig::new();

        assert_eq!(optim.lr_shift, 2);
        assert_eq!(optim.b1_shift, 3);
        assert_eq!(optim.b2_shift, 4);
        assert_eq!(optim.eps, 1);
    }

    #[test]
    fn test_display_adam() {
        let optim = AdamConfig::new();

        let content = format!("{:}", optim);

        assert_eq!(content, "Adam(lr=2, β1=3, β2=4)");
    }

    #[test]
    fn test_adam_learn_rate_to_shift(){
        let optim = AdamConfig::new();

        let samples = vec![
            1.0, 0.5, 0.25,
            0.125, 0.0625, 0.03125,
            0.015625, 0.0078125, 0.00390625,
        ];
        let expected_values = vec![
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ];
        for (val, expected_f32) in samples.iter().zip(expected_values) {
            assert_eq!(optim.learn_rate_to_shift(*val), expected_f32);
        }
    }

    #[test]
    fn test_adam_beta_to_shift(){
        let optim = AdamConfig::new();

        let samples = vec![
            0.997, 0.99, 0.98,
            0.97, 0.95, 0.88,
            0.82, 0.50, 0.20,
        ];
        let expected_values = vec![
            8, 7, 6, 5, 4, 3, 2, 1, 0
        ];
        for (val, expected_f32) in samples.iter().zip(expected_values) {
            assert_eq!(optim.beta_to_shift(*val), expected_f32);
        }
    }

    #[test]
    fn test_adam_with_betas_sets_betas(){
        let mut optim = AdamConfig::new();

        optim = optim.with_betas(0.997, 0.5);

        assert_eq!(optim.b1_shift, 8);
        assert_eq!(optim.b2_shift, 1);
    }

    #[test]
    fn test_with_learn_rate_sets_learn_rate(){
        let mut optim = AdamConfig::new();

        optim = optim.with_learn_rate(0.0625);

        assert_eq!(optim.lr_shift, 4);
    }

    #[test]
    fn test_with_eps_sets_eps(){
        let mut optim = AdamConfig::new();

        optim = optim.with_eps(0.0625);

        assert_eq!(optim.eps, 0);

        optim = optim.with_eps(2.0625);

        assert_eq!(optim.eps, 2);
    }

    #[test]
    fn test_adam_init_state_creates_correct_length_vectors(){
        let optim = AdamConfig::new();

        for size in [0, 1, 12] {
            let state = optim.init_state(size);

            let OptimizerState::Adam {m: momentum, v: velocities} = state else {
                unreachable!("AdamConfig::init_state must return Adam");
            };

            assert_eq!(momentum, vec![0; size]);
            assert_eq!(velocities, vec![0i32; size]);
        }
    }

    #[test]
    fn test_adam_update_with_sgd_state(){
        let sgd = SGDConfig::new();

        let adam = AdamConfig::new();
        let mut weights = vec![1; 2];
        let grads = vec![1; 2];

        adam.update(&mut weights, &grads, &mut sgd.init_state(2), 0);
    }

    #[test]
    fn test_adam_update(){
        let optim = AdamConfig::new()
            .with_learn_rate(1.0)
            .with_betas(0.0, 0.0)
            .with_eps(0.0);

        assert_eq!(optim.lr_shift, 0);
        assert_eq!(optim.b1_shift, 0);
        assert_eq!(optim.b2_shift, 0);
        assert_eq!(optim.eps, 0);

        let mut weights = vec![1; 2];
        let mut bias = vec![0; 1];

        let mut w_state = optim.init_state(2);
        let mut b_state = optim.init_state(1);
        let w_grads = vec![4; 2];
        let b_grads = vec![4; 1];

        optim.update(&mut weights, &w_grads, &mut w_state, 0);
        optim.update(&mut bias, &b_grads, &mut b_state, 0);

        let (w_momentum, w_velocities) = match &w_state {
            OptimizerState::Adam { m, v } => (m.clone(), v.clone()),
            _ => panic!("Should not happen!"),
        };

        let (b_momentum, b_velocities) = match &b_state {
            OptimizerState::Adam { m, v } => (m.clone(), v.clone()),
            _ => panic!("Should not happen!"),
        };

        // Computes for each momentum value: 0 - (0 / 1) + 4 / 1 = 4;
        assert_eq!(w_momentum[0], 4);
        assert_eq!(w_momentum[1], 4);
        assert_eq!(b_momentum[0], 4);

        // Computes for each velocity value: 0 - (0 / 1) + (4 * 4) / 1 = 16;
        assert_eq!(w_velocities[0], 16);
        assert_eq!(w_velocities[1], 16);
        assert_eq!(b_velocities[0], 16);


        // w - (momentum / 1) / (sqrt(velocity) + eps)
        // 1 - ((4 / 1) / (4 + 0)) = 1 - 4/4 = 0
        assert_eq!(weights[0], 0);
        assert_eq!(weights[1], 0);
        // 0 - ((4 / 1) / (4 + 0)) = 0 - 4/4 = -1
        assert_eq!(bias[0], -1);


    }

    #[test]
    fn test_adam_update_with_learn_rate(){
        let optim = AdamConfig::new()
            .with_learn_rate(2.0)
            .with_betas(0.0, 0.0)
            .with_eps(0.0);

        assert_eq!(optim.lr_shift, 0);
        assert_eq!(optim.b1_shift, 0);
        assert_eq!(optim.b2_shift, 0);
        assert_eq!(optim.eps, 0);

        let mut weights = vec![1; 2];
        let mut bias = vec![0; 1];

        let grads = vec![4; 2];

        let mut state = optim.init_state(2);
        optim.update(&mut weights, &grads, &mut state, 0);
        optim.update(&mut bias, &grads, &mut state, 0);

        assert_eq!(weights[0], 0);
        assert_eq!(weights[1], 0);

        assert_eq!(bias[0], -1);
    }

    #[test]
    fn test_adam_update_with_eps(){
        let optim = AdamConfig::new()
            .with_learn_rate(1.0)
            .with_betas(0.0, 0.0)
            .with_eps(1.0);

        assert_eq!(optim.lr_shift, 0);
        assert_eq!(optim.b1_shift, 0);
        assert_eq!(optim.b2_shift, 0);
        assert_eq!(optim.eps, 1);

        let mut weights = vec![1; 2];
        let mut bias = vec![0; 1];

        let grads = vec![4; 2];

        let mut state = optim.init_state(2);
        optim.update(&mut weights, &grads, &mut state, 0);
        optim.update(&mut bias, &grads, &mut state, 0);

        assert_eq!(weights[0], 1);
        assert_eq!(weights[1], 1);

        assert_eq!(bias[0], 0);
    }
}
