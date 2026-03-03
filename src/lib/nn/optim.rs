use crate::nn::kernels;
use crate::{checked_sub_counting, checked_add_counting};

use std::fmt;

pub trait OptimizerConfig {
    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState);
    fn init_state(&self, len: usize) -> OptimizerState;
}

#[derive(Debug, PartialEq)]
pub enum OptimizerState {
    None,
    SGD { velocity: Vec<i32> },
    Adam { m: Vec<i32>, v: Vec<i32> },
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
        let shift = inv.round() as u32;
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
                    *w = checked_sub_counting!(
                        w,
                        *m / lr_div,
                        backward_wraps
                    );
                }
            }
            _ => {
                for (w, g) in weights.iter_mut().zip(grads) {
                    *w = checked_sub_counting!(
                        w,
                        *g / lr_div,
                        backward_wraps
                    );
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
impl OptimizerConfig for AdamConfig {
    fn init_state(&self, len: usize) -> OptimizerState {
        OptimizerState::Adam {
            m: vec![0; len],
            v: vec![0i32; len],
        }
    }

    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState) {
        if let OptimizerState::Adam { m, v } = state {
            let b1_div = 1 << self.b1_shift;
            let b2_div = 1 << self.b2_shift;
            let lr_div = 1 << self.lr_shift;

            for i in 0..weights.len() {
                let g = grads[i];
                let g_64 = grads[i] as i32;
                m[i] = checked_add_counting!(
                    m[i].wrapping_sub(m[i] / b1_div),
                    g / b1_div,
                    backward_wraps
                );
                v[i] = checked_add_counting!(
                    v[i].wrapping_sub(v[i] / (b2_div as i32)),
                    g_64 * g_64 / b2_div as i32,
                    backward_wraps
                );
                let denom = kernels::isqrt_64(v[i].max(0) as u64) as i32 + self.eps;
                weights[i] = checked_sub_counting!(
                    weights[i],
                    (m[i] / lr_div) / denom,
                    backward_wraps
                );
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
    fn test_sgd_with_momentum(){
        let mut optim = SGDConfig::new();

        assert_eq!(optim.momentum_shift, None);

        let samples = vec![
            0.88, 0.85, 0.82,
            0.80, 0.75, 0.70,
            0.50, 0.25, 0.125,
            0.0625,
        ];
        let expected_values = vec![
            8, 7, 6, 5, 4, 3, 2,
            1, 1, 1,
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
        let mut optim = SGDConfig::new();

        let mut state = optim.init_state(5);

        let mut input = vec![2; 2];
        let grad = vec![2; 3];
        optim.update(&mut input, &grad, &mut state);
    }

    #[test]
    fn test_sgd_update_sets_velocities(){
        let mut optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0);

        let mut state = optim.init_state(0);

        match state {
            OptimizerState::SGD { ref velocity } => {
                let grad = vec![2; 2];
                // will be [0, 0] -> [-2, -2]
                let mut weights = vec![0; 2];

                let og_velocity = velocity.clone();
                optim.update(&mut weights, &grad, &mut state);

                let new_velocity = velocity.clone();

                for (og_vel, new_vel) in og_velocity.into_iter().zip(new_velocity.into_iter()) {
                    assert!(og_vel != new_vel);
                }
            }
            OptimizerState::Adam { m: _, v: _ } => {}
            OptimizerState::None => {}
        }

    }

    #[test]
    fn test_sgd_no_momentum_update(){
        let mut optim = SGDConfig::new()
            // grad update will be applied fully
            .with_learn_rate(1.0);

        let mut state = optim.init_state(0);

        // somewhere we computed this gradient, we will apply it to 
        // some weights and some bias
        let grad = vec![2; 2];
        // will be [0, 0] -> [-2, -2]
        let mut weights = vec![0; 2];

        optim.update(&mut weights, &grad, &mut state);

        // will be [1, 1] -> [-1, -1]
        let mut bias = vec![1; 2];

        optim.update(&mut bias, &grad, &mut state);

        assert_eq!(weights, vec![-2, -2]);
        assert_eq!(bias, vec![-1, -1]);
    }

    #[test]
    fn test_sgd_no_momentum_update_check_lr_influence(){
        let mut optim = SGDConfig::new()
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

        optim.update(&mut weights, &grad, &mut state);
        optim.update(&mut bias, &grad, &mut state);

        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }

    #[test]
    fn test_sgd_with_momentum_update(){
        let mut optim = SGDConfig::new()
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

        optim.update(&mut weights, &grad, &mut state);
        optim.update(&mut bias, &grad_bias, &mut state);

        assert_eq!(weights, vec![-2, -2]);
        assert_eq!(bias, vec![-2]);
    }

    #[test]
    fn test_sgd_with_momentum_update_check_lr_influence(){
        let mut optim = SGDConfig::new()
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

        optim.update(&mut weights, &grad, &mut state);
        optim.update(&mut bias, &grad, &mut state);

        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }
    
    #[test]
    fn test_sgd_update_too_small_state_size(){
        let mut optim = SGDConfig::new()
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

        optim.update(&mut weights, &grad, &mut state);
        optim.update(&mut bias, &grad, &mut state);

        // weights[1] and bias[1] are not updated
        assert_eq!(weights, vec![-1, 0]);
        assert_eq!(bias, vec![0, 1]);
    }

    #[test]
    fn test_sgd_update_too_big_state_size_no_issue_except_velocity(){
        let mut optim = SGDConfig::new()
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
        println!("Vel: {:?}", state);

        optim.update(&mut weights, &grad, &mut state);
        println!("Vel: {:?}", state);
        optim.update(&mut bias, &grad, &mut state);

        println!("Vel: {:?}", state);

        // the full matrix was updated
        assert_eq!(weights, vec![-1, -1]);
        assert_eq!(bias, vec![0, 0]);
    }
}
