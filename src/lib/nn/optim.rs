use crate::nn::kernels;
use crate::{checked_sub_i16, checked_add_i16};

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
                    *w = checked_sub_i16!(
                        w,
                        *m / lr_div,
                        backward_wraps
                    );
                }
            }
            _ => {
                for (w, g) in weights.iter_mut().zip(grads) {
                    *w = checked_sub_i16!(
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
                m[i] = checked_add_i16!(
                    m[i].wrapping_sub(m[i] / b1_div),
                    g,
                    backward_wraps
                );
                v[i] = checked_add_i16!(
                    v[i].wrapping_sub(v[i] / (b2_div as i64)),
                    g_64 * g_64,
                    backward_wraps
                );
                let denom = kernels::isqrt_64(v[i].max(0) as u64) as i32 + self.eps;
                weights[i] = checked_sub_i16!(
                    weights[i],
                    (m[i] * self.lr_mult) / denom,
                    backward_wraps
                );
            }
        }
    }
}

