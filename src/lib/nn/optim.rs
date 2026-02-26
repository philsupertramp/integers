use crate::nn::kernels;
use crate::{checked_sub_i16, checked_add_i16};

use std::fmt;

/// Weight Quantization Configuration
///
/// Configure how to scale weights from i32 (master) to i8 (storage).
///
#[derive(Clone, Copy, Debug)]
pub enum WeightQuantization {
    Bits(u32),
    Scale(f32),
    ActivationRange(f32)
}

impl WeightQuantization {
    pub fn to_shift(self) -> u32 {
        match self {
            WeightQuantization::Bits(n) => {
                assert!(n >= 1 && n <= 8, "Bits must be in 1-8, got {}", n);
                n
            }
            WeightQuantization::Scale(scale) => {
                assert!(scale > 0.0, "Scale must be positive, got {}", scale);
                let inv = 1.0 / scale;
                let shift = inv.log2().round() as u32;
                assert!(shift >= 1 && shift <= 8, "Scale {} gives shift {}, must be in 1-8", scale, shift);
                shift
            }
            WeightQuantization::ActivationRange(max) => {
                assert!(max > 0.0, "ActivationRange must be positive, got {}", max);
                let shift = (127.0 / max).log2().ceil() as u32;
                assert!(shift >= 1 && shift <= 8, "Range {} gives shift {}, must be in range 1-8", max, shift);
                shift
            }
        }
    }

    pub fn scale_factor(self) -> f32 {
        1.0 / (2_u32.pow(self.to_shift()) as f32)
    }

    pub fn shift(self) -> u32 {
        self.to_shift()
    }
}

impl fmt::Display for WeightQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightQuantization::Bits(n) => write!(f, "{}B quantization", n),
            WeightQuantization::Scale(s) => write!(f, "1/{:.0} scale", 1.0/s),
            WeightQuantization::ActivationRange(m) => write!(f, "range ±{:.1}", m),
        }
    }
}

/// Gradient Scaling Configuration
///
/// Configure how to scale gradients during a backward pass.
///
/// In deep networks, gradients multiply through layers and can overflow i16
/// This controls the divide-by-2^N applied to all gradients.
#[derive(Clone, Copy, Debug)]
pub enum GradientScaling {
    DivideBy(u32),

    ForDepth(usize),
}

impl GradientScaling {
    pub fn to_shift(self) -> u32 {
        match self {
            GradientScaling::DivideBy(divisor) => {
                let shift = divisor.next_power_of_two().trailing_zeros();
                assert!(shift >= 4 && shift <= 10, "DivideBy {} gives shift {}, must be in 4-10.", divisor, shift);
                shift
            }
            GradientScaling::ForDepth(depth) => {
                let shift = 5 + (depth as u32);

                if shift > 30 {
                    eprintln!(
                        "WARNING: GradientScaling::ForDepth({}) gives shift={}\n
                         Dividing gradients by 2^{} (very aggressive).\n
                         This may indicate you need architectural changes like:\n
                         - Batch normalization\n
                         - Layer normalization\n
                         - Residual connections\n
                         - Or explicitly use GradientScaling::DivideBy() if intentional",
                        depth, shift, shift
                    )
                }
                shift
            }
        }
    }

    pub fn divisor(self) -> u32 {
        2_u32.pow(self.to_shift())
    }

    pub fn shift(self) -> u32 {
        self.to_shift()
    }
}

impl fmt::Display for GradientScaling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradientScaling::DivideBy(div) => write!(f, "÷{}", div),
            GradientScaling::ForDepth(d) => write!(f, "auto(depth={})", d),
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
    pub fn new() -> Self {
        Self {
            lr_shift: 2,
            momentum_shift: None,
        }
    }

    fn learn_rate_to_shift(&self, learn_rate: f32) -> u32 {
        let inv = 1.0 / learn_rate;
        let shift = inv.log2().round() as u32;
        assert!(shift >= 0 && shift <= 8, "Learning rate {} gives shift {}, should be 0-8", learn_rate, shift);
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
        assert!(shift >= 0 && shift <= 8, "Momentum {} gives shift {}, should be 0-8", momentum, shift);
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
        assert!(shift >= 0 && shift <= 8, "Learning rate {} gives shift {}, should be 0-8", learn_rate, shift);
        shift
    }

    fn beta_to_shift(&self, beta: f32) -> u32 {
        // beta = 1 - 1/2^shift
        // shift = log2(1 / (1 - beta))
        let inv = 1.0 / (1.0 - beta);
        let shift = inv.log2().round() as u32;
        assert!(shift >= 0 && shift <= 8, "Beta {} gives shift {}, should be 0-8", beta, shift);
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
            v: vec![0i64; len],
        }
    }

    fn update(&self, weights: &mut [i32], grads: &[i32], state: &mut OptimizerState) {
        if let OptimizerState::Adam { m, v } = state {
            let b1_div = 1 << self.b1_shift;
            let b2_div = 1 << self.b2_shift;
            let lr_div = 1 << self.lr_shift;

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
                    (m[i] / lr_div) / denom,
                    backward_wraps
                );
            }
        }
    }
}

