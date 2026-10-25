//! Independent scalar FP32 counterpart of the fixed MNIST CNN.
//! No dyadic forward/backward/update helpers are called. Only initialization
//! copies decoded integer parameters, before either backend begins training.
use integers::dyadic::Dyadic;
use integers::nn::{Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential};
use integers::rng::seed_rng;

const SHIFT: u32 = 7;
const BITS: u32 = 31;
const GRAD_CLIP: i32 = 8192;
const MOM_SHIFT: u32 = 1;

fn decode(x: Dyadic) -> f32 { (x.v as f32) * 2f32.powi(-(x.s as i32)) }

struct Parameters {
    weights: Vec<f32>, bias: Vec<f32>,
    grad_weights: Vec<f32>, grad_bias: Vec<f32>,
    velocity_weights: Vec<f32>, velocity_bias: Vec<f32>,
}
impl Parameters {
    fn new(weights: &[Dyadic], bias: &[Dyadic]) -> Self {
        let weights: Vec<_> = weights.iter().copied().map(decode).collect();
        let bias: Vec<_> = bias.iter().copied().map(decode).collect();
        Self {
            grad_weights: vec![0.0; weights.len()],
            grad_bias: vec![0.0; bias.len()],
            velocity_weights: vec![0.0; weights.len()],
            velocity_bias: vec![0.0; bias.len()],
            weights, bias,
        }
    }
    fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    }
    fn update(&mut self, lr_shift: u32, samples: usize) {
        assert!(samples > 0, "cannot update FP32 without samples");
        let scale = 1.0 / samples as f32;
        let momentum = 2f32.powi(-(MOM_SHIFT as i32));
        let lr = 2f32.powi(-(lr_shift as i32));
        for i in 0..self.weights.len() {
            self.velocity_weights[i] = momentum * self.velocity_weights[i] + scale * self.grad_weights[i];
            self.weights[i] -= lr * self.velocity_weights[i];
        }
        for i in 0..self.bias.len() {
            self.velocity_bias[i] = momentum * self.velocity_bias[i] + scale * self.grad_bias[i];
            self.bias[i] -= lr * self.velocity_bias[i];
        }
    }
    fn nonfinite(&self) -> usize {
        self.weights.iter().chain(self.bias.iter())
            .chain(self.velocity_weights.iter()).chain(self.velocity_bias.iter())
            .filter(|v| !v.is_finite()).count()
    }
}

struct Convolution {
    cin: usize, cout: usize, h: usize, w: usize, kh: usize, kw: usize,
    params: Parameters,
}
impl Convolution {
    fn from_integer(src: &Conv2D) -> Self {
        Self {
            cin: src.in_channels, cout: src.out_channels,
            h: src.in_h, w: src.in_w, kh: src.kernel_h, kw: src.kernel_w,
            params: Parameters::new(&src.kernels.data, &src.biases.data),
        }
    }
    fn out_h(&self) -> usize { self.h - self.kh + 1 }
    fn out_w(&self) -> usize { self.w - self.kw + 1 }
    fn kernel_index(&self, oc: usize, ic: usize, y: usize, x: usize) -> usize {
        ((oc * self.cin + ic) * self.kh + y) * self.kw + x
    }
    fn input_index(&self, ic: usize, y: usize, x: usize) -> usize { (ic * self.h + y) * self.w + x }
    fn output_index(&self, oc: usize, y: usize, x: usize) -> usize {
        (oc * self.out_h() + y) * self.out_w() + x
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cin * self.h * self.w, "FP32 conv input shape");
        let mut y = vec![0.0; self.cout * self.out_h() * self.out_w()];
        for oc in 0..self.cout {
            for oh in 0..self.out_h() {
                for ow in 0..self.out_w() {
                    let mut sum = self.params.bias[oc];
                    for ic in 0..self.cin {
                        for kh in 0..self.kh {
                            for kw in 0..self.kw {
                                sum += x[self.input_index(ic, oh + kh, ow + kw)]
                                    * self.params.weights[self.kernel_index(oc, ic, kh, kw)];
                            }
                        }
                    }
                    y[self.output_index(oc, oh, ow)] = sum;
                }
            }
        }
        y
    }
    fn backward(&mut self, input: &[f32], grad: &[f32]) -> Vec<f32> {
        assert_eq!(grad.len(), self.cout * self.out_h() * self.out_w(), "FP32 conv gradient shape");
        let mut dx = vec![0.0; input.len()];
        let max_grad = GRAD_CLIP as f32 * 2f32.powi(-(SHIFT as i32));
        for oc in 0..self.cout {
            for oh in 0..self.out_h() {
                for ow in 0..self.out_w() {
                    let g = grad[self.output_index(oc, oh, ow)].clamp(-max_grad, max_grad);
                    self.params.grad_bias[oc] += g;
                    for ic in 0..self.cin {
                        for kh in 0..self.kh {
                            for kw in 0..self.kw {
                                let wi = self.kernel_index(oc, ic, kh, kw);
                                let ii = self.input_index(ic, oh + kh, ow + kw);
                                self.params.grad_weights[wi] += g * input[ii];
                                dx[ii] += g * self.params.weights[wi];
                            }
                        }
                    }
                }
            }
        }
        dx
    }
}

struct Dense { input: usize, output: usize, params: Parameters }
impl Dense {
    fn from_integer(src: &Linear) -> Self {
        Self {
            input: src.in_features, output: src.out_features,
            params: Parameters::new(&src.weights.data, &src.biases.data),
        }
    }
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input, "FP32 dense input shape");
        let mut output = self.params.bias.clone();
        for j in 0..self.output {
            for i in 0..self.input {
                output[j] += input[i] * self.params.weights[j * self.input + i];
            }
        }
        output
    }
    fn backward(&mut self, input: &[f32], grad: &[f32]) -> Vec<f32> {
        assert_eq!(grad.len(), self.output, "FP32 dense gradient shape");
        let mut dx = vec![0.0; self.input];
        let max_grad = GRAD_CLIP as f32 * 2f32.powi(-(SHIFT as i32));
        for j in 0..self.output {
            let g = grad[j].clamp(-max_grad, max_grad);
            self.params.grad_bias[j] += g;
            for i in 0..self.input {
                let wi = j * self.input + i;
                self.params.grad_weights[wi] += g * input[i];
                dx[i] += g * self.params.weights[wi];
            }
        }
        dx
    }
}

struct Pool {
    channels: usize, h: usize, w: usize,
    kernel: usize, stride: usize, winners: Vec<usize>,
}
impl Pool {
    fn new(channels: usize, h: usize, w: usize, kernel: usize, stride: usize) -> Self {
        Self { channels, h, w, kernel, stride, winners: Vec::new() }
    }
    fn output_dims(&self) -> (usize, usize) {
        ((self.h - self.kernel) / self.stride + 1, (self.w - self.kernel) / self.stride + 1)
    }
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.channels * self.h * self.w);
        self.winners.clear();
        let (oh, ow) = self.output_dims();
        let mut out = Vec::with_capacity(self.channels * oh * ow);
        for c in 0..self.channels {
            for y in 0..oh {
                for x in 0..ow {
                    let mut winner = (c * self.h + y * self.stride) * self.w + x * self.stride;
                    for ky in 0..self.kernel {
                        for kx in 0..self.kernel {
                            let at = (c * self.h + y * self.stride + ky) * self.w + x * self.stride + kx;
                            // A strict comparison guarantees first-index tie routing.
                            if input[at] > input[winner] { winner = at; }
                        }
                    }
                    self.winners.push(winner);
                    out.push(input[winner]);
                }
            }
        }
        out
    }
    fn backward(&self, grad: &[f32]) -> Vec<f32> {
        assert_eq!(grad.len(), self.winners.len(), "FP32 pool gradient shape");
        let mut dx = vec![0.0; self.channels * self.h * self.w];
        for (&winner, &g) in self.winners.iter().zip(grad) { dx[winner] += g; }
        dx
    }
}

enum Op { Conv(Convolution), Relu, Pool(Pool), Flatten, Linear(Dense) }
pub struct CnnFp32 { ops: Vec<Op>, inputs: Vec<Vec<f32>> }
impl CnnFp32 {
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), 784, "MNIST CNN expects a 28x28 input");
        self.inputs.clear();
        let mut current = input.to_vec();
        for op in &mut self.ops {
            self.inputs.push(current.clone());
            current = match op {
                Op::Conv(layer) => layer.forward(&current),
                Op::Relu => current.iter().map(|&v| v.max(0.0)).collect(),
                Op::Pool(layer) => layer.forward(&current),
                Op::Flatten => current,
                Op::Linear(layer) => layer.forward(&current),
            };
        }
        assert_eq!(current.len(), 10);
        current
    }
    pub fn accumulate(&mut self, input: &[f32], target: &[f32]) -> usize {
        assert_eq!(target.len(), 10, "MNIST one-hot target");
        let y = self.forward(input);
        // Integer argmax chooses the first maximum on ties.
        let mut prediction = 0;
        for i in 1..y.len() {
            if y[i] > y[prediction] { prediction = i; }
        }
        // d/dy [0.5 sum_j (y_j - t_j)^2] = y - t.
        let mut grad: Vec<f32> = y.iter().zip(target).map(|(&v, &t)| v - t).collect();
        for (op, x) in self.ops.iter_mut().zip(&self.inputs).rev() {
            grad = match op {
                Op::Conv(layer) => layer.backward(x, &grad),
                Op::Relu => {
                    assert_eq!(x.len(), grad.len());
                    grad.iter().zip(x).map(|(&g, &v)| if v > 0.0 { g } else { 0.0 }).collect()
                }
                Op::Pool(layer) => layer.backward(&grad),
                Op::Flatten => grad,
                Op::Linear(layer) => layer.backward(x, &grad),
            };
        }
        assert_eq!(grad.len(), 784);
        prediction
    }
    pub fn zero_grad(&mut self) {
        for op in &mut self.ops {
            match op {
                Op::Conv(c) => c.params.zero_grad(),
                Op::Linear(c) => c.params.zero_grad(),
                _ => {}
            }
        }
    }
    pub fn update(&mut self, lr_shift: u32, samples: usize) {
        for op in &mut self.ops {
            match op {
                Op::Conv(c) => c.params.update(lr_shift, samples),
                Op::Linear(c) => c.params.update(lr_shift, samples),
                _ => {}
            }
        }
    }
    pub fn nonfinite_parameters(&self) -> usize {
        self.ops.iter().map(|op| match op {
            Op::Conv(c) => c.params.nonfinite(),
            Op::Linear(c) => c.params.nonfinite(),
            _ => 0,
        }).sum()
    }
}

/// The *only* shared numerical data are decoded initial parameters. Each model
/// has independent forward/backward/update implementations thereafter.
pub fn build_models(seed: u64) -> (Sequential, CnnFp32) {
    seed_rng(seed);
    let mut integer = Sequential::new();
    let mut ops = Vec::new();

    let a = Conv2D::new(1, 4, 3, 3, 28, 28, SHIFT, SHIFT, BITS)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT);
    ops.push(Op::Conv(Convolution::from_integer(&a)));
    integer.add(a);
    integer.add(ReLU::new()); ops.push(Op::Relu);
    integer.add(MaxPool2D::new(4, 26, 26, 2, 2));
    ops.push(Op::Pool(Pool::new(4, 26, 26, 2, 2)));

    let b = Conv2D::new(4, 8, 3, 3, 13, 13, SHIFT, SHIFT, BITS)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT);
    ops.push(Op::Conv(Convolution::from_integer(&b)));
    integer.add(b);
    integer.add(ReLU::new()); ops.push(Op::Relu);
    integer.add(MaxPool2D::new(8, 11, 11, 2, 2));
    ops.push(Op::Pool(Pool::new(8, 11, 11, 2, 2)));

    integer.add(Flatten); ops.push(Op::Flatten);
    let c = Linear::new(200, 64, SHIFT, SHIFT, BITS)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT);
    ops.push(Op::Linear(Dense::from_integer(&c)));
    integer.add(c);
    integer.add(ReLU::new()); ops.push(Op::Relu);
    let d = Linear::new(64, 10, SHIFT, SHIFT, BITS)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT);
    ops.push(Op::Linear(Dense::from_integer(&d)));
    integer.add(d);
    (integer, CnnFp32 { ops, inputs: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_ties_overlap_and_gradient_scatter() {
        let mut pool = Pool::new(1, 3, 3, 2, 1);
        let x = [1., 2., 3., 4., 9., 9., 7., 9., 9.];
        assert_eq!(pool.forward(&x), vec![9., 9., 9., 9.]);
        assert_eq!(pool.winners, vec![4, 4, 4, 4]);
        assert_eq!(pool.backward(&[1., 2., 3., 4.]), vec![0.,0.,0.,0.,10.,0.,0.,0.,0.]);
    }
    #[test]
    fn multichannel_conv_accumulates_correct_weight_gradient() {
        let mut c = Conv2D::new(2, 1, 2, 2, 3, 3, 0, 0, 31);
        c.kernels.data.fill(Dyadic::new(1, 0));
        c.biases.data[0] = Dyadic::new(0, 0);
        let mut f = Convolution::from_integer(&c);
        let x: Vec<f32> = (1..=18).map(|n| n as f32).collect();
        assert_eq!(f.forward(&x), vec![60.,68.,84.,92.]);
        let dx = f.backward(&x, &[1.,1.,1.,1.]);
        assert_eq!(dx[4], 4.0); // center of first channel in all 4 windows
        assert_eq!(dx[13], 4.0); // center of second channel
        assert_eq!(f.params.grad_bias, vec![4.]);
        assert_eq!(f.params.grad_weights[0], 1. + 2. + 4. + 5.);
        assert_eq!(f.params.grad_weights[4], 10. + 11. + 13. + 14.);
    }
}
