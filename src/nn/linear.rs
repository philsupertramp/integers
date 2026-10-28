use crate::dyadic::{
    Dyadic, Tensor, TensorView, add, discard_mul_with_rng, mul_with_rng, requantize, signed_bounds,
    ste_requantize,
};
use crate::nn::{Module, TrainingArithmetic, apply_updates_mean};
use crate::rng::{rng_range, with_rng_state};

/// Fully-connected affine layer: `y = ℛ(W⊗x ⊕ b)`.
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Tensor,
    pub biases: Tensor,
    pub quant_shift: u32,
    pub output_bits: u32,
    pub grad_clip: i32,
    pub momentum_shift: Option<u32>,
    // per-sample caches (single call)
    input_cache: Tensor,
    output_cache: Tensor,
    // batch caches (batch call)
    input_batch_cache: Tensor,
    output_batch_cache: Tensor,
    clip_cache: Vec<bool>,
    clip_batch_cache: Vec<Vec<bool>>,
    // gradients + velocity
    grad_w: Tensor,
    grad_b: Tensor,
    vel_w: Tensor,
    vel_b: Tensor,
    grad_samples: usize,

    q_min: i32,
    q_max: i32,
}

/// Same-scale gradient accumulation in the dense backward hot loop.
/// When exponents match, add() performs exactly saturating mantissa addition
/// without RNG draws; otherwise keep the full dyadic alignment path.
/// No multiplication, arithmetic order, overflow, or RNG-stream changes.
#[inline(always)]
fn add_backward_accumulator(a: Dyadic, b: Dyadic) -> Dyadic {
    if a.s == b.s {
        Dyadic::new(a.v.saturating_add(b.v), a.s)
    } else {
        add(a, b)
    }
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight_shift: u32,
        quant_shift: u32,
        output_bits: u32,
    ) -> Self {
        let n_w = out_features * in_features;
        let k = ((1u32 << weight_shift) as f64 / (in_features as f64).sqrt()).round() as i32;
        let k = k.max(1);
        let z = |n: usize| vec![Dyadic::new(0, weight_shift); n];
        let weights = Tensor::from_vec(
            (0..n_w)
                .map(|_| Dyadic::new(-k + rng_range(2 * k as u32) as i32, weight_shift))
                .collect(),
            vec![out_features, in_features],
        );
        let biases = Tensor::from_vec(z(out_features), vec![out_features]);
        let (q_min, q_max) = signed_bounds(output_bits);
        Self {
            in_features,
            out_features,
            weights,
            biases,
            quant_shift,
            output_bits,
            grad_clip: i32::MAX,
            momentum_shift: None,
            input_cache: Tensor::new(),
            output_cache: Tensor::new(),
            input_batch_cache: Tensor::new(),
            output_batch_cache: Tensor::new(),
            clip_cache: Vec::new(),
            clip_batch_cache: Vec::new(),
            grad_w: Tensor::zeros(vec![n_w]),
            grad_b: Tensor::zeros(vec![out_features]),
            vel_w: Tensor::zeros(vec![n_w]),
            vel_b: Tensor::zeros(vec![out_features]),
            grad_samples: 0,
            q_min,
            q_max,
        }
    }

    pub fn with_grad_clip(mut self, c: i32) -> Self {
        self.grad_clip = c.abs();
        self
    }
    pub fn with_momentum(mut self, s: u32) -> Self {
        self.momentum_shift = Some(s);
        self
    }

    #[inline]
    fn w(&self, j: usize, i: usize) -> Dyadic {
        self.weights.data[j * self.in_features + i]
    }
    #[inline]
    fn flat(&self, j: usize, i: usize) -> usize {
        j * self.in_features + i
    }

    fn forward_one(&mut self, input: &TensorView) -> (Tensor, Vec<bool>) {
        assert_eq!(
            input.data.len(),
            self.in_features,
            "Linear: input feature count mismatch"
        );
        let mut out = Vec::with_capacity(self.out_features);
        let mut clipped = Vec::with_capacity(self.out_features);
        // Keep the original output/feature traversal and one stochastic draw
        // per dyadic product (including zero products at q > 0). add() only
        // aligns to the finer exponent, and requantize(acc, acc.s, ...) does
        // not draw RNG. No global TLS RNG calls may be introduced inside
        // this scoped closure without revisiting the state-commit contract.
        with_rng_state(|rng| {
            for j in 0..self.out_features {
                let mut acc = self.biases.data[j];
                for i in 0..self.in_features {
                    acc = add(
                        acc,
                        mul_with_rng(self.w(j, i), input.data[i], self.quant_shift, rng),
                    );
                }
                let (y, was_clipped) = requantize(acc, acc.s, self.q_min, self.q_max);
                out.push(y);
                clipped.push(was_clipped);
            }
        });
        (Tensor::from_vec(out, vec![self.out_features]), clipped)
    }

    fn backward_one(
        &mut self,
        grad_output: &TensorView,
        input: &TensorView,
        output: &TensorView,
        clipped: &[bool],
    ) -> Tensor {
        assert_eq!(
            grad_output.data.len(),
            self.out_features,
            "Linear: output gradient count mismatch"
        );
        assert_eq!(
            input.data.len(),
            self.in_features,
            "Linear: missing or invalid forward cache"
        );
        assert_eq!(
            output.data.len(),
            self.out_features,
            "Linear: missing output cache"
        );
        assert_eq!(
            clipped.len(),
            self.out_features,
            "Linear: missing saturation cache"
        );
        let g_s = grad_output.data.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.in_features];
        // One TLS access per sample instead of one per dyadic product.
        // Both mul calls retain their original order. add() aligns to the
        // finer exponent and never draws RNG; do not introduce nested TLS
        // draws inside this scope without revisiting its contract.
        with_rng_state(|rng| {
            for (j, &is_clipped) in clipped.iter().enumerate() {
                let gr = if is_clipped {
                    Dyadic::new(0, grad_output.data[j].s)
                } else {
                    ste_requantize(
                        grad_output.data[j],
                        output.data[j].v,
                        self.q_min,
                        self.q_max,
                    )
                };
                let gj = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
                for (i, gi_value) in gi.iter_mut().enumerate() {
                    let idx = self.flat(j, i);

                    self.grad_w.data[idx] = add_backward_accumulator(
                        self.grad_w.data[idx],
                        mul_with_rng(gj, input.data[i], self.quant_shift, rng),
                    );

                    *gi_value = add_backward_accumulator(
                        *gi_value,
                        mul_with_rng(gj, self.w(j, i), self.quant_shift, rng),
                    );
                }
                self.grad_b.data[j] = add(self.grad_b.data[j], gj);
            }
        });
        self.grad_samples = self
            .grad_samples
            .checked_add(1)
            .expect("Linear: sample counter overflow");
        Tensor::from_vec(gi, vec![self.in_features])
    }

    /// Training-only variant: the model input is not trainable, so the first
    /// layer's input gradient would be thrown away by the caller.
    ///
    /// Preserve *every* weight gradient, bias gradient, mask, error and
    /// stochastic RNG draw in the same j/i traversal order as backward_one.
    /// Only the discarded input-gradient multiplication/accumulation is
    /// elided; discard_mul_with_rng handles its exact RNG contract.
    fn backward_one_discard_input_grad(
        &mut self,
        grad_output: &TensorView,
        input: &TensorView,
        output: &TensorView,
        clipped: &[bool],
    ) {
        assert_eq!(
            grad_output.data.len(),
            self.out_features,
            "Linear: output gradient count mismatch"
        );
        assert_eq!(
            input.data.len(),
            self.in_features,
            "Linear: missing or invalid forward cache"
        );
        assert_eq!(
            output.data.len(),
            self.out_features,
            "Linear: missing output cache"
        );
        assert_eq!(
            clipped.len(),
            self.out_features,
            "Linear: missing saturation cache"
        );

        with_rng_state(|rng| {
            for (j, &is_clipped) in clipped.iter().enumerate() {
                let gr = if is_clipped {
                    Dyadic::new(0, grad_output.data[j].s)
                } else {
                    ste_requantize(
                        grad_output.data[j],
                        output.data[j].v,
                        self.q_min,
                        self.q_max,
                    )
                };
                let gj = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
                for i in 0..self.in_features {
                    let idx = self.flat(j, i);
                    self.grad_w.data[idx] = add_backward_accumulator(
                        self.grad_w.data[idx],
                        mul_with_rng(gj, input.data[i], self.quant_shift, rng),
                    );
                    // This result is unused, but the corresponding product
                    // must still consume the exact original RNG stream.
                    discard_mul_with_rng(gj, self.w(j, i), self.quant_shift, rng);
                }
                self.grad_b.data[j] = add(self.grad_b.data[j], gj);
            }
        });
        self.grad_samples = self
            .grad_samples
            .checked_add(1)
            .expect("Linear: sample counter overflow");
    }
}

impl Module for Linear {
    fn training_arithmetic(&self) -> TrainingArithmetic {
        TrainingArithmetic::IntegerOnly
    }
    fn name(&self) -> &'static str {
        "Linear"
    }
    fn describe(&self) -> String {
        let ws = self.weights.data.first().map_or(0, |w| w.s);
        let clip = if self.grad_clip == i32::MAX {
            "off".into()
        } else {
            format!("2^{}", (self.grad_clip as f64).log2() as i32)
        };
        let mom = self
            .momentum_shift
            .map_or("off".into(), |m| format!("shift={m}"));
        format!(
            "Linear(in={}, out={}, w_shift={ws}, q={}, bits={}, clip={clip}, mom={mom})",
            self.in_features, self.out_features, self.quant_shift, self.output_bits
        )
    }

    fn forward(&mut self, input: &TensorView) -> Tensor {
        let (out, clipped) = self.forward_one(input);
        self.input_cache = input.to_tensor();
        self.output_cache = out.clone();
        self.clip_cache = clipped;
        out
    }

    fn backward(&mut self, grad_output: &TensorView) -> Tensor {
        let input = self.input_cache.clone();
        let output = self.output_cache.clone();
        let clipped = self.clip_cache.clone();
        self.backward_one(grad_output, &input.view(), &output.view(), &clipped)
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        assert!(
            inputs.shape.len() >= 2,
            "Linear: expected [batch, features]"
        );
        assert_eq!(
            inputs.shape[1..].iter().product::<usize>(),
            self.in_features,
            "Linear: batch feature shape mismatch"
        );
        assert_eq!(
            inputs.data.len(),
            inputs.shape[0] * self.in_features,
            "Linear: batch buffer length mismatch"
        );
        let mut clips = Vec::with_capacity(inputs.shape[0]);
        let outputs: Tensor = inputs
            .iter()
            .map(|x| {
                let (out, clipped) = self.forward_one(&x);
                clips.push(clipped);
                out
            })
            .collect();
        self.clip_batch_cache = clips;
        self.input_batch_cache = inputs.clone();
        self.output_batch_cache = outputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let n = self
            .input_batch_cache
            .shape
            .first()
            .copied()
            .expect("Linear: missing batch forward cache");
        assert!(grads.shape.len() >= 2, "Linear: expected batched gradients");
        assert_eq!(grads.shape[0], n, "Linear: gradient batch size mismatch");
        assert_eq!(
            grads.shape[1..].iter().product::<usize>(),
            self.out_features,
            "Linear: gradient feature shape mismatch"
        );
        assert_eq!(
            grads.data.len(),
            n * self.out_features,
            "Linear: gradient buffer mismatch"
        );
        assert_eq!(
            self.clip_batch_cache.len(),
            n,
            "Linear: missing batch saturation cache"
        );
        let inputs = std::mem::take(&mut self.input_batch_cache);
        let outputs = std::mem::take(&mut self.output_batch_cache);
        let clips = std::mem::take(&mut self.clip_batch_cache);
        grads
            .iter()
            .zip(inputs.iter().zip(outputs.iter()))
            .zip(clips.iter())
            .map(|((g, (inp, out)), clipped)| self.backward_one(&g, &inp, &out, clipped))
            .collect()
    }

    fn backward_batch_discard_input_grad(&mut self, grads: &Tensor) {
        let n = self
            .input_batch_cache
            .shape
            .first()
            .copied()
            .expect("Linear: missing batch forward cache");
        assert!(grads.shape.len() >= 2, "Linear: expected batched gradients");
        assert_eq!(grads.shape[0], n, "Linear: gradient batch size mismatch");
        assert_eq!(
            grads.shape[1..].iter().product::<usize>(),
            self.out_features,
            "Linear: gradient feature shape mismatch"
        );
        assert_eq!(
            grads.data.len(),
            n * self.out_features,
            "Linear: gradient buffer mismatch"
        );
        assert_eq!(
            self.clip_batch_cache.len(),
            n,
            "Linear: missing batch saturation cache"
        );

        let inputs = std::mem::take(&mut self.input_batch_cache);
        let outputs = std::mem::take(&mut self.output_batch_cache);
        let clips = std::mem::take(&mut self.clip_batch_cache);
        for ((g, (inp, out)), clipped) in grads
            .iter()
            .zip(inputs.iter().zip(outputs.iter()))
            .zip(clips.iter())
        {
            self.backward_one_discard_input_grad(&g, &inp, &out, clipped);
        }
    }

    fn update(&mut self, lr: u32) {
        if self.grad_samples == 0 {
            return;
        }
        apply_updates_mean(
            &mut self.weights.data,
            &self.grad_w,
            &mut self.vel_w.data,
            lr,
            self.momentum_shift,
            self.grad_samples,
        );
        apply_updates_mean(
            &mut self.biases.data,
            &self.grad_b,
            &mut self.vel_b.data,
            lr,
            self.momentum_shift,
            self.grad_samples,
        );
    }

    fn zero_grad(&mut self) {
        self.grad_samples = 0;
        self.grad_w.data.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.data.iter_mut().for_each(|g| g.v = 0);
    }
}

#[cfg(test)]
mod backward_accumulator_tests {
    use super::*;
    use crate::dyadic::mul;
    use crate::rng::{rng_next, seed_rng};

    fn d(v: i32, s: u32) -> Dyadic {
        Dyadic::new(v, s)
    }

    #[test]
    fn equal_and_mixed_scale_accumulation_match_reference_and_rng() {
        // Include saturation, zeros, extreme shift distances and
        // rescalings that consume stochastic RNG draws.
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -8192,
            -127,
            -1,
            0,
            1,
            127,
            8192,
            i32::MAX - 1,
            i32::MAX,
        ];
        let shifts = [0, 1, 7, 14, 31, 64, 128];
        for &sa in &shifts {
            for &sb in &shifts {
                for &av in &values {
                    for &bv in &values {
                        let a = d(av, sa);
                        let b = d(bv, sb);
                        seed_rng(0x2345_6789);
                        let expected = add(a, b);
                        let expected_next = rng_next();
                        seed_rng(0x2345_6789);
                        let actual = add_backward_accumulator(a, b);
                        let actual_next = rng_next();
                        assert_eq!(actual, expected, "a={a:?} b={b:?}");
                        assert_eq!(
                            actual_next, expected_next,
                            "RNG diverged for a={a:?} b={b:?}"
                        );
                    }
                }
            }
        }
    }

    // Frozen reference (pre-optimization), intentionally uses only add().
    fn reference_backward_one(
        layer: &mut Linear,
        grad_output: &TensorView,
        input: &TensorView,
        output: &TensorView,
        clipped: &[bool],
    ) -> Tensor {
        let g_s = grad_output.data.first().map_or(0, |g| g.s);
        let mut gi = vec![d(0, g_s); layer.in_features];
        for (j, &is_clipped) in clipped.iter().enumerate() {
            let gr = if is_clipped {
                d(0, grad_output.data[j].s)
            } else {
                ste_requantize(
                    grad_output.data[j],
                    output.data[j].v,
                    layer.q_min,
                    layer.q_max,
                )
            };
            let gj = d(gr.v.clamp(-layer.grad_clip, layer.grad_clip), gr.s);
            for (i, gi_value) in gi.iter_mut().enumerate() {
                let idx = j * layer.in_features + i;
                layer.grad_w.data[idx] = add(
                    layer.grad_w.data[idx],
                    mul(gj, input.data[i], layer.quant_shift),
                );
                *gi_value = add(
                    *gi_value,
                    mul(gj, layer.weights.data[idx], layer.quant_shift),
                );
            }
            layer.grad_b.data[j] = add(layer.grad_b.data[j], gj);
        }
        layer.grad_samples += 1;
        Tensor::from_vec(gi, vec![layer.in_features])
    }

    #[test]
    fn full_backward_matches_frozen_reference_and_rng_state() {
        for &samples in &[1usize, 7, 32] {
            for &grad_shift in &[0u32, 7, 10] {
                let mut fast = Linear::new(5, 3, 7, 7, 31).with_grad_clip(8192);
                let mut reference = Linear::new(5, 3, 7, 7, 31).with_grad_clip(8192);
                // Explicit parameters: independent constructors cannot bias
                // the two fixed-seed arithmetic executions.
                for i in 0..15 {
                    let weight = d((i as i32 * 19 - 132) * 3, 7);
                    fast.weights.data[i] = weight;
                    reference.weights.data[i] = weight;
                }
                let mut inputs = Vec::new();
                let mut outputs = Vec::new();
                let mut clips = Vec::new();
                let mut grads = Vec::new();
                for sample in 0..samples {
                    let data = (0..5)
                        .map(|i| {
                            d(
                                ((i + sample as i32) * 17 % 223) - 111,
                                if i % 2 == 0 { 7 } else { 10 },
                            )
                        })
                        .collect();
                    inputs.push(Tensor::from_vec(data, vec![5]));
                    outputs.push(Tensor::from_vec(
                        (0..3).map(|j| d(j * 11 - 7, 7)).collect(),
                        vec![3],
                    ));
                    clips.push([sample % 3 == 0, false, sample % 5 == 0]);
                    grads.push(Tensor::from_vec(
                        (0..3)
                            .map(|j| d(sample as i32 * 103 + j * 1001 - 1024, grad_shift))
                            .collect(),
                        vec![3],
                    ));
                }
                seed_rng(0x17a2_b3c4);
                let got: Vec<Tensor> = (0..samples)
                    .map(|n| {
                        fast.backward_one(
                            &grads[n].view(),
                            &inputs[n].view(),
                            &outputs[n].view(),
                            &clips[n],
                        )
                    })
                    .collect();
                let next_fast = rng_next();
                seed_rng(0x17a2_b3c4);
                let expected: Vec<Tensor> = (0..samples)
                    .map(|n| {
                        reference_backward_one(
                            &mut reference,
                            &grads[n].view(),
                            &inputs[n].view(),
                            &outputs[n].view(),
                            &clips[n],
                        )
                    })
                    .collect();
                let next_reference = rng_next();
                assert_eq!(
                    got.iter().map(|t| &t.data).collect::<Vec<_>>(),
                    expected.iter().map(|t| &t.data).collect::<Vec<_>>()
                );
                assert_eq!(fast.grad_w.data, reference.grad_w.data);
                assert_eq!(fast.grad_b.data, reference.grad_b.data);
                assert_eq!(fast.grad_samples, reference.grad_samples);
                assert_eq!(next_fast, next_reference, "stochastic RNG diverged");
            }
        }
    }
    /// Native-architecture diagnostic; compares the identical frozen reference
    /// and scoped RNG path on a realistic dense input layer. Not a Pi 5 result
    /// unless invoked on an actual Pi 5 under controlled conditions.
    #[test]
    #[ignore = "run with cargo test --release --lib benchmark_scoped_rng_vs_tls -- --ignored --nocapture"]
    fn benchmark_scoped_rng_vs_tls() {
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 64;
        let mut optimized = Linear::new(784, 128, 7, 7, 31).with_grad_clip(8192);
        let mut baseline = Linear::new(784, 128, 7, 7, 31).with_grad_clip(8192);
        baseline.weights = optimized.weights.clone();
        baseline.biases = optimized.biases.clone();
        let inputs = Tensor::from_vec(
            (0..784)
                .map(|i| d(if i % 3 == 0 { 0 } else { i % 255 - 127 }, 7))
                .collect(),
            vec![784],
        );
        let gradients = Tensor::from_vec((0..128).map(|j| d(j % 17 - 8, 7)).collect(), vec![128]);
        let outputs = Tensor::from_vec(vec![d(0, 7); 128], vec![128]);
        let clipped = vec![false; 128];

        seed_rng(0x1234_5678);
        let before = Instant::now();
        let mut baseline_last = None;
        for _ in 0..SAMPLES {
            baseline_last = Some(black_box(reference_backward_one(
                &mut baseline,
                &gradients.view(),
                &inputs.view(),
                &outputs.view(),
                &clipped,
            )));
        }
        let baseline_time = before.elapsed();
        let next_baseline = rng_next();

        seed_rng(0x1234_5678);
        let before = Instant::now();
        let mut optimized_last = None;
        for _ in 0..SAMPLES {
            optimized_last = Some(black_box(optimized.backward_one(
                &gradients.view(),
                &inputs.view(),
                &outputs.view(),
                &clipped,
            )));
        }
        let optimized_time = before.elapsed();
        let next_optimized = rng_next();

        assert_eq!(baseline_last.unwrap().data, optimized_last.unwrap().data);
        assert_eq!(baseline.grad_w.data, optimized.grad_w.data);
        assert_eq!(baseline.grad_b.data, optimized.grad_b.data);
        assert_eq!(next_baseline, next_optimized);
        println!(
            "Linear 784x128 backward, {SAMPLES} samples: legacy_TLS={:?}, scoped_RNG={:?}, ratio={:.3}x (native host diagnostic, not Pi 5 proof)",
            baseline_time,
            optimized_time,
            baseline_time.as_secs_f64() / optimized_time.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod forward_scoped_rng_tests {
    use super::*;
    use crate::dyadic::mul;
    use crate::rng::{rng_next, seed_rng};

    // Frozen pre-scoping forward path: uses the public mul() with TLS state
    // acquisition for every product, in the exact original traversal order.
    fn reference_forward_one(layer: &Linear, input: &TensorView) -> (Tensor, Vec<bool>) {
        assert_eq!(input.data.len(), layer.in_features);
        let mut out = Vec::with_capacity(layer.out_features);
        let mut clipped = Vec::with_capacity(layer.out_features);
        for j in 0..layer.out_features {
            let mut acc = layer.biases.data[j];
            for i in 0..layer.in_features {
                acc = add(acc, mul(layer.w(j, i), input.data[i], layer.quant_shift));
            }
            let (y, was_clipped) = requantize(acc, acc.s, layer.q_min, layer.q_max);
            out.push(y);
            clipped.push(was_clipped);
        }
        (Tensor::from_vec(out, vec![layer.out_features]), clipped)
    }

    #[test]
    fn forward_one_matches_frozen_reference_outputs_clips_and_rng() {
        // Exercise zero products, saturation, negative mantissas, mixed
        // exponents, q=0/no draws, and q >= 64/rare-event draw paths.
        for &(n_in, n_out) in &[(1usize, 1usize), (5, 3), (64, 32)] {
            for &q in &[0u32, 7, 31, 64, 128] {
                for &bits in &[4u32, 8, 31] {
                    let mut layer = Linear::new(n_in, n_out, 7, q, bits);
                    for (idx, weight) in layer.weights.data.iter_mut().enumerate() {
                        let mantissa = match idx % 11 {
                            0 => 0,
                            1 => i32::MAX,
                            2 => i32::MIN,
                            _ => ((idx * 17 % 251) as i32) - 125,
                        };
                        *weight = Dyadic::new(mantissa, q + (idx % 3) as u32);
                    }
                    for (j, bias) in layer.biases.data.iter_mut().enumerate() {
                        *bias = Dyadic::new(
                            if j % 3 == 0 {
                                i32::MAX
                            } else {
                                (j * 5) as i32 - 20
                            },
                            q + (j % 2) as u32,
                        );
                    }
                    let inputs: Vec<Tensor> = (0..17usize)
                        .map(|sample| {
                            Tensor::from_vec(
                                (0..n_in)
                                    .map(|i| {
                                        Dyadic::new(
                                            if (i + sample) % 7 == 0 {
                                                0
                                            } else {
                                                ((i * 43 + sample * 7) % 255) as i32 - 127
                                            },
                                            q + ((i + sample) % 2) as u32,
                                        )
                                    })
                                    .collect(),
                                vec![n_in],
                            )
                        })
                        .collect();
                    let seed = 0x1357_2468_9abc_def0;
                    seed_rng(seed);
                    let fast: Vec<_> = inputs
                        .iter()
                        .map(|x| layer.forward_one(&x.view()))
                        .collect();
                    let fast_next = rng_next();

                    seed_rng(seed);
                    let reference: Vec<_> = inputs
                        .iter()
                        .map(|x| reference_forward_one(&layer, &x.view()))
                        .collect();
                    let reference_next = rng_next();

                    for (i, ((out, clips), (expected, expected_clips))) in
                        fast.iter().zip(reference.iter()).enumerate()
                    {
                        assert_eq!(
                            out.data, expected.data,
                            "forward output sample={i} dims={n_in}x{n_out} q={q} bits={bits}"
                        );
                        assert_eq!(
                            clips, expected_clips,
                            "clipping sample={i} dims={n_in}x{n_out} q={q} bits={bits}"
                        );
                    }
                    assert_eq!(
                        fast_next, reference_next,
                        "forward RNG dims={n_in}x{n_out} q={q} bits={bits}"
                    );

                    // Public batch entry point must preserve both sample order
                    // and per-sample clipping caches, including next RNG state.
                    let batched = Tensor::from_vec(
                        inputs.iter().flat_map(|x| x.data.iter().copied()).collect(),
                        vec![inputs.len(), n_in],
                    );
                    seed_rng(seed);
                    let result = layer.forward_batch(&batched);
                    let batch_next = rng_next();
                    assert_eq!(result.shape, vec![inputs.len(), n_out]);
                    for (i, (expected, clips)) in reference.iter().enumerate() {
                        assert_eq!(
                            &result.data[i * n_out..(i + 1) * n_out],
                            &expected.data,
                            "batch output sample={i} q={q} bits={bits}",
                        );
                        assert_eq!(&layer.clip_batch_cache[i], clips);
                    }
                    assert_eq!(
                        batch_next, reference_next,
                        "batch RNG dims={n_in}x{n_out} q={q} bits={bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_single_entry_point_keeps_output_and_cache() {
        let mut layer = Linear::new(5, 3, 7, 7, 8);
        let input = Tensor::from_vec(
            vec![
                Dyadic::new(0, 7),
                Dyadic::new(-127, 7),
                Dyadic::new(127, 7),
                Dyadic::new(1, 7),
                Dyadic::new(-2, 7),
            ],
            vec![5],
        );
        seed_rng(127);
        let (expected, clips) = reference_forward_one(&layer, &input.view());
        let next_reference = rng_next();

        seed_rng(127);
        let result = layer.forward(&input.view());
        let next_fast = rng_next();
        assert_eq!(result.data, expected.data);
        assert_eq!(layer.input_cache.data, input.data);
        assert_eq!(layer.output_cache.data, expected.data);
        assert_eq!(layer.clip_cache, clips);
        assert_eq!(next_fast, next_reference);
    }

    /// Host-only, identical-binary diagnostic of the original TLS path and
    /// optimized path. Do not infer physical Pi speedup from ARM64 CI runners.
    #[test]
    #[ignore = "run with cargo test --release --lib benchmark_forward_scoped_rng_vs_tls -- --ignored --nocapture"]
    fn benchmark_forward_scoped_rng_vs_tls() {
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 64;
        let mut layer = Linear::new(784, 128, 7, 7, 31);
        let inputs = Tensor::from_vec(
            (0..784)
                .map(|i| Dyadic::new(if i % 3 == 0 { 0 } else { (i % 255) - 127 }, 7))
                .collect(),
            vec![784],
        );
        for (i, bias) in layer.biases.data.iter_mut().enumerate() {
            *bias = Dyadic::new((i % 13) as i32 - 6, 7);
        }

        seed_rng(0x1234_5678);
        let before = Instant::now();
        let mut baseline_last = None;
        for _ in 0..SAMPLES {
            baseline_last = Some(black_box(reference_forward_one(
                black_box(&layer),
                black_box(&inputs.view()),
            )));
        }
        let baseline_time = before.elapsed();
        let next_baseline = rng_next();

        seed_rng(0x1234_5678);
        let before = Instant::now();
        let mut optimized_last = None;
        for _ in 0..SAMPLES {
            optimized_last = Some(black_box(layer.forward_one(black_box(&inputs.view()))));
        }
        let optimized_time = before.elapsed();
        let next_optimized = rng_next();

        let (baseline_out, baseline_clips) = baseline_last.unwrap();
        let (optimized_out, optimized_clips) = optimized_last.unwrap();
        assert_eq!(baseline_out.data, optimized_out.data);
        assert_eq!(baseline_clips, optimized_clips);
        assert_eq!(next_baseline, next_optimized);
        println!(
            "Linear 784x128 forward, {SAMPLES} samples: legacy_TLS={:?}, scoped_RNG={:?}, ratio={:.3}x (native host diagnostic, not Pi 5 proof)",
            baseline_time,
            optimized_time,
            baseline_time.as_secs_f64() / optimized_time.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod discarded_input_gradient_tests {
    use super::*;
    use crate::rng::{rng_next, seed_rng};

    fn d(v: i32, s: u32) -> Dyadic {
        Dyadic::new(v, s)
    }

    #[test]
    fn discarded_first_layer_gradient_preserves_parameters_and_rng_after_update() {
        // q >= 64 exercises the product-dependent RNG fallback, including
        // zero products; q=0 tests zero draws, and q=63 the one-draw boundary.
        for &q in &[0u32, 1, 7, 31, 63, 64, 128] {
            for &(n_in, n_out) in &[(5usize, 3usize), (16, 8)] {
                for &samples in &[1usize, 7] {
                    for &bits in &[4u32, 31] {
                        let mut full = Linear::new(n_in, n_out, 7, q, bits)
                            .with_grad_clip(8192)
                            .with_momentum(1);
                        for (idx, w) in full.weights.data.iter_mut().enumerate() {
                            *w = d(
                                match idx % 17 {
                                    0 => i32::MAX,
                                    1 => i32::MIN,
                                    2 => 0,
                                    _ => (idx % 253) as i32 - 126,
                                },
                                q + (idx % 2) as u32,
                            );
                        }
                        for (j, b) in full.biases.data.iter_mut().enumerate() {
                            *b = d(if j % 4 == 0 { i32::MAX } else { 0 }, q);
                        }
                        let mut optimized = Linear::new(n_in, n_out, 7, q, bits)
                            .with_grad_clip(8192)
                            .with_momentum(1);
                        optimized.weights = full.weights.clone();
                        optimized.biases = full.biases.clone();

                        let inputs = Tensor::from_vec(
                            (0..samples * n_in)
                                .map(|i| {
                                    d(
                                        if i % 7 == 0 {
                                            0
                                        } else {
                                            (i % 201) as i32 - 100
                                        },
                                        q + (i % 2) as u32,
                                    )
                                })
                                .collect(),
                            vec![samples, n_in],
                        );
                        let grads = Tensor::from_vec(
                            (0..samples * n_out)
                                .map(|i| {
                                    d(
                                        if i % 5 == 0 {
                                            i32::MIN
                                        } else {
                                            (i % 47) as i32 - 23
                                        },
                                        q,
                                    )
                                })
                                .collect(),
                            vec![samples, n_out],
                        );

                        let seed = 0x1bad_b002_1234_5678;
                        seed_rng(seed);
                        let full_outputs = full.forward_batch(&inputs);
                        let _unused_grad = full.backward_batch(&grads);
                        let full_grads = (full.grad_w.data.clone(), full.grad_b.data.clone());
                        let full_samples = full.grad_samples;
                        full.update(7);
                        let full_next_rng = rng_next();

                        seed_rng(seed);
                        let fast_outputs = optimized.forward_batch(&inputs);
                        optimized.backward_batch_discard_input_grad(&grads);
                        let fast_grads =
                            (optimized.grad_w.data.clone(), optimized.grad_b.data.clone());
                        let fast_samples = optimized.grad_samples;
                        optimized.update(7);
                        let fast_next_rng = rng_next();

                        let label =
                            format!("q={q} size={n_in}x{n_out} samples={samples} bits={bits}");
                        assert_eq!(full_outputs.data, fast_outputs.data, "{label}: forward");
                        assert_eq!(full_grads, fast_grads, "{label}: parameter gradients");
                        assert_eq!(full_samples, fast_samples, "{label}: sample count");
                        assert_eq!(
                            full.weights.data, optimized.weights.data,
                            "{label}: updated weights"
                        );
                        assert_eq!(
                            full.biases.data, optimized.biases.data,
                            "{label}: updated biases"
                        );
                        assert_eq!(
                            full.vel_w.data, optimized.vel_w.data,
                            "{label}: weight momentum"
                        );
                        assert_eq!(
                            full.vel_b.data, optimized.vel_b.data,
                            "{label}: bias momentum"
                        );
                        assert_eq!(full_next_rng, fast_next_rng, "{label}: RNG stream");
                    }
                }
            }
        }
    }

    fn model(seed: u64) -> crate::nn::Sequential {
        seed_rng(seed);
        let mut seq = crate::nn::Sequential::new();
        seq.add(Linear::new(6, 4, 7, 7, 8).with_momentum(1));
        seq.add(crate::nn::ReLU::new());
        seq.add(Linear::new(4, 3, 7, 7, 8).with_momentum(1));
        seq
    }

    #[test]
    fn sequential_training_and_timed_path_preserve_predictions_and_rng_across_steps() {
        let mut full = model(1337);
        let mut optimized = model(1337);
        let input = Tensor::from_vec((0..42).map(|i| d(i % 23 - 11, 7)).collect(), vec![7, 6]);
        let grads = Tensor::from_vec((0..21).map(|i| d(i % 19 - 9, 7)).collect(), vec![7, 3]);
        seed_rng(0x1234_5678);
        let mut expected = Vec::new();
        for _ in 0..4 {
            full.zero_grad();
            expected.push(full.forward_batch(&input).data);
            let _ = full.backward_batch(&grads);
            full.update(7);
        }
        let expected_after = full.forward_batch(&input).data;
        let full_rng = rng_next();

        seed_rng(0x1234_5678);
        let mut actual = Vec::new();
        for step in 0..4 {
            optimized.zero_grad();
            actual.push(optimized.forward_batch(&input).data);
            if step % 2 == 0 {
                let mut layer_ns = [0u128; 3];
                optimized.backward_batch_timed_discard_input_grad(&grads, &mut layer_ns);
            } else {
                optimized.backward_batch_discard_input_grad(&grads);
            }
            optimized.update(7);
        }
        let actual_after = optimized.forward_batch(&input).data;
        let fast_rng = rng_next();
        assert_eq!(expected, actual, "predictions during training");
        assert_eq!(expected_after, actual_after, "predictions after updates");
        assert_eq!(full_rng, fast_rng, "RNG after multiple updates");
    }

    #[test]
    fn public_backward_still_produces_full_input_gradients() {
        let mut linear = Linear::new(6, 4, 7, 7, 8);
        let inputs = Tensor::from_vec(vec![d(2, 7); 12], vec![2, 6]);
        let grads = Tensor::from_vec(vec![d(3, 7); 8], vec![2, 4]);
        linear.forward_batch(&inputs);
        let input_grads = linear.backward_batch(&grads);
        assert_eq!(input_grads.shape, vec![2, 6]);
        assert_eq!(input_grads.data.len(), 12);
    }

    #[test]
    #[ignore = "run with cargo test --release --lib benchmark_discard_unused_input_grad -- --ignored --nocapture"]
    fn benchmark_discard_unused_input_grad() {
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 64;
        let mut full = Linear::new(784, 128, 7, 7, 31).with_grad_clip(8192);
        let mut optimized = Linear::new(784, 128, 7, 7, 31).with_grad_clip(8192);
        optimized.weights = full.weights.clone();
        optimized.biases = full.biases.clone();
        let input = Tensor::from_vec(
            (0..784)
                .map(|i| d(if i % 3 == 0 { 0 } else { i % 255 - 127 }, 7))
                .collect(),
            vec![784],
        );
        let gradient = Tensor::from_vec((0..128).map(|j| d(j % 17 - 8, 7)).collect(), vec![128]);
        let output = Tensor::from_vec(vec![d(0, 7); 128], vec![128]);
        let clipped = vec![false; 128];
        seed_rng(0x1234_5678);
        let start = Instant::now();
        for _ in 0..SAMPLES {
            black_box(full.backward_one(&gradient.view(), &input.view(), &output.view(), &clipped));
        }
        let full_time = start.elapsed();
        let next_full = rng_next();

        seed_rng(0x1234_5678);
        let start = Instant::now();
        for _ in 0..SAMPLES {
            optimized.backward_one_discard_input_grad(
                black_box(&gradient.view()),
                black_box(&input.view()),
                black_box(&output.view()),
                black_box(&clipped),
            );
        }
        let fast_time = start.elapsed();
        let next_fast = rng_next();
        assert_eq!(full.grad_w.data, optimized.grad_w.data);
        assert_eq!(full.grad_b.data, optimized.grad_b.data);
        assert_eq!(full.grad_samples, optimized.grad_samples);
        assert_eq!(next_full, next_fast);
        println!(
            "Linear 784x128 backward, {SAMPLES} samples: full_input_gradient={:?}, discard_unused={:?}, ratio={:.3}x (native-host diagnostic, NOT physical Pi 5 performance)",
            full_time,
            fast_time,
            full_time.as_secs_f64() / fast_time.as_secs_f64(),
        );
    }
}
