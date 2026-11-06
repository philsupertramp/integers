//! Independent, dataset-free numerical oracles for issue #14.
//! The reference computations decode (v,s) directly to FP64 and do not call
//! production dyadic operators. All stochastic paths use fixed seeds.
use integers::nn::{Conv2D, Dropout, Flatten, Linear, MaxPool2D, Module, ReLU, Softmax};
use integers::rng::seed_rng;
use integers::{Dyadic, Tensor, cross_entropy_grad, mse_grad};

fn d(v: i32, s: u32) -> Dyadic {
    Dyadic::new(v, s)
}
fn t(v: &[i32], s: u32, shape: &[usize]) -> Tensor {
    Tensor::from_vec(v.iter().map(|&x| d(x, s)).collect(), shape.to_vec())
}
fn decoded(x: Dyadic) -> f64 {
    (x.v as f64) * 2f64.powi(-(x.s as i32))
}
fn approx(actual: Dyadic, expected: f64, steps: f64) {
    // One LSB per rounded multiply/accumulate, with a small FP64 margin.
    let bound = steps * 2f64.powi(-(actual.s as i32)) + 1e-12;
    assert!(
        (decoded(actual) - expected).abs() <= bound,
        "got {:?} => {}, expected {expected}, bound {bound}",
        actual,
        decoded(actual)
    );
}
fn set_linear(layer: &mut Linear) {
    layer.weights.data = vec![d(4, 2), d(-2, 2), d(2, 2), d(4, 2)];
    layer.biases.data = vec![d(1, 2), d(2, 2)];
}
fn set_conv(layer: &mut Conv2D) {
    layer.kernels.data = vec![d(1, 0), d(0, 0), d(0, 0), d(-1, 0)];
    layer.biases.data = vec![d(1, 0)];
}
fn fp_linear(weights: &[Dyadic], biases: &[Dyadic], x: &[Dyadic], n_in: usize) -> Vec<f64> {
    biases
        .iter()
        .enumerate()
        .map(|(j, b)| {
            decoded(*b)
                + (0..n_in)
                    .map(|i| decoded(weights[j * n_in + i]) * decoded(x[i]))
                    .sum::<f64>()
        })
        .collect()
}
fn fp_conv(x: &[Dyadic], kernel: &[Dyadic], bias: Dyadic) -> Vec<f64> {
    (0..2)
        .flat_map(|h| {
            (0..2).map(move |w| {
                decoded(bias)
                    + (0..2)
                        .flat_map(|kh| {
                            (0..2).map(move |kw| {
                                decoded(x[(h + kh) * 3 + w + kw]) * decoded(kernel[kh * 2 + kw])
                            })
                        })
                        .sum::<f64>()
            })
        })
        .collect()
}
fn panics(f: impl FnOnce()) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
}

#[test]
fn linear_forward_backward_fp64_and_finite_differences() {
    seed_rng(14);
    let mut layer = Linear::new(2, 2, 2, 2, 16);
    set_linear(&mut layer);
    let x = t(&[8, -4], 2, &[2]);
    let y = layer.forward(&x.view());
    let oracle = fp_linear(&layer.weights.data, &layer.biases.data, &x.data, 2);
    for (got, reference) in y.data.iter().zip(&oracle) {
        approx(*got, *reference, 0.0); // all products are exact on these dyadic fixtures
    }
    let g = t(&[4, -4], 2, &[2]);
    let gi = layer.backward(&g.view());
    let expected_input = [
        decoded(g.data[0]) * decoded(layer.weights.data[0])
            + decoded(g.data[1]) * decoded(layer.weights.data[2]),
        decoded(g.data[0]) * decoded(layer.weights.data[1])
            + decoded(g.data[1]) * decoded(layer.weights.data[3]),
    ];
    for (a, e) in gi.data.iter().zip(expected_input) {
        approx(*a, e, 0.0);
    }
    // Independently differentiate FP64 dot(g, W*x+b) by central differences.
    let h = 1e-5;
    let x0 = x.data.iter().map(|&v| decoded(v)).collect::<Vec<_>>();
    let wf = layer
        .weights
        .data
        .iter()
        .map(|&v| decoded(v))
        .collect::<Vec<_>>();
    let gf = g.data.iter().map(|&v| decoded(v)).collect::<Vec<_>>();
    let bf = layer
        .biases
        .data
        .iter()
        .map(|&v| decoded(v))
        .collect::<Vec<_>>();
    let objective = |xv: &[f64], wv: &[f64], bv: &[f64]| -> f64 {
        (0..2)
            .map(|j| gf[j] * (bv[j] + (0..2).map(|i| wv[j * 2 + i] * xv[i]).sum::<f64>()))
            .sum()
    };
    for i in 0..2 {
        let mut plus = x0.clone();
        plus[i] += h;
        let mut minus = x0.clone();
        minus[i] -= h;
        let fd = (objective(&plus, &wf, &bf) - objective(&minus, &wf, &bf)) / (2.0 * h);
        assert!((decoded(gi.data[i]) - fd).abs() < 1e-8);
    }
    // A no-momentum update exposes accumulated weight and bias gradients.
    layer.update(0);
    let wg = [8, -4, -8, 4];
    for (i, &dv) in wg.iter().enumerate() {
        assert_eq!(layer.weights.data[i], d([4, -2, 2, 4][i] - dv, 2));
    }
    assert_eq!(layer.biases.data, vec![d(-3, 2), d(6, 2)]);
}

#[test]
fn linear_accumulation_zero_grad_and_momentum() {
    seed_rng(16);
    let mut layer = Linear::new(1, 1, 2, 2, 32);
    layer.weights.data[0] = d(4, 2);
    layer.biases.data[0] = d(0, 2);
    let x = t(&[8], 2, &[1]);
    let g = t(&[8], 2, &[1]);
    layer.zero_grad();
    layer.forward(&x.view());
    layer.backward(&g.view());
    layer.backward(&g.view()); // two samples, same cached input
    layer.update(0);
    // Two identical backward samples have mean grad_w = 2*2 = 4,
    // rather than the historical SUM gradient of 8.
    assert_eq!(layer.weights.data[0], d(-12, 2));
    assert_eq!(layer.biases.data[0], d(-8, 2));
    layer.zero_grad();
    layer.update(0);
    assert_eq!(layer.weights.data[0], d(-12, 2));

    let mut momentum = Linear::new(1, 1, 2, 2, 32).with_momentum(1);
    momentum.weights.data[0] = d(4, 2);
    momentum.biases.data[0] = d(0, 2);
    momentum.forward(&t(&[4], 2, &[1]).view());
    momentum.backward(&t(&[8], 2, &[1]).view());
    momentum.update(0);
    assert_eq!(momentum.weights.data[0], d(-4, 2)); // velocity 2.0
    momentum.zero_grad();
    momentum.update(0);
    assert_eq!(momentum.weights.data[0], d(-4, 2)); // no accumulated examples: no update
}

#[test]
fn linear_batch_equivalence_and_cache_isolation() {
    seed_rng(17);
    let mut batched = Linear::new(2, 2, 2, 2, 16);
    let mut single = Linear::new(2, 2, 2, 2, 16);
    set_linear(&mut batched);
    set_linear(&mut single);
    let x = t(&[8, -4, 4, 4], 2, &[2, 2]);
    let grads = t(&[4, -4, -4, 4], 2, &[2, 2]);
    let y = batched.forward_batch(&x);
    let gi = batched.backward_batch(&grads);
    let mut ys = vec![];
    let mut gs = vec![];
    for (xi, gg) in x.iter().zip(grads.iter()) {
        ys.extend(single.forward(&xi).data);
        gs.extend(single.backward(&gg).data);
    }
    assert_eq!(y.data, ys);
    assert_eq!(gi.data, gs);
    batched.update(0);
    single.update(0);
    assert_eq!(batched.weights.data, single.weights.data);
    assert_eq!(batched.biases.data, single.biases.data);
    // Repeated forward replaces prior cache, rather than using stale sample.
    let a = t(&[4, 0], 2, &[2]);
    let b = t(&[0, 4], 2, &[2]);
    let mut isolated = Linear::new(2, 2, 2, 2, 16);
    set_linear(&mut isolated);
    isolated.forward(&a.view());
    isolated.forward(&b.view());
    isolated.backward(&t(&[4, 0], 2, &[2]).view());
    isolated.update(0);
    assert_eq!(isolated.weights.data[0], d(4, 2)); // x[0] was zero in b
    assert_eq!(isolated.weights.data[1], d(-6, 2));
}

#[test]
fn relu_boundaries_batch_and_new_forward_mask() {
    let mut relu = ReLU::new();
    let x = Tensor::from_vec(vec![d(-5, 1), d(0, 3), d(5, 4)], vec![3]);
    let g = Tensor::from_vec(vec![d(7, 2); 3], vec![3]);
    assert_eq!(
        relu.forward(&x.view()).data,
        vec![d(0, 1), d(0, 3), d(5, 4)]
    );
    assert_eq!(
        relu.backward(&g.view()).data,
        vec![d(0, 2), d(0, 2), d(7, 2)]
    );
    relu.forward(&t(&[2, -1, 0], 2, &[3]).view());
    assert_eq!(
        relu.backward(&g.view()).data,
        vec![d(7, 2), d(0, 2), d(0, 2)]
    );

    let xb = t(&[2, -1, 0, -2, 3, 4], 2, &[2, 3]);
    let gb = t(&[1, 2, 3, 4, 5, 6], 2, &[2, 3]);
    assert_eq!(
        relu.forward_batch(&xb).data,
        t(&[2, 0, 0, 0, 3, 4], 2, &[2, 3]).data
    );
    assert_eq!(
        relu.backward_batch(&gb).data,
        t(&[1, 0, 0, 0, 5, 6], 2, &[2, 3]).data
    );
}

#[test]
fn conv_forward_gradient_oracle_accumulation_and_update() {
    seed_rng(18);
    let mut conv = Conv2D::new(1, 1, 2, 2, 3, 3, 0, 0, 32);
    set_conv(&mut conv);
    let x = t(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 0, &[9]);
    let y = conv.forward(&x.view());
    let oracle = fp_conv(&x.data, &conv.kernels.data, conv.biases.data[0]);
    for (v, expected) in y.data.iter().zip(oracle) {
        approx(*v, expected, 0.0);
    }
    let g = t(&[1, 2, 3, 4], 0, &[4]);
    let gi = conv.backward(&g.view());
    // Independent transposed-convolution/index accumulation reference.
    let mut gi_ref = [0f64; 9];
    let mut gk_ref = [0f64; 4];
    for oh in 0..2 {
        for ow in 0..2 {
            for kh in 0..2 {
                for kw in 0..2 {
                    let op = oh * 2 + ow;
                    let inp = (oh + kh) * 3 + ow + kw;
                    let k = kh * 2 + kw;
                    gi_ref[inp] += decoded(g.data[op]) * decoded(conv.kernels.data[k]);
                    gk_ref[k] += decoded(g.data[op]) * decoded(x.data[inp]);
                }
            }
        }
    }
    for (v, e) in gi.data.iter().zip(gi_ref) {
        approx(*v, e, 0.0);
    }
    conv.update(0);
    for (k, &expected) in conv.kernels.data.iter().zip(&gk_ref) {
        approx(
            *k,
            [1.0, 0.0, 0.0, -1.0][gk_ref.iter().position(|&x| x == expected).unwrap()] - expected,
            0.0,
        );
    }
    assert_eq!(conv.biases.data[0], d(-9, 0));
    // zero_grad prevents re-applying the same accumulated derivative
    let after = conv.kernels.data.clone();
    conv.zero_grad();
    conv.update(0);
    assert_eq!(conv.kernels.data, after);
}

#[test]
fn conv_batch_matches_per_sample_and_cache() {
    seed_rng(19);
    let mut batch = Conv2D::new(1, 1, 2, 2, 3, 3, 0, 0, 32);
    let mut serial = Conv2D::new(1, 1, 2, 2, 3, 3, 0, 0, 32);
    set_conv(&mut batch);
    set_conv(&mut serial);
    let inputs = t(
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        0,
        &[2, 9],
    );
    let grads = t(&[1, 2, 3, 4, 4, 3, 2, 1], 0, &[2, 4]);
    let y = batch.forward_batch(&inputs);
    let dx = batch.backward_batch(&grads);
    let mut expected_y = vec![];
    let mut expected_dx = vec![];
    for (x, g) in inputs.iter().zip(grads.iter()) {
        expected_y.extend(serial.forward(&x).data);
        expected_dx.extend(serial.backward(&g).data);
    }
    assert_eq!(y.data, expected_y);
    assert_eq!(dx.data, expected_dx);
    batch.update(0);
    serial.update(0);
    assert_eq!(batch.kernels.data, serial.kernels.data);
    assert_eq!(batch.biases.data, serial.biases.data);
}

#[test]
fn pool_selection_ties_mixed_scales_and_gradient_routing() {
    let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
    let x = t(&[1, 4, 2, 3], 0, &[4]);
    assert_eq!(pool.forward(&x.view()).data, vec![d(4, 0)]);
    assert_eq!(
        pool.backward(&t(&[7], 0, &[1]).view()).data,
        t(&[0, 7, 0, 0], 0, &[4]).data
    );
    assert_eq!(
        pool.forward(&t(&[5, 5, 2, 3], 0, &[4]).view()).data,
        vec![d(5, 0)]
    );
    assert_eq!(
        pool.backward(&t(&[2], 0, &[1]).view()).data,
        t(&[2, 0, 0, 0], 0, &[4]).data
    );
    let mixed = Tensor::from_vec(vec![d(3, 1), d(2, 0), d(-8, 2), d(1, 0)], vec![4]);
    assert_eq!(pool.forward(&mixed.view()).data, vec![d(2, 0)]);
    assert_eq!(
        pool.backward(&t(&[4], 0, &[1]).view()).data,
        t(&[0, 4, 0, 0], 0, &[4]).data
    );

    let xb = t(&[1, 4, 2, 3, 5, 3, 2, 1], 0, &[2, 4]);
    let gb = t(&[7, 8], 0, &[2, 1]);
    assert_eq!(pool.forward_batch(&xb).data, t(&[4, 5], 0, &[2, 1]).data);
    assert_eq!(
        pool.backward_batch(&gb).data,
        t(&[0, 7, 0, 0, 8, 0, 0, 0], 0, &[2, 4]).data
    );
}

#[test]
fn flatten_is_value_and_shape_preserving() {
    let mut flat = Flatten;
    let x = t(&[1, -1, 3, 4], 2, &[2, 2]);
    assert_eq!(flat.forward(&x.view()).data, x.data);
    assert_eq!(flat.forward(&x.view()).shape, x.shape);
    assert_eq!(flat.backward(&x.view()).data, x.data);
    let batch = t(&[1, 2, 3, 4, 5, 6, 7, 8], 2, &[2, 2, 2]);
    assert_eq!(flat.forward_batch(&batch).data, batch.data);
    assert_eq!(flat.backward_batch(&batch).data, batch.data);
}

#[test]
fn loss_gradients_agree_with_independent_fp64_and_validate_lengths() {
    let out = vec![d(4, 2), d(-3, 2)];
    let target = vec![d(2, 2), d(1, 2)];
    for (got, (&o, &tgt)) in mse_grad(&out, &target).iter().zip(out.iter().zip(&target)) {
        approx(*got, decoded(o) - decoded(tgt), 0.0);
    }
    let probs = vec![d(96, 7), d(32, 7)];
    let onehot = vec![d(128, 7), d(0, 7)];
    for (got, (&p, &tgt)) in cross_entropy_grad(&probs, &onehot, 7)
        .iter()
        .zip(probs.iter().zip(&onehot))
    {
        approx(*got, decoded(p) - decoded(tgt), 0.0);
    }
    panics(|| {
        mse_grad(&out, &target[..1]);
    });
    panics(|| {
        cross_entropy_grad(&probs, &onehot[..1], 7);
    });
    panics(|| {
        mse_grad(&[d(4, 2)], &[d(2, 3)]);
    });
    panics(|| {
        cross_entropy_grad(&[d(4, 2)], &[d(2, 2)], 7);
    });
    assert!(mse_grad(&[], &[]).is_empty());
    assert!(cross_entropy_grad(&[], &[], 7).is_empty());
}

#[test]
fn softmax_reference_and_identity_ste_not_true_jacobian() {
    let mut softmax = Softmax::new(7);
    let input = t(&[-4, 0, 4], 2, &[3]);
    let output = softmax.forward(&input.view());
    let logits: Vec<f64> = input.data.iter().map(|&x| decoded(x)).collect();
    let es: Vec<f64> = logits.iter().map(|x| x.exp()).collect();
    let sum: f64 = es.iter().sum();
    for (actual, e) in output.data.iter().zip(es) {
        approx(*actual, e / sum, 0.5);
    }
    let g = t(&[16, -32, 16], 7, &[3]);
    assert_eq!(softmax.backward(&g.view()).data, g.data);
    // Contract is identity STE, not J_softmax^T * g.
    let probs: Vec<f64> = logits.iter().map(|x| x.exp() / sum).collect();
    let jac_g0 = probs[0]
        * (decoded(g.data[0])
            - probs
                .iter()
                .zip(&g.data)
                .map(|(p, g)| p * decoded(*g))
                .sum::<f64>());
    assert!((jac_g0 - decoded(g.data[0])).abs() > 0.01);
}

#[test]
fn dropout_seeded_mask_backward_and_eval() {
    let mut drop = Dropout::new(0.5);
    let x = t(&[128, 128, 128, 128, 128, 128, 128, 128], 7, &[8]);
    let grads = t(&[128, 128, 128, 128, 128, 128, 128, 128], 7, &[8]);
    seed_rng(0x14);
    let first = drop.forward(&x.view());
    let back = drop.backward(&grads.view());
    assert_eq!(first.data, back.data);
    assert!(first.data.iter().all(|v| v.v == 0 || v.v == 256));
    seed_rng(0x14);
    assert_eq!(drop.forward(&x.view()).data, first.data);
    drop.set_training(false);
    assert_eq!(drop.forward(&x.view()).data, x.data);
    assert_eq!(drop.backward(&grads.view()).data, grads.data);

    drop.set_training(true);
    seed_rng(0x140);
    let xb = t(&[128, 128, 128, 128, 128, 128, 128, 128], 7, &[2, 4]);
    let yb = drop.forward_batch(&xb);
    let gb = drop.backward_batch(&xb);
    assert_eq!(yb.data, gb.data);
    seed_rng(0x140);
    assert_eq!(drop.forward_batch(&xb).data, yb.data);
}

#[test]
fn shape_mismatches_must_not_silently_truncate() {
    let mut linear = Linear::new(2, 1, 2, 2, 16);
    panics(|| {
        linear.forward(&t(&[4], 2, &[1]).view());
    });
    linear.forward(&t(&[4, 4], 2, &[2]).view());
    panics(|| {
        linear.backward(&t(&[2, 2], 2, &[2]).view());
    });
    let xb = t(&[4, 4, 4, 4], 2, &[2, 2]);
    linear.forward_batch(&xb);
    panics(|| {
        linear.backward_batch(&t(&[2], 2, &[1, 1]));
    });
    let mut conv = Conv2D::new(1, 1, 2, 2, 3, 3, 0, 0, 16);
    panics(|| {
        conv.forward(&t(&[1, 2], 0, &[2]).view());
    });
    conv.forward(&t(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 0, &[9]).view());
    panics(|| {
        conv.backward(&t(&[1, 2], 0, &[2]).view());
    });
    conv.forward_batch(&t(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 0, &[1, 9]));
    panics(|| {
        conv.backward_batch(&t(&[1, 2, 3], 0, &[1, 3]));
    });
    let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
    pool.forward(&t(&[1, 2, 3, 4], 0, &[4]).view());
    panics(|| {
        pool.backward(&t(&[1, 2], 0, &[2]).view());
    });
    panics(|| {
        pool.forward(&t(&[1, 2, 3], 0, &[3]).view());
    });
    let mut relu = ReLU::new();
    relu.forward_batch(&t(&[1, 2], 0, &[1, 2]));
    panics(|| {
        relu.backward_batch(&t(&[1], 0, &[1, 1]));
    });
    panics(|| {
        relu.forward_batch(&Tensor::from_vec(vec![], vec![0, 2]));
    });
}

#[test]
fn saturation_masks_true_clipping_but_not_exact_boundary_values() {
    seed_rng(20);
    let mut linear = Linear::new(1, 1, 0, 0, 8);
    linear.weights.data[0] = d(2, 0);
    linear.biases.data[0] = d(0, 0);
    let clipped = linear.forward(&t(&[100], 0, &[1]).view());
    assert_eq!(clipped.data, vec![d(127, 0)]);
    assert_eq!(
        linear.backward(&t(&[7], 0, &[1]).view()).data,
        vec![d(0, 0)]
    );
    linear.update(0);
    assert_eq!(linear.weights.data[0], d(2, 0));
    assert_eq!(linear.biases.data[0], d(0, 0));
    linear.weights.data[0] = d(127, 0);
    assert_eq!(
        linear.forward(&t(&[1], 0, &[1]).view()).data,
        vec![d(127, 0)]
    );
    assert_eq!(
        linear.backward(&t(&[1], 0, &[1]).view()).data,
        vec![d(127, 0)]
    );

    let mut conv = Conv2D::new(1, 1, 1, 1, 1, 1, 0, 0, 8);
    conv.kernels.data[0] = d(2, 0);
    let batch = t(&[100, 1], 0, &[2, 1]);
    assert_eq!(conv.forward_batch(&batch).data, vec![d(127, 0), d(2, 0)]);
    let gi = conv.backward_batch(&t(&[7, 7], 0, &[2, 1]));
    assert_eq!(gi.data, vec![d(0, 0), d(14, 0)]);
    conv.update(0);
    // Only the second of two examples contributes. Mean gradient = 7/2;
    // stochastic SGD rounding yields an integer delta of 3 or 4.
    assert!(matches!(conv.kernels.data[0].v, -2 | -1));
    assert!(matches!(conv.biases.data[0].v, -4 | -3));
}

#[test]
fn pool_mixed_scale_negative_and_boundary_ties() {
    let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
    let x = Tensor::from_vec(vec![d(-2, 0), d(-3, 1), d(-1, 0), d(-8, 3)], vec![4]);
    // -1.0 ties at indices 2 and 3; first spatial index wins.
    assert_eq!(pool.forward(&x.view()).data, vec![d(-1, 0)]);
    assert_eq!(
        pool.backward(&t(&[4], 0, &[1]).view()).data,
        t(&[0, 0, 4, 0], 0, &[4]).data
    );
}

#[test]
fn independent_minibatch_accumulation_and_conv_momentum() {
    seed_rng(21);
    let mut linear = Linear::new(1, 1, 2, 2, 32);
    linear.weights.data[0] = d(4, 2);
    linear.biases.data[0] = d(0, 2);
    let input = t(&[4], 2, &[1, 1]);
    let grad = t(&[4], 2, &[1, 1]);
    for _ in 0..2 {
        linear.forward_batch(&input);
        assert_eq!(linear.backward_batch(&grad).data, vec![d(4, 2)]);
    }
    linear.update(0);
    // Four identical backward examples: mean, not sum.
    assert_eq!(linear.weights.data[0], d(0, 2));
    assert_eq!(linear.biases.data[0], d(-4, 2));
    linear.zero_grad();
    let w_before = linear.weights.data.clone();
    linear.update(0);
    assert_eq!(linear.weights.data, w_before);

    let mut conv = Conv2D::new(1, 1, 1, 1, 1, 1, 2, 2, 32).with_momentum(1);
    conv.kernels.data[0] = d(4, 2);
    conv.biases.data[0] = d(0, 2);
    conv.forward_batch(&t(&[4], 2, &[1, 1]));
    assert_eq!(
        conv.backward_batch(&t(&[8], 2, &[1, 1])).data,
        vec![d(8, 2)]
    );
    conv.update(0);
    assert_eq!(conv.kernels.data[0], d(-4, 2));
    assert_eq!(conv.biases.data[0], d(-8, 2));
    conv.zero_grad();
    conv.update(0);
    assert_eq!(conv.kernels.data[0], d(-4, 2));
    assert_eq!(conv.biases.data[0], d(-8, 2)); // no gradient samples: no step
}

#[test]
fn successive_batch_forwards_replace_linear_and_conv_caches() {
    seed_rng(22);
    let mut linear = Linear::new(1, 1, 0, 0, 32);
    linear.weights.data[0] = d(3, 0);
    linear.biases.data[0] = d(0, 0);
    linear.forward_batch(&t(&[5], 0, &[1, 1]));
    linear.forward_batch(&t(&[2], 0, &[1, 1]));
    linear.backward_batch(&t(&[4], 0, &[1, 1]));
    linear.update(0);
    assert_eq!(linear.weights.data[0], d(-5, 0)); // 3 - (2*4)

    let mut conv = Conv2D::new(1, 1, 1, 1, 1, 1, 0, 0, 32);
    conv.kernels.data[0] = d(3, 0);
    conv.biases.data[0] = d(0, 0);
    conv.forward_batch(&t(&[5], 0, &[1, 1]));
    conv.forward_batch(&t(&[2], 0, &[1, 1]));
    conv.backward_batch(&t(&[4], 0, &[1, 1]));
    conv.update(0);
    assert_eq!(conv.kernels.data[0], d(-5, 0));
}

#[test]
fn sequential_backward_layer_timing_preserves_values_and_rng() {
    use integers::nn::Sequential;
    use integers::rng::rng_next;
    fn run(profile: bool) -> (Vec<Dyadic>, Vec<Dyadic>, Vec<Dyadic>, u64, Vec<u128>) {
        seed_rng(0x4e52);
        let mut model = Sequential::new();
        model.add(Linear::new(3, 4, 7, 7, 31).with_grad_clip(8192));
        model.add(ReLU::new());
        model.add(Linear::new(4, 2, 7, 7, 31).with_grad_clip(8192));
        let x = t(&[16, -31, 42, 5, 64, -13, 127, -57, 80], 7, &[3, 3]);
        let grad = t(&[4, -7, -13, 8, 0, 34], 7, &[3, 2]);
        // Reseed after constructor draws, to compare exactly the same
        // stochastic rounding stream across timed vs original methods.
        seed_rng(0x8e67);
        model.zero_grad();
        let output = model.forward_batch(&x);
        let mut per_layer = vec![0u128; model.layers.len()];
        let dx = if profile {
            model.backward_batch_timed(&grad, &mut per_layer)
        } else {
            model.backward_batch(&grad)
        };
        model.update(7);
        let updated = model.forward_batch(&x);
        let next_rng = rng_next();
        (output.data, dx.data, updated.data, next_rng, per_layer)
    }
    let (forward, gradient, update, next, _) = run(false);
    let (forward_profiled, gradient_profiled, update_profiled, next_profiled, timings) = run(true);
    assert_eq!(forward, forward_profiled);
    assert_eq!(gradient, gradient_profiled);
    assert_eq!(update, update_profiled);
    assert_eq!(next, next_profiled, "profiling changed the RNG stream");
    assert_eq!(timings.len(), 3);
    // Durations can be zero under extremely coarse timer granularity.
    assert!(timings.iter().all(|&duration| duration < 600_000_000_000));
}
