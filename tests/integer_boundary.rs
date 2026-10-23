//! ADR-002: the standard Sequential path must fail closed for runtime FP layers.
//! Numeric oracle checks below inspect the *experimental* compatibility path;
//! they do not certify its computations as integer-only.
use integers::nn::{
    BatchNorm1D, BatchNorm2D, Conv2D, Dropout, Flatten, Linear, MaxPool2D, Module, ReLU,
    Sequential, Softmax, TrainingArithmetic,
};
use integers::rng::seed_rng;
use integers::{Dyadic, Tensor, mse_grad};

fn d(v: i32, s: u32) -> Dyadic {
    Dyadic::new(v, s)
}
fn t(xs: &[i32], s: u32, shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.iter().map(|&v| d(v, s)).collect(), shape.to_vec())
}
fn decode(v: Dyadic) -> f64 {
    (v.v as f64) * 2f64.powi(-(v.s as i32))
}
fn panics(f: impl FnOnce()) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
}
fn near(v: Dyadic, expected: f64, lsb: f64) {
    let abs = (decode(v) - expected).abs();
    let bound = lsb * 2f64.powi(-(v.s as i32)) + 1e-12;
    assert!(
        abs <= bound,
        "got {:?} vs {expected}, abs {abs}, max {bound}",
        v
    );
}

#[test]
fn trusted_builtin_arithmetic_is_explicit_and_unverified_defaults_fail_closed() {
    struct Unknown;
    impl Module for Unknown {
        fn name(&self) -> &'static str {
            "unverified"
        }
        fn forward(&mut self, x: &integers::dyadic::TensorView) -> Tensor {
            x.to_tensor()
        }
        fn backward(&mut self, x: &integers::dyadic::TensorView) -> Tensor {
            x.to_tensor()
        }
        fn update(&mut self, _: u32) {}
        fn zero_grad(&mut self) {}
    }
    assert_eq!(
        Unknown.training_arithmetic(),
        TrainingArithmetic::Unverified
    );
    assert_eq!(
        Linear::new(1, 1, 4, 4, 16).training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        Conv2D::new(1, 1, 1, 1, 1, 1, 4, 4, 16).training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        ReLU::new().training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        MaxPool2D::new(1, 2, 2, 2, 2).training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        Flatten.training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        Dropout::new(0.5).training_arithmetic(),
        TrainingArithmetic::IntegerOnly
    );
    assert_eq!(
        Softmax::new(7).training_arithmetic(),
        TrainingArithmetic::FloatingPoint
    );
    assert_eq!(
        BatchNorm1D::new(1, 7).training_arithmetic(),
        TrainingArithmetic::FloatingPoint
    );
    assert_eq!(
        BatchNorm2D::new(1, 1, 2, 7).training_arithmetic(),
        TrainingArithmetic::FloatingPoint
    );
    panics(|| Sequential::new().add(Softmax::new(7)));
    panics(|| Sequential::new().add(BatchNorm1D::new(1, 7)));
    panics(|| Sequential::new().add(BatchNorm2D::new(1, 1, 2, 7)));
    panics(|| Sequential::new().add(Unknown));
}

#[test]
fn public_layers_vector_cannot_bypass_the_integer_boundary() {
    let mut model = Sequential::new();
    model.layers.push(Box::new(Softmax::new(7)));
    panics(|| {
        model.forward(&t(&[0, 128], 7, &[2]).view());
    });
    panics(|| {
        model.backward(&t(&[0, 128], 7, &[2]).view());
    });
    panics(|| {
        model.forward_batch(&t(&[0, 128], 7, &[1, 2]));
    });
    panics(|| {
        model.backward_batch(&t(&[0, 128], 7, &[1, 2]));
    });
    panics(|| {
        model.update(1);
    });
    panics(|| {
        model.zero_grad();
    });
    panics(|| {
        model.set_training(false);
    });

    let mut another = Sequential::new();
    another.layers.push(Box::new(BatchNorm1D::new(1, 7)));
    panics(|| {
        another.forward(&t(&[0], 7, &[1]).view());
    });
    panics(|| {
        another.update(1);
    });
}

#[test]
fn strict_integer_only_model_can_train_with_momentum_and_dropout() {
    seed_rng(0x260);
    let mut model = Sequential::new();
    model.add(Linear::new(2, 2, 7, 7, 16).with_momentum(1));
    model.add(ReLU::new());
    model.add(Dropout::new(0.25));
    model.add(Linear::new(2, 1, 7, 7, 16));
    let x = t(&[64, -32, 32, 64], 7, &[2, 2]);
    let target = t(&[16, 64], 7, &[2, 1]);
    model.zero_grad();
    let out = model.forward_batch(&x);
    let grad = Tensor::from_vec(mse_grad(&out.data, &target.data), out.shape.clone());
    let dx = model.backward_batch(&grad);
    assert_eq!(dx.shape, x.shape);
    assert_eq!(dx.data.len(), x.data.len());
    model.update(7);
    model.zero_grad();
    let next = model.forward_batch(&x);
    assert_eq!(next.data.len(), 2);
}

#[test]
fn explicitly_opted_in_experimental_softmax_retains_reference_behavior() {
    let mut experimental = Sequential::new_experimental_float();
    experimental.add(Softmax::new(7));
    let inp = t(&[-2, 0, 2], 1, &[3]); // [-1,0,1]
    let out = experimental.forward(&inp.view());
    let es: Vec<f64> = [-1.0f64, 0.0, 1.0].iter().map(|v| v.exp()).collect();
    let denom: f64 = es.iter().sum();
    for (actual, reference) in out.data.iter().zip(es.iter()) {
        near(*actual, reference / denom, 0.5);
    }
    let g = t(&[1, -2, 1], 7, &[3]);
    assert_eq!(experimental.backward(&g.view()).data, g.data); // identity STE
}

#[test]
fn experimental_batch_norm_matches_fp64_and_zero_variance_cases() {
    const S: u32 = 7;
    let mut batch1d = Sequential::new_experimental_float();
    batch1d.add(BatchNorm1D::new(1, S));
    let x = t(&[-128, 128], S, &[2, 1]); // [-1, 1], mean=0, variance=1
    let out = batch1d.forward_batch(&x);
    for (got, expected) in out.data.iter().zip([-1.0f64, 1.0]) {
        near(*got, expected / (1.0 + 1e-5f64).sqrt(), 0.5);
    }
    let zero = t(&[32, 32], S, &[2, 1]);
    let outputs = batch1d.forward_batch(&zero);
    assert!(outputs.data.iter().all(|x| x.v == 0));
    let grads = t(&[128, 128], S, &[2, 1]);
    let dx = batch1d.backward_batch(&grads);
    assert_eq!(dx.data.len(), 2);
    assert!(dx.data.iter().all(|x| x.v == 0));

    let mut batch2d = Sequential::new_experimental_float();
    batch2d.add(BatchNorm2D::new(1, 1, 2, S));
    let x2 = t(&[-128, 0, 128, 256], S, &[2, 2]); // [-1,0,1,2]; mean .5 variance 1.25
    let y = batch2d.forward_batch(&x2);
    let inv = (1.25f64 + 1e-5).sqrt();
    for (got, expected) in y.data.iter().zip([-1.0, -0.0, 1.0, 2.0]) {
        near(*got, (expected - 0.5) / inv, 1.5);
    }
    let zero2d = t(&[64, 64, 64, 64], S, &[2, 2]);
    let z = batch2d.forward_batch(&zero2d);
    assert!(z.data.iter().all(|x| x.v == 0));
    let zero_g = batch2d.backward_batch(&t(&[128; 4], S, &[2, 2]));
    assert!(zero_g.data.iter().all(|x| x.v == 0));
}

#[test]
fn experimental_batchnorm_multifeature_and_multichannel_layouts_use_real_batch_size() {
    const S: u32 = 7;
    let mut bn1 = Sequential::new_experimental_float();
    bn1.add(BatchNorm1D::new(2, S));
    let input1 = t(&[-128, 0, 0, 256, 128, 512], S, &[3, 2]);
    let output1 = bn1.forward_batch(&input1);
    assert_eq!(output1.shape, vec![3, 2]);
    let inputs = [[-1.0f64, 0.0], [0.0, 2.0], [1.0, 4.0]];
    for feature in 0..2 {
        let mean: f64 = inputs.iter().map(|x| x[feature]).sum::<f64>() / 3.0;
        let variance = inputs
            .iter()
            .map(|x| (x[feature] - mean).powi(2))
            .sum::<f64>()
            / 3.0;
        for sample in 0..3 {
            near(
                output1.data[sample * 2 + feature],
                (inputs[sample][feature] - mean) / (variance + 1e-5).sqrt(),
                0.51,
            );
        }
    }
    let grads1 = t(&[128, 0, 0, 128, -128, 128], S, &[3, 2]);
    assert_eq!(bn1.backward_batch(&grads1).shape, vec![3, 2]);
    panics(|| {
        bn1.backward_batch(&t(&[128; 4], S, &[2, 2]));
    });

    let mut bn2 = Sequential::new_experimental_float();
    bn2.add(BatchNorm2D::new(2, 1, 2, S));
    let input2 = t(&[-128, 0, -256, 128, 128, 256, 256, 384], S, &[2, 4]);
    let output2 = bn2.forward_batch(&input2);
    assert_eq!(output2.shape, vec![2, 4]);
    let decoded = [[-1.0f64, 0.0, -2.0, 1.0], [1.0, 2.0, 2.0, 3.0]];
    for channel in 0..2 {
        let vals = [
            decoded[0][2 * channel],
            decoded[0][2 * channel + 1],
            decoded[1][2 * channel],
            decoded[1][2 * channel + 1],
        ];
        let mean = vals.iter().sum::<f64>() / 4.0;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        for sample in 0..2 {
            for spatial in 0..2 {
                let idx = sample * 4 + channel * 2 + spatial;
                near(
                    output2.data[idx],
                    (decoded[sample][channel * 2 + spatial] - mean) / (var + 1e-5).sqrt(),
                    0.51,
                );
            }
        }
    }
    let grads2 = t(&[128, 0, 0, 128, 0, 128, 128, 0], S, &[2, 4]);
    let backward = bn2.backward_batch(&grads2);
    assert_eq!(backward.shape, vec![2, 4]);
    for channel in 0..2 {
        let vals = [
            decoded[0][2 * channel],
            decoded[0][2 * channel + 1],
            decoded[1][2 * channel],
            decoded[1][2 * channel + 1],
        ];
        let gg = [
            decode(grads2.data[channel * 2]),
            decode(grads2.data[channel * 2 + 1]),
            decode(grads2.data[4 + channel * 2]),
            decode(grads2.data[4 + channel * 2 + 1]),
        ];
        let mean = vals.iter().sum::<f64>() / 4.0;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let xhat = vals.map(|v| (v - mean) * inv_std);
        let sg: f64 = gg.iter().sum();
        let sgx: f64 = gg.iter().zip(xhat.iter()).map(|(g, x)| g * x).sum();
        for sample in 0..2 {
            for spatial in 0..2 {
                let i = sample * 2 + spatial;
                let expected = inv_std * (gg[i] - sg / 4.0 - xhat[i] * sgx / 4.0);
                near(
                    backward.data[sample * 4 + channel * 2 + spatial],
                    expected,
                    1.5,
                );
            }
        }
    }
    panics(|| {
        bn2.backward_batch(&t(&[128; 4], S, &[1, 4]));
    });
}
