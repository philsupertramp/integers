//! Held-out, paired integer/FP32 dense MLP experiment for issue #15.
//! A validation-only run never opens test data. A final run evaluates test
//! exactly once after the fixed training budget.
//! No FP arithmetic is performed in the integer Sequential training path.
// The shared FP32 oracle exposes training helpers used by other experiments.
// This target intentionally exercises only a subset of that reference API.
#[allow(dead_code)]
#[path = "../experiments/cnn_reference.rs"]
mod cnn_reference;
#[path = "mnist_nn/mnist_loader.rs"]
mod mnist_loader;
#[path = "../experiments/protocol.rs"]
mod protocol;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use integers::data::Dataset;
use integers::dyadic::{Dyadic, Tensor};
use integers::nn::{Linear, ReLU, Sequential};
use integers::rng::seed_rng;
use integers::{argmax, mse_grad};

const SHIFT: u32 = 7;
const LR_SHIFT: u32 = 7;
const IRIS_LR_SHIFT: u32 = 6; // selected from validation-only 3-split HPO
const MOM_SHIFT: u32 = 1;
const GRAD_CLIP: i32 = 8192;
const OUTPUT_BITS: u32 = 31;
// Fixed independently of model seed: the same examples are held out in all runs.
const SPLIT_SEED: u64 = 0x0123_4567_89AB_CDEF;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Integer,
    Fp32,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Task {
    Iris,
    Mnist,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Architecture {
    Dense,
    Cnn,
}
struct Config {
    task: Task,
    backend: Backend,
    architecture: Architecture,
    seed: u64,
    epochs: usize,
    batch: usize,
    lr_shift: u32,
    train_limit: Option<usize>,
    validation_limit: Option<usize>,
    data_dir: PathBuf,
    prepared_data: bool,
    report: PathBuf,
    smoke: bool,
    ci_sample: bool,
    validation_only: bool,
    final_test: bool,
    profile_training: bool,
    split_seed: u64,
}
impl Config {
    fn parse() -> Self {
        let args: Vec<String> = env::args().skip(1).collect();
        let mut cfg = Self {
            task: Task::Iris,
            backend: Backend::Integer,
            architecture: Architecture::Dense,
            seed: 1337,
            epochs: 0,
            train_limit: None,
            validation_limit: None,
            batch: 32,
            lr_shift: LR_SHIFT,
            data_dir: PathBuf::from("data/mnist"),
            prepared_data: false,
            report: PathBuf::from("reports/held_out.json"),
            smoke: false,
            ci_sample: false,
            validation_only: false,
            final_test: false,
            profile_training: false,
            split_seed: SPLIT_SEED,
        };
        let mut i = 0;
        let mut explicit_lr = false;
        while i < args.len() {
            let key = &args[i];
            if key == "--smoke" {
                cfg.smoke = true;
                i += 1;
                continue;
            }
            if key == "--ci-sample" {
                cfg.ci_sample = true;
                i += 1;
                continue;
            }
            if key == "--validation-only" {
                cfg.validation_only = true;
                i += 1;
                continue;
            }
            if key == "--prepared-data" {
                cfg.prepared_data = true;
                i += 1;
                continue;
            }
            if key == "--final-test" {
                cfg.final_test = true;
                i += 1;
                continue;
            }
            if key == "--profile-training" {
                cfg.profile_training = true;
                i += 1;
                continue;
            }
            let value = args
                .get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {key}"));
            match key.as_str() {
                "--dataset" => {
                    cfg.task = match value.as_str() {
                        "iris" => Task::Iris,
                        "mnist" => Task::Mnist,
                        _ => panic!("unknown dataset: {value}"),
                    }
                }
                "--architecture" => {
                    cfg.architecture = match value.as_str() {
                        "dense" => Architecture::Dense,
                        "cnn" => Architecture::Cnn,
                        _ => panic!("unknown architecture: {value}"),
                    }
                }
                "--train-limit" => {
                    cfg.train_limit = Some(value.parse().expect("invalid training sample limit"))
                }
                "--validation-limit" => {
                    cfg.validation_limit =
                        Some(value.parse().expect("invalid validation sample limit"))
                }
                "--backend" => {
                    cfg.backend = match value.as_str() {
                        "integer" => Backend::Integer,
                        "fp32" => Backend::Fp32,
                        _ => panic!("unknown backend: {value}"),
                    }
                }
                "--seed" => cfg.seed = value.parse().expect("invalid seed"),
                "--epochs" => cfg.epochs = value.parse().expect("invalid epochs"),
                "--batch-size" => cfg.batch = value.parse().expect("invalid batch size"),
                "--lr-shift" => {
                    cfg.lr_shift = value.parse().expect("invalid lr shift");
                    explicit_lr = true;
                }
                "--split-seed" => cfg.split_seed = value.parse().expect("invalid split seed"),
                "--data-dir" => cfg.data_dir = PathBuf::from(value),
                "--report" => cfg.report = PathBuf::from(value),
                _ => panic!("unknown option: {key}"),
            }
            i += 2;
        }
        assert!(
            cfg.architecture != Architecture::Cnn || cfg.task == Task::Mnist,
            "CNN requires MNIST"
        );
        assert!(
            cfg.train_limit.is_none_or(|v| v > 0) && cfg.validation_limit.is_none_or(|v| v > 0),
            "sample limits must be positive"
        );
        assert!(
            (cfg.train_limit.is_none() && cfg.validation_limit.is_none())
                || (cfg.validation_only && !cfg.ci_sample),
            "sample limits require validation-only mode"
        );
        if !explicit_lr && cfg.task == Task::Iris {
            cfg.lr_shift = IRIS_LR_SHIFT;
        }
        if cfg.epochs == 0 {
            cfg.epochs = if cfg.smoke {
                2
            } else if cfg.task == Task::Iris {
                200
            } else {
                10
            };
        }
        assert!(
            cfg.seed != 0
                && cfg.split_seed != 0
                && cfg.batch > 0
                && cfg.epochs > 0
                && cfg.lr_shift <= 30,
            "invalid seed, split seed, batch, epochs, or lr shift"
        );
        assert!(
            !cfg.profile_training
                || (cfg.architecture == Architecture::Dense && cfg.validation_only),
            "training profiling requires dense --validation-only mode"
        );
        assert!(
            !cfg.validation_only || !cfg.ci_sample,
            "--validation-only and --ci-sample are mutually exclusive"
        );
        assert!(
            !cfg.final_test || (!cfg.validation_only && !cfg.ci_sample && !cfg.smoke),
            "--final-test cannot be combined with validation-only, CI sample or smoke"
        );
        assert!(
            cfg.task != Task::Mnist || cfg.validation_only || cfg.ci_sample || cfg.final_test,
            "MNIST test access requires explicit --final-test; use --validation-only during development"
        );
        assert!(
            !cfg.ci_sample || (cfg.task == Task::Mnist && !cfg.smoke),
            "--ci-sample is only supported for MNIST and cannot be combined with --smoke"
        );
        cfg
    }
}

struct ExperimentData {
    pool: Dataset,
    test: Dataset,
    train_ids: Vec<usize>,
    validation_ids: Vec<usize>,
}

fn iris_rows(path: &Path) -> Vec<([f32; 4], u8)> {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts: Vec<_> = line.trim().split('\t').collect();
            assert_eq!(
                parts.len(),
                5,
                "Iris TSV must have four features and a class label"
            );
            let x = std::array::from_fn(|i| parts[i].parse::<f32>().expect("invalid Iris feature"));
            let y = parts[4].parse::<u8>().expect("invalid Iris label");
            assert!(y < 3, "Iris class must be 0..2");
            (x, y)
        })
        .collect()
}

/// Normalize using training features ONLY; validation/test features never
/// influence fit statistics. Their values may naturally exceed the fit range.
fn iris_dataset(rows: &[([f32; 4], u8)], limits: &[(f32, f32); 4]) -> Dataset {
    let mut data = Vec::with_capacity(rows.len() * 4);
    let mut targets = Vec::with_capacity(rows.len() * 3);
    for (x, y) in rows {
        for j in 0..4 {
            let (lo, hi) = limits[j];
            let q = if hi > lo {
                ((x[j] - lo) / (hi - lo) * 254.0 - 127.0).round()
            } else {
                0.0
            };
            data.push(Dyadic::new(
                q.clamp(i32::MIN as f32, i32::MAX as f32) as i32,
                SHIFT,
            ));
        }
        for c in 0..3 {
            targets.push(Dyadic::new(if c == *y as usize { 127 } else { 0 }, SHIFT));
        }
    }
    Dataset {
        inputs: Tensor::from_vec(data, vec![rows.len(), 4]),
        labels: rows.iter().map(|r| r.1).collect(),
        targets: Tensor::from_vec(targets, vec![rows.len(), 3]),
        n_classes: 3,
        input_shift: SHIFT as i32,
    }
}

fn load_data(cfg: &Config) -> ExperimentData {
    match cfg.task {
        Task::Iris => {
            // Historical repository split (130 development + 20 reserved).
            // IMPORTANT: the 20 were already evaluated in historical blogs;
            // this is a held-out protocol, not a pristine never-seen dataset.
            let pool = iris_rows(Path::new("data/iris_train.tsv"));
            // HPO must not even READ the historically reused test file.
            let test = if cfg.validation_only {
                Vec::new()
            } else {
                iris_rows(Path::new("data/iris_test.tsv"))
            };
            assert_eq!(pool.len(), 130, "expected bundled Iris development split");
            if !cfg.validation_only {
                assert_eq!(test.len(), 20, "expected Iris test split");
            }
            let labels: Vec<u8> = pool.iter().map(|r| r.1).collect();
            let (mut train_ids, validation_ids) = protocol::stratified_validation(
                &labels,
                if cfg.smoke { 1 } else { 8 },
                cfg.split_seed,
            );
            if cfg.smoke {
                let mut seen = [0; 3];
                train_ids.retain(|&idx| {
                    let c = labels[idx] as usize;
                    seen[c] += 1;
                    seen[c] <= 4
                });
            }
            let limits = std::array::from_fn(|j| {
                let lo = train_ids
                    .iter()
                    .map(|&i| pool[i].0[j])
                    .fold(f32::INFINITY, f32::min);
                let hi = train_ids
                    .iter()
                    .map(|&i| pool[i].0[j])
                    .fold(f32::NEG_INFINITY, f32::max);
                (lo, hi)
            });
            ExperimentData {
                pool: iris_dataset(&pool, &limits),
                test: iris_dataset(&test, &limits),
                train_ids,
                validation_ids,
            }
        }
        Task::Mnist => {
            let (pool, test) = if cfg.prepared_data {
                mnist_loader::load_prepared_mnist(&cfg.data_dir, cfg.validation_only)
            } else if cfg.validation_only {
                // Only canonical TRAINING IDX paths are constructed here.
                // No format auto-probe, metadata lookup, file read, or
                // allocation ever touches the official 10k test partition.
                let pool = mnist_loader::load_mnist_idx(
                    &cfg.data_dir.join("train-images-idx3-ubyte"),
                    &cfg.data_dir.join("train-labels-idx1-ubyte"),
                    true,
                )
                .expect("canonical MNIST training IDX loading failed");
                let empty_test = Dataset {
                    inputs: Tensor::from_vec(Vec::new(), vec![0, 784]),
                    labels: Vec::new(),
                    targets: Tensor::from_vec(Vec::new(), vec![0, 10]),
                    n_classes: 10,
                    input_shift: SHIFT as i32,
                };
                (pool, empty_test)
            } else {
                // Historical CI sampled mode and the explicit final-test mode
                // continue to require all four files. No test evaluation
                // occurs until the final training budget is complete.
                mnist_loader::load_mnist_auto(&cfg.data_dir, true)
                    .expect("MNIST IDX/Parquet loading failed")
            };
            assert_eq!(
                pool.len(),
                60000,
                "full canonical MNIST training split required; do not benchmark subsets"
            );
            assert_eq!(
                test.len(),
                if cfg.validation_only { 0 } else { 10000 },
                "MNIST test count mismatch"
            );
            assert_eq!((pool.n_features(), pool.n_classes), (784, 10));
            let validation_per_class = if cfg.smoke {
                1
            } else if cfg.ci_sample {
                20
            } else {
                500
            };
            let (mut train_ids, validation_ids) =
                protocol::stratified_validation(&pool.labels, validation_per_class, cfg.split_seed);
            if cfg.ci_sample || cfg.smoke {
                // The CI profile samples 64 examples per class from the ENTIRE
                // remaining training pool with an independent fixed RNG stream.
                // No validation or test row can be selected for training.
                let max_per_class = if cfg.ci_sample { 64 } else { 4 };
                let mut seen = [0; 10];
                let randomized = protocol::epoch_order(&train_ids, cfg.split_seed ^ 0xC1C1_0000, 0);
                train_ids = randomized
                    .into_iter()
                    .filter(|&idx| {
                        let c = pool.labels[idx] as usize;
                        if seen[c] == max_per_class {
                            return false;
                        }
                        seen[c] += 1;
                        true
                    })
                    .collect();
                assert!(
                    seen.iter().all(|&count| count == max_per_class),
                    "insufficient per-class MNIST rows for sampled run"
                );
                train_ids.sort_unstable();
            }
            ExperimentData {
                pool,
                test,
                train_ids,
                validation_ids,
            }
        }
    }
}

struct DenseFp32 {
    input: usize,
    output: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    grad_w: Vec<f32>,
    grad_b: Vec<f32>,
    vel_w: Vec<f32>,
    vel_b: Vec<f32>,
}
fn decoded(x: Dyadic) -> f32 {
    (x.v as f32) * 2f32.powi(-(x.s as i32))
}
impl DenseFp32 {
    fn from_integer(layer: &Linear) -> Self {
        let w: Vec<f32> = layer.weights.data.iter().copied().map(decoded).collect();
        let b: Vec<f32> = layer.biases.data.iter().copied().map(decoded).collect();
        Self {
            input: layer.in_features,
            output: layer.out_features,
            grad_w: vec![0.0; w.len()],
            grad_b: vec![0.0; b.len()],
            vel_w: vec![0.0; w.len()],
            vel_b: vec![0.0; b.len()],
            weights: w,
            biases: b,
        }
    }
    fn zero_grad(&mut self) {
        self.grad_w.fill(0.0);
        self.grad_b.fill(0.0);
    }
    fn update(&mut self, lr_shift: u32, samples: usize) {
        assert!(samples > 0, "FP32 update requires positive sample count");
        let inv_samples = 1.0f32 / samples as f32;
        // Apply the mean of summed per-example gradients BEFORE momentum.
        // Same architecture, starting weights, batch order, learning-rate and
        // momentum convention; different arithmetic (continuous FP32 vs dyadic).
        let lr = 2f32.powi(-(lr_shift as i32));
        let decay = 2f32.powi(-(MOM_SHIFT as i32));
        for i in 0..self.weights.len() {
            self.vel_w[i] = decay * self.vel_w[i] + self.grad_w[i] * inv_samples;
            self.weights[i] -= lr * self.vel_w[i];
        }
        for i in 0..self.biases.len() {
            self.vel_b[i] = decay * self.vel_b[i] + self.grad_b[i] * inv_samples;
            self.biases[i] -= lr * self.vel_b[i];
        }
    }
}

struct FloatMlp {
    layers: Vec<DenseFp32>,
}
impl FloatMlp {
    fn activations(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut history = vec![input.to_vec()];
        for (k, layer) in self.layers.iter().enumerate() {
            let x = history.last().unwrap();
            assert_eq!(x.len(), layer.input);
            let mut y = layer.biases.clone();
            for (j, yj) in y.iter_mut().enumerate() {
                for (i, &xi) in x.iter().enumerate() {
                    *yj += layer.weights[j * layer.input + i] * xi;
                }
                if k + 1 < self.layers.len() {
                    *yj = yj.max(0.0);
                }
            }
            history.push(y);
        }
        history
    }
    #[allow(dead_code)] // Reference-only prediction helper, not used by the smoke example.
    fn predict(&self, input: &[f32]) -> usize {
        let values = self.activations(input);
        values
            .last()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }
    fn zero_grad(&mut self) {
        for l in &mut self.layers {
            l.zero_grad();
        }
    }
    fn update(&mut self, lr_shift: u32, samples: usize) {
        for layer in &mut self.layers {
            layer.update(lr_shift, samples);
        }
    }
    // Shared FP32 arithmetic, split only at stopwatch boundaries.
    fn loss_and_prediction(activations: &[Vec<f32>], target: &[f32]) -> (usize, Vec<f32>) {
        let result = activations.last().unwrap();
        let pred = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(result.len(), target.len());
        let grad: Vec<f32> = result.iter().zip(target).map(|(p, t)| p - t).collect();
        (pred, grad)
    }

    fn backward_from_activations(&mut self, activations: &[Vec<f32>], mut grad: Vec<f32>) {
        for k in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[k];
            let input_act = &activations[k];
            let mut previous = vec![0.0; layer.input];
            // Match integer Linear::backward_one: clip the per-example
            // upstream gradient before parameter accumulation.
            let clip = GRAD_CLIP as f32 / (1u32 << SHIFT) as f32;
            for (j, &grad_j) in grad.iter().enumerate().take(layer.output) {
                let g = grad_j.clamp(-clip, clip);
                layer.grad_b[j] += g;
                for i in 0..layer.input {
                    let p = j * layer.input + i;
                    layer.grad_w[p] += g * input_act[i];
                    previous[i] += g * layer.weights[p];
                }
            }
            if k > 0 {
                for (g, &a) in previous.iter_mut().zip(&activations[k]) {
                    if a <= 0.0 {
                        *g = 0.0;
                    }
                }
            }
            grad = previous;
        }
    }

    fn accumulate(&mut self, input: &[f32], target: &[f32]) -> usize {
        let activations = self.activations(input);
        let (pred, grad) = Self::loss_and_prediction(&activations, target);
        self.backward_from_activations(&activations, grad);
        pred
    }

    fn accumulate_profiled(
        &mut self,
        input: &[f32],
        target: &[f32],
        profile: &mut TrainingProfile,
    ) -> usize {
        // Dense FP32 calls forward/backward per sample; integer uses batches.
        let activations = timed(&mut profile.forward_ns, || self.activations(input));
        let (pred, grad) = timed(&mut profile.loss_metrics_ns, || {
            Self::loss_and_prediction(&activations, target)
        });
        timed(&mut profile.backward_ns, || {
            self.backward_from_activations(&activations, grad)
        });
        pred
    }
}

fn build_models(seed: u64, dims: &[usize]) -> (Sequential, FloatMlp) {
    seed_rng(seed);
    let mut integer = Sequential::new();
    let mut float_layers = Vec::new();
    for (i, window) in dims.windows(2).enumerate() {
        let layer = Linear::new(window[0], window[1], SHIFT, SHIFT, OUTPUT_BITS)
            .with_grad_clip(GRAD_CLIP)
            .with_momentum(MOM_SHIFT);
        float_layers.push(DenseFp32::from_integer(&layer));
        integer.add(layer);
        if i + 1 < dims.len() - 1 {
            integer.add(ReLU::new());
        }
    }
    (
        integer,
        FloatMlp {
            layers: float_layers,
        },
    )
}

enum Fp32Model {
    Dense(FloatMlp),
    Cnn(cnn_reference::CnnFp32),
}
impl Fp32Model {
    fn forward_outputs(&mut self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(model) => model.activations(input).pop().expect("nonempty MLP"),
            Self::Cnn(model) => model.forward(input),
        }
    }
    fn accumulate(&mut self, input: &[f32], target: &[f32]) -> usize {
        match self {
            Self::Dense(model) => model.accumulate(input, target),
            Self::Cnn(model) => model.accumulate(input, target),
        }
    }
    fn zero_grad(&mut self) {
        match self {
            Self::Dense(model) => model.zero_grad(),
            Self::Cnn(model) => model.zero_grad(),
        }
    }
    fn update(&mut self, lr_shift: u32, samples: usize) {
        match self {
            Self::Dense(model) => model.update(lr_shift, samples),
            Self::Cnn(model) => model.update(lr_shift, samples),
        }
    }
    fn nonfinite_parameters(&self) -> usize {
        match self {
            Self::Dense(model) => model
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .weights
                        .iter()
                        .chain(layer.biases.iter())
                        .chain(layer.vel_w.iter())
                        .chain(layer.vel_b.iter())
                        .filter(|x| !x.is_finite())
                        .count()
                })
                .sum(),
            Self::Cnn(model) => model.nonfinite_parameters(),
        }
    }
}

fn float_row(ds: &Dataset, i: usize) -> Vec<f32> {
    ds.get_input(i).data.iter().copied().map(decoded).collect()
}
fn float_target(ds: &Dataset, i: usize) -> Vec<f32> {
    ds.get_target(i).data.iter().copied().map(decoded).collect()
}

struct Evaluation {
    correct: usize,
    predicted_counts: Vec<usize>,
    // Rows are true classes; columns are predicted classes.
    confusion: Vec<Vec<usize>>,
    nonfinite_outputs: usize,
    extreme_outputs: usize,
    max_abs_output: f64,
}
fn accuracy_integer(model: &mut Sequential, ds: &Dataset, ids: &[usize]) -> Evaluation {
    let mut stats = Evaluation {
        correct: 0,
        predicted_counts: vec![0; ds.n_classes],
        confusion: vec![vec![0; ds.n_classes]; ds.n_classes],
        nonfinite_outputs: 0,
        extreme_outputs: 0,
        max_abs_output: 0.0,
    };
    for &i in ids {
        let output = model.forward(&ds.get_input(i).view());
        let predicted = argmax(&output.data);
        stats.predicted_counts[predicted] += 1;
        stats.confusion[ds.labels[i] as usize][predicted] += 1;
        stats.correct += usize::from(predicted == ds.labels[i] as usize);
        for &value in &output.data {
            stats.max_abs_output = stats.max_abs_output.max(value.to_f64().abs());
            stats.extreme_outputs += usize::from(value.v == i32::MIN || value.v == i32::MAX);
        }
    }
    stats
}
fn accuracy_fp32(model: &mut Fp32Model, ds: &Dataset, ids: &[usize]) -> Evaluation {
    let mut stats = Evaluation {
        correct: 0,
        predicted_counts: vec![0; ds.n_classes],
        confusion: vec![vec![0; ds.n_classes]; ds.n_classes],
        nonfinite_outputs: 0,
        extreme_outputs: 0,
        max_abs_output: 0.0,
    };
    for &i in ids {
        let values = model.forward_outputs(&float_row(ds, i));
        let predicted = if let Fp32Model::Cnn(_) = model {
            let mut best = 0;
            for j in 1..values.len() {
                if values[j] > values[best] {
                    best = j;
                }
            }
            best
        } else {
            // Preserve the previously measured dense reference convention.
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        stats.predicted_counts[predicted] += 1;
        stats.confusion[ds.labels[i] as usize][predicted] += 1;
        stats.correct += usize::from(predicted == ds.labels[i] as usize);
        for &value in &values {
            if value.is_finite() {
                stats.max_abs_output = stats.max_abs_output.max(value.abs() as f64);
            } else {
                stats.nonfinite_outputs += 1;
            }
        }
    }
    stats
}
fn fp32_parameter_nonfinite(model: &Fp32Model) -> usize {
    model.nonfinite_parameters()
}

fn peak_rss_kb() -> Option<u64> {
    // Linux ru_maxrss is process-local. On non-Linux, do not invent values.
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
}
fn pct(correct: usize, size: usize) -> f64 {
    100.0 * correct as f64 / size as f64
}

#[derive(Default)]
struct TrainingProfile {
    preparation_ns: u128,
    zero_grad_ns: u128,
    forward_ns: u128,
    loss_metrics_ns: u128,
    backward_ns: u128,
    update_ns: u128,
    batches: usize,
    samples: usize,
    backward_layer_ns: Vec<u128>,
    backward_layer_names: Vec<&'static str>,
}

fn timed<T>(elapsed_ns: &mut u128, operation: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = operation();
    *elapsed_ns += started.elapsed().as_nanos();
    result
}

impl TrainingProfile {
    fn json(&self, wall_ns: u128) -> String {
        let measured = self.preparation_ns
            + self.zero_grad_ns
            + self.forward_ns
            + self.loss_metrics_ns
            + self.backward_ns
            + self.update_ns;
        // Unaccounted time includes shuffle, loop and timer overhead.
        let other_ns = wall_ns.saturating_sub(measured);
        let layers = self
            .backward_layer_ns
            .iter()
            .zip(&self.backward_layer_names)
            .enumerate()
            .map(|(index, (elapsed_ns, name))| {
                format!(
                    "{{\"index\":{},\"name\":\"{}\",\"elapsed_ns\":{}}}",
                    index, name, elapsed_ns
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":1,\"clock\":\"Instant\",\"unit\":\"ns\",\"batches\":{},\"samples\":{},\"training_wall_ns\":{},\"preparation_ns\":{},\"zero_grad_ns\":{},\"forward_ns\":{},\"loss_metrics_ns\":{},\"backward_ns\":{},\"update_ns\":{},\"other_ns\":{},\"backward_layers\":[{}]}}",
            self.batches,
            self.samples,
            wall_ns,
            self.preparation_ns,
            self.zero_grad_ns,
            self.forward_ns,
            self.loss_metrics_ns,
            self.backward_ns,
            self.update_ns,
            other_ns,
            layers
        )
    }
}

struct Epoch {
    epoch: usize,
    train_accuracy: f64,
    validation_accuracy: f64,
    training_samples: usize,
    elapsed_ms: u128,
    training_ms: u128,
    training_profile_json: String,
    validation_ms: u128,
    validation_predictions: Vec<usize>,
    validation_confusion: Vec<Vec<usize>>,
    validation_per_class_recall_json: String,
    validation_nonfinite: usize,
    validation_extremes: usize,
    validation_max_abs: f64,
    nonfinite_fp32_params: Option<usize>,
}
fn main() {
    let load_start = Instant::now();
    let cfg = Config::parse();
    let mut data = load_data(&cfg);
    // Validation-only subsampling is performed *after* the canonical stratified
    // split, with its own deterministic selection schedule. Never touch test.
    if let Some(n) = cfg.train_limit {
        data.train_ids = protocol::epoch_order(&data.train_ids, cfg.split_seed ^ 0xC035, 0)
            .into_iter()
            .take(n)
            .collect();
    }
    if let Some(n) = cfg.validation_limit {
        data.validation_ids =
            protocol::epoch_order(&data.validation_ids, cfg.split_seed ^ 0xCAFE, 0)
                .into_iter()
                .take(n)
                .collect();
    }
    let dataset_loading_ms = load_start.elapsed().as_millis();
    let model_start = Instant::now();
    let dims = match (cfg.task, cfg.architecture) {
        (Task::Iris, _) => vec![4, 8, 8, 3],
        (Task::Mnist, Architecture::Dense) => vec![784, 128, 128, 10],
        (Task::Mnist, Architecture::Cnn) => vec![784, 2704, 676, 968, 200, 64, 10],
    };
    let (mut integer, mut fp32) = match cfg.architecture {
        Architecture::Dense => {
            let (model, reference) = build_models(cfg.seed, &dims);
            (model, Fp32Model::Dense(reference))
        }
        Architecture::Cnn => {
            let (model, reference) = cnn_reference::build_models(cfg.seed);
            (model, Fp32Model::Cnn(reference))
        }
    };
    let model_initialization_ms = model_start.elapsed().as_millis();
    let mut curve = Vec::new();
    let start = Instant::now();
    for epoch in 0..cfg.epochs {
        let epoch_start = Instant::now();
        let order = protocol::epoch_order(&data.train_ids, cfg.seed, epoch);
        let mut correct = 0usize;
        let mut profile = TrainingProfile::default();
        if cfg.profile_training && cfg.backend == Backend::Integer {
            profile.backward_layer_ns = vec![0; integer.layers.len()];
            profile.backward_layer_names =
                integer.layers.iter().map(|layer| layer.name()).collect();
        }
        match cfg.backend {
            Backend::Integer => {
                integer.set_training(true);
                for ids in order.chunks(cfg.batch) {
                    if cfg.profile_training {
                        profile.batches += 1;
                        profile.samples += ids.len();
                        let (x, t) =
                            timed(&mut profile.preparation_ns, || data.pool.minibatch(ids));
                        timed(&mut profile.zero_grad_ns, || integer.zero_grad());
                        let y = timed(&mut profile.forward_ns, || integer.forward_batch(&x));
                        let gradient: Tensor = timed(&mut profile.loss_metrics_ns, || {
                            for (sample, &idx) in y.iter().zip(ids) {
                                if argmax(sample.data) == data.pool.labels[idx] as usize {
                                    correct += 1;
                                }
                            }
                            y.iter()
                                .zip(t.iter())
                                .map(|(y, t)| {
                                    Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec())
                                })
                                .collect()
                        });
                        let started = Instant::now();
                        integer.backward_batch_timed_discard_input_grad(
                            &gradient,
                            &mut profile.backward_layer_ns,
                        );
                        profile.backward_ns += started.elapsed().as_nanos();
                        timed(&mut profile.update_ns, || integer.update(cfg.lr_shift));
                    } else {
                        let (x, t) = data.pool.minibatch(ids);
                        integer.zero_grad();
                        let y = integer.forward_batch(&x);
                        for (sample, &idx) in y.iter().zip(ids) {
                            if argmax(sample.data) == data.pool.labels[idx] as usize {
                                correct += 1;
                            }
                        }
                        let gradient: Tensor = y
                            .iter()
                            .zip(t.iter())
                            .map(|(y, t)| {
                                Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec())
                            })
                            .collect();
                        integer.backward_batch_discard_input_grad(&gradient);
                        integer.update(cfg.lr_shift);
                    }
                }
                integer.set_training(false);
            }
            Backend::Fp32 => {
                for ids in order.chunks(cfg.batch) {
                    if cfg.profile_training {
                        profile.batches += 1;
                        profile.samples += ids.len();
                        let Fp32Model::Dense(model) = &mut fp32 else {
                            unreachable!("profiling requires dense MLP");
                        };
                        timed(&mut profile.zero_grad_ns, || model.zero_grad());
                        for &idx in ids {
                            let (x, t) = timed(&mut profile.preparation_ns, || {
                                (float_row(&data.pool, idx), float_target(&data.pool, idx))
                            });
                            if model.accumulate_profiled(&x, &t, &mut profile)
                                == data.pool.labels[idx] as usize
                            {
                                correct += 1;
                            }
                        }
                        timed(&mut profile.update_ns, || {
                            model.update(cfg.lr_shift, ids.len())
                        });
                    } else {
                        fp32.zero_grad();
                        for &idx in ids {
                            let x = float_row(&data.pool, idx);
                            let t = float_target(&data.pool, idx);
                            if fp32.accumulate(&x, &t) == data.pool.labels[idx] as usize {
                                correct += 1;
                            }
                        }
                        fp32.update(cfg.lr_shift, ids.len());
                    }
                }
            }
        }
        let training_wall_ns = epoch_start.elapsed().as_nanos();
        let training_ms = training_wall_ns / 1_000_000;
        let training_profile_json = if cfg.profile_training {
            assert_eq!(profile.samples, order.len());
            profile.json(training_wall_ns)
        } else {
            "null".to_string()
        };
        let validation_start = Instant::now();
        let val = match cfg.backend {
            Backend::Integer => accuracy_integer(&mut integer, &data.pool, &data.validation_ids),
            Backend::Fp32 => accuracy_fp32(&mut fp32, &data.pool, &data.validation_ids),
        };
        let validation_ms = validation_start.elapsed().as_millis();
        let recall_json = format!(
            "[{}]",
            val.confusion
                .iter()
                .enumerate()
                .map(|(class, row)| {
                    let total: usize = row.iter().sum();
                    if total == 0 {
                        "null".to_string()
                    } else {
                        format!("{:.8}", row[class] as f64 / total as f64)
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        );
        let metric = Epoch {
            epoch: epoch + 1,
            train_accuracy: pct(correct, data.train_ids.len()),
            validation_accuracy: pct(val.correct, data.validation_ids.len()),
            training_samples: order.len(),
            elapsed_ms: epoch_start.elapsed().as_millis(),
            training_ms,
            training_profile_json,
            validation_ms,
            validation_predictions: val.predicted_counts,
            validation_confusion: val.confusion,
            validation_per_class_recall_json: recall_json,
            validation_nonfinite: val.nonfinite_outputs,
            validation_extremes: val.extreme_outputs,
            validation_max_abs: val.max_abs_output,
            nonfinite_fp32_params: if cfg.backend == Backend::Fp32 {
                Some(fp32_parameter_nonfinite(&fp32))
            } else {
                None
            },
        };
        println!(
            "epoch={} train={:.2}% val={:.2}% wall_ms={} training_ms={} validation_ms={} predictions={:?} nonfinite_outputs={} extreme_outputs={} nonfinite_fp32_parameters={:?}",
            metric.epoch,
            metric.train_accuracy,
            metric.validation_accuracy,
            metric.elapsed_ms,
            metric.training_ms,
            metric.validation_ms,
            metric.validation_predictions,
            metric.validation_nonfinite,
            metric.validation_extremes,
            metric.nonfinite_fp32_params
        );
        if cfg.profile_training {
            println!(
                "TRAINING_PROFILE epoch={} backend={} {}",
                metric.epoch,
                if cfg.backend == Backend::Integer {
                    "integer"
                } else {
                    "fp32"
                },
                metric.training_profile_json
            );
        }
        curve.push(metric);
    }
    // Validation-only Iris/MNIST never reads or evaluates test data.
    // CI sample MNIST never evaluates the official test data. Full runs evaluate it
    // exactly once, after the fixed epoch budget has completed.
    let evaluation_start = Instant::now();
    let test_correct = if cfg.ci_sample || cfg.validation_only {
        None
    } else {
        let test_ids: Vec<usize> = (0..data.test.len()).collect();
        Some(match cfg.backend {
            Backend::Integer => accuracy_integer(&mut integer, &data.test, &test_ids).correct,
            Backend::Fp32 => accuracy_fp32(&mut fp32, &data.test, &test_ids).correct,
        })
    };
    let test_evaluation_ms = evaluation_start.elapsed().as_millis();
    let total_ms = start.elapsed().as_millis();
    let backend = if cfg.backend == Backend::Integer {
        "integer"
    } else {
        "fp32"
    };
    let task = if cfg.task == Task::Iris {
        "iris"
    } else {
        "mnist"
    };
    let architecture_kind = if cfg.architecture == Architecture::Dense {
        "dense_mlp"
    } else {
        "mnist_cnn_v1"
    };
    if let Some(correct) = test_correct {
        println!(
            "FINAL {task} {backend} seed={} test={}/{} ({:.2}%) duration_ms={total_ms}",
            cfg.seed,
            correct,
            data.test.len(),
            pct(correct, data.test.len())
        );
    } else {
        println!(
            "VALIDATION ONLY {task} {backend} seed={} test=NOT_EVALUATED duration_ms={total_ms}",
            cfg.seed
        );
    }

    let epochs: Vec<String> = curve.iter().map(|m| format!(
        "{{\"epoch\":{},\"train_accuracy_pct\":{:.6},\"validation_accuracy_pct\":{:.6},\"training_samples\":{},\"wall_ms\":{},\"training_ms\":{},\"training_profile\":{},\"validation_ms\":{},\"validation_predictions\":{:?},\"validation_confusion\":{:?},\"validation_per_class_recall\":{},\"nonfinite_outputs\":{},\"extreme_outputs\":{},\"max_abs_output\":{:.8},\"nonfinite_fp32_parameters\":{}}}",
        m.epoch, m.train_accuracy, m.validation_accuracy, m.training_samples, m.elapsed_ms, m.training_ms, m.training_profile_json, m.validation_ms,
        m.validation_predictions, m.validation_confusion, m.validation_per_class_recall_json, m.validation_nonfinite, m.validation_extremes, m.validation_max_abs,
        m.nonfinite_fp32_params.map_or("null".to_string(), |n| n.to_string()),
    )).collect();
    let rss = peak_rss_kb().map_or("null".to_owned(), |n| n.to_string());
    let test_correct_json = test_correct.map_or("null".to_owned(), |n| n.to_string());
    let test_accuracy_json = test_correct.map_or("null".to_owned(), |n| {
        format!("{:.6}", pct(n, data.test.len()))
    });
    let payload = format!(
        "{{\n  \"schema\": 1, \"dataset\": \"{task}\", \"backend\": \"{backend}\", \"smoke\": {}, \"ci_sample\": {},\n  \"seed\": {}, \"split_seed\": {}, \"epochs\": {}, \"batch_size\": {}, \"architecture_kind\": {:?}, \"train_limit\": {}, \"validation_limit\": {},\n  \"architecture\": {:?}, \"quantization_shift\": {}, \"output_bits\": {}, \"lr_shift\": {}, \"momentum_shift\": {}, \"grad_clip_mantissa\": {},\n  \"train_size\": {}, \"validation_size\": {}, \"test_size\": {},\n  \"validation_only\": {}, \"profile_training\": {}, \"test_evaluated\": {}, \"test_correct\": {}, \"test_accuracy_pct\": {}, \"wall_ms\": {}, \"peak_rss_kb\": {}, \"dataset_loading_ms\": {}, \"model_initialization_ms\": {}, \"test_evaluation_ms\": {},\n  \"learning_curve\": [{}]\n}}\n",
        cfg.smoke,
        cfg.ci_sample,
        cfg.seed,
        cfg.split_seed,
        cfg.epochs,
        cfg.batch,
        architecture_kind,
        cfg.train_limit
            .map_or("null".to_string(), |n| n.to_string()),
        cfg.validation_limit
            .map_or("null".to_string(), |n| n.to_string()),
        dims,
        SHIFT,
        OUTPUT_BITS,
        cfg.lr_shift,
        MOM_SHIFT,
        GRAD_CLIP,
        data.train_ids.len(),
        data.validation_ids.len(),
        data.test.len(),
        cfg.validation_only,
        cfg.profile_training,
        test_correct.is_some(),
        test_correct_json,
        test_accuracy_json,
        total_ms,
        rss,
        dataset_loading_ms,
        model_initialization_ms,
        test_evaluation_ms,
        epochs.join(","),
    );
    if let Some(parent) = cfg.report.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    fs::write(&cfg.report, payload).expect("write report JSON");
}
