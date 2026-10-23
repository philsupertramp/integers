//! Held-out, paired integer/FP32 dense MLP experiment for issue #15.
//! NEVER evaluate the final test set until the fixed epoch budget is complete.
//! No FP arithmetic is performed in the integer Sequential training path.
#[path = "../experiments/protocol.rs"]
mod protocol;
#[path = "mnist_nn/mnist_loader.rs"]
mod mnist_loader;

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
const MOM_SHIFT: u32 = 1;
const GRAD_CLIP: i32 = 8192;
const OUTPUT_BITS: u32 = 31;
// Fixed independently of model seed: the same examples are held out in all runs.
const SPLIT_SEED: u64 = 0x0123_4567_89AB_CDEF;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend { Integer, Fp32 }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Task { Iris, Mnist }
struct Config {
    task: Task, backend: Backend, seed: u64, epochs: usize, batch: usize,
    data_dir: PathBuf, report: PathBuf, smoke: bool,
}
impl Config {
    fn parse() -> Self {
        let args: Vec<String> = env::args().skip(1).collect();
        let mut cfg = Self {
            task: Task::Iris, backend: Backend::Integer, seed: 1337, epochs: 0,
            batch: 32, data_dir: PathBuf::from("data/mnist"),
            report: PathBuf::from("reports/held_out.json"), smoke: false,
        };
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            if key == "--smoke" { cfg.smoke = true; i += 1; continue; }
            let value = args.get(i + 1).unwrap_or_else(|| panic!("missing value for {key}"));
            match key.as_str() {
                "--dataset" => cfg.task = match value.as_str() {
                    "iris" => Task::Iris, "mnist" => Task::Mnist, _ => panic!("unknown dataset: {value}"),
                },
                "--backend" => cfg.backend = match value.as_str() {
                    "integer" => Backend::Integer, "fp32" => Backend::Fp32, _ => panic!("unknown backend: {value}"),
                },
                "--seed" => cfg.seed = value.parse().expect("invalid seed"),
                "--epochs" => cfg.epochs = value.parse().expect("invalid epochs"),
                "--batch-size" => cfg.batch = value.parse().expect("invalid batch size"),
                "--data-dir" => cfg.data_dir = PathBuf::from(value),
                "--report" => cfg.report = PathBuf::from(value),
                _ => panic!("unknown option: {key}"),
            }
            i += 2;
        }
        if cfg.epochs == 0 { cfg.epochs = if cfg.smoke { 2 } else if cfg.task == Task::Iris { 100 } else { 10 }; }
        assert!(cfg.seed != 0 && cfg.batch > 0 && cfg.epochs > 0, "seed, batch and epochs must be positive");
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
    raw.lines().filter(|line| !line.trim().is_empty()).map(|line| {
        let parts: Vec<_> = line.trim().split('\t').collect();
        assert_eq!(parts.len(), 5, "Iris TSV must have four features and a class label");
        let x = std::array::from_fn(|i| parts[i].parse::<f32>().expect("invalid Iris feature"));
        let y = parts[4].parse::<u8>().expect("invalid Iris label");
        assert!(y < 3, "Iris class must be 0..2");
        (x, y)
    }).collect()
}

/// Normalize using training features ONLY; validation/test features never
/// influence fit statistics. Their values may naturally exceed the fit range.
fn iris_dataset(rows: &[([f32; 4], u8)], limits: &[(f32, f32); 4]) -> Dataset {
    let mut data = Vec::with_capacity(rows.len() * 4);
    let mut targets = Vec::with_capacity(rows.len() * 3);
    for (x, y) in rows {
        for j in 0..4 {
            let (lo, hi) = limits[j];
            let q = if hi > lo { ((x[j] - lo) / (hi - lo) * 254.0 - 127.0).round() } else { 0.0 };
            data.push(Dyadic::new(q.clamp(i32::MIN as f32, i32::MAX as f32) as i32, SHIFT));
        }
        for c in 0..3 { targets.push(Dyadic::new(if c == *y as usize { 127 } else { 0 }, SHIFT)); }
    }
    Dataset {
        inputs: Tensor::from_vec(data, vec![rows.len(), 4]),
        labels: rows.iter().map(|r| r.1).collect(),
        targets: Tensor::from_vec(targets, vec![rows.len(), 3]),
        n_classes: 3, input_shift: SHIFT as i32,
    }
}

fn load_data(cfg: &Config) -> ExperimentData {
    match cfg.task {
        Task::Iris => {
            // Historical repository split (130 development + 20 reserved).
            // IMPORTANT: the 30 were already evaluated in historical blogs;
            // this is a held-out protocol, not a pristine never-seen dataset.
            let pool = iris_rows(Path::new("data/iris_train.tsv"));
            let test = iris_rows(Path::new("data/iris_test.tsv"));
            assert_eq!((pool.len(), test.len()), (130, 20), "expected bundled Iris split");
            let labels: Vec<u8> = pool.iter().map(|r| r.1).collect();
            let (mut train_ids, validation_ids) =
                protocol::stratified_validation(&labels, if cfg.smoke { 1 } else { 8 }, SPLIT_SEED);
            if cfg.smoke {
                let mut seen = [0; 3];
                train_ids.retain(|&idx| {
                    let c = labels[idx] as usize;
                    seen[c] += 1;
                    seen[c] <= 4
                });
            }
            let limits = std::array::from_fn(|j| {
                let lo = train_ids.iter().map(|&i| pool[i].0[j]).fold(f32::INFINITY, f32::min);
                let hi = train_ids.iter().map(|&i| pool[i].0[j]).fold(f32::NEG_INFINITY, f32::max);
                (lo, hi)
            });
            ExperimentData {
                pool: iris_dataset(&pool, &limits),
                test: iris_dataset(&test, &limits),
                train_ids, validation_ids,
            }
        }
        Task::Mnist => {
            let (pool, test) = mnist_loader::load_mnist_auto(&cfg.data_dir, true)
                .expect("MNIST IDX/Parquet loading failed");
            assert_eq!((pool.len(), test.len()), (60000, 10000),
                "full canonical MNIST split required; do not benchmark subsets");
            assert_eq!((pool.n_features(), pool.n_classes), (784, 10));
            let (mut train_ids, validation_ids) = protocol::stratified_validation(
                &pool.labels, if cfg.smoke { 1 } else { 500 }, SPLIT_SEED,
            );
            if cfg.smoke {
                let mut seen = [0; 10];
                train_ids.retain(|&idx| {
                    let c = pool.labels[idx] as usize;
                    seen[c] += 1;
                    seen[c] <= 4
                });
            }
            ExperimentData { pool, test, train_ids, validation_ids }
        }
    }
}

struct DenseFp32 {
    input: usize, output: usize,
    weights: Vec<f32>, biases: Vec<f32>,
    grad_w: Vec<f32>, grad_b: Vec<f32>,
    vel_w: Vec<f32>, vel_b: Vec<f32>,
}
fn decoded(x: Dyadic) -> f32 { (x.v as f32) * 2f32.powi(-(x.s as i32)) }
impl DenseFp32 {
    fn from_integer(layer: &Linear) -> Self {
        let w: Vec<f32> = layer.weights.data.iter().copied().map(decoded).collect();
        let b: Vec<f32> = layer.biases.data.iter().copied().map(decoded).collect();
        Self {
            input: layer.in_features, output: layer.out_features,
            grad_w: vec![0.0; w.len()], grad_b: vec![0.0; b.len()],
            vel_w: vec![0.0; w.len()], vel_b: vec![0.0; b.len()],
            weights: w, biases: b,
        }
    }
    fn zero_grad(&mut self) {
        self.grad_w.fill(0.0);
        self.grad_b.fill(0.0);
    }
    fn update(&mut self) {
        // Same architecture, starting weights, batch order, learning-rate and
        // momentum convention; different arithmetic (continuous FP32 vs dyadic).
        let lr = 2f32.powi(-(LR_SHIFT as i32));
        let decay = 2f32.powi(-(MOM_SHIFT as i32));
        for i in 0..self.weights.len() {
            self.vel_w[i] = decay * self.vel_w[i] + self.grad_w[i];
            self.weights[i] -= lr * self.vel_w[i];
        }
        for i in 0..self.biases.len() {
            self.vel_b[i] = decay * self.vel_b[i] + self.grad_b[i];
            self.biases[i] -= lr * self.vel_b[i];
        }
    }
}

struct FloatMlp { layers: Vec<DenseFp32> }
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
                if k + 1 < self.layers.len() { *yj = yj.max(0.0); }
            }
            history.push(y);
        }
        history
    }
    fn predict(&self, input: &[f32]) -> usize {
        let values = self.activations(input);
        values.last().unwrap().iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
    }
    fn zero_grad(&mut self) { for l in &mut self.layers { l.zero_grad(); } }
    fn update(&mut self) { for l in &mut self.layers { l.update(); } }
    fn accumulate(&mut self, input: &[f32], target: &[f32]) -> usize {
        let activations = self.activations(input);
        let result = activations.last().unwrap();
        let pred = result.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(result.len(), target.len());
        let mut grad: Vec<f32> = result.iter().zip(target).map(|(p, t)| p - t).collect();
        for k in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[k];
            let input_act = &activations[k];
            let mut previous = vec![0.0; layer.input];
            // Match integer Linear::backward_one: clip the *per-example*
            // upstream neuron gradient before parameter accumulation,
            // not the aggregate gradient at optimizer update time.
            let clip = GRAD_CLIP as f32 / (1u32 << SHIFT) as f32;
            for j in 0..layer.output {
                let g = grad[j].clamp(-clip, clip);
                layer.grad_b[j] += g;
                for i in 0..layer.input {
                    let p = j * layer.input + i;
                    layer.grad_w[p] += g * input_act[i];
                    previous[i] += g * layer.weights[p];
                }
            }
            if k > 0 {
                for (g, &a) in previous.iter_mut().zip(&activations[k]) {
                    if a <= 0.0 { *g = 0.0; }
                }
            }
            grad = previous;
        }
        pred
    }
}

fn build_models(seed: u64, dims: &[usize]) -> (Sequential, FloatMlp) {
    seed_rng(seed);
    let mut integer = Sequential::new();
    let mut float_layers = Vec::new();
    for (i, window) in dims.windows(2).enumerate() {
        let layer = Linear::new(window[0], window[1], SHIFT, SHIFT, OUTPUT_BITS)
            .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT);
        float_layers.push(DenseFp32::from_integer(&layer));
        integer.add(layer);
        if i + 1 < dims.len() - 1 { integer.add(ReLU::new()); }
    }
    (integer, FloatMlp { layers: float_layers })
}

fn float_row(ds: &Dataset, i: usize) -> Vec<f32> {
    ds.get_input(i).data.iter().copied().map(decoded).collect()
}
fn float_target(ds: &Dataset, i: usize) -> Vec<f32> {
    ds.get_target(i).data.iter().copied().map(decoded).collect()
}

fn accuracy_integer(model: &mut Sequential, ds: &Dataset, ids: &[usize]) -> usize {
    ids.iter().filter(|&&i| {
        argmax(&model.forward(&ds.get_input(i).view()).data) == ds.labels[i] as usize
    }).count()
}
fn accuracy_fp32(model: &FloatMlp, ds: &Dataset, ids: &[usize]) -> usize {
    ids.iter().filter(|&&i| model.predict(&float_row(ds, i)) == ds.labels[i] as usize).count()
}

fn peak_rss_kb() -> Option<u64> {
    // Linux ru_maxrss is process-local. On non-Linux, do not invent values.
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
}
fn pct(correct: usize, size: usize) -> f64 { 100.0 * correct as f64 / size as f64 }

struct Epoch {
    epoch: usize, train_accuracy: f64, validation_accuracy: f64,
    training_samples: usize, elapsed_ms: u128,
}
fn main() {
    let cfg = Config::parse();
    let data = load_data(&cfg);
    let dims = match cfg.task {
        Task::Iris => vec![4, 8, 8, 3],
        Task::Mnist => vec![784, 128, 128, 10],
    };
    let (mut integer, mut fp32) = build_models(cfg.seed, &dims);
    let mut curve = Vec::new();
    let start = Instant::now();
    for epoch in 0..cfg.epochs {
        let epoch_start = Instant::now();
        let order = protocol::epoch_order(&data.train_ids, cfg.seed, epoch);
        let mut correct = 0usize;
        match cfg.backend {
            Backend::Integer => {
                integer.set_training(true);
                for ids in order.chunks(cfg.batch) {
                    let (x, t) = data.pool.minibatch(ids);
                    integer.zero_grad();
                    let y = integer.forward_batch(&x);
                    for (sample, &idx) in y.iter().zip(ids) {
                        if argmax(sample.data) == data.pool.labels[idx] as usize { correct += 1; }
                    }
                    let gradient: Tensor = y.iter().zip(t.iter()).map(|(y, t)| {
                        Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec())
                    }).collect();
                    integer.backward_batch(&gradient);
                    integer.update(LR_SHIFT);
                }
                integer.set_training(false);
            }
            Backend::Fp32 => {
                for ids in order.chunks(cfg.batch) {
                    fp32.zero_grad();
                    for &idx in ids {
                        let x = float_row(&data.pool, idx);
                        let t = float_target(&data.pool, idx);
                        if fp32.accumulate(&x, &t) == data.pool.labels[idx] as usize { correct += 1; }
                    }
                    fp32.update();
                }
            }
        }
        let val_correct = match cfg.backend {
            Backend::Integer => accuracy_integer(&mut integer, &data.pool, &data.validation_ids),
            Backend::Fp32 => accuracy_fp32(&fp32, &data.pool, &data.validation_ids),
        };
        let metric = Epoch {
            epoch: epoch + 1,
            train_accuracy: pct(correct, data.train_ids.len()),
            validation_accuracy: pct(val_correct, data.validation_ids.len()),
            training_samples: order.len(),
            elapsed_ms: epoch_start.elapsed().as_millis(),
        };
        println!("epoch={} train={:.2}% val={:.2}% wall_ms={}",
            metric.epoch, metric.train_accuracy, metric.validation_accuracy, metric.elapsed_ms);
        curve.push(metric);
    }
    // This is intentionally the FIRST and ONLY read of the reserved test set.
    let test_ids: Vec<usize> = (0..data.test.len()).collect();
    let test_correct = match cfg.backend {
        Backend::Integer => accuracy_integer(&mut integer, &data.test, &test_ids),
        Backend::Fp32 => accuracy_fp32(&fp32, &data.test, &test_ids),
    };
    let total_ms = start.elapsed().as_millis();
    let backend = if cfg.backend == Backend::Integer { "integer" } else { "fp32" };
    let task = if cfg.task == Task::Iris { "iris" } else { "mnist" };
    println!("FINAL {task} {backend} seed={} test={}/{} ({:.2}%) duration_ms={total_ms}",
        cfg.seed, test_correct, data.test.len(), pct(test_correct, data.test.len()));

    let epochs: Vec<String> = curve.iter().map(|m| format!(
        "{{\"epoch\":{},\"train_accuracy_pct\":{:.6},\"validation_accuracy_pct\":{:.6},\"training_samples\":{},\"wall_ms\":{}}}",
        m.epoch, m.train_accuracy, m.validation_accuracy, m.training_samples, m.elapsed_ms,
    )).collect();
    let rss = peak_rss_kb().map_or("null".to_owned(), |n| n.to_string());
    let payload = format!(
        "{{\n  \"schema\": 1, \"dataset\": \"{task}\", \"backend\": \"{backend}\", \"smoke\": {},\n  \"seed\": {}, \"split_seed\": {}, \"epochs\": {}, \"batch_size\": {},\n  \"architecture\": {:?}, \"quantization_shift\": {}, \"output_bits\": {}, \"lr_shift\": {}, \"momentum_shift\": {}, \"grad_clip_mantissa\": {},\n  \"train_size\": {}, \"validation_size\": {}, \"test_size\": {},\n  \"test_correct\": {}, \"test_accuracy_pct\": {:.6}, \"wall_ms\": {}, \"peak_rss_kb\": {},\n  \"learning_curve\": [{}]\n}}\n",
        cfg.smoke, cfg.seed, SPLIT_SEED, cfg.epochs, cfg.batch,
        dims, SHIFT, OUTPUT_BITS, LR_SHIFT, MOM_SHIFT, GRAD_CLIP,
        data.train_ids.len(), data.validation_ids.len(), data.test.len(),
        test_correct, pct(test_correct, data.test.len()), total_ms, rss,
        epochs.join(","),
    );
    if let Some(parent) = cfg.report.parent() { fs::create_dir_all(parent).expect("create report directory"); }
    fs::write(&cfg.report, payload).expect("write report JSON");
}
