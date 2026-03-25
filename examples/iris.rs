//! Iris classification with a dyadic integer network.
//!
//! Architecture: 4 → 8 → ReLU → 8 → ReLU → 3 → Softmax
//! Loss:         Cross-entropy  (gradient = softmax_out − one_hot_target)
//! Optimizer:    SGD with momentum (momentum_shift = 1, lr_shift = 7)
//!
//! This matches the architecture reported in the blog post that achieves 95%
//! accuracy on i32, surpassing the f32 baseline of 90%.
//!
//! # Usage
//! ```sh
//! cargo run --release --example iris
//! cargo run --release --example iris -- path/to/iris.csv
//! ```
//!
//! Expects the UCI Iris CSV (no header, label in column 4):
//! ```text
//! 5.1,3.5,1.4,0.2,Iris-setosa
//! 4.9,3.0,1.4,0.2,Iris-setosa
//! …
//! ```

use std::time::Instant;

use integers::data::shuffled_indices;
use integers::data::dataset_loaders::{DatasetBuilder, QuantizationMethod, FileFormat};
use integers::nn::{Linear, ReLU, Sequential, Softmax};
use integers::{argmax, cross_entropy_grad, TrainingReporter};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let train_path = args.get(1).map(|s| s.as_str()).unwrap_or("data/iris_train.tsv");
    let test_path = args.get(1).map(|s| s.as_str()).unwrap_or("data/iris_test.tsv");

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const SHIFT:     u32 = 7;
    const LR_SHIFT:  u32 = 7;    // lr = 2^(-7) ≈ 0.0078
    const MOM_SHIFT: u32 = 1;    // momentum decay = 2^(-1) = 0.5
    const GRAD_CLIP: i32 = 8192; // 2^13
    const EPOCHS:    u32 = 500;
    const LOG_EVERY: u32 = 50;
    const BITS_OUT: u32 = 31;

    // ── Load dataset ──────────────────────────────────────────────────────────
    println!("Loading Iris train dataset from '{train_path}' …");
    let ds = DatasetBuilder::new_csv(train_path)
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::MinMax)
        .load()
        .unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            eprintln!("Download: https://archive.ics.uci.edu/ml/datasets/iris");
            std::process::exit(1);
        });
    println!("Loading Iris test dataset from '{test_path}' …");
    let test = DatasetBuilder::new_csv(test_path)
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::MinMax)
        .load()
        .unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            eprintln!("Download: https://archive.ics.uci.edu/ml/datasets/iris");
            std::process::exit(1);
        });

    let shift = ds.input_shift.max(0) as u32;
    println!("  {} samples,  {} features,  {} classes  (input_shift={shift})\n",
        ds.len(), ds.n_features(), ds.n_classes);

    // ── Model ─────────────────────────────────────────────────────────────────
    //  4 → Linear(8) → ReLU → Linear(8) → ReLU → Linear(3) → Softmax
    let mut model = Sequential::new();
    model.add(Linear::new(ds.n_features(), 8, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(Linear::new(8, 8, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(Linear::new(8, ds.n_classes, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    //model.add(Softmax::new(SHIFT));
    model.summary();
    println!();

    // ── Training loop ─────────────────────────────────────────────────────────
    let mut reporter = TrainingReporter::new(EPOCHS, LOG_EVERY, shift);
    reporter.print_header();

    let total_start = Instant::now();

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        
        reporter.reset();

        // Online SGD (batch_size = 1), shuffled each epoch.
        for &i in &shuffled_indices(ds.len()) {
            let x = ds.get_input(i).data;
            let t = ds.get_target(i).data;

            model.zero_grad();
            let y = model.forward(&x);

            let correct = argmax(&y) == ds.labels[i] as usize;
            let g = cross_entropy_grad(&y, &t, shift);
            let sq = g.iter().map(|d| (d.v as i64).pow(2)).sum::<i64>();

            reporter.record(sq, correct);
            model.backward(&g);
            model.update(LR_SHIFT);
        }
        
        // Evaluate on the test set at the end of each epoch.
        let test_correct: usize = (0..test.len()).filter(|&i| {
            let x = test.get_input(i).data;
            argmax(&model.forward(&x)) == test.labels[i] as usize
        }).count();
        let test_acc = test_correct as f64 / test.len() as f64 * 100.0;

        let epoch_elapsed = epoch_start.elapsed();

        reporter.maybe_print(epoch, Some(test_acc));
        
        if epoch % LOG_EVERY == 0 || epoch == EPOCHS - 1 {
            println!("  └─ Epoch {} duration: {:.4}s", epoch, epoch_elapsed.as_secs_f64());
        }
    }

    let total_elapsed = total_start.elapsed();

    // ── Final evaluation ──────────────────────────────────────────────────────
    let correct: usize = (0..ds.len()).filter(|&i| {
        let x = ds.get_input(i).data;
        argmax(&model.forward(&x)) == ds.labels[i] as usize
    }).count();

    println!();
    reporter.print_footer();
    println!();
    println!("Total training time: {:.4}s", total_elapsed.as_secs_f64());
    println!("Final pass accuracy: {}/{} = {:.1}%",
        correct, ds.len(), correct as f64 / ds.len() as f64 * 100.0);

    // ── Per-class breakdown ───────────────────────────────────────────────────
    let nc = ds.n_classes;
    let mut per_class = vec![(0usize, 0usize); nc]; // (correct, total)
    for i in 0..ds.len() {
        let x   = ds.get_input(i).data;
        let y   = model.forward(&x);
        let cls = ds.labels[i] as usize;
        per_class[cls].1 += 1;
        if argmax(&y) == cls { per_class[cls].0 += 1; }
    }
    println!();
    println!("Per-class accuracy:");
    for (c, (ok, total)) in per_class.iter().enumerate() {
        println!("  class {c}: {ok}/{total} = {:.1}%", *ok as f64 / *total as f64 * 100.0);
    }
}
