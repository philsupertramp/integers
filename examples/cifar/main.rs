//! CIFAR-10 classification with a dyadic convolutional network.
//!
//! # Architecture
//!
//! ```text
//! Input: 3×32×32 (3072 flat, channel-first)
//!   Conv2D(3→16,  3×3)  →  16×30×30   ReLU   MaxPool(2) → 16×15×15
//!   Conv2D(16→32, 3×3)  →  32×13×13   ReLU   MaxPool(2) → 32×6×6
//!   Flatten → 1152
//!   Linear(1152→128)  ReLU
//!   Linear(128→10)    Softmax
//! ```
//!
//! # Usage
//! ```sh
//! cargo run --release --example cifar --features "parquet-support image-decode" \
//!     -- data/cifar10/train.parquet data/cifar10/test.parquet
//! ```
//!
//! # Expectation setting
//!
//! This is a substantially harder task than MNIST:
//! - 10 visually similar categories vs 10 well-separated digit shapes
//! - 3 colour channels, more texture variation, viewpoint/lighting changes
//! - Float networks with BatchNorm reach ~75–80%; we don't have BatchNorm
//!
//! With this engine (in the current state), expect **50–65%** on the full
//! test set with this shallow architecture and integer-only arithmetic.
//! The early-stop criterion of "5× 100% on 20 eval samples" fires at
//! ~80%+ model accuracy — 20 samples is a weak signal so the bar is
//! lower than it sounds. Watch the train accuracy trend to understand
//! convergence.
//!
//! **Compile with `--release`** — CIFAR-10 images are 4× larger than MNIST.

mod cifar_loader;

use std::path::PathBuf;

use crate::cifar_loader::load_cifar10_parquet;
use integers::data::shuffled_indices;
use integers::nn::{BatchNorm1D, BatchNorm2D, Conv2D, Dropout, Flatten, Linear, MaxPool2D, ReLU, Sequential, Softmax};
use integers::rng::XorShift64;
use integers::dyadic::Dyadic;
use integers::{argmax, cross_entropy_grad};

const CIFAR_CLASSES: [&str; 10] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

fn main() {
    // ── CLI ───────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let train_path = PathBuf::from(
        args.get(1).map(|s| s.as_str()).unwrap_or("data/cifar10/train.parquet"));
    let test_path = PathBuf::from(
        args.get(2).map(|s| s.as_str()).unwrap_or("data/cifar10/test.parquet"));

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const SHIFT:         u32   = 7;
    const LR_SHIFT:      u32   = 7;
    const MOM_SHIFT:     u32   = 1;
    const BATCH_SIZE:    usize = 32;
    const GRAD_CLIP:     i32   = 8192;
    const N_TRAIN:       usize = 256;
    const N_EVAL:        usize = 20;
    const MAX_EPOCHS:    u32   = 500;
    const STOP_PATIENCE: u32   = 5;

    // ── Load data ─────────────────────────────────────────────────────────────
    println!("Loading CIFAR-10 …");
    println!("  train: {}", train_path.display());
    let train = load_cifar10_parquet(&train_path)
        .unwrap_or_else(|e| { eprintln!("Train load failed: {e}"); std::process::exit(1); });

    println!("  test:  {}", test_path.display());
    let test = load_cifar10_parquet(&test_path)
        .unwrap_or_else(|e| { eprintln!("Test load failed: {e}"); std::process::exit(1); });

    let shift = train.input_shift.max(0) as u32;
    println!("  train pool: {} samples,  {} features",  train.len(), train.n_features());
    println!("  test  pool: {} samples,  {} features\n", test.len(),  test.n_features());

    // ── Model ─────────────────────────────────────────────────────────────────
    //
    // 3×32×32 → Conv(3,16,3×3) → 16×30×30 → MaxPool(2) → 16×15×15
    //         → Conv(16,32,3×3) → 32×13×13 → MaxPool(2) → 32×6×6
    //         → Flatten(1152) → Linear(128) → ReLU → Linear(10) → Softmax
    //
    // MaxPool(2,2) on 30×30: out = (30-2)/2+1 = 15  ✓
    // MaxPool(2,2) on 13×13: out = (13-2)/2+1 = 6   ✓
    // Flatten: 32 × 6 × 6 = 1152

    let mut model = Sequential::new();
    model.add(Conv2D::new(3, 16, 3, 3, 32, 32, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(BatchNorm2D::new(16, 30, 30, SHIFT));
    model.add(ReLU::new());
    model.add(MaxPool2D::new(16, 30, 30, 2, 2));  // → 16×15×15

    model.add(Conv2D::new(16, 32, 3, 3, 15, 15, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(BatchNorm2D::new(32, 13, 13, SHIFT));
    model.add(ReLU::new());
    model.add(MaxPool2D::new(32, 13, 13, 2, 2));  // → 32×6×6

    model.add(Flatten);
    model.add(Linear::new(1152, 128, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(BatchNorm1D::new(128, SHIFT));
    model.add(ReLU::new());
    model.add(Dropout::new(0.5));
    model.add(Linear::new(128, 10, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(Softmax::new(SHIFT));

    model.summary();
    println!();
    println!("  max_epochs={MAX_EPOCHS},  n_train={N_TRAIN},  batch={BATCH_SIZE}");
    println!("  n_eval={N_EVAL},  stop_patience={STOP_PATIENCE}× 100%");
    println!("  lr=2^(-{LR_SHIFT}),  momentum={MOM_SHIFT},  clip=2^{}",
        (GRAD_CLIP as f64).log2() as i32);
    println!();
    println!("  Note: target ~55–65% test accuracy with BatchNorm + Dropout.");
    println!("  The early-stop fires at model accuracy ~80%+ (20 eval samples");
    println!("  is a weak signal — watch train accuracy for convergence).\n");

    // ── Fixed eval subset ─────────────────────────────────────────────────────
    let mut eval_rng = XorShift64::new(0xcafebabe);
    let eval_indices: Vec<usize> = shuffled_indices(test.len())
        .into_iter().take(N_EVAL).collect();

    // ── Training ──────────────────────────────────────────────────────────────
    let mut train_rng     = XorShift64::new(1337);
    let mut perfect_streak = 0u32;

    println!("{:>6}  {:>10}  {:>10}  {:>14}",
        "Epoch", "Train Acc", "Eval Acc", "Streak (100%)");
    println!("{}", "─".repeat(48));

    let mut stopped_at = None;

    'training: for epoch in 0..MAX_EPOCHS {
        let train_indices: Vec<usize> = shuffled_indices(train.len())
            .into_iter().take(N_TRAIN).collect();

        let mut train_correct = 0usize;

        model.set_training(true);
        for batch in train_indices.chunks(BATCH_SIZE) {
            model.zero_grad();

            let batch_x: Vec<Vec<Dyadic>> = batch.iter()
                .map(|&i| train.get_input(i).data)
                .collect();
            let batch_t: Vec<Vec<Dyadic>> = batch.iter()
                .map(|&i| train.get_target(i).data)
                .collect();

            let batch_y = model.forward_batch(&batch_x);

            for (&idx, y) in batch.iter().zip(batch_y.iter()) {
                if argmax(y) == train.labels[idx] as usize { train_correct += 1; }
            }

            let batch_g: Vec<Vec<Dyadic>> = batch_y.iter().zip(batch_t.iter())
                .map(|(y, t)| cross_entropy_grad(y, t, shift))
                .collect();

            model.backward_batch(&batch_g);
            model.update(LR_SHIFT);
        }

        let train_acc = train_correct as f64 / N_TRAIN as f64 * 100.0;

        // ── Evaluate on the fixed eval subset ──────────────────────────────────
        model.set_training(false);
        let eval_correct: usize = eval_indices.iter().filter(|&&i| {
            let x = test.get_input(i).data;
            argmax(&model.forward(&x)) == test.labels[i] as usize
        }).count();
        let eval_acc = eval_correct as f64 / N_EVAL as f64 * 100.0;

        if eval_correct == N_EVAL { perfect_streak += 1; } else { perfect_streak = 0; }

        let streak_str = if perfect_streak > 0 { format!("{perfect_streak}") } else { "—".to_string() };
        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>14}", epoch, train_acc, eval_acc, streak_str);

        if perfect_streak >= STOP_PATIENCE {
            stopped_at = Some(epoch);
            break 'training;
        }
    }

    println!("{}", "─".repeat(48));
    match stopped_at {
        Some(e) => println!("\nEarly stop at epoch {e}."),
        None    => println!("\nReached {MAX_EPOCHS} epochs without triggering early stop."),
    }

    // ── Full test evaluation ───────────────────────────────────────────────────
    println!("\nFull test evaluation ({} samples) …", test.len());
    model.set_training(false);
    let mut conf = [[0usize; 10]; 10];
    for i in 0..test.len() {
        let x    = test.get_input(i).data;
        let pred = argmax(&model.forward(&x));
        conf[test.labels[i] as usize][pred] += 1;
    }
    let total_correct: usize = (0..10).map(|c| conf[c][c]).sum();
    println!("Full test accuracy: {}/{} = {:.2}%\n",
        total_correct, test.len(),
        total_correct as f64 / test.len() as f64 * 100.0);

    // Confusion matrix with class names.
    let max_name = CIFAR_CLASSES.iter().map(|n| n.len()).max().unwrap_or(8);
    print!("{:>width$}  ", "", width = max_name);
    for c in CIFAR_CLASSES { print!("{c:>6}"); }
    println!("   Acc");
    println!("  {}", "─".repeat(max_name + 2 + 10 * 6 + 7));
    for r in 0..10 {
        print!("{:>width$}  |", CIFAR_CLASSES[r], width = max_name);
        for c in 0..10 {
            if r == c { print!("\x1b[1m{:>6}\x1b[0m", conf[r][c]); }
            else      { print!("{:>6}", conf[r][c]); }
        }
        let row_total: usize = conf[r].iter().sum();
        println!("  {:.1}%", conf[r][r] as f64 / row_total as f64 * 100.0);
    }
}
