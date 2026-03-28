//! MNIST digit classification with a dyadic convolutional network.
//!
//! Architecture: 784 -> 128 ReLU -> 128 -> ReLU -> 10
//!
//! # Training regime
//!
//! - Train on samples once per epoch, shuffle dataset before training
//! - evaluate the model on the full testing set each epoch
//! - Early stopping: halt once 100% eval accuracy is sustained for
//!   `STOP_PATIENCE` consecutive epochs
//! - Hard cap at 500 epochs
//!
//! # Usage
//! ```sh
//! cargo run --release --example mnist
//! cargo run --release --example mnist -- data/mnist
//! cargo run --release --example mnist --features parquet-support
//! ```
mod mnist_loader;
use crate::mnist_loader::load_mnist_auto;

use std::path::PathBuf;
use std::time::Instant;
use std::io::{self, Write}; // Added for the progress bar

use integers::data::shuffled_indices;
use integers::nn::{Linear, ReLU, Sequential};
use integers::rng::XorShift64;
use integers::dyadic::Dyadic;
use integers::{argmax, mse_grad};

fn main() {
    // ── CLI ───────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let data_dir: PathBuf = args.get(1).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/mnist"));

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const SHIFT:         u32   = 7;
    const BITS_OUT:      u32   = 31;
    const LR_SHIFT:      u32   = 7;    // lr = 2^(-7) ≈ 0.0078
    const MOM_SHIFT:     u32   = 1;    // momentum decay = 0.5
    const BATCH_SIZE:    usize = 32;
    const GRAD_CLIP:     i32   = 8192; // 2^13
    const N_TRAIN:       usize = 60000;  // training samples drawn per epoch
    const N_EVAL:        usize = 10000;   // test samples evaluated per epoch
    const MAX_EPOCHS:    u32   = 500;
    const STOP_PATIENCE: u32   = 5;    // consecutive 100% eval epochs before stopping

    // ── Load data ─────────────────────────────────────────────────────────────
    println!("Loading MNIST from '{}' …", data_dir.display());
    let (train, test) = load_mnist_auto(&data_dir, true)
        .unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1); });

    let shift = train.input_shift.max(0) as u32;
    println!("  train pool: {} samples", train.len());
    println!("  test  pool: {} samples\n", test.len());

    // ── Model ─────────────────────────────────────────────────────────────────
    let mut model = Sequential::new();
    model.add(Linear::new(784, 128, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(Linear::new(128, 128, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(Linear::new(128, 10, SHIFT, SHIFT, BITS_OUT)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));

    model.summary();
    println!();
    println!("  max_epochs={MAX_EPOCHS},  n_train={N_TRAIN},  batch={BATCH_SIZE}");
    println!("  n_eval={N_EVAL},  stop_patience={STOP_PATIENCE} consecutive 100%");
    println!("  lr=2^(-{LR_SHIFT}),  momentum_shift={MOM_SHIFT},  \
              grad_clip=2^{}", (GRAD_CLIP as f64).log2() as i32);
    println!();

    // ── Fix the eval subset once so progress is comparable across epochs ───────
    // Using a separate RNG seeded independently so the eval set doesn't depend
    // on how many training shuffles have been performed.
    let eval_indices: Vec<usize> = shuffled_indices(test.len())
        .into_iter().take(N_EVAL).collect();

    // ── Training loop ─────────────────────────────────────────────────────────
    let mut perfect_streak = 0u32;

    println!("{:>6}  {:>10}  {:>10}  {:>14}  {:>10}",
        "Epoch", "Train Acc", "Eval Acc", "Streak (100%)", "Time");
    println!("{}", "─".repeat(60));

    let mut stopped_at = None;

    'training: for epoch in 0..MAX_EPOCHS {
        let epoch_start = Instant::now();

        // ── Sample N_TRAIN indices from the training pool ──────────────────────
        let train_indices: Vec<usize> = shuffled_indices(train.len())
            .into_iter().take(N_TRAIN).collect();

        // ── Mini-batch SGD ─────────────────────────────────────────────────────
        let mut train_correct = 0usize;
        let chunks = train_indices.chunks(BATCH_SIZE);
        let total_batches = chunks.len();

        model.set_training(true);
        for (b_idx, batch) in chunks.enumerate() {
            model.zero_grad();

            // Assemble batch tensors.
            let batch_x: Vec<Vec<Dyadic>> = batch.iter()
                .map(|&i| train.get_input(i).data)
                .collect();
            let batch_t: Vec<Vec<Dyadic>> = batch.iter()
                .map(|&i| train.get_target(i).data)
                .collect();

            // Forward (BatchNorm sees all N samples at once).
            let batch_y = model.forward_batch(&batch_x);

            for (n, (&idx, y)) in batch.iter().zip(batch_y.iter()).enumerate() {
                if argmax(y) == train.labels[idx] as usize { train_correct += 1; }
                let _ = n;
            }

            // Compute loss gradients and backward.
            let batch_g: Vec<Vec<Dyadic>> = batch_y.iter().zip(batch_t.iter())
                .map(|(y, t)| mse_grad(y, t))
                .collect();

            model.backward_batch(&batch_g);
            model.update(LR_SHIFT);

            // Inline progress bar update (every 5 batches or on the very last batch)
            if b_idx % 5 == 0 || b_idx + 1 == total_batches {
                let progress = (b_idx + 1) as f64 / total_batches as f64;
                let bar_width = 25;
                let filled = (progress * bar_width as f64) as usize;
                print!(
                    "\r      [{}{}] {:>5.1}%",
                    "█".repeat(filled),
                    " ".repeat(bar_width - filled),
                    progress * 100.0
                );
                let _ = io::stdout().flush();
            }
        }

        let train_acc = train_correct as f64 / N_TRAIN as f64 * 100.0;

        // Clear the progress bar from the line before printing epoch evaluation
        print!("\r{:60}\r", "");

        // ── Evaluate on the fixed eval subset ──────────────────────────────────
        model.set_training(false);
        let eval_correct: usize = eval_indices.iter().filter(|&&i| {
            let x = test.get_input(i).data;
            argmax(&model.forward(&x)) == test.labels[i] as usize
        }).count();

        let eval_acc = eval_correct as f64 / N_EVAL as f64 * 100.0;

        let elapsed = epoch_start.elapsed();
        let time_str = format!("{:.2}s", elapsed.as_secs_f64());

        // ── Early stopping logic ───────────────────────────────────────────────
        if eval_correct == N_EVAL {
            perfect_streak += 1;
        } else {
            perfect_streak = 0;
        }

        let streak_str = if perfect_streak > 0 {
            format!("{perfect_streak}")
        } else {
            "—".to_string()
        };

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>14}  {:>10}",
            epoch, train_acc, eval_acc, streak_str, time_str);

        if perfect_streak >= STOP_PATIENCE {
            stopped_at = Some(epoch);
            break 'training;
        }
    }

    // ── Final report ──────────────────────────────────────────────────────────
    println!("{}", "─".repeat(60));
    match stopped_at {
        Some(e) => println!(
            "\nEarly stop at epoch {e}: {STOP_PATIENCE} consecutive epochs \
             with 100% on the {N_EVAL}-sample eval set."
        ),
        None => println!(
            "\nReached {MAX_EPOCHS} epochs without sustaining {STOP_PATIENCE}× \
             100% on eval."
        ),
    }

    // ── Final full-test evaluation ─────────────────────────────────────────────
    println!("\nRunning final evaluation on the full test set ({} samples) …", test.len());
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

    println!("Confusion matrix (rows = true label, cols = predicted):");
    print!("     ");
    for c in 0..10 { print!("{c:>5}"); }
    println!("   ← predicted");
    println!("     {}", "─".repeat(50));
    for r in 0..10 {
        print!("{r:>3}  |");
        for c in 0..10 {
            if r == c { print!("\x1b[1m{:>5}\x1b[0m", conf[r][c]); }
            else      { print!("{:>5}", conf[r][c]); }
        }
        let row_total: usize = conf[r].iter().sum();
        println!("  {:.1}%", conf[r][r] as f64 / row_total as f64 * 100.0);
    }
}
