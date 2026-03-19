//! MNIST digit classification with a dyadic convolutional network.
//!
//! # Architecture
//!
//! ```text
//! Input: 1×28×28 (784 flat)
//!   Conv2D(1→4, 3×3)  →  4×26×26 = 2704   ReLU   MaxPool(2) → 4×13×13 = 676
//!   Conv2D(4→8, 3×3)  →  8×11×11 = 968    ReLU   MaxPool(2) → 8×5×5   = 200
//!   Flatten
//!   Linear(200→64)  ReLU
//!   Linear(64→10)   Softmax
//! ```
//!
//! # Training regime
//!
//! - 256 randomly sampled training images per epoch, in 8 batches of 32
//! - 20 randomly sampled test images evaluated each epoch
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

use std::path::PathBuf;

use integers::data::shuffled_indices;
use integers::mnist_loader::load_mnist_auto;
use integers::nn::{Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential, Softmax};
use integers::rng::XorShift64;
use integers::{argmax, cross_entropy_grad, sample_to_dyadic, target_to_dyadic};

fn main() {
    // ── CLI ───────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let data_dir: PathBuf = args.get(1).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/mnist"));

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const SHIFT:         u32   = 7;
    const LR_SHIFT:      u32   = 7;    // lr = 2^(-7) ≈ 0.0078
    const MOM_SHIFT:     u32   = 1;    // momentum decay = 0.5
    const BATCH_SIZE:    usize = 32;
    const GRAD_CLIP:     i32   = 8192; // 2^13
    const N_TRAIN:       usize = 256;  // training samples drawn per epoch
    const N_EVAL:        usize = 20;   // test samples evaluated per epoch
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
    model.add(Conv2D::new(1, 4, 3, 3, 28, 28, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(MaxPool2D::new(4, 26, 26, 2, 2));

    model.add(Conv2D::new(4, 8, 3, 3, 13, 13, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(MaxPool2D::new(8, 11, 11, 2, 2));

    model.add(Flatten);
    model.add(Linear::new(200, 64, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(ReLU::new());
    model.add(Linear::new(64, 10, SHIFT, SHIFT, 32)
        .with_grad_clip(GRAD_CLIP).with_momentum(MOM_SHIFT));
    model.add(Softmax::new(SHIFT));

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
    let mut eval_rng  = XorShift64::new(0xdeadbeef);
    let eval_indices: Vec<usize> = shuffled_indices(test.len(), &mut eval_rng)
        .into_iter().take(N_EVAL).collect();

    // ── Training loop ─────────────────────────────────────────────────────────
    let mut train_rng     = XorShift64::new(1337);
    let mut perfect_streak = 0u32;

    println!("{:>6}  {:>10}  {:>10}  {:>14}",
        "Epoch", "Train Acc", "Eval Acc", "Streak (100%)");
    println!("{}", "─".repeat(48));

    let mut stopped_at = None;

    'training: for epoch in 0..MAX_EPOCHS {

        // ── Sample N_TRAIN indices from the training pool ──────────────────────
        let train_indices: Vec<usize> = shuffled_indices(train.len(), &mut train_rng)
            .into_iter().take(N_TRAIN).collect();

        // ── Mini-batch SGD ─────────────────────────────────────────────────────
        let mut train_correct = 0usize;

        for batch in train_indices.chunks(BATCH_SIZE) {
            model.zero_grad();

            for &i in batch {
                let x = sample_to_dyadic(&train.get_input(i).data,  shift);
                let t = target_to_dyadic(&train.get_target(i).data, shift);

                let y = model.forward(&x);
                if argmax(&y) == train.labels[i] as usize { train_correct += 1; }

                let g = cross_entropy_grad(&y, &t, shift);
                model.backward(&g);
            }
            model.update(LR_SHIFT);
        }

        let train_acc = train_correct as f64 / N_TRAIN as f64 * 100.0;

        // ── Evaluate on the fixed eval subset ──────────────────────────────────
        let eval_correct: usize = eval_indices.iter().filter(|&&i| {
            let x = sample_to_dyadic(&test.get_input(i).data, shift);
            argmax(&model.forward(&x)) == test.labels[i] as usize
        }).count();

        let eval_acc = eval_correct as f64 / N_EVAL as f64 * 100.0;

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

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>14}",
            epoch, train_acc, eval_acc, streak_str);

        if perfect_streak >= STOP_PATIENCE {
            stopped_at = Some(epoch);
            break 'training;
        }
    }

    // ── Final report ──────────────────────────────────────────────────────────
    println!("{}", "─".repeat(48));
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
    let mut conf = [[0usize; 10]; 10];
    for i in 0..test.len() {
        let x    = sample_to_dyadic(&test.get_input(i).data, shift);
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
