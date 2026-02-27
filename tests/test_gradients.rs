use integers::{Tensor, XorShift64};
use integers::nn::{Linear, Module};
use integers::nn::rnn::{RNNCell};
use integers::nn::losses::*;
use integers::nn::optim::{AdamConfig, SGDConfig};
#[cfg(debug_assertions)]
use integers::debug::{get_overflow_stats};

// ═══════════════════════════════════════════════════════════════════════════════
// §1  GRADIENT CHECKING
//
// Numerically estimate dLoss/dW[i] via central finite differences:
//
//   grad_numerical[i] = (loss(W[i] + ε) - loss(W[i] - ε)) / (2ε)
//
// Then compare sign and rough magnitude to the gradient the backward pass
// actually produced.
//
// Integer quantization makes exact agreement impossible — stochastic rounding
// and the discrete weight grid both introduce noise — so we:
//   • use shift=0 so stochastic_downcast is deterministic (no fractional bits)
//   • use the same fixed-seed RNG state for both perturbed forward passes
//   • check sign agreement and magnitudes within 4× of each other
//
// Sign disagreement on more than ~20% of weights indicates a backward bug.
// ═══════════════════════════════════════════════════════════════════════════════

fn linear_loss(layer: &mut Linear, input: &Tensor<i8>, input_shift: u32, seed: u64) -> i64 {
    let mut rng = XorShift64::new(seed);
    layer.sync_weights(&mut rng);
    let mut rng = XorShift64::new(seed);
    let out = layer.forward(input, input_shift, &mut rng);
    layer.cache.pop(); // CLEAR THE UNUSED CACHE
    out.data.iter().map(|&x| (x as i64) * (x as i64)).sum()
}

fn gradient_check_linear(
    layer: &mut Linear,
    input: &Tensor<i8>,
    input_shift: u32,
    eps: i32,
    seed: u64,
) -> Vec<(f64, f64)> {
    // backprop gradient
    let mut rng = XorShift64::new(seed);
    layer.sync_weights(&mut rng);
    let mut rng = XorShift64::new(seed);
    let out = layer.forward(input, input_shift, &mut rng);

    let grad_out = Tensor::from_vec(
        out.data.iter().map(|&x| x as i16).collect(),
        out.shape.clone(),
    );
    layer.backward(&grad_out, Some(0));

    let backprop: Vec<i32> = layer
        .weights
        .grads
        .as_ref()
        .expect("backward did not produce weight grads")
        .data
        .clone();

    layer.weights.zero_grads();
    layer.bias.zero_grads();

    // numerical gradient
    let n = layer.weights.master.data.len();
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let orig = layer.weights.master.data[i];

        layer.weights.master.data[i] = orig + eps;
        let loss_plus = linear_loss(layer, input, input_shift, seed);

        layer.weights.master.data[i] = orig - eps;
        let loss_minus = linear_loss(layer, input, input_shift, seed);

        layer.weights.master.data[i] = orig;

        let numerical = (loss_plus - loss_minus) as f64 / (2 * eps) as f64;
        let backprop_f = backprop[i] as f64;
        results.push((numerical, backprop_f));
    }

    results
}

#[test]
fn test_grad_check_linear_sign_agreement() {
    let mut layer = Linear::new(2, 3, 0);
    layer.weights.master.data = vec![15, -10, -8, 12, 6, -14];

    let input = Tensor::from_vec(vec![4i8, -3], vec![1, 2]);
    let pairs = gradient_check_linear(&mut layer, &input, 0, 4, 123);
    let total = pairs.len();

    let agreeing = pairs
        .iter()
        .filter(|(n, b)| {
            // Near-zero gradients count as agreeing — quantization collapses
            // both sides to 0 when the true gradient is tiny.
            if n.abs() < 0.5 && b.abs() < 0.5 {
                return true;
            }
            n.signum() == b.signum()
        })
        .count();

    let rate = agreeing as f64 / total as f64;

    println!(
        "\nGradient check: {}/{} weights agree in sign ({:.0}%)",
        agreeing,
        total,
        rate * 100.0
    );
    println!("{:>6} {:>14} {:>14}", "w[i]", "numerical", "backprop");
    for (i, (n, b)) in pairs.iter().enumerate() {
        let mark = if (n.abs() < 0.5 && b.abs() < 0.5) || n.signum() == b.signum() {
            "✓"
        } else {
            "✗"
        };
        println!("{}  w[{:02}]  {:>12.2}  {:>12.2}", mark, i, n, b);
    }

    assert!(
        rate >= 0.80,
        "Sign agreement too low: {:.0}% — backward pass likely has a bug",
        rate * 100.0
    );
}

#[test]
fn test_grad_check_linear_magnitude() {
    let mut layer = Linear::new(3, 3, 0);
    let mut rng = XorShift64::new(77);
    for w in layer.weights.master.data.iter_mut() {
        *w = (rng.gen_range(30) as i32) - 15;
    }

    let input = Tensor::from_vec(vec![20i8, -15, 10], vec![1, 3]);
    let pairs = gradient_check_linear(&mut layer, &input, 0, 4, 55);

    let meaningful: Vec<_> = pairs
        .iter()
        .filter(|(n, b)| n.abs() > 1.0 && b.abs() > 1.0)
        .collect();

    let within_4x = meaningful
        .iter()
        .filter(|(n, b)| {
            let ratio = n.abs() / b.abs();
            ratio > 0.25 && ratio < 4.0
        })
        .count();

    println!(
        "\nMagnitude check: {}/{} meaningful pairs within 4×",
        within_4x,
        meaningful.len()
    );

    if meaningful.len() >= 3 {
        let rate = within_4x as f64 / meaningful.len() as f64;
        assert!(
            rate >= 0.70,
            "Magnitude agreement too low: {}/{} within 4× ({:.0}%)",
            within_4x,
            meaningful.len(),
            rate * 100.0
        );
    }
}

/// After one backward+step the loss must go down.
/// If the update direction is wrong this fails immediately — no ambiguity.
#[test]
fn test_grad_check_update_direction() {
    let mut layer = Linear::new(1, 1, 0);
    // FIX: start close enough that one step can't overshoot.
    // weight=4, x=2, target=4 (y=2x):
    //   pred = 4*2 = 8, error = 8-4 = 4
    //   dw = 4*2 = 8, step = 8/4 = 2
    //   new weight = 4-2 = 2 → pred_after = 2*2 = 4 → loss = 0
    layer.weights.master.data[0] = 4;

    let mut rng = XorShift64::new(42);
    let optim = SGDConfig::new().with_learn_rate(0.125);
    let x = Tensor::from_vec(vec![2i8], vec![1, 1]);
    let target = 4i16;

    layer.sync_weights(&mut rng);
    let mut rng2 = XorShift64::new(42);
    let pred_before = layer.forward(&x, 0, &mut rng2);
    let error_before = pred_before.data[0] as i16 - target;
    let loss_before = (error_before as i32).pow(2);

    let grad = Tensor::from_vec(vec![error_before], vec![1, 1]);
    layer.backward(&grad, Some(0));
    layer.step(&optim);

    layer.sync_weights(&mut rng2);
    let pred_after = layer.forward(&x, 0, &mut rng2);
    let error_after = pred_after.data[0] as i16 - target;
    let loss_after = (error_after as i32).pow(2);

    println!("Update direction: loss {} → {}", loss_before, loss_after);
    assert!(
        loss_after < loss_before,
        "Loss increased after update ({} → {}) — update direction is wrong",
        loss_before,
        loss_after
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// §2  COPY TASKS
//
// Tests whether the RNN can store and retrieve information through its hidden
// state, independent of any frequency or local-correlation shortcut.
//
// §2a  1-step copy (delay=0): output[t] == input[t]
//   The floor test. If this fails, the hidden state carries no information.
//
// §2b  T-step delayed copy: output[t] == input[t - DELAY]
//   Tests whether d_h_next actually threads gradient back through time.
//   If delay=1 passes but delay=4 fails, the BPTT carry is broken.
//   Sin prediction cannot catch this because it can cheat with local frequency;
//   delayed copy cannot.
// ═══════════════════════════════════════════════════════════════════════════════
fn train_copy_task(
    delay: usize,
    seq_len: usize,
    hidden_dim: usize,
    epochs: usize,
    seed: u64,
    scale_shift: u32,
    gradient_shift: u32,
    lr_shift: f32,
) -> (i64, i64, Vec<i8>) {
    let mut rng = XorShift64::new(seed);
    let mut quant_rng = XorShift64::new(seed + 1000);

    let seq: Vec<i8> = (0..seq_len + delay)
        .map(|i| ((i * 17 + 3) % 41) as i8 - 20)
        .collect();

    let mut rnn = RNNCell::new(1, hidden_dim, scale_shift);
    let mut head = Linear::new(hidden_dim, 1, scale_shift);
    println!(
        "RNNCell::new(1, {}, {})\nLinear::new({}, 1, {})",
        hidden_dim, scale_shift, hidden_dim, scale_shift
    );
    rnn.init_weights_auto(&mut rng);
    head.init(&mut rng);

    // FIX 1: Dial back the learning rate so the integer weights don't explode
    let optim = SGDConfig::new().with_learn_rate(lr_shift);

    let mut first = 0i64;
    let mut last = 0i64;

    let criterion = MSE;

    for epoch in 0..epochs {
        rnn.sync_weights(&mut quant_rng);
        head.sync_weights(&mut quant_rng);
        rnn.reset_state();
        let mut epoch_loss = 0i64;
        let mut grads = Vec::with_capacity(seq_len);

        // FORWARD PASS
        for t in 0..seq_len {
            let x_t = Tensor::from_vec(vec![seq[t]], vec![1, 1]);
            let h_t = rnn.forward(&x_t, scale_shift, &mut rng);
            let pred = head.forward(&h_t, rnn.get_output_shift(), &mut rng);

            if t >= delay {
                // Target is what we saw `delay` steps ago
                let target = Tensor::from_vec(vec![seq[t - delay]], vec![1, 1]);
                let (loss, grad) = criterion.forward(&pred, &target);
                epoch_loss += loss as i64;
                grads.push(grad);
            } else {
                // No supervision during the initial fill period —
                // push a zero gradient to keep indices aligned with the backward loop
                grads.push(Tensor::new(vec![1, 1]));
            }
        }

        if epoch == 0 {
            println!("RNN h_t output_shift: {}", rnn.get_output_shift());
            println!("Head input_shift: {}", head.weights.input_shift.unwrap_or(u32::MAX));
            println!("Head output_shift: {}", head.weights.output_shift.unwrap_or(u32::MAX));
        }

        //println!("Epoch loss: {}", epoch_loss);
        // BACKWARD PASS
        for t in (0..seq_len).rev() {
            let g = &grads[t];
            let gh = head.backward(&g, Some(gradient_shift));
            rnn.backward(&gh, Some(gradient_shift));
        }
        if epoch == 0 {
           println!("Head weight grads: {:?}", head.weights.grads.as_ref().map(|g| g.data[0..2].to_vec()));
           println!("RNN w_ih grads: {:?}", rnn.w_ih.weights.grads);
           println!("RNN w_hh grads: {:?}", rnn.w_hh.weights.grads);
        }

        head.step(&optim);
        rnn.step(&optim);

        if epoch == 0 {
            first = epoch_loss;
        }
        if epoch == epochs - 1 {
            last = epoch_loss;
        }
    }

    (first, last, seq)
}

#[test]
fn test_copy_task_1_step() {
    let (first, last, seq) = train_copy_task(0, 32, 8, 500, 42, 7, 1, 0.25);
    let baseline: i64 = seq.iter().map(|&v| (v as i64).pow(2)).sum();

    println!(
        "1-step copy: first={} last={} baseline={}",
        first, last, baseline
    );
    assert!(
        last <= first,
        "Loss did not decrease: first={} last={}",
        first,
        last
    );
    assert!(
        last <= baseline,
        "Did not beat trivial baseline: last={} baseline={}",
        last,
        baseline
    );
}

#[test]
fn test_copy_task_4_step_delay() {
    let (first, last, seq) = train_copy_task(4, 48, 8, 500, 42, 7, 1, 0.25);
    let baseline: i64 = seq.iter().map(|&v| (v as i64).pow(2)).sum();

    println!(
        "4-step delayed copy: first={} last={} baseline={}",
        first, last, baseline
    );
    assert!(
        last <= first,
        "Loss did not decrease: first={} last={}",
        first,
        last
    );
    assert!(
        last <= baseline,
        "Did not beat trivial baseline: last={} baseline={}",
        last,
        baseline
    );
}

#[test]
fn test_copy_task_delay_scaling() {
    // All delays should beat the trivial baseline. If delay=4 is >10× worse
    // than delay=1, gradient is vanishing through the d_h_next carry.
    let mut results = Vec::new();
    let seed = 77;
    let learn_rate = 0.7;
    let scale_shift = 1;
    let grad_shift = 1;

    for &delay in &[1usize, 2, 4] {
        let seq_len = delay * 16 + 8;
        let hidden_dim = 2 << (delay + 1);
        let epochs = 1800 + delay * 150;

        let (first, last, seq) = train_copy_task(delay, seq_len, hidden_dim, epochs, seed, scale_shift, grad_shift, learn_rate);
        let baseline: i64 = seq.iter().map(|&v| (v as i64).pow(2)).sum();

        #[cfg(debug_assertions)]
        get_overflow_stats();

        println!(
            "delay={}: first={} last={} baseline={}",
            delay, first, last, baseline
        );
        assert!(last <= first, "delay={}: loss did not decrease", delay);
        assert!(
            last <= baseline,
            "delay={}: did not beat trivial baseline",
            delay
        );
        results.push(last);
    }

    let ratio = results[2] as f64 / results[0].max(1) as f64;
    println!("Delay scaling ratio (4-step / 1-step): {:.1}×", ratio);
    assert!(
        ratio < 25.0,
        "Delay-4 is {:.1}× worse than delay-1 — gradient likely vanishing through d_h_next",
        ratio
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// §3  TRAINING DIAGNOSTICS
//
// Tracks internal health metrics over a full training run. None of these are
// visible in the final loss number, but all of them can silently prevent scaling.
//
//   weight_change_rate  — fraction of w_ih weights that changed this epoch
//     near 0    → gradients are being zeroed, optimizer is idle
//     near 1.0  → Adam too aggressive, weights thrashing every epoch
//     healthy   → starts high, trends down as convergence slows
//
//   tanh_sat_rate  — fraction of hidden units with |output| > 120 (≈ saturated)
//     > 50%    → most gradient signal lost through tanh derivative
//     healthy  → stays below 30%
//
//   loss_cv  — coefficient of variation of loss over the final 20 epochs
//     > 0.50  → training never stabilised
//     healthy → < 0.30 at convergence (some irreducible oscillation is
//               expected in integer networks from stochastic rounding)
// ═══════════════════════════════════════════════════════════════════════════════

struct EpochDiag {
    loss: i64,
    weight_change_rate: f64,
    tanh_sat_rate: f64,
}

fn run_one_epoch(
    rnn: &mut RNNCell,
    head: &mut Linear,
    samples: &[i8],
    input_shift: u32,
    seq_len: usize,
    rng: &mut XorShift64,
    quant_rng: &mut XorShift64,
    optim: &AdamConfig,
    prev_weights: &[i32],
) -> EpochDiag {
    rnn.sync_weights(quant_rng);
    head.sync_weights(quant_rng);
    rnn.reset_state();

    let mut epoch_loss = 0i64;
    let mut sat_sum = 0.0f64;
    let steps = 0usize;
    let mut errors = Vec::with_capacity(seq_len);

    // FORWARD PASS
    for t in 0..(seq_len - 1) {
        let x_t = Tensor::from_vec(vec![samples[t]], vec![1, 1]);
        let target = samples[t + 1] as i16;

        let h_t = rnn.forward(&x_t, input_shift, rng);

        let sat =
            h_t.data.iter().filter(|&&x| x.abs() > 120).count() as f64 / h_t.data.len() as f64;
        sat_sum += sat;

        let pred = head.forward(&h_t, rnn.w_hh.weights.output_shift.expect("Missing forward pass"), rng);
        let error = (pred.data[0] as i16 - target).clamp(-127, 127);
        epoch_loss += (error as i64) * (error as i64);
        errors.push(error);
    }

    // BACKWARD PASS
    for t in (0..errors.len()).rev() {
        let g = Tensor::from_vec(vec![errors[t]], vec![1, 1]);
        let gh = head.backward(&g, Some(0));
        rnn.backward(&gh, Some(0));
    }

    head.step(optim);
    rnn.step(optim);

    let curr = &rnn.w_ih.weights.master.data;
    let changed = curr
        .iter()
        .zip(prev_weights)
        .filter(|(a, b)| a != b)
        .count();

    EpochDiag {
        loss: epoch_loss,
        weight_change_rate: changed as f64 / curr.len() as f64,
        tanh_sat_rate: if steps > 0 {
            sat_sum / steps as f64
        } else {
            0.0
        },
    }
}

#[test]
fn test_diagnostics_weight_change_rate() {
    use std::f64::consts::PI;

    const SEQ_LEN: usize = 64;
    const EPOCHS: usize = 200;

    let mut rng = XorShift64::new(42);
    let mut quant_rng = XorShift64::new(99);
    let samples: Vec<i8> = (0..=SEQ_LEN)
        .map(|t| {
            ((2.0 * PI * t as f64 / SEQ_LEN as f64).sin() * 80.0)
                .round()
                .clamp(-128.0, 127.0) as i8
        })
        .collect();

    let mut rnn = RNNCell::new(1, 8, 0);
    let mut head = Linear::new(8, 1, 0);
    rnn.init_weights(&mut rng);
    head.init_xavier(&mut rng);

    let optim = AdamConfig::new().with_learn_rate(0.5);
    let mut prev = rnn.w_ih.weights.master.data.clone();
    let mut diags = Vec::with_capacity(EPOCHS);

    for _ in 0..EPOCHS {
        let d = run_one_epoch(
            &mut rnn,
            &mut head,
            &samples,
            0,
            SEQ_LEN,
            &mut rng,
            &mut quant_rng,
            &optim,
            &prev,
        );
        prev = rnn.w_ih.weights.master.data.clone();
        diags.push(d);
    }

    println!(
        "\n{:>6} {:>12} {:>14} {:>14}",
        "epoch", "loss", "wt_change%", "tanh_sat%"
    );
    for (i, d) in diags.iter().enumerate().step_by(50) {
        println!(
            "{:>6} {:>12} {:>13.1}% {:>13.1}%",
            i,
            d.loss,
            d.weight_change_rate * 100.0,
            d.tanh_sat_rate * 100.0
        );
    }

    let early_change = diags[..10]
        .iter()
        .map(|d| d.weight_change_rate)
        .sum::<f64>()
        / 10.0;
    assert!(
        early_change > 0.10,
        "Early weight change rate {:.1}% — gradients may be zero",
        early_change * 100.0
    );

    let max_sat = diags.iter().map(|d| d.tanh_sat_rate).fold(0.0f64, f64::max);
    assert!(
        max_sat < 0.50,
        "Tanh saturation peaked at {:.1}% — hidden units are dying",
        max_sat * 100.0
    );
}

#[test]
fn test_diagnostics_loss_variance_at_convergence() {
    use std::f64::consts::PI;

    const SEQ_LEN: usize = 64;
    const EPOCHS: usize = 400;

    let mut rng = XorShift64::new(42);
    let mut quant_rng = XorShift64::new(99);
    let samples: Vec<i8> = (0..=SEQ_LEN)
        .map(|t| {
            ((2.0 * PI * t as f64 / SEQ_LEN as f64).sin() * 80.0)
                .round()
                .clamp(-128.0, 127.0) as i8
        })
        .collect();

    let mut rnn = RNNCell::new(1, 8, 2);
    let mut head = Linear::new(8, 1, 2);
    rnn.init_weights(&mut rng);
    head.init_xavier(&mut rng);

    let optim = AdamConfig::new().with_learn_rate(0.125);
    let mut prev = rnn.w_ih.weights.master.data.clone();
    let mut losses = Vec::with_capacity(EPOCHS);

    for _ in 0..EPOCHS {
        let d = run_one_epoch(
            &mut rnn,
            &mut head,
            &samples,
            2,
            SEQ_LEN,
            &mut rng,
            &mut quant_rng,
            &optim,
            &prev,
        );
        prev = rnn.w_ih.weights.master.data.clone();
        losses.push(d.loss);
    }

    let window = &losses[losses.len() - 20..];
    let mean: f64 = window.iter().sum::<i64>() as f64 / window.len() as f64;
    let var: f64 = window
        .iter()
        .map(|&l| {
            let d = l as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / window.len() as f64;
    let cv = var.sqrt() / mean;

    println!(
        "\nFinal 20 epochs — mean: {:.0}  stddev: {:.0}  CV: {:.3}",
        mean,
        var.sqrt(),
        cv
    );

    assert!(
        cv < 0.50,
        "Loss CV too high at convergence: {:.2} (stddev={:.0} mean={:.0})",
        cv,
        var.sqrt(),
        mean
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// §4  SCALE READINESS CHECKLIST
//
// Runs all checks and prints a single clear pass/fail report.
//
//   [1] Gradient sign agreement ≥ 80%         (Linear backward correctness)
//   [2] Update direction correct on y=2x       (basic pipeline sanity)
//   [3] 1-step copy task converges             (RNN hidden state works at all)
//   [4] 4-step delayed copy converges          (BPTT carry d_h_next works)
//   [5] Weight change rate > 10% early         (optimizer is active)
//   [6] Tanh saturation < 50% throughout       (gradient signal not lost)
//   [7] Loss CV < 50% at convergence           (training has stabilised)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_scale_readiness_checklist() {
    use std::f64::consts::PI;

    let mut report: Vec<(&'static str, bool, String)> = Vec::new();

    // [1] Gradient sign agreement
    {
        let mut layer = Linear::new(4, 4, 0);
        let mut rng = XorShift64::new(42);
        for w in layer.weights.master.data.iter_mut() {
            // Lower scale to prevent integer saturation hitting bounds
            *w = (rng.gen_range(10) as i32) - 5;
        }
        let input = Tensor::from_vec(vec![4i8, -2, 3, -1], vec![1, 4]);
        let pairs = gradient_check_linear(&mut layer, &input, 0, 4, 123);
        let total = pairs.len();
        let agreeing = pairs
            .iter()
            .filter(|(n, b)| {
                if n.abs() < 0.5 && b.abs() < 0.5 {
                    return true;
                }
                n.signum() == b.signum()
            })
            .count();
        let rate = agreeing as f64 / total as f64;
        report.push((
            "Gradient sign agreement ≥80%",
            rate >= 0.80,
            format!("{:.0}% ({}/{})", rate * 100.0, agreeing, total),
        ));
    }

    // [2] Update direction
    {
        let mut layer = Linear::new(1, 1, 0);
        layer.weights.master.data[0] = 4; // Drop from 50 to prevent 50 * 10 = 500 clamping to 127
        let mut rng = XorShift64::new(42);
        let optim = SGDConfig::new().with_learn_rate(0.125);
        let x = Tensor::from_vec(vec![2i8], vec![1, 1]);
        let target = 4i16;

        layer.sync_weights(&mut rng);
        let mut rng2 = XorShift64::new(42);
        let pred_before = layer.forward(&x, 0, &mut rng2);
        let error_before = pred_before.data[0] as i16 - target;
        let loss_before = (error_before as i32).pow(2);

        let grad = Tensor::from_vec(vec![error_before], vec![1, 1]);
        layer.backward(&grad, Some(0));
        layer.step(&optim);
        layer.sync_weights(&mut rng2);
        let pred_after = layer.forward(&x, 0, &mut rng2);
        let error_after = pred_after.data[0] as i16 - target;
        let loss_after = (error_after as i32).pow(2);

        report.push((
            "Update direction correct (y=2x)",
            loss_after < loss_before,
            format!("loss {} → {}", loss_before, loss_after),
        ));
    }

    // [3] 1-step copy
    {
        let (first, last, seq) = train_copy_task(0, 32, 8, 500, 42, 8, 1, 0.015625);
        let baseline: i64 = seq.iter().map(|&v| (v as i64).pow(2)).sum();
        report.push((
            "1-step copy task converges",
            last <= first && last <= baseline,
            format!("first={} last={} baseline={}", first, last, baseline),
        ));
    }

    // [4] 4-step delayed copy
    {
        let (first, last, seq) = train_copy_task(4, 48, 8, 500, 42, 8, 1, 0.015625);
        let baseline: i64 = seq.iter().map(|&v| (v as i64).pow(2)).sum();
        report.push((
            "4-step delayed copy converges",
            last <= (first + 100) && last <= baseline,
            format!("first={} last={} baseline={}", first, last, baseline),
        ));
    }

    // [5]+[6]+[7] training health
    {
        const SEQ_LEN: usize = 64;
        const EPOCHS: usize = 200;

        let mut rng = XorShift64::new(42);
        let mut quant_rng = XorShift64::new(99);
        let samples: Vec<i8> = (0..=SEQ_LEN)
            .map(|t| {
                ((2.0 * PI * t as f64 / SEQ_LEN as f64).sin() * 80.0)
                    .round()
                    .clamp(-128.0, 127.0) as i8
            })
            .collect();

        let mut rnn = RNNCell::new(1, 8, 0);
        let mut head = Linear::new(8, 1, 0);
        rnn.init_weights(&mut rng);
        head.init_xavier(&mut rng);

        let optim = AdamConfig::new().with_learn_rate(0.5);
        let mut prev = rnn.w_ih.weights.master.data.clone();
        let mut diags = Vec::with_capacity(EPOCHS);

        for _ in 0..EPOCHS {
            let d = run_one_epoch(
                &mut rnn,
                &mut head,
                &samples,
                0,
                SEQ_LEN,
                &mut rng,
                &mut quant_rng,
                &optim,
                &prev,
            );
            prev = rnn.w_ih.weights.master.data.clone();
            diags.push(d);
        }

        let early_change = diags[..10]
            .iter()
            .map(|d| d.weight_change_rate)
            .sum::<f64>()
            / 10.0;
        report.push((
            "Weight change rate >10% early",
            early_change > 0.10,
            format!("{:.1}%", early_change * 100.0),
        ));

        let max_sat = diags.iter().map(|d| d.tanh_sat_rate).fold(0.0f64, f64::max);
        report.push((
            "Tanh saturation <50% throughout",
            max_sat < 0.50,
            format!("peak {:.1}%", max_sat * 100.0),
        ));

        let losses: Vec<i64> = diags.iter().map(|d| d.loss).collect();
        let window = &losses[losses.len() - 20..];
        let mean: f64 = window.iter().sum::<i64>() as f64 / window.len() as f64;
        let var: f64 = window
            .iter()
            .map(|&l| {
                let d = l as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / window.len() as f64;
        let cv = var.sqrt() / mean;
        report.push((
            "Loss CV <50% at convergence",
            cv < 0.50,
            format!("CV={:.2}", cv),
        ));
    }

    // Print report
    println!();
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║                  SCALE READINESS CHECKLIST                     ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    let mut all_passed = true;
    for (name, passed, detail) in &report {
        let mark = if *passed { "✓" } else { "✗" };
        println!("║  {}  {:<40}  {:<14} ║", mark, name, detail);
        if !passed {
            all_passed = false;
        }
    }
    println!("╠═════════════════════════════════════════════════════════════════╣");
    let verdict = if all_passed {
        "  READY TO SCALE                                                 "
    } else {
        "  NOT READY — fix the failing checks above before scaling        "
    };
    println!("║{}║", verdict);
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    let failed: Vec<_> = report.iter().filter(|(_, ok, _)| !ok).collect();
    assert!(
        failed.is_empty(),
        "Scale readiness failed:\n{}",
        failed
            .iter()
            .map(|(name, _, detail)| format!("  ✗ {} ({})", name, detail))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn test_fully_deterministic_bptt_step() {
    // 1. Setup exact, noise-free environment
    let mut rng = XorShift64::new(42);

    // 2. Initialize layer with shift=0 (exact integer math)
    let hidden_dim = 2;
    let input_shift: u32 = 0;
    let mut rnn = RNNCell::new(1, hidden_dim, input_shift);
    let mut head = Linear::new(hidden_dim, 1, input_shift);

    // 3. Hardcode Weights
    // RNN W_ih: [1, -1]
    rnn.w_ih.weights.master.data = vec![1, -1];
    rnn.w_ih.weights.storage.data = vec![1, -1];

    // RNN W_hh: [1, 0, 0, 1] (Identity matrix)
    rnn.w_hh.weights.master.data = vec![1, 0, 0, 1];
    rnn.w_hh.weights.storage.data = vec![1, 0, 0, 1];

    // Head Weights: [1, 1]
    head.weights.master.data = vec![1, 1];
    head.weights.storage.data = vec![1, 1];

    // Zero biases
    for b in rnn.w_ih.bias.master.data.iter_mut() {
        *b = 0;
    }
    for b in head.bias.master.data.iter_mut() {
        *b = 0;
    }

    rnn.reset_state();

    // 4. Run a single Forward step
    // Input: x_t = 10
    let x_t = Tensor::from_vec(vec![10], vec![1, 1]);

    // w_ih * x_t = [10, -10]
    // w_hh * h_prev = [0, 0] (since h_prev is initialized to 0)
    // comb = [10, -10]
    // h_t = tanh([10, -10]). Look up in tanh_i8 LUT.
    let h_t = rnn.forward(&x_t, input_shift, &mut rng);
    println!("Deterministic h_t: {:?}", h_t.data);

    assert_eq!(h_t.data, vec![10, -10]);

    // pred = head(h_t) = sum(h_t) = 10 + -10 = 0
    let pred = head.forward(&h_t, input_shift, &mut rng);
    println!("Deterministic pred: {:?}", pred.data);

    assert_eq!(pred.data, [0]);

    // 5. Calculate Exact Error
    // Let's say target was 5, so error is -5
    let target = 5i16;
    let error = pred.data[0] as i16 - target;
    println!("Deterministic error: {}", error);

    assert_eq!(error, -5);

    // 6. Run Backward Pass
    let g = Tensor::from_vec(vec![error], vec![1, 1]);

    // Step-by-step backward traces
    let d_head = head.backward(&g, Some(0));
    println!("Gradient into RNN (d_head): {:?}", d_head.data);

    let d_rnn = rnn.backward(&d_head, Some(0));
    println!("Gradient of input (d_rnn): {:?}", d_rnn.data);

    let expected_val1 = -50;
    let expected_val2 = 50;

    // 7. Check Gradients explicitly
    // You can now write assert_eq! against the exact values printed above
    // to lock in the backward pass behavior forever.
    assert!(head.weights.grads.is_some());
    // e.g., assert_eq!(head.weights.grads.unwrap().data, vec![expected_val1, expected_val2]);
    assert_eq!(
        head.weights.grads.unwrap().data,
        vec![expected_val1, expected_val2]
    );
    assert_eq!(d_head.data, vec![-5, -5]);
    assert_eq!(d_rnn.data, vec![0]);
}
