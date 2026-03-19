//! Quick smoke-test: train a dyadic integer network on XOR.
//! For the real examples run `cargo run --example iris` or `cargo run --example mnist`.

use integers::{argmax, mse_grad, sample_to_dyadic, target_to_dyadic, Dyadic};
use integers::nn::{Linear, ReLU, Sequential};

const SHIFT: u32 = 7;

fn enc(x: f64) -> Dyadic { Dyadic::new((x * 128.0).round() as i32, SHIFT) }

fn main() {
    let data: Vec<(Vec<Dyadic>, Vec<Dyadic>)> = vec![
        (vec![enc( 1.0), enc( 1.0)], vec![enc(-1.0)]),
        (vec![enc( 1.0), enc(-1.0)], vec![enc( 1.0)]),
        (vec![enc(-1.0), enc( 1.0)], vec![enc( 1.0)]),
        (vec![enc(-1.0), enc(-1.0)], vec![enc(-1.0)]),
    ];

    let mut model = Sequential::new();
    model.add(Linear::new(2, 8, SHIFT, SHIFT, 31).with_grad_clip(8192));
    model.add(ReLU::new());
    model.add(Linear::new(8, 1, SHIFT, SHIFT, 31).with_grad_clip(8192));
    model.summary();

    for epoch in 0..300 {
        let mut sq_err = 0i64;
        for (x, t) in &data {
            model.zero_grad();
            let y   = model.forward(x);
            let g   = mse_grad(&y, t);
            sq_err += g.iter().map(|d| (d.v as i64).pow(2)).sum::<i64>();
            model.backward(&g);
            model.update(5);
        }
        if epoch % 50 == 0 || epoch == 299 {
            let mse = (sq_err as f64 / data.len() as f64) * 2f64.powi(-2 * SHIFT as i32);
            println!("epoch {epoch:>3}  mse = {mse:.4}");
        }
    }

    println!("\nFinal predictions:");
    for (x, t) in &data {
        let y = model.forward(x);
        println!("  [{:5.2}, {:5.2}] → {:6.3}  (target {:5.2})",
            x[0].to_f64(), x[1].to_f64(), y[0].to_f64(), t[0].to_f64());
    }
}
