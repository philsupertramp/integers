use criterion::{black_box, criterion_group, criterion_main, Criterion};
use integers::{Dyadic, Tensor, nn::{ReLU, Module}};

fn bench_batch_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ReLU_Forward_Batch");
    
    // Create a realistically sized batch for an AI researcher:
    // Batch size 128, each item is a 64x64 feature map (4096 elements)
    // Total elements: 524,288
    let batch_size = 1024;
    let feature_size = 128 * 128;
    let dummy_data = vec![Dyadic { v: 1, s: 0 }; batch_size * feature_size];
    
    let batch = Tensor {
        data: dummy_data,
        shape: vec![batch_size, 128, 128],
    };

    let mut relu = ReLU::new();

    // TARGET 1: The new View way. Zero-copy iteration!
    group.bench_function("ZeroCopy_TensorView", |b| {
        b.iter(|| {
            let _result: Vec<Tensor> = black_box(&batch)
                .iter() 
                .map(|view_row| relu.forward(&view_row))
                .collect();
            black_box(_result);
        })
    });

    // TARGET 2: The batched way.
    group.bench_function("ZeroCopy_Tensor_batched", |b| {
        b.iter(|| {
            // black_box stops the compiler from optimizing our loop away
            let _result: Tensor = relu.forward_batch(black_box(&batch));
            black_box(_result);
        })
    });


    group.finish();
}

criterion_group!(benches, bench_batch_forward);
criterion_main!(benches);
