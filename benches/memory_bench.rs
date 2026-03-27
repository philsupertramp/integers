use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use integers::{Tensor, Dyadic, nn::{Module, ReLU}};

// ==========================================
// 1. THE MEMORY TRACKER
// ==========================================

/// A custom allocator that intercepts all memory requests and counts them.
struct TrackingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

pub fn reset_memory_counters() {
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    ALLOCATIONS.store(0, Ordering::SeqCst);
}

pub fn get_memory_stats() -> (usize, usize) {
    (
        ALLOCATED_BYTES.load(Ordering::SeqCst),
        ALLOCATIONS.load(Ordering::SeqCst),
    )
}


fn main() {
    let batch_size = 128;
    let feature_size = 64 * 64; // 4096 items per row (16 KB)
    
    // Allocate the input batch
    let batch = Tensor {
        data: vec![Dyadic{v: 1, s: 0}; batch_size * feature_size],
        shape: vec![batch_size, 64, 64],
    };

    let mut relu = ReLU::new();

    println!("Starting Memory Profiling...\n");

    // --- TEST 1: UNBATCHED (Returns Vec<Tensor>) ---
    reset_memory_counters();
    
    let _unbatched_result: Vec<Tensor> = batch.iter()
        .map(|view| relu.forward(view))
        .collect();
    
    let (unbatched_bytes, unbatched_count) = get_memory_stats();
    
    println!("❌ UNBATCHED (List of individual Tensors):");
    println!("   Total Allocations: {} times", unbatched_count);
    println!("   Memory Allocated:  {} bytes ({} MB)\n", unbatched_bytes, unbatched_bytes / 1_000_000);


    // --- TEST 2: BATCHED (Returns single Tensor using optimized FromIterator) ---
    // Drop the previous result to clean up, though it doesn't strictly affect our counters
    drop(_unbatched_result); 
    reset_memory_counters();
    
    let _batched_result: Tensor = batch.iter()
        .map(|view| relu.forward(view))
        .collect();
    
    let (batched_bytes, batched_count) = get_memory_stats();

    println!("✅ BATCH-WISE USING VIEWS:");
    println!("   Total Allocations: {} times", batched_count);
    println!("   Memory Allocated:  {} bytes ({} MB)", batched_bytes, batched_bytes / 1_000_000);
}
