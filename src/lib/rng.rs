//! XorShift64 — a fast, deterministic PRNG requiring no floating-point ops.
//!
//! Used for shuffle index generation during mini-batch training so that the
//! shuffling path is entirely integer-arithmetic, matching the spirit of the
//! dyadic training system.

/// A 64-bit XorShift PRNG.
///
/// Period: 2^{64} − 1 (all non-zero 64-bit states are visited).
/// Seed must be non-zero.
use std::cell::Cell;

#[derive(Clone, Copy)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new generator with the given seed.
    ///
    /// # Panics
    /// Panics if `seed == 0` (degenerate fixed point for XorShift).
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "XorShift64 requires a non-zero seed");
        Self { state: seed }
    }

    /// Advance the state and return the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a uniform pseudo-random value in `[0, n)`.
    ///
    /// Uses rejection-free modular reduction (slight bias for large `n`,
    /// negligible for shuffle indices where `n ≪ 2⁶⁴`).
    #[inline]
    pub fn gen_range(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
}


// ─── Thread-local XorShift64 RNG ─────────────────────────────────────────────

// Seeded with a Weyl-sequence constant — guaranteed non-zero.
thread_local! {
    static RNG: Cell<XorShift64> = Cell::new(XorShift64::new(1337));
}

/// Advance the thread-local XorShift64 state and return the next `u64`.
#[inline(always)]
pub(crate) fn rng_next() -> u64 {
    RNG.with(|c| {
        let mut x = c.get();
        let val = x.next_u64();
        c.set(x);
        val
    })
}

#[inline(always)]
pub(crate) fn rng_range(n: u32) -> u32 {
    RNG.with(|c| {
        c.get().gen_range(n)
    })
}

/// Seed the thread-local RNG.  Call this at the start of each thread if you
/// need reproducible stochastic rounding (e.g. in tests).
pub fn seed_rng(s: u64) {
    assert!(s != 0, "XorShift64 requires a non-zero seed");
    RNG.with(|c| {
        let mut rng = c.get();
        rng.state = s;
        c.set(rng);
        rng
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_zero_output() {
        let mut rng = XorShift64::new(1);
        let mut saw_nonzero = false;
        for _ in 0..100 {
            if rng.next_u64() != 0 { saw_nonzero = true; break; }
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn gen_range_in_bounds() {
        let mut rng = XorShift64::new(42);
        for _ in 0..1000 {
            let v = rng.gen_range(10);
            assert!(v < 10);
        }
    }

    #[test]
    #[should_panic]
    fn zero_seed_panics() {
        XorShift64::new(0);
    }
}
