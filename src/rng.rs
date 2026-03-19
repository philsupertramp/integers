//! XorShift64 — a fast, deterministic PRNG requiring no floating-point ops.
//!
//! Used for shuffle index generation during mini-batch training so that the
//! shuffling path is entirely integer-arithmetic, matching the spirit of the
//! dyadic training system.

/// A 64-bit XorShift PRNG.
///
/// Period: 2⁶⁴ − 1 (all non-zero 64-bit states are visited).
/// Seed must be non-zero.
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
