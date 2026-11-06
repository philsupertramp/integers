// Include the same split/sampling algorithm used by the executable.
// Its #[cfg(test)] fixtures verify stratification, disjointness, and seeds.
#[path = "../experiments/protocol.rs"]
mod protocol;
