//! Reproducible split and sampling helpers for issue #15.
//! This module is shared by the experiment and its independent protocol tests.
use integers::rng::XorShift64;

/// Class-balanced, deterministic hold-out split. The validation quota is
/// applied to every class; the test set is NEVER passed to this function.
pub fn stratified_validation(
    labels: &[u8],
    per_class: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    assert!(seed != 0);
    assert!(per_class > 0);
    let classes = labels.iter().copied().max().map_or(0, |x| x as usize + 1);
    assert!(classes > 0, "empty labels");
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for class in 0..classes {
        let mut candidates: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label as usize == class).then_some(i))
            .collect();
        assert!(
            candidates.len() > per_class,
            "class {class} has too few samples for validation"
        );
        shuffle(
            &mut candidates,
            seed ^ (class as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15),
        );
        validation.extend_from_slice(&candidates[..per_class]);
        train.extend_from_slice(&candidates[per_class..]);
    }
    train.sort_unstable();
    validation.sort_unstable();
    (train, validation)
}

pub fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = XorShift64::new(seed.max(1));
    for i in (1..items.len()).rev() {
        let j = rng.gen_range((i + 1) as u32) as usize;
        items.swap(i, j);
    }
}

/// Separate RNG streams for per-seed/per-epoch sampling. Training never
/// consumes the validation/test split's RNG stream.
pub fn epoch_order(indices: &[usize], seed: u64, epoch: usize) -> Vec<usize> {
    let mut order = indices.to_vec();
    let mixed = seed ^ ((epoch as u64).wrapping_add(1)).wrapping_mul(0xD1B54A32D192ED03);
    shuffle(&mut order, mixed);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disjoint_balanced_and_complete() {
        let labels: Vec<u8> = (0..30).map(|i| (i % 3) as u8).collect();
        let (train, val) = stratified_validation(&labels, 2, 1776);
        assert_eq!(train.len(), 24);
        assert_eq!(val.len(), 6);
        for c in 0..3 {
            assert_eq!(val.iter().filter(|&&i| labels[i] == c).count(), 2);
        }
        let mut all = train.clone();
        all.extend(val.iter().copied());
        all.sort_unstable();
        assert_eq!(all, (0..30).collect::<Vec<_>>());
        assert_eq!((train, val), stratified_validation(&labels, 2, 1776));
    }
    #[test]
    fn independent_epochs_are_reproducible() {
        let source: Vec<usize> = (0..100).collect();
        assert_eq!(epoch_order(&source, 5, 7), epoch_order(&source, 5, 7));
        assert_ne!(epoch_order(&source, 5, 7), epoch_order(&source, 5, 8));
        assert_eq!(source, (0..100).collect::<Vec<_>>());
    }
    #[test]
    #[should_panic(expected = "too few samples")]
    fn refuses_empty_train_class() {
        stratified_validation(&[0, 0, 1], 1, 1);
    }
}
