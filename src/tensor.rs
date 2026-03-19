//! Generic tensor type and the `Scalar` trait that quantised-data pipelines
//! depend on.
//!
//! The only concrete `Scalar` implementation shipped here is `i32`, which is
//! the natural storage type for dyadic integers.  Adding a new precision (e.g.
//! `i8` for a low-bitwidth experiment) just requires one more `impl Scalar`.

// ─── Scalar trait ─────────────────────────────────────────────────────────────

/// Trait for types that can serve as dataset element values.
///
/// Implementors must be `Clone + Copy + Default + PartialOrd` so that tensors
/// can be zeroed and elements can be compared (for argmax etc.).
pub trait Scalar: Clone + Copy + Default + PartialOrd + std::fmt::Debug + Send + Sync {
    /// Construct a value from a quantised `i32` mantissa (e.g. from a
    /// per-column min-max quantiser).
    fn from_quantized(v: i32) -> Self;

    /// Transform the raw quantisation shift produced by a quantiser into
    /// whatever form the `Dataset` stores it as.
    ///
    /// For `i32` this is the identity; a hypothetical `f32` scalar type might
    /// return `0` and absorb the scale into the values themselves.
    fn dataset_input_shift(shift: i32) -> i32;
}

// ─── i32 implementation ────────────────────────────────────────────────────────

impl Scalar for i32 {
    /// `i32` *is* the quantised form — pass through unchanged.
    #[inline] fn from_quantized(v: i32) -> Self { v }
    /// The shift is stored verbatim in `Dataset::input_shift`.
    #[inline] fn dataset_input_shift(shift: i32) -> i32 { shift }
}

// ─── Tensor ───────────────────────────────────────────────────────────────────

/// A row-major n-dimensional array of `Scalar` elements.
///
/// Indexing convention: `shape = [d₀, d₁, …, dₙ]`, element `[i₀, i₁, …, iₙ]`
/// lives at `data[i₀·(d₁·…·dₙ) + i₁·(d₂·…·dₙ) + … + iₙ]`.
#[derive(Clone, Debug)]
pub struct Tensor<S: Scalar> {
    pub data:  Vec<S>,
    pub shape: Vec<usize>,
}

impl<S: Scalar> Tensor<S> {
    /// Construct from a flat buffer and a shape.  Panics in debug mode if the
    /// buffer length does not match the product of the shape dimensions.
    pub fn from_vec(data: Vec<S>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor::from_vec: buffer length {} ≠ shape product {}",
            data.len(), shape.iter().product::<usize>(),
        );
        Self { data, shape }
    }

    /// Allocate an all-zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { data: vec![S::default(); n], shape }
    }

    /// Total number of elements.
    #[inline] pub fn len(&self) -> usize { self.data.len() }

    /// `true` if the tensor has no elements.
    #[inline] pub fn is_empty(&self) -> bool { self.data.is_empty() }
}
