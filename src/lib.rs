//! Lattice-point decisions: closest-vector, sphere, and maximum-likelihood
//! decoding over the lattices [`lattica`] represents.
//!
//! # The boundary with `lattica`
//!
//! `lattica` computes facts about a lattice — representation, exact integer
//! linear algebra, orthogonalization, reduction, named constructions,
//! structural enumeration. This crate decides points on it: every entry point
//! receives a real target (or a code seam) and either selects a lattice point
//! — nearest, listed, or provably maximum-likelihood — or reports why it
//! cannot. [`closed`] has the `O(n)` closed-form decoders for the classical
//! families, [`babai`] the bounded-work approximations, [`enumerate`] the
//! budgeted Schnorr–Euchner searches, [`highdim`] the exact `BW_16` and
//! Leech maximum-likelihood decoders, [`modulo`] the quantization-error side,
//! and [`construction`] decodes Construction A lattices through the caller's
//! code.
//!
//! # The one type floating point is for
//!
//! The input is a received vector — a genuinely real quantity — and the
//! output is an exact lattice point, returned as scaled integers. That makes
//! this the crate where `f64` is the right type, and the only place in the
//! lattice family where cross-platform agreement has to be argued rather than
//!
//! The argument is structural, not statistical. Every decision in
//! [`closed`] uses exactly four operations: add, subtract, compare, and
//! round-to-nearest-integer. All four are correctly rounded and exactly
//! specified by IEEE-754, so two peers on different architectures compute
//! bit-identical results from bit-identical inputs. There is no accumulation
//! whose order could differ, no transcendental, and no fused multiply-add.
//! **Do not introduce one.** A `mul_add` on a decision path would be a
//! wire-format break that no unit test would catch.
//!
//! What floating point does *not* buy is exactness of the geometry: two
//! lattice points whose distances to `x` differ by less than one ulp are
//! ordered by the arithmetic rather than by the mathematics. That is inherent
//! to a real-valued input, it is confined to genuine Voronoi boundaries, and
//! it is deterministic — which is the property that matters for a codec.
//!
//! # Tie rules are format
//!
//! Two peers decoding the same boundary point must return the same lattice
//! point, so both ties in the algorithm are specified, not incidental:
//!
//! - **Rounding tie.** `x = k + 0.5` rounds *away from zero*. Chosen over
//!   ties-to-even because it makes `f(-x) == -f(x)`, so the decoders commute
//!   with negation — a property that is directly testable, unlike "the tie
//!   went the way we expected".
//! - **Worst-coordinate tie.** When several coordinates are equally far from
//!   an integer, the lowest index wins.
//! - **Coset tie.** [`DnPlus`] prefers the `D_n` coset when the two
//!   candidates are equidistant.
//!
//! Changing any of these is a format break, not a refactor.
//!
//! # Example
//!
//! ```
//! use lattice_engine::{Quantizer, Scratch, Zn};
//!
//! let q = Zn::new(4).unwrap();
//! let mut scratch = Scratch::new(4);
//! let mut out = [0i64; 4];
//! q.nearest(&[0.4, -0.6, 1.5, -2.5], &mut out, &mut scratch).unwrap();
//! // Half-way values round away from zero.
//! assert_eq!(out, [0, -1, 2, -3]);
//! # Ok::<(), lattice_engine::DecodeError>(())
//! ```

#![forbid(unsafe_code)]
pub mod babai;
pub mod closed;
pub mod construction;
pub mod enumerate;
pub mod highdim;
pub mod modulo;

pub use closed::{An, Dn, DnPlus, Zn, e8, round_nearest, round_nearest_flipped};
pub use construction::{CodeMembership, ConstructionA};
pub use enumerate::{
    EnumerationScratch, Enumerator, ListPoint, PreparedEnumerationScratch, PreparedEnumerator,
};
pub use highdim::{AmbientScratch, BarnesWall16, Leech24};
pub use modulo::{Scaled, mod_lattice, mod_lattice_dithered};

/// The decode error vocabulary, shared with the arithmetic below.
///
/// Raised for validation failures, exhausted budgets, and geometry that does
/// not describe a lattice. Rank or budget exhaustion is a typed outcome,
/// never an approximate answer.
pub use lattica::error::DecodeError;

/// Largest coordinate magnitude a decoder will accept.
///
/// Above `2^52` every `f64` is already an integer, and the doubled output of
/// [`DnPlus`] still fits `i64`. A larger input is rejected
/// rather than saturated.
pub const COORD_LIMIT: f64 = 4_503_599_627_370_496.0;

/// Reusable working buffers for the decoders.
///
/// Owned by the caller so that steady-state decoding allocates nothing
/// (invariant I7). One `Scratch` serves any lattice of at most its capacity,
/// and grows on demand.
#[derive(Debug, Clone, Default)]
pub struct Scratch {
    base: Vec<i64>,
    alt: Vec<i64>,
    delta: Vec<f64>,
    shifted: Vec<f64>,
    order: Vec<u32>,
    /// Lattice point in scaled integer coordinates, for `mod_lattice`.
    point: Vec<i64>,
    /// Rescaled input, for `Scaled`.
    divided: Vec<f64>,
    /// Per-symbol costs, for Construction A: `q` entries per coordinate.
    pub(crate) costs: Vec<f64>,
    /// Chosen residues, for Construction A.
    pub(crate) symbols: Vec<u32>,
}

impl Scratch {
    /// Allocates buffers for lattices of dimension up to `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            base: vec![0; dim],
            alt: vec![0; dim],
            delta: vec![0.0; dim],
            shifted: vec![0.0; dim],
            order: vec![0; dim],
            point: vec![0; dim],
            divided: vec![0.0; dim],
            costs: Vec::new(),
            symbols: vec![0; dim],
        }
    }

    /// Current capacity in coordinates.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.base.len()
    }

    /// Grows to `dim` if needed. A no-op once warm.
    pub(crate) fn ensure(&mut self, dim: usize) {
        if self.base.len() >= dim {
            return;
        }
        self.base.resize(dim, 0);
        self.alt.resize(dim, 0);
        self.delta.resize(dim, 0.0);
        self.shifted.resize(dim, 0.0);
        self.order.resize(dim, 0);
        self.point.resize(dim, 0);
        self.divided.resize(dim, 0.0);
        self.symbols.resize(dim, 0);
    }

    /// Grows the per-symbol cost table to `dim * symbols` entries.
    pub(crate) fn ensure_costs(&mut self, dim: usize, symbols: usize) {
        let needed = dim * symbols;
        if self.costs.len() < needed {
            self.costs.resize(needed, 0.0);
        }
    }
}

/// A closed-form nearest-point decoder for one lattice.
pub trait Quantizer {
    /// Dimension of the ambient space, which is the length of both the input
    /// and the output.
    fn dim(&self) -> usize;

    /// Denominator of the output coordinates.
    ///
    /// [`nearest`](Quantizer::nearest) writes `scale * v`, where `v` is the
    /// lattice point. It is 1 for lattices inside `Z^n` and 2 for
    /// [`DnPlus`], whose points are half-integers. Returning
    /// scaled integers rather than floats keeps the answer exact: a lattice
    /// point is a discrete object and should not come back as a rounding
    /// question.
    fn scale(&self) -> i64;

    /// Writes `scale * v` for the lattice point `v` nearest to `x`.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::LengthMismatch`] if either slice is not [`dim`](Quantizer::dim) long.
    /// - [`DecodeError::NonFinite`] if any coordinate is NaN or infinite.
    /// - [`DecodeError::Range`] if any coordinate exceeds [`COORD_LIMIT`].
    ///
    /// A rejected call leaves `out` untouched.
    fn nearest(&self, x: &[f64], out: &mut [i64], scratch: &mut Scratch)
    -> Result<(), DecodeError>;
}

/// A shared reference to a decoder is a decoder.
///
/// Lets a caller wrap a borrowed quantizer — `Scaled::new(&q, k)` — without
/// giving up ownership of the inner one.
impl<Q: Quantizer + ?Sized> Quantizer for &Q {
    fn dim(&self) -> usize {
        (**self).dim()
    }

    fn scale(&self) -> i64 {
        (**self).scale()
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        (**self).nearest(x, out, scratch)
    }
}

/// Checks that an input and an output slice both match the lattice dimension.
fn validate_lengths(input: usize, output: usize, dim: usize) -> Result<(), DecodeError> {
    if input != dim {
        return Err(DecodeError::LengthMismatch {
            expected: dim,
            found: input,
        });
    }
    if output != dim {
        return Err(DecodeError::LengthMismatch {
            expected: dim,
            found: output,
        });
    }
    Ok(())
}

/// Validates lengths and coordinate values before anything is written.
pub(crate) fn validate(x: &[f64], out: &[i64], dim: usize) -> Result<(), DecodeError> {
    validate_lengths(x.len(), out.len(), dim)?;
    for (index, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(DecodeError::NonFinite { index });
        }
        if v.abs() > COORD_LIMIT {
            return Err(DecodeError::Range(lattica::error::RangeError::Overflow {
                op: lattica::error::Op::Mul,
                width_bits: 64,
            }));
        }
    }
    Ok(())
}

/// Decodes a contiguous run of received vectors.
///
/// `points` and `out` are flat and strided by `q.dim()`. A flat buffer is the
/// primitive rather than a slice of slices because it is what a vector kernel
/// can walk without a pointer chase; callers holding scattered symbols can
/// adapt on their side.
///
/// # Errors
///
/// As [`Quantizer::nearest`], plus [`DecodeError::LengthMismatch`] if the
/// buffers are not whole multiples of the dimension or disagree in length. The
/// vectors are decoded in order, so an error leaves earlier outputs written and
/// later ones untouched; the failing vector itself is not written.
pub fn nearest_batch<Q: Quantizer + ?Sized>(
    q: &Q,
    points: &[f64],
    out: &mut [i64],
    scratch: &mut Scratch,
) -> Result<(), DecodeError> {
    let dim = q.dim();
    if dim == 0 || !points.len().is_multiple_of(dim) || points.len() != out.len() {
        return Err(DecodeError::LengthMismatch {
            expected: dim,
            found: points.len(),
        });
    }
    for (src, dst) in points.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        q.nearest(src, dst, scratch)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{COORD_LIMIT, Quantizer, Scratch, nearest_batch};
    use crate::closed::{Dn, Zn};
    use lattica::error::DecodeError;

    #[test]
    fn rejected_input_leaves_the_output_untouched() {
        let q = Zn::new(3).unwrap();
        let mut scratch = Scratch::new(3);
        let mut out = [7i64; 3];

        assert_eq!(
            q.nearest(&[1.0, f64::NAN, 3.0], &mut out, &mut scratch),
            Err(DecodeError::NonFinite { index: 1 })
        );
        assert_eq!(out, [7, 7, 7]);

        assert_eq!(
            q.nearest(&[1.0, f64::INFINITY, 3.0], &mut out, &mut scratch),
            Err(DecodeError::NonFinite { index: 1 })
        );
        assert_eq!(out, [7, 7, 7]);

        assert!(
            q.nearest(&[1.0, COORD_LIMIT * 2.0, 3.0], &mut out, &mut scratch)
                .is_err()
        );
        assert_eq!(out, [7, 7, 7]);

        assert_eq!(
            q.nearest(&[1.0, 2.0], &mut out, &mut scratch),
            Err(DecodeError::LengthMismatch {
                expected: 3,
                found: 2
            })
        );
        assert_eq!(out, [7, 7, 7]);
    }

    #[test]
    fn scratch_grows_once_and_then_stays_put() {
        // `Zn` needs no scratch at all; `Dn` does.
        let mut scratch = Scratch::default();
        assert_eq!(scratch.capacity(), 0);
        let q = Dn::new(4).unwrap();
        let mut out = [0i64; 4];
        q.nearest(&[0.1, 0.2, 0.3, 0.4], &mut out, &mut scratch)
            .unwrap();
        assert!(scratch.capacity() >= 4);
        let grown = scratch.capacity();
        q.nearest(&[0.9, 0.8, 0.7, 0.6], &mut out, &mut scratch)
            .unwrap();
        assert_eq!(scratch.capacity(), grown);
    }

    #[test]
    fn batch_matches_the_single_point_path() {
        let q = Zn::new(2).unwrap();
        let mut scratch = Scratch::new(2);
        let points = [0.4, -0.6, 1.5, 2.5, -1.5, -2.5];
        let mut batched = [0i64; 6];
        nearest_batch(&q, &points, &mut batched, &mut scratch).unwrap();

        let mut single = [0i64; 6];
        for (src, dst) in points.chunks_exact(2).zip(single.chunks_exact_mut(2)) {
            q.nearest(src, dst, &mut scratch).unwrap();
        }
        assert_eq!(batched, single);
        // Half-way values round away from zero.
        assert_eq!(batched, [0, -1, 2, 3, -2, -3]);
    }

    #[test]
    fn a_ragged_batch_is_rejected() {
        let q = Zn::new(2).unwrap();
        let mut scratch = Scratch::new(2);
        let mut out = [0i64; 3];
        assert!(nearest_batch(&q, &[1.0, 2.0, 3.0], &mut out, &mut scratch).is_err());
    }
}
