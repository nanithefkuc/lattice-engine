//! Construction A decoding: the code↔lattice bridge.
//!
//! # The seam
//!
//! This crate never owns a code. Construction A needs one, so it arrives
//! through [`CodeMembership`]: the caller answers membership and decoding
//! questions about residues, and the engine supplies the lattice decision
//! built around those answers. No field type crosses the boundary, which is
//! why this crate does not depend on `fgf` or `sgraph`.
//!
//! The *generator* construction (`construction_a_basis`) stays in
//! `lattica::construct` as code-free integer arithmetic; this module owns the
//! decodable side: the seam, the [`ConstructionA`] view of a code, and its
//! nearest-point decode.
//!
//! ## Why the seam carries costs, not residues
//!
//! A hard-decision seam — `decode(&self, residues)` — is not enough. The
//! nearest point of `Λ = qZ^n + lift(C)` to a real `x` is
//!
//! ```text
//! min over c in C of  sum_i dist(x_i, lift(c_i) + qZ)^2
//! ```
//!
//! which is a *soft* decoding problem with a per-symbol metric. Handing the
//! code only a rounded residue vector throws away exactly the information
//! that decides the answer, and yields a bounded-distance decoder wearing a
//! maximum-likelihood label — a correctness bug whose only symptom is a
//! slightly worse error rate. So [`CodeMembership::decode_costs`] takes the
//! metric, and [`ConstructionA`] is maximum-likelihood whenever the caller's
//! decoder is.

// Construction A converts between residues, real coordinates, and integer
// lattice points on every symbol. Each cast is on a value bounded by the
// modulus or by an already-validated coordinate.
#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use core::num::NonZeroU32;

use crate::{Quantizer, Scratch, round_away};
use lattica::error::{DecodeError, LatticeError, Op, RangeError};
use lattica::zq::Zq;

/// A linear code over `Z_q`, supplied by the caller.
///
/// Implementors own the code, its field or ring, and its decoder. The engine
/// only asks questions.
pub trait CodeMembership {
    /// The modulus `q`.
    fn modulus(&self) -> NonZeroU32;

    /// The code length `n`, which is the lattice dimension.
    fn length(&self) -> usize;

    /// The number of codewords, used to compute the lattice covolume.
    fn cardinality(&self) -> u64;

    /// Is this residue vector a codeword?
    fn contains(&self, residues: &[u32]) -> bool;

    /// Writes the codeword minimizing `Σ_i costs[i * q + c_i]`.
    ///
    /// `costs` is row-major with `q` entries per coordinate. An implementation
    /// that minimizes exactly makes [`ConstructionA`] a maximum-likelihood
    /// decoder; one that does not makes it bounded-distance, and should say so.
    ///
    /// # Errors
    ///
    /// Implementation-defined; [`DecodeError::LengthMismatch`] when the buffers
    /// do not match the code's geometry.
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError>;
}

/// The Construction A lattice `Λ = q·Z^n + lift(C)`.
///
/// Its covolume is `q^n / |C|`, so a `[n, k]` code over `Z_q` gives `q^(n-k)`.
/// The generator matrix of the lattice comes from
/// `lattica::construct::construction_a_basis`; this type is the decodable
/// view of the same lattice.
#[derive(Debug, Clone)]
pub struct ConstructionA<C> {
    code: C,
    zq: Zq,
}

impl<C: CodeMembership> ConstructionA<C> {
    /// Wraps a code as a lattice.
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadModulus`] for `q < 2`, and
    /// [`LatticeError::Degenerate`] for a zero-length code.
    pub fn new(code: C) -> Result<Self, LatticeError> {
        if code.length() == 0 {
            return Err(LatticeError::Degenerate);
        }
        let zq = Zq::new(code.modulus())?;
        Ok(Self { code, zq })
    }

    /// The code this lattice is built from.
    pub const fn code(&self) -> &C {
        &self.code
    }

    /// The covolume `q^n / |C|`, the volume of a fundamental region.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] if `q^n` overflows `u128`, or if the cardinality
    /// does not divide it — which means the caller's `cardinality` is wrong.
    pub fn covolume(&self) -> Result<u128, LatticeError> {
        let q = u128::from(self.zq.modulus());
        let mut total: u128 = 1;
        for _ in 0..self.code.length() {
            total = total.checked_mul(q).ok_or(RangeError::Overflow {
                op: Op::Mul,
                width_bits: 128,
            })?;
        }
        let size = u128::from(self.code.cardinality());
        if size == 0 || !total.is_multiple_of(size) {
            return Err(LatticeError::Degenerate);
        }
        Ok(total / size)
    }

    /// Is the integer point `x` in the lattice?
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadSupport`] if `x` is not the code's length.
    pub fn contains(&self, x: &[i64]) -> Result<bool, LatticeError> {
        if x.len() != self.code.length() {
            return Err(LatticeError::BadSupport);
        }
        let mut residues = vec![0u32; x.len()];
        for (dst, &v) in residues.iter_mut().zip(x) {
            *dst = self.zq.reduce_i64(v);
        }
        Ok(self.code.contains(&residues))
    }
}

impl<C: CodeMembership> Quantizer for ConstructionA<C> {
    fn dim(&self) -> usize {
        self.code.length()
    }

    fn scale(&self) -> i64 {
        1
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        let n = self.code.length();
        crate::validate(x, out, n)?;
        let q = self.zq.modulus();
        let symbols = usize::try_from(q).map_err(|_| DecodeError::NotInLattice)?;
        scratch.ensure(n);
        scratch.ensure_costs(n, symbols);

        // For each coordinate and each residue class, the nearest integer in
        // that class and the squared error it costs. The lattice's own `out`
        // doubles as storage for the winning representative, so the cost table
        // is the only extra state.
        let mut costs = core::mem::take(&mut scratch.costs);
        #[allow(clippy::cast_precision_loss)]
        let modulus = f64::from(q);
        for (i, &xi) in x.iter().enumerate() {
            for s in 0..symbols {
                #[allow(clippy::cast_precision_loss)]
                let lifted = f64::from(self.zq.lift(u32::try_from(s).unwrap_or(0)));
                let steps = round_away((xi - lifted) / modulus);
                #[allow(clippy::cast_precision_loss)]
                let candidate = lifted + steps as f64 * modulus;
                let d = xi - candidate;
                costs[i * symbols + s] = d * d;
            }
        }

        let mut chosen = core::mem::take(&mut scratch.symbols);
        let result = self
            .code
            .decode_costs(&costs[..n * symbols], &mut chosen[..n]);
        if result.is_ok() {
            for (i, (&xi, slot)) in x.iter().zip(out.iter_mut()).enumerate() {
                let lifted = f64::from(self.zq.lift(chosen[i]));
                let steps = round_away((xi - lifted) / modulus);
                #[allow(clippy::cast_possible_truncation)]
                let value = lifted as i64 + steps * i64::from(q);
                *slot = value;
            }
        }
        scratch.costs = costs;
        scratch.symbols = chosen;
        result
    }
}
