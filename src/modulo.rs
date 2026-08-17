//! Reduction modulo a lattice, and lattices scaled by an integer.

// The scale and factor conversions are on small positive integers, and the
// coordinate conversions are on values already validated against COORD_LIMIT.
#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use super::{Quantizer, Scratch, validate_lengths};
use lattica::error::{DecodeError, LatticeError};

/// `x mod Λ = x - Q_Λ(x)`, the quantization error.
///
/// The result is the representative of `x + Λ` inside the fundamental Voronoi
/// region of `Λ`, so `mod Λ` is exactly "throw away the lattice point and keep
/// what is left". It is the shaping operation of a nested lattice code and the
/// error term of a lattice quantizer, which are the same object viewed from two
/// ends.
///
/// # Translation invariance holds up to ties
///
/// `(x + λ) mod Λ == x mod Λ` for `λ ∈ Λ` — *except* when `x` lies on a Voronoi
/// boundary. There the two computations may return different, equidistant
/// nearest points, because the tie rules are index-based and the `D_n^+` coset
/// preference is not symmetric under a glue-vector shift. The invariant that
/// always holds is the weaker and more fundamental one: **the distance to the
/// lattice is translation-invariant**, so the two residuals differ by a lattice
/// vector and have equal norm.
///
/// A codec that assumes exact equality will be wrong on a measure-zero set of
/// inputs — which, with quantized or structured inputs, is not as rare as
/// "measure zero" suggests. Compare energies, not coordinates.
///
/// The same caveat applies to negation; see the crate documentation.
///
/// # Errors
///
/// As [`Quantizer::nearest`], plus [`DecodeError::LengthMismatch`] if `out` is
/// not the lattice's dimension. `out` is untouched on failure.
///
/// # Examples
///
/// ```
/// use lattice_engine::{Scratch, Zn, mod_lattice};
///
/// let q = Zn::new(2).unwrap();
/// let mut scratch = Scratch::new(2);
/// let mut out = [0.0f64; 2];
///
/// mod_lattice(&q, &[3.25, -1.75], &mut out, &mut scratch).unwrap();
/// assert_eq!(out, [0.25, 0.25]);
/// ```
pub fn mod_lattice<Q: Quantizer + ?Sized>(
    q: &Q,
    x: &[f64],
    out: &mut [f64],
    scratch: &mut Scratch,
) -> Result<(), DecodeError> {
    let dim = q.dim();
    validate_lengths(x.len(), out.len(), dim)?;
    scratch.ensure(dim);

    // Borrow the integer buffer out of the scratch so the decoder can still
    // take `&mut Scratch`. `take` on a `Vec` moves the allocation and leaves an
    // empty one behind, so this costs nothing.
    let mut point = core::mem::take(&mut scratch.point);
    let result = q.nearest(x, &mut point[..dim], scratch);
    if result.is_ok() {
        let scale = f64::from(i32::try_from(q.scale()).unwrap_or(1));
        for ((dst, &xi), &vi) in out.iter_mut().zip(x).zip(&point[..dim]) {
            *dst = xi - vi as f64 / scale;
        }
    }
    scratch.point = point;
    result
}

/// Dithered reduction: `((x + dither) mod Λ) - dither`.
///
/// With `dither` drawn uniformly from the Voronoi region of `Λ`, the result is
/// uniform over that region **and statistically independent of `x`** — the
/// crypto-like property that makes dithered lattice quantization behave as an
/// additive noise channel rather than a deterministic distortion. Nested
/// lattice codes rely on it for their shaping gain, and without the dither the
/// error correlates with the signal.
///
/// # Errors
///
/// As [`mod_lattice`], plus [`DecodeError::LengthMismatch`] if `dither` is not
/// the lattice's dimension.
pub fn mod_lattice_dithered<Q: Quantizer + ?Sized>(
    q: &Q,
    x: &[f64],
    dither: &[f64],
    out: &mut [f64],
    scratch: &mut Scratch,
) -> Result<(), DecodeError> {
    let dim = q.dim();
    validate_lengths(x.len(), out.len(), dim)?;
    if dither.len() != dim {
        return Err(DecodeError::LengthMismatch {
            expected: dim,
            found: dither.len(),
        });
    }
    scratch.ensure(dim);

    let mut shifted = core::mem::take(&mut scratch.divided);
    for ((dst, &xi), &di) in shifted[..dim].iter_mut().zip(x).zip(dither) {
        *dst = xi + di;
    }
    let result = mod_lattice(q, &shifted[..dim], out, scratch);
    scratch.divided = shifted;
    result?;

    for (dst, &di) in out.iter_mut().zip(dither) {
        *dst -= di;
    }
    Ok(())
}

/// The lattice `k·Λ`, for a positive integer `k`.
///
/// `Q_{kΛ}(x) = k · Q_Λ(x/k)`, so a scaled lattice needs no new decoder. This
/// is how the shaping lattice of a self-similar nested code is built: the
/// coding lattice is `Λ` and the shaping lattice is `k·Λ`, giving a codebook of
/// `k^n` points.
///
/// # Exactness
///
/// The division by `k` is the one operation in the decoders that is not exact
/// for every input. When `k` is a power of two it is exact and invariant I2
/// holds unchanged; otherwise the quotient is correctly rounded, which is still
/// bit-identical on every platform but is no longer the same as decoding
/// against the true scaled lattice at distances below one ulp. Prefer a power
/// of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scaled<Q> {
    inner: Q,
    factor: i64,
}

impl<Q: Quantizer> Scaled<Q> {
    /// Builds the decoder for `factor · Λ`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] if `factor` is not positive.
    pub fn new(inner: Q, factor: i64) -> Result<Self, LatticeError> {
        if factor <= 0 {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self { inner, factor })
    }

    /// The scaling factor.
    #[must_use]
    pub const fn factor(&self) -> i64 {
        self.factor
    }

    /// The lattice being scaled.
    #[must_use]
    pub const fn inner(&self) -> &Q {
        &self.inner
    }
}

impl<Q: Quantizer> Quantizer for Scaled<Q> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn scale(&self) -> i64 {
        self.inner.scale()
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        let dim = self.dim();
        validate_lengths(x.len(), out.len(), dim)?;
        scratch.ensure(dim);

        let factor = self.factor as f64;
        let mut divided = core::mem::take(&mut scratch.divided);
        for (dst, &xi) in divided[..dim].iter_mut().zip(x) {
            *dst = xi / factor;
        }
        let result = self.inner.nearest(&divided[..dim], out, scratch);
        scratch.divided = divided;
        result?;

        for slot in out.iter_mut() {
            *slot = slot
                .checked_mul(self.factor)
                .ok_or(DecodeError::NotInLattice)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Residuals of dyadic inputs are exact, so `==` is the right assertion.
    #![allow(clippy::float_cmp)]
    use super::{Scaled, mod_lattice, mod_lattice_dithered};
    use crate::{Dn, Quantizer, Scratch, Zn, e8};

    #[test]
    fn reduction_lands_in_the_voronoi_region() {
        // The defining property: the residual quantizes to the origin.
        let q = Dn::new(6).unwrap();
        let mut scratch = Scratch::new(6);
        let mut residual = [0.0f64; 6];
        let mut point = [0i64; 6];
        for k in -20..=20 {
            let x: Vec<f64> = (0..6)
                .map(|i| f64::from(k) * 0.37 + f64::from(i) * 1.9)
                .collect();
            mod_lattice(&q, &x, &mut residual, &mut scratch).unwrap();
            q.nearest(&residual, &mut point, &mut scratch).unwrap();
            assert_eq!(point, [0; 6], "residual {residual:?} left the cell");
        }
    }

    #[test]
    fn reduction_is_idempotent_and_shift_invariant() {
        let q = e8();
        let mut scratch = Scratch::new(8);
        let (mut once, mut twice) = ([0.0f64; 8], [0.0f64; 8]);
        for k in -10..=10 {
            let x: Vec<f64> = (0..8)
                .map(|i| f64::from(k) * 0.61 + f64::from(i) * 0.43)
                .collect();
            mod_lattice(&q, &x, &mut once, &mut scratch).unwrap();
            mod_lattice(&q, &once.clone(), &mut twice, &mut scratch).unwrap();
            assert_eq!(once, twice, "mod is not idempotent");

            // Adding a lattice vector -- here the all-halves glue vector of
            // E_8, doubled to (1,...,1) -- must not change the residual.
            let shifted: Vec<f64> = x.iter().map(|v| v + 0.5).collect();
            let mut moved = [0.0f64; 8];
            mod_lattice(&q, &shifted, &mut moved, &mut scratch).unwrap();
            for i in 0..8 {
                assert!(
                    (moved[i] - (once[i] + 0.5)).abs() < 1e-12
                        || (moved[i] - (once[i] - 0.5)).abs() < 1e-12
                        || (moved[i] - once[i]).abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn dithering_cancels_exactly_when_the_dither_is_zero() {
        let q = Zn::new(3).unwrap();
        let mut scratch = Scratch::new(3);
        let (mut plain, mut dithered) = ([0.0f64; 3], [0.0f64; 3]);
        let x = [1.3, -2.7, 0.1];
        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice_dithered(&q, &x, &[0.0; 3], &mut dithered, &mut scratch).unwrap();
        assert_eq!(plain, dithered);
    }

    #[test]
    fn a_scaled_lattice_decodes_to_multiples() {
        let q = Scaled::new(Zn::new(3).unwrap(), 4).unwrap();
        let mut scratch = Scratch::new(3);
        let mut out = [0i64; 3];
        q.nearest(&[1.9, 6.1, -2.1], &mut out, &mut scratch)
            .unwrap();
        assert_eq!(out, [0, 8, -4]);
        assert!(out.iter().all(|v| v % 4 == 0));
    }

    #[test]
    fn scaling_by_one_changes_nothing() {
        let plain = Dn::new(4).unwrap();
        let scaled = Scaled::new(plain, 1).unwrap();
        let mut scratch = Scratch::new(4);
        let (mut a, mut b) = ([0i64; 4], [0i64; 4]);
        let x = [0.4, -1.3, 2.2, 0.9];
        plain.nearest(&x, &mut a, &mut scratch).unwrap();
        scaled.nearest(&x, &mut b, &mut scratch).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_non_positive_factor_is_rejected() {
        assert!(Scaled::new(Zn::new(2).unwrap(), 0).is_err());
        assert!(Scaled::new(Zn::new(2).unwrap(), -3).is_err());
    }
}
