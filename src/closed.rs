//! Closed-form nearest-point decoders (Conway–Sloane, 1982).
//!
//! Each of these is `O(n)` — `O(n)` expected for [`An`] — and uses only add,
//! subtract, compare, and round. See the [module documentation](super) for why
//! that operation set is the load-bearing part.

// Casts here are `f64 -> i64` on values already validated to lie within
// `COORD_LIMIT`, where every `f64` is an exact integer well inside `i64`, and
// `i64 -> f64` on decoded coordinates of the same magnitude. Neither can lose
// information. `TryFrom` does not exist for this pair, and a fallible
// conversion on the hot path would buy nothing.
// The `&self` receivers mirror the `Quantizer` trait; taking these
// zero-cost types by value in the inherent helpers would only make the two
// shapes disagree.
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use super::{Quantizer, Scratch, validate};

use crate::kernel::BatchFamily;
use crate::round_away;
use lattica::error::{DecodeError, LatticeError};

/// Conway–Sloane `f`: coordinatewise nearest integer, ties away from zero.
///
/// The tie rule makes `f(-x) == -f(x)`, so every decoder built on it commutes
/// with negation. See invariant I3.
///
/// # Errors
///
/// [`DecodeError::LengthMismatch`] if the slices differ in length,
/// [`DecodeError::NonFinite`] for NaN or infinity, and [`DecodeError::Range`]
/// beyond [`super::COORD_LIMIT`].
///
/// # Examples
///
/// ```
/// use lattice_engine::round_nearest;
///
/// let mut out = [0i64; 4];
/// round_nearest(&[0.5, -0.5, 1.4, -1.6], &mut out).unwrap();
/// assert_eq!(out, [1, -1, 1, -2]);
/// ```
pub fn round_nearest(x: &[f64], out: &mut [i64]) -> Result<(), DecodeError> {
    validate(x, out, x.len())?;
    for (dst, &v) in out.iter_mut().zip(x) {
        *dst = round_away(v);
    }
    Ok(())
}

/// Conway–Sloane `g`: [`round_nearest`] with the worst coordinate rounded the
/// other way.
///
/// "Worst" means furthest from an integer; ties go to the lowest index. `f` and
/// `g` differ in exactly one coordinate, by exactly one — which is the entire
/// trick behind the [`Dn`] decoder, since their coordinate sums then have
/// opposite parity.
///
/// # Errors
///
/// As [`round_nearest`].
///
/// # Examples
///
/// ```
/// use lattice_engine::{round_nearest, round_nearest_flipped};
///
/// let x = [0.5, 0.5];
/// let (mut f, mut g) = ([0i64; 2], [0i64; 2]);
/// round_nearest(&x, &mut f).unwrap();
/// round_nearest_flipped(&x, &mut g).unwrap();
///
/// // Both coordinates are equally far from an integer, so the tie goes to
/// // index 0; `f` rounded it up, so `g` rounds it down.
/// assert_eq!(f, [1, 1]);
/// assert_eq!(g, [0, 1]);
/// ```
pub fn round_nearest_flipped(x: &[f64], out: &mut [i64]) -> Result<(), DecodeError> {
    validate(x, out, x.len())?;
    if x.is_empty() {
        return Ok(());
    }
    let mut worst = 0usize;
    let mut worst_distance = -1.0f64;
    let mut worst_delta = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        let rounded = round_away(v);
        out[i] = rounded;
        let delta = v - rounded as f64;
        let distance = if delta < 0.0 { -delta } else { delta };
        // Strict `>` keeps the lowest index on a tie.
        if distance > worst_distance {
            worst_distance = distance;
            worst_delta = delta;
            worst = i;
        }
    }
    out[worst] += if worst_delta >= 0.0 { 1 } else { -1 };
    Ok(())
}

/// Rounds into `out` and records `x - round(x)` in `delta`, returning the index
/// of the coordinate furthest from an integer.
fn round_with_deltas(x: &[f64], out: &mut [i64], delta: &mut [f64]) -> usize {
    let mut worst = 0usize;
    let mut worst_distance = -1.0f64;
    for (i, &v) in x.iter().enumerate() {
        let rounded = round_away(v);
        out[i] = rounded;
        let d = v - rounded as f64;
        delta[i] = d;
        let distance = if d < 0.0 { -d } else { d };
        if distance > worst_distance {
            worst_distance = distance;
            worst = i;
        }
    }
    worst
}

/// The `D_n` decoder proper, over caller-provided buffers.
///
/// Computes `f(x)`, and if its coordinate sum is odd replaces it with `g(x)`.
/// Exactly one of the two has an even sum, because they differ by one in a
/// single coordinate.
fn decode_dn(x: &[f64], out: &mut [i64], delta: &mut [f64]) {
    let worst = round_with_deltas(x, out, delta);
    // Parity of a sum is the exclusive-or of the low bits, which cannot
    // overflow the way an accumulating sum could.
    let mut parity = 0i64;
    for &v in out.iter() {
        parity ^= v;
    }
    if parity & 1 != 0 {
        out[worst] += if delta[worst] >= 0.0 { 1 } else { -1 };
    }
}

/// Squared distance from `x` to the integer point `v`, plus a constant offset
/// applied to every coordinate of `v`.
fn distance_sq(x: &[f64], v: &[i64], offset: f64) -> f64 {
    let mut total = 0.0f64;
    for (&xi, &vi) in x.iter().zip(v) {
        let d = xi - (vi as f64 + offset);
        total += d * d;
    }
    total
}

/// The integer lattice `Z^n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zn {
    n: usize,
}

impl Zn {
    /// Creates the decoder for `Z^n`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] if `n` is zero.
    pub const fn new(n: usize) -> Result<Self, LatticeError> {
        if n == 0 {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self { n })
    }
}

impl Quantizer for Zn {
    fn dim(&self) -> usize {
        self.n
    }

    fn scale(&self) -> i64 {
        1
    }

    fn batch_family(&self) -> Option<BatchFamily> {
        Some(BatchFamily::Zn)
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        _scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        validate(x, out, self.n)?;
        for (dst, &v) in out.iter_mut().zip(x) {
            *dst = round_away(v);
        }
        Ok(())
    }
}

/// The checkerboard lattice `D_n = {x ∈ Z^n : Σ x_i even}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dn {
    n: usize,
}

impl Dn {
    /// Creates the decoder for `D_n`.
    ///
    /// Requires `n >= 2`, below which the parity constraint is vacuous. Note
    /// that [`lattica::named::d_n`] requires `n >= 3` instead — a different
    /// bound for a different reason, since its Dynkin construction does not
    /// describe `D_2`. The decoder here is correct for every `n >= 2`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] if `n < 2`.
    pub const fn new(n: usize) -> Result<Self, LatticeError> {
        if n < 2 {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self { n })
    }
}

impl Quantizer for Dn {
    fn dim(&self) -> usize {
        self.n
    }

    fn scale(&self) -> i64 {
        1
    }

    fn batch_family(&self) -> Option<BatchFamily> {
        Some(BatchFamily::Dn)
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        validate(x, out, self.n)?;
        scratch.ensure(self.n);
        decode_dn(x, out, &mut scratch.delta[..self.n]);
        Ok(())
    }
}

/// The root lattice `A_n = {x ∈ Z^(n+1) : Σ x_i = 0}`, of rank `n` in ambient
/// dimension `n+1`.
///
/// The decoder projects onto the sum-zero hyperplane first. That is not an
/// optimisation but a requirement: for `x` off the hyperplane the squared
/// distance splits as `‖x - proj(x)‖² + ‖proj(x) - v‖²`, so the nearest point
/// is determined entirely by the projection, and the coordinate-adjustment step
/// below is only bounded when the coordinate sum is already near zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct An {
    rank: usize,
}

impl An {
    /// Creates the decoder for `A_n`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] if `n` is zero.
    pub const fn new(n: usize) -> Result<Self, LatticeError> {
        if n == 0 {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self { rank: n })
    }

    /// Decodes using a full sort of the coordinate residuals.
    ///
    /// `O(n log n)`, and the reference implementation: [`Quantizer::nearest`]
    /// uses a linear-time selection instead, and the two are differentially
    /// tested against each other. Both resolve ties by lowest index, so they
    /// agree on every input, not merely on generic ones.
    ///
    /// # Errors
    ///
    /// As [`Quantizer::nearest`].
    pub fn nearest_sorted(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        self.decode(x, out, scratch, true)
    }

    fn decode(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
        sorted: bool,
    ) -> Result<(), DecodeError> {
        let m = self.rank + 1;
        validate(x, out, m)?;
        scratch.ensure(m);

        let Scratch {
            delta,
            shifted,
            order,
            ..
        } = scratch;
        let (delta, shifted, order) = (&mut delta[..m], &mut shifted[..m], &mut order[..m]);

        // Project onto the sum-zero hyperplane.
        let mut total = 0.0f64;
        for &v in x {
            total += v;
        }
        let mean = total / m as f64;
        for (dst, &v) in shifted.iter_mut().zip(x) {
            *dst = v - mean;
        }

        round_with_deltas(shifted, out, delta);
        let mut deficiency = 0i64;
        for &v in out.iter() {
            deficiency += v;
        }
        if deficiency == 0 {
            return Ok(());
        }

        let count = usize::try_from(deficiency.abs()).map_err(|_| DecodeError::NotInLattice)?;
        if count > m {
            // Unreachable after projection, where |Σ round(x'_i)| ≤ m/2.
            return Err(DecodeError::NotInLattice);
        }

        for (slot, value) in order.iter_mut().enumerate() {
            *value = u32::try_from(slot).map_err(|_| DecodeError::NotInLattice)?;
        }

        // Decrementing coordinate i costs 2·δ_i + 1, so an excess is removed
        // from the smallest residuals; incrementing costs 1 - 2·δ_i, so a
        // deficit is made up from the largest. Ties go to the lowest index in
        // both directions, which is why the comparator carries the index.
        let ascending = deficiency > 0;
        let compare = |a: &u32, b: &u32| {
            let (ia, ib) = (*a as usize, *b as usize);
            let primary = if ascending {
                delta[ia].total_cmp(&delta[ib])
            } else {
                delta[ib].total_cmp(&delta[ia])
            };
            primary.then(ia.cmp(&ib))
        };

        if sorted {
            order.sort_unstable_by(compare);
        } else if count < m {
            order.select_nth_unstable_by(count, compare);
        }

        let step = if ascending { -1 } else { 1 };
        for &index in &order[..count] {
            out[index as usize] += step;
        }
        Ok(())
    }
}

impl Quantizer for An {
    fn dim(&self) -> usize {
        self.rank + 1
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
        self.decode(x, out, scratch, false)
    }
}

/// The lattice `D_n^+ = D_n ∪ (D_n + ½·1)`, for even `n`.
///
/// At `n = 8` this is `E_8`; see [`e8`]. Its points are half-integers, so
/// [`Quantizer::scale`] is 2 and the decoder writes `2v`.
///
/// `D_n^+` is a lattice for every even `n`, and an *integral* lattice when
/// `4 | n`. At `n = 4` it is a rescaled `Z^4`, and at `n = 8` it is the
/// exceptional `E_8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DnPlus {
    n: usize,
}

impl DnPlus {
    /// Creates the decoder for `D_n^+`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] if `n` is odd or below 2.
    pub const fn new(n: usize) -> Result<Self, LatticeError> {
        if n < 2 || !n.is_multiple_of(2) {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self { n })
    }
}

impl Quantizer for DnPlus {
    fn dim(&self) -> usize {
        self.n
    }

    fn scale(&self) -> i64 {
        2
    }

    fn batch_family(&self) -> Option<BatchFamily> {
        Some(BatchFamily::DnPlus)
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        let n = self.n;
        validate(x, out, n)?;
        scratch.ensure(n);

        let Scratch {
            base,
            alt,
            delta,
            shifted,
            ..
        } = scratch;
        let (base, alt) = (&mut base[..n], &mut alt[..n]);
        let (delta, shifted) = (&mut delta[..n], &mut shifted[..n]);

        // The D_n coset.
        decode_dn(x, base, delta);

        // The shifted coset: decode x - ½·1 in D_n, then add ½·1 back.
        for (dst, &v) in shifted.iter_mut().zip(x) {
            *dst = v - 0.5;
        }
        decode_dn(shifted, alt, delta);

        let plain = distance_sq(x, base, 0.0);
        let shifted_distance = distance_sq(x, alt, 0.5);

        // Strict `<` prefers the D_n coset on a tie (invariant I3).
        if shifted_distance < plain {
            for (dst, &v) in out.iter_mut().zip(alt.iter()) {
                *dst = 2 * v + 1;
            }
        } else {
            for (dst, &v) in out.iter_mut().zip(base.iter()) {
                *dst = 2 * v;
            }
        }
        Ok(())
    }
}

/// The `E_8` decoder.
///
/// For the *Gram matrix* see [`lattica::named::e8`]; this is the geometric side.
///
/// `E_8` is `D_8^+`, so this is not a second implementation — it is the same
/// code path at `n = 8`. Output coordinates are doubled; see
/// [`Quantizer::scale`].
///
/// # Panics
///
/// Never: 8 is even and at least 2.
#[must_use]
pub fn e8() -> DnPlus {
    match DnPlus::new(8) {
        Ok(q) => q,
        Err(_) => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use super::{An, Dn, DnPlus, Zn, e8, round_nearest, round_nearest_flipped};
    use crate::{Quantizer, Scratch};
    use lattica::error::LatticeError;

    fn decode<Q: Quantizer>(q: &Q, x: &[f64]) -> Vec<i64> {
        let mut out = vec![0i64; q.dim()];
        let mut scratch = Scratch::new(q.dim());
        q.nearest(x, &mut out, &mut scratch).unwrap();
        out
    }

    #[test]
    fn degenerate_dimensions_are_rejected() {
        assert_eq!(Zn::new(0), Err(LatticeError::Degenerate));
        assert_eq!(An::new(0), Err(LatticeError::Degenerate));
        assert_eq!(Dn::new(1), Err(LatticeError::Degenerate));
        assert_eq!(DnPlus::new(7), Err(LatticeError::Degenerate));
    }

    #[test]
    fn rounding_ties_go_away_from_zero() {
        let mut out = [0i64; 6];
        round_nearest(&[0.5, -0.5, 1.5, -1.5, 2.5, -2.5], &mut out).unwrap();
        assert_eq!(out, [1, -1, 2, -2, 3, -3]);
    }

    #[test]
    fn f_and_g_differ_in_exactly_one_coordinate_by_one() {
        let cases: [&[f64]; 4] = [
            &[0.1, 0.2, 0.3],
            &[0.5, 0.5, 0.5],
            &[-0.4, 0.4, 0.0],
            &[1.0, 2.0, 3.0],
        ];
        for x in cases {
            let (mut f, mut g) = ([0i64; 3], [0i64; 3]);
            round_nearest(x, &mut f).unwrap();
            round_nearest_flipped(x, &mut g).unwrap();
            let differing: Vec<usize> = (0..3).filter(|&i| f[i] != g[i]).collect();
            assert_eq!(differing.len(), 1, "x = {x:?}");
            assert_eq!((f[differing[0]] - g[differing[0]]).abs(), 1);
        }
    }

    #[test]
    fn dn_always_returns_an_even_sum() {
        let q = Dn::new(4).unwrap();
        for a in -3..=3 {
            for b in -3..=3 {
                let x = [f64::from(a) * 0.25, f64::from(b) * 0.25, 0.3, -0.7];
                let v = decode(&q, &x);
                assert_eq!(v.iter().sum::<i64>() % 2, 0, "x = {x:?}");
            }
        }
    }

    #[test]
    fn an_always_returns_a_zero_sum() {
        let q = An::new(4).unwrap();
        for k in -6..=6 {
            let x = [f64::from(k) * 0.3, 0.7, -1.2, 2.4, 0.0];
            let v = decode(&q, &x);
            assert_eq!(v.iter().sum::<i64>(), 0, "x = {x:?}");
        }
    }

    #[test]
    fn dnplus_returns_coordinates_of_uniform_parity() {
        // Every point is either in D_n (all doubled coordinates even) or in the
        // shifted coset (all odd). A mixture would not be a lattice point.
        let q = e8();
        for k in -4..=4 {
            let mut x = [0.0f64; 8];
            for (i, slot) in x.iter_mut().enumerate() {
                *slot = f64::from(k) * 0.2 + i as f64 * 0.11;
            }
            let v = decode(&q, &x);
            let odd = v[0] & 1;
            assert!(v.iter().all(|&c| c & 1 == odd), "mixed parity for {x:?}");
        }
    }

    #[test]
    fn lattice_points_are_their_own_nearest_points() {
        let q = Dn::new(6).unwrap();
        let point = [2.0, -3.0, 1.0, 4.0, 0.0, -4.0];
        assert!(point.iter().sum::<f64>().rem_euclid(2.0) < 1e-12);
        assert_eq!(decode(&q, &point), [2, -3, 1, 4, 0, -4]);

        let q = Zn::new(3).unwrap();
        assert_eq!(decode(&q, &[-7.0, 0.0, 12.0]), [-7, 0, 12]);
    }

    #[test]
    fn e8_is_dn_plus_at_eight() {
        assert_eq!(e8(), DnPlus::new(8).unwrap());
        assert_eq!(e8().dim(), 8);
        assert_eq!(e8().scale(), 2);
    }
}
