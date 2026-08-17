//! Babai's rounding and nearest-plane algorithms, for a general basis.
//!
//! # Coordinates in, coordinates out
//!
//! These take the target as a **real coefficient vector** `t`, meaning the
//! point `Σ t_i b_i`, and return integer coefficients. No ambient
//! representation appears, which is what lets them work for lattices whose
//! ambient basis is not integral — the same reason the rest of the crate is
//! coordinate-first. [`coefficients`] converts from the inner products
//! `⟨x, b_i⟩`, which is what a caller holding an ambient point can compute.
//!
//! # These are approximations, and say so
//!
//! Neither algorithm returns the nearest lattice point in general. Rounding is
//! cheap and weak; nearest-plane is `O(n²)` given the orthogonalization and, on
//! an LLL-reduced basis, lands within a factor `2^(n/2)` of optimal. Both are
//! *exact* when the basis is orthogonal, and both are only as good as the basis
//! — run [`lattica::reduce::lll`] first.
//!
//! When the true nearest point is required, enumerate.

// Converting exact integers into `f64` is the defining act of these modules:
// the lattice is integral, the target is real, and the two must meet. Every
// cast is on a Gram entry, a minor, or a coefficient already validated to be
// finite; where a minor could exceed 2^53 the loss is in a reported quantity,
// never in a decision the exact path also makes.
#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use lattica::basis::Gram;
use lattica::error::{DecodeError, ReduceError};
use lattica::gso::Gso;
use lattica::int::{Int, adjugate};

/// Nearest integer, ties away from zero. See the crate documentation for why this rule.
#[allow(clippy::cast_possible_truncation)]
fn round_away(v: f64) -> i64 {
    let truncated = v as i64;
    let fraction = v - truncated as f64;
    if fraction >= 0.5 {
        truncated + 1
    } else if fraction <= -0.5 {
        truncated - 1
    } else {
        truncated
    }
}

/// Converts inner products into basis coefficients: given `inner[i] = ⟨x, b_i⟩`,
/// writes the real `t` with `x = Σ t_i b_i`.
///
/// Solves `t = w · G⁻¹` as `w · adj(G) / det(G)`, so the matrix inversion is
/// exact and only the final division is floating point.
///
/// # Errors
///
/// [`ReduceError::Singular`] if the Gram matrix is singular, and
/// [`ReduceError::Range`] on a length mismatch or an overflow.
pub fn coefficients<T: Int>(
    gram: &Gram<T>,
    inner: &[f64],
    out: &mut [f64],
) -> Result<(), ReduceError> {
    let n = gram.dim();
    if inner.len() != n || out.len() != n {
        return Err(lattica::error::RangeError::Shape {
            expected: n,
            found: inner.len(),
        }
        .into());
    }
    let determinant = gram.det()?;
    if determinant.is_zero() {
        return Err(ReduceError::Singular);
    }
    let cofactors = adjugate(gram.as_matrix())?;

    let scale = determinant.widen() as f64;
    for (j, slot) in out.iter_mut().enumerate() {
        let mut total = 0.0f64;
        for (i, &w) in inner.iter().enumerate() {
            let entry = cofactors.get(i, j).widen() as f64;
            total += w * entry;
        }
        *slot = total / scale;
    }
    Ok(())
}

/// Babai rounding: round every coefficient independently.
///
/// `O(n)`, and exact only when the basis is orthogonal. On a skewed basis it
/// can be arbitrarily bad, which is the entire motivation for
/// [`nearest_plane`].
///
/// # Errors
///
/// [`DecodeError::LengthMismatch`] if the slices differ in length, and
/// [`DecodeError::NonFinite`] for a NaN or infinity.
pub fn round(coefficients: &[f64], out: &mut [i64]) -> Result<(), DecodeError> {
    if coefficients.len() != out.len() {
        return Err(DecodeError::LengthMismatch {
            expected: coefficients.len(),
            found: out.len(),
        });
    }
    for (index, &value) in coefficients.iter().enumerate() {
        if !value.is_finite() {
            return Err(DecodeError::NonFinite { index });
        }
    }
    for (slot, &value) in out.iter_mut().zip(coefficients) {
        *slot = round_away(value);
    }
    Ok(())
}

/// Babai's nearest-plane algorithm.
///
/// Walks the basis from the last vector to the first, at each step rounding the
/// target's coordinate along `b*_i` and subtracting the chosen multiple of
/// `b_i`. On an LLL-reduced basis the answer is within `2^(n/2)` of the nearest
/// point, and it is exactly the nearest point whenever the basis is orthogonal.
///
/// `coefficients` is consumed: the algorithm subtracts as it goes, and on
/// return it holds the residual coefficient vector `t - z`. That is not a
/// side effect to work around — it is the quantization error in coordinates,
/// which is usually what the caller wants next.
///
/// # Errors
///
/// [`DecodeError::LengthMismatch`] on a length disagreement with the
/// orthogonalization, and [`DecodeError::NonFinite`] for a NaN or infinity.
///
/// # Examples
///
/// ```
/// use lattica::basis::Gram;
/// use lattica::gso::Gso;
/// use lattice_engine::babai;
///
/// // An orthogonal basis: nearest-plane is then exactly the nearest point.
/// let g = Gram::<i64>::from_rows(2, &[4, 0, 0, 9])?;
/// let gso = Gso::new(&g)?;
///
/// let mut t = [1.4, -0.6];
/// let mut z = [0i64; 2];
/// babai::nearest_plane(&gso, &mut t, &mut z)?;
/// assert_eq!(z, [1, -1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn nearest_plane<T: Int>(
    gso: &Gso<T>,
    coefficients: &mut [f64],
    out: &mut [i64],
) -> Result<(), DecodeError> {
    let n = gso.dim();
    if coefficients.len() != n || out.len() != n {
        return Err(DecodeError::LengthMismatch {
            expected: n,
            found: coefficients.len(),
        });
    }
    for (index, &value) in coefficients.iter().enumerate() {
        if !value.is_finite() {
            return Err(DecodeError::NonFinite { index });
        }
    }

    for i in (0..n).rev() {
        // The target's coordinate along b*_i, given the multiples already
        // subtracted from the coefficients above it.
        let mut along = coefficients[i];
        for (j, &above) in coefficients.iter().enumerate().skip(i + 1) {
            along += above * gso.mu(j, i);
        }
        let chosen = round_away(along);
        out[i] = chosen;
        coefficients[i] -= chosen as f64;
    }
    Ok(())
}

/// Squared distance between a real coefficient vector and an integer one,
/// measured with the Gram matrix.
///
/// # Errors
///
/// [`DecodeError::LengthMismatch`] if either slice is the wrong length.
pub fn distance_sq<T: Int>(
    gram: &Gram<T>,
    target: &[f64],
    point: &[i64],
) -> Result<f64, DecodeError> {
    let n = gram.dim();
    if target.len() != n || point.len() != n {
        return Err(DecodeError::LengthMismatch {
            expected: n,
            found: target.len(),
        });
    }
    let mut total = 0.0f64;
    for i in 0..n {
        let di = target[i] - point[i] as f64;
        if di == 0.0 {
            continue;
        }
        for j in 0..n {
            let dj = target[j] - point[j] as f64;
            let entry = gram.entry(i, j).widen() as f64;
            total += di * entry * dj;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    // Exact on dyadic and integral inputs, so `==` is the right assertion.
    #![allow(clippy::float_cmp)]
    use super::{coefficients, distance_sq, nearest_plane, round};
    use lattica::basis::Gram;
    use lattica::gso::Gso;
    use lattica::named::{e8, zn};

    #[test]
    fn on_the_integer_lattice_both_agree_with_plain_rounding() {
        let g = zn::<i64>(4).unwrap();
        let gso = Gso::new(&g).unwrap();
        let target = [1.4, -0.6, 2.5, -2.5];

        let mut rounded = [0i64; 4];
        round(&target, &mut rounded).unwrap();
        assert_eq!(rounded, [1, -1, 3, -3]);

        let mut work = target;
        let mut plane = [0i64; 4];
        nearest_plane(&gso, &mut work, &mut plane).unwrap();
        assert_eq!(plane, rounded);
    }

    #[test]
    fn nearest_plane_is_exact_on_an_orthogonal_basis() {
        let g = Gram::<i64>::from_rows(3, &[4, 0, 0, 0, 9, 0, 0, 0, 25]).unwrap();
        let gso = Gso::new(&g).unwrap();
        for a in -4..=4 {
            for b in -4..=4 {
                let target = [f64::from(a) * 0.3, f64::from(b) * 0.4, 0.51];
                let mut work = target;
                let mut z = [0i64; 3];
                nearest_plane(&gso, &mut work, &mut z).unwrap();
                // Orthogonal: the optimum is coordinatewise rounding.
                let mut expected = [0i64; 3];
                round(&target, &mut expected).unwrap();
                assert_eq!(z, expected);
            }
        }
    }

    #[test]
    fn the_residual_is_left_in_the_coefficient_buffer() {
        let g = zn::<i64>(3).unwrap();
        let gso = Gso::new(&g).unwrap();
        let mut work = [1.25, -0.75, 4.5];
        let mut z = [0i64; 3];
        nearest_plane(&gso, &mut work, &mut z).unwrap();
        assert_eq!(z, [1, -1, 5]);
        assert_eq!(work, [0.25, 0.25, -0.5]);
    }

    #[test]
    fn coefficients_invert_the_gram_matrix() {
        // For b_i orthogonal with |b_i|^2 = g_ii, <x, b_i> = t_i * g_ii.
        let g = Gram::<i64>::from_rows(2, &[4, 0, 0, 9]).unwrap();
        let mut t = [0.0f64; 2];
        coefficients(&g, &[8.0, -18.0], &mut t).unwrap();
        assert!((t[0] - 2.0).abs() < 1e-12);
        assert!((t[1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn coefficients_round_trip_through_the_gram_matrix() {
        let g = e8::<i64>().unwrap();
        let t = [0.3f64, -1.2, 2.0, 0.5, -0.25, 1.75, 0.0, -3.5];
        // w_i = <x, b_i> = sum_j t_j G[j][i]
        let mut inner = [0.0f64; 8];
        for (i, slot) in inner.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let value: f64 = (0..8).map(|j| t[j] * g.entry(j, i) as f64).sum();
            *slot = value;
        }
        let mut recovered = [0.0f64; 8];
        coefficients(&g, &inner, &mut recovered).unwrap();
        for i in 0..8 {
            assert!((recovered[i] - t[i]).abs() < 1e-9, "coefficient {i}");
        }
    }

    #[test]
    fn distance_uses_the_gram_metric() {
        let g = Gram::<i64>::from_rows(2, &[4, 0, 0, 9]).unwrap();
        // (0.5, 0) in coefficients is a vector of squared length 0.25*4 = 1.
        assert!((distance_sq(&g, &[0.5, 0.0], &[0, 0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((distance_sq(&g, &[0.0, 1.0 / 3.0], &[0, 0]).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bad_input_is_rejected() {
        let g = zn::<i64>(2).unwrap();
        let gso = Gso::new(&g).unwrap();
        let mut z = [0i64; 2];
        assert!(nearest_plane(&gso, &mut [f64::NAN, 0.0], &mut z).is_err());
        assert!(nearest_plane(&gso, &mut [0.0], &mut z).is_err());
        assert!(round(&[0.0], &mut z).is_err());
        let mut t = [0.0f64; 2];
        assert!(coefficients(&g, &[1.0], &mut t).is_err());
    }
}
