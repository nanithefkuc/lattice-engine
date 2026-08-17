//! Runtime-dispatched decode kernels over layouts owned by `lattice-engine`.
//!
//! The engine owns the flat batch-decode layout of [`crate::nearest_batch`]:
//! `points` and `out` are contiguous, strided by the lattice dimension. Rule 2
//! puts the kernels over that layout here, not in `lattica`. Selection is
//! stack-wide through `simdispatch`; `archmage` supplies the safe capability
//! token for the tier already selected.
//!
//! # Bit-identity is structural
//!
//! The kernel never changes an operation, only who executes it. The rounding
//! pass uses truncation toward zero, one exact subtraction, and two exact
//! comparisons — the same four-operation set as the scalar `round_away`, so
//! every element computes the identical integer regardless of which lane it
//! lands in. AVX2 has no packed `f64 -> i64` convert, so lane values convert
//! through an exact per-tile scalar cast on integer-valued doubles. The
//! per-vector fixups (parity, worst-coordinate ties, coset distances) are the
//! scalar code itself, in scalar order. There is no FMA, no reassociation,
//! and no transcendental anywhere on the path.
//!
//! # The error contract is preserved by construction
//!
//! [`crate::nearest_batch`] documents that a failing vector leaves earlier
//! outputs written and the failing one untouched. A flat kernel writes the
//! whole buffer, so it must never start on input that will fail: the batch is
//! pre-validated in full, and any invalid coordinate returns `false` so the
//! caller runs the documented per-vector loop, which reproduces the contract
//! exactly.

use crate::{COORD_LIMIT, Scratch};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use simdispatch::{Backend, Selection};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const ENGINE_TIERS: &[Backend] = &[Backend::V3GfniCrypto, Backend::V3, Backend::Scalar];

/// Which engine-owned batch kernel a closed-form decoder maps to.
///
/// The variant names the lattice family, not the kernel implementation: the
/// same family runs on the dispatched SIMD kernel, the scalar reference, or
/// the caller's per-vector loop, with bit-identical results on all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFamily {
    /// `Z^n`: coordinatewise rounding, the whole flat batch elementwise.
    Zn,
    /// `D_n`: the rounding pass plus the parity fixup per vector.
    Dn,
    /// `D_n^+` and `E_8`: both cosets, distances, and the coset selection.
    DnPlus,
}

/// Batches below this many vectors stay on the caller's per-vector loop:
/// dispatch and the full-buffer validation scan cost more than they save.
/// Set by the crossover measurement in `BENCHMARKS.md`; re-measure before
/// changing it.
pub const DISPATCH_MIN_VECTORS: usize = 8;

/// Rounds one flat buffer of validated coordinates.
fn round_plane_scalar(x: &[f64], out: &mut [i64]) {
    for (dst, &v) in out.iter_mut().zip(x) {
        *dst = crate::round_away(v);
    }
}

/// The `D_n` fixup over one already-rounded vector, mirroring the scalar
/// decoder exactly: parity by XOR of low bits, worst coordinate by strict `>`
/// on `|x_i - round(x_i)|` with the lowest index on ties.
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
fn fixup_dn_scalar(x: &[f64], out: &mut [i64]) {
    let mut parity = 0i64;
    for &v in out.iter() {
        parity ^= v;
    }
    if parity & 1 == 0 {
        return;
    }
    let mut worst = 0usize;
    let mut worst_distance = -1.0f64;
    let mut worst_delta = 0.0f64;
    for (i, (&xi, &vi)) in x.iter().zip(out.iter()).enumerate() {
        let delta = xi - vi as f64;
        let distance = if delta < 0.0 { -delta } else { delta };
        // Strict `>` keeps the lowest index on a tie.
        if distance > worst_distance {
            worst_distance = distance;
            worst_delta = delta;
            worst = i;
        }
    }
    out[worst] += if worst_delta >= 0.0 { 1 } else { -1 };
}

/// Squared distance from `x` to the integer point `v`, plus a constant offset
/// applied to every coordinate of `v` — the closed-form decoder's own
/// expression, accumulated in the same coordinate order.
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
fn distance_sq_scalar(x: &[f64], v: &[i64], offset: f64) -> f64 {
    let mut total = 0.0f64;
    for (&xi, &vi) in x.iter().zip(v) {
        let d = xi - (vi as f64 + offset);
        total += d * d;
    }
    total
}

/// Are all coordinates finite and within [`COORD_LIMIT`]?
fn all_valid(points: &[f64]) -> bool {
    points
        .iter()
        .all(|&v| v.is_finite() && v.abs() <= COORD_LIMIT)
}

/// Attempts the engine-owned kernel for one batch of `points.len() / dim`
/// vectors. Returns `true` when the batch is fully decoded into `out`;
/// `false` when the caller must run its per-vector loop (no kernel for the
/// family, batch below the dispatch crossover, or an invalid coordinate
/// anywhere — the last so the documented partial-write error contract is
/// preserved bit for bit). Lengths are the caller's contract and must already
/// be checked.
pub(crate) fn decode_batch(
    family: BatchFamily,
    dim: usize,
    points: &[f64],
    out: &mut [i64],
    scratch: &mut Scratch,
) -> bool {
    if dim == 0 || points.len() / dim < DISPATCH_MIN_VECTORS || !all_valid(points) {
        return false;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if matches!(backend(), Backend::V3GfniCrypto | Backend::V3) {
            use archmage::SimdToken;
            // Selection remains simdispatch's single source of policy. Summon
            // only materializes archmage's safe capability token for the tier
            // already selected; it never chooses or upgrades a backend.
            if let Some(token) = archmage::X64V3Token::summon() {
                dispatch_simd(token, family, dim, points, out, scratch);
                return true;
            }
        }
    }

    dispatch_scalar(family, dim, points, out, scratch);
    true
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn dispatch_simd(
    token: archmage::X64V3Token,
    family: BatchFamily,
    dim: usize,
    points: &[f64],
    out: &mut [i64],
    scratch: &mut Scratch,
) {
    match family {
        BatchFamily::Zn => x86::round_plane_avx2(token, points, out),
        BatchFamily::Dn => {
            x86::round_plane_avx2(token, points, out);
            fixup_all_dn(dim, points, out);
        }
        BatchFamily::DnPlus => {
            let (shifted, alt) = batch_planes(points.len(), scratch);
            build_shifted_plane(points, shifted);
            x86::round_plane_avx2(token, points, out);
            x86::round_plane_avx2(token, shifted, alt);
            fixup_all_dnplus(dim, points, shifted, out, alt);
        }
    }
}

fn dispatch_scalar(
    family: BatchFamily,
    dim: usize,
    points: &[f64],
    out: &mut [i64],
    scratch: &mut Scratch,
) {
    match family {
        BatchFamily::Zn => round_plane_scalar(points, out),
        BatchFamily::Dn => {
            round_plane_scalar(points, out);
            fixup_all_dn(dim, points, out);
        }
        BatchFamily::DnPlus => {
            let (shifted, alt) = batch_planes(points.len(), scratch);
            build_shifted_plane(points, shifted);
            round_plane_scalar(points, out);
            round_plane_scalar(shifted, alt);
            fixup_all_dnplus(dim, points, shifted, out, alt);
        }
    }
}

/// Grows the batch planes to `len` and returns the `x - ½` input plane and
/// the shifted-coset output plane. The first call allocates; a warm scratch
/// never does.
fn batch_planes(len: usize, scratch: &mut Scratch) -> (&mut [f64], &mut [i64]) {
    scratch.ensure_batch(len);
    let Scratch { shifted, alt, .. } = scratch;
    (&mut shifted[..len], &mut alt[..len])
}

/// Writes `x - ½` elementwise: one exact subtraction per coordinate.
fn build_shifted_plane(points: &[f64], shifted: &mut [f64]) {
    for (dst, &v) in shifted.iter_mut().zip(points) {
        *dst = v - 0.5;
    }
}

/// Applies the `D_n` parity fixup to every rounded vector of the batch.
fn fixup_all_dn(dim: usize, points: &[f64], out: &mut [i64]) {
    for (x, o) in points.chunks(dim).zip(out.chunks_mut(dim)) {
        fixup_dn_scalar(x, o);
    }
}

/// Applies the `D_n^+` selection to every vector: the `D_n` fixup on both
/// cosets, the two coset distances in coordinate order, and the strict-`<`
/// preference for the `D_n` coset on a tie. `shifted` holds the already
/// materialized `x - ½` plane, so nothing here allocates.
fn fixup_all_dnplus(
    dim: usize,
    points: &[f64],
    shifted: &[f64],
    base: &mut [i64],
    alt: &mut [i64],
) {
    for vector in 0..points.len() / dim {
        let rows = vector * dim..(vector + 1) * dim;
        let (x, s) = (&points[rows.clone()], &shifted[rows.clone()]);
        let (b, a) = (&mut base[rows.clone()], &mut alt[rows]);
        fixup_dn_scalar(x, b);
        fixup_dn_scalar(s, a);

        let plain = distance_sq_scalar(x, b, 0.0);
        let shifted_distance = distance_sq_scalar(x, a, 0.5);

        // Strict `<` prefers the D_n coset on a tie (invariant I2).
        if shifted_distance < plain {
            for (dst, &v) in b.iter_mut().zip(a.iter()) {
                *dst = 2 * v + 1;
            }
        } else {
            for dst in b.iter_mut() {
                *dst *= 2;
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn backend() -> Backend {
    use std::sync::LazyLock;
    static BACKEND: LazyLock<Backend> = LazyLock::new(|| {
        Selection::new("SIMD_BACKEND")
            .supports(ENGINE_TIERS)
            .resolve()
    });
    *BACKEND
}

/// Unstable implementation access for differential tests and benchmarks.
#[cfg(feature = "internals")]
pub mod internals {
    /// Portable scalar reference for the flat rounding pass.
    pub fn round_plane_scalar(x: &[f64], out: &mut [i64]) {
        super::round_plane_scalar(x, out);
    }

    /// The dispatch threshold applied by the batch kernel.
    pub const DISPATCH_MIN_VECTORS: usize = super::DISPATCH_MIN_VECTORS;
}
