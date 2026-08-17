//! x86-64 kernels for the engine-owned decode layouts.
//!
//! Four coordinates per register; every lane is an independent coordinate of
//! the flat batch, so no lane ever observes another. The operation set is the
//! scalar one — truncate toward zero, one exact subtraction, two exact
//! comparisons, integer adjusts — with no FMA and no reassociation anywhere.

use archmage::prelude::*;

/// Rounds one flat plane of validated coordinates, ties away from zero.
///
/// Lane values finish as integer-valued doubles; AVX2 has no packed
/// `f64 -> i64` convert, so conversion lands through an exact per-tile scalar
/// cast, which is exact for every integer-valued double within `i64`.
#[allow(
    clippy::used_underscore_binding,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
#[arcane(import_intrinsics)]
pub(super) fn round_plane_avx2(_token: X64V3Token, x: &[f64], out: &mut [i64]) {
    let half = _mm256_set1_pd(0.5);
    let minus_half = _mm256_set1_pd(-0.5);
    let one = _mm256_set1_pd(1.0);
    let minus_one = _mm256_set1_pd(-1.0);

    let tile_end = x.len() / 4 * 4;
    let mut tile = [0.0f64; 4];
    for (values, destination) in x[..tile_end]
        .chunks_exact(4)
        .zip(out[..tile_end].chunks_exact_mut(4))
    {
        let values: &[f64; 4] = values.try_into().unwrap();
        let v = _mm256_loadu_pd(values);
        // Truncate toward zero: exact, and the scalar `as i64` truncation
        // for every value the batch validation already bounded.
        let truncated = _mm256_round_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let fraction = _mm256_sub_pd(v, truncated);
        let up = _mm256_cmp_pd(fraction, half, _CMP_GE_OQ);
        let down = _mm256_cmp_pd(fraction, minus_half, _CMP_LE_OQ);
        // `mask & 1.0` is 1.0 or 0.0; `mask & -1.0` is -1.0 or 0.0. The
        // adjust operands are the integers, not the tie bounds.
        let adjusted = _mm256_add_pd(
            _mm256_add_pd(truncated, _mm256_and_pd(up, one)),
            _mm256_and_pd(down, minus_one),
        );
        _mm256_storeu_pd(&mut tile, adjusted);
        let destination: &mut [i64; 4] = destination.try_into().unwrap();
        for (dst, value) in destination.iter_mut().zip(tile) {
            *dst = value as i64;
        }
    }
    for (dst, &value) in out[tile_end..].iter_mut().zip(x[tile_end..].iter()) {
        *dst = crate::round_away(value);
    }
}
