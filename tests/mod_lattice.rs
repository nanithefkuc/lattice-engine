//! Acceptance tests: `mod Λ`, dithering, and the shaping gain.
//!
//! The `mod Λ` identities are asserted *exactly*. Query points are dyadic and
//! lattice points are integers or half-integers, so every intermediate is
//! exactly representable and an approximate comparison would only hide bugs.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use lattica::named::e8_generator;
use lattice_engine::{
    Dn, Quantizer, Scaled, Scratch, Zn, e8 as e8_decoder, mod_lattice, mod_lattice_dithered,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A multiple of `2^-8` in `[-8, 8)`.
    fn dyadic(&mut self) -> f64 {
        (f64::from(u32::try_from(self.next() % 4096).unwrap()) - 2048.0) / 256.0
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

// ---------------------------------------------------------------- mod lattice

#[test]
fn reduction_is_idempotent_and_lands_in_the_voronoi_region() {
    let mut rng = Rng(0xA1B2_C3D4_E5F6_0718);
    let q = e8_decoder();
    let mut scratch = Scratch::new(8);
    let (mut once, mut twice) = ([0.0f64; 8], [0.0f64; 8]);
    let mut point = [0i64; 8];

    for _ in 0..2000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        mod_lattice(&q, &x, &mut once, &mut scratch).unwrap();
        let saved = once;
        mod_lattice(&q, &saved, &mut twice, &mut scratch).unwrap();
        assert_eq!(once, twice, "(x mod L) mod L != x mod L");

        q.nearest(&once, &mut point, &mut scratch).unwrap();
        assert_eq!(point, [0i64; 8], "the residual left the Voronoi region");
    }
}

#[test]
fn reduction_is_invariant_under_adding_a_lattice_vector() {
    // For an *integral* lattice vector the invariance is exact: rounding is
    // unchanged by an integer shift, so the residuals agree coordinate for
    // coordinate.
    let mut rng = Rng(0xB2C3_D4E5_F607_1829);
    let q = Dn::new(8).unwrap();
    let mut scratch = Scratch::new(8);
    let (mut plain, mut shifted) = ([0.0f64; 8], [0.0f64; 8]);

    for _ in 0..1000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        // An element of D_8: integral with an even coordinate sum.
        let mut lambda: Vec<f64> = (0..8)
            .map(|_| f64::from(i32::try_from(rng.next() % 7).unwrap()) - 3.0)
            .collect();
        if lambda.iter().sum::<f64>() % 2.0 != 0.0 {
            lambda[0] += 1.0;
        }
        let moved: Vec<f64> = x.iter().zip(&lambda).map(|(a, b)| a + b).collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice(&q, &moved, &mut shifted, &mut scratch).unwrap();
        assert_eq!(plain, shifted, "(x + lambda) mod L != x mod L");
    }
}

#[test]
fn distance_to_the_lattice_is_translation_invariant() {
    // The general statement, and the strongest one available. Translating by a
    // lattice vector cannot change the distance to the lattice -- but it *can*
    // change which of several equidistant nearest points is chosen, because
    // the tie rules are index-based and the D_n^+ coset preference is not
    // symmetric under a glue-vector shift. So the invariant is asserted on the
    // norm, which is a property of the lattice, rather than on the point,
    // which is a property of the specification. Same shape of limitation as
    // the negation caveat in the quantizer tests.
    let mut rng = Rng(0xB2C3_D4E5_F607_182A);
    let q = e8_decoder();
    let basis = e8_generator();
    let mut scratch = Scratch::new(8);
    let (mut plain, mut shifted) = ([0.0f64; 8], [0.0f64; 8]);
    let mut boundary_hits = 0usize;

    for _ in 0..2000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        let mut lambda = [0.0f64; 8];
        for row in &basis {
            let c = f64::from(i32::try_from(rng.next() % 5).unwrap()) - 2.0;
            for (dst, &b) in lambda.iter_mut().zip(row) {
                *dst += c * b;
            }
        }
        let moved: Vec<f64> = x.iter().zip(&lambda).map(|(a, b)| a + b).collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice(&q, &moved, &mut shifted, &mut scratch).unwrap();

        let a: f64 = plain.iter().map(|v| v * v).sum();
        let b: f64 = shifted.iter().map(|v| v * v).sum();
        assert_eq!(a, b, "distance to the lattice changed under translation");

        if plain != shifted {
            // The two residuals must then differ by a lattice vector.
            let diff: Vec<f64> = plain.iter().zip(&shifted).map(|(p, q)| p - q).collect();
            let mut point = [0i64; 8];
            let mut residue = [0.0f64; 8];
            mod_lattice(&q, &diff, &mut residue, &mut scratch).unwrap();
            q.nearest(&diff, &mut point, &mut scratch).unwrap();
            assert!(
                residue.iter().all(|v| v.abs() < 1e-12),
                "the two residuals differ by a non-lattice vector"
            );
            boundary_hits += 1;
        }
    }
    // The caveat is real, not theoretical: it fires on this input set.
    assert!(boundary_hits > 0, "no boundary case was exercised");
}

#[test]
fn reduction_distributes_over_addition() {
    // ((a mod L) + b) mod L == (a + b) mod L, because Q is Λ-periodic.
    let mut rng = Rng(0xC3D4_E5F6_0718_293A);
    let q = Dn::new(6).unwrap();
    let mut scratch = Scratch::new(6);
    let (mut folded, mut direct, mut partial) = ([0.0f64; 6], [0.0f64; 6], [0.0f64; 6]);

    for _ in 0..2000 {
        let a: Vec<f64> = (0..6).map(|_| rng.dyadic()).collect();
        let b: Vec<f64> = (0..6).map(|_| rng.dyadic()).collect();

        mod_lattice(&q, &a, &mut partial, &mut scratch).unwrap();
        let mixed: Vec<f64> = partial.iter().zip(&b).map(|(p, v)| p + v).collect();
        mod_lattice(&q, &mixed, &mut folded, &mut scratch).unwrap();

        let sum: Vec<f64> = a.iter().zip(&b).map(|(p, v)| p + v).collect();
        mod_lattice(&q, &sum, &mut direct, &mut scratch).unwrap();

        assert_eq!(folded, direct, "mod is not distributive");
    }
}

#[test]
fn dithering_round_trips() {
    // ((x + d) mod L) - d recovers the plain residual shifted by the dither,
    // and is exactly x mod L when d is itself a lattice point.
    let mut rng = Rng(0xD4E5_F607_1829_3A4B);
    let q = Zn::new(5).unwrap();
    let mut scratch = Scratch::new(5);
    let (mut plain, mut dithered) = ([0.0f64; 5], [0.0f64; 5]);

    for _ in 0..1000 {
        let x: Vec<f64> = (0..5).map(|_| rng.dyadic()).collect();
        let lattice_dither: Vec<f64> = (0..5)
            .map(|_| f64::from(i32::try_from(rng.next() % 7).unwrap()) - 3.0)
            .collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice_dithered(&q, &x, &lattice_dither, &mut dithered, &mut scratch).unwrap();
        // With a lattice-valued dither the two agree up to a lattice vector:
        // exactly equal away from a Voronoi boundary, and equidistant on one.
        let mut energy_a = 0.0;
        let mut energy_b = 0.0;
        for i in 0..5 {
            let folded = dithered[i] + lattice_dither[i];
            let gap = folded - plain[i];
            assert_eq!(gap, gap.round(), "residuals differ by a non-lattice vector");
            energy_a += folded * folded;
            energy_b += plain[i] * plain[i];
        }
        assert_eq!(energy_a, energy_b, "dithering changed the distance");
    }
}

// ---------------------------------------------------------------- shaping gain

/// The shaping gain of a lattice's Voronoi region over a cube of equal volume,
/// in decibels, measured directly through `mod_lattice`.
fn shaping_gain_db<Q: Quantizer>(
    coding: &Q,
    basis: &[[f64; 8]; 8],
    factor: i64,
    samples: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let dim = coding.dim();
    let shaping = Scaled::new(coding, factor).unwrap();
    let mut scratch = Scratch::new(dim);
    let mut residual = vec![0.0f64; dim];

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..samples {
        // Uniform over the fundamental parallelepiped of factor*Λ, which
        // `mod_lattice` folds onto the Voronoi region.
        let mut raw = vec![0.0f64; dim];
        for row in basis {
            let u = rng.unit() * factor as f64;
            for (dst, &b) in raw.iter_mut().zip(row) {
                *dst += u * b;
            }
        }
        mod_lattice(&shaping, &raw, &mut residual, &mut scratch).unwrap();
        let energy: f64 = residual.iter().map(|v| v * v).sum();
        sum += energy;
        sum_sq += energy * energy;
    }

    let count = samples as f64;
    let mean = sum / count;
    let power = mean / dim as f64;
    let variance = (sum_sq / count - mean * mean).max(0.0);
    let standard_error = (variance / count).sqrt() / dim as f64;

    let cube = (factor * factor) as f64 / 12.0;
    let gain = 10.0 * (cube / power).log10();
    let error = 10.0 / std::f64::consts::LN_10 * standard_error / power;
    (gain, error)
}

#[test]
fn e8_reproduces_its_published_shaping_gain() {
    // THE RELEASE GATE. G(E_8) = 0.0716821 against G(cube) = 1/12 gives
    // 10*log10((1/12)/0.0716821) = 0.6539 dB. The tolerance is five standard
    // errors of the sample, so it cannot be widened to pass.
    let mut rng = Rng(0x0E80_0E80_0E80_0E81);
    let (gain, error) = shaping_gain_db(&e8_decoder(), &e8_generator(), 4, 200_000, &mut rng);
    assert!(
        (gain - 0.6539).abs() <= 5.0 * error,
        "E_8 shaping gain {gain:.4} dB +/- {error:.4}, published 0.6539 dB"
    );
    // Sanity: it must be well under the ultimate 1.5329 dB and well over zero.
    assert!(gain > 0.4 && gain < 1.0);
}
