//! Acceptance tests: Construction A decoding through the code seam.
//!
//! The mock codes are deliberately exhaustive decoders, so the lattice decode
//! they support is maximum-likelihood and can be checked against brute force.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use std::num::NonZeroU32;

use lattica::construct::construction_a_basis;
use lattica::int::IntMatrix;
use lattica::named::d_n;
use lattice_engine::{CodeMembership, ConstructionA, DecodeError, Dn, Quantizer, Scratch};

#[cfg(miri)]
const ML_CASES: usize = 3;
#[cfg(not(miri))]
const ML_CASES: usize = 3_000;
#[cfg(miri)]
const DIFFERENTIAL_CASES: usize = 5;
#[cfg(not(miri))]
const DIFFERENTIAL_CASES: usize = 5_000;

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
}

/// The single parity check code over `Z_2`, length 4. Construction A over it is
/// `D_4`, which the stack can also build two other ways.
struct ParityCheck;

impl CodeMembership for ParityCheck {
    fn modulus(&self) -> NonZeroU32 {
        NonZeroU32::new(2).unwrap()
    }
    fn length(&self) -> usize {
        4
    }
    fn cardinality(&self) -> u64 {
        8
    }
    fn contains(&self, residues: &[u32]) -> bool {
        residues.iter().sum::<u32>() % 2 == 0
    }
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError> {
        // Exhaustive: 16 words, keep the even-weight one of least cost.
        let mut best = f64::INFINITY;
        let mut best_word = 0usize;
        for word in 0..16usize {
            if (word.count_ones() % 2) != 0 {
                continue;
            }
            let total: f64 = (0..4).map(|i| costs[i * 2 + ((word >> i) & 1)]).sum();
            if total < best {
                best = total;
                best_word = word;
            }
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::try_from((best_word >> i) & 1).unwrap();
        }
        Ok(())
    }
}

/// The ternary repetition code of length 3: `{(a, a, a)}` over `Z_3`.
struct Repetition3;

impl CodeMembership for Repetition3 {
    fn modulus(&self) -> NonZeroU32 {
        NonZeroU32::new(3).unwrap()
    }
    fn length(&self) -> usize {
        3
    }
    fn cardinality(&self) -> u64 {
        3
    }
    fn contains(&self, residues: &[u32]) -> bool {
        residues.iter().all(|&r| r == residues[0])
    }
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError> {
        let mut best = f64::INFINITY;
        let mut best_symbol = 0u32;
        for s in 0..3usize {
            let total: f64 = (0..3).map(|i| costs[i * 3 + s]).sum();
            if total < best {
                best = total;
                best_symbol = u32::try_from(s).unwrap();
            }
        }
        out.fill(best_symbol);
        Ok(())
    }
}

#[test]
fn construction_a_covolume_is_q_to_the_redundancy() {
    let parity = ConstructionA::new(ParityCheck).unwrap();
    // q^n / |C| = 2^4 / 8 = 2 = q^(n-k) with k = 3.
    assert_eq!(parity.covolume().unwrap(), 2);

    let repetition = ConstructionA::new(Repetition3).unwrap();
    // 3^3 / 3 = 9 = q^(n-k) with k = 1.
    assert_eq!(repetition.covolume().unwrap(), 9);
}

#[test]
fn construction_a_over_the_parity_code_is_the_checkerboard_lattice() {
    let lattice = ConstructionA::new(ParityCheck).unwrap();
    // Membership agrees with D_4's definition.
    for a in -3..=3i64 {
        for b in -3..=3i64 {
            for c in -3..=3i64 {
                for d in -3..=3i64 {
                    let point = [a, b, c, d];
                    let want = (a + b + c + d) % 2 == 0;
                    assert_eq!(lattice.contains(&point).unwrap(), want);
                }
            }
        }
    }
    // And so does the generator-matrix route.
    let generator =
        IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
    let basis = construction_a_basis(2i64, &generator).unwrap();
    assert_eq!(
        basis.gram().unwrap().det().unwrap(),
        d_n::<i64>(4).unwrap().det().unwrap()
    );
}

fn scaled_coordinates(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&coordinate| (coordinate * 256.0) as i64)
        .collect()
}

fn scaled_squared_distance(x: &[i64], point: &[i64]) -> i128 {
    x.iter()
        .zip(point)
        .map(|(&coordinate, &lattice_coordinate)| {
            let residual = i128::from(coordinate) - 256 * i128::from(lattice_coordinate);
            residual * residual
        })
        .sum()
}

/// Minimal squared distance from `x` to the lattice, by exhaustive search over
/// a box that provably contains the answer.
fn brute_force_min<C: CodeMembership>(lattice: &ConstructionA<C>, x: &[f64]) -> i128 {
    let n = x.len();
    // Rounding to the nearest integer point and repairing it into the lattice
    // gives some valid upper bound; a box of half-width `n` around `x` is
    // vastly more than enough for the toy geometries here.
    let lo: Vec<i64> = x.iter().map(|v| (v - 3.0).ceil() as i64).collect();
    let hi: Vec<i64> = x.iter().map(|v| (v + 3.0).floor() as i64).collect();
    let scaled = scaled_coordinates(x);

    let mut best = i128::MAX;
    let mut v = lo.clone();
    loop {
        if lattice.contains(&v).unwrap() {
            let d = scaled_squared_distance(&scaled, &v);
            if d < best {
                best = d;
            }
        }
        let mut i = 0;
        while i < n {
            v[i] += 1;
            if v[i] <= hi[i] {
                break;
            }
            v[i] = lo[i];
            i += 1;
        }
        if i == n {
            break;
        }
    }
    best
}

#[test]
fn construction_a_decoding_is_maximum_likelihood() {
    // The soft-cost seam is what makes this exact. A hard-decision seam would
    // land on a nearby lattice point that is not always the nearest, and the
    // only symptom would be a slightly worse error rate.
    let mut rng = Rng(0xE5F6_0718_293A_4B5C);

    let parity = ConstructionA::new(ParityCheck).unwrap();
    let mut scratch = Scratch::new(4);
    let mut out = [0i64; 4];
    for _ in 0..ML_CASES {
        let x: Vec<f64> = (0..4).map(|_| rng.dyadic()).collect();
        parity.nearest(&x, &mut out, &mut scratch).unwrap();
        assert!(parity.contains(&out).unwrap());
        let got = scaled_squared_distance(&scaled_coordinates(&x), &out);
        assert_eq!(got, brute_force_min(&parity, &x), "parity: {x:?}");
    }

    let repetition = ConstructionA::new(Repetition3).unwrap();
    let mut out = [0i64; 3];
    for _ in 0..ML_CASES {
        let x: Vec<f64> = (0..3).map(|_| rng.dyadic()).collect();
        repetition.nearest(&x, &mut out, &mut scratch).unwrap();
        assert!(repetition.contains(&out).unwrap());
        let got = scaled_squared_distance(&scaled_coordinates(&x), &out);
        assert_eq!(got, brute_force_min(&repetition, &x), "repetition: {x:?}");
    }
}

#[test]
fn construction_a_over_the_parity_code_matches_the_closed_form_decoder() {
    // Two decoders with nothing in common -- a soft-decision code search over
    // Z_2 versus the Conway-Sloane f/g construction -- for the same lattice.
    let mut rng = Rng(0xF607_1829_3A4B_5C6D);
    let coded = ConstructionA::new(ParityCheck).unwrap();
    let closed = Dn::new(4).unwrap();
    let mut scratch = Scratch::new(4);
    let (mut a, mut b) = ([0i64; 4], [0i64; 4]);

    for _ in 0..DIFFERENTIAL_CASES {
        let x: Vec<f64> = (0..4).map(|_| rng.dyadic()).collect();
        coded.nearest(&x, &mut a, &mut scratch).unwrap();
        closed.nearest(&x, &mut b, &mut scratch).unwrap();
        let scaled = scaled_coordinates(&x);
        let da = scaled_squared_distance(&scaled, &a);
        let db = scaled_squared_distance(&scaled, &b);
        assert_eq!(da, db, "decoders disagree on distance for {x:?}");
    }
}

#[test]
fn construction_a_rejects_bad_geometry() {
    let lattice = ConstructionA::new(ParityCheck).unwrap();
    assert!(lattice.contains(&[1, 2, 3]).is_err());
    let mut scratch = Scratch::new(4);
    let mut out = [0i64; 4];
    assert!(
        lattice
            .nearest(&[1.0, f64::NAN, 0.0, 0.0], &mut out, &mut scratch)
            .is_err()
    );
}
