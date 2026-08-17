//! Acceptance tests: Babai rounding and nearest-plane.
//!
//! # The oracle is exhaustive search, not a stored answer
//!
//! `exact_nearest` walks a box that provably contains the true nearest point
//! (sized by Babai's own answer through `G⁻¹ = adj(G)/det(G)`), so the oracle
//! is independent of the code it checks. The inequality "Babai never beats the
//! optimum" is the gate; how often it matches is the quality measurement.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use lattica::basis::Gram;
use lattica::gso::Gso;
use lattica::named::zn;
use lattica::reduce::{Delta, Reduced, lll};
use lattice_engine::babai;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn signed(&mut self, bound: i64) -> i64 {
        i64::try_from(self.next() % u64::try_from(2 * bound + 1).unwrap()).unwrap() - bound
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// A random positive-definite Gram matrix, from a random full-rank basis.
fn random_gram(rng: &mut Rng, n: usize, bound: i64) -> Option<Gram<i64>> {
    let data: Vec<i64> = (0..n * n).map(|_| rng.signed(bound)).collect();
    let basis = lattica::Basis::from_rows(n, n, &data).ok()?;
    let gram = basis.gram().ok()?;
    if gram.det().ok()? <= 0 {
        return None;
    }
    Gso::new(&gram).ok()?;
    Some(gram)
}

/// The reduction certificate, re-derived in integers: the transform is
/// unimodular, `U G Uᵀ` reproduces the output, the output is reduced, and the
/// determinant is invariant.
fn check_certificate(original: &Gram<i64>, reduced: &Reduced<i64>, delta: Delta) {
    assert_eq!(
        reduced.transform.det().unwrap().abs(),
        1,
        "transform is not unimodular"
    );

    let congruence = reduced
        .transform
        .mul(original.as_matrix())
        .unwrap()
        .mul(&reduced.transform.transpose().unwrap())
        .unwrap();
    assert_eq!(&congruence, reduced.gram.as_matrix(), "U G U^T != G_out");

    assert!(
        lattica::reduce::is_reduced(&reduced.gram, delta).unwrap(),
        "output is not LLL-reduced"
    );

    assert_eq!(reduced.gram.det().unwrap(), original.det().unwrap());
    assert_eq!(reduced.gram.dim(), original.dim());
}

/// The true nearest lattice point to the coefficient vector `t`, by exhaustive
/// search over a box that provably contains it.
///
/// If `z0` is any lattice point at squared distance `R²`, then any nearer point
/// satisfies `(t_i - z_i)² ≤ R²·(G⁻¹)_ii`, and `G⁻¹ = adj(G)/det(G)`. The box
/// is therefore exact, and Babai's own answer supplies `R²`.
fn exact_nearest(gram: &Gram<i64>, t: &[f64], seed: &[i64]) -> (Vec<i64>, f64) {
    let n = gram.dim();
    let det = gram.det().unwrap() as f64;
    let adj = gram.adjugate().unwrap();

    let mut best = seed.to_vec();
    let mut best_distance = babai::distance_sq(gram, t, seed).unwrap();

    let widths: Vec<i64> = (0..n)
        .map(|i| {
            let variance = best_distance * (adj.entry(i, i) as f64) / det;
            (variance.max(0.0).sqrt() + 1.0).ceil() as i64
        })
        .collect();
    let centre: Vec<i64> = t.iter().map(|v| v.round() as i64).collect();
    let lo: Vec<i64> = (0..n).map(|i| centre[i] - widths[i]).collect();
    let hi: Vec<i64> = (0..n).map(|i| centre[i] + widths[i]).collect();

    let mut z = lo.clone();
    loop {
        let d = babai::distance_sq(gram, t, &z).unwrap();
        if d < best_distance {
            best_distance = d;
            best.copy_from_slice(&z);
        }
        let mut i = 0;
        while i < n {
            z[i] += 1;
            if z[i] <= hi[i] {
                break;
            }
            z[i] = lo[i];
            i += 1;
        }
        if i == n {
            break;
        }
    }
    (best, best_distance)
}

#[test]
fn babai_never_beats_the_true_nearest_point_and_often_matches_it() {
    let mut rng = Rng(0x66AA_7799_8811_9922);
    let mut exact_hits = 0usize;
    let mut trials = 0usize;

    for n in 2..=4 {
        for _ in 0..30 {
            let Some(gram) = random_gram(&mut rng, n, 4) else {
                continue;
            };
            // Reduce first: nearest-plane is only as good as its basis.
            let Ok(reduced) = lll(&gram, Delta::STRONG) else {
                continue;
            };
            let Ok(gso) = Gso::new(&reduced.gram) else {
                continue;
            };

            for _ in 0..20 {
                let t: Vec<f64> = (0..n).map(|_| (rng.unit() - 0.5) * 8.0).collect();

                let mut work = t.clone();
                let mut plane = vec![0i64; n];
                babai::nearest_plane(&gso, &mut work, &mut plane).unwrap();
                let plane_distance = babai::distance_sq(&reduced.gram, &t, &plane).unwrap();

                let (_, optimal) = exact_nearest(&reduced.gram, &t, &plane);

                assert!(
                    plane_distance >= optimal - 1e-9,
                    "Babai beat the optimum, so the oracle is wrong"
                );
                trials += 1;
                if (plane_distance - optimal).abs() < 1e-9 {
                    exact_hits += 1;
                }

                // Rounding is never better than nearest-plane's guarantee, but
                // it must still land on a lattice point at a finite distance.
                let mut rounded = vec![0i64; n];
                babai::round(&t, &mut rounded).unwrap();
                assert!(babai::distance_sq(&reduced.gram, &t, &rounded).unwrap() >= optimal - 1e-9);
            }
        }
    }

    assert!(trials > 300, "only {trials} Babai trials ran");
    // On a strongly reduced basis in low dimension it should usually be exact;
    // this is a quality measurement, and the inequality above is the gate.
    println!("nearest-plane was optimal on {exact_hits} of {trials} targets");
    assert!(
        exact_hits * 10 >= trials * 7,
        "nearest-plane was optimal only {exact_hits}/{trials} times"
    );
}

#[test]
fn nearest_plane_is_exact_on_the_integer_lattice_at_any_dimension() {
    let mut rng = Rng(0x7799_8811_9922_AA33);
    for n in [2usize, 5, 8, 16] {
        let gram = zn::<i64>(n).unwrap();
        let gso = Gso::new(&gram).unwrap();
        for _ in 0..200 {
            let t: Vec<f64> = (0..n).map(|_| (rng.unit() - 0.5) * 20.0).collect();
            let mut work = t.clone();
            let mut plane = vec![0i64; n];
            babai::nearest_plane(&gso, &mut work, &mut plane).unwrap();

            let mut rounded = vec![0i64; n];
            babai::round(&t, &mut rounded).unwrap();
            assert_eq!(plane, rounded, "Z^n: nearest-plane is plain rounding");
        }
    }
}

#[test]
fn reduction_improves_what_babai_can_do() {
    // The point of reducing before decoding: on a skewed basis nearest-plane is
    // poor, and on the reduced basis of the *same lattice* it is not.
    let gram = Gram::<i64>::from_rows(2, &[10_001, 9999, 9999, 10_001]).unwrap();
    let reduced = lll(&gram, Delta::STRONG).unwrap();
    check_certificate(&gram, &reduced, Delta::STRONG);

    let raw = Gso::new(&gram).unwrap();
    let good = Gso::new(&reduced.gram).unwrap();

    let mut skewed_loss = 0.0f64;
    let mut reduced_loss = 0.0f64;
    let mut rng = Rng(0x8811_9922_AA33_BB44);
    for _ in 0..200 {
        let t: Vec<f64> = (0..2).map(|_| (rng.unit() - 0.5) * 6.0).collect();

        let mut work = t.clone();
        let mut z = [0i64; 2];
        babai::nearest_plane(&raw, &mut work, &mut z).unwrap();
        let (_, optimal) = exact_nearest(&gram, &t, &z);
        skewed_loss += babai::distance_sq(&gram, &t, &z).unwrap() - optimal;

        // The reduced basis expresses the same lattice, so compare against its
        // own optimum.
        let mut work = t.clone();
        let mut z = [0i64; 2];
        babai::nearest_plane(&good, &mut work, &mut z).unwrap();
        let (_, optimal) = exact_nearest(&reduced.gram, &t, &z);
        reduced_loss += babai::distance_sq(&reduced.gram, &t, &z).unwrap() - optimal;
    }
    assert!(
        reduced_loss < skewed_loss,
        "reduction did not help: {reduced_loss} vs {skewed_loss}"
    );
}
