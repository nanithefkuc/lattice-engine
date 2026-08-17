//! Acceptance tests: the closed-form quantizers.
//!
//! # Two oracles, each provably correct, covering different ground
//!
//! **Exhaustive box search** (`brute_force_min`) at dimension ≤ 6. A crude but
//! obviously-valid lattice point gives an upper bound `R` on the distance to
//! the nearest one; every candidate then satisfies `|v_i - x_i| ≤ R`
//! coordinatewise, so enumerating that box and filtering by membership cannot
//! miss the minimum. This probes Voronoi boundaries and ties, which is where
//! decoders are wrong.
//!
//! **Packing-radius perturbation** at every dimension including 8, 16 and 24.
//! If `‖x - v‖ < d_min/2` then `v` is the *unique* nearest lattice point, no
//! search required. Perturbing a known lattice point by less than the packing
//! radius therefore gives a correct answer for free, at any dimension the box
//! search could never reach.
//!
//! # Why the comparisons are exact
//!
//! Query coordinates are multiples of `2^-10` with magnitude below 4. Lattice
//! coordinates are integers or half-integers, so every difference is a multiple
//! of `2^-10`, every square a multiple of `2^-20` below 256, and every sum of
//! nine such squares needs 28 significand bits. All of it is exact in binary64.
//! A disagreement between oracle and decoder is therefore a real bug and never
//! an artifact of the comparison.
//!
//! The differential asserts *equal minimal distance*, not an equal point: at a
//! tie any nearest point is correct, and which one is returned is pinned
//! separately by `tests/vectors.rs`.

// The oracles here juggle scaled integer coordinates against real ones, so
// widening and narrowing casts are the substance of the file rather than an
// accident. Every one is on a value bounded well inside its target type.
//
// `float_cmp` is allowed for the same reason it is normally denied: query
// coordinates are dyadic and the squared distances are therefore exact, so
// `==` between two of them is the strongest available assertion. An
// approximate comparison here would hide precisely the bugs being hunted.
#![allow(
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use lattice_engine::{An, Dn, DnPlus, Quantizer, Scratch, Zn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Zn,
    Dn,
    An,
    DnPlus,
}

/// Ambient dimension for a family at parameter `n`.
fn ambient(family: Family, n: usize) -> usize {
    match family {
        Family::An => n + 1,
        _ => n,
    }
}

/// Output scale: coordinates are returned as `scale * v`.
fn scale_of(family: Family) -> i64 {
    match family {
        Family::DnPlus => 2,
        _ => 1,
    }
}

/// Is the scaled integer vector `v` a lattice point?
fn contains(family: Family, v: &[i64]) -> bool {
    match family {
        Family::Zn => true,
        Family::Dn => v.iter().sum::<i64>() % 2 == 0,
        Family::An => v.iter().sum::<i64>() == 0,
        Family::DnPlus => {
            // Doubled coordinates: all even (the D_n coset) or all odd (the
            // shifted coset), and in both cases the underlying D_n point must
            // have an even coordinate sum.
            let parity = v[0] & 1;
            if !v.iter().all(|&c| c & 1 == parity) {
                return false;
            }
            let halves: i64 = v.iter().map(|&c| (c - parity) / 2).sum();
            halves % 2 == 0
        }
    }
}

/// Some lattice point near `x`, by a construction that is obviously valid and
/// shares no logic with the decoders under test.
fn crude_point(family: Family, x: &[f64]) -> Vec<i64> {
    let m = x.len();
    match family {
        Family::Zn => x.iter().map(|v| v.round() as i64).collect(),
        Family::Dn => {
            let mut v: Vec<i64> = x.iter().map(|c| c.round() as i64).collect();
            if v.iter().sum::<i64>() % 2 != 0 {
                v[0] += 1;
            }
            v
        }
        Family::An => {
            let mean = x.iter().sum::<f64>() / m as f64;
            let mut v: Vec<i64> = x.iter().map(|c| (c - mean).round() as i64).collect();
            let excess: i64 = v.iter().sum();
            v[0] -= excess;
            v
        }
        Family::DnPlus => {
            let mut v: Vec<i64> = x.iter().map(|c| c.round() as i64).collect();
            if v.iter().sum::<i64>() % 2 != 0 {
                v[0] += 1;
            }
            v.iter().map(|c| 2 * c).collect()
        }
    }
}

/// Squared distance from `x` to the lattice point whose scaled coordinates are
/// `v`.
fn distance_sq(x: &[f64], v: &[i64], scale: i64) -> f64 {
    let s = scale as f64;
    let mut total = 0.0;
    for (&xi, &vi) in x.iter().zip(v) {
        let d = xi - vi as f64 / s;
        total += d * d;
    }
    total
}

/// Minimal squared distance from `x` to the lattice, by exhaustive search over
/// a provably sufficient box.
fn brute_force_min(family: Family, x: &[f64]) -> f64 {
    let m = x.len();
    let scale = scale_of(family);
    let s = scale as f64;

    let seed = crude_point(family, x);
    assert!(
        contains(family, &seed),
        "the crude construction left the lattice"
    );
    let bound_sq = distance_sq(x, &seed, scale);
    // A hair of slack absorbs the rounding in `sqrt`; an insufficient box would
    // show up as the oracle losing to the decoder, which the caller asserts
    // against, so this can only cost time and never correctness.
    let bound = bound_sq.sqrt() * (1.0 + 1e-9) + 1e-9;

    let lo: Vec<i64> = (0..m).map(|i| ((x[i] - bound) * s).ceil() as i64).collect();
    let hi: Vec<i64> = (0..m)
        .map(|i| ((x[i] + bound) * s).floor() as i64)
        .collect();

    let mut best = bound_sq;
    let mut v = lo.clone();
    loop {
        if contains(family, &v) {
            let d = distance_sq(x, &v, scale);
            if d < best {
                best = d;
            }
        }
        let mut i = 0;
        while i < m {
            v[i] += 1;
            if v[i] <= hi[i] {
                break;
            }
            v[i] = lo[i];
            i += 1;
        }
        if i == m {
            break;
        }
    }
    best
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A multiple of `2^-10` in `[-4, 4)`, so that squared distances are exact
    /// in binary64.
    fn dyadic(&mut self) -> f64 {
        (f64::from(u32::try_from(self.next() % 8192).unwrap()) - 4096.0) / 1024.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        // 53 bits of mantissa from the top of the word.
        ((self.next() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn quantizer(family: Family, n: usize) -> Box<dyn Quantizer> {
    match family {
        Family::Zn => Box::new(Zn::new(n).unwrap()),
        Family::Dn => Box::new(Dn::new(n).unwrap()),
        Family::An => Box::new(An::new(n).unwrap()),
        Family::DnPlus => Box::new(DnPlus::new(n).unwrap()),
    }
}

/// Every (family, parameter) pair the box search can reach.
fn small_cases() -> Vec<(Family, usize)> {
    let mut cases = Vec::new();
    for n in 1..=6 {
        cases.push((Family::Zn, n));
    }
    for n in 2..=6 {
        cases.push((Family::Dn, n));
    }
    for n in 1..=5 {
        cases.push((Family::An, n));
    }
    for n in [2usize, 4, 6] {
        cases.push((Family::DnPlus, n));
    }
    cases
}

#[test]
fn closed_forms_attain_the_true_minimum() {
    let mut rng = Rng(0x1357_9BDF_0246_8ACE);
    for (family, n) in small_cases() {
        let m = ambient(family, n);
        let q = quantizer(family, n);
        let mut scratch = Scratch::new(m);
        let mut out = vec![0i64; m];

        for _ in 0..800 {
            let x: Vec<f64> = (0..m).map(|_| rng.dyadic()).collect();
            q.nearest(&x, &mut out, &mut scratch).unwrap();

            assert!(
                contains(family, &out),
                "{family:?}({n}) returned a non-lattice point {out:?} for {x:?}"
            );
            let got = distance_sq(&x, &out, scale_of(family));
            let want = brute_force_min(family, &x);
            assert_eq!(
                got, want,
                "{family:?}({n}) is not nearest for {x:?}: got {got}, best {want}"
            );
        }
    }
}

/// A random lattice point, in scaled coordinates.
fn random_point(family: Family, m: usize, rng: &mut Rng) -> Vec<i64> {
    let mut v: Vec<i64> = (0..m)
        .map(|_| i64::try_from(rng.below(21)).unwrap() - 10)
        .collect();
    match family {
        Family::Zn => v,
        Family::Dn => {
            if v.iter().sum::<i64>() % 2 != 0 {
                v[0] += 1;
            }
            v
        }
        Family::An => {
            let excess: i64 = v.iter().sum();
            v[0] -= excess;
            v
        }
        Family::DnPlus => {
            if v.iter().sum::<i64>() % 2 != 0 {
                v[0] += 1;
            }
            let coset = rng.below(2);
            v.iter()
                .map(|&c| 2 * c + i64::try_from(coset).unwrap())
                .collect()
        }
    }
}

/// Squared minimal distance of the lattice, in true coordinates.
fn min_norm_sq(family: Family, n: usize) -> f64 {
    match family {
        Family::Zn => 1.0,
        Family::Dn | Family::An => 2.0,
        // D_n^+ has minimum min(2, n/4): the glue vector has squared norm n/4.
        Family::DnPlus => 2.0f64.min(n as f64 / 4.0),
    }
}

#[test]
fn a_point_inside_the_packing_radius_decodes_to_its_own_centre() {
    // If ‖x - v‖ < d_min/2 then v is the unique nearest lattice point. No
    // search is needed to know the right answer, so this reaches dimensions the
    // box search never could.
    let mut rng = Rng(0x0FED_CBA9_8765_4321);
    let cases: Vec<(Family, usize)> = vec![
        (Family::Zn, 8),
        (Family::Zn, 24),
        (Family::Dn, 8),
        (Family::Dn, 16),
        (Family::Dn, 24),
        (Family::An, 8),
        (Family::An, 15),
        (Family::DnPlus, 8),
        (Family::DnPlus, 16),
        (Family::DnPlus, 24),
    ];

    for (family, n) in cases {
        let m = ambient(family, n);
        let q = quantizer(family, n);
        let mut scratch = Scratch::new(m);
        let mut out = vec![0i64; m];
        let packing = min_norm_sq(family, n).sqrt() / 2.0;
        let scale = scale_of(family) as f64;

        for _ in 0..400 {
            let point = random_point(family, m, &mut rng);
            assert!(contains(family, &point));

            // A random direction, scaled to a norm strictly inside the packing
            // radius.
            let raw: Vec<f64> = (0..m).map(|_| rng.unit() - 0.5).collect();
            let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm == 0.0 {
                continue;
            }
            let factor = packing * 0.98 * rng.unit() / norm;

            let x: Vec<f64> = point
                .iter()
                .zip(&raw)
                .map(|(&p, &d)| p as f64 / scale + d * factor)
                .collect();

            q.nearest(&x, &mut out, &mut scratch).unwrap();
            assert_eq!(
                out, point,
                "{family:?}({n}) missed a point well inside its packing radius"
            );
        }
    }
}

#[test]
fn an_selection_and_sort_paths_agree_exactly() {
    // Not merely on generic inputs: both resolve ties by lowest index, so they
    // must agree everywhere, including on the boundary points where an
    // unstable selection would be free to differ.
    let mut rng = Rng(0x2468_ACE0_1357_9BDF);
    for n in 1..=10usize {
        let m = n + 1;
        let q = An::new(n).unwrap();
        let mut scratch = Scratch::new(m);
        let (mut fast, mut reference) = (vec![0i64; m], vec![0i64; m]);

        for round in 0..3000 {
            // Half the rounds use a coarse grid so exact residual ties are
            // common rather than vanishingly rare.
            let x: Vec<f64> = (0..m)
                .map(|_| {
                    if round % 2 == 0 {
                        (f64::from(u32::try_from(rng.below(9)).unwrap()) - 4.0) / 2.0
                    } else {
                        rng.dyadic()
                    }
                })
                .collect();

            q.nearest(&x, &mut fast, &mut scratch).unwrap();
            q.nearest_sorted(&x, &mut reference, &mut scratch).unwrap();
            assert_eq!(fast, reference, "A_{n} paths diverged on {x:?}");
        }
    }
}

#[test]
fn decoding_is_stable_under_negation_up_to_ties() {
    // The lattices are all symmetric under negation, so the *distance* achieved
    // must be. The chosen point need not be: at a tie the index-based rule
    // cannot flip with the sign, which is a deliberate limitation of I3 rather
    // than an oversight, and is documented on the decoders.
    let mut rng = Rng(0xF0E1_D2C3_B4A5_9687);
    for (family, n) in small_cases() {
        let m = ambient(family, n);
        let q = quantizer(family, n);
        let mut scratch = Scratch::new(m);
        let (mut a, mut b) = (vec![0i64; m], vec![0i64; m]);

        for _ in 0..400 {
            let x: Vec<f64> = (0..m).map(|_| rng.dyadic()).collect();
            let negated: Vec<f64> = x.iter().map(|v| -v).collect();
            q.nearest(&x, &mut a, &mut scratch).unwrap();
            q.nearest(&negated, &mut b, &mut scratch).unwrap();
            assert_eq!(
                distance_sq(&x, &a, scale_of(family)),
                distance_sq(&negated, &b, scale_of(family)),
                "{family:?}({n}) is asymmetric under negation at {x:?}"
            );
        }
    }
}

#[test]
fn every_decoded_point_is_a_lattice_point() {
    let mut rng = Rng(0x5A5A_A5A5_3C3C_C3C3);
    for (family, n) in small_cases() {
        let m = ambient(family, n);
        let q = quantizer(family, n);
        let mut scratch = Scratch::new(m);
        let mut out = vec![0i64; m];
        for _ in 0..500 {
            // Deliberately far from the origin, to exercise the A_n projection.
            let x: Vec<f64> = (0..m).map(|_| rng.dyadic() * 100.0 + 37.0).collect();
            q.nearest(&x, &mut out, &mut scratch).unwrap();
            assert!(contains(family, &out), "{family:?}({n}) left the lattice");
        }
    }
}

/// Basis of each lattice in true ambient coordinates, one vector per row.
fn ambient_basis(family: Family, n: usize) -> Vec<Vec<f64>> {
    let m = ambient(family, n);
    let mut rows = Vec::new();
    match family {
        Family::Zn => {
            for i in 0..n {
                let mut r = vec![0.0; m];
                r[i] = 1.0;
                rows.push(r);
            }
        }
        Family::An => {
            for i in 0..n {
                let mut r = vec![0.0; m];
                r[i] = 1.0;
                r[i + 1] = -1.0;
                rows.push(r);
            }
        }
        Family::Dn => {
            for i in 0..n - 1 {
                let mut r = vec![0.0; m];
                r[i] = 1.0;
                r[i + 1] = -1.0;
                rows.push(r);
            }
            let mut last = vec![0.0; m];
            last[n - 2] = 1.0;
            last[n - 1] = 1.0;
            rows.push(last);
        }
        Family::DnPlus => {
            // 2e_0, then e_i - e_{i-1}, then the glue vector: the standard
            // generator matrix of D_n^+, of determinant 1.
            let mut first = vec![0.0; m];
            first[0] = 2.0;
            rows.push(first);
            for i in 1..n - 1 {
                let mut r = vec![0.0; m];
                r[i] = 1.0;
                r[i - 1] = -1.0;
                rows.push(r);
            }
            rows.push(vec![0.5; m]);
        }
    }
    rows
}

/// Monte-Carlo normalized second moment.
///
/// Sampling uniformly over one fundamental parallelepiped makes the
/// quantization error uniform over the Voronoi region, because the
/// parallelepiped tiles space and quantization folds it onto the cell.
fn normalized_second_moment(
    family: Family,
    n: usize,
    covolume: f64,
    samples: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let m = ambient(family, n);
    let basis = ambient_basis(family, n);
    let rank = basis.len();
    let q = quantizer(family, n);
    let scale = scale_of(family) as f64;
    let mut scratch = Scratch::new(m);
    let mut out = vec![0i64; m];

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for _ in 0..samples {
        let mut x = vec![0.0f64; m];
        for row in basis.iter().take(rank) {
            let u = rng.unit();
            for (dst, &b) in x.iter_mut().zip(row) {
                *dst += u * b;
            }
        }
        q.nearest(&x, &mut out, &mut scratch).unwrap();
        let mut energy = 0.0f64;
        for (&xi, &vi) in x.iter().zip(&out) {
            let d = xi - vi as f64 / scale;
            energy += d * d;
        }
        sum += energy;
        sum_sq += energy * energy;
    }

    let count = samples as f64;
    let mean = sum / count;
    let variance = (sum_sq / count - mean * mean).max(0.0);
    let standard_error = (variance / count).sqrt();

    // G = E[energy] / (rank * V^(2/rank)).
    let denominator = rank as f64 * covolume.powf(2.0 / rank as f64);
    (mean / denominator, standard_error / denominator)
}

#[test]
fn normalized_second_moments_match_the_published_values() {
    // Conway & Sloane table 2.3. The tolerance is five standard errors of the
    // sample itself, with no floor and nothing hand-tuned, so it cannot be
    // widened until the test passes. It is also sharp: the three published
    // values differ from one another by twenty to fifty standard errors, so a
    // decoder that returned the right lattice's points by the wrong rule --
    // giving a Voronoi cell of the wrong shape -- would fail here even though
    // every pointwise test still passed.
    let samples = 120_000;
    let cases: [(Family, usize, f64, f64); 4] = [
        // (family, n, covolume, published G)
        (Family::Zn, 1, 1.0, 1.0 / 12.0),
        (Family::Zn, 8, 1.0, 1.0 / 12.0),
        (Family::Dn, 4, 2.0, 0.076_603),
        (Family::DnPlus, 8, 1.0, 0.071_682),
    ];

    let mut rng = Rng(0x7E57_10E5_7E57_10E5);
    for (family, n, covolume, published) in cases {
        let (estimate, standard_error) =
            normalized_second_moment(family, n, covolume, samples, &mut rng);
        let tolerance = 5.0 * standard_error;
        println!(
            "{family:?}({n}): G = {estimate:.6} +/- {standard_error:.6} (5se = {tolerance:.6}), published {published:.6}"
        );
        assert!(
            (estimate - published).abs() <= tolerance,
            "{family:?}({n}): G = {estimate:.6} +/- {standard_error:.6}, published {published:.6}"
        );
    }
}
