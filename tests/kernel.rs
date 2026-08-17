//! Differential tests for the engine-owned batch kernels.
//!
//! The oracle is the per-vector path: `q.nearest` in a loop is the documented
//! scalar decoder, and `nearest_batch` — whether it ran the dispatched SIMD
//! kernel, the scalar reference kernel, or its per-vector fallback — must
//! produce identical bytes for every input, including the tie set, and the
//! identical error with the identical partial-write state for invalid input.

#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use lattice_engine::{Dn, DnPlus, Quantizer, Scratch, Zn, e8, nearest_batch};

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

/// The per-vector reference: exactly what the documented fallback loop does.
fn reference_decode(q: &dyn Quantizer, points: &[f64], out: &mut [i64], scratch: &mut Scratch) {
    let dim = q.dim();
    for (src, dst) in points.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        q.nearest(src, dst, scratch).unwrap();
    }
}

fn check_family(q: &dyn Quantizer, name: &str, rng_seed: u64) {
    let dim = q.dim();
    let mut rng = Rng(rng_seed);
    for vectors in [1usize, 2, 7, 8, 9, 16, 17, 63, 64, 65, 257] {
        // Half the coordinates are exact halves or integers: the tie set,
        // where the away-from-zero and lowest-index rules decide.
        let points: Vec<f64> = (0..dim * vectors)
            .map(|i| {
                let raw = rng.dyadic();
                if i % 2 == 0 {
                    (raw * 2.0).round() / 2.0
                } else {
                    raw
                }
            })
            .collect();

        let mut reference = vec![0i64; points.len()];
        let mut got = vec![0i64; points.len()];
        let (mut a, mut b) = (Scratch::new(dim), Scratch::new(dim));
        reference_decode(q, &points, &mut reference, &mut a);
        nearest_batch(q, &points, &mut got, &mut b).unwrap();
        assert_eq!(reference, got, "{name} diverged at {vectors} vectors");

        // Negation symmetry: same corpus, negated.
        let negated: Vec<f64> = points.iter().map(|v| -v).collect();
        reference_decode(q, &negated, &mut reference, &mut a);
        nearest_batch(q, &negated, &mut got, &mut b).unwrap();
        assert_eq!(reference, got, "{name} diverged under negation");
    }
}

#[test]
fn zn_kernel_matches_the_per_vector_path() {
    for n in [2usize, 3, 5, 8, 24] {
        check_family(
            &Zn::new(n).unwrap(),
            &format!("zn{n}"),
            0x2100_0000 + n as u64,
        );
    }
}

#[test]
fn dn_kernel_matches_the_per_vector_path() {
    for n in [2usize, 3, 5, 8, 24] {
        check_family(
            &Dn::new(n).unwrap(),
            &format!("dn{n}"),
            0x2200_0000 + n as u64,
        );
    }
}

#[test]
fn dnplus_and_e8_kernels_match_the_per_vector_path() {
    for n in [2usize, 4, 6, 8, 16] {
        check_family(
            &DnPlus::new(n).unwrap(),
            &format!("dnplus{n}"),
            0x2300_0000 + n as u64,
        );
    }
    check_family(&e8(), "e8", 0x2300_0008);
}

#[test]
fn an_and_scaled_batches_stay_on_the_scalar_path_unchanged() {
    use lattice_engine::{An, Scaled};
    check_family(&An::new(7).unwrap(), "an7", 0x2400_0007);
    let inner = Dn::new(8).unwrap();
    let scaled = Scaled::new(&inner, 4).unwrap();
    check_family(&scaled, "scaled-dn8", 0x2500_0008);
}

#[test]
fn invalid_batches_reproduce_the_documented_partial_writes() {
    for (q, name) in [
        (&Zn::new(8).unwrap() as &dyn Quantizer, "zn8"),
        (&Dn::new(8).unwrap() as &dyn Quantizer, "dn8"),
        (&e8() as &dyn Quantizer, "e8"),
    ] {
        let dim = q.dim();
        let vectors = 16usize;
        let mut rng = Rng(0xBA0D_BA7C_0000_0001);
        let mut points: Vec<f64> = (0..dim * vectors).map(|_| rng.dyadic()).collect();
        let mut reference = vec![0i64; points.len()];
        let mut got = vec![0i64; points.len()];
        let (mut a, mut b) = (Scratch::new(dim), Scratch::new(dim));

        for bad_index in [0, dim, dim * 5, points.len() - 1] {
            let saved = points[bad_index];
            for poison in [f64::NAN, f64::INFINITY, 30_000_000_000_000_000.0] {
                points[bad_index] = poison;

                // The reference: the documented loop, decoded in order,
                // stopping at the failure with earlier outputs written.
                let mut expected_error = None;
                for (i, (src, dst)) in points
                    .chunks_exact(dim)
                    .zip(reference.chunks_exact_mut(dim))
                    .enumerate()
                {
                    if let Err(error) = q.nearest(src, dst, &mut a) {
                        expected_error = Some(error);
                        let _ = i;
                        break;
                    }
                }
                got.fill(0);
                let got_error = nearest_batch(q, &points, &mut got, &mut b);
                assert_eq!(
                    got_error.err(),
                    expected_error,
                    "{name}: error differs with poison at {bad_index}"
                );
                assert_eq!(
                    got, reference,
                    "{name}: partial-write state differs with poison at {bad_index}"
                );

                points[bad_index] = saved;
            }
        }
    }
}

#[test]
fn ragged_batches_are_rejected_unchanged() {
    let q = Zn::new(2).unwrap();
    let mut scratch = Scratch::new(2);
    let mut out = [0i64; 3];
    // Three points cannot be whole vectors of dimension two.
    assert!(nearest_batch(&q, &[1.0, 2.0, 3.0], &mut out, &mut scratch).is_err());
    // Input and output lengths disagree.
    assert!(nearest_batch(&q, &[1.0, 2.0], &mut out, &mut scratch).is_err());
}

#[cfg(feature = "internals")]
#[test]
fn scalar_round_plane_matches_round_away_elementwise() {
    use lattice_engine::kernel::internals::round_plane_scalar;
    let mut rng = Rng(0x5CA1_0000_0000_0001);
    let x: Vec<f64> = (0..1000)
        .map(|i| {
            let raw = rng.dyadic();
            if i % 3 == 0 { raw.round() } else { raw }
        })
        .collect();
    let mut got = vec![0i64; x.len()];
    round_plane_scalar(&x, &mut got);
    for (dst, &v) in got.iter().zip(&x) {
        let mut one = [7i64; 1];
        lattice_engine::round_nearest(&[v], &mut one).unwrap();
        assert_eq!(*dst, one[0]);
    }
}
