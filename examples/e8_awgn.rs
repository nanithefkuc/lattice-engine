//! Proof: a nested `E_8` lattice code over an AWGN channel.
//!
//! This is the gate that decides whether release 0.1 exists. A decode loop that
//! merely *runs* proves nothing; the test is whether the constellation
//! reproduces a **published constant** — the `0.65 dB` shaping gain of the
//! `E_8` Voronoi region over a cube.
//!
//! ```text
//! coding lattice   Λ_f = E_8
//! shaping lattice  Λ_s = M · E_8
//! codebook         Λ_f / Λ_s, of size M^8
//! encode           x = (λ + d) mod Λ_s,  d uniform over V(Λ_s)
//! channel          y = x + n,  n ~ N(0, σ²I)
//! decode           â = coords(Q_{Λ_f}(y - d)) mod M
//! ```
//!
//! The dither is what makes `x` uniform over `V(Λ_s)` and independent of the
//! message, and the uniformity is where the shaping gain comes from. Remove it
//! and the average power depends on which codeword was sent.
//!
//! Run with `cargo run --release --example e8_awgn`.

// A simulation converts between message indices, lattice coordinates and real
// samples constantly; every cast here is on a small bounded value.
#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use lattica::int::IntMatrix;
use lattica::named::{e8 as e8_gram, e8_generator};
use lattica::nested::Nested;
use lattice_engine::{Quantizer, Scaled, Scratch, e8 as e8_decoder, mod_lattice};

const DIM: usize = 8;
/// Scaling of the shaping lattice. A power of two keeps `Scaled`'s division
/// exact.
const M: i64 = 4;

/// xorshift64 with Box–Muller on top, so the run is reproducible.
struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// Standard normal.
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // Polar Box-Muller, rejecting outside the unit disc.
        loop {
            let u = 2.0 * self.unit() - 1.0;
            let v = 2.0 * self.unit() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }
}

/// Gauss–Jordan inverse of a small dense matrix.
fn invert(a: &[[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let mut work = *a;
    let mut inverse = [[0.0f64; DIM]; DIM];
    for (i, row) in inverse.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..DIM {
        let pivot = (col..DIM)
            .max_by(|&i, &j| work[i][col].abs().total_cmp(&work[j][col].abs()))
            .expect("nonempty");
        work.swap(col, pivot);
        inverse.swap(col, pivot);
        let scale = 1.0 / work[col][col];
        for j in 0..DIM {
            work[col][j] *= scale;
            inverse[col][j] *= scale;
        }
        for row in 0..DIM {
            if row == col {
                continue;
            }
            let factor = work[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..DIM {
                work[row][j] -= factor * work[col][j];
                inverse[row][j] -= factor * inverse[col][j];
            }
        }
    }
    inverse
}

/// `coords · basis`, the ambient point of a coordinate vector.
fn to_ambient(coords: &[i64], basis: &[[f64; DIM]; DIM]) -> [f64; DIM] {
    let mut out = [0.0f64; DIM];
    for (c, row) in coords.iter().zip(basis) {
        if *c == 0 {
            continue;
        }
        let c = *c as f64;
        for (dst, &b) in out.iter_mut().zip(row) {
            *dst += c * b;
        }
    }
    out
}

/// `point · inverse`, rounded: the coordinate vector of an ambient lattice
/// point.
fn to_coords(point: &[f64], inverse: &[[f64; DIM]; DIM]) -> [i64; DIM] {
    let mut out = [0i64; DIM];
    for (j, slot) in out.iter_mut().enumerate() {
        let value: f64 = (0..DIM).map(|k| point[k] * inverse[k][j]).sum();
        *slot = value.round() as i64;
    }
    out
}

fn main() {
    let basis = e8_generator();
    let inverse = invert(&basis);
    let shaping_basis = {
        let mut b = basis;
        for row in &mut b {
            for v in row {
                *v *= M as f64;
            }
        }
        b
    };

    // Codebook: Λ_f / Λ_s with Λ_s = M·Λ_f, so the transform is M·I.
    let mut transform = IntMatrix::<i64>::zeros(DIM, DIM).unwrap();
    for i in 0..DIM {
        transform.set(i, i, M);
    }
    let pair = Nested::new(e8_gram::<i64>().unwrap(), transform).unwrap();
    let codebook = u64::try_from(pair.index()).unwrap();

    let coding = e8_decoder();
    let shaping = Scaled::new(coding, M).unwrap();
    let mut scratch = Scratch::new(DIM);

    println!("Nested E_8 lattice code over AWGN");
    println!("  coding lattice   E_8");
    println!("  shaping lattice  {M} * E_8");
    println!("  codebook size    {codebook}  ({} bits/dimension)", {
        let bits = (codebook as f64).log2() / DIM as f64;
        format!("{bits:.2}")
    });
    println!();

    // ---------------------------------------------------------------- shaping
    let samples = 400_000usize;
    let mut rng = Rng::new(0x5EED_E805_EEDE_8001);
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    let mut coords = vec![0i64; DIM];
    let mut transmitted = [0.0f64; DIM];

    for _ in 0..samples {
        pair.coset_representative(rng.below(codebook), &mut coords)
            .unwrap();
        let word = to_ambient(&coords, &basis);

        // Dither uniform over V(Λ_s): sample the fundamental parallelepiped,
        // then fold it into the Voronoi region.
        let mut raw = [0.0f64; DIM];
        for row in &shaping_basis {
            let u = rng.unit();
            for (dst, &b) in raw.iter_mut().zip(row) {
                *dst += u * b;
            }
        }
        let mut dither = [0.0f64; DIM];
        mod_lattice(&shaping, &raw, &mut dither, &mut scratch).unwrap();

        let mut shifted = [0.0f64; DIM];
        for i in 0..DIM {
            shifted[i] = word[i] + dither[i];
        }
        mod_lattice(&shaping, &shifted, &mut transmitted, &mut scratch).unwrap();

        let energy: f64 = transmitted.iter().map(|v| v * v).sum();
        sum += energy;
        sum_sq += energy * energy;
    }

    let count = samples as f64;
    let mean_energy = sum / count;
    let power = mean_energy / DIM as f64;
    let variance = (sum_sq / count - mean_energy * mean_energy).max(0.0);
    let standard_error = (variance / count).sqrt() / DIM as f64;

    let cube_power = (M * M) as f64 / 12.0;
    let gain_db = 10.0 * (cube_power / power).log10();
    // d(gain)/d(power), for propagating the Monte-Carlo error into dB.
    let gain_error_db = 10.0 / core::f64::consts::LN_10 * standard_error / power;
    let ultimate_db = 10.0 * (2.0 * core::f64::consts::PI * core::f64::consts::E / 12.0).log10();

    println!("Shaping ({samples} samples)");
    println!("  average power / dimension   {power:.6}  +/- {standard_error:.6}");
    println!("  cube of the same volume     {cube_power:.6}");
    println!("  shaping gain                {gain_db:.4} dB  +/- {gain_error_db:.4} dB");
    println!("  published for E_8           0.6539 dB");
    println!("  ultimate (sphere)           {ultimate_db:.4} dB");
    println!();

    // ---------------------------------------------------------------- channel
    println!("Decoding");
    println!(
        "  {:>8}  {:>10}  {:>12}",
        "sigma", "SNR (dB)", "word errors"
    );
    let trials = 20_000usize;
    for &sigma in &[0.40f64, 0.30, 0.25, 0.20, 0.15] {
        let mut errors = 0usize;
        let mut received = [0.0f64; DIM];
        let mut decoded = vec![0i64; DIM];

        for _ in 0..trials {
            let message = rng.below(codebook);
            pair.coset_representative(message, &mut coords).unwrap();
            let word = to_ambient(&coords, &basis);

            let mut raw = [0.0f64; DIM];
            for row in &shaping_basis {
                let u = rng.unit();
                for (dst, &b) in raw.iter_mut().zip(row) {
                    *dst += u * b;
                }
            }
            let mut dither = [0.0f64; DIM];
            mod_lattice(&shaping, &raw, &mut dither, &mut scratch).unwrap();

            let mut shifted = [0.0f64; DIM];
            for i in 0..DIM {
                shifted[i] = word[i] + dither[i];
            }
            mod_lattice(&shaping, &shifted, &mut transmitted, &mut scratch).unwrap();

            for i in 0..DIM {
                received[i] = transmitted[i] + sigma * rng.normal() - dither[i];
            }

            coding
                .nearest(&received, &mut decoded, &mut scratch)
                .unwrap();
            // The E_8 decoder returns doubled coordinates.
            let ambient: Vec<f64> = decoded.iter().map(|&v| v as f64 / 2.0).collect();
            let recovered = to_coords(&ambient, &inverse);

            let matches = coords
                .iter()
                .zip(&recovered)
                .all(|(&want, &got)| got.rem_euclid(M) == want);
            if !matches {
                errors += 1;
            }
        }

        let snr_db = 10.0 * (power / (sigma * sigma)).log10();
        println!("  {sigma:>8.2}  {snr_db:>10.2}  {errors:>7} / {trials}");
    }
    println!();

    // ------------------------------------------------------------------- gate
    assert!(
        (gain_db - 0.6539).abs() < 5.0 * gain_error_db.max(0.002),
        "shaping gain {gain_db:.4} dB is not the published 0.6539 dB"
    );
    println!("PASS: the E_8 Voronoi region reproduces its published shaping gain.");
}
