//! Measures decoding beyond the guaranteed packing radius of `BW_16` and
//! `Λ_24`.

use lattica::error::DecodeError;
use lattice_engine::{AmbientScratch, BarnesWall16, Leech24};

const SAMPLES: usize = 2_000;
const NODE_BUDGET: u64 = 1 << 24;
const RADII: [f64; 4] = [0.95, 1.05, 1.25, 1.50];

struct Rng(u64);

impl Rng {
    fn next_signed(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let bits = u32::try_from(self.0 >> 32).unwrap();
        f64::from(bits) / f64::from(u32::MAX) * 2.0 - 1.0
    }
}

fn direction<const N: usize>(rng: &mut Rng, radius: f64) -> [f64; N] {
    let mut point = [0.0; N];
    let mut norm_sq = 0.0;
    for value in &mut point {
        *value = rng.next_signed();
        norm_sq += *value * *value;
    }
    let scale = radius / norm_sq.sqrt();
    for value in &mut point {
        *value *= scale;
    }
    point
}

fn measure<const N: usize>(
    name: &str,
    mut decode: impl FnMut(&[f64], &mut [i64]) -> Result<u64, DecodeError>,
) {
    let mut rng = Rng(0x4c41_5454_4943_4101 ^ u64::try_from(N).unwrap());
    for radius in RADII {
        let mut errors = 0usize;
        let mut exhausted = 0usize;
        let mut out = [0i64; N];
        for _ in 0..SAMPLES {
            let received = direction::<N>(&mut rng, radius);
            match decode(&received, &mut out) {
                Ok(_) => errors += usize::from(out.iter().any(|&value| value != 0)),
                Err(DecodeError::BudgetExhausted { .. }) => exhausted += 1,
                Err(error) => panic!("{name} failed at radius {radius}: {error}"),
            }
        }
        println!(
            "{name} radius={radius:.2} word_errors={errors}/{SAMPLES} budget_exhausted={exhausted}/{SAMPLES}"
        );
    }
}

fn main() {
    let barnes_wall = BarnesWall16::new().unwrap();
    let mut bw_scratch = AmbientScratch::new();
    measure::<16>("BW_16", |received, out| {
        barnes_wall.nearest(received, out, NODE_BUDGET, &mut bw_scratch)
    });

    let leech = Leech24::new().unwrap();
    let mut leech_scratch = AmbientScratch::new();
    measure::<24>("Lambda_24", |received, out| {
        leech.nearest(received, out, NODE_BUDGET, &mut leech_scratch)
    });
}
