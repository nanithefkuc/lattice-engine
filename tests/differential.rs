//! Migration differential: `lattice_engine` versus the `lattica::quant` it was
//! extracted from.
//!
//! While `lattica` still publishes its decoder module, this suite runs both
//! implementations on identical inputs and requires bit-identical outputs and
//! identical error variants. It exists to prove the extraction is a pure move;
//! it is deleted when the engine re-pins a `lattica` revision that no longer
//! carries `quant`, leaving the moved tests and `tests/data/ties.txt` as the
//! frozen expectations.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use std::num::NonZeroU32;

use lattica::construct as old_construct;
use lattica::construct::CodeMembership as OldMembership;
use lattica::named::{d_n, e8 as e8_gram, zn};
use lattica::quant as old;
use lattica::reduce::Delta;
use lattice_engine as engine;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn dyadic(&mut self) -> f64 {
        (f64::from(u32::try_from(self.next() % 4096).unwrap()) - 2048.0) / 256.0
    }
}

/// The single parity check code over `Z_2`, length 4, exhaustive decoder.
struct ParityCheck;

impl engine::CodeMembership for ParityCheck {
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
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), engine::DecodeError> {
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

impl OldMembership for ParityCheck {
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
    fn decode_costs(
        &self,
        costs: &[f64],
        out: &mut [u32],
    ) -> Result<(), lattica::error::DecodeError> {
        engine::CodeMembership::decode_costs(self, costs, out)
    }
}

/// One closed-form family in both implementations: old and new side by side.
type Pair = (
    &'static str,
    usize,
    Box<dyn old::Quantizer>,
    Box<dyn engine::Quantizer>,
);

#[test]
fn closed_form_families_agree_bit_for_bit() {
    let mut rng = Rng(0xD1FF_E0E0_0D1D_1F1E);
    let cases = 2_000usize;

    let pairs: Vec<Pair> = vec![
        (
            "zn8",
            8,
            Box::new(old::Zn::new(8).unwrap()),
            Box::new(engine::Zn::new(8).unwrap()),
        ),
        (
            "dn8",
            8,
            Box::new(old::Dn::new(8).unwrap()),
            Box::new(engine::Dn::new(8).unwrap()),
        ),
        (
            "an7",
            8,
            Box::new(old::An::new(7).unwrap()),
            Box::new(engine::An::new(7).unwrap()),
        ),
        (
            "dnplus8",
            8,
            Box::new(old::DnPlus::new(8).unwrap()),
            Box::new(engine::DnPlus::new(8).unwrap()),
        ),
        ("e8", 8, Box::new(old::e8()), Box::new(engine::e8())),
        (
            "dn5",
            5,
            Box::new(old::Dn::new(5).unwrap()),
            Box::new(engine::Dn::new(5).unwrap()),
        ),
    ];

    for (name, dim, old_q, new_q) in pairs {
        let mut old_scratch = old::Scratch::new(dim);
        let mut new_scratch = engine::Scratch::new(dim);
        let (mut a, mut b) = (vec![0i64; dim], vec![0i64; dim]);
        for _ in 0..cases {
            // Half of the inputs are exact halves or integers: the tie set.
            let x: Vec<f64> = (0..dim)
                .map(|i| {
                    let raw = rng.dyadic();
                    if i % 2 == 0 {
                        (raw * 2.0).round() / 2.0
                    } else {
                        raw
                    }
                })
                .collect();
            old_q.nearest(&x, &mut a, &mut old_scratch).unwrap();
            new_q.nearest(&x, &mut b, &mut new_scratch).unwrap();
            assert_eq!(a, b, "{name} disagreed on {x:?}");
        }

        // Rejections agree exactly.
        let bad: Vec<Vec<f64>> = vec![
            vec![f64::NAN; dim],
            vec![f64::INFINITY; dim],
            vec![0.0; dim - 1],
        ];
        for x in bad {
            let mut a = vec![7i64; dim];
            let mut b = vec![7i64; dim];
            assert_eq!(
                old_q.nearest(&x, &mut a, &mut old_scratch),
                new_q.nearest(&x, &mut b, &mut new_scratch),
                "{name} rejection differed for {x:?}"
            );
            assert_eq!(a, b);
        }
    }
}

#[test]
fn enumeration_agrees_bit_for_bit() {
    const BUDGET: u64 = 1 << 20;
    let mut rng = Rng(0xE0E0_1D1D_2E2E_3F3F);

    for gram in [
        zn::<i64>(12).unwrap(),
        d_n::<i64>(8).unwrap(),
        e8_gram::<i64>().unwrap(),
    ] {
        let dim = gram.dim();
        let old_enum = old::Enumerator::new(&gram).unwrap();
        let new_enum = engine::Enumerator::new(&gram).unwrap();
        let (mut old_scratch, mut new_scratch) = (
            old::EnumerationScratch::new(),
            engine::EnumerationScratch::new(),
        );
        let (mut a, mut b) = (vec![0i64; dim], vec![0i64; dim]);

        for _ in 0..200 {
            let x: Vec<f64> = (0..dim).map(|_| rng.dyadic()).collect();
            let old_nodes = old_enum
                .nearest(&x, &mut a, 64.0, BUDGET, &mut old_scratch)
                .unwrap();
            let new_nodes = new_enum
                .nearest(&x, &mut b, 64.0, BUDGET, &mut new_scratch)
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(old_nodes, new_nodes, "node counts diverged");
        }

        let old_prepared = old::PreparedEnumerator::new(&gram, Delta::STRONG).unwrap();
        let new_prepared = engine::PreparedEnumerator::new(&gram, Delta::STRONG).unwrap();
        let (mut old_scratch, mut new_scratch) = (
            old::PreparedEnumerationScratch::new(),
            engine::PreparedEnumerationScratch::new(),
        );
        for _ in 0..200 {
            let x: Vec<f64> = (0..dim).map(|_| rng.dyadic()).collect();
            let old_nodes = old_prepared
                .nearest_ml(&x, &mut a, BUDGET, &mut old_scratch)
                .unwrap();
            let new_nodes = new_prepared
                .nearest_ml(&x, &mut b, BUDGET, &mut new_scratch)
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(old_nodes, new_nodes);
        }

        // Budget exhaustion agrees, including the reported radius.
        let x: Vec<f64> = (0..dim).map(|_| rng.dyadic()).collect();
        let (mut o, mut n) = (
            old::EnumerationScratch::new(),
            engine::EnumerationScratch::new(),
        );
        assert_eq!(
            old_enum.nearest(&x, &mut a, 64.0, 0, &mut o),
            new_enum.nearest(&x, &mut b, 64.0, 0, &mut n),
        );
    }
}

#[test]
fn maximum_likelihood_decoders_agree_bit_for_bit() {
    let targets = [
        vec![0.31; 16],
        vec![0.5; 16],
        vec![-0.5; 16],
        vec![1.25; 16],
    ];
    let (mut old_scratch, mut new_scratch) =
        (old::AmbientScratch::new(), engine::AmbientScratch::new());
    let (mut a, mut b) = (vec![0i64; 16], vec![0i64; 16]);

    let old_bw = old::BarnesWall16::new().unwrap();
    let new_bw = engine::BarnesWall16::new().unwrap();
    for x in &targets {
        assert_eq!(
            old_bw.nearest(x, &mut a, 1 << 20, &mut old_scratch),
            new_bw.nearest(x, &mut b, 1 << 20, &mut new_scratch),
        );
        assert_eq!(a, b);
    }
    // Exhaustion is reported identically.
    assert_eq!(
        old_bw.nearest(&targets[0], &mut a, 0, &mut old_scratch),
        new_bw.nearest(&targets[0], &mut b, 0, &mut new_scratch),
    );
}

#[test]
fn modulo_and_scaled_agree_bit_for_bit() {
    let mut rng = Rng(0xBADA_5555_0D1D_1F1E);
    let q_old = old::Dn::new(6).unwrap();
    let q_new = engine::Dn::new(6).unwrap();
    let scaled_old = old::Scaled::new(&q_old, 4).unwrap();
    let scaled_new = engine::Scaled::new(&q_new, 4).unwrap();
    let (mut old_scratch, mut new_scratch) = (old::Scratch::new(6), engine::Scratch::new(6));
    let (mut a, mut b, mut c, mut d) = (
        vec![0.0f64; 6],
        vec![0.0f64; 6],
        vec![0i64; 6],
        vec![0i64; 6],
    );

    for _ in 0..500 {
        let x: Vec<f64> = (0..6).map(|_| rng.dyadic()).collect();
        let dither: Vec<f64> = (0..6).map(|_| rng.dyadic() / 8.0).collect();

        old::mod_lattice(&q_old, &x, &mut a, &mut old_scratch).unwrap();
        engine::mod_lattice(&q_new, &x, &mut b, &mut new_scratch).unwrap();
        assert_eq!(a, b);

        old::mod_lattice_dithered(&q_old, &x, &dither, &mut a, &mut old_scratch).unwrap();
        engine::mod_lattice_dithered(&q_new, &x, &dither, &mut b, &mut new_scratch).unwrap();
        assert_eq!(a, b);

        old::Quantizer::nearest(&scaled_old, &x, &mut c, &mut old_scratch).unwrap();
        engine::Quantizer::nearest(&scaled_new, &x, &mut d, &mut new_scratch).unwrap();
        assert_eq!(c, d);
    }
}

#[test]
fn construction_a_agrees_bit_for_bit() {
    let mut rng = Rng(0xC0DE_0D1D_1F1E_5A5A);
    let old_lattice = old_construct::ConstructionA::new(ParityCheck).unwrap();
    let new_lattice = engine::ConstructionA::new(ParityCheck).unwrap();
    let (mut old_scratch, mut new_scratch) = (old::Scratch::new(4), engine::Scratch::new(4));
    let (mut a, mut b) = ([0i64; 4], [0i64; 4]);

    for _ in 0..2_000 {
        let x: Vec<f64> = (0..4).map(|_| rng.dyadic()).collect();
        old::Quantizer::nearest(&old_lattice, &x, &mut a, &mut old_scratch).unwrap();
        engine::Quantizer::nearest(&new_lattice, &x, &mut b, &mut new_scratch).unwrap();
        assert_eq!(a, b);
    }
}
