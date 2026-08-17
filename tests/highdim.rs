//! Acceptance checks for the Barnes–Wall and Leech lattice decoders.
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use lattica::named::{bw16, leech24};
use lattica::shortvec::{DEFAULT_NODE_BUDGET, census};
use lattice_engine::{AmbientScratch, BarnesWall16, Leech24};

const DECODE_BUDGET: u64 = 1 << 24;

#[test]
fn named_gram_matrices_reproduce_published_constants() {
    let barnes_wall = bw16::<i64>().unwrap();
    assert_eq!(barnes_wall.det().unwrap(), 256);
    let bw_census = census(&barnes_wall, DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(bw_census.min_norm_sq, Some(4));
    assert_eq!(bw_census.kissing_number, 4320);

    let leech = leech24::<i64>().unwrap();
    assert_eq!(leech.det().unwrap(), 1);
    let leech_census = census(&leech, DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(leech_census.min_norm_sq, Some(4));
    assert_eq!(leech_census.kissing_number, 196_560);
}

#[test]
fn coefficient_decoders_are_exact_inside_the_packing_radius() {
    let barnes_wall = BarnesWall16::new().unwrap();
    let mut bw_target = [0.0; 16];
    let mut bw_want = [0i64; 16];
    for i in 0..16 {
        bw_want[i] = i64::from((i % 3) as i8) - 1;
        bw_target[i] = bw_want[i] as f64 + (i as f64 - 7.5) / 512.0;
    }
    let mut bw_got = [0i64; 16];
    barnes_wall
        .nearest_coefficients(
            &bw_target,
            &mut bw_got,
            DECODE_BUDGET,
            &mut AmbientScratch::new(),
        )
        .unwrap();
    assert_eq!(bw_got, bw_want);

    let leech = Leech24::new().unwrap();
    let mut leech_target = [0.0; 24];
    let mut leech_want = [0i64; 24];
    for i in 0..24 {
        leech_want[i] = i64::from((i % 3) as i8) - 1;
        leech_target[i] = leech_want[i] as f64 + (i as f64 - 11.5) / 1024.0;
    }
    let mut leech_got = [0i64; 24];
    leech
        .nearest_coefficients(
            &leech_target,
            &mut leech_got,
            DECODE_BUDGET,
            &mut AmbientScratch::new(),
        )
        .unwrap();
    assert_eq!(leech_got, leech_want);
}

#[test]
fn ambient_decoders_preserve_exact_algebraic_scaling() {
    let barnes_wall = BarnesWall16::new().unwrap();
    assert_eq!(barnes_wall.coordinate_denominator_sq(), 4);
    let mut bw_out = [7i64; 16];
    barnes_wall
        .nearest(
            &[0.01; 16],
            &mut bw_out,
            DECODE_BUDGET,
            &mut AmbientScratch::new(),
        )
        .unwrap();
    assert_eq!(bw_out, [0; 16]);

    let leech = Leech24::new().unwrap();
    assert_eq!(leech.coordinate_denominator_sq(), 8);
    let mut leech_out = [7i64; 24];
    leech
        .nearest(
            &[0.01; 24],
            &mut leech_out,
            DECODE_BUDGET,
            &mut AmbientScratch::new(),
        )
        .unwrap();
    assert_eq!(leech_out, [0; 24]);
}
