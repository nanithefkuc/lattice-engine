//! Independent acceptance checks for budgeted real-target enumeration.
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

use lattica::Basis;
use lattica::error::DecodeError;
use lattica::named::{d_n, d_n_basis, e8, e8_generator, zn};
use lattica::reduce::Delta;
use lattica::relevant::relevant_vectors;
use lattica::shortvec::DEFAULT_NODE_BUDGET;
use lattice_engine::babai::distance_sq;
use lattice_engine::closed::{Dn, e8 as e8_quantizer};
use lattice_engine::{
    EnumerationScratch, Enumerator, PreparedEnumerationScratch, PreparedEnumerator, Quantizer,
    Scratch,
};

const DECODE_BUDGET: u64 = 1 << 20;

fn for_each_box(
    dimension: usize,
    low: i64,
    high: i64,
    point: &mut [i64],
    depth: usize,
    visit: &mut impl FnMut(&[i64]),
) {
    if depth == dimension {
        visit(point);
        return;
    }
    for value in low..=high {
        point[depth] = value;
        for_each_box(dimension, low, high, point, depth + 1, visit);
    }
}

#[test]
fn nearest_agrees_with_brute_force_through_dimension_six() {
    for n in 1..=6 {
        let mut basis_data = vec![0i64; n * n];
        for i in 0..n {
            basis_data[i * n + i] = 1;
            if i > 0 {
                basis_data[i * n + i - 1] = if i % 2 == 0 { -1 } else { 1 };
            }
        }
        let gram = Basis::from_rows(n, n, &basis_data).unwrap().gram().unwrap();
        let enumerator = Enumerator::new(&gram).unwrap();
        let target: Vec<f64> = (0..n)
            .map(|i| (f64::from(i as u32) - f64::from(n as u32) / 2.0) / 3.0 + 0.125)
            .collect();

        let mut brute = vec![0i64; n];
        let mut candidate = vec![0i64; n];
        let mut brute_distance = f64::INFINITY;
        for_each_box(n, -3, 3, &mut candidate, 0, &mut |point| {
            let distance = distance_sq(&gram, &target, point).unwrap();
            if distance < brute_distance || (distance == brute_distance && point < &brute) {
                brute_distance = distance;
                brute.copy_from_slice(point);
            }
        });

        let mut got = vec![i64::MAX; n];
        let mut scratch = EnumerationScratch::new();
        enumerator
            .nearest(&target, &mut got, 100.0, DECODE_BUDGET, &mut scratch)
            .unwrap();
        assert_eq!(got, brute, "dimension {n}");
        assert_eq!(distance_sq(&gram, &target, &got).unwrap(), brute_distance);
    }
}

#[cfg(feature = "internals")]
#[test]
fn seeded_radius_is_validated_before_proof_search() {
    let gram = zn::<i64>(3).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let target = [0.2, -0.3, 0.1];
    let candidate = [0, 0, 0];
    let mut scratch = EnumerationScratch::new();
    let mut out = [91; 3];

    assert_eq!(
        enumerator
            .nearest_seeded(
                &target,
                &mut out,
                &candidate,
                0.0,
                DECODE_BUDGET,
                &mut scratch,
            )
            .unwrap_err(),
        DecodeError::InvalidRadius { radius_sq: 0.0 }
    );
    assert_eq!(out, [91; 3]);

    enumerator
        .nearest_seeded(
            &target,
            &mut out,
            &candidate,
            1.0,
            DECODE_BUDGET,
            &mut scratch,
        )
        .unwrap();
    assert_eq!(out, candidate);
}

#[test]
fn prepared_reduced_basis_maps_the_proved_answer_back() {
    let basis = Basis::from_rows(
        4,
        4,
        &[
            7i64, 2, 0, 0, //
            5, 3, 1, 0, //
            4, -2, 5, 1, //
            9, 1, -3, 4,
        ],
    )
    .unwrap();
    let gram = basis.gram().unwrap();
    let direct = Enumerator::new(&gram).unwrap();
    let prepared = PreparedEnumerator::new(&gram, Delta::STRONG).unwrap();
    let target = [1.25, -0.375, 2.125, -1.75];

    let mut direct_out = [0; 4];
    direct
        .nearest_ml(
            &target,
            &mut direct_out,
            DECODE_BUDGET,
            &mut EnumerationScratch::new(),
        )
        .unwrap();
    let mut prepared_out = [0; 4];
    prepared
        .nearest_ml(
            &target,
            &mut prepared_out,
            DECODE_BUDGET,
            &mut PreparedEnumerationScratch::new(),
        )
        .unwrap();

    assert_eq!(prepared_out, direct_out);
    assert_eq!(
        distance_sq(&gram, &target, &prepared_out).unwrap(),
        distance_sq(&gram, &target, &direct_out).unwrap()
    );
}

#[test]
fn list_mode_matches_brute_force_and_pins_order() {
    let gram = zn::<i64>(2).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let target = [0.25, -0.375];
    let radius_sq = 2.0;
    let mut scratch = EnumerationScratch::new();
    let got = enumerator
        .list(&target, radius_sq, DECODE_BUDGET, &mut scratch)
        .unwrap();

    let mut want = Vec::new();
    let mut point = [0i64; 2];
    for_each_box(2, -2, 2, &mut point, 0, &mut |candidate| {
        let distance = distance_sq(&gram, &target, candidate).unwrap();
        if distance <= radius_sq {
            want.push((distance, candidate.to_vec()));
        }
    });
    want.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    assert_eq!(got.len(), want.len());
    for (actual, (distance, point)) in got.iter().zip(want) {
        assert_eq!(actual.point(), point);
        assert_eq!(actual.distance_sq(), distance);
    }
}

#[test]
fn budget_exhaustion_is_deterministic_and_never_writes_a_candidate() {
    let gram = zn::<i64>(6).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let mut out = [77i64; 6];
    let mut scratch = EnumerationScratch::new();
    let error = enumerator
        .nearest(&[0.1; 6], &mut out, 100.0, 1, &mut scratch)
        .unwrap_err();
    assert_eq!(
        error,
        DecodeError::BudgetExhausted {
            nodes: 1,
            radius_sq: 100.0,
        }
    );
    assert_eq!(out, [77; 6]);
}

#[test]
fn enumeration_agrees_with_closed_dn_decoder() {
    for n in 3..=6 {
        let gram = d_n::<i64>(n).unwrap();
        let basis = d_n_basis::<i64>(n).unwrap();
        let enumerator = Enumerator::new(&gram).unwrap();
        let quantizer = Dn::new(n).unwrap();
        let target: Vec<f64> = (0..n)
            .map(|i| f64::from(i as u32) * 0.271 - 0.613)
            .collect();
        let mut ambient = vec![0.0; n];
        for (i, &coefficient) in target.iter().enumerate() {
            for (j, slot) in ambient.iter_mut().enumerate() {
                *slot += coefficient * basis.as_matrix().get(i, j) as f64;
            }
        }

        let mut closed = vec![0i64; n];
        quantizer
            .nearest(&ambient, &mut closed, &mut Scratch::new(n))
            .unwrap();
        let mut coordinates = vec![0i64; n];
        enumerator
            .nearest(
                &target,
                &mut coordinates,
                100.0,
                DECODE_BUDGET,
                &mut EnumerationScratch::new(),
            )
            .unwrap();
        let mut enumerated_ambient = vec![0i64; n];
        for (i, &coefficient) in coordinates.iter().enumerate() {
            for (j, slot) in enumerated_ambient.iter_mut().enumerate() {
                *slot += coefficient * basis.as_matrix().get(i, j);
            }
        }
        assert_eq!(enumerated_ambient, closed, "D_{n}");
    }
}

#[test]
fn enumeration_agrees_with_closed_e8_decoder() {
    let generator = e8_generator();
    let mut gram_data = [0i64; 64];
    for i in 0..8 {
        for j in 0..8 {
            let inner: f64 = (0..8).map(|k| generator[i][k] * generator[j][k]).sum();
            gram_data[i * 8 + j] = inner.round() as i64;
        }
    }
    let gram = lattica::Gram::from_rows(8, &gram_data).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let target = [-0.61, -0.39, -0.17, 0.05, 0.27, 0.49, 0.71, 0.93];
    let mut ambient = [0.0; 8];
    for i in 0..8 {
        for (j, slot) in ambient.iter_mut().enumerate() {
            *slot += target[i] * generator[i][j];
        }
    }

    let mut closed = [0i64; 8];
    e8_quantizer()
        .nearest(&ambient, &mut closed, &mut Scratch::new(8))
        .unwrap();
    let mut coordinates = [0i64; 8];
    enumerator
        .nearest(
            &target,
            &mut coordinates,
            100.0,
            DECODE_BUDGET,
            &mut EnumerationScratch::new(),
        )
        .unwrap();
    let mut doubled_ambient = [0i64; 8];
    for i in 0..8 {
        for (j, slot) in doubled_ambient.iter_mut().enumerate() {
            *slot += coordinates[i] * (2.0 * generator[i][j]) as i64;
        }
    }
    assert_eq!(doubled_ambient, closed);
}

#[test]
fn relevant_vector_counts_match_d4_and_e8() {
    let d4 = relevant_vectors(&d_n::<i64>(4).unwrap(), DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(d4.len(), 24);
    let e8_vectors = relevant_vectors(&e8::<i64>().unwrap(), DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(e8_vectors.len(), 240);
}
