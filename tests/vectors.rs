//! Pinned tie behaviour, driven by `tests/data/ties.txt`.
#![allow(clippy::doc_markdown)]
//!
//! Invariant I3: two peers quantizing the same boundary point must return the
//! same lattice point. The fixture file is the specification of which point,
//! and a change to an expected value there is a wire-format break.
//!
//! This test also runs under cross-*execution* in CI — on x86_64, AArch64 and
//! Wasm — because a build-only job proves nothing about floating-point
//! agreement between peers.

use lattice_engine::{
    An, Dn, DnPlus, Quantizer, Scratch, Zn, round_nearest, round_nearest_flipped,
};

fn parse_floats(field: &str) -> Vec<f64> {
    field
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect()
}

fn parse_ints(field: &str) -> Vec<i64> {
    field
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect()
}

#[test]
fn pinned_tie_fixtures() {
    let text = include_str!("data/ties.txt");
    let mut scratch = Scratch::new(16);
    let mut cases = 0;

    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('|').map(str::trim).collect();
        assert_eq!(fields.len(), 3, "line {}: malformed", lineno + 1);

        let head: Vec<&str> = fields[0].split_whitespace().collect();
        let (op, param) = (head[0], head[1]);
        let x = parse_floats(fields[1]);
        let want = parse_ints(fields[2]);
        let mut got = vec![0i64; want.len()];

        match op {
            "f" => round_nearest(&x, &mut got).unwrap(),
            "g" => round_nearest_flipped(&x, &mut got).unwrap(),
            "zn" => Zn::new(param.parse().unwrap())
                .unwrap()
                .nearest(&x, &mut got, &mut scratch)
                .unwrap(),
            "dn" => Dn::new(param.parse().unwrap())
                .unwrap()
                .nearest(&x, &mut got, &mut scratch)
                .unwrap(),
            "an" => An::new(param.parse().unwrap())
                .unwrap()
                .nearest(&x, &mut got, &mut scratch)
                .unwrap(),
            "dnplus" => DnPlus::new(param.parse().unwrap())
                .unwrap()
                .nearest(&x, &mut got, &mut scratch)
                .unwrap(),
            other => panic!("line {}: unknown op {other}", lineno + 1),
        }

        assert_eq!(
            got,
            want,
            "line {}: {op} {param} on {x:?} -- this is a format break, not a test failure",
            lineno + 1
        );
        cases += 1;
    }

    assert!(
        cases >= 17,
        "only {cases} fixtures ran; the file may be truncated"
    );
}

#[test]
fn rounding_commutes_with_negation() {
    // The reason ties go away from zero rather than to even: f(-x) == -f(x),
    // which is checkable rather than merely intended.
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Multiples of 1/64 in [-8, 8), so half-way cases occur often.
        (f64::from(u32::try_from(state % 1024).unwrap()) - 512.0) / 64.0
    };
    for _ in 0..2000 {
        let x: Vec<f64> = (0..7).map(|_| next()).collect();
        let negated: Vec<f64> = x.iter().map(|v| -v).collect();

        let mut a = vec![0i64; 7];
        let mut b = vec![0i64; 7];
        round_nearest(&x, &mut a).unwrap();
        round_nearest(&negated, &mut b).unwrap();
        for i in 0..7 {
            assert_eq!(a[i], -b[i], "f(-x) != -f(x) at {:?}", x[i]);
        }

        round_nearest_flipped(&x, &mut a).unwrap();
        round_nearest_flipped(&negated, &mut b).unwrap();
        // `g` inherits the symmetry except where the worst-coordinate tie is
        // resolved by index, which negation cannot flip. It still holds
        // whenever the flipped coordinate has a nonzero residual.
        let differing = (0..7).filter(|&i| a[i] != -b[i]).count();
        assert!(
            differing <= 1,
            "g broke symmetry in {differing} coordinates"
        );
    }
}
