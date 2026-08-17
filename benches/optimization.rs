//! Workload-separated decode measurements and correctness fingerprints.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattica::Basis;
use lattice_engine::kernel::internals::round_plane_scalar;
use lattice_engine::{
    AmbientScratch, An, BarnesWall16, Dn, DnPlus, EnumerationScratch, Enumerator, Leech24,
    Quantizer, Scratch, Zn, e8 as e8_quantizer, nearest_batch,
};

const DIMENSIONS: [usize; 3] = [8, 16, 24];
const SAMPLES: usize = 11;
const NODE_BUDGET: u64 = 1 << 28;

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measured<R>(mut body: impl FnMut() -> R) -> Duration {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        black_box(body());
        samples.push(start.elapsed());
    }
    median(samples)
}

fn cvp_gram(dimension: usize) -> lattica::Gram<i128> {
    let mut rows = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        rows[row * dimension + row] = 2;
        if row + 1 < dimension {
            rows[row * dimension + row + 1] = 1;
        }
    }
    Basis::from_rows(dimension, dimension, &rows)
        .unwrap()
        .gram()
        .unwrap()
}

fn benchmark_cvp() {
    for dimension in DIMENSIONS {
        let gram = cvp_gram(dimension);
        let prepare = measured(|| black_box(Enumerator::new(black_box(&gram)).unwrap()));
        println!(
            "cvp_prepare_ns,{dimension},single,{:.2},{}",
            prepare.as_secs_f64() * 1e9,
            gram.det().unwrap()
        );
        let enumerator = Enumerator::new(&gram).unwrap();
        let mut scratch = EnumerationScratch::new();
        let mut out = vec![0i64; dimension];
        for (class, offset) in [("easy", 0.0625), ("median", 0.4375), ("boundary", 0.5)] {
            let target: Vec<f64> = (0..dimension)
                .map(|index| if index % 2 == 0 { offset } else { -offset })
                .collect();
            let expected = enumerator
                .nearest_ml(&target, &mut out, NODE_BUDGET, &mut scratch)
                .unwrap();
            let fingerprint: i128 = out
                .iter()
                .enumerate()
                .map(|(index, value)| i128::try_from(index + 1).unwrap() * i128::from(*value))
                .sum();
            let elapsed = measured(|| {
                black_box(
                    enumerator
                        .nearest_ml(
                            black_box(&target),
                            black_box(&mut out),
                            NODE_BUDGET,
                            black_box(&mut scratch),
                        )
                        .unwrap(),
                );
            });
            let ns = elapsed.as_secs_f64() * 1e9;
            println!("cvp_nodes,{dimension},{class},{expected},{fingerprint}");
            println!("cvp_warm_ns,{dimension},{class},{ns:.2},{fingerprint}");
            println!(
                "cvp_ns_per_node,{dimension},{class},{:.2},{fingerprint}",
                ns / f64::from(u32::try_from(expected.max(1)).unwrap())
            );
            println!("cvp_max_depth,{dimension},{class},{dimension},{fingerprint}");
        }
    }
}

fn benchmark_named() {
    let bw_setup = measured(|| black_box(BarnesWall16::new().unwrap()));
    let leech_setup = measured(|| black_box(Leech24::new().unwrap()));
    println!(
        "named_setup_ns,16,bw16,{:.2},16",
        bw_setup.as_secs_f64() * 1e9
    );
    println!(
        "named_setup_ns,24,leech,{:.2},24",
        leech_setup.as_secs_f64() * 1e9
    );

    let bw = BarnesWall16::new().unwrap();
    let leech = Leech24::new().unwrap();
    let mut scratch = AmbientScratch::new();
    let mut out16 = [0i64; 16];
    let mut out24 = [0i64; 24];
    let ambient16 = [0.31; 16];
    let ambient24 = [0.31; 24];
    let nodes16 = bw
        .nearest(&ambient16, &mut out16, NODE_BUDGET, &mut scratch)
        .unwrap();
    let nodes24 = leech
        .nearest(&ambient24, &mut out24, NODE_BUDGET, &mut scratch)
        .unwrap();
    let total16 = measured(|| {
        black_box(
            bw.nearest(&ambient16, &mut out16, NODE_BUDGET, &mut scratch)
                .unwrap(),
        );
    });
    let total24 = measured(|| {
        black_box(
            leech
                .nearest(&ambient24, &mut out24, NODE_BUDGET, &mut scratch)
                .unwrap(),
        );
    });
    println!(
        "named_total_ns,16,bw16,{:.2},{}",
        total16.as_secs_f64() * 1e9,
        out16.iter().sum::<i64>()
    );
    println!(
        "named_total_ns,24,leech,{:.2},{}",
        total24.as_secs_f64() * 1e9,
        out24.iter().sum::<i64>()
    );
    println!(
        "named_nodes,16,bw16,{nodes16},{}",
        out16.iter().sum::<i64>()
    );
    println!(
        "named_nodes,24,leech,{nodes24},{}",
        out24.iter().sum::<i64>()
    );
}

fn benchmark_quantizers() {
    let quantizers: Vec<(&str, Box<dyn Quantizer>)> = vec![
        ("zn24", Box::new(Zn::new(24).unwrap())),
        ("dn24", Box::new(Dn::new(24).unwrap())),
        ("an23", Box::new(An::new(23).unwrap())),
        ("dnplus24", Box::new(DnPlus::new(24).unwrap())),
        ("e8", Box::new(e8_quantizer())),
    ];
    for (name, quantizer) in quantizers {
        for vectors in [1usize, 8, 64, 257] {
            let dimension = quantizer.dim();
            let input: Vec<f64> = (0..dimension * vectors)
                .map(|index| f64::from(u32::try_from(index % 31).unwrap()) / 16.0 - 0.9375)
                .collect();
            let mut output = vec![0i64; input.len()];
            let mut scratch = Scratch::new(dimension);
            let duration = measured(|| {
                nearest_batch(
                    quantizer.as_ref(),
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
            });
            let fingerprint: i128 = output.iter().map(|value| i128::from(*value)).sum();
            println!(
                "quantizer_batch_ns,{dimension},{name}_{vectors},{:.2},{fingerprint}",
                duration.as_secs_f64() * 1e9
            );
        }
    }
}

fn benchmark_kernel_crossover() {
    // Dispatched batch decode versus the scalar per-vector path, per family.
    // The crossover sets `DISPATCH_MIN_VECTORS`; see BENCHMARKS.md.
    println!("kernel,operation,dimension,family,vectors,total_ns");
    let families: Vec<(&str, usize, Box<dyn Quantizer>)> = vec![
        ("zn24", 24, Box::new(Zn::new(24).unwrap())),
        ("dn24", 24, Box::new(Dn::new(24).unwrap())),
        ("dnplus24", 24, Box::new(DnPlus::new(24).unwrap())),
        ("e8", 8, Box::new(e8_quantizer())),
    ];
    for (name, dim, q) in families {
        for vectors in [1usize, 2, 4, 8, 16, 32, 64, 257] {
            let input: Vec<f64> = (0..dim * vectors)
                .map(|index| f64::from(u32::try_from(index % 31).unwrap()) / 16.0 - 0.9375)
                .collect();
            let mut output = vec![0i64; input.len()];
            let mut scalar_output = vec![0i64; input.len()];
            let mut scratch = Scratch::new(dim);
            let scalar = measured(|| {
                for (src, dst) in input
                    .chunks_exact(dim)
                    .zip(scalar_output.chunks_exact_mut(dim))
                {
                    q.nearest(src, dst, &mut scratch).unwrap();
                }
            });
            let dispatched = measured(|| {
                nearest_batch(
                    q.as_ref(),
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
            });
            assert_eq!(output, scalar_output, "{name} at {vectors} vectors");
            println!(
                "kernel,scalar,{dim},{name},{vectors},{:.2}",
                scalar.as_secs_f64() * 1e9
            );
            println!(
                "kernel,dispatched,{dim},{name},{vectors},{:.2}",
                dispatched.as_secs_f64() * 1e9
            );
        }
    }
    // The rounding primitive alone, both executors.
    let plane: Vec<f64> = (0..24 * 257)
        .map(|index| f64::from(u32::try_from(index % 31).unwrap()) / 16.0 - 0.9375)
        .collect();
    let mut rounded = vec![0i64; plane.len()];
    let scalar = measured(|| round_plane_scalar(black_box(&plane), black_box(&mut rounded)));
    let zn = Zn::new(24).unwrap();
    let zn_ref: &dyn Quantizer = &zn;
    let mut zn_scratch = Scratch::new(24);
    let dispatched = measured(|| {
        nearest_batch(
            zn_ref,
            black_box(&plane),
            black_box(&mut rounded),
            black_box(&mut zn_scratch),
        )
        .unwrap()
    });
    println!(
        "kernel,round_plane_scalar,24,zn24,6168,{:.2}",
        scalar.as_secs_f64() * 1e9
    );
    println!(
        "kernel,round_plane_dispatched,24,zn24,6168,{:.2}",
        dispatched.as_secs_f64() * 1e9
    );
}

fn main() {
    benchmark_cvp();
    benchmark_named();
    benchmark_quantizers();
    benchmark_kernel_crossover();
}
