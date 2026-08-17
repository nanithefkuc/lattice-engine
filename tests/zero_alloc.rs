//! Invariant I5: the hot path allocates nothing in steady state.
//!
//! A decoder that allocates per received vector is not a decoder for a
//! real-time codec, and the cost is invisible in a throughput benchmark on an
//! unloaded machine. This measures it directly.
//!
//! The counter is thread-local rather than global: the test harness allocates
//! on its own threads, and a global count would measure that noise instead of
//! the decoder. `Cell` with a `const` initializer keeps the counting path
//! itself allocation-free, so it cannot recurse into the allocator.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the only addition is a
// thread-local increment that performs no allocation of its own.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn bump() {
    let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `body` and returns how many allocations it made on this thread.
fn allocations_during<F: FnOnce()>(body: F) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    body();
    ALLOCATIONS.with(Cell::get) - before
}

use core::num::NonZeroU32;

use lattica::error::DecodeError;
use lattica::named::d_n;
use lattica::reduce::Delta;
use lattica::relevant::relevant_vectors;
use lattica::shortvec::census;
use lattice_engine::{
    An, Dn, DnPlus, EnumerationScratch, Enumerator, PreparedEnumerationScratch, PreparedEnumerator,
    Quantizer, Scratch, Zn, e8, nearest_batch,
};
use lattice_engine::{CodeMembership, ConstructionA};

/// Deterministic dyadic coordinates, so the measurement is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (f64::from(u32::try_from(self.0 % 8192).unwrap()) - 4096.0) / 1024.0
    }
}

fn check_batch(name: &str, q: &dyn Quantizer, vectors: usize) {
    let dim = q.dim();
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let points: Vec<f64> = (0..dim * vectors).map(|_| rng.next()).collect();
    let mut out = vec![0i64; dim * vectors];
    let mut scratch = Scratch::new(dim);

    // Warm every buffer to its high-water mark before measuring, including
    // the batch kernel planes, which grow to the full batch length once.
    nearest_batch(q, &points, &mut out, &mut scratch).unwrap();

    let allocations = allocations_during(|| {
        nearest_batch(q, &points, &mut out, &mut scratch).unwrap();
    });
    assert_eq!(
        allocations, 0,
        "{name} allocated {allocations} times over {vectors} vectors"
    );
}

#[test]
fn steady_state_batch_decoding_allocates_nothing() {
    // Roughly one million decoded coordinates per family.
    check_batch("Z^24", &Zn::new(24).unwrap(), 40_000);
    check_batch("D_24", &Dn::new(24).unwrap(), 40_000);
    check_batch("A_23", &An::new(23).unwrap(), 40_000);
    check_batch("D_24^+", &DnPlus::new(24).unwrap(), 40_000);
    check_batch("E_8", &e8(), 125_000);
}

#[test]
fn a_cold_scratch_grows_once_and_never_again() {
    let q = Dn::new(16).unwrap();
    let mut scratch = Scratch::default();
    let mut out = [0i64; 16];
    let x = [0.3f64; 16];

    // The first call is allowed to allocate; it is the only one that may.
    let cold = allocations_during(|| {
        q.nearest(&x, &mut out, &mut scratch).unwrap();
    });
    assert!(cold > 0, "expected the cold path to allocate its buffers");

    let warm = allocations_during(|| {
        for _ in 0..10_000 {
            q.nearest(&x, &mut out, &mut scratch).unwrap();
        }
    });
    assert_eq!(warm, 0, "warm decoding allocated {warm} times");
}

#[test]
fn steady_state_enumeration_allocates_nothing() {
    let gram = d_n::<i64>(8).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let mut scratch = EnumerationScratch::new();
    let mut out = [0i64; 8];
    let target = [0.17, -0.31, 0.43, -0.59, 0.71, -0.83, 0.97, -1.09];

    enumerator
        .nearest(&target, &mut out, 100.0, 1 << 20, &mut scratch)
        .unwrap();
    let allocations = allocations_during(|| {
        for _ in 0..1_000 {
            enumerator
                .nearest(&target, &mut out, 100.0, 1 << 20, &mut scratch)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "warm enumeration allocated {allocations} times"
    );
}

#[test]
fn steady_state_prepared_enumeration_allocates_nothing() {
    let gram = d_n::<i64>(8).unwrap();
    let enumerator = PreparedEnumerator::new(&gram, Delta::STRONG).unwrap();
    let mut scratch = PreparedEnumerationScratch::new();
    let mut out = [0i64; 8];
    let target = [0.17, -0.31, 0.43, -0.59, 0.71, -0.83, 0.97, -1.09];

    enumerator
        .nearest_ml(&target, &mut out, 1 << 20, &mut scratch)
        .unwrap();
    let allocations = allocations_during(|| {
        for _ in 0..1_000 {
            enumerator
                .nearest_ml(&target, &mut out, 1 << 20, &mut scratch)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "warm prepared enumeration allocated {allocations} times"
    );
}

struct ParityCheck;

impl CodeMembership for ParityCheck {
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
        residues.iter().sum::<u32>().is_multiple_of(2)
    }

    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError> {
        if costs.len() != 8 || out.len() != 4 {
            return Err(DecodeError::LengthMismatch {
                expected: 4,
                found: out.len(),
            });
        }
        out.fill(0);
        Ok(())
    }
}

#[test]
fn repeated_materialized_workload_allocations_are_counted() {
    let gram = d_n::<i64>(4).unwrap();
    let enumerator = Enumerator::new(&gram).unwrap();
    let mut scratch = EnumerationScratch::new();
    let target = [0.17, -0.31, 0.43, -0.59];

    let list_allocations = allocations_during(|| {
        let points = enumerator
            .list(&target, 4.0, 1 << 20, &mut scratch)
            .unwrap();
        assert!(!points.is_empty());
    });
    assert!(
        list_allocations > 0,
        "list output was not allocation-proportional"
    );

    let relevant_allocations = allocations_during(|| {
        assert!(!relevant_vectors(&gram, 1 << 20).unwrap().is_empty());
    });
    assert!(
        relevant_allocations > 0,
        "relevant-vector materialization did not allocate"
    );

    let census_allocations = allocations_during(|| {
        assert!(census(&gram, 1 << 20).unwrap().min_norm_sq.is_some());
    });
    assert!(census_allocations > 0, "cold exact census did not allocate");

    let construction = ConstructionA::new(ParityCheck).unwrap();
    let membership_allocations = allocations_during(|| {
        for _ in 0..100 {
            assert!(construction.contains(&[1, 1, 0, 0]).unwrap());
        }
    });
    assert_eq!(
        membership_allocations, 100,
        "Construction A membership allocation count changed"
    );
}

#[test]
fn error_paths_do_not_allocate_either() {
    let q = Dn::new(8).unwrap();
    let mut scratch = Scratch::new(8);
    let mut out = [0i64; 8];
    let bad = [f64::NAN; 8];
    let short = [0.0f64; 4];

    // Warm up.
    q.nearest(&[0.5f64; 8], &mut out, &mut scratch).unwrap();

    let allocations = allocations_during(|| {
        for _ in 0..1000 {
            assert!(q.nearest(&bad, &mut out, &mut scratch).is_err());
            assert!(q.nearest(&short, &mut out, &mut scratch).is_err());
        }
    });
    assert_eq!(allocations, 0, "rejection allocated {allocations} times");
}
