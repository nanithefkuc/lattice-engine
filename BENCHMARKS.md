# Benchmarks and decoder measurements

Performance thresholds and decoder behavior are recorded here rather than in
API documentation. Re-run the named harness before changing either policy.

The decoder records marked *moved* were measured in `lattica` before the
extraction (2026-08-12/13, Intel Core Ultra 7 258V, `rustc 1.93.0`, pinned
corpora) and moved with the code they describe; the fingerprints and word
counts are properties of the algorithms, which moved verbatim. Post-move
verification: `cargo run --release --example highdim_ml` reproduces the
`Λ_24` word-error table exactly (385/2000 at radius 1.25, zero budget
exhaustion), and `cargo run --release --example e8_awgn` passes the shaping
gate. A full pinned-core re-baseline is recorded in the first section
below.

## Post-extraction re-baseline

Commands, pinned to CPU 2:

```sh
taskset -c 2 cargo bench --bench optimization
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-17 on the original Intel Core Ultra 7 258V, `rustc 1.93.0`,
`lattica` `6178a52` pinned, after the `round_away` dedup. This is the
regression gate for the extraction itself: same corpora, same fingerprints.

The CVP comparison corpus reproduces the pre-move fingerprints *exactly*
(target `-356691156`/`13549574`/`-217921229`, point `-364109`/`15984`/
`-153113`, distance `20641984069`/`42334915811`/`61652689299` at dimensions
8/16/24) with warm medians of `616.84 ns`, `5.014 µs`, and `25.662 µs`
against the recorded in-`lattica` `0.635 µs`, `4.795 µs`, and `24.699 µs` —
within ~3%, so the cross-crate boundary is performance-neutral. Cold medians:
`1.510 µs`, `11.694 µs`, `45.919 µs`.

The optimization corpus keeps its recorded structure: CVP preparation
`1.021 µs`/`5.778 µs`/`18.395 µs` (recorded post-Bareiss:
`1.098`/`6.558`/`21.134`); node counts identical — 8/12/16, 16/37/88, and
24/133/402 on the easy/median/boundary classes; and the named decoders visit
**18 nodes for `BW_16` and 19,202 for `Λ_24`**, exactly the recorded values,
with setup medians of `55.8 µs` and `164.4 µs`. Closed-form batch decoding
stays scalar and in family: `e8_257` at `10.07 µs`, `dn24_257` at
`12.21 µs`, `an23_257` at `25.00 µs`.

Decision: no dispatch or crossover changes. The extraction is
performance-neutral; the first engine-owned kernel decision (06-optimizations
P2 class) starts from these numbers.

## Barnes–Wall and Leech decoding beyond packing radius *(moved)*

Command:

```sh
cargo run --release --example highdim_ml
```

Measured 2026-08-12 on the original machine and toolchain. For each radius,
2,000 deterministic directions were sampled uniformly from a normalized cube
vector on the ambient sphere. The transmitted point was zero. A word error
means the maximum-likelihood point was nonzero; budget exhaustion is reported
separately. The node budget was `2^24` per point.

| Lattice | Radius | Word errors | Budget exhausted |
| --- | ---: | ---: | ---: |
| `BW_16` | 0.95 | 0 / 2000 | 0 / 2000 |
| `BW_16` | 1.05 | 0 / 2000 | 0 / 2000 |
| `BW_16` | 1.25 | 738 / 2000 | 0 / 2000 |
| `BW_16` | 1.50 | 2000 / 2000 | 0 / 2000 |
| `Λ_24` | 0.95 | 0 / 2000 | 0 / 2000 |
| `Λ_24` | 1.05 | 0 / 2000 | 0 / 2000 |
| `Λ_24` | 1.25 | 385 / 2000 | 0 / 2000 |
| `Λ_24` | 1.50 | 2000 / 2000 | 0 / 2000 |

Both lattices have minimal squared norm 4, so radius 1 is the guaranteed
unique decoding radius. The implementation deliberately uses
maximum-likelihood Schnorr–Euchner search instead of a bounded-distance
recursion or hexacode shortcut: success is the globally nearest point, and an
insufficient node budget is an error. This avoids a second, lattice-specific
failure region. The out-of-radius table measures channel word errors, not
hidden algorithm errors.

Ambient outputs retain exact algebraic scaling. `BW_16` returns numerators
over 2; `Λ_24` returns numerators over `sqrt(8)`. Returning approximate
`f64` lattice points was rejected because it would turn a discrete answer
into a rounding question.

## Optimization corpus, decode half *(moved)*

Command:

```sh
taskset -c 2 cargo bench --bench optimization
```

Measured 2026-08-12 as above. The decode groups carry CVP preparation, nodes,
and nanoseconds per node; named-decoder setup and total latency; and
closed-form batch quantizers. Every row carries a geometry name and
deterministic correctness fingerprint.

Warm CVP on the comparison corpus improved from `1.512 µs`, `16.540 µs`, and
`120.107 µs` to `0.635 µs`, `4.795 µs`, and `24.699 µs` at dimensions 8, 16,
and 24, with unchanged target, point, and distance fingerprints. CVP
preparation improved a further `12.8%`, `33.6%`, and `32.6%` with the
symmetric Bareiss factorization pass.

Strong reduced-basis preconditioning is the selected proof-tree optimization.
On the named target `[0.31; n]`, `BW_16` visits 18 nodes and `Λ_24` visits
19,202 nodes; the full 2,000-word radius sweep above reports no budget
exhaustion. Stronger floating lower bounds and deterministic multi-start Babai
candidates were therefore not added: neither has a measured exhaustion case to
solve, and both would add per-node or per-word work to the default path.
Single-word subtree scheduling remains deferred; independent received words
are the deterministic parallel boundary.

The specialized Barnes–Wall recursion and Leech hexacode candidate engines
were not selected after this measurement. Preconditioned exhaustive search
already meets the ML sweep without exhaustion, so a second candidate
implementation would add tables, scratch, and a new membership proof without a
failing workload to recover. Decoder construction remains setup work: measured
medians are `69.6 µs` for `BW_16` and `177.7 µs` for `Λ_24`. Precomputed dual
tables were also rejected because they would replace independently checked
exact construction for an unmeasured cold-path saving.

Closed-form quantizer batches remain scalar: the corpus identifies no
layout-preserving consumer or repeatable gain that would justify a second
semantic implementation. Construction-A membership allocates one residue
buffer per call; no consumer repeats it in a hot path, so prepared public APIs
and extra scratch types were not added speculatively.

## Comparison target selection *(moved, decoder half)*

The specialized decoders do not yet have one contract-equivalent library
target. The published
[`leech-decoding`](https://github.com/avanpo/leech-decoding) implementation is
a worthwhile algorithmic ceiling for `Λ_24`, but it accepts bounded integer
representatives and targets a constant-time cryptographic contract rather
than arbitrary real maximum-likelihood queries. BLAS is likewise only a
throughput ceiling for the real transform because it may reassociate or use
FMA, while the stack promises scalar accumulation order and bit-identical
dispatched results. Neither should receive a headline ratio until an adapter
verifies identical input, output, distance, and tie behavior.

## fplll CVP comparison *(moved)*

The harness pins fplll 5.5.0 at commit
`a8dedce384689047daba154bd50d6215e35bf03b`. Build the fplll side and this
crate's side from this repository:

```sh
git clone --depth 1 --branch 5.5.0 \
  https://github.com/fplll/fplll.git target/fplll-5.5.0
cd target/fplll-5.5.0
./autogen.sh
./configure --disable-shared CXXFLAGS="-O3 -march=native -DNDEBUG"
make -j
cd ../..

c++ -O3 -march=native -DNDEBUG -std=c++17 \
  -Itarget/fplll-5.5.0 benches/fplll_compare.cpp \
  target/fplll-5.5.0/fplll/.libs/libfplll.a \
  -lmpfr -lgmp -lpthread -o target/fplll-compare

taskset -c 2 target/fplll-compare
taskset -c 2 cargo bench --bench fplll_compare
```

The CVP corpus uses one exactly `δ = 0.99` LLL-reduced upper-bidiagonal basis
per dimension and 128 deterministic targets. The diagonal is 2 and the
superdiagonal is 1. Targets have denominator 1009; fplll receives the
equivalent integer problem with both basis and target scaled by 1009.

The pre-extraction measurement (same corpus, then inside `lattica`):

| Dimension | cold | warm | fplll `FAST` | fplll `PROVED` |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1.561 µs | 0.629 µs | 21.425 µs | 49.089 µs |
| 16 | 11.499 µs | 4.952 µs | 57.355 µs | 103.699 µs |
| 24 | 44.192 µs | 25.381 µs | 131.268 µs | 210.412 µs |

Against fplll `FAST`, warm decoding was 34.1x, 11.6x, and 5.17x faster at
dimensions 8, 16, and 24; cold was 13.7x, 4.99x, and 2.97x faster. These are
throughput comparisons, not equivalent guarantees: this crate retains exact
pruning plus an explicit node budget, while fplll `FAST` does not promise the
closest point.

The output fingerprints expose a correctness problem in fplll 5.5.0
`CVPM_PROVED`. `FAST` and this crate reported identical target, point, and
distance fingerprints. `PROVED` reported different point and larger distance
fingerprints; one independently checkable miss is frozen in
`benches/data/fplll_proved_miss.txt`:

```sh
target/fplll-5.5.0/fplll/fplll -a cvp \
  < benches/data/fplll_proved_miss.txt
```

fplll returns
`[-26234, -1009, 14126, -2018, 18162, 11099, -13117, -2018]`, at squared
distance 3602525. The lattice point
`[-26234, -1009, 14126, -2018, 18162, 12108, -14126, -4036]` has squared
distance 3053629 to the same target. A lower-distance lattice point is enough
to disprove the claimed closest result. Consequently, the `PROVED` timings
above are diagnostic and are not used for a speedup claim.

### fplll Babai cycle

The first attempted corpus also exposed a non-terminating fplll CVP prepass.
The minimal input is retained in `benches/data/fplll_babai_cycle.txt`:

```sh
timeout 1 target/fplll-5.5.0/fplll/fplll -a cvp \
  < benches/data/fplll_babai_cycle.txt
```

The process emits `warning: possible infinite loop in Babai's algorithm` and
does not terminate before the timeout. Instrumenting the pinned source showed
the residual alternating between
`[1, 0, 1, -1, 1, -1, 1, 0]` and
`[-1, 1, -1, 0, -1, 0, -1, 0]`; the rounded Babai coefficients alternate with
opposite signs, so neither iteration satisfies the stop condition.

The relevant fplll loop
[only warns at power-of-two iteration counts and has no hard limit](https://github.com/fplll/fplll/blob/5.5.0/fplll/svpcvp.cpp#L571-L595).
The comparison corpus uses denominator-1009 targets in general position so
the timing harness terminates, but the reproducer remains part of the
benchmark record rather than being hidden by that dataset choice.
