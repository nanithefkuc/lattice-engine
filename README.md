> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things can break. Audit the code yourself, or with
> your own agent before using.

# lattice-engine

`lattice-engine` is the lattice decoding engine of the FEC stack: the
closest-vector, sphere, and maximum-likelihood decisions that turn a received
real vector into an exact lattice point. It sits one layer above
[`lattica`](https://github.com/nanithefkuc/lattica), which owns the lattice
arithmetic — Gram matrices, exact integer linear algebra, fraction-free
orthogonalization, LLL reduction, named constructions, and the real-vector
kernels this crate composes.

It provides:

- The `Quantizer` trait — the decode seam (`dim`, `scale`, `nearest`) — with
  caller-owned `Scratch`, batch decoding, and strict validation.
- Closed-form Conway–Sloane nearest-point decoders for `Z^n`, `A_n`, `D_n`,
  `D_n^+` (and `E_8`), `O(n)` per point and using only add, subtract,
  compare, and round, so two peers on different architectures agree at a
  Voronoi boundary.
- Babai rounding and nearest-plane: bounded-work approximations, exact on
  orthogonal bases.
- Budgeted Schnorr–Euchner nearest-point and list enumeration over any
  integral Gram matrix, prepared or per-call; node budgets are hard limits
  and exhaustion is an error, never an approximate answer.
- Exact maximum-likelihood decoding of `BW_16` and the Leech lattice over
  their published generators.
- `mod Λ` with dithering and scaled lattices: the quantization-error side
  that nested lattice codes shape with.
- Construction A decoding through the caller-supplied `CodeMembership` seam —
  maximum-likelihood whenever the caller's code decoder is.

The break line with `lattica` is facts versus decisions: `lattica` computes
facts about a lattice, including the code-free Construction A/D generator
constructions; this crate decides lattice points. Tie rules (rounding away
from zero, lowest-index worst coordinate, `D_n^+` coset preference) are wire
format, guarded by frozen fixtures.

The `E_8` release gate reproduces the published `0.6539 dB` shaping gain of
the `E_8` Voronoi region:

```sh
cargo run --release --example e8_awgn
```

Decoded points are decisions about noisy inputs, not authenticated data;
 FEC supplies no integrity. See the crate documentation for the exactness and
determinism argument.

## Usage

The MSRV is Rust 1.89.

`lattice-engine` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
lattice-engine = { git = "https://github.com/nanithefkuc/lattice-engine" }
```

### Features

| Feature | Result |
| --- | --- |
| default (`simd`) | the engine-owned AVX2 batch-decode kernel over `nearest_batch`'s flat layout (`Z^n`, `D_n`, `D_n^+`/`E_8`; dispatched from 8 vectors, bit-identical to the scalar path), plus `lattica`'s dispatched real-vector kernels — selection single-source through `simdispatch` in both cases |
| `internals` | unstable implementation APIs, exempt from compatibility guarantees |

## Building

```sh
cargo build                     # default: simd
cargo build --features internals
cargo test
```

Decode measurements — the `BW_16`/`Λ_24` word-error curves, enumeration node
counts, and the reproducible fplll closest-vector comparison — are recorded in
[`BENCHMARKS.md`](BENCHMARKS.md).

## License

MIT.
