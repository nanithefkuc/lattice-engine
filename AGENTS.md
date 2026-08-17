# AGENTS.md

Working rules for `lattice-engine`: the operational summary of how the crate
is built and tested.

## What this crate is

The lattice decoding engine — closest-vector, sphere, and maximum-likelihood
decisions — one layer above `lattica`. **`lattica` computes facts about a
lattice; this crate decides points on it.** Not a codec. Not field or graph
arithmetic. Not lattice cryptography, ever.

The break line, applied literally: a function that receives a real target (or
a code seam) and selects a lattice point lives here; a function that computes
a quantity, transform, certificate, or constructor of a lattice lives in
`lattica`.

## Hard rules

1. **`lattica` is the only runtime dependency.** Every lattice operation —
   Gram matrices, GSO, LLL, integer linear algebra, named generators,
   real-vector kernels — is `lattica`'s at a pinned git revision. No local
   elimination, orthogonalization, or reduction. No `fgf`, `sgraph`, or `gfm`
   ever; CI fails the build if one appears.
2. **No `unsafe`.** Forbidden at the crate root. SIMD is reached through
   `archmage`'s safe intrinsics under `simdispatch` selection — the batch
   decode kernel owns this crate's flat layout — and through `lattica`'s
   kernels everywhere else.
3. **Decision paths use add, subtract, compare, and round.** No `mul_add`, no
   transcendental, no reassociation, no `f64::round` anywhere a lattice point
   is chosen. Two peers must decode a boundary point identically; the
   operation set is the proof. Do not "simplify" this.
4. **Tie rules are wire format.** Rounding ties away from zero; the
   worst-coordinate tie at the lowest index; `D_n^+` prefers the `D_n` coset.
   `tests/data/ties.txt` is format: changing an expected value is a wire
   break requiring a versioned decoder, not an edited line.
5. **Budgets are hard limits.** Enumeration node budgets and ML budgets bound
   every search; exhaustion is a typed error, never an approximate answer,
   and never a silent radius narrowing.
6. **Validate before mutating.** A rejected call leaves every output buffer
   and all internal state exactly as it was, including the batch contract's
   partial-write order.
7. **`DecodeError` is `lattica`'s.** Re-exported here; one vocabulary across
   the seam. Engine-specific outcomes extend status types, not a parallel
   error enum.

## Testing

A test whose expected value came from this crate's own output is not a test.
Every algorithm has an independent oracle: brute force, `lattica::shortvec`,
packing radius, published constants, or an exhaustive mock code.

Fixtures under `tests/data/` are format. Moving a file is fine; changing an
expected value is a wire break. Assert on *distances* under translation and
negation, never on points — the tie set makes point-equality false on Voronoi
boundaries by design.

## Commands

```sh
cargo test --all-features
cargo test --no-default-features
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

## Release gate

`cargo run --release --example e8_awgn` must reproduce the published
`0.6539 dB` `E_8` shaping gain within five standard errors. If the measured
gain leaves that band, something is broken — the tolerance cannot be widened
to pass.
