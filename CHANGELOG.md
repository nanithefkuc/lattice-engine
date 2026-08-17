# Changelog

All notable changes to `lattice-engine` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.0]

Initial release: the lattice decoding engine, extracted from `lattica`.

### Added

- **The decode seam.** The `Quantizer` trait (`dim`/`scale`/`nearest`),
  caller-owned `Scratch`, `nearest_batch` over flat strided buffers, strict
  input validation with an exact `COORD_LIMIT`, and the shared
  `lattica::error::DecodeError` vocabulary, re-exported.
- **Closed-form Conway–Sloane decoders** for `Z^n`, `A_n`, `D_n`, `D_n^+`
  (`E_8` at `n = 8`) with specified tie rules: rounding ties away from zero,
  the worst-coordinate tie at the lowest index, and the `D_n^+` `D_n`-coset
  preference. `O(n)` per point using add, subtract, compare, and round only.
- **Babai rounding and nearest-plane**, the coefficient solve against the
  exact adjugate, and the Gram-metric distance.
- **Budgeted Schnorr–Euchner enumeration**: `Enumerator` over the exact
  fraction-free factorization, `PreparedEnumerator` with LLL preprocessing
  and the stored unimodular transform, and radius-list enumeration. Node
  budgets are hard limits; exhaustion is a typed error.
- **Exact maximum-likelihood `BW_16` and Leech decoding** over the published
  generators, with budget exhaustion reported separately from word errors.
- **`mod Λ`** with dithered modulo and `Scaled` lattices — the
  quantization-error side consumed by nested lattice codes.
- **Construction A decoding** through the `CodeMembership` seam, which
  carries per-symbol costs so the decode is maximum-likelihood whenever the
  caller's code decoder is. The generator construction stays in `lattica`.

### Decoding not included

Everything else about lattices — representation, exact integer linear
algebra, reduction, named constructions, structural enumeration, real-vector
kernels — remains `lattica`'s and is consumed at a pinned revision.
