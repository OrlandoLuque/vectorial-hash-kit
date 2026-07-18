# Changelog

All notable changes to this workspace are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crates are
pre-1.0, so the public API may still change between minor versions.

## [Unreleased]

### Added
- **k-NN on every structure.** `knn(q, k)` now exists on `Tree`, `QuadTree`,
  `IntegerTree`, `Tree3`, `Octree3` and `MortonGrid3` (best-first / ring-by-ring
  with bounding-box pruning). Each is brute-force gated in tests.
- **Stable `ItemRef` handles** on all five trees: `insert_ref` / `update_ref` /
  `remove_ref` relocate an item in O(1), skipping the predicate `update`'s leaf
  scan (~5–10× cheaper on relocation-heavy workloads — the decision-map win).
- **Parallel batch cull** behind the `parallel` feature: `cull_many` (serial,
  always available) and `cull_many_par` (rayon) on every structure. Measured
  crossover in [`docs/PARALLEL.md`](docs/PARALLEL.md).
- **Morton / Z-order linear grids** — `MortonGrid3` (3D) and `MortonGrid`
  (**2D**, new): pointer-free flat spatial hashes with `cull`, `knn`
  (ring-by-ring), `clear`, parallel `extend_par`, and Morton encode/decode.
- **Ray-casting** — the DDA leaf-walk (`Tree::raycast` / `raycast_first`, 2D
  variable-cell with selectable `WalkNeighbors`) and the Amanatides–Woo grid
  walk (`MortonGrid`/`MortonGrid3::raycast` / `raycast_first`), plus the capsule
  shapes (`Capsule` 2D, `Segment3` 3D with analytic `classify_box`/`classify_aabb`)
  for the exact thick band. Full study + benchmarks in [`docs/RAYCAST.md`](docs/RAYCAST.md).
- **Benchmarks + regression gate.** A Criterion suite (`benches/spatial.rs`) and
  a deterministic, committed-baseline regression gate
  (`examples/regression_gate.rs`, min-of-N + clock calibration) that can fail a
  build on a real regression. See [`crates/vectorial-hash/benches/README.md`](crates/vectorial-hash/benches/README.md).
- **CI** (`.github/workflows/ci.yml`): clippy + tests gated on the flagship
  crate, bench/example compile, and a wasm build of the web demos.
- **WebAssembly demos** on GitHub Pages (2D and 3D `critters`, built by CI).
- **Docs:** [`docs/CHOOSING.md`](docs/CHOOSING.md) (structure flowchart),
  [`docs/PARALLEL.md`](docs/PARALLEL.md), and the 3D decision map in
  [`docs/THREE_D.md`](docs/THREE_D.md).

### Changed
- **Nudge-free 3D DDA walk.** `Tree3` and `Octree3` raycast (`raycast_dda` /
  `raycast_dda_first`) now step to the next cell by finding the exact face-neighbour
  via **ascend-to-LCA** (Samet's rope-free neighbour) instead of a `locate`+epsilon
  nudge — exact, no epsilon. Gated by a new completeness test (every crossed leaf is
  visited) plus a config-fuzz, on top of the existing subset-of-capsule / first-hit
  gates. See [`docs/RAYCAST.md`](docs/RAYCAST.md).
- Cleaned the `vectorial-hash` crate to be clippy-clean under `-D warnings`
  (redundant closures, a derivable `Default`).

### Known issues
- `vectorial-hash-templates`' fingerprint-regression test and ~21 clippy lints
  are not yet portable to the Linux CI runner (advisory there for now). Tracked
  in [`docs/BACKLOG.md`](docs/BACKLOG.md).

## [0.1.0]

Initial workspace: the binary-split `Tree` with template-driven culling, the
dynamic `remove` / `update` contract (paper merge-up rule), `QuadTree` and
`IntegerTree` reference structures, the 3D `Tree3` / `Octree3`, the
templates-crate adapter, the CLI, and the 2D/3D `critters` demos.
