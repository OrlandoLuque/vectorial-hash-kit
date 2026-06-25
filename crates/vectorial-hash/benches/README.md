# Benchmarks

Two complementary tools — one for *exploring* performance, one for *gating* it.

## 1. Criterion — rich local exploration

`benches/spatial.rs` benchmarks the structures across build / cull / update /
knn with the [Criterion](https://github.com/bheisler/criterion.rs) harness
(statistical sampling, outlier detection, HTML reports under `target/criterion/`).

```bash
cargo bench -p vectorial-hash                          # everything
cargo bench -p vectorial-hash -- cull                  # filter by name
cargo bench -p vectorial-hash -- --save-baseline main  # snapshot a baseline
cargo bench -p vectorial-hash -- --baseline main       # compare against it
```

Use this while tuning: it tells you *how much* faster/slower a change made each
op, with confidence intervals. Its baselines live under `target/` (not
committed) and it does not fail a build.

## 2. The regression gate — committed, deterministic, can fail a build

`examples/regression_gate.rs` is the opposite trade-off: a small fixed set of
timings (min-of-N, not statistical), checked against a **committed** baseline
(`benches/baseline.tsv`), exiting non-zero when an op regresses past a threshold.

```bash
# capture the baseline on THIS machine, ideally quiet (close heavy apps):
cargo run -p vectorial-hash --example regression_gate --release -- --save
# later / in CI — compare, exit 1 if any op regressed > threshold (default 25%):
cargo run -p vectorial-hash --example regression_gate --release
cargo run -p vectorial-hash --example regression_gate --release -- --threshold 0.40
```

Two design choices make it robust enough to gate on:

- **min-of-N** — the fastest sample is the truest cost; noise only ever adds
  time, so the minimum is the least-disturbed run. Far steadier than a mean or
  median for sub-millisecond ops.
- **clock calibration** — a fixed CPU loop (`_calib`) is timed in the same run
  and every op is compared as a *ratio* to it. A machine running 1.3× slower
  overall (turbo down, thermal, background load) scales `_calib` the same way,
  so the ratio cancels and a uniform slowdown does **not** read as a regression.
  This is what took back-to-back runs from ±60% to ±6%.

It catches **gross / algorithmic** regressions (an accidental O(n²), a lost
short-circuit, a dropped fast path) — not 5% micro-tuning, which is Criterion's
job. The baseline is hardware-specific: regenerate with `--save` when you change
machines, and keep the saved `baseline.tsv` under version control so the gate
has something to compare against.

### Current baseline (this machine, orientation only)

Captured on the dev box (16-thread). Absolute numbers vary by hardware; the
*shape* is the point — note `update_predicate` ≈ 5–6× `update_ref` (the stable
`ItemRef` win) and Morton's cheap build + cull.

| op | ns | note |
| --- | ---: | --- |
| `build_tree3` | ~6.1M | binary build, 20k pts |
| `build_octree3` | ~4.6M | 8-way build |
| `build_morton3` | ~1.7M | flat grid build (cheapest) |
| `cull_tree3_x64` | ~125k | 64 sphere culls |
| `cull_octree3_x64` | ~114k | |
| `cull_morton3_x64` | ~64k | flat grid cull (cheapest) |
| `knn_tree3_k16_x256` | ~576k | 256 k-NN queries, k=16 |
| `update_predicate_frame` | ~3.16M | 20k relocations, predicate `update` |
| `update_ref_frame` | ~653k | 20k relocations, `update_ref` (**~5× faster**) |
