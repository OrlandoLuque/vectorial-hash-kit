# vectorial-hash-cli

Command-line tools for the [`vectorial-hash`](../vectorial-hash) workspace. Installs as the `vh` binary.

## Subcommands

| Subcommand | Storage | Notes |
| --- | --- | --- |
| `generate` | in-memory | Default. Single process, `HashMap`-backed dedup. Accepts `--angle-step`, `--scale`, `--grid`. |
| `compare` | in-memory | Benchmark across grids 16 + 32. |
| `heavy` | in-memory | Benchmark across grids 16 + 32 + 64 + 128. |
| `bench` | in-memory | 4-way cull benchmark: binary-split tree vs quadtree, templates on/off. Accepts `--points`, `--culls`, `--item-limit`, `--seed`. |
| `bench-sizes` | in-memory | Per-cell-size template benchmark: no templates vs the old single-grid snap method vs the hierarchical bank capped at ≤16/≤32/≤64 px cells, with and without the 1×1 leaf raster. Same flags as `bench`. |
| `bench-walk` | in-memory | Traversal benchmark: tree descent vs flood-fill over leaf neighbours (Samet ascent / locate probe / stored ropes). Adds `--scale`; run with `--no-default-features` to build without rope bookkeeping. |
| `generate-redis` | Redis | Requires feature `redis-store`. Multi-process via per-task locks. Accepts the same generation params as `generate`. |

## Usage

```bash
# default in-memory generation (0.5deg step, scale 128, grid 16)
cargo run -p vectorial-hash-cli --release -- generate

# quick smoke generation
cargo run -p vectorial-hash-cli --release -- generate --angle-step 15 --scale 64 --grid 16

# template-generation benchmarks
cargo run -p vectorial-hash-cli --release -- compare
cargo run -p vectorial-hash-cli --release -- heavy

# runtime cull benchmark (4 configs over the same point cloud)
cargo run -p vectorial-hash-cli --release -- bench
cargo run -p vectorial-hash-cli --release -- bench --points 1000000 --culls 100 --item-limit 8

# Redis-backed (requires the feature)
cargo run -p vectorial-hash-cli --release --features redis-store -- generate-redis \
    --redis-host 127.0.0.1 --redis-port 6379
```

## `bench`: what it measures

Builds a deterministic random point cloud in a 4096×4096 world, indexes it twice (the `vectorial-hash` binary-split tree and a reference quadtree), and culls a large rotated drop polygon repeatedly in four configurations:

1. binary-split tree + precomputed `TemplateGrid` (green/yellow/white short-circuit)
2. binary-split tree, per-point polygon test only
3. quadtree + the same `TemplateGrid`
4. quadtree, per-point polygon test only

All four must return the same hit count (asserted) before timing starts. Reported: per-config totals, per-cull averages, and the speedup ratios.

## Features

| Feature | Effect |
| --- | --- |
| `redis-store` | Enables the `generate-redis` subcommand. Propagates to `vectorial-hash-templates/redis-store`. |
