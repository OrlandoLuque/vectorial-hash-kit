# vectorial-hash-kit

Index and hash techniques for vectorial spaces, plus tooling to precompute spatial templates that accelerate culling on evolved quadtree/octree structures.

**Author:** Orlando Jose Luque Moraira

This is a Cargo workspace with several independently-versioned crates.

## Documentation

| Document | What it is |
| --- | --- |
| [`docs/INDEX.md`](docs/INDEX.md) | Map of every doc, crate, test layer, and roadmap pointer in one page. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Methodology + reproducible empirical results for every `vh bench-*` command, plus industry-context references. Five result sections covering single-template culling, per-cell-size selection, descent vs neighbour walk, granularity-as-fallback aggregation, and figure↔grid scale equivalence. |
| [`docs/DEFECTS_FOUND.md`](docs/DEFECTS_FOUND.md) | Living log of every bug the test suite has caught: reproducer, cause, fix, and which test pins it permanently. |
| Per-crate READMEs | [`vectorial-hash`](crates/vectorial-hash/README.md), [`vectorial-hash-templates`](crates/vectorial-hash-templates/README.md), [`vectorial-hash-cli`](crates/vectorial-hash-cli/README.md), [`vectorial-hash-demos`](crates/vectorial-hash-demos/README.md). |
| Publication paper | [`OrlandoLuque/vectorialHash`](https://github.com/OrlandoLuque/vectorialHash) — original May 2019 + June 2026 addendum reporting this reference implementation. |

## Crates

| Crate | Role |
| --- | --- |
| [`vectorial-hash`](crates/vectorial-hash) | Core indexing/hashing algorithm: binary-split spatial tree, `cull` with bbox fallback. Dependency-light, no I/O. |
| [`vectorial-hash-templates`](crates/vectorial-hash-templates) | Precomputes polygon × scale × angle vs. grid intersection templates. |
| [`vectorial-hash-cli`](crates/vectorial-hash-cli) | `vh` binary with subcommands for generation, benchmarks, inspection. |
| [`vectorial-hash-demos`](crates/vectorial-hash-demos) | Runnable demos (`publish = false`). |

## Quick start

```bash
# in-memory template generation (no Redis needed)
cargo run -p vectorial-hash-cli -- generate

# template-generation benchmarks
cargo run -p vectorial-hash-cli --release -- compare   # grids 16 + 32
cargo run -p vectorial-hash-cli --release -- heavy     # grids 16 + 32 + 64 + 128

# runtime cull benchmarks (full results + analysis in docs/BENCHMARKS.md)
cargo run -p vectorial-hash-cli --release -- bench           # tree vs quadtree, single template on/off
cargo run -p vectorial-hash-cli --release -- bench-sizes     # per-cell-size selection (the paper's scheme)
cargo run -p vectorial-hash-cli --release -- bench-walk      # descent vs neighbour-walk traversal
cargo run -p vectorial-hash-cli --release -- bench-fallback  # granularity-as-fallback aggregation
cargo run -p vectorial-hash-cli --release -- bench-scale     # figure↔grid scale equivalence

# console demos (template dedup + end-to-end cull)
cargo run -p vectorial-hash-demos

# visual demo: critters on a live-subdividing map (macroquad)
cargo run -p vectorial-hash-demos --bin critters --release
```

## Template generation modes

The template generator has two storage backends:

### In-memory (default)

Single process, `HashMap`-based template store. Use this for testing, smaller workloads, or when you don't need multi-process coordination.

```bash
cargo run -p vectorial-hash-cli -- generate
```

### Redis-backed (multi-process)

Behind the `redis-store` feature. Multiple processes coordinate via Redis to share template dedup state, with per-task locks and keep-alive TTLs.

```bash
cargo run -p vectorial-hash-cli --features redis-store -- generate-redis \
    --redis-host 127.0.0.1 --redis-port 6379
```

**Redis on this machine:** Redis 3.0.504 is installed at `C:\Program Files\Redis\` and runs as a Windows service. `redis-cli ping` should return `PONG`. No manual start needed.

## Algorithm internals

### Runtime culling (`vectorial-hash`)

- **Binary-split tree**: items live in leaf cells. Rectangles split along the long axis; squares pick the axis that distributes items most evenly.
- **Green / yellow / white short-circuit**: a `Shape` may expose a `TemplateGrid` classifying its coverage cell-by-cell. During `cull`, each tree-node bbox is classified — *green* takes the whole subtree without per-point checks, *white* skips it, *yellow* recurses. Without a template the path falls back to bbox-intersect + per-point.

### Template generation (`vectorial-hash-templates`)

- **Cell classification**: each grid cell is `OUT` / `MAYBE` / `IN` relative to the polygon. The fast path uses bbox rejection + lazy cell construction (`get_template_grid_fast`).
- **8-symmetry dedup**: each generated template is compared (via binary hash) against all 8 rotations/flips (`eq, rCC, rC, r180, fLR, fTB, fTLBR, fTRBL`) so we keep only canonical templates.
- **Binary encoding**: 2-byte header (width, height) + 2 bits per cell. The encoding preserves a known PHP bug for hash compatibility with templates produced by the original `multiDimensionalIndexTemplateCreation` project — bits aren't flushed if cells aren't divisible by 4.
- **Workload split**: subtasks are created with `max_per_task = 500_000` combinations. Rayon parallelises subtasks across CPU threads.

## License & AI/ML training reservation

Dual-licensed under **MIT OR Apache-2.0** — pick whichever fits your project (see [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE)).

In addition, the author reserves all rights regarding the use of this work — source code, documentation, generated artifacts and any derivative works — for **AI/ML training, fine-tuning, evaluation, or any form of text and data mining (TDM)**. The reservation is declared in [`NOTICE`](NOTICE), and expressed in machine-readable form via [`ai.txt`](ai.txt) (Spawning AI) and [`tdmrep.json`](tdmrep.json) (W3C TDM Reservation Protocol). To negotiate a license for those uses, contact **orlando.luque@gmail.com**.
