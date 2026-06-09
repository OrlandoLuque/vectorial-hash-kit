# vectorial-hash-kit

Index and hash techniques for vectorial spaces, plus tooling to precompute spatial templates that accelerate culling on evolved quadtree/octree structures.

**Author:** Orlando Jose Luque Moraira

This is a Cargo workspace with several independently-versioned crates.

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

# benchmark variants
cargo run -p vectorial-hash-cli -- compare   # grids 16 + 32
cargo run -p vectorial-hash-cli -- heavy     # grids 16 + 32 + 64 + 128

# tiny demo
cargo run -p vectorial-hash-demos
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

## Algorithm internals (templates crate)

- **Cell classification**: each grid cell is `OUT` / `MAYBE` / `IN` relative to the polygon. The fast path uses bbox rejection + lazy cell construction (`get_template_grid_fast`).
- **8-symmetry dedup**: each generated template is compared (via binary hash) against all 8 rotations/flips (`eq, rCC, rC, r180, fLR, fTB, fTLBR, fTRBL`) so we keep only canonical templates.
- **Binary encoding**: 2-byte header (width, height) + 2 bits per cell. The encoding preserves a known PHP bug for hash compatibility with templates produced by the original `multiDimensionalIndexTemplateCreation` project — bits aren't flushed if cells aren't divisible by 4.
- **Workload split**: subtasks are created with `max_per_task = 500_000` combinations. Rayon parallelises subtasks across CPU threads.
