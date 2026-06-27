# Documentation index

The workspace is split between the published paper, the runnable crates,
and a growing set of design notes that came out of the implementation.
This file is the map; everything else is one click away.

## Theory

| Document | Where | What |
| --- | --- | --- |
| [Multidimensional vector index](https://github.com/OrlandoLuque/vectorialHash) | Sibling repo `vectorialHash` | The paper. Original (May 2019) + June 2026 addendum covering the reference implementation and the empirical results below. |

## Implementation overview

| Crate | Role | Surface |
| --- | --- | --- |
| [`vectorial-hash`](../crates/vectorial-hash/README.md) | Core spatial index | `Tree`, `Shape`, `Tree::cull`, `Tree::cull_walk`, `TemplateGrid`, `PlacedTemplate`, `Side`, `WalkNeighbors` |
| [`vectorial-hash-templates`](../crates/vectorial-hash-templates/README.md) | Template generator and runtime bank | `polygon`, `templates`, `matrix`, `adapter`, `bank::TemplateBank`, `fingerprint` |
| [`vectorial-hash-cli`](../crates/vectorial-hash-cli/README.md) | `vh` command — generation, inspection, benchmarks | `generate`, `compare`, `heavy`, `bench`, `bench-sizes`, `bench-walk`, `bench-fallback`, `bench-scale`, `templates-fingerprint`, `generate-redis` |
| [`vectorial-hash-demos`](../crates/vectorial-hash-demos/README.md) | Runnable demos (visual + console + headless) | Console end-to-end pipeline, the live macroquad critters demo (binary / quadtree / dual mode), and `critters_headless` — the same deterministic sim core without a window, reporting per-operation statistics |

## Empirical results & methodology

| Document | Covers |
| --- | --- |
| [BENCHMARKS.md](BENCHMARKS.md) | Methodology and reproducible results for every `vh bench-*` command, with industry-context references. Five result sections (single-template, per-cell-size selection, descent vs neighbour walk, granularity-as-fallback, figure↔grid scale equivalence). |
| [UPDATE_STRATEGIES.md](UPDATE_STRATEGIES.md) | Empirical comparison of `Tree::update` relocation strategies (Legacy / Lca / LcaRopes), cost of the `neighbors` feature, and the IntegerTree<T> bit-shift experiment. 135-cell formal sweep + IntegerTree head-to-head, item_limit×merge_limit tuning, cull_walk break-even, and quad-vs-binary, with verdict on the API defaults. |
| [THREE_D.md](THREE_D.md) | 3D indexing: true 3D tree (`Tree3`, binary split, analytic sphere classification, 1×1×1 `VoxelRaster`) vs the author's projection-indexing idea (three 2D trees + exact narrowphase). Static time/precision comparison + the `critters3d_headless` dynamic workload. |
| [PARALLEL.md](PARALLEL.md) | Where threads pay and where they don't: the `cull_many_par` batch crossover (measured), and the per-unit AI fan-out (`par_iter_mut`, ~11–12× on 16 threads) that the `siege` demo runs on. Reads parallelise; writes stay serial. |
| [RAYCAST.md](RAYCAST.md) | Ray-casting across every structure: thin-ray DDA vs thick-capsule cull, the analytic `classify_box`/`classify_aabb` pruning hooks, the SoA + SIMD narrowphase, and the exhaustive anti-contamination benchmarks. |
| [SIEGE.md](SIEGE.md) | The `siege` demo: a 3D battlefield (pirates vs undead, animated glTF models) where eight troop types each map onto a different index query (k-NN targeting, first-hit vs all-hits `raycast`, sphere-cull AoE, friendly-k-NN heal), the parallel decide→serial-apply AI loop, boids, smoke/projectiles, and the `siege_wgpu` GPU-skinning twin. |
| [MAP_DESIGN.md](MAP_DESIGN.md) | Research notes for the voxel battlefield: blocky greedy-meshed voxels + baked vertex AO + height/slope colour ramps (look), and domain-warp + particle-erosion rivers with bridges at choke-points (design). |

## Quality & correctness

| Document | Covers |
| --- | --- |
| [DEFECTS_FOUND.md](DEFECTS_FOUND.md) | Living log of every bug the test suite has caught, with reproducer, cause, fix, and the test that pins each one against regressions. Currently four entries (D-1 to D-4). |

## Regression infrastructure (at a glance)

| Test | What it does | Source |
| --- | --- | --- |
| Unit tests | Single-operation contracts (`Tree`, `TemplateGrid`, `TemplateBank`, `Polygon`, `Rect`) | `crates/*/src/**/*.rs` |
| Boundary regressions | Permanent reproducers for the geometric configurations that have ever been broken | `crates/vectorial-hash-templates/tests/boundary_regressions.rs` |
| Snapshot fingerprint | Byte-for-byte diff of a deterministic template set against a versioned fixture | `crates/vectorial-hash-templates/tests/fingerprint_regression.rs` |
| Cell-by-cell verification | Every template that ever differed from a previous reference, classified against pure-math ground truth | `crates/vectorial-hash-templates/tests/verify_88_ray_fix_templates.rs` |
| Exhaustive culling campaign | Property/fuzz over random churned trees × random figures, every cull config equality-gated against brute force | `crates/vectorial-hash-templates/tests/exhaustive_culling.rs` |

```bash
cargo test --workspace                                                       # everything in the default set
cargo test -p vectorial-hash-templates --release --test exhaustive_culling \  # long campaign (~60 s)
    -- --ignored
cargo test -p vectorial-hash --features neighbors                            # opt-in ropes path
```

## Roadmap & open ideas

The main roadmap lives in
[`crates/vectorial-hash/README.md`](../crates/vectorial-hash/README.md#roadmap)
(implementation-side ideas, ordered by impact). The templates-side notes
live in
[`crates/vectorial-hash-templates/README.md`](../crates/vectorial-hash-templates/README.md#pending-design-notes)
(generation-side ideas). Highlights: items with area/volume (full
`Areal` and the Minkowski-flavoured "index dilation" alternative),
parametric circle templates, partial-symmetry templates beyond the 8-way
dedup, bit-shift paths for power-of-two worlds, and specific-case
validation (donut, long line, wide corridor).
