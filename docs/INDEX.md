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
| [`bench-runner`](../crates/bench-runner/README.md) | Runs every benchmark in the repo on an idle machine and writes result tables | `--group core/query/gpu/sim/demos/cli/cold/gate/all`, `--repeat N`, `--only <substr>`; Markdown + CSV under `bench-results/` |
| [`vectorial-hash-demos`](../crates/vectorial-hash-demos/README.md) | Runnable demos (visual + console + headless) | Console end-to-end pipeline, the live macroquad critters demo (binary / quadtree / dual mode), and `critters_headless` — the same deterministic sim core without a window, reporting per-operation statistics |

## Empirical results & methodology

**How the numbers are taken — and what that got wrong first:**
[MEASURING.md](MEASURING.md). Every rule in it replaced an answer that looked right,
including three published figures that did not survive being re-measured.

**Reproducing any number below:** `cargo run -p bench-runner --release -- --group all`
(or a single group; `--repeat 3` for anything you intend to quote). It waits for the
machine to be idle, repeats, and reports the spread — see
[the runner's README](../crates/bench-runner/README.md) for the three published figures
that did **not** survive being re-measured that way.

| Document | Covers |
| --- | --- |
| [BENCHMARKS.md](BENCHMARKS.md) | Headline conclusions for the `vh bench-*` culling benchmarks (single-template ~4×, per-cell-size 12–19×, descent vs walk, fallback, scale equivalence). *Full methodology + reproducible tables in the [research repo](https://github.com/OrlandoLuque/vectorialHash/blob/master/research/BENCHMARKS.md).* |
| [UPDATE_STRATEGIES.md](UPDATE_STRATEGIES.md) | Verdicts on the `Tree::update` relocation strategies (Legacy / Lca / LcaRopes) + the IntegerTree bit-shift experiment that set the API defaults. *Full 135-cell sweep + analysis in the [research repo](https://github.com/OrlandoLuque/vectorialHash/blob/master/research/UPDATE_STRATEGIES.md).* |
| [HORDE.md](HORDE.md) | The horde demo (`horde_wgpu`): TAB-style assault where noise-wake cascades, tower targeting, breaches and the infection chain are all index queries — and **100k dormant zombies cost 0.42 ms/step to keep indexed** (the keep-index headline at scale). Design/research in [HORDE_DESIGN.md](HORDE_DESIGN.md). |
| [FORMATIONS.md](FORMATIONS.md) | The formations demo (`formations_wgpu`): Total War-style automatic battle — melee k-NN pairing, flank/rear **sector classify** → morale, cavalry charges armed over a **raycast** corridor, k-NN(1) arrow landings with honest friendly fire; the regiment level maneuvers with honest brute force (an index loses at N≈60). Research in [FORMATIONS_DESIGN.md](FORMATIONS_DESIGN.md). |
| [FLUID.md](FLUID.md) | The `fluid_wgpu` demo: an interactive 2D **position-based fluid** you stir with mouse or finger — SPH is the workload an index is *for* (every particle needs its kernel neighbours every step). Head-to-head per frame, maintain / query / physics: the kept `Tree`+`ItemRef` maintains **3.5–3.9× cheaper** than either rebuild but gives it back on query; the neighbour search *is* the simulation cost. |
| [ADAPTIVE_LAB.md](ADAPTIVE_LAB.md) | The `adaptive_lab_wgpu` demo: **four sliders, one per threshold**, and a timeline strip showing which backend `AdaptiveIndex` held each step. It is the only place the layer's reason to exist is *visible* — flapping, lag and a wrong choice are one number in a total and three different pictures on a strip. Drag the radius alone and watch the policy leave the grid: same population, same query load, queries now finding 1.2 items each. |
| [POINTCLOUD.md](POINTCLOUD.md) | The `pointcloud_wgpu` demo: a large **static, strongly skewed** scanned cloud coloured by local density — i.e. *N k-NN queries per pass*. Both trees answer k-NN **~1.5× faster than the flat grid**; `KdTree3` **builds 1.7× faster than the midpoint octree** and ties it on query; the grid still wins the build outright. |
| [STEALTH.md](STEALTH.md) | The `stealth_wgpu` demo: guards whose **view cone is a real frustum cull** (`Polyhedron3`, six half-spaces) and whose **line of sight is a real segment↔solid test** (capsule broadphase → `segment_hit`). Races the index against a linear scan every frame: **the crossover is ~1000 agents** — below it a loop honestly wins, at 40k the index is 6.7×. |
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
