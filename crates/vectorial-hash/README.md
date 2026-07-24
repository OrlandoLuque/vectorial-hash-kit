# vectorial-hash

Core index and hash algorithms for vectorial spaces, designed with modern CPU architecture in mind.

Part of the [`vectorial-hash-kit`](https://github.com/OrlandoLuque/vectorial-hash-kit) workspace.

**New here?** Seven structures (2D + 3D) answer the same queries — see
[`docs/CHOOSING.md`](../../docs/CHOOSING.md) for a one-glance flowchart on which
to pick, [`docs/THREE_D.md`](../../docs/THREE_D.md) for the quantitative decision
map, [`docs/RAYCAST.md`](../../docs/RAYCAST.md) for the ray-cast surface,
[`docs/PARALLEL.md`](../../docs/PARALLEL.md) for when threads pay, and
[`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) for how to make the chosen
structure fast (build flags, keep-index, threads, GPU), and
[`docs/GPU.md`](../../docs/GPU.md) for the GPU-compute strand (demos, benches, and
the measured verdict on when the GPU wins vs a parallel CPU path).

## Status

Mature. **Seven structures** share one arena layout and query surface: `Tree`
(2D binary), `QuadTree` (2D 4-way), `IntegerTree` (2D, `i32` / power-of-two),
`Tree3` (3D binary), `Octree3` (3D 8-way), the pointer-free Z-order grids
`MortonGrid` (2D) and `MortonGrid3` (3D), and the adaptive pointer-free
`LinearOctree3` (3D). Each supports template-driven `cull`,
dynamic `insert`/`remove`/`update` (merge-up rule), an **O(1) stable `ItemRef`**
relocation handle (`insert_ref`/`update_ref`/`remove_ref`), **k-NN**, **ray-cast**
(thick capsule + DDA leaf-walk, `raycast`/`raycast_dda`/`raycast_first`), and
dependency-free `serialize`/`deserialize` of the built index. Query volumes: 2D
`Shape` (`Circle`, boxes, polygons) and 3D `Shape3` (`Sphere3`, `Polyhedron3`
convex half-spaces incl. `from_corners` frustum + `segment_hit` line-of-sight,
`Segment3` capsule). Parallel batch cull (`cull_many_par`), parallel `bulk_load`,
and a self-tuning structure [`advisor`] round it out. `Tree::visit_leaves`
exposes the live regions (used by the visual demos).

## What it does

A binary-split spatial tree where items live in leaf cells. When a cell exceeds `item_limit`, it splits:

- Rectangles split along the long axis so children are closer to square.
- Squares pick the axis that distributes their items most evenly.

Queries (`Tree::cull`) walk the tree against a [`Shape`]. Each node's bbox is classified as **green** (fully inside the shape — take every item without per-point checks), **white** (fully outside — skip the subtree) or **yellow** (recurse). Two template mechanisms drive this, tried in order:

1. **Per-cell-size selection** (`Shape::template_for_cell`, the paper's scheme): the shape resolves, for each tree-cell size, the precomputed template whose generation offset matches the figure's real position within the global virtual grid of that size. Template cells align 1:1 with same-size tree cells, so a node classifies with one direct cell read; `cull` resolves each size once per execution and caches it. The figure is never moved — the matching template is selected.
2. **Single fixed grid** (`Shape::template_grid`): one grid covering the whole shape, classified per node via `classify_region`.

Leaf items can additionally be answered by a 1×1 raster (`Shape::point_template`): `In`/`Out` pixels skip geometry entirely and only boundary (`Maybe`) pixels run the exact `contains_point`, after a bounding-box pre-filter. Without any template, the path falls back to bbox-intersect + per-point check.

## Public surface

| Module | Type | Purpose |
| --- | --- | --- |
| `geom` | `Point`, `Rect` | 2D primitives (half-open `Rect`). |
| `template` | `CellState`, `TemplateGrid` | Runtime cull template: classify a region as In/Out/Maybe; `translated` re-anchors a template anywhere. |
| `tree` | `Tree<T>`, `Node<T>`, `NodeId`, `Positioned` | Arena-backed binary-split tree (`insert`, `remove`, `update`, `locate`, `cull`, `visit_leaves`). `Tree::with_limits` sets a separate merge-up threshold (`merge_limit <= item_limit`) for split/merge hysteresis. |
| `culling` | `Shape`, `Tree::cull`, `Tree::cull_walk`, `WalkNeighbors` | Query items inside a shape, with optional template short-circuit; `cull_walk` traverses by flood fill over leaf neighbours instead of descending. |
| `quadtree` | `QuadTree<T>`, `QNode<T>`, `QNodeId` | Reference 4-way structure with the tree's full dynamic contract (`insert`, `remove`, `update`, 4-way merge rule, `cull` through the same template machinery) for head-to-head comparisons. |
| `itree` | `IntegerTree<T>` | 2D binary tree on `i32` coordinates with a power-of-two root extent (bit-shift `locate`); converts IRect↔Rect at the `Shape` boundary so all the float template machinery works unchanged. |
| `morton` | `MortonGrid<T>` | 2D pointer-free Z-order (linear-quadtree) hash grid: quantise → interleave bits → bucket by code. Fastest index on uniform data; `raycast` too. |
| `tree3` | `Tree3<T>`, `Node3`, `Node3Id`, `Positioned3`, `Point3`, `Aabb` | 3D binary-split tree. `insert`/`remove`/`update`(LCA)/`cull`/`knn`/`raycast`(+DDA)/`serialize` + the `ItemRef` handle path (`insert_ref`/`update_ref`/`update_ref_tracked`→`Crossing`/`remove_ref`), `bulk_load`(+`_par`). |
| `tree3` (shapes) | `Shape3`, `Sphere3`, `Polyhedron3`, `Segment3`, `VoxelRaster` | 3D query volumes. `Polyhedron3` = convex half-spaces (`from_corners` builds a frustum; `segment_hit` is the exact line-of-sight/occlusion test); `Segment3` = capsule behind `raycast`. |
| `octree3` | `Octree3<T>`, `ONode`, `ONodeId` | 3D 8-way (2×2×2) split — the 3D `QuadTree`. Same dynamic contract + `knn` + DDA `raycast` + `bulk_load` + `serialize`. |
| `morton3` | `MortonGrid3<T>` | 3D Z-order linear octree: fastest on uniform data, cheapest build, `knn` (ring shell) + `raycast`, `extend_par` bulk build. |
| `linear_octree3` | `LinearOctree3<T>` | 3D **adaptive** linear octree: sparse hash of leaf buckets keyed by a self-describing Morton location code (path + level in one `u64`), pointer-free; a leaf splits into 8 only where points cluster. `from_items`/`insert`/`cull`/`knn`/`serialize`. Grid-cheap builds (~2× faster than `Octree3`) + ~5× faster clustered k-NN than the uniform `MortonGrid3` — the pick for skewed data you rebuild often (measured, `docs/THREE_D.md`). |
| `advisor` | `SpatialProfile`, `StructureHint` | Self-tuning structure selection: track a region's relocation rate + query:move ratio (EMA); `recommend()` returns `BruteForce`/`KeepIndexTree`/`CoarserOrRebuild` from measured crossovers. |

Beyond `cull`, every structure answers **`knn`** (k-nearest, best-first with bbox
pruning), **`raycast`** (thick capsule + a front-to-back DDA leaf-walk with
`raycast_first` early-exit — see [`docs/RAYCAST.md`](../../docs/RAYCAST.md)), and
carries a stable **`ItemRef`** so a moving item relocates in O(1) (no locate walk,
no predicate scan) — the highest-leverage win for per-frame relocation workloads.

## Features

| Feature | Effect |
| --- | --- |
| `neighbors` | Per-leaf stored neighbour lists ("ropes", `Side`-indexed), rewired on every split/merge, plus `Tree::neighbors_ropes` and `WalkNeighbors::Ropes`. Off by default — none of the bookkeeping exists in the compiled code without it. The zero-storage neighbour finders (`neighbors_samet`, `neighbors_probe`) are always available. |
| `parallel` | rayon-backed batch cull `cull_many_par` on every structure (the serial `cull_many` is always available). Reads fan out over a thread pool; writes stay serial. Off by default — rayon is not in the dependency tree (nor the wasm build) without it. Measured crossover for when threads pay vs. lose: `docs/PARALLEL.md`. |

## Example

```rust
use vectorial_hash::{Point, Rect, Tree, Positioned, Shape};

#[derive(Clone, Copy)]
struct Pt(Point);
impl Positioned for Pt {
    fn position(&self) -> Point { self.0 }
}

struct Box2 { rect: Rect }
impl Shape for Box2 {
    fn bounding_box(&self) -> Rect { self.rect }
    fn contains_point(&self, p: Point) -> bool { self.rect.contains(p) }
}

let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
tree.insert(Pt(Point::new(10.0, 10.0)));
tree.insert(Pt(Point::new(50.0, 50.0)));
tree.insert(Pt(Point::new(90.0, 90.0)));

let hits = tree.cull(&Box2 { rect: Rect::new(0.0, 0.0, 60.0, 60.0) });
assert_eq!(hits.len(), 2);
```

## Design notes

- **Arena storage**: nodes live in `Vec<Node<T>>`, referenced by `NodeId(u32)`. Cache-friendly and side-steps the `Rc<RefCell<>>` graph dance that parent pointers would otherwise demand.
- **No I/O, no storage backends**: this crate stays dependency-light. Template generation (incl. Redis coordination) lives in [`vectorial-hash-templates`](../vectorial-hash-templates).
- **Items as points** (for now): the PDF outlines extending to area/volume items by adjusting insert+cull. That's deferred.

## Validation

The cull is validated by an exhaustive property/fuzz campaign
(`vectorial-hash-templates/tests/exhaustive_culling.rs`): deterministic seeds
generate churned trees × random figures, scales, angles and integer origins,
and every configuration (plain bbox, single grid, per-size bank ± raster, and
all `cull_walk` strategies) must return exactly the brute-force item set.
**Exactness contract**: results are exact for items farther than ~2×EPSILON
(1e-5) from the figure boundary; items inside that epsilon halo may classify
either way (the exact-geometry templates vs the epsilon-tolerant on-edge
rule). The campaign found and fixed two real defects: unbounded subdivision
when items share one position (now a soft limit), and unstable point-in-
polygon answers when the casting ray grazed a vertex (now a safe-ray pick).
Run the long campaign with
`cargo test -p vectorial-hash-templates --release --test exhaustive_culling -- --ignored`
(add `--features neighbors` for the ropes strategy).

## Honest limitations (measured)

This kit reports negative results too — the conclusions come from benchmarks, not
intuition. In short:

- **No single structure wins.** The binary tree is rarely the outright best: the
  8-way `Octree3` culls ~14–19% faster, and the pointer-free `MortonGrid3` wins on
  uniform data and on rebuild-per-frame. Choose per workload — see
  [`docs/THREE_D.md`](../../docs/THREE_D.md), or let the `advisor` pick from
  measured local rates.
- **Templates and the 1×1×1 raster are conditional.** For cheap analytic shapes
  (spheres, circles) a direct test beats a lookup — a memory load can't outrun one
  distance compare (the memory wall). The template/raster path pays only for an
  *expensive* `contains_point` (a many-faced/non-analytic figure; crossover ~24–48
  faces of a `Polyhedron3`). Don't raster a sphere.
- **The keep-index (`ItemRef`) is the real lever — and a well-known pattern.** The
  arena + stable-handle relocation isn't novel (slot maps / ECS storage / in-place
  broad-phase updates are standard); the useful finding is that it *dominates*
  moving-point maintenance here (5–11×).
- **Some precomputation measured slower than recompute** — a boids force table and
  a moving-data GPU offload both lost to the simple path (the memory wall again);
  see [`docs/PERF_NOTES.md`](../../docs/PERF_NOTES.md). The full self-critique lives
  in the research study's `LESSONS_LEARNED`.

## Roadmap

Ordered roughly by impact. The first block was harvested from the original design notes (the unpublished draft of the publication) and is the next round of work.

- **Items with area or volume**: an item may live in more than one leaf when its extent overlaps several cells. Two complementary approaches:
  - **Index dilation (Minkowski-flavoured)**: when the agent has a known fixed extent (e.g. a radius), keep items as points but generate the **figure's template inflated by the agent extent** — a hit on the inflated raster means the agent's centre is at most `r` away from the real figure. The runtime data structure stays untouched; only the bank gains an inflated set per (shape, agent radius). Cheap and orthogonal to everything that already exists. WIP exists in `vectorial-hash-templates/src/polygon.rs::inflated_convex` (convex-only Minkowski offset: edges shift by `r`, arcs grow `R→R+r`, vertices gain a joining arc) — one failing test at a sharp convex corner (the drop's tail tip) is the known gap. **Open design choice for the runtime narrowphase** (the per-point test in `Maybe` cells): test against the *built inflated polygon* (`inflated.is_inside(p)`, reuses existing geometry) vs. *distance-to-original* (`dist_to_boundary(orig, p) <= r`, no polygon construction). Likely: distance for circles/spheres (trivial), inflated polygon for many-edged shapes. Dilation is also the model for *every* extent-aware query, not just attack: inflate the vision circle by the prey radius; agent-agent collision is `disk(r1) ⊕ disk(r2) = disk(r1+r2)`; the fully general extent-vs-extent test is "origin ∈ A ⊕ (−B)".
  - **Full extent on the index** (paper's variant): generalize `Positioned` to an `Areal` returning a `Rect` / `Aabb`, distribute across children on split, count once per item under the merge-up rule, and deduplicate in `cull`. More general (mixed agent extents, arbitrary overlap) but a larger refactor; needed for collisions among differently-shaped extended objects.
- **Figure↔grid scale equivalence** (implemented): `PlacedTemplate::with_scale` and `TemplateBank::placed_for_scaled` reuse one canonical set across query scales via a uniform multiplier — no extra precomputation, no cell-data clones. See `vh bench-scale` for the runtime trade-off (excellent for low factors; cull slows at high factors).
- **Parametric circle templates**: instead of `In/Out/Maybe` per cell, store, per (offset, cell), the minimum radius that makes the cell `Maybe` and the minimum that makes it `In`. One parametric grid then answers every radius — exact, no aggregation. Generalizable to any one-parameter family (squares, regular polygons, …).
- **Partial-symmetry templates beyond 8 ops**: a quarter circle stamped 4 times reproduces the full circle; half a drop reproduces the whole drop. Goes further than the current 8-way dedup by exploiting the figure's own symmetries to halve/quarter the precomputed payload.
- **Bit-shifts for power-of-two worlds** (implemented as `IntegerTree<T>` in `src/itree.rs`): mirror of the binary-split tree with `i32` coordinates and a power-of-two root extent asserted at construction, including a `cull` method that converts IRect↔Rect at the shape boundary so the existing float `Shape` machinery (templates, raster, contains_point) works unchanged. The split policy matches `Tree<T>` exactly. Empirical: **conditional win** — ~22% faster on `move+update` only when items store positions natively as integers (synthetic bench). In a workload where items need both float and integer positions, the duplicated state cost reverses the result (~23% slower on update, ~3–5% slower in total throughput vs the float tree). See `docs/UPDATE_STRATEGIES.md` for the head-to-head numbers and the recommended scenarios.
- **`update` with ascend-to-LCA** (implemented as `UpdateStrategy::Lca` and `LcaRopes` in `tree.rs`): default since 2026-06. Up to ~10% faster than the legacy remove+insert path on update-heavy workloads, and consistently ~30% lower arena footprint. The `LcaRopes` variant (enabled with the `neighbors` feature) adds a 0.5–4.5% margin on top. Empirical 135-cell sweep + analysis in `docs/UPDATE_STRATEGIES.md`.
- **Stable `ItemRef`** (on **all five trees** — `Tree`, `QuadTree`, `IntegerTree`, `Tree3`, `Octree3`): `insert_ref` returns an `ItemRef` handle that stays valid as the item moves between leaves (a parallel per-leaf handle vector + a handle→location slot-map, maintained through splits/merges); `update_ref`/`remove_ref` then reach the item in **O(1)** — no locate walk, no predicate scan. The 3D decision-map sweep showed the predicate `update`'s O(item_limit) leaf scan is what made the trees lose the per-frame relocate race to a flat grid rebuild; the handle path removes it (~5–11× faster maintain, 10× at item_limit 64) and flips the winner back to the trees (`docs/THREE_D.md` § "The fix: Stable ItemRef"). Default `update`/`remove` keep the predicate API for callers without a handle (the Lca path preserves the handle; the Legacy strategy reassigns it). Churn-tested per structure. (`MortonGrid3` doesn't need it — it re-buckets, so there's no scan to skip.)
- **Range-to-value hash for float `locate`**: the integer / power-of-two path uses shifts; for arbitrary float worlds, precompute a per-axis range→cell-id table so `locate` becomes a hash lookup instead of a descent with division at each level. Trade-off vs. the current descent depends on tree depth and would need measuring.
- **Specific-case validation**: add explicit scenarios to the exhaustive campaign for query shapes the design notes call out — a donut (annulus, hole-in-middle figures), a long thin line / wide corridor (route-like GPS queries), and irregular concave polygons — so the contract holds for the harder geometries too.

Continuing work:

- **Validation contract refinement**: the cull is exact beyond a ~2×EPSILON halo around the figure boundary (1e-5 in the intersector); inside the halo, items may classify either way. Tightening this either with exact-arithmetic predicates or with consistent epsilon-tolerant templates would close the last source of discrepancy.
- **Aggregation cache across culls**: today the fallback rebuilds the aggregated template once per cull. Memoizing it on the bank (keyed by figure / angle / origin / target size) would close the gap to the fully precomputed variant when memory matters more than precompute.
- **Arena free-list** (implemented for `Tree`, `QuadTree`, `IntegerTree`, `Tree3`): merge-ups return the orphaned child slots to a free-list that the next split reuses, so the arena `Vec` stabilises at the high-water-mark of live nodes instead of growing without bound under churn (e.g. the 3D critters arena dropped from ~99k to ~7.3k nodes for ~3.6k leaves). `node_count()` now means arena capacity; `live_node_count()` is the reachable count. Freed `NodeId`s are NOT stable — a later `alloc` may reuse one (relevant only if the **stable `ItemRef`** item below is built on raw ids).
- **Stable `ItemRef`** so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- **3D variant** (prototyped — see `src/tree3.rs` and `docs/THREE_D.md`). Both strategies were built and measured:
  - **True 3D tree** (`Tree3`, binary 3D split): geometry → `Point3`/`Aabb`, split the longest of 3 axes, sphere classified analytically per node box (green/white/yellow), 1×1×1 `VoxelRaster` at leaves. **14–16× faster than brute force** at realistic query sizes; has `insert`/`locate`/`update` (LCA) / `remove` / `cull` / `knn` / `serialize` / **`insert_ref`+`update_ref`+`remove_ref`** (stable `ItemRef` handle), validated by a 6000-op churn test. The **`ItemRef`** is the resolution of the decision-map finding: a stable O(1) handle (valid as the item moves leaves) that skips `update`'s O(item_limit) predicate scan — in a per-frame relocate-everything workload it makes maintain **~5–11× faster** and flips the decision-map maintain winner from the Morton grid back to the binary tree (15/16 configs). See `docs/THREE_D.md` § Synthesis. `knn` (k-nearest-neighbour, best-first with bbox pruning and a bounded max-heap — on both `Tree3` and `Octree3`) answers "the k closest to q" in 2–13 µs/query at 50k for k=1..50, brute-force gated. `serialize`/`deserialize` round-trip the **built** tree (exact arena + free-list, no rebuild on load) to any `std::io::Write`/`Read` — dependency-free, with items written by a caller closure so it works for any `T`; a round-trip test preserves cull + knn + arena exactly and rejects corrupt input. (`Octree3` and the adaptive `LinearOctree3` carry the same round-trip; only the 2D trees are not wired yet.) For a non-analytic shape it would need an N³ voxel template (the memory wall) — the parametric-voxel and sub-block-dedup ideas become load-bearing there.
  - **Projection indexing (three 2D trees)** — *author's idea*: index items on the (x,y), (x,z), (y,z) projections, cull each with the sphere's shadow, intersect the candidate id sets, exact 3D narrowphase on survivors. Provably a **superset** (no false negatives). Measured: the 3-way intersection has near-exact precision (**1.12× candidates**) but is **slow** (1.5×) because intersecting three dense shadow sets dominates; a **1-projection** variant (one plane + exact filter) is the sweet spot at **7×** when the 3D test is cheap. Reuses all the 2D machinery, no N³ memory. The full time/precision comparison and guidance is in `docs/THREE_D.md`.
  - **Octree** (`Octree3`, 8-way 2×2×2 split — the 3D analogue of `QuadTree`): culls 14–19% faster than the binary `Tree3` (the descent walks fewer levels), at the cost of more nodes — same direction as 2D quad-vs-binary, wider gap. Now has the same ascend-to-LCA `update` (churn-tested) as `Tree3`, so the *dynamic* octree can be compared head-to-head: its per-frame `update` is also ~5–15% faster than the binary tree's, with cull ≈ equal and identical id sets (`critters3d_headless`, which now drives all four structures). Both exact, both through the shared `Shape3` machinery. The 1×1×1 `VoxelRaster` loses to a sphere's analytic test (a memory lookup can't beat one distance compare) but wins for an expensive `contains_point` — crossover at ~24–48 faces of a `Polyhedron3`, in `docs/THREE_D.md`.
  - **Morton / Z-order grid** (`MortonGrid3`, pointer-free linear octree): quantise points to integer cells, pack each cell's bits into a 64-bit Z-order code, bucket by code in a hash; a cull visits only the cells overlapping the query bbox (green/white/yellow per cell). **On uniform data it is the fastest index** — 14.5× brute at realistic selectivity vs the binary tree's 11.4× and octree's 12.6×, with the cheapest build (no rebalancing) — because a query touches a few O(1) cell lookups instead of descending ~15 pointer levels. The catch is its single fixed resolution: on non-uniform (stacked) data the adaptive octree retakes the lead (17.4× vs 15.5×), and a query far from the cell size degrades it. Grid for uniform-ish data + known query scale; tree when density/query-size varies. Build-and-cull only (no `update`) — yet the decision-map sweep found that for a *full per-frame relocation* workload its flat re-bucket also **beats the trees on the maintain side**, because each tree `update` pays an O(item_limit) predicate scan to find the item (a stable `ItemRef` would close that gap). Full tables (cull crossover + decision map) in `docs/THREE_D.md`.
- **SIMD-friendly cell layout** where it pays off.
- **Multithreading**: the runtime today is single-threaded — `cull`, `update`, `insert`/`remove` all assume exclusive `&mut Tree<T>` access. Designs to explore once the single-thread paths are settled: locked shards (one lock per top-level partition), copy-on-write snapshots for concurrent read-only culls during a frame, or a SoA `update_many` that batches movement so a single mutator pass can apply all critter motion. Independent of bit-shift and 3D and worth its own milestone.
- Stable `ItemRef` so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- 3D variant (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- SIMD-friendly cell layout where it pays off.
