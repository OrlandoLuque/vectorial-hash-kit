# vectorial-hash

Core index and hash algorithms for vectorial spaces, designed with modern CPU architecture in mind.

Part of the [`vectorial-hash-kit`](https://github.com/OrlandoLuque/vectorial-hash-kit) workspace.

## Status

Runtime tree, template-driven culling, the dynamic `remove` / `update` operations (with the paper's merge-up rule), and the templates-crate adapter are wired up. `Tree::visit_leaves` exposes the live regions (used by the visual demo) and `TemplateGrid::translated` re-anchors a precomputed template at any world position. 3D support is next.

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

## Features

| Feature | Effect |
| --- | --- |
| `neighbors` | Per-leaf stored neighbour lists ("ropes", `Side`-indexed), rewired on every split/merge, plus `Tree::neighbors_ropes` and `WalkNeighbors::Ropes`. Off by default — none of the bookkeeping exists in the compiled code without it. The zero-storage neighbour finders (`neighbors_samet`, `neighbors_probe`) are always available. |

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
- **Range-to-value hash for float `locate`**: the integer / power-of-two path uses shifts; for arbitrary float worlds, precompute a per-axis range→cell-id table so `locate` becomes a hash lookup instead of a descent with division at each level. Trade-off vs. the current descent depends on tree depth and would need measuring.
- **Specific-case validation**: add explicit scenarios to the exhaustive campaign for query shapes the design notes call out — a donut (annulus, hole-in-middle figures), a long thin line / wide corridor (route-like GPS queries), and irregular concave polygons — so the contract holds for the harder geometries too.

Continuing work:

- **Validation contract refinement**: the cull is exact beyond a ~2×EPSILON halo around the figure boundary (1e-5 in the intersector); inside the halo, items may classify either way. Tightening this either with exact-arithmetic predicates or with consistent epsilon-tolerant templates would close the last source of discrepancy.
- **Aggregation cache across culls**: today the fallback rebuilds the aggregated template once per cull. Memoizing it on the bank (keyed by figure / angle / origin / target size) would close the gap to the fully precomputed variant when memory matters more than precompute.
- **Arena free-list** so `remove` reclaims orphaned nodes (today they stay as zombies; `NodeId`s are stable but `node_count()` overstates live nodes).
- **Stable `ItemRef`** so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- **3D variant**. Two candidate strategies:
  - **True 3D tree** (octree / 3D binary split): the textbook approach. Geometry → `Point3`/`Aabb`, split the longest of 3 axes, templates become voxel grids (In/Out/Maybe), the 8-fold square symmetry becomes the 48-fold cube symmetry. Exact, keeps the green short-circuit, but templates explode N²→N³ — the parametric-template and sub-block-dedup ideas become load-bearing, and it's a lot of new code.
  - **Projection indexing (two/three 2D trees)** — *author's idea, likely the better first 3D*: index items in two 2D trees, one on (x,y) and one on (x,z); cull each projection with the existing 2D machinery and intersect the candidate ID sets. Provably a **superset** of the true 3D result (no false negatives), so it's a broadphase that needs an exact 3D `contains_point` narrowphase to drop false positives (the "Steinmetz" corners where the two cylinders exceed the sphere). Reuses *all* the optimized 2D code and the 2D template bank — sidesteps the N³ memory explosion entirely. Cost: items live in 2 trees (≈2× update), and the 2D green short-circuit does NOT carry to 3D (every survivor needs the exact test). Wins when shapes are convex/blob-like and the 3D point test is cheap (our case: `dx²+dy²+dz² ≤ r²`); degrades for thin/diagonal shapes with a large bicylinder over-approximation. Refinement: a third projection (yz) recovers the lost y↔z correlation (visual-hull with 3 axis-aligned silhouettes) for ~1.5× cull cost. **Proposed first experiment**: measure the broadphase/exact ratio (false-positive rate) per shape with the 2-projection scheme *before* building any 3D tree — cheap to measure, decides the approach.
- **SIMD-friendly cell layout** where it pays off.
- **Multithreading**: the runtime today is single-threaded — `cull`, `update`, `insert`/`remove` all assume exclusive `&mut Tree<T>` access. Designs to explore once the single-thread paths are settled: locked shards (one lock per top-level partition), copy-on-write snapshots for concurrent read-only culls during a frame, or a SoA `update_many` that batches movement so a single mutator pass can apply all critter motion. Independent of bit-shift and 3D and worth its own milestone.
- Stable `ItemRef` so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- 3D variant (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- SIMD-friendly cell layout where it pays off.
