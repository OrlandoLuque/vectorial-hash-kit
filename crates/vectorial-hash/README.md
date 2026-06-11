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
  - **Index dilation (Minkowski-flavoured)**: when the agent has a known fixed extent (e.g. a radius), keep items as points but generate the **figure's template inflated by the agent extent** — a hit on the inflated raster means the agent's centre is at most `r` away from the real figure. The runtime data structure stays untouched; only the bank gains an inflated set per (shape, agent radius). Cheap and orthogonal to everything that already exists.
  - **Full extent on the index** (paper's variant): generalize `Positioned` to an `Areal` returning a `Rect` / `Aabb`, distribute across children on split, count once per item under the merge-up rule, and deduplicate in `cull`. More general (mixed agent extents, arbitrary overlap) but a larger refactor; needed for collisions among differently-shaped extended objects.
- **Figure↔grid scale equivalence** (implemented): `PlacedTemplate::with_scale` and `TemplateBank::placed_for_scaled` reuse one canonical set across query scales via a uniform multiplier — no extra precomputation, no cell-data clones. See `vh bench-scale` for the runtime trade-off (excellent for low factors; cull slows at high factors).
- **Parametric circle templates**: instead of `In/Out/Maybe` per cell, store, per (offset, cell), the minimum radius that makes the cell `Maybe` and the minimum that makes it `In`. One parametric grid then answers every radius — exact, no aggregation. Generalizable to any one-parameter family (squares, regular polygons, …).
- **Partial-symmetry templates beyond 8 ops**: a quarter circle stamped 4 times reproduces the full circle; half a drop reproduces the whole drop. Goes further than the current 8-way dedup by exploiting the figure's own symmetries to halve/quarter the precomputed payload.
- **Bit-shifts for power-of-two worlds**: when the root extent is a power of two and splits are 1:1 binary, cell locating and divisions can use shifts instead of float arithmetic. The `Tree<T>` API stays the same; a typed `IntegerTree` (or a const generic) selects the shift-based path at compile time.
- **Specific-case validation**: add explicit scenarios to the exhaustive campaign for query shapes the design notes call out — a donut (annulus, hole-in-middle figures), a long thin line / wide corridor (route-like GPS queries), and irregular concave polygons — so the contract holds for the harder geometries too.

Continuing work:

- **Validation contract refinement**: the cull is exact beyond a ~2×EPSILON halo around the figure boundary (1e-5 in the intersector); inside the halo, items may classify either way. Tightening this either with exact-arithmetic predicates or with consistent epsilon-tolerant templates would close the last source of discrepancy.
- **Aggregation cache across culls**: today the fallback rebuilds the aggregated template once per cull. Memoizing it on the bank (keyed by figure / angle / origin / target size) would close the gap to the fully precomputed variant when memory matters more than precompute.
- **Arena free-list** so `remove` reclaims orphaned nodes (today they stay as zombies; `NodeId`s are stable but `node_count()` overstates live nodes).
- **Stable `ItemRef`** so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- **3D variant** (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- **SIMD-friendly cell layout** where it pays off.
- Stable `ItemRef` so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- 3D variant (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- SIMD-friendly cell layout where it pays off.
