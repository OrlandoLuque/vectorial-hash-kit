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
| `culling` | `Shape`, `Tree::cull` | Query items inside a shape, with optional template short-circuit. |

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

## Roadmap

- Arena free-list so `remove` reclaims orphaned nodes (today they stay as zombies; `NodeId`s are stable but `node_count()` overstates live nodes).
- Stable `ItemRef` so callers can hold onto items across mutations without re-locating them by point + predicate every time.
- 3D variant (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- SIMD-friendly cell layout where it pays off.
