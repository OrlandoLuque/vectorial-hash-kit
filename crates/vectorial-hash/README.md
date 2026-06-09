# vectorial-hash

Core index and hash algorithms for vectorial spaces, designed with modern CPU architecture in mind.

Part of the [`vectorial-hash-kit`](https://github.com/<your-user>/vectorial-hash-kit) workspace.

## Status

First port of the runtime tree from the [`Multidimensional vector index`](../../Multidimensional%20vector%20index.pdf) paper landed. The bbox-fallback culling path is wired up; template-driven culling will follow in the next iteration alongside `remove` / `move` and 3D support.

## What it does

A binary-split spatial tree where items live in leaf cells. When a cell exceeds `item_limit`, it splits:

- Rectangles split along the long axis so children are closer to square.
- Squares pick the axis that distributes their items most evenly.

Queries (`Tree::cull`) walk the tree against a [`Shape`]: cells whose bbox doesn't touch the shape are pruned; leaf items get a final per-point check. A future template-aware path will short-circuit the per-point step for cells fully covered by the shape.

## Public surface

| Module | Type | Purpose |
| --- | --- | --- |
| `geom` | `Point`, `Rect` | 2D primitives (half-open `Rect`). |
| `tree` | `Tree<T>`, `Node<T>`, `NodeId`, `Positioned` | Arena-backed binary-split tree. |
| `culling` | `Shape`, `Tree::cull` | Query items inside a shape. |

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

- `Tree::remove` + `Tree::move_item` with the merge-up rule from the paper.
- Optional template lookup in `cull` to recover the green/yellow/white short-circuit.
- 3D variant (probably feature-gated or via a generic dimension parameter once the 2D shape settles).
- SIMD-friendly cell layout where it pays off.
