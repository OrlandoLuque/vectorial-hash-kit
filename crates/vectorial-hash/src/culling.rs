//! Culling: find tree items inside a shape.

use crate::geom::{Point, Rect};
use crate::tree::{NodeId, Positioned, Tree};

/// A shape the culling algorithm can test against.
///
/// Future versions will accept a template set to short-circuit per-item checks
/// for cells where the shape is fully known to cover everything; for now this
/// trait describes only what the bbox-fallback path needs.
pub trait Shape {
    fn bounding_box(&self) -> Rect;
    fn contains_point(&self, point: Point) -> bool;
}

impl<T: Positioned> Tree<T> {
    /// Return references to every item inside `shape`.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bbox = shape.bounding_box();
        self.cull_recurse(self.root, shape, &bbox, false, &mut out);
        out
    }

    fn cull_recurse<'a, S: Shape>(
        &'a self,
        node_id: NodeId,
        shape: &S,
        shape_bbox: &Rect,
        fully_inside: bool,
        out: &mut Vec<&'a T>,
    ) {
        let node = self.get(node_id);

        if fully_inside {
            match node.children {
                Some([a, b]) => {
                    self.cull_recurse(a, shape, shape_bbox, true, out);
                    self.cull_recurse(b, shape, shape_bbox, true, out);
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }

        match node.children {
            Some([a, b]) => {
                // No template lookup yet: use the bounding box as the conservative filter.
                for child_id in [a, b] {
                    if self.get(child_id).bbox.intersects(shape_bbox) {
                        self.cull_recurse(child_id, shape, shape_bbox, false, out);
                    }
                }
            }
            None => {
                for it in &node.items {
                    if shape.contains_point(it.position()) {
                        out.push(it);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Pt(Point);
    impl Positioned for Pt {
        fn position(&self) -> Point { self.0 }
    }

    struct Circle { center: Point, radius: f64 }
    impl Shape for Circle {
        fn bounding_box(&self) -> Rect {
            Rect::new(
                self.center.x - self.radius,
                self.center.y - self.radius,
                self.radius * 2.0,
                self.radius * 2.0,
            )
        }
        fn contains_point(&self, p: Point) -> bool {
            let dx = p.x - self.center.x;
            let dy = p.y - self.center.y;
            dx * dx + dy * dy <= self.radius * self.radius
        }
    }

    #[test]
    fn cull_collects_only_points_inside_the_circle() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
        for &x in &[10.0_f64, 50.0, 90.0] {
            for &y in &[10.0_f64, 50.0, 90.0] {
                tree.insert(Pt(Point::new(x, y)));
            }
        }
        let circle = Circle { center: Point::new(50.0, 50.0), radius: 20.0 };
        let hit = tree.cull(&circle);
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].0, Point::new(50.0, 50.0));
    }

    #[test]
    fn cull_returns_empty_when_shape_outside_root() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(50.0, 50.0)));
        let circle = Circle { center: Point::new(500.0, 500.0), radius: 10.0 };
        assert!(tree.cull(&circle).is_empty());
    }
}
