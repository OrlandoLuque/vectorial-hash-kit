//! Culling: find tree items inside a shape.

use crate::geom::{Point, Rect};
use crate::template::{CellState, TemplateGrid};
use crate::tree::{NodeId, Positioned, Tree};

/// A shape the culling algorithm can test against.
///
/// `bounding_box` and `contains_point` are required. Implementations may
/// additionally provide a [`TemplateGrid`] to enable the green/yellow/white
/// short-circuit: tree nodes whose bbox falls entirely on green cells are
/// included wholesale, white cells let us skip whole subtrees, and only
/// yellow cells fall back to per-point checks.
pub trait Shape {
    fn bounding_box(&self) -> Rect;
    fn contains_point(&self, point: Point) -> bool;

    /// Optional precomputed cull template. Default: none (bbox fallback).
    fn template_grid(&self) -> Option<&TemplateGrid> {
        None
    }
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
                for child_id in [a, b] {
                    let child_bbox = self.get(child_id).bbox;
                    match classify_child(shape, shape_bbox, &child_bbox) {
                        CellState::Out => {}
                        CellState::In => {
                            self.cull_recurse(child_id, shape, shape_bbox, true, out);
                        }
                        CellState::Maybe => {
                            self.cull_recurse(child_id, shape, shape_bbox, false, out);
                        }
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

fn classify_child<S: Shape>(shape: &S, shape_bbox: &Rect, child_bbox: &Rect) -> CellState {
    if let Some(grid) = shape.template_grid() {
        grid.classify_region(child_bbox)
    } else if child_bbox.intersects(shape_bbox) {
        CellState::Maybe
    } else {
        CellState::Out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::TemplateGrid;

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

    /// A "shape" backed only by a TemplateGrid; `contains_point` would be
    /// wrong on purpose (always returns false) so we can detect green-cell
    /// short-circuits — if any item came back, the template path included it
    /// without ever calling `contains_point`.
    struct GridShape {
        bbox: Rect,
        grid: TemplateGrid,
    }
    impl Shape for GridShape {
        fn bounding_box(&self) -> Rect { self.bbox }
        fn contains_point(&self, _p: Point) -> bool { false }
        fn template_grid(&self) -> Option<&TemplateGrid> { Some(&self.grid) }
    }

    #[test]
    fn green_template_cell_short_circuits_per_point_check() {
        use CellState::*;
        // Root 100x100 split into 2x2 cells of 50x50; only top-right cell is In.
        let grid = TemplateGrid::new(
            Point::new(0.0, 0.0),
            50.0,
            50.0,
            2,
            2,
            vec![
                Out, Out,
                Out, In,
            ],
        );
        let shape = GridShape { bbox: Rect::new(0.0, 0.0, 100.0, 100.0), grid };

        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        tree.insert(Pt(Point::new(10.0, 10.0))); // Out cell
        tree.insert(Pt(Point::new(60.0, 60.0))); // In  cell
        tree.insert(Pt(Point::new(90.0, 90.0))); // In  cell

        let hits = tree.cull(&shape);
        // Two items live in the In cell; both must be included even though
        // `contains_point` always returns false.
        let mut positions: Vec<_> = hits.iter().map(|p| p.0).collect();
        positions.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        assert_eq!(positions, vec![Point::new(60.0, 60.0), Point::new(90.0, 90.0)]);
    }

    /// Mirror of the above for white cells: a shape whose template is all Out
    /// must return an empty cull result, regardless of `contains_point`.
    #[test]
    fn white_template_skips_whole_subtree() {
        use CellState::*;
        let grid = TemplateGrid::new(
            Point::new(0.0, 0.0),
            50.0,
            50.0,
            2,
            2,
            vec![Out, Out, Out, Out],
        );
        struct AllInside { bbox: Rect, grid: TemplateGrid }
        impl Shape for AllInside {
            fn bounding_box(&self) -> Rect { self.bbox }
            fn contains_point(&self, _p: Point) -> bool { true } // would include everything
            fn template_grid(&self) -> Option<&TemplateGrid> { Some(&self.grid) }
        }
        let shape = AllInside { bbox: Rect::new(0.0, 0.0, 100.0, 100.0), grid };

        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        tree.insert(Pt(Point::new(25.0, 25.0)));
        tree.insert(Pt(Point::new(75.0, 75.0)));

        assert!(tree.cull(&shape).is_empty());
    }
}
