//! Culling: find tree items inside a shape.

use std::collections::HashMap;
use std::sync::Arc;

use crate::geom::{Point, Rect};
use crate::template::{CellState, TemplateGrid};
use crate::tree::{NodeId, Positioned, Tree};

/// A shape the culling algorithm can test against.
///
/// `bounding_box` and `contains_point` are required. Implementations may
/// additionally provide templates to enable the green/yellow/white
/// short-circuit: tree nodes whose bbox falls entirely on green cells are
/// included wholesale, white cells let us skip whole subtrees, and only
/// yellow cells fall back to per-point checks.
///
/// Two template mechanisms are supported, tried in this order:
///
/// 1. **Per-cell-size selection** ([`Shape::template_for_cell`], the paper's
///    scheme): for each tree-cell size touched by the query, the shape
///    resolves the precomputed template whose generation offset matches the
///    figure's real position within the global virtual grid of that cell
///    size. Template cells then align 1:1 with same-size tree cells, so each
///    node classifies with a single direct cell read. The figure is **never
///    moved** to fit the grid — the matching template is selected instead.
///    `cull` resolves at most one template per distinct cell size per
///    execution and caches it for the rest of that query.
/// 2. **Single fixed grid** ([`Shape::template_grid`]): one grid covering the
///    whole shape, classified per node via [`TemplateGrid::classify_region`].
pub trait Shape {
    fn bounding_box(&self) -> Rect;
    fn contains_point(&self, point: Point) -> bool;

    /// Optional precomputed cull template. Default: none (bbox fallback).
    fn template_grid(&self) -> Option<&TemplateGrid> {
        None
    }

    /// Resolve the template aligned to the global virtual grid of cells
    /// `cell_w` × `cell_h` for this shape at its current position, if one
    /// exists. Returning `None` falls back to `template_grid` / bbox.
    ///
    /// Contract: the returned grid's cells must be exactly `cell_w` ×
    /// `cell_h` and anchored on multiples of the cell size, so that aligned
    /// tree nodes map 1:1 onto template cells.
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<Arc<TemplateGrid>> {
        let _ = (cell_w, cell_h);
        None
    }

    /// Optional 1×1-cell raster of the shape used for per-item tests in
    /// leaf cells: `In`/`Out` pixels answer immediately and only `Maybe`
    /// (boundary) pixels fall back to the exact `contains_point`.
    fn point_template(&self) -> Option<&TemplateGrid> {
        None
    }
}

/// Per-execution cache: one resolved template per distinct cell size.
type SizeCache = HashMap<(u64, u64), Option<Arc<TemplateGrid>>>;

impl<T: Positioned> Tree<T> {
    /// Return references to every item inside `shape`.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bbox = shape.bounding_box();
        let mut sizes = SizeCache::new();
        self.cull_recurse(self.root, shape, &bbox, false, &mut sizes, &mut out);
        out
    }

    fn cull_recurse<'a, S: Shape>(
        &'a self,
        node_id: NodeId,
        shape: &S,
        shape_bbox: &Rect,
        fully_inside: bool,
        sizes: &mut SizeCache,
        out: &mut Vec<&'a T>,
    ) {
        let node = self.get(node_id);

        if fully_inside {
            match node.children {
                Some([a, b]) => {
                    self.cull_recurse(a, shape, shape_bbox, true, sizes, out);
                    self.cull_recurse(b, shape, shape_bbox, true, sizes, out);
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }

        match node.children {
            Some([a, b]) => {
                for child_id in [a, b] {
                    let child_bbox = self.get(child_id).bbox;
                    match classify_child(shape, shape_bbox, &child_bbox, sizes) {
                        CellState::Out => {}
                        CellState::In => {
                            self.cull_recurse(child_id, shape, shape_bbox, true, sizes, out);
                        }
                        CellState::Maybe => {
                            self.cull_recurse(child_id, shape, shape_bbox, false, sizes, out);
                        }
                    }
                }
            }
            None => {
                let point_grid = shape.point_template();
                for it in &node.items {
                    let p = it.position();
                    // Bbox pre-filter: anything outside the shape's bounding
                    // box is out, no geometry needed.
                    if !shape_bbox.contains(p) {
                        continue;
                    }
                    match point_grid.map(|g| g.cell_at_world(p)) {
                        Some(CellState::In) => out.push(it),
                        Some(CellState::Out) => {}
                        // Boundary pixel or no raster: exact test.
                        _ => {
                            if shape.contains_point(p) {
                                out.push(it);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn classify_child<S: Shape>(
    shape: &S,
    shape_bbox: &Rect,
    child_bbox: &Rect,
    sizes: &mut SizeCache,
) -> CellState {
    // 1. Per-cell-size template, resolved once per size per execution. The
    //    node bbox is exactly one cell of the global virtual grid of its own
    //    size, so its centre reads the matching template cell directly.
    let key = (child_bbox.width.to_bits(), child_bbox.height.to_bits());
    let per_size = sizes
        .entry(key)
        .or_insert_with(|| shape.template_for_cell(child_bbox.width, child_bbox.height))
        .clone();
    if let Some(grid) = per_size {
        return grid.cell_at_world(Point::new(
            child_bbox.x + child_bbox.width / 2.0,
            child_bbox.y + child_bbox.height / 2.0,
        ));
    }

    // 2. Single fixed grid (region classification), then bbox fallback.
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

    /// Per-cell-size selection: the shape serves a template aligned to each
    /// requested cell size; nodes classify via a single direct cell read.
    /// `contains_point` always false would drop everything if the green
    /// short-circuit didn't include the In-cell subtree wholesale.
    struct PerSizeShape {
        bbox: Rect,
        resolved: std::cell::RefCell<Vec<(f64, f64)>>,
    }
    impl Shape for PerSizeShape {
        fn bounding_box(&self) -> Rect {
            self.bbox
        }
        fn contains_point(&self, _p: Point) -> bool {
            false
        }
        fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<Arc<TemplateGrid>> {
            use CellState::*;
            self.resolved.borrow_mut().push((cell_w, cell_h));
            // Whatever the size, mark the cell range covering x >= 50 as In
            // within the right half of a 100x100 world.
            let cols = (50.0 / cell_w).max(1.0) as u32;
            let rows = (100.0 / cell_h).max(1.0) as u32;
            Some(Arc::new(TemplateGrid::new(
                Point::new(50.0, 0.0),
                cell_w,
                cell_h,
                cols,
                rows,
                vec![In; (cols * rows) as usize],
            )))
        }
    }

    #[test]
    fn per_size_template_short_circuits_and_caches_per_size() {
        let shape = PerSizeShape {
            bbox: Rect::new(50.0, 0.0, 50.0, 100.0),
            resolved: std::cell::RefCell::new(Vec::new()),
        };
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        tree.insert(Pt(Point::new(10.0, 10.0))); // left half: Out
        tree.insert(Pt(Point::new(60.0, 60.0))); // right half: In
        tree.insert(Pt(Point::new(90.0, 90.0))); // right half: In

        let hits = tree.cull(&shape);
        let mut positions: Vec<_> = hits.iter().map(|p| p.0).collect();
        positions.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        assert_eq!(positions, vec![Point::new(60.0, 60.0), Point::new(90.0, 90.0)]);

        // The per-execution cache must resolve each distinct size only once.
        let resolved = shape.resolved.borrow();
        let mut unique = resolved.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        assert_eq!(resolved.len(), unique.len(), "sizes resolved more than once: {resolved:?}");
    }

    /// Leaf fallback uses the 1x1 raster: In pixels accepted without exact
    /// tests, Out pixels rejected even if contains_point says otherwise,
    /// Maybe pixels defer to contains_point.
    struct RasterShape {
        bbox: Rect,
        raster: TemplateGrid,
    }
    impl Shape for RasterShape {
        fn bounding_box(&self) -> Rect {
            self.bbox
        }
        fn contains_point(&self, p: Point) -> bool {
            p.y < 2.0 // only used for Maybe pixels
        }
        fn point_template(&self) -> Option<&TemplateGrid> {
            Some(&self.raster)
        }
    }

    #[test]
    fn point_template_resolves_leaf_items_with_exact_test_only_on_maybe() {
        use CellState::*;
        // 3x1 raster of 1x1 cells at x = 0..3: In, Maybe, Out.
        let raster = TemplateGrid::new(
            Point::new(0.0, 0.0),
            1.0,
            1.0,
            3,
            3,
            vec![
                In, Maybe, Out,
                In, Maybe, Out,
                In, Maybe, Out,
            ],
        );
        let shape = RasterShape { bbox: Rect::new(0.0, 0.0, 3.0, 3.0), raster };
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 10.0, 10.0), 16);
        tree.insert(Pt(Point::new(0.5, 0.5))); // In pixel -> hit
        tree.insert(Pt(Point::new(1.5, 0.5))); // Maybe pixel, y < 2 -> exact says yes
        tree.insert(Pt(Point::new(1.5, 2.5))); // Maybe pixel, y >= 2 -> exact says no
        tree.insert(Pt(Point::new(2.5, 0.5))); // Out pixel -> miss (contains_point not consulted)
        tree.insert(Pt(Point::new(8.0, 8.0))); // outside bbox -> pre-filtered

        let mut positions: Vec<_> = tree.cull(&shape).iter().map(|p| p.0).collect();
        positions.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        assert_eq!(positions, vec![Point::new(0.5, 0.5), Point::new(1.5, 0.5)]);
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
