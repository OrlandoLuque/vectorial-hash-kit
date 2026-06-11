//! Cull template: precomputed grid classifying a shape's coverage cell by cell.
//!
//! Each cell of a [`TemplateGrid`] is either [`CellState::In`] (every point of
//! the cell is inside the shape — *green*), [`CellState::Out`] (no point is
//! inside — *white*) or [`CellState::Maybe`] (the cell straddles the shape's
//! boundary — *yellow*). During culling, classifying a tree node's bbox lets
//! us short-circuit:
//!
//! - **green**: take every item in the subtree without per-point checks.
//! - **white**: skip the subtree.
//! - **yellow**: recurse and fall back to per-point.

use std::sync::Arc;

use crate::geom::{Point, Rect};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CellState {
    Out,
    Maybe,
    In,
}

/// A shared template grid *placed* at a world displacement and optional
/// uniform scale factor.
///
/// Resolving a template at query time never clones cell data: the canonical
/// grid stays behind an `Arc` and classification simply translates the query
/// point by `(dx, dy)` and divides by `scale` before reading. This is what
/// [`crate::Shape::template_for_cell`] and
/// [`crate::Shape::point_template`] hand to the culling walk.
///
/// The `scale` field implements the figure↔grid scale equivalence (the
/// classifications of a template generated for figure F with cells C are
/// identical to those of F·k with cells C·k for any k > 0). One stored set
/// per shape canonical size therefore covers infinitely many query scales
/// — see [`crate::Shape::template_for_cell`] and the templates crate's
/// `TemplateBank::placed_for_scaled` for the consumer side.
#[derive(Clone, Debug)]
pub struct PlacedTemplate {
    pub grid: Arc<TemplateGrid>,
    /// World displacement added to the grid's own (scaled) origin.
    pub dx: f64,
    pub dy: f64,
    /// World cell size = `grid.cell_w * scale` (likewise for height). 1.0
    /// when the stored grid is consumed at its generation scale.
    pub scale: f64,
}

impl PlacedTemplate {
    pub fn new(grid: Arc<TemplateGrid>, dx: f64, dy: f64) -> Self {
        Self { grid, dx, dy, scale: 1.0 }
    }

    /// Place the grid at `(dx, dy)` and present its cells as `scale`×
    /// larger than their canonical size. Equivalent to using the same
    /// template for a figure scaled by `scale` with cells scaled by `scale`.
    pub fn with_scale(grid: Arc<TemplateGrid>, dx: f64, dy: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive");
        Self { grid, dx, dy, scale }
    }

    /// World-space cell width (= `grid.cell_w * scale`).
    pub fn world_cell_w(&self) -> f64 {
        self.grid.cell_w * self.scale
    }
    /// World-space cell height.
    pub fn world_cell_h(&self) -> f64 {
        self.grid.cell_h * self.scale
    }

    /// State of the cell containing world point `p`.
    pub fn cell_at_world(&self, p: Point) -> CellState {
        let local_x = (p.x - self.dx) / self.scale;
        let local_y = (p.y - self.dy) / self.scale;
        self.grid.cell_at_world(Point::new(local_x, local_y))
    }

    /// Bounding box of the placed grid, in world coordinates.
    pub fn bounding_box(&self) -> Rect {
        let b = self.grid.bounding_box();
        Rect::new(
            b.x * self.scale + self.dx,
            b.y * self.scale + self.dy,
            b.width * self.scale,
            b.height * self.scale,
        )
    }

    /// Aggregated copy with cells `fx`×`fy` times larger.
    ///
    /// The result is aligned to the **world** grid of the new cell size
    /// (origin `floor(world_origin / new_cell_size) * new_cell_size`), so a
    /// query node of that size — which always sits on the same lattice —
    /// reads exactly one aggregated cell. See [`TemplateGrid::aggregate`]
    /// for the exact (lossless) classification rule.
    pub fn aggregated(&self, fx: u32, fy: u32) -> PlacedTemplate {
        assert!(fx >= 1 && fy >= 1, "aggregate factors must be >= 1");
        let src_cw = self.world_cell_w();
        let src_ch = self.world_cell_h();
        let new_cw = src_cw * fx as f64;
        let new_ch = src_ch * fy as f64;

        // World extent of the source grid (respecting scale).
        let src_world_x = self.grid.origin_x * self.scale + self.dx;
        let src_world_y = self.grid.origin_y * self.scale + self.dy;
        let src_world_x_max = src_world_x + src_cw * self.grid.cols as f64;
        let src_world_y_max = src_world_y + src_ch * self.grid.rows as f64;

        // World origin of the aggregated grid: aligned to the global lattice.
        let new_world_x = (src_world_x / new_cw).floor() * new_cw;
        let new_world_y = (src_world_y / new_ch).floor() * new_ch;
        let new_cols =
            (((src_world_x_max - new_world_x) / new_cw).ceil() as u32).max(1);
        let new_rows =
            (((src_world_y_max - new_world_y) / new_ch).ceil() as u32).max(1);

        let mut cells = Vec::with_capacity((new_cols * new_rows) as usize);
        for j in 0..new_rows {
            for i in 0..new_cols {
                let cell_world_x = new_world_x + i as f64 * new_cw;
                let cell_world_y = new_world_y + j as f64 * new_ch;
                let mut saw_in = false;
                let mut saw_out = false;
                'outer: for sub_j in 0..fy {
                    let sub_y = cell_world_y + sub_j as f64 * src_ch;
                    for sub_i in 0..fx {
                        let sub_x = cell_world_x + sub_i as f64 * src_cw;
                        // Probe slightly inside each sub-cell for stable
                        // mapping back to (col, row) under float jitter.
                        let probe_x = sub_x + src_cw * 0.5;
                        let probe_y = sub_y + src_ch * 0.5;
                        match self.cell_at_world(Point::new(probe_x, probe_y)) {
                            CellState::In => saw_in = true,
                            CellState::Out => saw_out = true,
                            CellState::Maybe => {
                                saw_in = true;
                                saw_out = true;
                            }
                        }
                        if saw_in && saw_out {
                            break 'outer;
                        }
                    }
                }
                cells.push(match (saw_in, saw_out) {
                    (true, false) => CellState::In,
                    (false, true) => CellState::Out,
                    (true, true) => CellState::Maybe,
                    (false, false) => CellState::Out,
                });
            }
        }
        let grid = TemplateGrid {
            origin_x: 0.0,
            origin_y: 0.0,
            cell_w: new_cw,
            cell_h: new_ch,
            cols: new_cols,
            rows: new_rows,
            cells,
        };
        PlacedTemplate::new(Arc::new(grid), new_world_x, new_world_y)
    }
}

/// A regular grid covering some region; each cell carries an [`CellState`].
///
/// Cells are stored row-major: `cells[row * cols + col]`. Coordinates outside
/// the grid are treated as [`CellState::Out`] — the grid is assumed to cover
/// the shape's bounding box, so anything outside it is, by construction,
/// outside the shape.
#[derive(Clone, Debug)]
pub struct TemplateGrid {
    pub origin_x: f64,
    pub origin_y: f64,
    pub cell_w: f64,
    pub cell_h: f64,
    pub cols: u32,
    pub rows: u32,
    pub cells: Vec<CellState>,
}

impl TemplateGrid {
    pub fn new(
        origin: Point,
        cell_w: f64,
        cell_h: f64,
        cols: u32,
        rows: u32,
        cells: Vec<CellState>,
    ) -> Self {
        assert!(cell_w > 0.0 && cell_h > 0.0, "cell size must be positive");
        assert_eq!(
            cells.len(),
            (cols as usize) * (rows as usize),
            "cells length must equal cols * rows",
        );
        Self {
            origin_x: origin.x,
            origin_y: origin.y,
            cell_w,
            cell_h,
            cols,
            rows,
            cells,
        }
    }

    pub fn cell(&self, col: u32, row: u32) -> CellState {
        self.cells[(row as usize) * (self.cols as usize) + (col as usize)]
    }

    /// Bounding box of the grid itself, in world coordinates.
    pub fn bounding_box(&self) -> Rect {
        Rect::new(
            self.origin_x,
            self.origin_y,
            self.cell_w * self.cols as f64,
            self.cell_h * self.rows as f64,
        )
    }

    /// State of the single cell containing world point `p`.
    /// Points outside the grid are `Out` (the grid covers the shape's
    /// bounding box, so beyond it there is no shape).
    pub fn cell_at_world(&self, p: Point) -> CellState {
        if p.x < self.origin_x || p.y < self.origin_y {
            return CellState::Out;
        }
        let col = ((p.x - self.origin_x) / self.cell_w).floor();
        let row = ((p.y - self.origin_y) / self.cell_h).floor();
        if col >= self.cols as f64 || row >= self.rows as f64 {
            return CellState::Out;
        }
        self.cell(col as u32, row as u32)
    }

    /// Aggregate this grid into one whose cells are `fx`×`fy` times larger.
    ///
    /// Each output cell summarizes a `fx`×`fy` block of input cells with the
    /// rule: every block cell `In` → `In`; every block cell `Out` → `Out`;
    /// anything else → `Maybe`. This rule is **exactly** the classification
    /// the larger-cell template would carry if it had been generated
    /// directly: a cell is `In` only when the *whole* cell is inside the
    /// figure, which is equivalent to every sub-cell being `In`; same for
    /// `Out`. So an aggregated grid loses no precision compared to the
    /// directly-generated one — only memory/time spent on precomputation.
    /// This is the "granularity as fallback" property recorded in the
    /// design notes (a small-cell set stands in for any missing larger set
    /// whose dimensions are an integer multiple).
    ///
    /// The output is anchored at the largest multiple of `(cell_w*fx,
    /// cell_h*fy)` not exceeding this grid's origin; uncovered sub-cells in
    /// the resulting blocks are treated as `Out` (outside the figure).
    pub fn aggregate(&self, fx: u32, fy: u32) -> TemplateGrid {
        assert!(fx >= 1 && fy >= 1, "aggregate factors must be >= 1");
        let new_cw = self.cell_w * fx as f64;
        let new_ch = self.cell_h * fy as f64;

        // Aligned origin: largest multiple of new cell size <= our origin.
        let new_origin_x = (self.origin_x / new_cw).floor() * new_cw;
        let new_origin_y = (self.origin_y / new_ch).floor() * new_ch;

        let in_x_max = self.origin_x + self.cell_w * self.cols as f64;
        let in_y_max = self.origin_y + self.cell_h * self.rows as f64;
        let new_cols =
            (((in_x_max - new_origin_x) / new_cw).ceil() as u32).max(1);
        let new_rows =
            (((in_y_max - new_origin_y) / new_ch).ceil() as u32).max(1);

        // For each output cell, scan its `fx*fy` covering sub-cells.
        let mut cells = Vec::with_capacity((new_cols * new_rows) as usize);
        for j in 0..new_rows {
            for i in 0..new_cols {
                // World-space bounds of this output cell.
                let ox = new_origin_x + i as f64 * new_cw;
                let oy = new_origin_y + j as f64 * new_ch;
                let mut saw_in = false;
                let mut saw_out = false;
                for sub_j in 0..fy {
                    let world_y = oy + sub_j as f64 * self.cell_h;
                    // Map back to input-grid integer index.
                    let row_f = (world_y - self.origin_y) / self.cell_h;
                    let row = row_f.round() as i64;
                    for sub_i in 0..fx {
                        let world_x = ox + sub_i as f64 * self.cell_w;
                        let col_f = (world_x - self.origin_x) / self.cell_w;
                        let col = col_f.round() as i64;
                        let inside = col >= 0
                            && row >= 0
                            && (col as u32) < self.cols
                            && (row as u32) < self.rows;
                        match if inside {
                            self.cell(col as u32, row as u32)
                        } else {
                            CellState::Out // sub-cells outside the input grid
                        } {
                            CellState::In => saw_in = true,
                            CellState::Out => saw_out = true,
                            CellState::Maybe => {
                                saw_in = true;
                                saw_out = true;
                            }
                        }
                        if saw_in && saw_out {
                            break;
                        }
                    }
                    if saw_in && saw_out {
                        break;
                    }
                }
                cells.push(match (saw_in, saw_out) {
                    (true, false) => CellState::In,
                    (false, true) => CellState::Out,
                    (true, true) => CellState::Maybe,
                    (false, false) => CellState::Out, // empty block: never occurs after the asserts above
                });
            }
        }

        TemplateGrid {
            origin_x: new_origin_x,
            origin_y: new_origin_y,
            cell_w: new_cw,
            cell_h: new_ch,
            cols: new_cols,
            rows: new_rows,
            cells,
        }
    }

    /// Copy of this grid re-anchored at `origin + (dx, dy)`.
    ///
    /// Lets one precomputed template be stamped at any world position (the
    /// cells are cloned; classification data is identical).
    pub fn translated(&self, dx: f64, dy: f64) -> TemplateGrid {
        TemplateGrid {
            origin_x: self.origin_x + dx,
            origin_y: self.origin_y + dy,
            ..self.clone()
        }
    }

    /// Uniformly-scaled copy: every cell `k`× larger, classifications kept.
    /// Used by the figure↔grid scale equivalence: a template generated for
    /// figure F over cells of size C is identical to one for F·k over cells
    /// of size C·k. Cells are cloned; for a zero-clone version use
    /// [`PlacedTemplate::with_scale`].
    pub fn rescaled(&self, k: f64) -> TemplateGrid {
        assert!(k > 0.0, "scale factor must be positive");
        TemplateGrid {
            origin_x: self.origin_x * k,
            origin_y: self.origin_y * k,
            cell_w: self.cell_w * k,
            cell_h: self.cell_h * k,
            ..self.clone()
        }
    }

    /// Classify a rectangular region against the grid.
    ///
    /// - Returns [`CellState::Out`] if every overlapped cell is `Out` *and*
    ///   any portion of the region outside the grid is also (by definition)
    ///   `Out`.
    /// - Returns [`CellState::In`] only when the region lies entirely within
    ///   the grid *and* every overlapped cell is `In`.
    /// - Otherwise returns [`CellState::Maybe`].
    pub fn classify_region(&self, region: &Rect) -> CellState {
        let grid_bbox = self.bounding_box();

        // Empty intersection with the grid: only out-of-grid territory.
        if !region.intersects(&grid_bbox) {
            return CellState::Out;
        }

        let region_inside_grid = region.x >= grid_bbox.x
            && region.y >= grid_bbox.y
            && region.x_max() <= grid_bbox.x_max()
            && region.y_max() <= grid_bbox.y_max();

        // Cell index range overlapped by the (clamped) region.
        let col_start = ((region.x - self.origin_x) / self.cell_w).floor().max(0.0) as u32;
        let row_start = ((region.y - self.origin_y) / self.cell_h).floor().max(0.0) as u32;
        // ceil() then cap at cols/rows so a region flush with the right edge
        // doesn't index out of bounds.
        let col_end_f = ((region.x_max() - self.origin_x) / self.cell_w).ceil();
        let row_end_f = ((region.y_max() - self.origin_y) / self.cell_h).ceil();
        let col_end = (col_end_f.max(0.0) as u32).min(self.cols);
        let row_end = (row_end_f.max(0.0) as u32).min(self.rows);

        if col_start >= col_end || row_start >= row_end {
            // Region's overlap with the grid has zero area; treat as Out.
            return CellState::Out;
        }

        let mut saw_in = false;
        let mut saw_out = !region_inside_grid; // out-of-grid portion is Out
        for row in row_start..row_end {
            for col in col_start..col_end {
                match self.cell(col, row) {
                    CellState::Maybe => return CellState::Maybe,
                    CellState::In => saw_in = true,
                    CellState::Out => saw_out = true,
                }
                if saw_in && saw_out {
                    return CellState::Maybe;
                }
            }
        }

        if saw_in {
            // saw_out is false here (otherwise we'd have returned Maybe).
            CellState::In
        } else {
            CellState::Out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x3 grid, cells 10x10, origin (0, 0). Centre cell is In, the rest Out.
    fn centre_in_grid() -> TemplateGrid {
        use CellState::*;
        TemplateGrid::new(
            Point::new(0.0, 0.0),
            10.0,
            10.0,
            3,
            3,
            vec![
                Out, Out, Out,
                Out, In,  Out,
                Out, Out, Out,
            ],
        )
    }

    #[test]
    fn classify_centre_cell_is_in() {
        let g = centre_in_grid();
        assert_eq!(g.classify_region(&Rect::new(10.0, 10.0, 10.0, 10.0)), CellState::In);
    }

    #[test]
    fn classify_corner_cell_is_out() {
        let g = centre_in_grid();
        assert_eq!(g.classify_region(&Rect::new(0.0, 0.0, 10.0, 10.0)), CellState::Out);
    }

    #[test]
    fn classify_region_spanning_in_and_out_is_maybe() {
        let g = centre_in_grid();
        assert_eq!(g.classify_region(&Rect::new(0.0, 10.0, 20.0, 10.0)), CellState::Maybe);
    }

    #[test]
    fn classify_region_outside_grid_is_out() {
        let g = centre_in_grid();
        assert_eq!(g.classify_region(&Rect::new(100.0, 100.0, 10.0, 10.0)), CellState::Out);
    }

    #[test]
    fn classify_region_overflowing_grid_with_in_cell_is_maybe() {
        let g = centre_in_grid();
        // straddles the In cell and pokes outside the grid on the right.
        assert_eq!(g.classify_region(&Rect::new(10.0, 10.0, 30.0, 10.0)), CellState::Maybe);
    }

    #[test]
    fn classify_region_overflowing_grid_with_only_out_cells_is_out() {
        let g = centre_in_grid();
        // straddles two corner Out cells and pokes outside on the right.
        assert_eq!(g.classify_region(&Rect::new(0.0, 0.0, 100.0, 10.0)), CellState::Out);
    }

    #[test]
    fn rescaled_grid_classifies_proportionally() {
        use CellState::*;
        // 3x1 of cell 2 with cells In, Maybe, Out.
        let g = TemplateGrid::new(
            Point::new(0.0, 0.0),
            2.0,
            2.0,
            3,
            1,
            vec![In, Maybe, Out],
        );
        let big = g.rescaled(3.0);
        assert_eq!(big.cell_w, 6.0);
        // Classifications unchanged at the scaled coordinates.
        assert_eq!(big.cell_at_world(Point::new(3.0, 3.0)), In);
        assert_eq!(big.cell_at_world(Point::new(9.0, 3.0)), Maybe);
        assert_eq!(big.cell_at_world(Point::new(15.0, 3.0)), Out);
    }

    #[test]
    fn placed_template_with_scale_matches_zero_clone_rescale() {
        use CellState::*;
        let g = std::sync::Arc::new(TemplateGrid::new(
            Point::new(0.0, 0.0),
            2.0,
            2.0,
            3,
            1,
            vec![In, Maybe, Out],
        ));
        // Same grid presented at 3x scale should classify like a manually
        // rescaled standalone grid translated to the same world origin.
        let placed = PlacedTemplate::with_scale(g.clone(), 0.0, 0.0, 3.0);
        let manual = g.rescaled(3.0);
        for x in (1..=17).step_by(2) {
            let p = Point::new(x as f64, 3.0);
            assert_eq!(placed.cell_at_world(p), manual.cell_at_world(p), "x={x}");
        }
        // World cell sizes reflect the scale.
        assert_eq!(placed.world_cell_w(), 6.0);
        assert_eq!(placed.world_cell_h(), 6.0);
    }

    #[test]
    fn cell_at_world_reads_single_cells_and_outside_is_out() {
        let g = centre_in_grid();
        assert_eq!(g.cell_at_world(Point::new(15.0, 15.0)), CellState::In);
        assert_eq!(g.cell_at_world(Point::new(5.0, 5.0)), CellState::Out);
        assert_eq!(g.cell_at_world(Point::new(-1.0, 15.0)), CellState::Out);
        assert_eq!(g.cell_at_world(Point::new(30.0, 15.0)), CellState::Out);
    }

    #[test]
    fn aggregate_collapses_blocks_with_the_exact_rule() {
        use CellState::*;
        // 4 cols × 2 rows of 1×1, then aggregate to 2×2 blocks of 2×1.
        let g = TemplateGrid::new(
            Point::new(0.0, 0.0),
            1.0,
            1.0,
            4,
            2,
            vec![
                In, In, Out, Maybe,   // row 0: blocks (In,In) (Out,Maybe)
                In, In, Out, Out,     // row 1: blocks (In,In) (Out,Out)
            ],
        );
        let agg = g.aggregate(2, 1);
        assert_eq!(agg.cell_w, 2.0);
        assert_eq!(agg.cell_h, 1.0);
        assert_eq!(agg.cols, 2);
        assert_eq!(agg.rows, 2);
        assert_eq!(agg.cell(0, 0), In);    // pure In
        assert_eq!(agg.cell(1, 0), Maybe); // Out + Maybe → Maybe
        assert_eq!(agg.cell(0, 1), In);    // pure In
        assert_eq!(agg.cell(1, 1), Out);   // pure Out
    }

    #[test]
    fn aggregate_with_misaligned_origin_pads_with_out() {
        use CellState::*;
        // Origin 10; aggregating to cells of 20 anchors at 0, extending one
        // (Out) column to the left.
        let g = TemplateGrid::new(
            Point::new(10.0, 0.0),
            10.0,
            10.0,
            2,
            1,
            vec![In, In],
        );
        let agg = g.aggregate(2, 1);
        assert_eq!(agg.origin_x, 0.0);
        assert_eq!(agg.cols, 2);
        assert_eq!(agg.cell(0, 0), Maybe); // half outside grid (Out) + In
        assert_eq!(agg.cell(1, 0), Maybe); // half In + half outside (Out)
    }

    #[test]
    fn translated_grid_classifies_at_the_new_anchor() {
        let g = centre_in_grid().translated(100.0, 50.0);
        assert_eq!(g.classify_region(&Rect::new(110.0, 60.0, 10.0, 10.0)), CellState::In);
        assert_eq!(g.classify_region(&Rect::new(10.0, 10.0, 10.0, 10.0)), CellState::Out);
    }

    #[test]
    fn maybe_cell_short_circuits() {
        use CellState::*;
        let g = TemplateGrid::new(
            Point::new(0.0, 0.0),
            10.0,
            10.0,
            2,
            1,
            vec![Maybe, In],
        );
        assert_eq!(g.classify_region(&Rect::new(0.0, 0.0, 20.0, 10.0)), CellState::Maybe);
    }
}
