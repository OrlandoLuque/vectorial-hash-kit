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

use crate::geom::{Point, Rect};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CellState {
    Out,
    Maybe,
    In,
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
