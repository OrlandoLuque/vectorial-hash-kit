//! Hierarchical, deduplicated runtime bank of precomputed templates.
//!
//! Index hierarchy (per the paper): **shape type → shape dimensions →
//! cell width → cell height → angle → offset x → offset y**.
//!
//! The figure is never moved to fit a grid. At query time the bank receives
//! the figure's real (integer) origin, computes its displacement within the
//! global virtual grid of the requested cell size, and returns the template
//! that was *generated* with exactly that displacement — re-anchored to the
//! world so its cells align 1:1 with the static lattice.
//!
//! Many (shape, dims, cell size, angle, offset) combinations produce the
//! same template content (same cols × rows, same cell states). The bank
//! stores each unique template once (`Arc`) and lets every matching index
//! leaf share it.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use vectorial_hash::{PlacedTemplate, Point, TemplateGrid};

use crate::matrix;
use crate::polygon::{rotated_copy, Polygon};
use crate::templates::{angle_to_radians, get_template_grid_fast};
use crate::adapter::matrix_to_template_grid;

/// Identifies a figure family: shape type plus its dimension vector.
/// The dimension vector is whatever minimally parameterizes the shape
/// (circle: 1, square: 1, rectangle: 2, drop: 2 — or 1 with fixed
/// proportions). Stored as bit patterns so the key is `Hash + Eq`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FigureKey {
    pub shape_id: u32,
    dims_bits: Vec<u64>,
}

impl FigureKey {
    pub fn new(shape_id: u32, dims: &[f64]) -> Self {
        Self {
            shape_id,
            dims_bits: dims.iter().map(|d| d.to_bits()).collect(),
        }
    }
}

/// One index leaf: where this key's template sits relative to the offset
/// cell (anchor) plus the shared, deduplicated grid (anchored at (0,0)).
#[derive(Clone)]
struct Entry {
    anchor_x: f64,
    anchor_y: f64,
    grid: Arc<TemplateGrid>,
}

#[derive(Default)]
struct OffsetIndex {
    /// (angle bits, offset x, offset y) → entry.
    map: HashMap<(u64, u32, u32), Entry>,
}

#[derive(Default)]
struct SizeIndex {
    sizes: HashMap<(u32, u32), OffsetIndex>,
}

/// Content-dedup key: cell size + grid dimensions + row-major cell states.
type DedupKey = (u32, u32, u32, u32, Vec<u8>);

#[derive(Default)]
pub struct TemplateBank {
    figures: HashMap<FigureKey, SizeIndex>,
    dedup: HashMap<DedupKey, Arc<TemplateGrid>>,
    entries: usize,
}

impl TemplateBank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate the full template set for `figure` (whose scaled, unrotated
    /// polygon is `base`) over the given `angles` (degrees) and one cell
    /// size, enumerating every integer origin offset `0..cell_w × 0..cell_h`
    /// (1-unit steps, like the original pipeline). Angles run in parallel.
    ///
    /// Use `cell_w = cell_h = 1` to build the 1×1 point raster: with integer
    /// origins the sub-cell offset is always (0, 0), so it costs a single
    /// template per angle.
    pub fn generate_size(
        &mut self,
        figure: &FigureKey,
        base: &Polygon,
        angles: &[f64],
        cell_w: u32,
        cell_h: u32,
    ) {
        assert!(cell_w >= 1 && cell_h >= 1);
        let cw = cell_w as i64;
        let ch = cell_h as i64;

        let per_angle: Vec<Vec<(f64, u32, u32, f64, f64, matrix::Matrix)>> = angles
            .par_iter()
            .map(|&angle| {
                let rotated = rotated_copy(base, angle_to_radians(angle));
                let mut results = Vec::with_capacity((cell_w * cell_h) as usize);
                for ox in 0..cell_w {
                    for oy in 0..cell_h {
                        let mut moved = rotated.clone();
                        moved.move_by(ox as f64, oy as f64);
                        let gx0 = (moved.x_min / cw as f64).floor() as i64;
                        let gx1 = (moved.x_max / cw as f64).ceil() as i64;
                        let gy0 = (moved.y_min / ch as f64).floor() as i64;
                        let gy1 = (moved.y_max / ch as f64).ceil() as i64;
                        let m = get_template_grid_fast(gx0, gy0, gx1, gy1, cw, ch, &moved);
                        results.push((
                            angle,
                            ox,
                            oy,
                            gx0 as f64 * cw as f64,
                            gy0 as f64 * ch as f64,
                            m,
                        ));
                    }
                }
                results
            })
            .collect();

        for results in per_angle {
            for (angle, ox, oy, anchor_x, anchor_y, m) in results {
                let (cols, rows) = matrix::dimensions(&m);
                let mut flat = Vec::with_capacity(cols * rows);
                for y in 0..rows {
                    for x in 0..cols {
                        flat.push(m[x][y]);
                    }
                }
                let grid = self
                    .dedup
                    .entry((cell_w, cell_h, cols as u32, rows as u32, flat))
                    .or_insert_with(|| {
                        Arc::new(matrix_to_template_grid(
                            &m,
                            Point::new(0.0, 0.0),
                            cell_w as f64,
                            cell_h as f64,
                        ))
                    })
                    .clone();
                self.figures
                    .entry(figure.clone())
                    .or_default()
                    .sizes
                    .entry((cell_w, cell_h))
                    .or_default()
                    .map
                    .insert((angle.to_bits(), ox, oy), Entry { anchor_x, anchor_y, grid });
                self.entries += 1;
            }
        }
    }

    /// Whether a template set exists for this figure and cell size.
    pub fn has_size(&self, figure: &FigureKey, cell_w: u32, cell_h: u32) -> bool {
        self.figures
            .get(figure)
            .is_some_and(|s| s.sizes.contains_key(&(cell_w, cell_h)))
    }

    /// Resolve the template for `figure` at `angle_deg`, applied with its
    /// origin at the integer world position `origin`, aligned to the global
    /// virtual grid of `cell_w` × `cell_h` cells.
    ///
    /// The origin's displacement within its virtual cell selects which
    /// precomputed template to use; the figure itself is never moved.
    pub fn template_for(
        &self,
        figure: &FigureKey,
        cell_w: u32,
        cell_h: u32,
        angle_deg: f64,
        origin: (i64, i64),
    ) -> Option<TemplateGrid> {
        let idx = self.figures.get(figure)?.sizes.get(&(cell_w, cell_h))?;
        let cw = cell_w as i64;
        let ch = cell_h as i64;
        let ox = origin.0.rem_euclid(cw);
        let oy = origin.1.rem_euclid(ch);
        let base_x = origin.0 - ox;
        let base_y = origin.1 - oy;
        let entry = idx.map.get(&(angle_deg.to_bits(), ox as u32, oy as u32))?;
        Some(entry.grid.translated(
            entry.anchor_x + base_x as f64,
            entry.anchor_y + base_y as f64,
        ))
    }

    /// 1×1-cell raster of the figure for per-point tests, re-anchored at the
    /// integer world `origin`. Requires `generate_size(.., 1, 1)`.
    pub fn point_raster(
        &self,
        figure: &FigureKey,
        angle_deg: f64,
        origin: (i64, i64),
    ) -> Option<TemplateGrid> {
        self.template_for(figure, 1, 1, angle_deg, origin)
    }

    /// Zero-clone variant of [`TemplateBank::template_for`]: returns a
    /// [`PlacedTemplate`] sharing the canonical grid behind its `Arc`, with
    /// the world displacement carried alongside. This is what the cull hot
    /// path should use — no cell data is copied per resolution.
    pub fn placed_for(
        &self,
        figure: &FigureKey,
        cell_w: u32,
        cell_h: u32,
        angle_deg: f64,
        origin: (i64, i64),
    ) -> Option<PlacedTemplate> {
        let idx = self.figures.get(figure)?.sizes.get(&(cell_w, cell_h))?;
        let cw = cell_w as i64;
        let ch = cell_h as i64;
        let ox = origin.0.rem_euclid(cw);
        let oy = origin.1.rem_euclid(ch);
        let base_x = origin.0 - ox;
        let base_y = origin.1 - oy;
        let entry = idx.map.get(&(angle_deg.to_bits(), ox as u32, oy as u32))?;
        Some(PlacedTemplate::new(
            entry.grid.clone(),
            entry.anchor_x + base_x as f64,
            entry.anchor_y + base_y as f64,
        ))
    }

    /// Zero-clone variant of [`TemplateBank::point_raster`].
    pub fn placed_raster(
        &self,
        figure: &FigureKey,
        angle_deg: f64,
        origin: (i64, i64),
    ) -> Option<PlacedTemplate> {
        self.placed_for(figure, 1, 1, angle_deg, origin)
    }

    /// Total index leaves (key combinations).
    pub fn entry_count(&self) -> usize {
        self.entries
    }

    /// Unique template instances actually stored.
    pub fn unique_count(&self) -> usize {
        self.dedup.len()
    }

    /// Estimated heap memory, split into what stores the templates
    /// themselves and what stores the lookup structure. Hash-map overhead is
    /// estimated from each map's allocated capacity.
    pub fn memory_usage(&self) -> BankMemory {
        use std::mem::size_of;

        // Unique template grids (shared behind Arcs).
        let mut grids_bytes = 0usize;
        for grid in self.dedup.values() {
            grids_bytes += size_of::<TemplateGrid>() + grid.cells.capacity();
        }

        // The dedup map also retains a flat copy of each unique template's
        // cells as its key (build-time only; could be dropped after
        // generation).
        let mut dedup_keys_bytes = 0usize;
        for (key, _) in self.dedup.iter() {
            dedup_keys_bytes += size_of::<DedupKey>() + key.4.capacity();
        }
        dedup_keys_bytes +=
            self.dedup.capacity() * (size_of::<DedupKey>() + size_of::<Arc<TemplateGrid>>() + 1);

        // Index levels: figure map -> size maps -> offset maps of Entry.
        let mut index_bytes = self.figures.capacity()
            * (size_of::<FigureKey>() + size_of::<SizeIndex>() + 1);
        for fig in self.figures.values() {
            index_bytes += fig.sizes.capacity()
                * (size_of::<(u32, u32)>() + size_of::<OffsetIndex>() + 1);
            for offsets in fig.sizes.values() {
                index_bytes += offsets.map.capacity()
                    * (size_of::<(u64, u32, u32)>() + size_of::<Entry>() + 1);
            }
        }

        BankMemory { grids_bytes, index_bytes, dedup_keys_bytes }
    }
}

/// Heap breakdown returned by [`TemplateBank::memory_usage`].
#[derive(Debug, Clone, Copy)]
pub struct BankMemory {
    /// Unique, deduplicated template grids (the actual cell data).
    pub grids_bytes: usize,
    /// The hierarchical lookup index (keys, anchors, shared pointers).
    pub index_bytes: usize,
    /// Flat cell copies retained as dedup-map keys (generation-time aid).
    pub dedup_keys_bytes: usize,
}

impl BankMemory {
    pub fn total(&self) -> usize {
        self.grids_bytes + self.index_bytes + self.dedup_keys_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon::create_square;
    use vectorial_hash::CellState;

    #[test]
    fn bank_template_matches_geometry_at_arbitrary_integer_origin() {
        let base = create_square(0.0, 0.0, 8.0, 8.0);
        let fig = FigureKey::new(7, &[8.0]);
        let mut bank = TemplateBank::new();
        bank.generate_size(&fig, &base, &[0.0], 4, 4);

        let origin = (37i64, 21i64);
        let grid = bank.template_for(&fig, 4, 4, 0.0, origin).unwrap();
        // Grid must be aligned to the global 4px lattice.
        assert_eq!(grid.origin_x.rem_euclid(4.0), 0.0);
        assert_eq!(grid.origin_y.rem_euclid(4.0), 0.0);

        let mut moved = base.clone();
        moved.move_by(origin.0 as f64, origin.1 as f64);
        for cx in (28..56).step_by(4) {
            for cy in (12..40).step_by(4) {
                let state =
                    grid.cell_at_world(Point::new(cx as f64 + 2.0, cy as f64 + 2.0));
                let cell = create_square(cx as f64, cy as f64, cx as f64 + 4.0, cy as f64 + 4.0);
                let expected = if cell.x_min >= moved.x_max
                    || cell.x_max <= moved.x_min
                    || cell.y_min >= moved.y_max
                    || cell.y_max <= moved.y_min
                {
                    CellState::Out
                } else if moved.completely_contains(&cell) {
                    CellState::In
                } else {
                    CellState::Maybe
                };
                assert_eq!(state, expected, "cell at ({cx},{cy})");
            }
        }
    }

    #[test]
    fn bank_dedups_identical_templates_across_angles() {
        // A square rotated 90 deg produces the same template contents, so
        // two angles must not double the unique count.
        let base = create_square(0.0, 0.0, 8.0, 8.0);
        let fig = FigureKey::new(7, &[8.0]);
        let mut bank = TemplateBank::new();
        bank.generate_size(&fig, &base, &[0.0, 90.0], 4, 4);
        assert_eq!(bank.entry_count(), 2 * 16);
        assert!(
            bank.unique_count() < bank.entry_count(),
            "expected shared templates: {} unique of {}",
            bank.unique_count(),
            bank.entry_count(),
        );
    }

    #[test]
    fn point_raster_is_single_offset_and_classifies_pixels() {
        let base = create_square(0.0, 0.0, 8.0, 8.0);
        let fig = FigureKey::new(7, &[8.0]);
        let mut bank = TemplateBank::new();
        bank.generate_size(&fig, &base, &[0.0], 1, 1);

        let raster = bank.point_raster(&fig, 0.0, (100, 50)).unwrap();
        // Pixel well inside the square -> In; far outside -> Out.
        assert_eq!(raster.cell_at_world(Point::new(104.5, 54.5)), CellState::In);
        assert_eq!(raster.cell_at_world(Point::new(120.5, 54.5)), CellState::Out);
    }

    #[test]
    fn placed_for_matches_materialized_template() {
        let base = create_square(0.0, 0.0, 8.0, 8.0);
        let fig = FigureKey::new(7, &[8.0]);
        let mut bank = TemplateBank::new();
        bank.generate_size(&fig, &base, &[0.0], 4, 4);
        let origin = (37i64, 21i64);
        let materialized = bank.template_for(&fig, 4, 4, 0.0, origin).unwrap();
        let placed = bank.placed_for(&fig, 4, 4, 0.0, origin).unwrap();
        for cx in (28..56).step_by(2) {
            for cy in (12..40).step_by(2) {
                let p = Point::new(cx as f64 + 1.0, cy as f64 + 1.0);
                assert_eq!(placed.cell_at_world(p), materialized.cell_at_world(p), "at {p:?}");
            }
        }
    }

    #[test]
    fn missing_size_or_angle_returns_none() {
        let base = create_square(0.0, 0.0, 8.0, 8.0);
        let fig = FigureKey::new(7, &[8.0]);
        let mut bank = TemplateBank::new();
        bank.generate_size(&fig, &base, &[0.0], 4, 4);
        assert!(bank.template_for(&fig, 8, 8, 0.0, (0, 0)).is_none());
        assert!(bank.template_for(&fig, 4, 4, 15.0, (0, 0)).is_none());
        assert!(!bank.has_size(&fig, 8, 8));
        assert!(bank.has_size(&fig, 4, 4));
    }
}
