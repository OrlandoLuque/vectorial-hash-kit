//! Culling: find tree items inside a shape.

use std::collections::{HashMap, HashSet};

use crate::geom::{Point, Rect};
use crate::template::{CellState, PlacedTemplate, TemplateGrid};
use crate::tree::{NodeId, Positioned, Side, Tree};

/// How [`Tree::cull_walk`] finds each leaf's neighbours.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WalkNeighbors {
    /// Ascend/descend through the existing parent pointers (Samet-style).
    /// Zero extra storage; O(1) amortized per neighbour.
    Samet,
    /// Probe a point just across each edge and `locate` it from the root.
    /// Zero extra storage; O(depth) per neighbour.
    Probe,
    /// Stored per-leaf neighbour lists, O(1); requires the `neighbors`
    /// feature (maintained on every split/merge).
    #[cfg(feature = "neighbors")]
    Ropes,
}

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
    /// The [`PlacedTemplate`] shares the canonical grid (`Arc`) — no cell
    /// data is cloned at resolution time. Contract: the placed grid's cells
    /// must be exactly `cell_w` × `cell_h` and sit on multiples of the cell
    /// size, so that aligned tree nodes map 1:1 onto template cells.
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        let _ = (cell_w, cell_h);
        None
    }

    /// Optional 1×1-cell raster of the shape used for per-item tests in
    /// leaf cells: `In`/`Out` pixels answer immediately and only `Maybe`
    /// (boundary) pixels fall back to the exact `contains_point`.
    fn point_template(&self) -> Option<&PlacedTemplate> {
        None
    }

    /// Optional **analytic** classification of an axis-aligned node box against
    /// the shape: `In` (box fully inside → take all its items, no per-point
    /// test), `Out` (fully outside → prune the subtree) or `Maybe` (straddles →
    /// descend). Default `None` falls back to the template / bbox-overlap path.
    ///
    /// Implement it for shapes with a cheap exact box test (circle, capsule, …):
    /// the cull then prunes **tightly without a precomputed template**, and the
    /// tree's recursion handles arbitrary shape "thickness" for free — it only
    /// descends the nodes the shape actually reaches, coarse in the interior and
    /// fine at the boundary, with no manual neighbour chasing. Takes precedence
    /// over the template path when it returns `Some`.
    fn classify_box(&self, b: &Rect) -> Option<CellState> {
        let _ = b;
        None
    }

    /// Opt into the **SoA batch narrowphase**: when `true`, the leaf-level
    /// per-item test runs [`Shape::contains_batch`] over contiguous position
    /// arrays instead of `contains_point` one at a time. Default `false` — only
    /// analytic shapes with a branchless, auto-vectorising kernel (e.g.
    /// [`Capsule`]) benefit, and only on leaves with many items; everything else
    /// keeps the exact current path (zero behaviour change).
    fn wants_batch(&self) -> bool { false }

    /// Batch `contains_point` over SoA position arrays `xs`/`ys` (parallel),
    /// writing a hit mask of length `xs.len()` into `out`. The default loops
    /// `contains_point`; analytic shapes override with a **branchless** kernel
    /// that LLVM auto-vectorises (SoA + no data-dependent branch → SIMD). Only
    /// consulted when [`Shape::wants_batch`] is `true`.
    fn contains_batch(&self, xs: &[f64], ys: &[f64], out: &mut Vec<bool>) {
        out.clear();
        out.extend(xs.iter().zip(ys).map(|(&x, &y)| self.contains_point(Point::new(x, y))));
    }
}

/// A **2D capsule**: the segment `a`–`b` thickened by radius `r`. As a [`Shape`]
/// it answers "every item within `r` of the segment" — a thick ray-cast via the
/// normal `cull` (the 2D analogue of [`crate::Segment3`]). All the segment
/// invariants are precomputed in [`Capsule::new`]; `contains_point` is the
/// branch-on-projection perpendicular distance (no division); and `classify_box`
/// prunes the cull to the radius-`r` band with a cheap conservative slab test in
/// segment-aligned coords + a centre-based `In` — **no template needed**, the
/// tree recursion handles the thickness. For the *nearest* hit along the ray,
/// `cull` then pick the minimum projection `t` (or use the thin-corridor
/// [`crate::Tree::raycast_first`] when the ray is thin).
pub struct Capsule {
    a: Point,
    abx: f64,
    aby: f64,
    len2: f64,
    inv_len2: f64,
    len: f64,
    ux: f64,
    uy: f64,
    nx: f64,
    ny: f64,
    r: f64,
    r2: f64,
    bbox: Rect,
}

impl Capsule {
    pub fn new(a: Point, b: Point, r: f64) -> Self {
        let (abx, aby) = (b.x - a.x, b.y - a.y);
        let len2 = abx * abx + aby * aby;
        let len = len2.sqrt();
        let (ux, uy) = if len > 0.0 { (abx / len, aby / len) } else { (1.0, 0.0) };
        let (nx, ny) = (-uy, ux);
        let bbox = Rect::new(a.x.min(b.x) - r, a.y.min(b.y) - r, (a.x.max(b.x) - a.x.min(b.x)) + 2.0 * r, (a.y.max(b.y) - a.y.min(b.y)) + 2.0 * r);
        Self { a, abx, aby, len2, inv_len2: if len2 > 0.0 { 1.0 / len2 } else { 0.0 }, len, ux, uy, nx, ny, r, r2: r * r, bbox }
    }
    /// Squared distance from `p` to the spine, perpendicular form (no division).
    #[inline]
    pub fn spine_dist2(&self, p: Point) -> f64 {
        let (apx, apy) = (p.x - self.a.x, p.y - self.a.y);
        let dot = apx * self.abx + apy * self.aby;
        if dot <= 0.0 {
            apx * apx + apy * apy
        } else if dot >= self.len2 {
            let (bpx, bpy) = (p.x - (self.a.x + self.abx), p.y - (self.a.y + self.aby));
            bpx * bpx + bpy * bpy
        } else {
            (apx * apx + apy * apy) - dot * dot * self.inv_len2
        }
    }
}

impl Shape for Capsule {
    fn bounding_box(&self) -> Rect { self.bbox }
    fn contains_point(&self, p: Point) -> bool { self.spine_dist2(p) <= self.r2 }
    fn classify_box(&self, b: &Rect) -> Option<CellState> {
        // Conservative slab reject: the whole capsule lives in the oriented box
        // [−r, len+r] (along u) × [−r, r] (along n) about `a`. Project the query
        // box onto u and n (AABB → corners by sign); no overlap ⟹ Out. Cheap, safe.
        let pick = |dx: f64, dy: f64| {
            let lo = dx * (if dx > 0.0 { b.x } else { b.x_max() }) + dy * (if dy > 0.0 { b.y } else { b.y_max() });
            let hi = dx * (if dx > 0.0 { b.x_max() } else { b.x }) + dy * (if dy > 0.0 { b.y_max() } else { b.y });
            (lo, hi)
        };
        let off_u = self.ux * self.a.x + self.uy * self.a.y;
        let off_n = self.nx * self.a.x + self.ny * self.a.y;
        let (u_lo, u_hi) = pick(self.ux, self.uy);
        let (n_lo, n_hi) = pick(self.nx, self.ny);
        let (u_lo, u_hi) = (u_lo - off_u, u_hi - off_u);
        let (n_lo, n_hi) = (n_lo - off_n, n_hi - off_n);
        if u_hi < -self.r || u_lo > self.len + self.r || n_hi < -self.r || n_lo > self.r {
            return Some(CellState::Out);
        }
        // Conservative In: centre within r − half-diagonal ⟹ whole box inside.
        let (cx, cy) = (b.x + b.width * 0.5, b.y + b.height * 0.5);
        let half_diag = 0.5 * (b.width * b.width + b.height * b.height).sqrt();
        if self.r > half_diag && self.spine_dist2(Point::new(cx, cy)) <= (self.r - half_diag) * (self.r - half_diag) {
            return Some(CellState::In);
        }
        Some(CellState::Maybe)
    }
    fn wants_batch(&self) -> bool { true }
    fn contains_batch(&self, xs: &[f64], ys: &[f64], out: &mut Vec<bool>) {
        out.clear();
        out.resize(xs.len(), false);
        // Branchless clamped-projection distance (no cap branches) — same result
        // as `contains_point`, but the tight zip loop auto-vectorises (SIMD).
        let (ax, ay, abx, aby, inv, r2) = (self.a.x, self.a.y, self.abx, self.aby, self.inv_len2, self.r2);
        for ((&x, &y), o) in xs.iter().zip(ys).zip(out.iter_mut()) {
            let (apx, apy) = (x - ax, y - ay);
            let t = ((apx * abx + apy * aby) * inv).clamp(0.0, 1.0);
            let (dx, dy) = (apx - abx * t, apy - aby * t);
            *o = dx * dx + dy * dy <= r2;
        }
    }
}

/// Per-execution cache: one resolved template per distinct cell size.
/// Shared with the reference quadtree so both structures classify cells
/// through the exact same machinery.
pub(crate) type SizeCache = HashMap<(u64, u64), Option<PlacedTemplate>>;

/// Per-item resolution of a `Maybe` leaf, shared by every structure:
/// bbox pre-filter (closed bounds — the figure boundary belongs to the
/// figure), then the 1×1 raster when available (exact geometry only on
/// boundary pixels).
/// Reusable SoA batch-narrowphase scratch: `xs`, `ys`, the hit-mask, and the
/// original item indices of the bbox-prefiltered points.
type SoaScratch = (Vec<f64>, Vec<f64>, Vec<bool>, Vec<u32>);

thread_local! {
    // Per-thread scratch — no allocation per leaf, no cross-thread sharing.
    static SOA_SCRATCH: std::cell::RefCell<SoaScratch> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

pub(crate) fn collect_matching_items<'a, T: Positioned, S: Shape>(
    items: &'a [T],
    shape: &S,
    shape_bbox: &Rect,
    out: &mut Vec<&'a T>,
) {
    // SoA batch path — opt-in (analytic shapes with a vectorising kernel). The
    // bbox pre-filter still runs (scalar), then the kernel tests the survivors
    // contiguously. Only taken when there's no per-item raster.
    if shape.wants_batch() && shape.point_template().is_none() {
        SOA_SCRATCH.with(|cell| {
            let (xs, ys, mask, idx) = &mut *cell.borrow_mut();
            xs.clear();
            ys.clear();
            idx.clear();
            for (i, it) in items.iter().enumerate() {
                let p = it.position();
                if shape_bbox.contains_closed(p) {
                    xs.push(p.x);
                    ys.push(p.y);
                    idx.push(i as u32);
                }
            }
            shape.contains_batch(xs, ys, mask);
            for (j, &hit) in mask.iter().enumerate() {
                if hit {
                    out.push(&items[idx[j] as usize]);
                }
            }
        });
        return;
    }

    let point_grid = shape.point_template();
    for it in items {
        let p = it.position();
        if !shape_bbox.contains_closed(p) {
            continue;
        }
        match point_grid.map(|g| g.cell_at_world(p)) {
            Some(CellState::In) => out.push(it),
            Some(CellState::Out) => {}
            _ => {
                if shape.contains_point(p) {
                    out.push(it);
                }
            }
        }
    }
}

impl<T: Positioned> Tree<T> {
    /// Return references to every item inside `shape`.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bbox = shape.bounding_box();
        let mut sizes = SizeCache::new();
        self.cull_recurse(self.root, shape, &bbox, false, &mut sizes, &mut out);
        out
    }

    /// Batch cull — see [`crate::Tree3::cull_many`].
    pub fn cull_many<'a, S: Shape>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// Parallel batch cull — see [`crate::Tree3::cull_many_par`].
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
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
            None => collect_matching_items(&self.get(node_id).items, shape, shape_bbox, out),
        }
    }

    /// Flood-fill cull: start at the leaf containing `seed` (a point that
    /// must lie inside `shape`) and expand through leaf neighbours instead
    /// of descending the tree. Expansion continues through `In`/`Maybe`
    /// leaves and stops at `Out` ones; the collected result is identical to
    /// [`Tree::cull`] for connected shapes.
    pub fn cull_walk<'a, S: Shape>(
        &'a self,
        shape: &S,
        seed: Point,
        strategy: WalkNeighbors,
    ) -> Vec<&'a T> {
        let mut out = Vec::new();
        if !self.get(self.root).bbox.contains(seed) {
            return out;
        }
        let shape_bbox = shape.bounding_box();
        let mut sizes = SizeCache::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        let start = self.locate(seed);
        visited.insert(start);
        let mut queue: Vec<NodeId> = vec![start];
        let mut nbuf: Vec<NodeId> = Vec::new();

        while let Some(leaf) = queue.pop() {
            let lb = self.get(leaf).bbox;
            match classify_child(shape, &shape_bbox, &lb, &mut sizes) {
                CellState::Out => continue, // don't collect, don't expand
                CellState::In => out.extend(self.get(leaf).items.iter()),
                CellState::Maybe => {
                    collect_matching_items(&self.get(leaf).items, shape, &shape_bbox, &mut out)
                }
            }
            for side in Side::ALL {
                nbuf.clear();
                match strategy {
                    WalkNeighbors::Samet => self.neighbors_samet(leaf, side, &mut nbuf),
                    WalkNeighbors::Probe => self.neighbors_probe(leaf, side, &mut nbuf),
                    #[cfg(feature = "neighbors")]
                    WalkNeighbors::Ropes => {
                        nbuf.extend_from_slice(self.neighbors_ropes(leaf, side))
                    }
                }
                for &n in &nbuf {
                    if visited.insert(n) {
                        queue.push(n);
                    }
                }
            }
        }
        out
    }
}

pub(crate) fn classify_child<S: Shape>(
    shape: &S,
    shape_bbox: &Rect,
    child_bbox: &Rect,
    sizes: &mut SizeCache,
) -> CellState {
    // 0. Analytic box classification (circle / capsule / …) — tight distance
    //    pruning with no template; takes precedence when the shape provides it.
    if let Some(state) = shape.classify_box(child_bbox) {
        return state;
    }

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
        fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
            use CellState::*;
            self.resolved.borrow_mut().push((cell_w, cell_h));
            // Whatever the size, mark the cell range covering x >= 50 as In
            // within the right half of a 100x100 world.
            let cols = (50.0 / cell_w).max(1.0) as u32;
            let rows = (100.0 / cell_h).max(1.0) as u32;
            Some(PlacedTemplate::new(
                std::sync::Arc::new(TemplateGrid::new(
                    Point::new(50.0, 0.0),
                    cell_w,
                    cell_h,
                    cols,
                    rows,
                    vec![In; (cols * rows) as usize],
                )),
                0.0,
                0.0,
            ))
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
        raster: PlacedTemplate,
    }
    impl Shape for RasterShape {
        fn bounding_box(&self) -> Rect {
            self.bbox
        }
        fn contains_point(&self, p: Point) -> bool {
            p.y < 2.0 // only used for Maybe pixels
        }
        fn point_template(&self) -> Option<&PlacedTemplate> {
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
        let shape = RasterShape {
            bbox: Rect::new(0.0, 0.0, 3.0, 3.0),
            raster: PlacedTemplate::new(std::sync::Arc::new(raster), 0.0, 0.0),
        };
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

    /// `cull_walk` must agree with `cull` for every neighbour strategy, on a
    /// churned tree (splits and merges) with a circle query.
    #[test]
    fn cull_walk_matches_descent_for_all_strategies() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 2);
        let mut x = 0xABCD1234u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut pts = Vec::new();
        for _ in 0..150 {
            let p = Point::new(next() * 256.0, next() * 256.0);
            pts.push(p);
            tree.insert(Pt(p));
        }
        for p in pts.iter().step_by(4) {
            tree.remove(*p, |it| it.0 == *p);
        }

        let circle = Circle { center: Point::new(120.0, 140.0), radius: 70.0 };
        let mut expected: Vec<Point> = tree.cull(&circle).iter().map(|p| p.0).collect();
        expected.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));

        let strategies: Vec<WalkNeighbors> = vec![
            WalkNeighbors::Samet,
            WalkNeighbors::Probe,
            #[cfg(feature = "neighbors")]
            WalkNeighbors::Ropes,
        ];
        for strategy in strategies {
            let mut got: Vec<Point> = tree
                .cull_walk(&circle, circle.center, strategy)
                .iter()
                .map(|p| p.0)
                .collect();
            got.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
            assert_eq!(got, expected, "strategy {strategy:?}");
        }
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
