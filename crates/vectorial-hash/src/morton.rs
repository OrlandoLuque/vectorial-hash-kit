//! Morton / Z-order linear grid in **2D** — the 2D analogue of
//! [`crate::MortonGrid3`]. A pointer-free spatial index: quantise each point to
//! an integer cell `(ix, iy)`, interleave the cell's bits into one Morton code
//! (the Z-order curve), and bucket points by that code in a hash map. The
//! "spatial hash" most 2D engines reach for — flat, O(1) cell lookup, trivial
//! build, but a *single* resolution (the adaptive [`crate::Tree`] /
//! [`crate::QuadTree`] tune leaf size to local density; this does not).

use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::culling::{collect_matching_items, Shape};
use crate::geom::{Point, Rect};
use crate::serde_io::{corrupt, r_rect, r_u32, r_u64, r_u8, w_rect, w_u32, w_u64, w_u8};
use crate::template::CellState;
use crate::tree::{knn_offer2, Positioned, RaycastOut};
use crate::tree3::{knn_worst, KnnEntry};

/// Interleave the low 32 bits of `n` with one zero bit between each (`abc` →
/// `a.b.c`), so two OR'd (shifted 0/1) pack `(x, y)` into a u64 Z-order code.
#[inline]
fn part1by1(mut n: u64) -> u64 {
    n &= 0xffff_ffff;
    n = (n | (n << 16)) & 0x0000_ffff_0000_ffff;
    n = (n | (n << 8)) & 0x00ff_00ff_00ff_00ff;
    n = (n | (n << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    n = (n | (n << 2)) & 0x3333_3333_3333_3333;
    n = (n | (n << 1)) & 0x5555_5555_5555_5555;
    n
}

/// The 2D Morton (Z-order) code of integer cell `(x, y)`. Each axis fits in 32
/// bits (`levels <= 31`). Bit `i` of `x` lands at code position `2i`, `y` at `2i+1`.
#[inline]
pub fn morton2(x: u32, y: u32) -> u64 { part1by1(x as u64) | (part1by1(y as u64) << 1) }

/// Inverse of [`part1by1`]: gather every other bit back down.
#[inline]
fn compact1by1(mut n: u64) -> u64 {
    n &= 0x5555_5555_5555_5555;
    n = (n ^ (n >> 1)) & 0x3333_3333_3333_3333;
    n = (n ^ (n >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    n = (n ^ (n >> 4)) & 0x00ff_00ff_00ff_00ff;
    n = (n ^ (n >> 8)) & 0x0000_ffff_0000_ffff;
    n = (n ^ (n >> 16)) & 0x0000_0000_ffff_ffff;
    n
}

/// Decode a Morton code back to its integer cell `(x, y)` — inverse of [`morton2`].
#[inline]
pub fn demorton2(code: u64) -> (u32, u32) { (compact1by1(code) as u32, compact1by1(code >> 1) as u32) }

/// Clamp a world coordinate to a cell index in `0..n` along one axis.
#[inline]
fn axis_index_n(v: f64, lo: f64, cell: f64, n: u32) -> u32 {
    if cell <= 0.0 {
        return 0;
    }
    let i = ((v - lo) / cell).floor();
    if i < 0.0 {
        0
    } else if i as u64 >= n as u64 {
        n - 1
    } else {
        i as u32
    }
}

/// Clip the ray `o + t·u` to `world` → the `[t_enter, t_exit]` it spends inside
/// (capped at `max_t`), or `None` if it misses. Slab test.
fn clip_ray_rect(o: Point, ux: f64, uy: f64, world: &Rect, max_t: f64) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0_f64, max_t);
    for (oa, ua, lo, hi) in [(o.x, ux, world.x, world.x_max()), (o.y, uy, world.y, world.y_max())] {
        if ua == 0.0 {
            if oa < lo || oa > hi {
                return None;
            }
        } else {
            let (mut ta, mut tb) = ((lo - oa) / ua, (hi - oa) / ua);
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((t0, t1))
}

pub struct MortonGrid<T: Positioned> {
    world: Rect,
    levels: u32,
    cells_per_axis: u32,
    cw: f64,
    ch: f64,
    cells: HashMap<u64, Vec<T>>,
    len: usize,
}

impl<T: Positioned> MortonGrid<T> {
    /// `levels` sets the resolution: `2^levels` cells per axis. 1..=31.
    pub fn new(world: Rect, levels: u32) -> Self {
        assert!((1..=31).contains(&levels), "levels must be in 1..=31 (2×31 = 62 bits)");
        let n = 1u32 << levels;
        Self { world, levels, cells_per_axis: n, cw: world.width / n as f64, ch: world.height / n as f64, cells: HashMap::new(), len: 0 }
    }

    /// Pick the smallest `levels` whose cell is at least `target` wide on the
    /// larger axis — "I want cells ≈ the query radius".
    pub fn levels_for_cell_size(world: Rect, target: f64) -> u32 {
        let span = world.width.max(world.height);
        let mut levels = 1u32;
        while levels < 31 && span / (1u64 << (levels + 1)) as f64 >= target {
            levels += 1;
        }
        levels
    }

    #[inline]
    fn cell_of(&self, p: Point) -> (u32, u32) {
        (axis_index_n(p.x, self.world.x, self.cw, self.cells_per_axis), axis_index_n(p.y, self.world.y, self.ch, self.cells_per_axis))
    }

    fn cell_box(&self, ix: u32, iy: u32) -> Rect {
        Rect::new(self.world.x + ix as f64 * self.cw, self.world.y + iy as f64 * self.ch, self.cw, self.ch)
    }

    /// Bucket an item by its Morton cell. Out-of-world points are rejected.
    pub fn insert(&mut self, item: T) -> bool {
        let p = item.position();
        if !self.world.contains(p) {
            return false;
        }
        let (ix, iy) = self.cell_of(p);
        self.cells.entry(morton2(ix, iy)).or_default().push(item);
        self.len += 1;
        true
    }

    /// Bulk-insert from a parallel iterator (feature `parallel`) — see
    /// [`crate::MortonGrid3::extend_par`].
    #[cfg(feature = "parallel")]
    pub fn extend_par<I>(&mut self, items: I) -> usize
    where
        I: rayon::iter::IntoParallelIterator<Item = T>,
        T: Send,
    {
        use rayon::prelude::*;
        let (world, cw, ch, n) = (self.world, self.cw, self.ch, self.cells_per_axis);
        let coded: Vec<(u64, T)> = items
            .into_par_iter()
            .filter_map(|it| {
                let p = it.position();
                if !world.contains(p) {
                    return None;
                }
                Some((morton2(axis_index_n(p.x, world.x, cw, n), axis_index_n(p.y, world.y, ch, n)), it))
            })
            .collect();
        let added = coded.len();
        for (code, it) in coded {
            self.cells.entry(code).or_default().push(it);
        }
        self.len += added;
        added
    }

    /// Empty the grid, retaining the hash-map capacity.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.len = 0;
    }

    pub fn item_count(&self) -> usize { self.len }
    /// Measure what the cells hold — see [`crate::morton3::Occupancy`]. O(non-empty cells).
    pub fn occupancy(&self) -> crate::morton3::Occupancy {
        let mut max = 0usize;
        for b in self.cells.values() { max = max.max(b.len()); }
        let cells = self.cells.len();
        crate::morton3::Occupancy {
            cells,
            items: self.len,
            mean: if cells == 0 { 0.0 } else { self.len as f64 / cells as f64 },
            max,
        }
    }

    pub fn cell_count(&self) -> usize { self.cells.len() }
    pub fn levels(&self) -> u32 { self.levels }

    /// Visit each occupied cell's box and item count (for visualisation).
    pub fn visit_cells<F: FnMut(&Rect, usize)>(&self, mut f: F) {
        for (&code, bucket) in &self.cells {
            let (ix, iy) = demorton2(code);
            f(&self.cell_box(ix, iy), bucket.len());
        }
    }

    /// Same cull contract as [`crate::Tree::cull`]: every item inside `shape`.
    /// Visits only the cells overlapping the shape's bbox; a cell fully inside
    /// (`classify_box` → `In`) contributes all its points with no per-point
    /// test, a straddling cell runs the shared narrowphase (template / raster /
    /// exact). Analytic shapes ([`crate::Capsule`]) prune tightly via `classify_box`.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bb = shape.bounding_box();
        let ix0 = axis_index_n(bb.x, self.world.x, self.cw, self.cells_per_axis);
        let iy0 = axis_index_n(bb.y, self.world.y, self.ch, self.cells_per_axis);
        let ix1 = axis_index_n(bb.x_max(), self.world.x, self.cw, self.cells_per_axis);
        let iy1 = axis_index_n(bb.y_max(), self.world.y, self.ch, self.cells_per_axis);
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let bucket = match self.cells.get(&morton2(ix, iy)) {
                    Some(b) => b,
                    None => continue,
                };
                let cb = self.cell_box(ix, iy);
                let state = shape.classify_box(&cb).unwrap_or(if cb.intersects(&bb) { CellState::Maybe } else { CellState::Out });
                match state {
                    CellState::Out => {}
                    CellState::In => out.extend(bucket.iter()),
                    CellState::Maybe => collect_matching_items(bucket, shape, &bb, &mut out),
                }
            }
        }
        out
    }

    /// **Coarse-tier cull** (the 2D twin of [`MortonGrid3::cull_layered`]): skip
    /// empty regions of a large / sparse query in O(1) rather than probing every
    /// fine cell of the bbox. Derives a coarse occupancy set from the live cells
    /// (a coarse cell = `2^shift` fine cells per axis; its Morton code is the fine
    /// code with the low `2·shift` bits dropped), then visits only the fine cells
    /// inside **occupied** coarse blocks. Identical result to [`cull`](Self::cull);
    /// a win when the query bbox is large and the world has empty regions.
    pub fn cull_layered<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let shift = 2u32;
        if self.levels <= shift { return self.cull(shape); }
        let bb = shape.bounding_box();
        let ix0 = axis_index_n(bb.x, self.world.x, self.cw, self.cells_per_axis);
        let iy0 = axis_index_n(bb.y, self.world.y, self.ch, self.cells_per_axis);
        let ix1 = axis_index_n(bb.x_max(), self.world.x, self.cw, self.cells_per_axis);
        let iy1 = axis_index_n(bb.y_max(), self.world.y, self.ch, self.cells_per_axis);
        let mut coarse: std::collections::HashSet<u64> = std::collections::HashSet::with_capacity(self.cells.len());
        for &code in self.cells.keys() { coarse.insert(code >> (2 * shift)); }
        let mut out = Vec::new();
        let probe = |ix: u32, iy: u32, out: &mut Vec<&'a T>| {
            let bucket = match self.cells.get(&morton2(ix, iy)) { Some(b) => b, None => return };
            let cb = self.cell_box(ix, iy);
            let state = shape.classify_box(&cb).unwrap_or(if cb.intersects(&bb) { CellState::Maybe } else { CellState::Out });
            match state { CellState::Out => {}, CellState::In => out.extend(bucket.iter()), CellState::Maybe => collect_matching_items(bucket, shape, &bb, out) }
        };
        let (cx0, cx1) = (ix0 >> shift, ix1 >> shift);
        let (cy0, cy1) = (iy0 >> shift, iy1 >> shift);
        for cy in cy0..=cy1 { for cx in cx0..=cx1 {
            if !coarse.contains(&morton2(cx, cy)) { continue; }
            let (fx0, fx1) = ((cx << shift).max(ix0), ((cx << shift) + (1 << shift) - 1).min(ix1));
            let (fy0, fy1) = ((cy << shift).max(iy0), ((cy << shift) + (1 << shift) - 1).min(iy1));
            for iy in fy0..=fy1 { for ix in fx0..=fx1 { probe(ix, iy, &mut out); } }
        } }
        out
    }

    /// Batch cull — see [`crate::Tree3::cull_many`].
    pub fn cull_many<'a, S: Shape>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// The `k` nearest items to `q`, sorted ascending by distance — ring-by-ring
    /// shell expansion (the 2D analogue of [`crate::MortonGrid3::knn`]). Scans
    /// the query cell, then each surrounding Chebyshev ring, keeping a bounded
    /// max-heap of the k best, and stops once the nearest unscanned point (the
    /// distance from `q` to the scanned square's nearest face — an exact lower
    /// bound) exceeds the k-th best.
    pub fn knn<'a>(&'a self, q: Point, k: usize) -> Vec<(f64, &'a T)> {
        if k == 0 || self.len == 0 {
            return Vec::new();
        }
        let (cx, cy) = self.cell_of(q);
        let (cx, cy) = (cx as i64, cy as i64);
        let n = self.cells_per_axis as i64;
        let mut heap: std::collections::BinaryHeap<KnnEntry<T>> = std::collections::BinaryHeap::new();

        // **Per-axis expansion** — see [`crate::MortonGrid3::knn`], which this mirrors and
        // where it was measured. `levels` is one number for both axes, so a world that is not
        // square does not have square cells, and a ring that grows both radii together is
        // isotropic in CELL space and anisotropic in WORLD space: it has to over-scan the
        // wide axis to reach along the narrow one. Growing whichever axis is currently
        // narrowest in world units keeps the scanned region near-square. The stopping rule is
        // unchanged and still exact: the region is still a box, so the nearest unscanned point
        // is still at least `safe` away.
        let (mut rx, mut ry) = (0i64, 0i64);
        let cells = &self.cells;
        let scan = |heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>, xs: (i64, i64), ys: (i64, i64)| {
            for ix in xs.0.max(0)..=xs.1.min(n - 1) {
                for iy in ys.0.max(0)..=ys.1.min(n - 1) {
                    if let Some(bucket) = cells.get(&morton2(ix as u32, iy as u32)) {
                        for it in bucket {
                            knn_offer2(heap, k, it, q);
                        }
                    }
                }
            }
        };
        scan(&mut heap, (cx, cx), (cy, cy));

        loop {
            let xlo = self.world.x + (cx - rx) as f64 * self.cw;
            let xhi = self.world.x + (cx + rx + 1) as f64 * self.cw;
            let ylo = self.world.y + (cy - ry) as f64 * self.ch;
            let yhi = self.world.y + (cy + ry + 1) as f64 * self.ch;
            let safe = (q.x - xlo).min(xhi - q.x).min(q.y - ylo).min(yhi - q.y);
            if heap.len() >= k && safe > 0.0 && safe * safe >= knn_worst(&heap, k) {
                break;
            }
            if rx > n && ry > n {
                break;
            }
            if (rx as f64 + 0.5) * self.cw <= (ry as f64 + 0.5) * self.ch {
                rx += 1;
                scan(&mut heap, (cx - rx, cx - rx), (cy - ry, cy + ry));
                scan(&mut heap, (cx + rx, cx + rx), (cy - ry, cy + ry));
            } else {
                ry += 1;
                scan(&mut heap, (cx - rx, cx + rx), (cy - ry, cy - ry));
                scan(&mut heap, (cx - rx, cx + rx), (cy + ry, cy + ry));
            }
        }

        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    /// **DDA ray-cast** — the 2D **Amanatides–Woo** voxel walk over the uniform
    /// grid (the grid analogue of [`crate::Tree::raycast`], constant
    /// `tMax`/`tDelta` so each step is one add + compare). Collects items within
    /// `radius` of the ray segment, sorted by distance along the ray, + stats.
    /// Thin corridor (use `cull(&Capsule)` for the exact thick band).
    pub fn raycast(&self, origin: Point, dir: Point, max_t: f64, radius: f64) -> RaycastOut<'_, T> {
        let mut out = RaycastOut { hits: Vec::new(), leaves_visited: 0, items_tested: 0 };
        let m = (dir.x * dir.x + dir.y * dir.y).sqrt();
        if m == 0.0 {
            return out;
        }
        let (ux, uy) = (dir.x / m, dir.y / m);
        let (t0, t1) = match clip_ray_rect(origin, ux, uy, &self.world, max_t) {
            Some(v) => v,
            None => return out,
        };
        let r2 = radius * radius;
        let end = Point::new(origin.x + ux * max_t, origin.y + uy * max_t);
        let n = self.cells_per_axis as i64;
        let s = Point::new(origin.x + ux * t0, origin.y + uy * t0);
        let mut ix = axis_index_n(s.x, self.world.x, self.cw, self.cells_per_axis) as i64;
        let mut iy = axis_index_n(s.y, self.world.y, self.ch, self.cells_per_axis) as i64;
        let setup = |u: f64, lo: f64, cell: f64, i: i64, o: f64| -> (i64, f64, f64) {
            if u > 0.0 {
                (1, (lo + (i + 1) as f64 * cell - o) / u, cell / u)
            } else if u < 0.0 {
                (-1, (lo + i as f64 * cell - o) / u, cell / (-u))
            } else {
                (0, f64::INFINITY, f64::INFINITY)
            }
        };
        let (sx, mut tmx, tdx) = setup(ux, self.world.x, self.cw, ix, origin.x);
        let (sy, mut tmy, tdy) = setup(uy, self.world.y, self.ch, iy, origin.y);
        let t_end = max_t.min(t1);
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > n as usize * 2 + 16 {
                break;
            }
            if (0..n).contains(&ix) && (0..n).contains(&iy) {
                out.leaves_visited += 1;
                if let Some(bucket) = self.cells.get(&morton2(ix as u32, iy as u32)) {
                    for it in bucket {
                        out.items_tested += 1;
                        let p = it.position();
                        let (apx, apy) = (p.x - origin.x, p.y - origin.y);
                        let proj = apx * ux + apy * uy;
                        let d2 = if proj <= 0.0 {
                            apx * apx + apy * apy
                        } else if proj >= max_t {
                            let (bx, by) = (p.x - end.x, p.y - end.y);
                            bx * bx + by * by
                        } else {
                            (apx * apx + apy * apy) - proj * proj
                        };
                        if d2 <= r2 {
                            out.hits.push((proj.clamp(0.0, max_t), it));
                        }
                    }
                }
            }
            if tmx <= tmy {
                if tmx > t_end { break; }
                ix += sx;
                tmx += tdx;
                if ix < 0 || ix >= n { break; }
            } else {
                if tmy > t_end { break; }
                iy += sy;
                tmy += tdy;
                if iy < 0 || iy >= n { break; }
            }
        }
        out.hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// **First hit** along the ray with front-to-back early-exit — see
    /// [`crate::Tree::raycast_first`]. Same DDA walk, stops at the first cell
    /// beyond the best hit.
    pub fn raycast_first(&self, origin: Point, dir: Point, max_t: f64, radius: f64) -> Option<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y).sqrt();
        if m == 0.0 {
            return None;
        }
        let (ux, uy) = (dir.x / m, dir.y / m);
        let (t0, t1) = clip_ray_rect(origin, ux, uy, &self.world, max_t)?;
        let r2 = radius * radius;
        let end = Point::new(origin.x + ux * max_t, origin.y + uy * max_t);
        let n = self.cells_per_axis as i64;
        let s = Point::new(origin.x + ux * t0, origin.y + uy * t0);
        let mut ix = axis_index_n(s.x, self.world.x, self.cw, self.cells_per_axis) as i64;
        let mut iy = axis_index_n(s.y, self.world.y, self.ch, self.cells_per_axis) as i64;
        let setup = |u: f64, lo: f64, cell: f64, i: i64, o: f64| -> (i64, f64, f64) {
            if u > 0.0 { (1, (lo + (i + 1) as f64 * cell - o) / u, cell / u) }
            else if u < 0.0 { (-1, (lo + i as f64 * cell - o) / u, cell / (-u)) }
            else { (0, f64::INFINITY, f64::INFINITY) }
        };
        let (sx, mut tmx, tdx) = setup(ux, self.world.x, self.cw, ix, origin.x);
        let (sy, mut tmy, tdy) = setup(uy, self.world.y, self.ch, iy, origin.y);
        let t_end = max_t.min(t1);
        let mut best: Option<(f64, &T)> = None;
        let mut t_enter = t0;
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > n as usize * 2 + 16 {
                break;
            }
            if let Some((bt, _)) = best {
                if t_enter - radius > bt {
                    break;
                }
            }
            if (0..n).contains(&ix) && (0..n).contains(&iy) {
                if let Some(bucket) = self.cells.get(&morton2(ix as u32, iy as u32)) {
                    for it in bucket {
                        let p = it.position();
                        let (apx, apy) = (p.x - origin.x, p.y - origin.y);
                        let proj = apx * ux + apy * uy;
                        let d2 = if proj <= 0.0 {
                            apx * apx + apy * apy
                        } else if proj >= max_t {
                            let (bx, by) = (p.x - end.x, p.y - end.y);
                            bx * bx + by * by
                        } else {
                            (apx * apx + apy * apy) - proj * proj
                        };
                        if d2 <= r2 {
                            let t = proj.clamp(0.0, max_t);
                            if best.is_none_or(|(bt, _)| t < bt) {
                                best = Some((t, it));
                            }
                        }
                    }
                }
            }
            if tmx <= tmy {
                if tmx > t_end { break; }
                t_enter = tmx;
                ix += sx;
                tmx += tdx;
                if ix < 0 || ix >= n { break; }
            } else {
                if tmy > t_end { break; }
                t_enter = tmy;
                iy += sy;
                tmy += tdy;
                if iy < 0 || iy >= n { break; }
            }
        }
        best
    }
}

// ------------------------------------------------------------- serialization

const MORTON2_MAGIC: &[u8; 4] = b"VHM2";
const MORTON2_VERSION: u8 = 1;

impl<T: Positioned> MortonGrid<T> {
    /// Serialize the grid (world + resolution + occupied buckets) to `w`. The 2D
    /// analogue of [`crate::MortonGrid3::serialize`] — see it for the format
    /// notes (no arena; bucket order is the hash map's, so not canonical).
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(MORTON2_MAGIC)?;
        w_u8(w, MORTON2_VERSION)?;
        w_rect(w, &self.world)?;
        w_u32(w, self.levels)?;
        w_u32(w, self.cells.len() as u32)?;
        for (&code, bucket) in &self.cells {
            w_u64(w, code)?;
            w_u32(w, bucket.len() as u32)?;
            for it in bucket { write_item(w, it)?; }
        }
        Ok(())
    }

    /// Inverse of [`MortonGrid::serialize`]: rebuild an equivalent grid from `r`.
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MORTON2_MAGIC { return Err(corrupt("bad MortonGrid magic")); }
        if r_u8(r)? != MORTON2_VERSION { return Err(corrupt("unsupported MortonGrid version")); }
        let world = r_rect(r)?;
        let levels = r_u32(r)?;
        if !(1..=31).contains(&levels) { return Err(corrupt("MortonGrid levels out of 1..=31")); }
        let mut grid = Self::new(world, levels);
        let ncells = r_u32(r)? as usize;
        grid.cells.reserve(ncells);
        for _ in 0..ncells {
            let code = r_u64(r)?;
            let n = r_u32(r)? as usize;
            let mut bucket = Vec::with_capacity(n);
            for _ in 0..n { bucket.push(read_item(r)?); }
            grid.len += bucket.len();
            grid.cells.insert(code, bucket);
        }
        Ok(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct P(Point);
    impl Positioned for P { fn position(&self) -> Point { self.0 } }

    struct Disc { cx: f64, cy: f64, r: f64 }
    impl Shape for Disc {
        fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
        fn contains_point(&self, p: Point) -> bool { let (dx, dy) = (p.x - self.cx, p.y - self.cy); dx * dx + dy * dy <= self.r * self.r }
    }

    fn rng_pts(n: usize, seed: u64) -> Vec<P> {
        let mut x = seed | 1;
        (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let a = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64; x ^= x << 13; x ^= x >> 7; x ^= x << 17; let b = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64; P(Point::new(a * 256.0, b * 256.0)) }).collect()
    }

    #[test]
    fn morton2_cull_layered_matches_cull_and_brute() {
        // The 2D coarse-tier cull == plain cull == brute, incl. a sparse world
        // (clumps in a big void) with a big query where the coarse skip fires.
        let mut x = 0x2D_1A7EDu64 & 0xFFFF_FFFF;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000).map(|i| { let (cx, cy) = if i % 2 == 0 { (300.0, 300.0) } else { (1700.0, 1500.0) };
            P(Point::new(cx + (rng() - 0.5) * 300.0, cy + (rng() - 0.5) * 300.0)) }).collect();
        let world = Rect::new(0.0, 0.0, 2048.0, 2048.0);
        for levels in [5u32, 7, 8] {
            let mut grid = MortonGrid::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            for (cx, cy, r) in [(1024.0, 1024.0, 1200.0), (300.0, 300.0, 200.0), (1700.0, 1500.0, 400.0), (1000.0, 1000.0, 2000.0)] {
                let s = Disc { cx, cy, r };
                let key = |v: &[&P]| { let mut k: Vec<(u64, u64)> = v.iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits())).collect(); k.sort(); k };
                let plain = key(&grid.cull(&s)); let layered = key(&grid.cull_layered(&s));
                let brute = { let mut b: Vec<(u64, u64)> = pts.iter().filter(|p| { let dx = p.0.x - cx; let dy = p.0.y - cy; dx * dx + dy * dy <= r * r }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits())).collect(); b.sort(); b };
                assert_eq!(layered, plain, "cull_layered != cull (levels={levels}) disc ({cx},{cy}) r={r}");
                assert_eq!(layered, brute, "cull_layered != brute (levels={levels}) disc ({cx},{cy}) r={r}");
            }
        }
    }

    #[test]
    fn morton2_serialize_roundtrip() {
        use std::io::{Cursor, Read, Write};
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point }
        impl Positioned for M { fn position(&self) -> Point { self.p } }
        let mut x = 0x0117_5E12u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Rect::new(0.0, 0.0, 256.0, 256.0);
        let mut grid = MortonGrid::<M>::new(world, 5);
        for id in 0..4000u32 {
            grid.insert(M { id, p: Point::new(rng() * 256.0, rng() * 256.0) });
        }
        let mut buf = Vec::new();
        grid.serialize(&mut buf, |w, it| {
            w.write_all(&it.id.to_le_bytes())?;
            w.write_all(&it.p.x.to_le_bytes())?; w.write_all(&it.p.y.to_le_bytes())
        }).unwrap();
        let loaded = MortonGrid::<M>::deserialize(&mut Cursor::new(&buf), |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?; let mut b = [0u8; 8];
            let id = u32::from_le_bytes(a);
            r.read_exact(&mut b)?; let px = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let py = f64::from_le_bytes(b);
            Ok(M { id, p: Point::new(px, py) })
        }).unwrap();
        assert_eq!(loaded.item_count(), grid.item_count(), "items");
        assert_eq!(loaded.cell_count(), grid.cell_count(), "cells");
        assert_eq!(loaded.levels(), grid.levels(), "levels");
        for (cx, cy, r) in [(128.0, 128.0, 30.0), (60.0, 200.0, 50.0), (10.0, 10.0, 80.0)] {
            let s = Disc { cx, cy, r };
            let mut a: Vec<u32> = grid.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = loaded.cull(&s).iter().map(|m| m.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "cull differs after round-trip ({cx},{cy}) r={r}");
        }
        assert!(MortonGrid::<M>::deserialize(&mut Cursor::new(&b"XXXXX"[..]), |_| unreachable!()).is_err());
    }

    #[test]
    fn morton2_encode_decode() {
        assert_eq!(morton2(0, 0), 0);
        assert_eq!(morton2(1, 0), 1);
        assert_eq!(morton2(0, 1), 2);
        assert_eq!(morton2(1, 1), 3);
        assert_eq!(morton2(2, 0), 4);
        for (x, y) in [(0u32, 0u32), (1, 2), (1023, 7), (65535, 65535), (12345, 54321)] {
            assert_eq!(demorton2(morton2(x, y)), (x, y));
        }
    }

    #[test]
    fn morton2_cull_matches_brute() {
        let pts = rng_pts(4000, 0xC0FE);
        let world = Rect::new(0.0, 0.0, 256.0, 256.0);
        for levels in [3u32, 5, 6] {
            let mut grid = MortonGrid::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            assert_eq!(grid.item_count(), pts.len());
            for (cx, cy, r) in [(128.0, 128.0, 40.0), (40.0, 40.0, 60.0), (250.0, 250.0, 30.0), (0.0, 0.0, 100.0)] {
                let mut want: Vec<(u64, u64)> = pts.iter().filter(|p| { let (dx, dy) = (p.0.x - cx, p.0.y - cy); dx * dx + dy * dy <= r * r }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits())).collect();
                let mut got: Vec<(u64, u64)> = grid.cull(&Disc { cx, cy, r }).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits())).collect();
                want.sort();
                got.sort();
                assert_eq!(want, got, "cull != brute (levels={levels}) disc ({cx},{cy}) r={r}");
            }
        }
    }

    #[test]
    fn morton2_knn_matches_brute() {
        let pts = rng_pts(3000, 0x51ED);
        let world = Rect::new(0.0, 0.0, 256.0, 256.0);
        for levels in [3u32, 5, 6] {
            let mut grid = MortonGrid::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            for (qx, qy) in [(128.0, 128.0), (10.0, 250.0), (0.0, 0.0), (255.0, 255.0)] {
                for k in [1usize, 8, 32] {
                    let mut brute: Vec<f64> = pts.iter().map(|p| { let (dx, dy) = (p.0.x - qx, p.0.y - qy); (dx * dx + dy * dy).sqrt() }).collect();
                    brute.sort_by(|a, b| a.total_cmp(b));
                    brute.truncate(k);
                    let got: Vec<f64> = grid.knn(Point::new(qx, qy), k).into_iter().map(|(d, _)| d).collect();
                    assert_eq!(got.len(), brute.len());
                    for (a, b) in got.iter().zip(brute.iter()) {
                        assert!((a - b).abs() < 1e-9, "knn dist != brute (levels={levels}, k={k})");
                    }
                }
            }
        }
    }

    #[test]
    fn morton2_raycast_subset_and_first() {
        use crate::Capsule;
        let pts = rng_pts(5000, 0x3ABC);
        let mut grid = MortonGrid::<P>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 5);
        for p in &pts { grid.insert(*p); }
        for (o, d, mt, r) in [
            (Point::new(8.0, 8.0), Point::new(1.0, 0.7), 360.0, 6.0),
            (Point::new(250.0, 10.0), Point::new(-1.0, 1.0), 360.0, 12.0),
            (Point::new(0.0, 128.0), Point::new(1.0, 0.0), 256.0, 20.0),
        ] {
            let dda = grid.raycast(o, d, mt, r);
            for w in dda.hits.windows(2) { assert!(w[0].0 <= w[1].0, "not sorted"); }
            let m = (d.x * d.x + d.y * d.y).sqrt();
            let end = Point::new(o.x + d.x / m * mt, o.y + d.y / m * mt);
            let cull: std::collections::HashSet<(u64, u64)> = grid.cull(&Capsule::new(o, end, r)).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits())).collect();
            for (_, p) in &dda.hits {
                assert!(cull.contains(&(p.0.x.to_bits(), p.0.y.to_bits())), "DDA hit not in exact capsule cull");
            }
            // first-hit == raycast nearest
            match (grid.raycast_first(o, d, mt, r), dda.hits.first()) {
                (None, None) => {}
                (Some((t1, _)), Some(&(t0, _))) => assert!((t1 - t0).abs() < 1e-9),
                _ => panic!("raycast_first / raycast nearest disagree"),
            }
        }
    }
}
