//! Morton / Z-order linear grid — a **pointer-free** 3D index, the fourth
//! structure in the 3D comparison (`Tree3` binary, `Octree3` 8-way, projection,
//! and this). Where the trees navigate parent→child pointers, a linear grid
//! quantises each point to an integer cell `(ix, iy, iz)`, interleaves the
//! cell's bits into a single 64-bit **Morton code** (the Z-order curve), and
//! buckets points by that code in a hash map. The spatial hierarchy is implicit
//! in the code's bit layout — no nodes, no splits, no merges. This is the
//! "spatial hash" most game engines reach for.
//!
//! Fixed resolution (all cells at one depth — the simplest linear octree): the
//! world is divided into `2^levels` cells per axis. A cull visits only the cells
//! overlapping the query's bounding box, with the same green / white / yellow
//! short-circuit per cell (a cell fully inside the shape contributes all its
//! points with no per-point test; a straddling cell runs the exact test, using
//! the shape's [`VoxelRaster`](crate::VoxelRaster) when present).
//!
//! Trade-off vs the trees: O(1) cell lookup and trivial build (no rebalancing),
//! but a *single* resolution — too coarse and each cell holds many points (more
//! exact tests), too fine and a large query touches many empty cells. The
//! trees adapt their leaf size to local density; the grid does not. The
//! comparison in `tree3d_bench` shows where each wins.

use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::serde_io::{corrupt, r_aabb, r_u32, r_u64, r_u8, w_aabb, w_u32, w_u64, w_u8};
use crate::template::CellState;
use crate::tree::RaycastOut;
use crate::tree3::{knn_offer, knn_worst, Aabb, KnnEntry, Point3, Positioned3, Shape3};

/// Clip the ray `o + t·u` to the world AABB → the `[t_enter, t_exit]` parameter
/// range it spends inside (capped at `max_t`), or `None` if it misses. Slab test.
fn clip_ray_aabb(o: Point3, ux: f64, uy: f64, uz: f64, w: &Aabb, max_t: f64) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0_f64, max_t);
    for (oa, ua, lo, hi) in [(o.x, ux, w.x, w.x_max()), (o.y, uy, w.y, w.y_max()), (o.z, uz, w.z, w.z_max())] {
        if ua == 0.0 {
            if oa < lo || oa > hi {
                return None; // parallel to this slab and outside it
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

/// Interleave the low 21 bits of `n` with two zero bits between each
/// (`abc` → `a..b..c`), so three of these OR'd together (shifted 0/1/2) pack
/// `(x, y, z)` into one Z-order code. 21 bits/axis × 3 = 63 bits fits a u64.
#[inline]
fn part1by2(mut n: u64) -> u64 {
    n &= 0x1f_ffff;
    n = (n | (n << 32)) & 0x1f00000000ffff;
    n = (n | (n << 16)) & 0x1f0000ff0000ff;
    n = (n | (n << 8)) & 0x100f00f00f00f00f;
    n = (n | (n << 4)) & 0x10c30c30c30c30c3;
    n = (n | (n << 2)) & 0x1249249249249249;
    n
}

/// The 3D Morton (Z-order) code of integer cell `(x, y, z)`. Each axis must fit
/// in 21 bits (`levels <= 21`). Bit `i` of `x` lands at code position `3i`,
/// `y` at `3i+1`, `z` at `3i+2`.
#[inline]
pub fn morton3(x: u32, y: u32, z: u32) -> u64 {
    part1by2(x as u64) | (part1by2(y as u64) << 1) | (part1by2(z as u64) << 2)
}

/// Inverse of [`part1by2`]: gather every third bit back down.
#[inline]
fn compact1by2(mut n: u64) -> u64 {
    n &= 0x1249249249249249;
    n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3;
    n = (n ^ (n >> 4)) & 0x100f00f00f00f00f;
    n = (n ^ (n >> 8)) & 0x1f0000ff0000ff;
    n = (n ^ (n >> 16)) & 0x1f00000000ffff;
    n = (n ^ (n >> 32)) & 0x1f_ffff;
    n
}

/// Decode a Morton code back to its integer cell `(x, y, z)` — inverse of
/// [`morton3`].
#[inline]
pub fn demorton3(code: u64) -> (u32, u32, u32) {
    (compact1by2(code) as u32, compact1by2(code >> 1) as u32, compact1by2(code >> 2) as u32)
}

/// Clamp a world coordinate to a cell index in `0..n` along one axis. Free fn
/// (not a method) so the parallel bulk build can call it without borrowing the
/// grid; `MortonGrid3::axis_index` delegates here.
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

pub struct MortonGrid3<T: Positioned3> {
    world: Aabb,
    levels: u32,
    cells_per_axis: u32,
    cw: f64,
    ch: f64,
    cd: f64,
    cells: HashMap<u64, Vec<T>>,
    len: usize,
}


/// What an in-place [`MortonGrid3::update`] did — the grid's counterpart to `Crossing`.
///
/// `Stayed` is the case worth building for: the item moved but not out of its cell, so the
/// grid did nothing at all beyond the lookup.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Crossed {
    /// Still in the same cell. Nothing was re-bucketed.
    Stayed,
    /// Re-bucketed into a different cell.
    Moved,
    /// The new position is outside the world box, so the item was dropped — the grid holds
    /// only what is inside it, exactly as `insert` does.
    Left,
    /// Nothing in the old cell matched the predicate. Either the old position was wrong or the
    /// item was never here.
    Missing,
}

impl Crossed {
    /// `true` only for [`Crossed::Missing`] — the one outcome that means the caller's idea of
    /// where the item was is wrong, and so the only one that needs recovering from.
    pub fn is_missing(self) -> bool { self == Crossed::Missing }
}

/// Set a point-like item's position — for [`MortonGrid3::relocate`] and
/// [`crate::MortonGrid::relocate`], which are conveniences over the predicate form.
pub trait SetPosition3: Positioned3 {
    fn set_position(&mut self, p: Point3);
}

#[inline]
fn morton3_of((x, y, z): (u32, u32, u32)) -> u64 { morton3(x, y, z) }

// ---------------------------------------------------------------- cell-visit counter

/// Cells looked up by grid queries on this thread since the last reset — see the `grid-stats`
/// feature. **Global and per-thread**, not per-grid: a tuning diagnostic, not instrumentation
/// to build on. A rayon-parallel query spreads its counts over the worker threads, and this
/// reports only the calling thread's share.
///
/// It exists because the repo's clock-free method has a structural blind spot. Wrapping the
/// query counts `classify_aabb`/`contains_point`; wrapping the ITEM counts `position()`. Both
/// measure work done on POINTS, so a grid query that crosses a hundred empty cells does no
/// countable work at all — which is how three separate speed-ups this week landed with the
/// point count essentially unchanged (`docs/MEASURING.md` § 8b).
///
/// Genuinely zero-cost when the feature is off: neither the counter nor the increments exist.
#[cfg(feature = "grid-stats")]
mod visits {
    use std::cell::Cell;
    thread_local! { pub static N: Cell<u64> = const { Cell::new(0) }; }
}

/// Cells visited by grid queries on this thread since [`reset_cell_visits`].
#[cfg(feature = "grid-stats")]
pub fn cell_visits() -> u64 { visits::N.with(|c| c.get()) }

/// Zero the counter and return what it held.
#[cfg(feature = "grid-stats")]
pub fn reset_cell_visits() -> u64 { visits::N.with(|c| c.replace(0)) }

#[cfg(feature = "grid-stats")]
#[inline]
pub(crate) fn count_cell() { visits::N.with(|c| c.set(c.get() + 1)); }

#[cfg(not(feature = "grid-stats"))]
#[inline(always)]
pub(crate) fn count_cell() {}

/// One cell lookup, counted. Every query path goes through this rather than touching the map
/// directly, so the counter cannot drift out of step with the traversals it measures.
#[inline]
pub(crate) fn bucket_of<T>(cells: &std::collections::HashMap<u64, Vec<T>>, key: u64) -> Option<&Vec<T>> {
    count_cell();
    cells.get(&key)
}

/// What a grid's cells actually hold — the number the query cost turns on.
///
/// A uniform grid is tuned by one knob, `levels`, and the natural way to set it is from a
/// length: "make the cell about the size of my query radius". Measured across four workloads
/// in this repo, length is the wrong variable and **occupancy** is the right one: what decides
/// the cost of a query is how many points it has to test per cell it visits, and a grid whose
/// cells hold 0.08 points spends all its time crossing empty ones (23x slower than the same
/// grid one level coarser — `examples/morton_knn_axis_bench`).
///
/// This is deliberately a **diagnostic and not a predictor**. There is no
/// `levels_for_occupancy` alongside `levels_for_cell_size`, because occupancy alone does not
/// determine the winner: re-picking the level of a cube-padded point-cloud grid to restore its
/// original points-per-cell still left it 1.3x slower than the anisotropic grid it replaced
/// (`docs/POINTCLOUD.md`). Cell *shape* matters too, and it should follow where the points are.
/// So: measure your grid, do not compute it.
///
/// `mean` is over **non-empty** cells, which is the only kind a traversal can pay for, and it
/// is the number to read first: far above the `k` you ask for means the grid is too coarse for
/// this data, far below means queries are spending their time crossing empty cells.
///
/// For skew, read `mean` and `cells` together rather than `max / mean`. That ratio looks like
/// the obvious skew signal and **degenerates exactly when skew is worst**: once the data
/// collapses into a single cell, `max == mean` and it reads 1.0, the same as a perfectly
/// uniform grid. Measured: 2000 points spread over a world gave `cells: 499, mean: 4.0,
/// max: 10` (ratio 2.5), and the same 2000 points in one corner gave `cells: 1, mean: 2000,
/// max: 2000` (ratio 1.0). The collapse is visible in `cells` and in `mean`, not in the
/// ratio. A `mean` in the hundreds is the signature that sends you to `KdTree3` or
/// `LinearOctree3` instead.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Occupancy {
    /// Non-empty cells. Empty ones are not stored (the backing store is a sparse hash) and
    /// cost nothing to hold — but they still cost a lookup each to cross.
    pub cells: usize,
    /// Items in the grid.
    pub items: usize,
    /// Items per non-empty cell. Aim for roughly the `k` you ask k-NN for.
    pub mean: f64,
    /// The fullest cell. `max / mean` is how skewed the data is.
    pub max: usize,
}

impl<T: Positioned3> MortonGrid3<T> {
    /// `levels` sets the resolution: `2^levels` cells per axis (so cell size is
    /// `world_dim / 2^levels`). 1..=21.
    pub fn new(world: Aabb, levels: u32) -> Self {
        assert!((1..=21).contains(&levels), "levels must be in 1..=21 (3×21 = 63 bits)");
        let n = 1u32 << levels;
        Self {
            world,
            levels,
            cells_per_axis: n,
            cw: world.w / n as f64,
            ch: world.h / n as f64,
            cd: world.d / n as f64,
            cells: HashMap::new(),
            len: 0,
        }
    }

    /// Pick the smallest `levels` whose cell is at least `target` units wide on
    /// the largest axis — a convenience for "I want cells ≈ the query radius".
    pub fn levels_for_cell_size(world: Aabb, target: f64) -> u32 {
        let span = world.w.max(world.h).max(world.d);
        let mut levels = 1u32;
        while levels < 21 && span / (1u64 << (levels + 1)) as f64 >= target {
            levels += 1;
        }
        levels
    }

    #[inline]
    fn axis_index(&self, v: f64, lo: f64, cell: f64) -> u32 {
        axis_index_n(v, lo, cell, self.cells_per_axis)
    }

    #[inline]
    fn cell_of(&self, p: Point3) -> (u32, u32, u32) {
        (
            self.axis_index(p.x, self.world.x, self.cw),
            self.axis_index(p.y, self.world.y, self.ch),
            self.axis_index(p.z, self.world.z, self.cd),
        )
    }

    fn cell_box(&self, ix: u32, iy: u32, iz: u32) -> Aabb {
        Aabb::new(
            self.world.x + ix as f64 * self.cw,
            self.world.y + iy as f64 * self.ch,
            self.world.z + iz as f64 * self.cd,
            self.cw,
            self.ch,
            self.cd,
        )
    }

    /// Bucket an item by its Morton cell. Out-of-world points are rejected
    /// (returns `false`), matching `Tree3`/`Octree3` insert semantics.
    pub fn insert(&mut self, item: T) -> bool {
        let p = item.position();
        if !self.world.contains(p) {
            return false;
        }
        let (ix, iy, iz) = self.cell_of(p);
        self.cells.entry(morton3(ix, iy, iz)).or_default().push(item);
        self.len += 1;
        true
    }

    /// **Move an item that is already in the grid, in place.** `old` is where it was, so the
    /// item can be found without scanning anything but its own cell; `predicate` picks it out
    /// of that cell, `mutate` changes it. Returns `Crossed::Missing` if no item in that cell
    /// matched.
    ///
    /// This is the grid's answer to the tree's `update_ref`, and it exists because "grids are
    /// rebuild-only" was a property of the API rather than of grids. A uniform grid has no
    /// handles to maintain and no structure to rebalance: if the item stays in its cell there
    /// is *nothing to do at all*, and if it leaves, the whole cost is one `swap_remove` and one
    /// push. What it cannot do is find the item without help, which is why the caller passes
    /// the old position — the same bargain `Tree3::update` makes with its predicate.
    ///
    /// Cost is O(occupancy of the old cell), so it is only a good deal on a grid whose cells
    /// hold few items — see [`Occupancy`], and note this is another reason the tuning knob
    /// matters. On a grid holding thousands per cell, `clear` + refill is the better call.
    pub fn update<P, M>(&mut self, old: Point3, predicate: P, mutate: M) -> Crossed
    where
        P: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        if !self.world.contains(old) { return Crossed::Missing; }
        let from = morton3_of(self.cell_of(old));
        // The new cell is computed before the mutable borrow is released, so `cell_of` (which
        // reads `self`) has to happen with the borrow already ended — hence the two phases.
        let (idx, p) = {
            let Some(bucket) = self.cells.get_mut(&from) else { return Crossed::Missing };
            let Some(idx) = bucket.iter().position(&predicate) else { return Crossed::Missing };
            mutate(&mut bucket[idx]);
            (idx, bucket[idx].position())
        };
        if !self.world.contains(p) {
            // Left the world: the grid holds only what is inside it, as `insert` promises.
            let bucket = self.cells.get_mut(&from).expect("bucket was just borrowed");
            bucket.swap_remove(idx);
            if bucket.is_empty() { self.cells.remove(&from); }
            self.len -= 1;
            return Crossed::Left;
        }
        let to = morton3_of(self.cell_of(p));
        if to == from { return Crossed::Stayed; }
        let bucket = self.cells.get_mut(&from).expect("bucket was just borrowed");
        let item = bucket.swap_remove(idx);
        if bucket.is_empty() { self.cells.remove(&from); }
        self.cells.entry(to).or_default().push(item);
        Crossed::Moved
    }

    /// **Remove an item, given where it was.** The companion to
    /// [`MortonGrid3::update`]: same bargain (the caller supplies the old position so only one
    /// cell is scanned), same cost, and the emptied bucket is dropped so the cell set stays
    /// exactly what a rebuild would produce.
    ///
    /// Returns the item, or `None` if nothing in that cell matched.
    pub fn remove<P: Fn(&T) -> bool>(&mut self, old: Point3, predicate: P) -> Option<T> {
        if !self.world.contains(old) { return None; }
        let key = morton3_of(self.cell_of(old));
        let bucket = self.cells.get_mut(&key)?;
        let idx = bucket.iter().position(&predicate)?;
        let item = bucket.swap_remove(idx);
        if bucket.is_empty() { self.cells.remove(&key); }
        self.len -= 1;
        Some(item)
    }

    /// Convenience over [`MortonGrid3::update`] for the common case of moving a point-like item
    /// whose identity is decided by comparing positions.
    pub fn relocate(&mut self, old: Point3, new: Point3) -> Crossed
    where
        T: SetPosition3,
    {
        self.update(old, |it| it.position() == old, |it| it.set_position(new))
    }

    /// Bulk-insert from a parallel iterator (feature `parallel`): each item's
    /// Morton cell is computed on rayon's thread pool, then the `(code, item)`
    /// pairs are grouped into buckets serially. The grouping is the serial tail
    /// (Amdahl), so the speedup is modest — the parallel part is the per-item
    /// quantise-and-encode, which pays for large `N`. Out-of-world items are
    /// skipped (as in [`MortonGrid3::insert`]); returns the count inserted.
    /// Pair with [`MortonGrid3::clear`] for a cheap parallel rebuild-per-frame.
    #[cfg(feature = "parallel")]
    pub fn extend_par<I>(&mut self, items: I) -> usize
    where
        I: rayon::iter::IntoParallelIterator<Item = T>,
        T: Send,
    {
        use rayon::prelude::*;
        let (world, cw, ch, cd, n) = (self.world, self.cw, self.ch, self.cd, self.cells_per_axis);
        let coded: Vec<(u64, T)> = items
            .into_par_iter()
            .filter_map(|it| {
                let p = it.position();
                if !world.contains(p) {
                    return None;
                }
                let ix = axis_index_n(p.x, world.x, cw, n);
                let iy = axis_index_n(p.y, world.y, ch, n);
                let iz = axis_index_n(p.z, world.z, cd, n);
                Some((morton3(ix, iy, iz), it))
            })
            .collect();
        let added = coded.len();
        for (code, it) in coded {
            self.cells.entry(code).or_default().push(it);
        }
        self.len += added;
        added
    }

    /// Empty the grid, **retaining the hash-map capacity** — the table is
    /// cleared, not freed. This is the cheap path for the rebuild-every-frame
    /// pattern Morton is built for: `grid.clear()` then re-`insert` avoids
    /// reallocating the bucket table each frame.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.len = 0;
    }

    /// Measure what the cells hold — see [`Occupancy`]. O(non-empty cells); call it when
    /// tuning, not per frame.
    pub fn occupancy(&self) -> Occupancy {
        let mut max = 0usize;
        for b in self.cells.values() { max = max.max(b.len()); }
        let cells = self.cells.len();
        Occupancy {
            cells,
            items: self.len,
            mean: if cells == 0 { 0.0 } else { self.len as f64 / cells as f64 },
            max,
        }
    }

    pub fn item_count(&self) -> usize { self.len }
    pub fn cell_count(&self) -> usize { self.cells.len() }
    pub fn levels(&self) -> u32 { self.levels }

    /// Visit each **occupied** cell's box and item count (for visualisation /
    /// debugging — the grid analogue of a tree's `visit_leaves`). Order is the
    /// hash map's, i.e. unspecified.
    pub fn visit_cells<F: FnMut(&Aabb, usize)>(&self, mut f: F) {
        for (&code, bucket) in &self.cells {
            let (ix, iy, iz) = demorton3(code);
            f(&self.cell_box(ix, iy, iz), bucket.len());
        }
    }

    /// Same cull contract as [`crate::Tree3::cull`]: every item inside `shape`.
    /// Visits only the cells overlapping the shape's bounding box; a cell fully
    /// inside short-circuits (all points, no test), a straddling cell runs the
    /// exact test (raster lookup when available).
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bb = shape.bounding_box();
        let ix0 = self.axis_index(bb.x, self.world.x, self.cw);
        let iy0 = self.axis_index(bb.y, self.world.y, self.ch);
        let iz0 = self.axis_index(bb.z, self.world.z, self.cd);
        let ix1 = self.axis_index(bb.x_max(), self.world.x, self.cw);
        let iy1 = self.axis_index(bb.y_max(), self.world.y, self.ch);
        let iz1 = self.axis_index(bb.z_max(), self.world.z, self.cd);
        let raster = shape.voxel_raster();
        for iz in iz0..=iz1 {
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    let bucket = match bucket_of(&self.cells, morton3(ix, iy, iz)) {
                        Some(b) => b,
                        None => continue,
                    };
                    match shape.classify_aabb(&self.cell_box(ix, iy, iz)) {
                        CellState::Out => {}
                        CellState::In => out.extend(bucket.iter()),
                        CellState::Maybe => {
                            for it in bucket {
                                let p = it.position();
                                match raster.map(|g| g.cell_at_world(p)) {
                                    Some(CellState::In) => out.push(it),
                                    Some(CellState::Out) => {}
                                    _ => if shape.contains_point(p) { out.push(it); },
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// **Coarse-tier cull** — the multi-level linear-octree query: skip whole
    /// empty regions of a large / sparse query in O(1) instead of probing every
    /// fine cell of the bounding box. It derives a coarse occupancy set from the
    /// live cells (a coarse cell = `2^shift` fine cells per axis; its Morton code
    /// is just the fine code with the low `3·shift` bits dropped — the Z-order
    /// prefix *is* the hierarchy), then visits only the fine cells inside
    /// **occupied** coarse blocks. Result is identical to [`cull`](Self::cull).
    ///
    /// Win profile (mirrors the on-disk layered study): a big query over sparse
    /// space skips the void cheaply; for a dense or small query plain `cull` is
    /// faster (no coarse pass), so reach for this only when the bbox is large and
    /// the world has empty regions.
    pub fn cull_layered<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let shift = 2u32; // coarse cell = 4×4×4 = 64 fine cells
        if self.levels <= shift { return self.cull(shape); }
        let bb = shape.bounding_box();
        let ix0 = self.axis_index(bb.x, self.world.x, self.cw); let ix1 = self.axis_index(bb.x_max(), self.world.x, self.cw);
        let iy0 = self.axis_index(bb.y, self.world.y, self.ch); let iy1 = self.axis_index(bb.y_max(), self.world.y, self.ch);
        let iz0 = self.axis_index(bb.z, self.world.z, self.cd); let iz1 = self.axis_index(bb.z_max(), self.world.z, self.cd);
        // coarse occupancy from the live fine codes (once)
        let mut coarse: std::collections::HashSet<u64> = std::collections::HashSet::with_capacity(self.cells.len());
        for &code in self.cells.keys() { coarse.insert(code >> (3 * shift)); }
        let raster = shape.voxel_raster();
        let mut out = Vec::new();
        let probe = |ix: u32, iy: u32, iz: u32, out: &mut Vec<&'a T>| {
            let bucket = match bucket_of(&self.cells, morton3(ix, iy, iz)) { Some(b) => b, None => return };
            match shape.classify_aabb(&self.cell_box(ix, iy, iz)) {
                CellState::Out => {}
                CellState::In => out.extend(bucket.iter()),
                CellState::Maybe => { for it in bucket { let p = it.position(); match raster.map(|g| g.cell_at_world(p)) { Some(CellState::In) => out.push(it), Some(CellState::Out) => {}, _ => if shape.contains_point(p) { out.push(it); } } } }
            }
        };
        let (cx0, cx1) = (ix0 >> shift, ix1 >> shift);
        let (cy0, cy1) = (iy0 >> shift, iy1 >> shift);
        let (cz0, cz1) = (iz0 >> shift, iz1 >> shift);
        for cz in cz0..=cz1 { for cy in cy0..=cy1 { for cx in cx0..=cx1 {
            if !coarse.contains(&morton3(cx, cy, cz)) { continue; } // whole coarse block empty → skip 2^(3·shift) fine cells
            let (fx0, fx1) = ((cx << shift).max(ix0), ((cx << shift) + (1 << shift) - 1).min(ix1));
            let (fy0, fy1) = ((cy << shift).max(iy0), ((cy << shift) + (1 << shift) - 1).min(iy1));
            let (fz0, fz1) = ((cz << shift).max(iz0), ((cz << shift) + (1 << shift) - 1).min(iz1));
            for iz in fz0..=fz1 { for iy in fy0..=fy1 { for ix in fx0..=fx1 { probe(ix, iy, iz, &mut out); } } }
        } } }
        out
    }

    /// Batch cull — see [`crate::Tree3::cull_many`].
    pub fn cull_many<'a, S: Shape3>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// **DDA ray-cast** over the Z-order grid — the 3D **Amanatides–Woo** voxel
    /// traversal (the grid analogue of [`crate::Tree::raycast`]). Walks the cells
    /// the centre ray crosses front-to-back; on a *uniform* grid `tMax`/`tDelta`
    /// are constant, so each step is one add + compare — no neighbour-finding, no
    /// per-cell recompute. Collects items within `radius` of the ray segment,
    /// sorted by distance along the ray, plus stats (`leaves_visited` = cells
    /// visited, `items_tested`). The ray is clipped to the world first.
    ///
    /// Thin corridor: items within `radius` in cells the centre line misses are
    /// not seen — use `cull(&Segment3)` for the exact thick band. `radius == 0`
    /// is the pure line.
    pub fn raycast(&self, origin: Point3, dir: Point3, max_t: f64, radius: f64) -> RaycastOut<'_, T> {
        let mut out = RaycastOut { hits: Vec::new(), leaves_visited: 0, items_tested: 0 };
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if m == 0.0 {
            return out;
        }
        let (ux, uy, uz) = (dir.x / m, dir.y / m, dir.z / m);
        let (t0, t1) = match clip_ray_aabb(origin, ux, uy, uz, &self.world, max_t) {
            Some(v) => v,
            None => return out, // ray misses the world
        };
        let r2 = radius * radius;
        let end = Point3::new(origin.x + ux * max_t, origin.y + uy * max_t, origin.z + uz * max_t);
        let n = self.cells_per_axis as i64;
        // Start cell = the entry point's cell.
        let s = Point3::new(origin.x + ux * t0, origin.y + uy * t0, origin.z + uz * t0);
        let mut ix = axis_index_n(s.x, self.world.x, self.cw, self.cells_per_axis) as i64;
        let mut iy = axis_index_n(s.y, self.world.y, self.ch, self.cells_per_axis) as i64;
        let mut iz = axis_index_n(s.z, self.world.z, self.cd, self.cells_per_axis) as i64;
        // (step, tMax-from-origin, tDelta) per axis on the uniform grid.
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
        let (sz, mut tmz, tdz) = setup(uz, self.world.z, self.cd, iz, origin.z);
        let t_end = max_t.min(t1);
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > n as usize * 3 + 16 {
                break;
            }
            if (0..n).contains(&ix) && (0..n).contains(&iy) && (0..n).contains(&iz) {
                out.leaves_visited += 1;
                if let Some(bucket) = bucket_of(&self.cells, morton3(ix as u32, iy as u32, iz as u32)) {
                    for it in bucket {
                        out.items_tested += 1;
                        let p = it.position();
                        let (apx, apy, apz) = (p.x - origin.x, p.y - origin.y, p.z - origin.z);
                        let proj = apx * ux + apy * uy + apz * uz;
                        let d2 = if proj <= 0.0 {
                            apx * apx + apy * apy + apz * apz
                        } else if proj >= max_t {
                            let (bx, by, bz) = (p.x - end.x, p.y - end.y, p.z - end.z);
                            bx * bx + by * by + bz * bz
                        } else {
                            (apx * apx + apy * apy + apz * apz) - proj * proj
                        };
                        if d2 <= r2 {
                            out.hits.push((proj.clamp(0.0, max_t), it));
                        }
                    }
                }
            }
            // Step the axis whose next boundary comes first.
            if tmx <= tmy && tmx <= tmz {
                if tmx > t_end { break; }
                ix += sx;
                tmx += tdx;
                if ix < 0 || ix >= n { break; }
            } else if tmy <= tmz {
                if tmy > t_end { break; }
                iy += sy;
                tmy += tdy;
                if iy < 0 || iy >= n { break; }
            } else {
                if tmz > t_end { break; }
                iz += sz;
                tmz += tdz;
                if iz < 0 || iz >= n { break; }
            }
        }
        out.hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// **First hit** along the ray (nearest by distance along it) with
    /// front-to-back **early-exit** — the grid analogue of
    /// [`crate::Tree::raycast_first`]. Same DDA walk, but it keeps the nearest
    /// item within `radius` and stops as soon as the next cell begins beyond the
    /// best hit (its entry `t` minus the `radius` slack exceeds the best `t`).
    /// Exact for thin rays; for thick rays it's the nearest hit *in the
    /// corridor*. Typically touches a handful of cells — the line-of-sight /
    /// picking query.
    pub fn raycast_first(&self, origin: Point3, dir: Point3, max_t: f64, radius: f64) -> Option<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if m == 0.0 {
            return None;
        }
        let (ux, uy, uz) = (dir.x / m, dir.y / m, dir.z / m);
        let (t0, t1) = clip_ray_aabb(origin, ux, uy, uz, &self.world, max_t)?;
        let r2 = radius * radius;
        let end = Point3::new(origin.x + ux * max_t, origin.y + uy * max_t, origin.z + uz * max_t);
        let n = self.cells_per_axis as i64;
        let s = Point3::new(origin.x + ux * t0, origin.y + uy * t0, origin.z + uz * t0);
        let mut ix = axis_index_n(s.x, self.world.x, self.cw, self.cells_per_axis) as i64;
        let mut iy = axis_index_n(s.y, self.world.y, self.ch, self.cells_per_axis) as i64;
        let mut iz = axis_index_n(s.z, self.world.z, self.cd, self.cells_per_axis) as i64;
        let setup = |u: f64, lo: f64, cell: f64, i: i64, o: f64| -> (i64, f64, f64) {
            if u > 0.0 { (1, (lo + (i + 1) as f64 * cell - o) / u, cell / u) }
            else if u < 0.0 { (-1, (lo + i as f64 * cell - o) / u, cell / (-u)) }
            else { (0, f64::INFINITY, f64::INFINITY) }
        };
        let (sx, mut tmx, tdx) = setup(ux, self.world.x, self.cw, ix, origin.x);
        let (sy, mut tmy, tdy) = setup(uy, self.world.y, self.ch, iy, origin.y);
        let (sz, mut tmz, tdz) = setup(uz, self.world.z, self.cd, iz, origin.z);
        let t_end = max_t.min(t1);
        let mut best: Option<(f64, &T)> = None;
        let mut t_enter = t0;
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > n as usize * 3 + 16 {
                break;
            }
            if let Some((bt, _)) = best {
                if t_enter - radius > bt {
                    break; // this cell and every later one start beyond the best hit
                }
            }
            if (0..n).contains(&ix) && (0..n).contains(&iy) && (0..n).contains(&iz) {
                if let Some(bucket) = bucket_of(&self.cells, morton3(ix as u32, iy as u32, iz as u32)) {
                    for it in bucket {
                        let p = it.position();
                        let (apx, apy, apz) = (p.x - origin.x, p.y - origin.y, p.z - origin.z);
                        let proj = apx * ux + apy * uy + apz * uz;
                        let d2 = if proj <= 0.0 {
                            apx * apx + apy * apy + apz * apz
                        } else if proj >= max_t {
                            let (bx, by, bz) = (p.x - end.x, p.y - end.y, p.z - end.z);
                            bx * bx + by * by + bz * bz
                        } else {
                            (apx * apx + apy * apy + apz * apz) - proj * proj
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
            if tmx <= tmy && tmx <= tmz {
                if tmx > t_end { break; }
                t_enter = tmx;
                ix += sx;
                tmx += tdx;
                if ix < 0 || ix >= n { break; }
            } else if tmy <= tmz {
                if tmy > t_end { break; }
                t_enter = tmy;
                iy += sy;
                tmy += tdy;
                if iy < 0 || iy >= n { break; }
            } else {
                if tmz > t_end { break; }
                t_enter = tmz;
                iz += sz;
                tmz += tdz;
                if iz < 0 || iz >= n { break; }
            }
        }
        best
    }

    /// The `k` nearest items to `q`, sorted ascending by distance — the grid
    /// analogue of [`crate::Tree3::knn`]. Where the tree descends nearest-child
    /// first, the grid expands **ring by ring**: it scans the query's own cell,
    /// then the surrounding Chebyshev shell (radius 1), then radius 2, … keeping
    /// a bounded max-heap of the k best. It stops once the whole *unscanned*
    /// region is provably farther than the current k-th nearest — i.e. when the
    /// distance from `q` to the nearest face of the already-scanned cell cube
    /// exceeds the k-th best distance (an exact lower bound, so no false stop).
    /// Fewer than `k` items → all of them; `k == 0` → empty.
    pub fn knn<'a>(&'a self, q: Point3, k: usize) -> Vec<(f64, &'a T)> {
        if k == 0 || self.len == 0 { return Vec::new(); }
        let (cx, cy, cz) = self.cell_of(q);
        let (cx, cy, cz) = (cx as i64, cy as i64, cz as i64);
        let n = self.cells_per_axis as i64;
        let mut heap: std::collections::BinaryHeap<KnnEntry<T>> = std::collections::BinaryHeap::new();

        // **Per-axis expansion**, not a Chebyshev shell. `levels` is one number for all three
        // axes, so on any world that is not a cube the cells are not cubes either: a
        // 1000x300x1000 world at levels=5 has cells 31.25 x 9.375 x 31.25. Growing all three
        // radii together is isotropic in CELL space and wildly anisotropic in WORLD space —
        // reaching 30 units along the short axis drags the wide axes out to +-125, and every
        // cell in between gets scanned. Measured on clustered data before this change:
        // 9 419 points tested per query, 47% of the entire population, for k = 8.
        //
        // So each axis grows on its own, always the one whose scanned world half-extent is
        // currently smallest, which keeps the scanned region near-cubic in world space. The
        // stopping rule is unchanged and still exact: everything inside the scanned box has
        // been offered, so the nearest unscanned point is at least `safe` away, and once the
        // k-th best beats that, no later cell can improve the answer.
        let (mut rx, mut ry, mut rz) = (0i64, 0i64, 0i64);
        let cells = &self.cells;
        let scan = |heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>, xs: (i64, i64), ys: (i64, i64), zs: (i64, i64)| {
            for ix in xs.0.max(0)..=xs.1.min(n - 1) {
                for iy in ys.0.max(0)..=ys.1.min(n - 1) {
                    for iz in zs.0.max(0)..=zs.1.min(n - 1) {
                        if let Some(bucket) = bucket_of(cells, morton3(ix as u32, iy as u32, iz as u32)) {
                            for it in bucket { knn_offer(heap, k, it, q); }
                        }
                    }
                }
            }
        };
        scan(&mut heap, (cx, cx), (cy, cy), (cz, cz));

        loop {
            // Distance from `q` to the nearest face of the scanned box.
            let xlo = self.world.x + (cx - rx) as f64 * self.cw;
            let xhi = self.world.x + (cx + rx + 1) as f64 * self.cw;
            let ylo = self.world.y + (cy - ry) as f64 * self.ch;
            let yhi = self.world.y + (cy + ry + 1) as f64 * self.ch;
            let zlo = self.world.z + (cz - rz) as f64 * self.cd;
            let zhi = self.world.z + (cz + rz + 1) as f64 * self.cd;
            let safe = (q.x - xlo).min(xhi - q.x).min(q.y - ylo).min(yhi - q.y).min(q.z - zlo).min(zhi - q.z);
            if heap.len() >= k && safe > 0.0 && safe * safe >= knn_worst(&heap, k) { break; }
            if rx > n && ry > n && rz > n { break; } // whole grid covered

            // Grow the narrowest axis in WORLD units — the whole point of the change.
            let (ex, ey, ez) = ((rx as f64 + 0.5) * self.cw, (ry as f64 + 0.5) * self.ch, (rz as f64 + 0.5) * self.cd);
            if ex <= ey && ex <= ez {
                rx += 1;
                scan(&mut heap, (cx - rx, cx - rx), (cy - ry, cy + ry), (cz - rz, cz + rz));
                scan(&mut heap, (cx + rx, cx + rx), (cy - ry, cy + ry), (cz - rz, cz + rz));
            } else if ey <= ez {
                ry += 1;
                scan(&mut heap, (cx - rx, cx + rx), (cy - ry, cy - ry), (cz - rz, cz + rz));
                scan(&mut heap, (cx - rx, cx + rx), (cy + ry, cy + ry), (cz - rz, cz + rz));
            } else {
                rz += 1;
                scan(&mut heap, (cx - rx, cx + rx), (cy - ry, cy + ry), (cz - rz, cz - rz));
                scan(&mut heap, (cx - rx, cx + rx), (cy - ry, cy + ry), (cz + rz, cz + rz));
            }
        }

        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    /// Parallel batch cull — see [`crate::Tree3::cull_many_par`].
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape3 + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
    }

    /// Batch k-NN — see [`crate::Tree3::knn_many`].
    pub fn knn_many(&self, queries: &[Point3], k: usize) -> Vec<Vec<(f64, &T)>> {
        queries.iter().map(|&q| self.knn(q, k)).collect()
    }

    /// Parallel batch k-NN — see [`crate::Tree3::knn_many_par`].
    #[cfg(feature = "parallel")]
    pub fn knn_many_par(&self, queries: &[Point3], k: usize) -> Vec<Vec<(f64, &T)>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        queries.par_iter().map(|&q| self.knn(q, k)).collect()
    }
}

// ------------------------------------------------------------- serialization

const MORTON3_MAGIC: &[u8; 4] = b"VHM3";
const MORTON3_VERSION: u8 = 1;

impl<T: Positioned3> MortonGrid3<T> {
    /// Serialize the grid (world + resolution + occupied buckets) to `w`. Items
    /// are written by `write_item`. Unlike the trees there is no arena to
    /// preserve — the cell layout is implicit in `world`/`levels`, so only the
    /// occupied `(code → bucket)` pairs are stored. Iteration order is the hash
    /// map's (unspecified), so the byte stream is not canonical, but a
    /// round-trip reproduces an equivalent grid.
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(MORTON3_MAGIC)?;
        w_u8(w, MORTON3_VERSION)?;
        w_aabb(w, &self.world)?;
        w_u32(w, self.levels)?;
        w_u32(w, self.cells.len() as u32)?;
        for (&code, bucket) in &self.cells {
            w_u64(w, code)?;
            w_u32(w, bucket.len() as u32)?;
            for it in bucket { write_item(w, it)?; }
        }
        Ok(())
    }

    /// Inverse of [`MortonGrid3::serialize`]: rebuild an equivalent grid from
    /// `r`, reading each item with `read_item` (must mirror the writer's layout).
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MORTON3_MAGIC { return Err(corrupt("bad MortonGrid3 magic")); }
        if r_u8(r)? != MORTON3_VERSION { return Err(corrupt("unsupported MortonGrid3 version")); }
        let world = r_aabb(r)?;
        let levels = r_u32(r)?;
        if !(1..=21).contains(&levels) { return Err(corrupt("MortonGrid3 levels out of 1..=21")); }
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
mod occupancy_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct P { p: Point3 }
    impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

    /// Each field is checked against a layout whose answer is known by construction, because
    /// "it returns some numbers" is not a test of a diagnostic.
    #[test]
    fn occupancy_reports_what_the_cells_actually_hold() {
        let world = Aabb::new(0.0, 0.0, 0.0, 64.0, 64.0, 64.0);
        let mut g: MortonGrid3<P> = MortonGrid3::new(world, 2); // 4 cells/axis, side 16
        assert_eq!(g.occupancy(), Occupancy { cells: 0, items: 0, mean: 0.0, max: 0 }, "empty grid");

        // 5 points inside one cell, 1 in another: 6 items, 2 cells, mean 3, max 5.
        for i in 0..5 { g.insert(P { p: Point3::new(1.0 + i as f64, 1.0, 1.0) }); }
        g.insert(P { p: Point3::new(50.0, 50.0, 50.0) });
        let o = g.occupancy();
        assert_eq!((o.cells, o.items, o.max), (2, 6, 5));
        assert!((o.mean - 3.0).abs() < 1e-12, "mean over NON-EMPTY cells, got {}", o.mean);

        // The empty cells are the point: 4^3 = 64 exist geometrically, 2 are stored.
        assert_eq!(g.cell_count(), 2, "empty cells must not be stored");
    }

    /// The skew signature the docs tell callers to look for — and, just as importantly, the
    /// one they are told NOT to use. `max / mean` reads 1.0 for data collapsed into a single
    /// cell, which is the worst case, so this pins both halves of that claim.
    #[test]
    fn clustering_shows_up_in_mean_and_cells_not_in_max_over_mean() {
        let world = Aabb::new(0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0);
        let mut uniform: MortonGrid3<P> = MortonGrid3::new(world, 3);
        let mut clumped: MortonGrid3<P> = MortonGrid3::new(world, 3);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 40) as f64) / (1u64 << 24) as f64 };
        for _ in 0..2000 {
            uniform.insert(P { p: Point3::new(rnd() * 999.0, rnd() * 999.0, rnd() * 999.0) });
            clumped.insert(P { p: Point3::new(rnd() * 40.0, rnd() * 40.0, rnd() * 40.0) });
        }
        let (u, c) = (uniform.occupancy(), clumped.occupancy());
        assert!(c.mean > 20.0 * u.mean, "clustering must show up in mean: {c:?} vs {u:?}");
        assert!(c.cells * 20 < u.cells, "and in the cell count collapsing: {c:?} vs {u:?}");
        // The trap, asserted so the doc's warning cannot rot: the ratio says the collapsed
        // grid is LESS skewed than the uniform one.
        assert!(c.max as f64 / c.mean <= u.max as f64 / u.mean,
            "max/mean is expected to be useless here — if this ever fails, revisit the docs: {c:?} vs {u:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy)]
    struct P(Point3);
    impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

    #[test]
    fn morton3_serialize_roundtrip() {
        use std::io::{Cursor, Read, Write};
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }
        let mut x = 0x0117_5E12u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut grid = MortonGrid3::<M>::new(world, 5);
        for id in 0..4000u32 {
            grid.insert(M { id, p: Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0) });
        }
        let mut buf = Vec::new();
        grid.serialize(&mut buf, |w, it| {
            w.write_all(&it.id.to_le_bytes())?;
            w.write_all(&it.p.x.to_le_bytes())?; w.write_all(&it.p.y.to_le_bytes())?; w.write_all(&it.p.z.to_le_bytes())
        }).unwrap();
        let loaded = MortonGrid3::<M>::deserialize(&mut Cursor::new(&buf), |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?; let mut b = [0u8; 8];
            let id = u32::from_le_bytes(a);
            r.read_exact(&mut b)?; let px = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let py = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let pz = f64::from_le_bytes(b);
            Ok(M { id, p: Point3::new(px, py, pz) })
        }).unwrap();
        assert_eq!(loaded.item_count(), grid.item_count(), "items");
        assert_eq!(loaded.cell_count(), grid.cell_count(), "cells");
        assert_eq!(loaded.levels(), grid.levels(), "levels");
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 30.0), (60.0, 200.0, 90.0, 50.0), (10.0, 10.0, 10.0, 80.0)] {
            let s = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut a: Vec<u32> = grid.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = loaded.cull(&s).iter().map(|m| m.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "cull differs after round-trip ({cx},{cy},{cz}) r={r}");
        }
        assert!(MortonGrid3::<M>::deserialize(&mut Cursor::new(&b"XXXXX"[..]), |_| unreachable!()).is_err());
    }

    #[test]
    fn morton_encode_known_values() {
        assert_eq!(morton3(0, 0, 0), 0);
        assert_eq!(morton3(1, 0, 0), 1); // x bit 0 → position 0
        assert_eq!(morton3(0, 1, 0), 2); // y bit 0 → position 1
        assert_eq!(morton3(0, 0, 1), 4); // z bit 0 → position 2
        assert_eq!(morton3(1, 1, 1), 7);
        assert_eq!(morton3(2, 0, 0), 8); // x bit 1 → position 3
    }

    #[test]
    fn morton_decode_roundtrips() {
        for (x, y, z) in [(0u32, 0u32, 0u32), (1, 2, 3), (1023, 7, 511), (1_048_575, 0, 1_048_575), (12345, 67890, 13579)] {
            assert_eq!(demorton3(morton3(x, y, z)), (x, y, z), "demorton3∘morton3 != id for ({x},{y},{z})");
        }
    }

    #[test]
    fn morton_cull_matches_brute() {
        // Build a grid over uniform points and verify the cull equals brute
        // force for a spread of sphere positions/radii, across resolutions.
        let mut x = 0x017E_0C75u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..4000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        for levels in [3u32, 5, 6] {
            let mut grid = MortonGrid3::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            assert_eq!(grid.item_count(), pts.len());
            for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 40.0), (40.0, 40.0, 40.0, 60.0), (250.0, 250.0, 250.0, 30.0), (0.0, 0.0, 0.0, 100.0)] {
                let s = Sphere3::new(cx, cy, cz, r).with_raster();
                let mut want: Vec<(u64, u64, u64)> = pts.iter().filter(|p| {
                    let dx = p.0.x - cx; let dy = p.0.y - cy; let dz = p.0.z - cz; dx * dx + dy * dy + dz * dz <= r * r
                }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
                let mut got: Vec<(u64, u64, u64)> = grid.cull(&s).iter()
                    .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
                want.sort(); got.sort();
                assert_eq!(want, got, "morton cull != brute (levels={levels}) for sphere ({cx},{cy},{cz}) r={r}");
            }
        }
    }

    #[test]
    fn morton_cull_layered_matches_cull_and_brute() {
        // The coarse-tier cull must return exactly what plain cull (and brute)
        // does — including a SPARSE world with a big query (where the coarse skip
        // actually fires: most coarse blocks over the void are empty).
        let mut x = 0xC0A4_5E11u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        // two dense clumps in a big world → lots of empty coarse space
        let pts: Vec<P> = (0..3000).map(|i| { let (cx, cy, cz) = if i % 2 == 0 { (200.0, 200.0, 200.0) } else { (1500.0, 1500.0, 1500.0) };
            P(Point3::new(cx + (rng() - 0.5) * 200.0, cy + (rng() - 0.5) * 200.0, cz + (rng() - 0.5) * 200.0)) }).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, 2048.0, 2048.0, 2048.0);
        for levels in [5u32, 7, 8] {
            let mut grid = MortonGrid3::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            for (cx, cy, cz, r) in [(1024.0, 1024.0, 1024.0, 1200.0), (200.0, 200.0, 200.0, 150.0), (1500.0, 1500.0, 1500.0, 400.0), (1000.0, 1000.0, 1000.0, 2000.0)] {
                let s = Sphere3::new(cx, cy, cz, r); // analytic (no raster: a 4000³ voxel raster would be 64 GB)
                let key = |v: &[&P]| { let mut k: Vec<(u64, u64, u64)> = v.iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect(); k.sort(); k };
                let plain = key(&grid.cull(&s)); let layered = key(&grid.cull_layered(&s));
                let brute = { let mut b: Vec<(u64, u64, u64)> = pts.iter().filter(|p| { let dx = p.0.x - cx; let dy = p.0.y - cy; let dz = p.0.z - cz; dx * dx + dy * dy + dz * dz <= r * r }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect(); b.sort(); b };
                assert_eq!(layered, plain, "cull_layered != cull (levels={levels}) sphere ({cx},{cy},{cz}) r={r}");
                assert_eq!(layered, brute, "cull_layered != brute (levels={levels}) sphere ({cx},{cy},{cz}) r={r}");
            }
        }
    }

    #[test]
    fn clear_empties_and_refills() {
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut g = MortonGrid3::<P>::new(world, 5);
        for i in 0..400u32 { g.insert(P(Point3::new((i % 20) as f64 * 12.0, 10.0, 10.0))); }
        assert!(g.item_count() > 0 && g.cell_count() > 0);
        g.clear();
        assert_eq!(g.item_count(), 0);
        assert_eq!(g.cell_count(), 0);
        for _ in 0..100 { g.insert(P(Point3::new(50.0, 50.0, 50.0))); }
        assert_eq!(g.item_count(), 100);
        assert_eq!(g.cull(&Sphere3::new(50.0, 50.0, 50.0, 5.0)).len(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn extend_par_matches_serial_insert() {
        // The parallel bulk build must produce an identical grid to inserting
        // serially — same items, same buckets, same culls.
        let mut x = 0x5EED_1234u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..5000).map(|_| P(Point3::new(rng() * 300.0 - 22.0, rng() * 300.0 - 22.0, rng() * 300.0 - 22.0))).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut a = MortonGrid3::<P>::new(world, 5);
        for p in &pts { a.insert(*p); }
        let mut b = MortonGrid3::<P>::new(world, 5);
        let added = b.extend_par(pts.clone());
        assert_eq!(added, a.item_count(), "extend_par count != serial inserted count");
        assert_eq!(a.item_count(), b.item_count());
        assert_eq!(a.cell_count(), b.cell_count());
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 40.0), (20.0, 240.0, 60.0, 50.0), (0.0, 0.0, 0.0, 100.0)] {
            let s = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut wa: Vec<(u64, u64, u64)> = a.cull(&s).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut wb: Vec<(u64, u64, u64)> = b.cull(&s).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            wa.sort(); wb.sort();
            assert_eq!(wa, wb, "extend_par cull != serial cull");
        }
    }

    #[test]
    fn morton_raycast_subset_of_capsule_and_sorted() {
        // The DDA hits must be a subset of the exact capsule cull (no invented
        // items / wrong cells), sorted by t, and non-empty on a populated ray.
        use crate::Segment3;
        let mut x = 0x3ABC_1199u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..6000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut grid = MortonGrid3::<P>::new(world, 5);
        for p in &pts { grid.insert(*p); }
        let rays = [
            (Point3::new(10.0, 10.0, 10.0), Point3::new(1.0, 0.8, 0.6), 400.0, 8.0),
            (Point3::new(250.0, 10.0, 128.0), Point3::new(-1.0, 1.0, 0.2), 400.0, 12.0),
            (Point3::new(128.0, 250.0, 5.0), Point3::new(0.1, -1.0, 0.3), 360.0, 5.0),
            (Point3::new(0.0, 128.0, 128.0), Point3::new(1.0, 0.0, 0.0), 256.0, 20.0),
        ];
        let mut any = false;
        for (o, d, mt, r) in rays {
            let dda = grid.raycast(o, d, mt, r);
            for w in dda.hits.windows(2) { assert!(w[0].0 <= w[1].0, "DDA hits not sorted by t"); }
            let m = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            let (ux, uy, uz) = (d.x / m, d.y / m, d.z / m);
            let end = Point3::new(o.x + ux * mt, o.y + uy * mt, o.z + uz * mt);
            let cull: std::collections::HashSet<(u64, u64, u64)> = grid.cull(&Segment3::new(o, end, r)).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            for (_, p) in &dda.hits {
                let k = (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits());
                assert!(cull.contains(&k), "DDA hit not in the exact capsule cull (wrong cell or false positive)");
                any = true;
            }
        }
        assert!(any, "DDA found nothing on any ray — likely a traversal bug");
    }

    #[test]
    fn morton_raycast_first_matches_nearest() {
        let mut x = 0x77AA_3311u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..6000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut grid = MortonGrid3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 5);
        for p in &pts { grid.insert(*p); }
        for (o, d, mt, r) in [
            (Point3::new(10.0, 10.0, 10.0), Point3::new(1.0, 0.8, 0.6), 400.0, 8.0),
            (Point3::new(250.0, 10.0, 128.0), Point3::new(-1.0, 1.0, 0.2), 400.0, 12.0),
            (Point3::new(0.0, 128.0, 128.0), Point3::new(1.0, 0.0, 0.0), 256.0, 20.0),
        ] {
            let first = grid.raycast_first(o, d, mt, r);
            let all = grid.raycast(o, d, mt, r);
            match (first, all.hits.first()) {
                (None, None) => {}
                (Some((t1, _)), Some(&(t0, _))) => assert!((t1 - t0).abs() < 1e-9, "raycast_first t {t1} != raycast nearest t {t0}"),
                _ => panic!("raycast_first / raycast nearest disagree on hit/miss"),
            }
        }
    }

    #[test]
    fn morton_knn_matches_brute() {
        // k-NN over the grid must return the same k smallest distances as brute
        // force. Compare distances (not item identity) so exact ties don't
        // spuriously fail. Sweep resolutions, query points, and k.
        let mut x = 0x51ED_270Bu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        for levels in [3u32, 5, 6] {
            let mut grid = MortonGrid3::<P>::new(world, levels);
            for p in &pts { grid.insert(*p); }
            for (qx, qy, qz) in [(128.0, 128.0, 128.0), (10.0, 250.0, 40.0), (0.0, 0.0, 0.0), (255.0, 255.0, 255.0)] {
                let q = Point3::new(qx, qy, qz);
                for k in [1usize, 8, 32] {
                    let mut brute: Vec<f64> = pts.iter().map(|p| { let d = (p.0.x - qx, p.0.y - qy, p.0.z - qz); (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt() }).collect();
                    brute.sort_by(|a, b| a.total_cmp(b));
                    brute.truncate(k);
                    let got: Vec<f64> = grid.knn(q, k).into_iter().map(|(d, _)| d).collect();
                    assert_eq!(got.len(), brute.len(), "knn count (levels={levels}, k={k})");
                    for (a, b) in got.iter().zip(brute.iter()) {
                        assert!((a - b).abs() < 1e-9, "knn dist != brute (levels={levels}, k={k}): {a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn levels_for_cell_size_picks_reasonable_resolution() {
        let world = Aabb::new(0.0, 0.0, 0.0, 512.0, 512.0, 512.0);
        // target cell ≈ 16 → 32 cells/axis → levels 5 (512/32 = 16).
        let levels = MortonGrid3::<P>::levels_for_cell_size(world, 16.0);
        let cells = 1u64 << levels;
        let cell = 512.0 / cells as f64;
        let half = 512.0 / (cells * 2) as f64;
        assert!(cell >= 16.0 && half < 16.0, "levels={levels} gives cell {cell} (want ≈16)");
    }
}

#[cfg(test)]
mod update_tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct M { id: u32, p: Point3 }
    impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }
    impl SetPosition3 for M { fn set_position(&mut self, p: Point3) { self.p = p; } }

    fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0) }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f64) / (1u64 << 24) as f64 }
        fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }

    /// The property the whole feature rests on: a grid MAINTAINED with `update` must answer
    /// exactly like one REBUILT from the same positions. Not "similar" — the same set, after
    /// thousands of moves, including moves that cross cells and moves that do not.
    #[test]
    fn maintained_grid_answers_identically_to_a_rebuilt_one() {
        let mut rng = Rng(0x5EED_1234);
        let n = 800usize;
        let mut pts: Vec<M> = (0..n).map(|i| M { id: i as u32, p: Point3::new(rng.r(0.0, 255.9), rng.r(0.0, 255.9), rng.r(0.0, 255.9)) }).collect();
        let mut kept: MortonGrid3<M> = MortonGrid3::new(world(), 4);
        for it in &pts { kept.insert(*it); }

        let (mut stayed, mut moved) = (0u32, 0u32);
        for round in 0..40 {
            for (i, pt) in pts.iter_mut().enumerate() {
                let old = pt.p;
                // Half tiny nudges (which should mostly stay), half long jumps (which should
                // mostly re-bucket) — a test that only did one of the two would miss a path.
                let step = if (i + round) % 2 == 0 { 1.0 } else { 90.0 };
                let np = Point3::new(
                    (old.x + rng.r(-step, step)).clamp(0.0, 255.9),
                    (old.y + rng.r(-step, step)).clamp(0.0, 255.9),
                    (old.z + rng.r(-step, step)).clamp(0.0, 255.9));
                let id = pt.id;
                match kept.update(old, |it| it.id == id, |it| it.p = np) {
                    Crossed::Stayed => stayed += 1,
                    Crossed::Moved => moved += 1,
                    other => panic!("update failed on item {id}: {other:?}"),
                }
                pt.p = np;
            }

            let mut fresh: MortonGrid3<M> = MortonGrid3::new(world(), 4);
            for it in &pts { fresh.insert(*it); }
            assert_eq!(kept.item_count(), fresh.item_count(), "count drifted at round {round}");
            assert_eq!(kept.cell_count(), fresh.cell_count(), "cell set drifted at round {round} — an emptied bucket was left behind");
            for (cx, cy, cz, r) in [(40.0, 40.0, 40.0, 30.0), (128.0, 128.0, 128.0, 50.0), (200.0, 60.0, 210.0, 25.0)] {
                let s = Sphere3::new(cx, cy, cz, r);
                let mut a: Vec<u32> = kept.cull(&s).iter().map(|m| m.id).collect();
                let mut b: Vec<u32> = fresh.cull(&s).iter().map(|m| m.id).collect();
                a.sort_unstable(); b.sort_unstable();
                assert_eq!(a, b, "maintained != rebuilt at round {round}");
            }
            let q = Point3::new(90.0, 110.0, 70.0);
            let ka: Vec<u64> = kept.knn(q, 12).iter().map(|(d, _)| d.to_bits()).collect();
            let kb: Vec<u64> = fresh.knn(q, 12).iter().map(|(d, _)| d.to_bits()).collect();
            assert_eq!(ka, kb, "knn diverged at round {round}");
        }
        // Non-vacuous: both branches must have been exercised, or this proves half of it.
        assert!(stayed > 1000 && moved > 1000, "test did not exercise both paths: stayed {stayed}, moved {moved}");
    }

    /// The edge cases, each asserted rather than assumed.
    #[test]
    fn update_reports_what_it_did() {
        let mut g: MortonGrid3<M> = MortonGrid3::new(world(), 4); // 16 cells/axis, side 16
        g.insert(M { id: 1, p: Point3::new(8.0, 8.0, 8.0) });
        g.insert(M { id: 2, p: Point3::new(200.0, 200.0, 200.0) });

        // within the same cell
        assert_eq!(g.update(Point3::new(8.0, 8.0, 8.0), |it| it.id == 1, |it| it.p = Point3::new(9.0, 9.0, 9.0)), Crossed::Stayed);
        // into another cell
        assert_eq!(g.update(Point3::new(9.0, 9.0, 9.0), |it| it.id == 1, |it| it.p = Point3::new(100.0, 100.0, 100.0)), Crossed::Moved);
        assert_eq!(g.item_count(), 2);
        // wrong old position: nothing in that cell matches
        assert_eq!(g.update(Point3::new(9.0, 9.0, 9.0), |it| it.id == 1, |it| it.p = Point3::new(1.0, 1.0, 1.0)), Crossed::Missing);
        // out of the world: dropped, exactly as insert would have refused it
        assert_eq!(g.update(Point3::new(100.0, 100.0, 100.0), |it| it.id == 1, |it| it.p = Point3::new(9999.0, 0.0, 0.0)), Crossed::Left);
        assert_eq!(g.item_count(), 1, "an item that left the world must be gone, not stale");
        assert!(g.cull(&Sphere3::new(100.0, 100.0, 100.0, 50.0)).is_empty(), "the vacated cell still answers");
        // and an old position outside the world is Missing, not a panic
        assert_eq!(g.update(Point3::new(-5.0, 0.0, 0.0), |_| true, |_| {}), Crossed::Missing);
    }

    /// Removal has to leave the grid indistinguishable from one built without the item —
    /// including the cell set, which is where a forgotten empty bucket would show.
    #[test]
    fn remove_leaves_the_grid_as_if_the_item_had_never_been_inserted() {
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut rng = Rng(0xBEEF);
        let pts: Vec<M> = (0..300).map(|i| M { id: i as u32, p: Point3::new(rng.r(0.0, 255.9), rng.r(0.0, 255.9), rng.r(0.0, 255.9)) }).collect();
        let mut g: MortonGrid3<M> = MortonGrid3::new(world, 4);
        for it in &pts { g.insert(*it); }

        let doomed: Vec<usize> = (0..300).step_by(3).collect();
        for &i in &doomed {
            let got = g.remove(pts[i].p, |it| it.id == i as u32).expect("item was there");
            assert_eq!(got.id, i as u32);
            assert!(g.remove(pts[i].p, |it| it.id == i as u32).is_none(), "removing twice must be None");
        }
        let survivors: Vec<M> = pts.iter().enumerate().filter(|(i, _)| !doomed.contains(i)).map(|(_, m)| *m).collect();
        let mut fresh: MortonGrid3<M> = MortonGrid3::new(world, 4);
        for it in &survivors { fresh.insert(*it); }
        assert_eq!(g.item_count(), fresh.item_count());
        assert_eq!(g.cell_count(), fresh.cell_count(), "an emptied bucket was left behind");
        for (cx, cy, cz, r) in [(60.0, 60.0, 60.0, 40.0), (180.0, 120.0, 200.0, 55.0)] {
            let s = Sphere3::new(cx, cy, cz, r);
            let mut a: Vec<u32> = g.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = fresh.cull(&s).iter().map(|m| m.id).collect();
            a.sort_unstable(); b.sort_unstable();
            assert_eq!(a, b, "removed grid != rebuilt grid");
        }
    }

    /// `relocate` is the same thing with the predicate written for you.
    #[test]
    fn relocate_matches_the_predicate_form() {
        let mut g: MortonGrid3<M> = MortonGrid3::new(world(), 4);
        g.insert(M { id: 7, p: Point3::new(20.0, 20.0, 20.0) });
        assert_eq!(g.relocate(Point3::new(20.0, 20.0, 20.0), Point3::new(150.0, 150.0, 150.0)), Crossed::Moved);
        let hit = g.cull(&Sphere3::new(150.0, 150.0, 150.0, 1.0));
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].id, 7);
    }
}
