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

use crate::template::CellState;
use crate::tree3::{Aabb, Point3, Positioned3, Shape3};

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
        if cell <= 0.0 {
            return 0;
        }
        let i = ((v - lo) / cell).floor();
        if i < 0.0 {
            0
        } else if i as u64 >= self.cells_per_axis as u64 {
            self.cells_per_axis - 1
        } else {
            i as u32
        }
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
                    let bucket = match self.cells.get(&morton3(ix, iy, iz)) {
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

    /// Batch cull — see [`crate::Tree3::cull_many`].
    pub fn cull_many<'a, S: Shape3>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy)]
    struct P(Point3);
    impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

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
