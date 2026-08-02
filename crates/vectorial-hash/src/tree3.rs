//! Experimental 3D spatial index — the "true 3D tree" half of the 3D
//! roadmap (the other half is projection indexing, benched against this in
//! `vectorial-hash-demos/src/bin/tree3d_bench.rs`).
//!
//! A binary-split tree in 3D: the direct analogue of [`crate::Tree`] with
//! `Point3`/`Aabb` instead of `Point`/`Rect`, splitting the longest of the
//! three axes. (An octree — 8-way 2×2×2 split — is the analogue of
//! [`crate::QuadTree`]; this binary version mirrors the primary 2D
//! structure and keeps the comparison apples-to-apples.)
//!
//! Culling uses the same green / white / yellow short-circuit as 2D:
//! - **green** (Aabb fully inside the shape) → take the whole subtree,
//! - **white** (fully outside) → skip it,
//! - **yellow** → recurse, and at leaves run a per-point test.
//!
//! The per-point leaf test can use either the shape's exact `contains_point`
//! or a precomputed **1×1×1 voxel raster** ([`VoxelRaster`]) — the 3D
//! analogue of the 2D 1×1 raster: In/Out voxels resolve by lookup, only
//! boundary (Maybe) voxels run the exact test.

use std::io::{self, Read, Write};

use crate::template::CellState;
use crate::tree::{RaycastOut, DEAD_HANDLE};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Axis-aligned box, half-open on every axis: `[x,x+w) × [y,y+h) × [z,z+d)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
    pub h: f64,
    pub d: f64,
}

impl Aabb {
    pub const fn new(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64) -> Self {
        Self { x, y, z, w, h, d }
    }
    #[inline] pub fn x_max(&self) -> f64 { self.x + self.w }
    #[inline] pub fn y_max(&self) -> f64 { self.y + self.h }
    #[inline] pub fn z_max(&self) -> f64 { self.z + self.d }
    #[inline]
    pub fn contains(&self, p: Point3) -> bool {
        p.x >= self.x && p.x < self.x_max()
            && p.y >= self.y && p.y < self.y_max()
            && p.z >= self.z && p.z < self.z_max()
    }
    #[inline]
    pub fn contains_closed(&self, p: Point3) -> bool {
        p.x >= self.x && p.x <= self.x_max()
            && p.y >= self.y && p.y <= self.y_max()
            && p.z >= self.z && p.z <= self.z_max()
    }
}

pub trait Positioned3 {
    fn position(&self) -> Point3;
}

/// A 3D query volume. `bounding_box` + `contains_point` are required;
/// `classify_aabb` enables the green/white/yellow short-circuit (default
/// falls back to bbox-intersect → Maybe/Out, i.e. no green).
pub trait Shape3 {
    fn bounding_box(&self) -> Aabb;
    fn contains_point(&self, p: Point3) -> bool;
    /// Classify a node box against the volume. Default: any box overlapping
    /// the bounding box is `Maybe`, otherwise `Out` (never green — forces
    /// per-point checks). Shapes with an analytic inside/outside test
    /// (e.g. a sphere) should override to return `In` for fully-contained
    /// boxes so whole subtrees short-circuit.
    fn classify_aabb(&self, b: &Aabb) -> CellState {
        let bb = self.bounding_box();
        let overlap = b.x < bb.x_max() && bb.x < b.x_max()
            && b.y < bb.y_max() && bb.y < b.y_max()
            && b.z < bb.z_max() && bb.z < b.z_max();
        if overlap { CellState::Maybe } else { CellState::Out }
    }
    /// Optional precomputed 1×1×1 voxel raster for the per-point leaf test.
    fn voxel_raster(&self) -> Option<&VoxelRaster> { None }
}

// ----------------------------------------------------------------- sphere

/// A sphere query volume with an exact analytic AABB classification.
pub struct Sphere3 {
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub r: f64,
    pub raster: Option<VoxelRaster>,
}

impl Sphere3 {
    pub fn new(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        Self { cx, cy, cz, r, raster: None }
    }
    /// Attach a precomputed 1×1×1 voxel raster covering the sphere's bbox.
    pub fn with_raster(mut self) -> Self {
        self.raster = Some(VoxelRaster::for_sphere(self.cx, self.cy, self.cz, self.r));
        self
    }
}

impl Shape3 for Sphere3 {
    fn bounding_box(&self) -> Aabb {
        Aabb::new(self.cx - self.r, self.cy - self.r, self.cz - self.r,
                  2.0 * self.r, 2.0 * self.r, 2.0 * self.r)
    }
    fn contains_point(&self, p: Point3) -> bool {
        let dx = p.x - self.cx; let dy = p.y - self.cy; let dz = p.z - self.cz;
        dx * dx + dy * dy + dz * dz <= self.r * self.r
    }
    fn classify_aabb(&self, b: &Aabb) -> CellState {
        // Nearest point of the box to the centre, and farthest corner.
        let nx = self.cx.clamp(b.x, b.x_max());
        let ny = self.cy.clamp(b.y, b.y_max());
        let nz = self.cz.clamp(b.z, b.z_max());
        let near2 = (nx - self.cx).powi(2) + (ny - self.cy).powi(2) + (nz - self.cz).powi(2);
        if near2 > self.r * self.r {
            return CellState::Out; // closest point already beyond r
        }
        let fx = if (self.cx - b.x).abs() > (self.cx - b.x_max()).abs() { b.x } else { b.x_max() };
        let fy = if (self.cy - b.y).abs() > (self.cy - b.y_max()).abs() { b.y } else { b.y_max() };
        let fz = if (self.cz - b.z).abs() > (self.cz - b.z_max()).abs() { b.z } else { b.z_max() };
        let far2 = (fx - self.cx).powi(2) + (fy - self.cy).powi(2) + (fz - self.cz).powi(2);
        if far2 <= self.r * self.r {
            CellState::In // farthest corner within r → whole box inside
        } else {
            CellState::Maybe
        }
    }
    fn voxel_raster(&self) -> Option<&VoxelRaster> {
        self.raster.as_ref()
    }
}

// --------------------------------------------------------------- polyhedron

/// A convex polyhedron = intersection of half-spaces `n·p <= d`. Unlike the
/// sphere, its `contains_point` is **expensive** — one dot product per face
/// — which is the regime where the 1×1×1 [`VoxelRaster`] pays for itself
/// (a single memory lookup beats N plane evaluations). Build one from a set
/// of planes, or via [`Polyhedron3::faceted_ball`] for an N-face ball-like
/// solid.
pub struct Polyhedron3 {
    /// Each plane is `(nx, ny, nz, d)` meaning the half-space `n·p <= d`.
    pub planes: Vec<(f64, f64, f64, f64)>,
    pub bbox: Aabb,
    pub raster: Option<VoxelRaster>,
}

impl Polyhedron3 {
    pub fn new(planes: Vec<(f64, f64, f64, f64)>, bbox: Aabb) -> Self {
        Self { planes, bbox, raster: None }
    }

    /// A ball of radius `r` centred at `c`, approximated by `faces` tangent
    /// half-spaces with pseudo-evenly-spread normals (Fibonacci sphere),
    /// **clipped to the ball's bounding cube** (6 extra axis-aligned planes)
    /// so the solid is guaranteed to lie inside its declared `bbox` — needed
    /// for the voxel raster to cover it. A good stand-in for a many-faced
    /// non-analytic convex shape.
    pub fn faceted_ball(cx: f64, cy: f64, cz: f64, r: f64, faces: usize) -> Self {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut planes = Vec::with_capacity(faces + 6);
        for i in 0..faces {
            let zf = 1.0 - 2.0 * (i as f64 + 0.5) / faces as f64;
            let rad = (1.0 - zf * zf).max(0.0).sqrt();
            let th = golden * i as f64;
            let (nx, ny, nz) = (rad * th.cos(), rad * th.sin(), zf);
            // tangent plane at c + r*n: n·p <= n·c + r
            planes.push((nx, ny, nz, nx * cx + ny * cy + nz * cz + r));
        }
        // Clip to the bounding cube [c±r]³ (the few-face intersection bulges
        // past it otherwise, escaping the raster grid).
        planes.push((1.0, 0.0, 0.0, cx + r));
        planes.push((-1.0, 0.0, 0.0, -(cx - r)));
        planes.push((0.0, 1.0, 0.0, cy + r));
        planes.push((0.0, -1.0, 0.0, -(cy - r)));
        planes.push((0.0, 0.0, 1.0, cz + r));
        planes.push((0.0, 0.0, -1.0, -(cz - r)));
        let b = Aabb::new(cx - r, cy - r, cz - r, 2.0 * r, 2.0 * r, 2.0 * r);
        Self { planes, bbox: b, raster: None }
    }

    /// Build the convex solid from **8 corner points** — the camera-frustum
    /// constructor (a view frustum is just 6 half-spaces). Corners are the near
    /// face then the far face, each ordered (bottom-left, bottom-right,
    /// top-right, top-left): indices `0..4` near, `4..8` far. Each face plane is
    /// derived with an **inward** normal (oriented against the corner centroid),
    /// so the winding need not be exact and an axis-aligned box passed as corners
    /// recovers its own six faces. `bbox` is the corners' AABB. Use it to cull a
    /// camera frustum: `tree.cull(&Polyhedron3::from_corners(frustum_corners))`.
    pub fn from_corners(corners: [Point3; 8]) -> Self {
        let cen = {
            let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
            for p in &corners { sx += p.x; sy += p.y; sz += p.z; }
            Point3::new(sx / 8.0, sy / 8.0, sz / 8.0)
        };
        // Three non-collinear points on each face: near, far, left, right, bottom, top.
        let faces = [[0, 1, 2], [4, 5, 6], [0, 3, 7], [1, 2, 6], [0, 1, 5], [3, 2, 6]];
        let mut planes = Vec::with_capacity(6);
        for [i, j, k] in faces {
            let (a, b, c) = (corners[i], corners[j], corners[k]);
            let e1 = (b.x - a.x, b.y - a.y, b.z - a.z);
            let e2 = (c.x - a.x, c.y - a.y, c.z - a.z);
            let mut nx = e1.1 * e2.2 - e1.2 * e2.1;
            let mut ny = e1.2 * e2.0 - e1.0 * e2.2;
            let mut nz = e1.0 * e2.1 - e1.1 * e2.0;
            let mut d = nx * a.x + ny * a.y + nz * a.z;
            // Orient inward: the centroid must satisfy the half-space n·p <= d.
            if nx * cen.x + ny * cen.y + nz * cen.z > d {
                nx = -nx; ny = -ny; nz = -nz; d = -d;
            }
            planes.push((nx, ny, nz, d));
        }
        let (mut lo, mut hi) = (corners[0], corners[0]);
        for p in &corners {
            lo.x = lo.x.min(p.x); lo.y = lo.y.min(p.y); lo.z = lo.z.min(p.z);
            hi.x = hi.x.max(p.x); hi.y = hi.y.max(p.y); hi.z = hi.z.max(p.z);
        }
        let bbox = Aabb::new(lo.x, lo.y, lo.z, hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
        Self { planes, bbox, raster: None }
    }

    pub fn with_raster(mut self) -> Self {
        // Borrow-checker: build against a raster-less copy of self.
        let probe = Polyhedron3 { planes: self.planes.clone(), bbox: self.bbox, raster: None };
        self.raster = Some(VoxelRaster::for_shape(&probe));
        self
    }

    /// **Line-of-sight / occlusion test.** The parameter `t ∈ [0, 1]` at which
    /// the segment `a`→`b` **first enters** this convex solid, or `None` if the
    /// segment never touches it. Liang–Barsky clipping against the half-spaces
    /// `n·p ≤ d` (each plane bounds the valid `t` interval; convex ⇒ one run).
    ///
    /// Use it as a hard occlusion test: a viewer at `a` sees a target at `b`
    /// **unless** some occluder returns `Some(t)` with `t < 1` — the solid blocks
    /// the line before the target is reached. (`Segment3`/`raycast` is the *thick*
    /// capsule that finds items *near* a ray; this is the exact segment↔solid
    /// surface hit the doc-comment there points to.)
    pub fn segment_hit(&self, a: Point3, b: Point3) -> Option<f64> {
        let (dx, dy, dz) = (b.x - a.x, b.y - a.y, b.z - a.z);
        let (mut t0, mut t1) = (0.0_f64, 1.0_f64); // inside-interval, clamped to the segment
        for &(nx, ny, nz, d) in &self.planes {
            let den = nx * dx + ny * dy + nz * dz;      // n·(b−a)
            let num = d - (nx * a.x + ny * a.y + nz * a.z); // constraint: t·den ≤ num
            if den.abs() < 1e-12 {
                if num < 0.0 { return None; } // parallel to this face and outside it
            } else {
                let t = num / den;
                if den > 0.0 { if t < t1 { t1 = t; } } // upper bound
                else if t > t0 { t0 = t; }             // lower bound
                if t0 > t1 { return None; }            // interval collapsed → misses the solid
            }
        }
        if t0 <= t1 { Some(t0) } else { None }
    }

    /// **Dilation** (Minkowski-flavoured): the convex solid grown outward by `r`.
    /// Each face half-space is pushed out by `r` (`n·p ≤ d` → `n·p ≤ d + r‖n‖`,
    /// so a normal of any scale moves its face by geometric distance `r`) and the
    /// bbox grows by `r`. The 3D analogue of the 2D `inflated_convex`: cull with
    /// `inflated(r)` to find every agent whose **centre** is within `r` of the
    /// figure — keep items as points, inflate the *query* (agent-body radius,
    /// `disk(r1) ⊕ disk(r2) = disk(r1+r2)`, vision-radius grows, …). Conservative
    /// at sharp corners (a superset of the exact `r`-neighbourhood → no false
    /// negatives in a cull); filter in narrowphase for exactness.
    pub fn inflated(&self, r: f64) -> Polyhedron3 {
        let planes = self.planes.iter().map(|&(nx, ny, nz, d)| { let m = (nx * nx + ny * ny + nz * nz).sqrt(); (nx, ny, nz, d + r * m) }).collect();
        let b = self.bbox;
        Polyhedron3 { planes, bbox: Aabb::new(b.x - r, b.y - r, b.z - r, b.w + 2.0 * r, b.h + 2.0 * r, b.d + 2.0 * r), raster: None }
    }
}

impl Shape3 for Polyhedron3 {
    fn bounding_box(&self) -> Aabb { self.bbox }
    fn contains_point(&self, p: Point3) -> bool {
        for &(nx, ny, nz, d) in &self.planes {
            if nx * p.x + ny * p.y + nz * p.z > d {
                return false;
            }
        }
        true
    }
    fn classify_aabb(&self, b: &Aabb) -> CellState {
        let mut all_inside = true;
        for &(nx, ny, nz, d) in &self.planes {
            // min / max of n·p over the box corners.
            let mnx = nx * b.x; let mxx = nx * b.x_max();
            let mny = ny * b.y; let mxy = ny * b.y_max();
            let mnz = nz * b.z; let mxz = nz * b.z_max();
            let lo = mnx.min(mxx) + mny.min(mxy) + mnz.min(mxz);
            let hi = mnx.max(mxx) + mny.max(mxy) + mnz.max(mxz);
            if lo > d {
                return CellState::Out; // whole box violates this face
            }
            if hi > d {
                all_inside = false; // box straddles this face
            }
        }
        if all_inside { CellState::In } else { CellState::Maybe }
    }
    fn voxel_raster(&self) -> Option<&VoxelRaster> {
        self.raster.as_ref()
    }
}

// --------------------------------------------------------------- segment

/// A **capsule**: a line segment `a`–`b` thickened by radius `r`. As a
/// [`Shape3`] it's the query volume behind a "thick ray-cast" — every item
/// within `r` of the segment (see [`Tree3::raycast`]). Point items need a
/// radius to be hit; a hard surface-intersection ray is a different primitive.
pub struct Segment3 {
    pub a: Point3,
    pub b: Point3,
    pub r: f64,
    pub raster: Option<VoxelRaster>,
}

impl Segment3 {
    pub fn new(a: Point3, b: Point3, r: f64) -> Self {
        Self { a, b, r, raster: None }
    }
    /// A capsule from a ray: `origin` to `origin + normalize(dir) * len`,
    /// thickened by `r`. Zero-length `dir` collapses to a sphere at `origin`.
    pub fn from_ray(origin: Point3, dir: Point3, len: f64, r: f64) -> Self {
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        let (ux, uy, uz) = if m > 0.0 { (dir.x / m, dir.y / m, dir.z / m) } else { (0.0, 0.0, 0.0) };
        Self::new(origin, Point3::new(origin.x + ux * len, origin.y + uy * len, origin.z + uz * len), r)
    }
}

/// Squared distance from `p` to segment `a`–`b` (clamped projection).
#[inline]
pub(crate) fn seg_point_dist2(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = (b.x - a.x, b.y - a.y, b.z - a.z);
    let ap = (p.x - a.x, p.y - a.y, p.z - a.z);
    let denom = ab.0 * ab.0 + ab.1 * ab.1 + ab.2 * ab.2;
    let t = if denom > 0.0 { ((ap.0 * ab.0 + ap.1 * ab.1 + ap.2 * ab.2) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy, dz) = (ap.0 - ab.0 * t, ap.1 - ab.1 * t, ap.2 - ab.2 * t);
    dx * dx + dy * dy + dz * dz
}

impl Shape3 for Segment3 {
    fn bounding_box(&self) -> Aabb {
        let lx = self.a.x.min(self.b.x) - self.r;
        let ly = self.a.y.min(self.b.y) - self.r;
        let lz = self.a.z.min(self.b.z) - self.r;
        let hx = self.a.x.max(self.b.x) + self.r;
        let hy = self.a.y.max(self.b.y) + self.r;
        let hz = self.a.z.max(self.b.z) + self.r;
        Aabb::new(lx, ly, lz, hx - lx, hy - ly, hz - lz)
    }
    fn contains_point(&self, p: Point3) -> bool {
        seg_point_dist2(p, self.a, self.b) <= self.r * self.r
    }
    fn classify_aabb(&self, b: &Aabb) -> CellState {
        // Conservative sphere classify (cheap, safe): the box's bounding sphere
        // (centre `c`, radius = half-diagonal) vs the capsule spine. Centre
        // farther than `r + half_diag` → whole box outside → Out; within
        // `r − half_diag` → whole box inside → In; else Maybe. One segment↔point
        // distance, no exact segment↔box distance. Looser than an exact test but
        // prunes the cells far from the diagonal that a bbox-overlap test keeps.
        let c = Point3::new(b.x + b.w * 0.5, b.y + b.h * 0.5, b.z + b.d * 0.5);
        let half_diag = 0.5 * (b.w * b.w + b.h * b.h + b.d * b.d).sqrt();
        let d = seg_point_dist2(c, self.a, self.b).sqrt();
        if d > self.r + half_diag {
            CellState::Out
        } else if d + half_diag <= self.r {
            CellState::In
        } else {
            CellState::Maybe
        }
    }
    fn voxel_raster(&self) -> Option<&VoxelRaster> { self.raster.as_ref() }
}

// ----------------------------------------------------------- voxel raster

/// A 1×1×1 voxel grid classifying each unit cell as In/Out/Maybe relative
/// to a shape. The 3D analogue of the 2D 1×1 raster: the per-point leaf
/// test becomes a lookup, and only `Maybe` voxels run the exact geometry.
pub struct VoxelRaster {
    origin: (i64, i64, i64),
    dims: (usize, usize, usize),
    cells: Vec<CellState>, // x-major: idx = (x*dy + y)*dz + z
}

impl VoxelRaster {
    /// Build the raster covering a sphere's bounding box. Each voxel is a
    /// unit cube; classify by distance from the sphere centre to the
    /// voxel's nearest and farthest corner (exact In/Out/Maybe).
    pub fn for_sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        let ox = (cx - r).floor() as i64;
        let oy = (cy - r).floor() as i64;
        let oz = (cz - r).floor() as i64;
        let ex = (cx + r).ceil() as i64;
        let ey = (cy + r).ceil() as i64;
        let ez = (cz + r).ceil() as i64;
        let (dx, dy, dz) = ((ex - ox) as usize, (ey - oy) as usize, (ez - oz) as usize);
        let mut cells = vec![CellState::Out; dx * dy * dz];
        let r2 = r * r;
        for ix in 0..dx {
            for iy in 0..dy {
                for iz in 0..dz {
                    let vx = (ox + ix as i64) as f64;
                    let vy = (oy + iy as i64) as f64;
                    let vz = (oz + iz as i64) as f64;
                    // nearest corner to centre, farthest corner
                    let nx = cx.clamp(vx, vx + 1.0);
                    let ny = cy.clamp(vy, vy + 1.0);
                    let nz = cz.clamp(vz, vz + 1.0);
                    let near2 = (nx - cx).powi(2) + (ny - cy).powi(2) + (nz - cz).powi(2);
                    let fx = if (cx - vx).abs() > (cx - (vx + 1.0)).abs() { vx } else { vx + 1.0 };
                    let fy = if (cy - vy).abs() > (cy - (vy + 1.0)).abs() { vy } else { vy + 1.0 };
                    let fz = if (cz - vz).abs() > (cz - (vz + 1.0)).abs() { vz } else { vz + 1.0 };
                    let far2 = (fx - cx).powi(2) + (fy - cy).powi(2) + (fz - cz).powi(2);
                    let state = if near2 > r2 { CellState::Out }
                        else if far2 <= r2 { CellState::In }
                        else { CellState::Maybe };
                    cells[(ix * dy + iy) * dz + iz] = state;
                }
            }
        }
        Self { origin: (ox, oy, oz), dims: (dx, dy, dz), cells }
    }

    /// Build the raster for an arbitrary [`Shape3`] by classifying each unit
    /// voxel of the shape's bounding box with the shape's own
    /// `classify_aabb` (In/Out/Maybe). This is the general 1×1×1 raster: it
    /// turns the per-point leaf test into a lookup for *any* shape, which is
    /// the win when `contains_point` is expensive (a many-faced polyhedron)
    /// rather than a single distance compare (a sphere).
    pub fn for_shape<S: Shape3>(shape: &S) -> Self {
        let b = shape.bounding_box();
        let ox = b.x.floor() as i64;
        let oy = b.y.floor() as i64;
        let oz = b.z.floor() as i64;
        let ex = b.x_max().ceil() as i64;
        let ey = b.y_max().ceil() as i64;
        let ez = b.z_max().ceil() as i64;
        let (dx, dy, dz) = ((ex - ox) as usize, (ey - oy) as usize, (ez - oz) as usize);
        let mut cells = vec![CellState::Out; dx * dy * dz];
        for ix in 0..dx {
            for iy in 0..dy {
                for iz in 0..dz {
                    let voxel = Aabb::new(
                        (ox + ix as i64) as f64, (oy + iy as i64) as f64, (oz + iz as i64) as f64,
                        1.0, 1.0, 1.0,
                    );
                    cells[(ix * dy + iy) * dz + iz] = shape.classify_aabb(&voxel);
                }
            }
        }
        Self { origin: (ox, oy, oz), dims: (dx, dy, dz), cells }
    }

    /// Classify the voxel containing world point `p`. Out-of-grid → Out.
    #[inline]
    pub fn cell_at_world(&self, p: Point3) -> CellState {
        let ix = p.x.floor() as i64 - self.origin.0;
        let iy = p.y.floor() as i64 - self.origin.1;
        let iz = p.z.floor() as i64 - self.origin.2;
        if ix < 0 || iy < 0 || iz < 0 { return CellState::Out; }
        let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
        let (dx, dy, dz) = self.dims;
        if ix >= dx || iy >= dy || iz >= dz { return CellState::Out; }
        self.cells[(ix * dy + iy) * dz + iz]
    }
}

// ------------------------------------------------------------------- tree

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node3Id(pub u32);

/// A **stable handle** to an inserted item (the "Stable `ItemRef`"): returned by
/// [`Tree3::insert_ref`] and accepted by [`Tree3::update_ref`] /
/// [`Tree3::remove_ref`]. It stays valid as the item moves between leaves, so
/// those calls reach the item in **O(1)** — no locate walk, no per-leaf
/// predicate scan (the cost that made the predicate [`Tree3::update`] lose the
/// relocate race to a flat rebuild; see `docs/THREE_D.md` § Synthesis).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemRef(pub u32);

/// Result of [`Tree3::update_ref_tracked`] — reports whether a moved item stayed
/// in its leaf, crossed into a different one, or left the world. Useful when a
/// caller needs to react to an item **changing leaf/cell** (streaming, LOD
/// tiers, dirty-region tracking, partition-boundary logic).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Crossing {
    /// The item moved but stayed in the same leaf (leaf id carried along).
    Stayed(Node3Id),
    /// The item crossed into a different leaf. Note a leaf is finer than any
    /// coarse "region" a caller might define, so debounce against a
    /// leaf→region map if you only care about coarse crossings.
    Moved { from: Node3Id, to: Node3Id },
    /// The item left the world; its handle was freed.
    Left,
}

/// Where a handle's item currently lives: its leaf node and slot within that
/// leaf's `items`/`hs` vectors.
#[derive(Copy, Clone)]
struct ItemLoc { node: Node3Id, slot: u32 }

pub struct Node3<T> {
    pub bbox: Aabb,
    pub parent: Option<Node3Id>,
    pub children: Option<[Node3Id; 2]>,
    pub items: Vec<T>,
    /// Parallel to `items`: `hs[i]` is the handle of `items[i]`. Lets a
    /// `swap_remove` fix the moved item's location in O(1). Empty on internal
    /// (non-leaf) nodes.
    hs: Vec<u32>,
}

pub struct Tree3<T: Positioned3> {
    nodes: Vec<Node3<T>>,
    /// Slots freed by merge-ups, reused before the arena grows — see
    /// [`crate::Tree`]'s free-list for the rationale.
    free: Vec<Node3Id>,
    /// Handle → current location. Indexed by [`ItemRef`]'s u32. Stale entries
    /// (freed handles) sit on `free_handles`.
    locs: Vec<ItemLoc>,
    free_handles: Vec<u32>,
    pub item_limit: usize,
    pub merge_limit: usize,
    min_cell: f64,
    pub root: Node3Id,
}

impl<T: Positioned3> Tree3<T> {
    /// Read an item through its stable [`ItemRef`] — `None` if the handle has been retired by
    /// `remove_ref`.
    ///
    /// The handle layer could **move** an item and could **delete** it, but not look at it, so
    /// any caller wanting to read one had to keep a parallel copy or abuse `update_ref`'s
    /// mutator to smuggle a value out. O(1), no descent and no scan: the handle *is* the dense
    /// index into the location table.
    pub fn get_ref(&self, r: ItemRef) -> Option<&T> {
        let loc = self.live_loc(r)?;
        self.get(loc.node).items.get(loc.slot as usize)
    }

    pub fn new(bbox: Aabb, item_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let min_cell = bbox.w.max(bbox.h).max(bbox.d) * 1e-12;
        Self {
            nodes: vec![Node3 { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() }],
            free: Vec::new(),
            locs: Vec::new(),
            free_handles: Vec::new(),
            item_limit,
            merge_limit: item_limit,
            min_cell,
            root: Node3Id(0),
        }
    }

    /// Empty the tree, **retaining allocated capacity** — the node arena, the
    /// handle table and the free lists are reset, not freed. Cheaper than
    /// dropping and rebuilding when you refill the index every frame (e.g. a
    /// per-frame projection rebuild). All existing [`ItemRef`]s are invalidated.
    pub fn clear(&mut self) {
        let bbox = self.get(self.root).bbox;
        self.nodes.clear();
        self.nodes.push(Node3 { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() });
        self.free.clear();
        self.locs.clear();
        self.free_handles.clear();
        self.root = Node3Id(0);
    }

    #[inline] pub fn get(&self, id: Node3Id) -> &Node3<T> { &self.nodes[id.0 as usize] }
    #[inline] fn get_mut(&mut self, id: Node3Id) -> &mut Node3<T> { &mut self.nodes[id.0 as usize] }
    fn alloc(&mut self, n: Node3<T>) -> Node3Id {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0 as usize] = n;
            id
        } else {
            let id = Node3Id(self.nodes.len() as u32);
            self.nodes.push(n);
            id
        }
    }

    // ---- handle bookkeeping (the Stable ItemRef layer) ----

    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() {
            h
        } else {
            let h = self.locs.len() as u32;
            self.locs.push(ItemLoc { node: Node3Id(0), slot: 0 });
            h
        }
    }
    /// Retire a handle: mark its location [`DEAD_HANDLE`] before recycling the id, so
    /// a stale `ItemRef` can't alias whatever item later lands in that slot.
    fn free_handle(&mut self, h: u32) {
        self.locs[h as usize] = ItemLoc { node: Node3Id(DEAD_HANDLE), slot: 0 };
        self.free_handles.push(h);
    }
    /// The live location behind a handle, or `None` if it was freed (item removed or
    /// dropped out of the root) or never belonged to this tree.
    fn live_loc(&self, r: ItemRef) -> Option<ItemLoc> {
        let loc = *self.locs.get(r.0 as usize)?;
        (loc.node.0 != DEAD_HANDLE).then_some(loc)
    }

    /// Push `item` (with handle `h`) into leaf `node`, recording its location.
    fn push_h(&mut self, node: Node3Id, item: T, h: u32) {
        let slot = self.get(node).items.len() as u32;
        let n = self.get_mut(node);
        n.items.push(item);
        n.hs.push(h);
        self.locs[h as usize] = ItemLoc { node, slot };
    }

    /// `swap_remove` slot `slot` of leaf `node`, fixing the moved item's
    /// recorded location. Returns `(item, its handle)`.
    fn swap_remove_h(&mut self, node: Node3Id, slot: usize) -> (T, u32) {
        let (item, h, moved) = {
            let n = self.get_mut(node);
            let item = n.items.swap_remove(slot);
            let h = n.hs.swap_remove(slot);
            let moved = if slot < n.hs.len() { Some(n.hs[slot]) } else { None };
            (item, h, moved)
        };
        if let Some(m) = moved { self.locs[m as usize].slot = slot as u32; }
        (item, h)
    }
    /// Arena capacity (high-water-mark). [`Tree3::live_node_count`] is the
    /// reachable count.
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn live_node_count(&self) -> usize { self.nodes.len() - self.free.len() }

    /// Reorder the node arena into DFS pre-order and drop freed slots — a pure
    /// **cache-locality** pass (van-Emde-Boas-flavoured: a node lands adjacent to
    /// its first child, so a root→leaf descent walks mostly-contiguous memory).
    /// The tree shape, items, bboxes, handles and *every query result* are
    /// unchanged; only the `nodes` Vec order and the internal `Node3Id`s move.
    ///
    /// Why: after a long churny run the split/merge slots land wherever the free
    /// list points, so insertion-order nodes drift into a scramble and descents
    /// thrash cache. `compact()` restores locality (and reclaims freed slots).
    /// O(live nodes), one pass. Invalidates any raw `Node3Id` you cached — but
    /// **not** `ItemRef` handles, which are remapped. See `docs/PERF_NOTES.md`.
    pub fn compact(&mut self) {
        let n = self.nodes.len();
        let mut old2new = vec![u32::MAX; n];
        let mut order: Vec<Node3Id> = Vec::with_capacity(self.live_node_count());
        // DFS pre-order from the root: emit a node, then recurse its children —
        // push right before left so the left child pops first and lands right
        // after its parent (the contiguous descent spine).
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            old2new[id.0 as usize] = order.len() as u32;
            order.push(id);
            if let Some([a, b]) = self.get(id).children { stack.push(b); stack.push(a); }
        }
        let remap = |id: Node3Id| Node3Id(old2new[id.0 as usize]);
        let mut new_nodes: Vec<Node3<T>> = Vec::with_capacity(order.len());
        for &old in &order {
            let bbox = self.nodes[old.0 as usize].bbox;
            let mut node = std::mem::replace(&mut self.nodes[old.0 as usize],
                Node3 { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() });
            node.parent = node.parent.map(remap);
            node.children = node.children.map(|[a, b]| [remap(a), remap(b)]);
            new_nodes.push(node);
        }
        // Remap the handle table (live handles always point at a reachable node;
        // stale free-handle locs may not, so guard on the sentinel).
        for loc in self.locs.iter_mut() {
            if loc.node.0 == DEAD_HANDLE { continue; } // freed handle: no live node to remap
            let nn = old2new[loc.node.0 as usize];
            if nn != u32::MAX { loc.node = Node3Id(nn); }
        }
        self.root = remap(self.root);
        self.nodes = new_nodes;
        self.free.clear();
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.insert_ref(item).is_some()
    }

    /// Insert and return a [stable handle](ItemRef) for O(1) future
    /// `update_ref`/`remove_ref` — `None` if the point is outside the root.
    pub fn insert_ref(&mut self, item: T) -> Option<ItemRef> {
        let p = item.position();
        if !self.get(self.root).bbox.contains(p) { return None; }
        let leaf = self.locate(p);
        let h = self.alloc_handle();
        self.push_h(leaf, item, h);
        if self.get(leaf).items.len() > self.item_limit { self.divide(leaf); }
        Some(ItemRef(h))
    }

    pub fn locate(&self, p: Point3) -> Node3Id {
        let mut cur = self.root;
        loop {
            match self.get(cur).children {
                None => return cur,
                Some([a, b]) => {
                    cur = if self.get(a).bbox.contains(p) { a } else { b };
                }
            }
        }
    }

    /// Build a tree from **all `items` at once** — a top-down partition (the same
    /// longest-axis-midpoint split `divide` uses) rather than N `insert`s. Handle
    /// `i` addresses `items[i]` (so `ItemRef(i)` is stable, as after inserting in
    /// order). A per-frame *rebuild* via `bulk_load` avoids the repeated
    /// root-descents of `clear()` + `insert` — one partition pass instead. Items
    /// are assumed within `bbox` (as the demos' clamped positions are).
    pub fn bulk_load(bbox: Aabb, item_limit: usize, items: Vec<T>) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let min_cell = bbox.w.max(bbox.h).max(bbox.d) * 1e-12;
        let n = items.len();
        let mut nodes: Vec<Node3<T>> = Vec::new();
        let mut locs = vec![ItemLoc { node: Node3Id(0), slot: 0 }; n];
        let indexed: Vec<(u32, T)> = items.into_iter().enumerate().map(|(i, it)| (i as u32, it)).collect();
        build3_serial(&mut nodes, &mut locs, item_limit, min_cell, bbox, None, indexed);
        Tree3 { nodes, free: Vec::new(), locs, free_handles: Vec::new(), item_limit, merge_limit: item_limit, min_cell, root: Node3Id(0) }
    }

    /// Parallel [`Tree3::bulk_load`] (feature `parallel`): the top-down partition
    /// — the O(n log n) work — fans out over rayon (`join` per split); the arena
    /// flatten is a cheap serial tail. The lever for the per-frame index rebuild
    /// that the serial `insert` loop can't parallelise (see `docs/PARALLEL.md`).
    #[cfg(feature = "parallel")]
    pub fn bulk_load_par(bbox: Aabb, item_limit: usize, items: Vec<T>) -> Self
    where T: Send {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let min_cell = bbox.w.max(bbox.h).max(bbox.d) * 1e-12;
        let n = items.len();
        let indexed: Vec<(u32, T)> = items.into_iter().enumerate().map(|(i, it)| (i as u32, it)).collect();
        let build = build3_par(item_limit, min_cell, bbox, indexed);
        let mut nodes: Vec<Node3<T>> = Vec::new();
        let mut locs = vec![ItemLoc { node: Node3Id(0), slot: 0 }; n];
        flatten3(&mut nodes, &mut locs, bbox, None, build);
        Tree3 { nodes, free: Vec::new(), locs, free_handles: Vec::new(), item_limit, merge_limit: item_limit, min_cell, root: Node3Id(0) }
    }

    fn divide(&mut self, id: Node3Id) {
        let (bbox, items, hs) = {
            let n = self.get_mut(id);
            (n.bbox, std::mem::take(&mut n.items), std::mem::take(&mut n.hs))
        };
        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.w.max(bbox.h).max(bbox.d) <= self.min_cell {
            // Put them back unchanged — same slots, so `locs` stays valid.
            let n = self.get_mut(id);
            n.items = items;
            n.hs = hs;
            return;
        }
        // Split the longest axis at its midpoint.
        let (a_box, b_box) = if bbox.w >= bbox.h && bbox.w >= bbox.d {
            let half = bbox.w / 2.0;
            (Aabb::new(bbox.x, bbox.y, bbox.z, half, bbox.h, bbox.d),
             Aabb::new(bbox.x + half, bbox.y, bbox.z, half, bbox.h, bbox.d))
        } else if bbox.h >= bbox.d {
            let half = bbox.h / 2.0;
            (Aabb::new(bbox.x, bbox.y, bbox.z, bbox.w, half, bbox.d),
             Aabb::new(bbox.x, bbox.y + half, bbox.z, bbox.w, half, bbox.d))
        } else {
            let half = bbox.d / 2.0;
            (Aabb::new(bbox.x, bbox.y, bbox.z, bbox.w, bbox.h, half),
             Aabb::new(bbox.x, bbox.y, bbox.z + half, bbox.w, bbox.h, half))
        };
        let a = self.alloc(Node3 { bbox: a_box, parent: Some(id), children: None, items: Vec::new(), hs: Vec::new() });
        let b = self.alloc(Node3 { bbox: b_box, parent: Some(id), children: None, items: Vec::new(), hs: Vec::new() });
        for (item, h) in items.into_iter().zip(hs) {
            let p = item.position();
            let dest = if self.get(a).bbox.contains(p) { a } else { b };
            self.push_h(dest, item, h);
        }
        self.get_mut(id).children = Some([a, b]);
        if self.get(a).items.len() > self.item_limit { self.divide(a); }
        if self.get(b).items.len() > self.item_limit { self.divide(b); }
    }

    /// Relocate via ascend-to-LCA (the 2D winner): find the item by `old` +
    /// `predicate`, mutate in place, and if it leaves its leaf, ascend to the
    /// lowest ancestor containing the new position and descend from there.
    /// Returns `false` if not found or pushed out of the root. This is the
    /// **predicate** path (O(item_limit) leaf scan); for O(1) relocation hold a
    /// stable [`ItemRef`] from [`Tree3::insert_ref`] and use [`Tree3::update_ref`].
    pub fn update<F, M>(&mut self, old: Point3, predicate: F, mutator: M) -> bool
    where F: Fn(&T) -> bool, M: FnOnce(&mut T) {
        if !self.get(self.root).bbox.contains(old) { return false; }
        let leaf = self.locate(old);
        let idx = match self.get(leaf).items.iter().position(&predicate) {
            Some(i) => i, None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);
        let np = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(np) { return true; }
        self.relocate(leaf, idx, np).is_some()
    }

    /// O(1) relocation through a stable [`ItemRef`]: no locate walk, no
    /// predicate scan — go straight to the item, mutate it, and only pay the
    /// ascend-to-LCA descent if it actually leaves its leaf. Returns `false`
    /// (and frees the handle) if it left the root.
    pub fn update_ref<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> bool {
        let Some(loc) = self.live_loc(r) else { return false }; // stale handle: item already gone
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) { return true; }
        self.relocate(node, slot, np).is_some()
    }

    /// Boundary-crossing variant of [`update_ref`] — reports whether the item
    /// stayed in its leaf, crossed into a **different leaf** (with both leaf
    /// ids), or left the world. The hook for reacting to an item **changing
    /// cell** (re-streaming, LOD tier changes, dirty-region tracking, coarse
    /// partition logic).
    ///
    /// **Caveat:** a *leaf* is finer than any coarse *region* a caller defines
    /// (a leaf splits at `item_limit` density, so an item can cross many leaves
    /// in one step). If you only care about coarse crossings, map
    /// `leaf → region` (a small side table over the tree) and act only when the
    /// *region* changes — [`Crossing::Moved`] is the cheap exact leaf signal;
    /// the coarse debounce is yours. Cost is identical to `update_ref` (the
    /// from/to leaf ids were already computed internally, just not surfaced).
    pub fn update_ref_tracked<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> Crossing {
        let Some(loc) = self.live_loc(r) else { return Crossing::Left }; // stale handle: item already gone
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) { return Crossing::Stayed(node); }
        match self.relocate(node, slot, np) {
            Some(dest) => Crossing::Moved { from: node, to: dest },
            None => Crossing::Left,
        }
    }

    /// The leaf a stable [`ItemRef`] currently lives in (O(1)) — pair with a
    /// `leaf → coarse-region` side table if you group leaves into larger cells.
    pub fn ref_leaf(&self, r: ItemRef) -> Node3Id { self.locs[r.0 as usize].node }

    /// Shared tail of `update`/`update_ref`: the item at `(leaf, slot)` has
    /// moved to `np` outside `leaf`. Ascend to the LCA, re-descend, move it.
    /// Returns the destination leaf, or `None` if it left the root (dropped +
    /// merged, handle freed).
    fn relocate(&mut self, leaf: Node3Id, slot: usize, np: Point3) -> Option<Node3Id> {
        let mut anc = self.get(leaf).parent;
        let lca = loop {
            match anc {
                Some(a) if self.get(a).bbox.contains(np) => break a,
                Some(a) => anc = self.get(a).parent,
                None => { // out of bounds: drop + merge, free the handle
                    let (_, h) = self.swap_remove_h(leaf, slot);
                    self.free_handle(h);
                    self.try_merge_up(leaf);
                    return None;
                }
            }
        };
        let (item, h) = self.swap_remove_h(leaf, slot);
        let dest = self.locate_from(lca, np);
        self.push_h(dest, item, h);
        if self.get(dest).items.len() > self.item_limit { self.divide(dest); }
        self.try_merge_up(leaf);
        Some(dest)
    }

    fn locate_from(&self, start: Node3Id, p: Point3) -> Node3Id {
        let mut cur = start;
        loop {
            match self.get(cur).children {
                None => return cur,
                Some([a, b]) => cur = if self.get(a).bbox.contains(p) { a } else { b },
            }
        }
    }

    pub fn remove<F: Fn(&T) -> bool>(&mut self, p: Point3, predicate: F) -> Option<T> {
        if !self.get(self.root).bbox.contains(p) { return None; }
        let leaf = self.locate(p);
        let idx = self.get(leaf).items.iter().position(&predicate)?;
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.free_handle(h);
        self.try_merge_up(leaf);
        Some(item)
    }

    /// Remove the item behind a stable [`ItemRef`] in O(1) (no scan). The
    /// handle is consumed; reusing it is a logic error.
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.live_loc(r)?; // stale handle: already removed
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handle(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    fn try_merge_up(&mut self, mut node: Node3Id) {
        loop {
            let parent = match self.get(node).parent { Some(p) => p, None => return };
            let [a, b] = self.get(parent).children.expect("parent has children");
            if self.get(a).children.is_some() || self.get(b).children.is_some() { return; }
            let combined = self.get(a).items.len() + self.get(b).items.len();
            if combined > self.merge_limit { return; }
            let mut ia = std::mem::take(&mut self.get_mut(a).items);
            let mut iha = std::mem::take(&mut self.get_mut(a).hs);
            let mut ib = std::mem::take(&mut self.get_mut(b).items);
            let mut ihb = std::mem::take(&mut self.get_mut(b).hs);
            ia.append(&mut ib);
            iha.append(&mut ihb);
            let pnode = self.get_mut(parent);
            pnode.items = ia;
            pnode.hs = iha;
            pnode.children = None;
            // The merged items now live in `parent` (a leaf again) — re-point
            // their handle locations.
            let len = self.get(parent).hs.len();
            for slot in 0..len {
                let h = self.get(parent).hs[slot];
                self.locs[h as usize] = ItemLoc { node: parent, slot: slot as u32 };
            }
            self.free.push(a);
            self.free.push(b);
            node = parent;
        }
    }


    /// Every live item's stable [`ItemRef`], in **depth-first leaf order** — the order the tree
    /// has already sorted them into, which is spatially coherent by construction.
    ///
    /// This exists for *warm-start migration*: a structure being replaced can hand its
    /// successor a spatially ordered sequence instead of an arbitrary one, which measures
    /// **1.42x on a `KdTree3` build and 1.81x on `Tree3` inserts**
    /// (`examples/migration_warm_start`). The grids could already do it via `iter_z_order`; a
    /// tree could not, because its per-item handles are parallel to `items` inside each leaf
    /// and were not reachable from outside the module. They are now.
    ///
    /// Costs one `Vec` of handles and a traversal — paid only when something asks, never in
    /// steady state.
    pub fn handles_dfs(&self) -> Vec<ItemRef> {
        let mut out = Vec::with_capacity(self.item_count());
        self.visit_leaves(|n| out.extend(n.hs.iter().map(|&h| ItemRef(h))));
        out
    }

    pub fn visit_leaves<F: FnMut(&Node3<T>)>(&self, mut f: F) {
        self.visit_from(self.root, &mut f);
    }
    fn visit_from<F: FnMut(&Node3<T>)>(&self, id: Node3Id, f: &mut F) {
        match self.get(id).children {
            Some([a, b]) => { self.visit_from(a, f); self.visit_from(b, f); }
            None => f(self.get(id)),
        }
    }
    pub fn item_count(&self) -> usize {
        let mut n = 0; self.visit_leaves(|l| n += l.items.len()); n
    }
    pub fn leaf_count(&self) -> usize {
        let mut n = 0; self.visit_leaves(|_| n += 1); n
    }

    /// Return references to every item inside `shape`.
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        self.cull_recurse(self.root, shape, false, &mut out);
        out
    }

    /// Cull many independent shapes in one call, returning one hit-list per
    /// shape (`out[i]` is the cull for `shapes[i]`). Always available and
    /// serial; see [`Tree3::cull_many_par`] for the rayon-backed version. This
    /// is the batch shape of a per-attacker combat sweep: each attacker culls
    /// its own attack volume against the shared, immutable index.
    pub fn cull_many<'a, S: Shape3>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// "Thick ray-cast": every item within `radius` of the ray
    /// `origin + t·normalize(dir)`, `t ∈ [0, max_dist]`, returned as
    /// `(t, &item)` sorted by `t` (nearest first). Built on the [`Segment3`]
    /// capsule + `cull`, then projected onto the ray. A zero-length `dir`
    /// returns nothing. This is a *thick* ray — point items need the radius to
    /// register; a hard surface-intersection ray is a separate primitive.
    pub fn raycast(&self, origin: Point3, dir: Point3, max_dist: f64, radius: f64) -> Vec<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if m == 0.0 {
            return Vec::new();
        }
        let (ux, uy, uz) = (dir.x / m, dir.y / m, dir.z / m);
        let end = Point3::new(origin.x + ux * max_dist, origin.y + uy * max_dist, origin.z + uz * max_dist);
        let seg = Segment3::new(origin, end, radius);
        let mut hits: Vec<(f64, &T)> = self
            .cull(&seg)
            .into_iter()
            .map(|it| {
                let p = it.position();
                let t = ((p.x - origin.x) * ux + (p.y - origin.y) * uy + (p.z - origin.z) * uz).clamp(0.0, max_dist);
                (t, it)
            })
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits
    }

    /// **DDA leaf-walk ray-cast** over the *variable-size* tree — the 3D analogue
    /// of the 2D [`crate::Tree::raycast`], adapted to the binary 3D tree. Walks
    /// the leaves the centre ray crosses front-to-back; `Tree3` has no 3D
    /// neighbour links, so the step is **Probe-style**: the slab test on the
    /// current leaf gives the exit `t`, and `locate` finds the next leaf just
    /// across that face. Collects items within `radius` of the ray, sorted by
    /// `t`, + stats (`leaves_visited`, `items_tested`). Thin corridor — use
    /// `raycast` / `cull(&Segment3)` for the exact thick band.
    ///
    /// The step is **nudge-free** (see [`Tree3::ray_step3_lca`]): the face
    /// neighbour is found by ascending to the least-common-ancestor, so no
    /// epsilon can skip a thin sliver. What the walk still is, by construction,
    /// is a **thin corridor**: it visits only the leaves the centre ray crosses,
    /// so an item within `radius` of the ray but sitting in a leaf the centreline
    /// misses is not reported. The result is therefore a strict *subset* of the
    /// exact capsule — never a false positive, and `examples/work_counters.rs`
    /// checks that subset relation and reports the recall it buys.
    pub fn raycast_dda(&self, origin: Point3, dir: Point3, max_t: f64, radius: f64) -> RaycastOut<'_, T> {
        let mut out = RaycastOut { hits: Vec::new(), leaves_visited: 0, items_tested: 0 };
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if m == 0.0 {
            return out;
        }
        let (ux, uy, uz) = (dir.x / m, dir.y / m, dir.z / m);
        let end = Point3::new(origin.x + ux * max_t, origin.y + uy * max_t, origin.z + uz * max_t);
        let r2 = radius * radius;
        let mut leaf = match self.raycast_start_leaf(origin, ux, uy, uz, max_t) {
            Some(l) => l,
            None => return out,
        };
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > self.nodes.len() * 3 + 32 {
                break;
            }
            out.leaves_visited += 1;
            for it in &self.get(leaf).items {
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
            match self.ray_step3_lca(leaf, origin, ux, uy, uz, max_t) {
                Some((_, next)) => leaf = next,
                None => break,
            }
        }
        out.hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// First hit (nearest along the ray) over the variable-size tree with
    /// front-to-back **early-exit** — see [`crate::Tree::raycast_first`]. Same
    /// Probe-style walk as [`Tree3::raycast_dda`], stopping at the first leaf
    /// beyond the best hit.
    pub fn raycast_dda_first(&self, origin: Point3, dir: Point3, max_t: f64, radius: f64) -> Option<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if m == 0.0 {
            return None;
        }
        let (ux, uy, uz) = (dir.x / m, dir.y / m, dir.z / m);
        let end = Point3::new(origin.x + ux * max_t, origin.y + uy * max_t, origin.z + uz * max_t);
        let r2 = radius * radius;
        let mut leaf = self.raycast_start_leaf(origin, ux, uy, uz, max_t)?;
        let mut best: Option<(f64, &T)> = None;
        let mut t_enter = 0.0;
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > self.nodes.len() * 3 + 32 {
                break;
            }
            if let Some((bt, _)) = best {
                if t_enter - radius > bt {
                    break;
                }
            }
            for it in &self.get(leaf).items {
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
            match self.ray_step3_lca(leaf, origin, ux, uy, uz, max_t) {
                Some((t_exit, next)) => {
                    t_enter = t_exit;
                    leaf = next;
                }
                None => break,
            }
        }
        best
    }

    /// The leaf the ray enters the world at (clipped + nudged inside), or `None`
    /// if it misses or starts past `max_t`. Shared by the Probe-style DDAs.
    fn raycast_start_leaf(&self, origin: Point3, ux: f64, uy: f64, uz: f64, max_t: f64) -> Option<Node3Id> {
        let world = self.get(self.root).bbox;
        let span = world.w.max(world.h).max(world.d);
        let mut t = 0.0_f64;
        for (oa, ua, lo, hi) in [(origin.x, ux, world.x, world.x_max()), (origin.y, uy, world.y, world.y_max()), (origin.z, uz, world.z, world.z_max())] {
            if ua == 0.0 {
                if oa < lo || oa > hi {
                    return None;
                }
            } else {
                let (mut ta, tb) = ((lo - oa) / ua, (hi - oa) / ua);
                if ta > tb {
                    ta = tb;
                }
                t = t.max(ta);
            }
        }
        let t0 = t.max(0.0) + span * 1e-9;
        if t0 >= max_t {
            return None;
        }
        let entry = Point3::new(origin.x + ux * t0, origin.y + uy * t0, origin.z + uz * t0);
        if !world.contains(entry) {
            return None;
        }
        Some(self.locate(entry))
    }

    /// One **nudge-free** DDA step from `leaf`: the slab test gives the exit `t`
    /// and which face the ray leaves by; the neighbour leaf just across that face
    /// is found by **ascending to the least-common-ancestor** whose sibling lies on
    /// the far side, then descending to the leaf touching the exit point — exact, no
    /// `locate`+epsilon nudge (Samet's rope-free face neighbour). `(t_exit,
    /// next_leaf)`, or `None` if the ray ends (`t_exit ≥ max_t`) or leaves the world.
    fn ray_step3_lca(&self, leaf: Node3Id, origin: Point3, ux: f64, uy: f64, uz: f64, max_t: f64) -> Option<(f64, Node3Id)> {
        let b = self.get(leaf).bbox;
        let tx = if ux > 0.0 { (b.x_max() - origin.x) / ux } else if ux < 0.0 { (b.x - origin.x) / ux } else { f64::INFINITY };
        let ty = if uy > 0.0 { (b.y_max() - origin.y) / uy } else if uy < 0.0 { (b.y - origin.y) / uy } else { f64::INFINITY };
        let tz = if uz > 0.0 { (b.z_max() - origin.z) / uz } else if uz < 0.0 { (b.z - origin.z) / uz } else { f64::INFINITY };
        let t_exit = tx.min(ty).min(tz);
        if t_exit >= max_t {
            return None;
        }
        let (ax, dir_pos) = if t_exit == tx { (0usize, ux > 0.0) } else if t_exit == ty { (1usize, uy > 0.0) } else { (2usize, uz > 0.0) };
        let p_exit = Point3::new(origin.x + ux * t_exit, origin.y + uy * t_exit, origin.z + uz * t_exit);
        let next = self.face_neighbor3(leaf, ax, dir_pos, p_exit)?;
        if next == leaf {
            return None;
        }
        Some((t_exit, next))
    }

    /// The leaf on the far side of `node`'s exit face (axis `ax`; `dir_pos` = the
    /// max face) at `p_exit`. Ascends via parents until the split *is* on `ax` and
    /// `node` is the near child, so the sibling is the neighbour subtree, then
    /// descends it. `None` ⇒ the face is the world boundary (the ray leaves).
    fn face_neighbor3(&self, mut node: Node3Id, ax: usize, dir_pos: bool, p_exit: Point3) -> Option<Node3Id> {
        loop {
            let par = self.get(node).parent?;
            let [c0, c1] = self.get(par).children.expect("internal node has children");
            if split_axis3(self.get(c0).bbox, self.get(c1).bbox) == ax {
                if dir_pos && node == c0 { return Some(self.descend_face3(c1, ax, dir_pos, p_exit)); }
                if !dir_pos && node == c1 { return Some(self.descend_face3(c0, ax, dir_pos, p_exit)); }
            }
            node = par; // split off-axis, or we are already on the far side → keep climbing
        }
    }

    /// Descend `node` to the leaf touching `p_exit` on the entry (`ax`) face:
    /// off-axis, follow `p_exit`'s side of each split; on `ax`, take the **near**
    /// child (we enter from that face) — so the walk needs no epsilon.
    fn descend_face3(&self, mut node: Node3Id, ax: usize, dir_pos: bool, p_exit: Point3) -> Node3Id {
        while let Some([c0, c1]) = self.get(node).children {
            let (lo, hi) = (self.get(c0).bbox, self.get(c1).bbox);
            let sax = split_axis3(lo, hi);
            node = if sax == ax {
                if dir_pos { c0 } else { c1 }
            } else if axis_coord3(p_exit, sax) < axis_min3(hi, sax) {
                c0
            } else {
                c1
            };
        }
        node
    }

    /// Parallel [`Tree3::cull_many`]: the independent, read-only queries run on
    /// rayon's thread pool. Worth it only when many queries hit a large index
    /// (see `docs/PARALLEL.md` for the measured crossover); for a handful of
    /// queries the serial version wins (no fork/join overhead). The index is
    /// shared `&self` — culls never mutate, so there is no contention.
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape3 + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
    }

    /// Batch k-NN — one result list per query point (`out[i]` for `queries[i]`).
    /// Serial; see [`Tree3::knn_many_par`].
    pub fn knn_many(&self, queries: &[Point3], k: usize) -> Vec<Vec<(f64, &T)>> {
        queries.iter().map(|&q| self.knn(q, k)).collect()
    }

    /// Parallel batch k-NN — the independent queries fan out over rayon (feature
    /// `parallel`). Like [`Tree3::cull_many_par`]: worth it for many queries over
    /// a large index; see `docs/PARALLEL.md` for the crossover.
    #[cfg(feature = "parallel")]
    pub fn knn_many_par(&self, queries: &[Point3], k: usize) -> Vec<Vec<(f64, &T)>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        queries.par_iter().map(|&q| self.knn(q, k)).collect()
    }

    fn cull_recurse<'a, S: Shape3>(
        &'a self, id: Node3Id, shape: &S, fully_inside: bool, out: &mut Vec<&'a T>,
    ) {
        let node = self.get(id);
        if fully_inside {
            match node.children {
                Some([a, b]) => {
                    self.cull_recurse(a, shape, true, out);
                    self.cull_recurse(b, shape, true, out);
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }
        match node.children {
            Some([a, b]) => {
                for child in [a, b] {
                    let cb = self.get(child).bbox;
                    match shape.classify_aabb(&cb) {
                        CellState::Out => {}
                        CellState::In => self.cull_recurse(child, shape, true, out),
                        CellState::Maybe => self.cull_recurse(child, shape, false, out),
                    }
                }
            }
            None => {
                let raster = shape.voxel_raster();
                for it in &node.items {
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

    /// The `k` nearest items to `q`, sorted ascending by distance (returns
    /// `(distance, &item)`). Best-first descent with bounding-box pruning: a
    /// bounded max-heap holds the current k best (its top is the k-th nearest
    /// so far), the nearer child is visited first to tighten the bound, and a
    /// subtree is skipped when its box's nearest point is already farther than
    /// the current k-th. Fewer than `k` items → all of them. `k == 0` → empty.
    pub fn knn(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        if k == 0 { return Vec::new(); }
        let mut heap: std::collections::BinaryHeap<KnnEntry<T>> = std::collections::BinaryHeap::new();
        self.knn_recurse(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    fn knn_recurse<'a>(&'a self, id: Node3Id, q: Point3, k: usize, heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>) {
        let node = self.get(id);
        match node.children {
            None => {
                for it in &node.items {
                    knn_offer(heap, k, it, q);
                }
            }
            Some([a, b]) => {
                let da = aabb_min_dist2(&self.get(a).bbox, q);
                let db = aabb_min_dist2(&self.get(b).bbox, q);
                let (first, dfirst, second, dsecond) = if da <= db { (a, da, b, db) } else { (b, db, a, da) };
                if dfirst < knn_worst(heap, k) { self.knn_recurse(first, q, k, heap); }
                if dsecond < knn_worst(heap, k) { self.knn_recurse(second, q, k, heap); }
            }
        }
    }
}

// ------------------------------------------------------------ k-NN helpers
// Shared by Tree3 and Octree3 (best-first nearest-neighbour search).

/// A heap entry keyed only by squared distance (items need no `Ord`). Max-heap
/// ordering so [`std::collections::BinaryHeap::peek`] is the current worst of
/// the k best — the one to evict when a closer point arrives.
pub(crate) struct KnnEntry<'a, T> {
    pub(crate) d2: f64,
    pub(crate) item: &'a T,
}
impl<T> PartialEq for KnnEntry<'_, T> { fn eq(&self, o: &Self) -> bool { self.d2 == o.d2 } }
impl<T> Eq for KnnEntry<'_, T> {}
impl<T> PartialOrd for KnnEntry<'_, T> { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl<T> Ord for KnnEntry<'_, T> { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.d2.total_cmp(&o.d2) } }

/// Current k-th-nearest squared distance (the pruning bound); `+∞` until the
/// heap is full.
#[inline]
pub(crate) fn knn_worst<T>(heap: &std::collections::BinaryHeap<KnnEntry<T>>, k: usize) -> f64 {
    if heap.len() < k { f64::INFINITY } else { heap.peek().unwrap().d2 }
}

/// Offer one item to the bounded heap: keep it if the heap isn't full yet, or
/// if it's closer than the current worst (evicting that worst).
#[inline]
pub(crate) fn knn_offer<'a, T: Positioned3>(heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>, k: usize, it: &'a T, q: Point3) {
    let p = it.position();
    let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
    let d2 = dx * dx + dy * dy + dz * dz;
    if heap.len() < k {
        heap.push(KnnEntry { d2, item: it });
    } else if d2 < heap.peek().unwrap().d2 {
        heap.pop();
        heap.push(KnnEntry { d2, item: it });
    }
}

// ------------------------------------------- nudge-free face-neighbour helpers
/// The axis a longest-axis-midpoint split cut, read back from its two child boxes
/// (`lo` = lower half, `hi` = upper): the one axis where the upper child starts
/// higher (the other two share the parent's extent, so their delta is 0).
fn split_axis3(lo: Aabb, hi: Aabb) -> usize {
    let d = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z];
    if d[0] >= d[1] && d[0] >= d[2] { 0 } else if d[1] >= d[2] { 1 } else { 2 }
}
#[inline] fn axis_min3(b: Aabb, a: usize) -> f64 { [b.x, b.y, b.z][a] }
#[inline] fn axis_coord3(p: Point3, a: usize) -> f64 { [p.x, p.y, p.z][a] }

// ------------------------------------------------------------- bulk build
// Shared top-down build for `Tree3::bulk_load` / `bulk_load_par`. Splits the
// longest axis at its midpoint — the same rule `divide` uses incrementally.

/// Longest-axis-midpoint split of `b` into (lower, upper) halves.
fn split_halves3(b: Aabb) -> (Aabb, Aabb) {
    if b.w >= b.h && b.w >= b.d {
        let h = b.w / 2.0;
        (Aabb::new(b.x, b.y, b.z, h, b.h, b.d), Aabb::new(b.x + h, b.y, b.z, h, b.h, b.d))
    } else if b.h >= b.d {
        let h = b.h / 2.0;
        (Aabb::new(b.x, b.y, b.z, b.w, h, b.d), Aabb::new(b.x, b.y + h, b.z, b.w, h, b.d))
    } else {
        let h = b.d / 2.0;
        (Aabb::new(b.x, b.y, b.z, b.w, b.h, h), Aabb::new(b.x, b.y, b.z + h, b.w, b.h, h))
    }
}

/// A node should split iff it overflows `item_limit`, hasn't hit the `min_cell`
/// floor, and its items aren't all at one point (inseparable) — mirrors `divide`.
fn splittable3<T: Positioned3>(items: &[(u32, T)], item_limit: usize, min_cell: f64, bbox: Aabb) -> bool {
    items.len() > item_limit
        && bbox.w.max(bbox.h).max(bbox.d) > min_cell
        && { let first = items[0].1.position(); !items.iter().all(|(_, it)| it.position() == first) }
}

/// Recursively build the arena in DFS order. Returns the new node's id.
fn build3_serial<T: Positioned3>(nodes: &mut Vec<Node3<T>>, locs: &mut [ItemLoc], item_limit: usize, min_cell: f64, bbox: Aabb, parent: Option<Node3Id>, items: Vec<(u32, T)>) -> Node3Id {
    let id = Node3Id(nodes.len() as u32);
    nodes.push(Node3 { bbox, parent, children: None, items: Vec::new(), hs: Vec::new() });
    if !splittable3(&items, item_limit, min_cell, bbox) {
        let node = &mut nodes[id.0 as usize];
        for (h, it) in items {
            let slot = node.items.len() as u32;
            locs[h as usize] = ItemLoc { node: id, slot };
            node.items.push(it); node.hs.push(h);
        }
        return id;
    }
    let (ab, bb) = split_halves3(bbox);
    let (mut ai, mut bi) = (Vec::new(), Vec::new());
    for (h, it) in items { if ab.contains(it.position()) { ai.push((h, it)); } else { bi.push((h, it)); } }
    let a = build3_serial(nodes, locs, item_limit, min_cell, ab, Some(id), ai);
    let b = build3_serial(nodes, locs, item_limit, min_cell, bb, Some(id), bi);
    nodes[id.0 as usize].children = Some([a, b]);
    id
}

/// A subtree built off-arena (so the recursion can fan out over threads); the
/// child boxes are recomputed on flatten (deterministic), so they aren't stored.
#[cfg(feature = "parallel")]
enum Build3<T> { Leaf(Vec<(u32, T)>), Split(Box<Build3<T>>, Box<Build3<T>>) }

/// Parallel partition (rayon `join` per split) producing a [`Build3`] tree.
#[cfg(feature = "parallel")]
fn build3_par<T: Positioned3 + Send>(item_limit: usize, min_cell: f64, bbox: Aabb, items: Vec<(u32, T)>) -> Build3<T> {
    if !splittable3(&items, item_limit, min_cell, bbox) { return Build3::Leaf(items); }
    let (ab, bb) = split_halves3(bbox);
    let (mut ai, mut bi) = (Vec::new(), Vec::new());
    for (h, it) in items { if ab.contains(it.position()) { ai.push((h, it)); } else { bi.push((h, it)); } }
    let (a, b) = rayon::join(
        || build3_par(item_limit, min_cell, ab, ai),
        || build3_par(item_limit, min_cell, bb, bi),
    );
    Build3::Split(Box::new(a), Box::new(b))
}

/// Serial flatten of a [`Build3`] tree into the arena (cheap tail after the
/// parallel partition). Recomputes the child boxes with `split_halves3`.
#[cfg(feature = "parallel")]
fn flatten3<T: Positioned3>(nodes: &mut Vec<Node3<T>>, locs: &mut [ItemLoc], bbox: Aabb, parent: Option<Node3Id>, build: Build3<T>) -> Node3Id {
    let id = Node3Id(nodes.len() as u32);
    nodes.push(Node3 { bbox, parent, children: None, items: Vec::new(), hs: Vec::new() });
    match build {
        Build3::Leaf(items) => {
            let node = &mut nodes[id.0 as usize];
            for (h, it) in items {
                let slot = node.items.len() as u32;
                locs[h as usize] = ItemLoc { node: id, slot };
                node.items.push(it); node.hs.push(h);
            }
        }
        Build3::Split(a, b) => {
            let (ab, bb) = split_halves3(bbox);
            let ca = flatten3(nodes, locs, ab, Some(id), *a);
            let cb = flatten3(nodes, locs, bb, Some(id), *b);
            nodes[id.0 as usize].children = Some([ca, cb]);
        }
    }
    id
}

/// Squared distance from `q` to the nearest point of box `b` (0 if inside).
#[inline]
pub(crate) fn aabb_min_dist2(b: &Aabb, q: Point3) -> f64 {
    let dx = if q.x < b.x { b.x - q.x } else if q.x > b.x_max() { q.x - b.x_max() } else { 0.0 };
    let dy = if q.y < b.y { b.y - q.y } else if q.y > b.y_max() { q.y - b.y_max() } else { 0.0 };
    let dz = if q.z < b.z { b.z - q.z } else if q.z > b.z_max() { q.z - b.z_max() } else { 0.0 };
    dx * dx + dy * dy + dz * dz
}

// ------------------------------------------------------------- serialization
// Little-endian primitive read/write helpers live in `crate::serde_io`, shared
// by every structure's serializer (so the byte layout is defined once).
use crate::serde_io::{corrupt, r_aabb, r_f64, r_u32, r_u64, r_u8, w_aabb, w_f64, w_u32, w_u64};

const TREE3_MAGIC: &[u8; 4] = b"VHT3";
const TREE3_VERSION: u8 = 2; // v2 adds per-leaf handle ids (the ItemRef layer)

impl<T: Positioned3> Tree3<T> {
    /// Serialize the **built** tree (exact arena, free-list, and params — no
    /// rebuild on load) to `w`. Items are written by the caller's `write_item`
    /// closure, so this is dependency-free and works for any `T`. A loader must
    /// pass a `read_item` that mirrors the same byte layout.
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(TREE3_MAGIC)?;
        w.write_all(&[TREE3_VERSION])?;
        w_u64(w, self.item_limit as u64)?;
        w_u64(w, self.merge_limit as u64)?;
        w_f64(w, self.min_cell)?;
        w_u32(w, self.root.0)?;
        w_u32(w, self.free.len() as u32)?;
        for f in &self.free { w_u32(w, f.0)?; }
        w_u32(w, self.nodes.len() as u32)?;
        for n in &self.nodes {
            w_aabb(w, &n.bbox)?;
            match n.parent {
                Some(p) => { w.write_all(&[1])?; w_u32(w, p.0)?; }
                None => w.write_all(&[0])?,
            }
            match n.children {
                Some([a, b]) => { w.write_all(&[1])?; w_u32(w, a.0)?; w_u32(w, b.0)?; }
                None => w.write_all(&[0])?,
            }
            w_u32(w, n.items.len() as u32)?;
            for it in &n.items { write_item(w, it)?; }
            // Per-leaf stable-handle ids (parallel to items) — preserves
            // ItemRefs across a save/load round-trip.
            for &h in &n.hs { w_u32(w, h)?; }
        }
        Ok(())
    }

    /// Inverse of [`Tree3::serialize`]: rebuild the exact tree from `r`, reading
    /// each item with `read_item` (must mirror the writer's layout).
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != TREE3_MAGIC { return Err(corrupt("bad Tree3 magic")); }
        if r_u8(r)? != TREE3_VERSION { return Err(corrupt("unsupported Tree3 version")); }
        let item_limit = r_u64(r)? as usize;
        let merge_limit = r_u64(r)? as usize;
        let min_cell = r_f64(r)?;
        let root = Node3Id(r_u32(r)?);
        let nfree = r_u32(r)? as usize;
        let mut free = Vec::with_capacity(nfree);
        for _ in 0..nfree { free.push(Node3Id(r_u32(r)?)); }
        let nnodes = r_u32(r)? as usize;
        let mut nodes = Vec::with_capacity(nnodes);
        for _ in 0..nnodes {
            let bbox = r_aabb(r)?;
            let parent = if r_u8(r)? == 1 { Some(Node3Id(r_u32(r)?)) } else { None };
            let children = if r_u8(r)? == 1 { Some([Node3Id(r_u32(r)?), Node3Id(r_u32(r)?)]) } else { None };
            let nitems = r_u32(r)? as usize;
            let mut items = Vec::with_capacity(nitems);
            for _ in 0..nitems { items.push(read_item(r)?); }
            let mut hs = Vec::with_capacity(nitems);
            for _ in 0..nitems { hs.push(r_u32(r)?); }
            nodes.push(Node3 { bbox, parent, children, items, hs });
        }
        if root.0 as usize >= nnodes { return Err(corrupt("root index out of range")); }
        // Rebuild the handle → location map (and the freed-handle list) from the
        // leaves' handle ids, so ItemRefs stay valid across the round-trip.
        let max_h = nodes.iter().flat_map(|n| n.hs.iter().copied()).max().map_or(0, |m| m + 1) as usize;
        let mut locs = vec![ItemLoc { node: Node3Id(0), slot: 0 }; max_h];
        let mut used = vec![false; max_h];
        for (ni, n) in nodes.iter().enumerate() {
            for (slot, &h) in n.hs.iter().enumerate() {
                locs[h as usize] = ItemLoc { node: Node3Id(ni as u32), slot: slot as u32 };
                used[h as usize] = true;
            }
        }
        let free_handles: Vec<u32> = (0..max_h as u32).filter(|&h| !used[h as usize]).collect();
        Ok(Tree3 { nodes, free, locs, free_handles, item_limit, merge_limit, min_cell, root })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct P(Point3);
    impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

    fn brute(pts: &[P], s: &Sphere3) -> Vec<usize> {
        pts.iter().enumerate()
            .filter(|(_, p)| s.contains_point(p.0))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn cull_matches_brute_force() {
        let mut x = 0x1234_5678u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000)
            .map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0)))
            .collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }
        // Tag items by index for comparison.
        for (cx, cy, cz, r) in [(128.0,128.0,128.0,40.0),(40.0,40.0,40.0,60.0),(250.0,250.0,250.0,30.0),(0.0,0.0,0.0,100.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            // brute set as positions
            let mut want: Vec<(u64,u64,u64)> = pts.iter().filter(|p| sphere.contains_point(p.0))
                .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut got: Vec<(u64,u64,u64)> = tree.cull(&sphere).iter()
                .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }
        let _ = brute(&pts, &Sphere3::new(0.0,0.0,0.0,10.0));
    }

    #[test]
    fn compact_preserves_queries_and_handles() {
        // compact() is a pure layout pass. After churn (inserts + relocations +
        // removes leave the arena scrambled with free slots), reordering must
        // change NO cull/knn result, must reclaim the free slots, and every live
        // ItemRef must still resolve to its own item.
        let mut x = 0x0BAD_C0DEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Tree3::<P>::new(world, 8);
        let mut refs: Vec<Option<ItemRef>> = Vec::new();
        let mut expected: Vec<Option<Point3>> = Vec::new();
        for _ in 0..3000 {
            let p = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
            refs.push(tree.insert_ref(P(p)));
            expected.push(Some(p));
        }
        // Churn: relocate every 3rd (scatters split slots), remove every 5th.
        for i in 0..3000 {
            if i % 3 == 0 { if let Some(r) = refs[i] {
                let np = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
                tree.update_ref(r, |q| q.0 = np); expected[i] = Some(np);
            } }
            if i % 5 == 0 { if let Some(r) = refs[i].take() { tree.remove_ref(r); expected[i] = None; } }
        }
        assert!(tree.node_count() > tree.live_node_count(), "churn should leave free slots to reclaim");

        let spheres = [(128.0,128.0,128.0,40.0),(40.0,40.0,40.0,60.0),(250.0,250.0,250.0,30.0),(0.0,0.0,0.0,100.0)];
        let snap_cull = |t: &Tree3<P>| -> Vec<Vec<(u64,u64,u64)>> {
            spheres.iter().map(|&(cx,cy,cz,r)| {
                let s = Sphere3::new(cx,cy,cz,r).with_raster();
                let mut v: Vec<(u64,u64,u64)> = t.cull(&s).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
                v.sort(); v
            }).collect()
        };
        let queries = [Point3::new(50.0,60.0,70.0), Point3::new(200.0,200.0,10.0), Point3::new(128.0,128.0,128.0)];
        let snap_knn = |t: &Tree3<P>| -> Vec<Vec<u64>> {
            queries.iter().map(|&q| { let mut d: Vec<u64> = t.knn(q, 12).iter().map(|(dist,_)| dist.to_bits()).collect(); d.sort(); d }).collect()
        };
        let cull_before = snap_cull(&tree);
        let knn_before = snap_knn(&tree);

        tree.compact();

        assert_eq!(tree.node_count(), tree.live_node_count(), "compact must reclaim every free slot");
        assert_eq!(cull_before, snap_cull(&tree), "compact changed a cull result");
        assert_eq!(knn_before, snap_knn(&tree), "compact changed a knn result");
        // Every live handle still resolves to exactly its own (possibly relocated) item.
        for i in 0..3000 {
            if let Some(r) = refs[i] {
                let got = tree.remove_ref(r).expect("live handle must resolve after compact").0;
                let want = expected[i].expect("live handle should have an expected position");
                assert_eq!((got.x.to_bits(), got.y.to_bits(), got.z.to_bits()),
                           (want.x.to_bits(), want.y.to_bits(), want.z.to_bits()),
                           "handle {i} resolved to the wrong item after compact");
            }
        }
    }

    #[test]
    fn bulk_load_matches_insert_and_brute_force() {
        // A tree built with `bulk_load` (one top-down partition) must answer
        // `cull` exactly like an insert-by-insert tree and brute force, and its
        // handles must be stable: `ItemRef(i)` addresses `items[i]`.
        let mut x = 0xC0FF_EE42u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let pts: Vec<P> = (0..3000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let bulk = Tree3::<P>::bulk_load(world, 8, pts.clone());
        for (cx, cy, cz, r) in [(128.0,128.0,128.0,40.0),(40.0,40.0,40.0,60.0),(250.0,250.0,250.0,30.0),(0.0,0.0,0.0,100.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<(u64,u64,u64)> = pts.iter().filter(|p| sphere.contains_point(p.0))
                .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut got: Vec<(u64,u64,u64)> = bulk.cull(&sphere).iter()
                .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "bulk_load cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }
        assert_eq!(bulk.item_count(), pts.len(), "bulk_load dropped items");
        // Handle stability: remove_ref(i) yields exactly items[i].
        let mut t = Tree3::<P>::bulk_load(world, 8, pts.clone());
        for i in (0..pts.len()).step_by(97) {
            let got = t.remove_ref(ItemRef(i as u32)).expect("handle i must resolve");
            assert_eq!(got.0.x.to_bits(), pts[i].0.x.to_bits(), "ItemRef({i}) addressed the wrong item");
            assert_eq!(got.0.z.to_bits(), pts[i].0.z.to_bits(), "ItemRef({i}) addressed the wrong item");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn bulk_load_par_matches_serial() {
        // The parallel partition must produce a structurally identical tree to
        // the serial one — same cull answers, same stable handles.
        let mut x = 0x51E6_E00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let pts: Vec<P> = (0..8000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let ser = Tree3::<P>::bulk_load(world, 8, pts.clone());
        let par = Tree3::<P>::bulk_load_par(world, 8, pts.clone());
        for (cx, cy, cz, r) in [(128.0,128.0,128.0,50.0),(200.0,60.0,90.0,45.0),(0.0,0.0,0.0,300.0)] {
            let s = Sphere3::new(cx, cy, cz, r);
            let sphere = if r <= 100.0 { s.with_raster() } else { s }; // don't raster a world-covering sphere
            let mut a: Vec<(u64,u64,u64)> = ser.cull(&sphere).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut b: Vec<(u64,u64,u64)> = par.cull(&sphere).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "bulk_load_par cull != serial for sphere ({cx},{cy},{cz}) r={r}");
        }
        // Handles must map the same way in both.
        let mut sr = Tree3::<P>::bulk_load(world, 8, pts.clone());
        let mut pr = Tree3::<P>::bulk_load_par(world, 8, pts.clone());
        for i in (0..pts.len()).step_by(131) {
            let a = sr.remove_ref(ItemRef(i as u32)).unwrap();
            let b = pr.remove_ref(ItemRef(i as u32)).unwrap();
            assert_eq!(a.0.x.to_bits(), b.0.x.to_bits(), "par handle {i} differs from serial");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn knn_many_par_matches_serial() {
        let mut x = 0x9A3C_77E1u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..5000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }
        let qs: Vec<Point3> = (0..200).map(|_| Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0)).collect();
        let s = tree.knn_many(&qs, 8);
        let p = tree.knn_many_par(&qs, 8);
        assert_eq!(s.len(), p.len());
        for (a, b) in s.iter().zip(p.iter()) {
            let da: Vec<u64> = a.iter().map(|(d, _)| d.to_bits()).collect();
            let db: Vec<u64> = b.iter().map(|(d, _)| d.to_bits()).collect();
            assert_eq!(da, db, "knn_many_par distances differ from serial");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn cull_many_par_matches_serial() {
        // The parallel batch cull must return exactly what the serial batch
        // does, query for query — it is the same `cull` fanned over rayon.
        let mut x = 0x0BAD_F00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..5000)
            .map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0)))
            .collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }
        let shapes: Vec<Sphere3> = (0..200)
            .map(|_| Sphere3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0, 8.0 + rng() * 40.0).with_raster())
            .collect();
        let serial = tree.cull_many(&shapes);
        let par = tree.cull_many_par(&shapes);
        assert_eq!(serial.len(), par.len());
        for (i, (s, p)) in serial.iter().zip(par.iter()).enumerate() {
            let mut a: Vec<(u64,u64,u64)> = s.iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut b: Vec<(u64,u64,u64)> = p.iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "parallel batch cull disagrees with serial at query {i}");
        }
    }

    #[test]
    fn raycast_thick_matches_brute_and_sorted() {
        // The thick ray-cast must return exactly the points within `radius` of
        // the ray segment, sorted by distance along the ray.
        let mut x = 0x0FAB_CA57u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..4000).map(|_| P(Point3::new(rng() * 100.0, rng() * 100.0, rng() * 100.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0), 8);
        for p in &pts { tree.insert(*p); }
        let (origin, dir, max_dist, radius) = (Point3::new(0.0, 50.0, 50.0), Point3::new(1.0, 0.0, 0.0), 100.0, 6.0);
        // Brute force: project onto the ray, clamp, measure perpendicular distance.
        let (ux, uy, uz) = (1.0, 0.0, 0.0);
        let mut want: Vec<(u64, u64, u64)> = pts.iter().filter(|p| {
            let q = p.0;
            let t = ((q.x - origin.x) * ux + (q.y - origin.y) * uy + (q.z - origin.z) * uz).clamp(0.0, max_dist);
            let (cx, cy, cz) = (origin.x + ux * t, origin.y + uy * t, origin.z + uz * t);
            let (dx, dy, dz) = (q.x - cx, q.y - cy, q.z - cz);
            dx * dx + dy * dy + dz * dz <= radius * radius
        }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        let hits = tree.raycast(origin, dir, max_dist, radius);
        // sorted ascending by t
        for w in hits.windows(2) { assert!(w[0].0 <= w[1].0, "raycast hits not sorted by t"); }
        let mut got: Vec<(u64, u64, u64)> = hits.iter().map(|(_, p)| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "raycast set != brute capsule");
    }

    #[test]
    fn raycast_dda_subset_of_capsule_and_first() {
        // The variable-cell DDA hits must be a subset of the exact capsule
        // raycast (no invented items / wrong cells), sorted; first == nearest.
        let mut x = 0x5DDA_3D11u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..6000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 6);
        for p in &pts { tree.insert(*p); }
        let mut any = false;
        for (o, d, mt, r) in [
            (Point3::new(10.0, 10.0, 10.0), Point3::new(1.0, 0.8, 0.6), 400.0, 8.0),
            (Point3::new(250.0, 10.0, 128.0), Point3::new(-1.0, 1.0, 0.2), 400.0, 12.0),
            (Point3::new(0.0, 128.0, 128.0), Point3::new(1.0, 0.0, 0.0), 256.0, 20.0),
        ] {
            let dda = tree.raycast_dda(o, d, mt, r);
            for w in dda.hits.windows(2) { assert!(w[0].0 <= w[1].0, "not sorted"); }
            let cap: std::collections::HashSet<(u64, u64, u64)> = tree.raycast(o, d, mt, r).iter().map(|(_, p)| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            for (_, p) in &dda.hits {
                assert!(cap.contains(&(p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())), "DDA hit not in the exact capsule set");
                any = true;
            }
            match (tree.raycast_dda_first(o, d, mt, r), dda.hits.first()) {
                (None, None) => {}
                (Some((t1, _)), Some(&(t0, _))) => assert!((t1 - t0).abs() < 1e-9, "first {t1} != nearest {t0}"),
                _ => panic!("raycast_dda_first / raycast_dda nearest disagree"),
            }
        }
        assert!(any, "DDA found nothing — likely a traversal bug");
    }

    #[test]
    fn raycast_dda_lca_visits_every_crossed_leaf() {
        // Completeness gate for the nudge-free (ascend-to-LCA) walk, independent of
        // the old nudge: densely sample the centre ray, `locate` each interior
        // sample, and assert every such leaf is one the walk actually stepped into.
        // (A leaf the ray demonstrably passes through must never be skipped.)
        let mut x = 0x11CA_7EE1u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..8000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 5);
        for p in &pts { tree.insert(*p); }
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        for (o, d, mt) in [
            (Point3::new(-40.0, 30.0, 20.0), Point3::new(1.0, 0.61, 0.37), 500.0),
            (Point3::new(300.0, 200.0, 130.0), Point3::new(-1.0, -0.43, 0.29), 600.0),
            (Point3::new(128.0, -20.0, 300.0), Point3::new(0.13, 1.0, -0.77), 600.0),
            (Point3::new(5.0, 5.0, 5.0), Point3::new(0.9, 0.7, 1.0), 500.0),
        ] {
            let m = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            let (ux, uy, uz) = (d.x / m, d.y / m, d.z / m);
            let mut leaf = match tree.raycast_start_leaf(o, ux, uy, uz, mt) { Some(l) => l, None => continue };
            let mut visited: Vec<Node3Id> = vec![leaf];
            let mut guard = 0usize;
            loop {
                guard += 1;
                if guard > tree.nodes.len() * 3 + 32 { break; }
                match tree.ray_step3_lca(leaf, o, ux, uy, uz, mt) {
                    Some((_, n)) => { leaf = n; if !visited.contains(&n) { visited.push(n); } }
                    None => break,
                }
            }
            let steps = 4000;
            for i in 0..=steps {
                let t = mt * i as f64 / steps as f64;
                let p = Point3::new(o.x + ux * t, o.y + uy * t, o.z + uz * t);
                if world.contains(p) {
                    let l = tree.locate(p);
                    assert!(visited.contains(&l), "LCA walk skipped a leaf the ray crosses (t={t})");
                }
            }
        }
    }

    #[test]
    fn raycast_dda_lca_fuzz_across_configs() {
        // Broadens the completeness gate (deterministic seed → not flaky): varied
        // item_limits + point distributions (uniform / clustered → deep, uneven
        // subdivision that stresses the ascend-to-LCA) + many random rays.
        let mut x = 0x5A3E_7F0Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let w = 256.0;
        let world = Aabb::new(0.0, 0.0, 0.0, w, w, w);
        for &il in &[3usize, 6, 12] {
            for clustered in [false, true] {
                let pts: Vec<P> = (0..5000).map(|_| {
                    if clustered {
                        let cx = (rng() * 6.0) as i64 as f64 * 40.0 + 20.0;
                        let cy = (rng() * 5.0) as i64 as f64 * 45.0 + 20.0;
                        P(Point3::new((cx + (rng() - 0.5) * 30.0).clamp(0.0, w), (cy + (rng() - 0.5) * 30.0).clamp(0.0, w), rng() * w))
                    } else {
                        P(Point3::new(rng() * w, rng() * w, rng() * w))
                    }
                }).collect();
                let mut tree = Tree3::<P>::new(world, il);
                for p in &pts { tree.insert(*p); }
                for _ in 0..30 {
                    let o = Point3::new(rng() * w * 1.4 - w * 0.2, rng() * w * 1.4 - w * 0.2, rng() * w * 1.4 - w * 0.2);
                    let d = Point3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5);
                    let m = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
                    if m < 1e-3 { continue; }
                    let (ux, uy, uz) = (d.x / m, d.y / m, d.z / m);
                    let mt = 800.0;
                    let mut leaf = match tree.raycast_start_leaf(o, ux, uy, uz, mt) { Some(l) => l, None => continue };
                    let mut visited: Vec<Node3Id> = vec![leaf];
                    let mut guard = 0usize;
                    loop {
                        guard += 1;
                        if guard > tree.nodes.len() * 3 + 32 { break; }
                        match tree.ray_step3_lca(leaf, o, ux, uy, uz, mt) {
                            Some((_, n)) => { leaf = n; if !visited.contains(&n) { visited.push(n); } }
                            None => break,
                        }
                    }
                    let steps = 1600;
                    for i in 0..=steps {
                        let t = mt * i as f64 / steps as f64;
                        let p = Point3::new(o.x + ux * t, o.y + uy * t, o.z + uz * t);
                        if world.contains(p) {
                            assert!(visited.contains(&tree.locate(p)), "LCA fuzz skipped a crossed leaf (il={il}, clustered={clustered})");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn frustum_from_corners_box_recovers_faces() {
        // An axis-aligned box passed as 8 corners must cull exactly the points
        // inside that box — the derived six planes are the box faces.
        let corners = [
            Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 10.0, 0.0), Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 10.0), Point3::new(10.0, 0.0, 10.0), Point3::new(10.0, 10.0, 10.0), Point3::new(0.0, 10.0, 10.0),
        ];
        let poly = Polyhedron3::from_corners(corners);
        let mut x = 0x00C0_FFEEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000).map(|_| P(Point3::new(rng() * 20.0 - 5.0, rng() * 20.0 - 5.0, rng() * 20.0 - 5.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(-5.0, -5.0, -5.0, 30.0, 30.0, 30.0), 8);
        for p in &pts { tree.insert(*p); }
        let inside = |q: Point3| q.x >= 0.0 && q.x <= 10.0 && q.y >= 0.0 && q.y <= 10.0 && q.z >= 0.0 && q.z <= 10.0;
        let mut want: Vec<(u64, u64, u64)> = pts.iter().filter(|p| inside(p.0)).map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        let mut got: Vec<(u64, u64, u64)> = tree.cull(&poly).iter().map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "frustum-from-box cull != box contains");
    }

    #[test]
    fn segment_hit_matches_box_slab_reference() {
        // segment_hit on a general convex polytope must agree with an INDEPENDENT
        // analytic slab (ray-box) clip when the polytope IS an axis-aligned box.
        // Both are exact (no sampling) → deterministic entry-t agreement.
        let (lo, hi) = (Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 8.0, 12.0));
        let corners = [
            Point3::new(lo.x, lo.y, lo.z), Point3::new(hi.x, lo.y, lo.z), Point3::new(hi.x, hi.y, lo.z), Point3::new(lo.x, hi.y, lo.z),
            Point3::new(lo.x, lo.y, hi.z), Point3::new(hi.x, lo.y, hi.z), Point3::new(hi.x, hi.y, hi.z), Point3::new(lo.x, hi.y, hi.z),
        ];
        let poly = Polyhedron3::from_corners(corners);
        // reference: standard slab clip of the segment a→b against [lo,hi].
        let slab = |a: Point3, b: Point3| -> Option<f64> {
            let d = [b.x - a.x, b.y - a.y, b.z - a.z];
            let ac = [a.x, a.y, a.z]; let l = [lo.x, lo.y, lo.z]; let h = [hi.x, hi.y, hi.z];
            let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
            for i in 0..3 {
                if d[i].abs() < 1e-12 { if ac[i] < l[i] || ac[i] > h[i] { return None; } }
                else {
                    let (mut tn, mut tf) = ((l[i] - ac[i]) / d[i], (h[i] - ac[i]) / d[i]);
                    if tn > tf { std::mem::swap(&mut tn, &mut tf); }
                    t0 = t0.max(tn); t1 = t1.min(tf);
                    if t0 > t1 { return None; }
                }
            }
            Some(t0)
        };
        let mut x = 0xF00D_CAFEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut hits = 0;
        for _ in 0..20_000 {
            let a = Point3::new(rng() * 20.0 - 5.0, rng() * 18.0 - 5.0, rng() * 22.0 - 5.0);
            let b = Point3::new(rng() * 20.0 - 5.0, rng() * 18.0 - 5.0, rng() * 22.0 - 5.0);
            match (poly.segment_hit(a, b), slab(a, b)) {
                (None, None) => {}
                (Some(t), Some(tr)) => { assert!((t - tr).abs() < 1e-6, "entry t {t} != slab {tr}"); hits += 1; }
                (g, r) => panic!("segment_hit {g:?} disagrees with slab {r:?} for {a:?}->{b:?}"),
            }
        }
        assert!(hits > 1000, "test should exercise many real occlusions (got {hits})");
    }

    #[test]
    fn inflated_is_superset_of_r_neighbourhood() {
        // Polyhedron3::inflated(r) must contain every point within r of the solid
        // (no false negatives) and stay tight (⊆ the L∞-r shell ⊆ r√3). Reference:
        // exact L2 distance from a point to an axis-aligned box.
        let (lo, hi) = (Point3::new(20.0, 10.0, 30.0), Point3::new(60.0, 50.0, 70.0));
        let corners = [
            Point3::new(lo.x, lo.y, lo.z), Point3::new(hi.x, lo.y, lo.z), Point3::new(hi.x, hi.y, lo.z), Point3::new(lo.x, hi.y, lo.z),
            Point3::new(lo.x, lo.y, hi.z), Point3::new(hi.x, lo.y, hi.z), Point3::new(hi.x, hi.y, hi.z), Point3::new(lo.x, hi.y, hi.z),
        ];
        let poly = Polyhedron3::from_corners(corners);
        let r = 8.0;
        let inf = poly.inflated(r);
        let box_dist = |p: Point3| -> f64 { let (cx, cy, cz) = (p.x.clamp(lo.x, hi.x), p.y.clamp(lo.y, hi.y), p.z.clamp(lo.z, hi.z)); ((p.x - cx).powi(2) + (p.y - cy).powi(2) + (p.z - cz).powi(2)).sqrt() };
        let mut x = 0x0D11_A7EDu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut near = 0;
        for _ in 0..80_000 {
            let p = Point3::new(rng() * 100.0 - 5.0, rng() * 90.0 - 5.0, rng() * 110.0 - 5.0);
            let d = box_dist(p);
            let cont = inf.contains_point(p);
            if d <= r - 1e-6 { assert!(cont, "false negative: dist {d} ≤ r {r} but not in inflated"); near += 1; }
            if cont { assert!(d <= r * 3f64.sqrt() + 1e-6, "too loose: in inflated but dist {d} > r√3"); }
        }
        // r=0 is the identity (same in/out as the original) on a sample
        let z = poly.inflated(0.0);
        for _ in 0..5_000 { let p = Point3::new(rng() * 100.0 - 5.0, rng() * 90.0 - 5.0, rng() * 110.0 - 5.0); assert_eq!(poly.contains_point(p), z.contains_point(p)); }
        assert!(near > 2000, "should exercise many within-r points (got {near})");
    }

    #[test]
    fn clear_empties_and_refills() {
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut t = Tree3::<P>::new(world, 4);
        for i in 0..500u32 { t.insert(P(Point3::new((i % 16) as f64 * 15.0, (i / 16 % 16) as f64 * 15.0, 10.0))); }
        assert!(t.item_count() > 0 && t.leaf_count() > 1, "tree should have split");
        t.clear();
        assert_eq!(t.item_count(), 0);
        assert_eq!(t.leaf_count(), 1, "clear leaves a single root leaf");
        assert!(t.cull(&Sphere3::new(128.0, 128.0, 128.0, 1000.0)).is_empty());
        // Refilling after clear works (and reuses handles from 0).
        let r0 = t.insert_ref(P(Point3::new(50.0, 50.0, 50.0))).unwrap();
        assert_eq!(r0.0, 0);
        for i in 0..300u32 { t.insert(P(Point3::new((i % 10) as f64 * 20.0, 20.0, 20.0))); }
        assert_eq!(t.item_count(), 301);
        assert!(!t.cull(&Sphere3::new(50.0, 50.0, 50.0, 5.0)).is_empty());
    }

    #[test]
    fn cull_matches_brute_after_churn() {
        // Deep dynamic test: build, churn with update/remove/insert, then
        // verify cull still equals brute force and the item count is sane.
        let mut x = 0xC0FFEEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };

        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Tree3::<M>::new(world, 6);
        // Shadow map id -> current position (ground truth).
        let mut live: std::collections::HashMap<u32, Point3> = std::collections::HashMap::new();
        let mut next_id = 0u32;

        let rp = |rng: &mut dyn FnMut() -> f64| Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);

        // Seed.
        for _ in 0..2000 {
            let p = rp(&mut rng);
            tree.insert(M { id: next_id, p });
            live.insert(next_id, p);
            next_id += 1;
        }

        // Churn: mix of moves, removes, inserts.
        for _ in 0..6000 {
            let roll = rng();
            if roll < 0.6 && !live.is_empty() {
                // move a random live item
                let ids: Vec<u32> = live.keys().copied().collect();
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id];
                let np = rp(&mut rng);
                let ok = tree.update(old, |m| m.id == id, |m| m.p = np);
                if ok { live.insert(id, np); }
                else { live.remove(&id); } // pushed out (shouldn't happen in-bounds) — keep consistent
            } else if roll < 0.8 && !live.is_empty() {
                let ids: Vec<u32> = live.keys().copied().collect();
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id];
                tree.remove(old, |m| m.id == id);
                live.remove(&id);
            } else {
                let p = rp(&mut rng);
                tree.insert(M { id: next_id, p });
                live.insert(next_id, p);
                next_id += 1;
            }
        }

        assert_eq!(tree.item_count(), live.len(), "item count drifted from ground truth after churn");

        // Cull equality vs brute force on the live set.
        for (cx, cy, cz, r) in [(128.0,128.0,128.0,30.0),(60.0,200.0,90.0,50.0),(10.0,10.0,10.0,80.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<u32> = live.iter()
                .filter(|(_, p)| { let dx=p.x-cx; let dy=p.y-cy; let dz=p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
                .map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = tree.cull(&sphere).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "post-churn cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }
    }

    #[test]
    fn voxel_raster_classifies_sphere() {
        let raster = VoxelRaster::for_sphere(10.0, 10.0, 10.0, 5.0);
        // Centre voxel must be In, a far voxel Out.
        assert_eq!(raster.cell_at_world(Point3::new(10.0, 10.0, 10.0)), CellState::In);
        assert_eq!(raster.cell_at_world(Point3::new(100.0, 100.0, 100.0)), CellState::Out);
    }

    #[test]
    fn polyhedron_cull_matches_brute_and_raster_agrees() {
        // A faceted ball (many-faced convex polyhedron) culled by Tree3 must
        // match brute force, and its 1×1×1 raster must agree with the
        // analytic contains_point everywhere off the boundary.
        let mut x = 0xBEEF1234u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000)
            .map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0)))
            .collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 16);
        for p in &pts { tree.insert(*p); }

        let poly = Polyhedron3::faceted_ball(120.0, 130.0, 110.0, 55.0, 48).with_raster();
        let mut want: Vec<(u64, u64, u64)> = pts.iter().filter(|p| poly.contains_point(p.0))
            .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        let mut got: Vec<(u64, u64, u64)> = tree.cull(&poly).iter()
            .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "polyhedron cull != brute force");

        // Raster vs analytic, away from the voxel-boundary halo.
        let raster = poly.raster.as_ref().unwrap();
        for p in &pts {
            match raster.cell_at_world(p.0) {
                CellState::In => assert!(poly.contains_point(p.0)),
                CellState::Out => assert!(!poly.contains_point(p.0)),
                CellState::Maybe => {} // boundary voxel — either answer allowed
            }
        }
    }

    #[test]
    fn knn_matches_brute_force() {
        // The k nearest by Tree3::knn must have the same k smallest distances as
        // a full brute-force sort, for varied query points and k. (The set of
        // distances is unique even if two items tie at the k-boundary.)
        let mut x = 0x7E57_C0DEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..4000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }

        let d2 = |a: Point3, q: Point3| { let (dx, dy, dz) = (a.x - q.x, a.y - q.y, a.z - q.z); dx * dx + dy * dy + dz * dz };
        for _ in 0..40 {
            let q = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
            for k in [1usize, 5, 17] {
                let got = tree.knn(q, k);
                assert_eq!(got.len(), k.min(pts.len()));
                assert!(got.windows(2).all(|w| w[0].0 <= w[1].0), "knn result not sorted ascending");
                let mut bf: Vec<f64> = pts.iter().map(|p| d2(p.0, q)).collect();
                bf.sort_by(|a, b| a.total_cmp(b));
                for (i, (dist, _)) in got.iter().enumerate() {
                    assert!((dist * dist - bf[i]).abs() <= 1e-6 * (1.0 + bf[i]),
                        "knn dist #{i} mismatch: got {} vs brute {}", dist * dist, bf[i]);
                }
            }
        }
        // k == 0 → empty; k > n → all.
        assert!(tree.knn(Point3::new(1.0, 1.0, 1.0), 0).is_empty());
        assert_eq!(tree.knn(Point3::new(1.0, 1.0, 1.0), pts.len() + 10).len(), pts.len());
    }

    #[test]
    fn serialize_roundtrip_preserves_tree() {
        use std::io::{Cursor, Read, Write};

        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        // Build with churn so the free-list is non-empty and the arena has dead
        // slots — the serializer must round-trip the exact arena, not a rebuild.
        let mut x = 0x5E21_A112u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Tree3::<M>::new(world, 6);
        // BTreeMap (not HashMap) so selection order is deterministic — the test
        // must not depend on randomized hash iteration order.
        let mut live: std::collections::BTreeMap<u32, Point3> = std::collections::BTreeMap::new();
        let mut next = 0u32;
        for _ in 0..1500 {
            let p = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
            tree.insert(M { id: next, p }); live.insert(next, p); next += 1;
        }
        for _ in 0..3000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.5 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id]; let np = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
                if tree.update(old, |m| m.id == id, |m| m.p = np) { live.insert(id, np); }
            } else if roll < 0.7 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                tree.remove(live[&id], |m| m.id == id); live.remove(&id);
            } else {
                let p = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
                tree.insert(M { id: next, p }); live.insert(next, p); next += 1;
            }
        }
        // Force dead slots deterministically: remove ~40% (no inserts after, so
        // the merge-freed slots stay in the free-list) — this guarantees the
        // serializer's free-list path is exercised regardless of churn outcome.
        let doomed: Vec<u32> = live.keys().copied().take(live.len() * 2 / 5).collect();
        for id in doomed {
            tree.remove(live[&id], |m| m.id == id);
            live.remove(&id);
        }
        assert!(tree.node_count() > tree.live_node_count(), "expected dead slots from churn + removals");

        // Serialize → deserialize with a matching item codec (id + x,y,z).
        let mut buf = Vec::new();
        tree.serialize(&mut buf, |w, it| {
            w.write_all(&it.id.to_le_bytes())?;
            w.write_all(&it.p.x.to_le_bytes())?;
            w.write_all(&it.p.y.to_le_bytes())?;
            w.write_all(&it.p.z.to_le_bytes())
        }).unwrap();
        let mut cur = Cursor::new(&buf);
        let loaded = Tree3::<M>::deserialize(&mut cur, |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?;
            let mut b = [0u8; 8];
            let id = u32::from_le_bytes(a);
            r.read_exact(&mut b)?; let px = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let py = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let pz = f64::from_le_bytes(b);
            Ok(M { id, p: Point3::new(px, py, pz) })
        }).unwrap();

        // Exact arena + reachable structure preserved.
        assert_eq!(loaded.node_count(), tree.node_count(), "arena size");
        assert_eq!(loaded.live_node_count(), tree.live_node_count(), "live nodes");
        assert_eq!(loaded.leaf_count(), tree.leaf_count(), "leaves");
        assert_eq!(loaded.item_count(), tree.item_count(), "items");

        // Culls identical for several spheres.
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 30.0), (60.0, 200.0, 90.0, 50.0), (10.0, 10.0, 10.0, 80.0)] {
            let s = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut a: Vec<u32> = tree.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = loaded.cull(&s).iter().map(|m| m.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "cull differs after round-trip ({cx},{cy},{cz}) r={r}");
        }
        // And knn identical.
        let ka: Vec<f64> = tree.knn(Point3::new(120.0, 120.0, 120.0), 8).iter().map(|(d, _)| *d).collect();
        let kb: Vec<f64> = loaded.knn(Point3::new(120.0, 120.0, 120.0), 8).iter().map(|(d, _)| *d).collect();
        assert_eq!(ka, kb, "knn differs after round-trip");

        // Corruption is rejected, not panicked.
        assert!(Tree3::<M>::deserialize(&mut Cursor::new(&b"XXXXX"[..]), |_| unreachable!()).is_err());
    }

    #[test]
    fn update_ref_churn_matches_brute() {
        // Build with insert_ref, churn with update_ref / remove_ref / insert_ref
        // (the O(1) stable-handle path), and verify the cull still equals brute
        // force and the item count tracks ground truth — i.e. the handle layer's
        // location bookkeeping stays consistent through splits and merges.
        use std::io::{Cursor, Read, Write};
        let mut x = 0x1772_EF00u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };

        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Tree3::<M>::new(world, 6);
        let rp = |rng: &mut dyn FnMut() -> f64| Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
        // id -> (handle, ground-truth position)
        let mut live: std::collections::HashMap<u32, (ItemRef, Point3)> = std::collections::HashMap::new();
        let mut next = 0u32;
        for _ in 0..2000 {
            let p = rp(&mut rng);
            let r = tree.insert_ref(M { id: next, p }).unwrap();
            live.insert(next, (r, p));
            next += 1;
        }
        for _ in 0..6000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.6 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let (r, _) = live[&id];
                let np = rp(&mut rng);
                assert!(tree.update_ref(r, |m| m.p = np), "in-bounds update_ref should not drop");
                live.get_mut(&id).unwrap().1 = np;
            } else if roll < 0.8 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let (r, _) = live[&id];
                tree.remove_ref(r);
                live.remove(&id);
            } else {
                let p = rp(&mut rng);
                let r = tree.insert_ref(M { id: next, p }).unwrap();
                live.insert(next, (r, p));
                next += 1;
            }
        }

        assert_eq!(tree.item_count(), live.len(), "item count drifted under handle churn");
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 30.0), (60.0, 200.0, 90.0, 50.0), (10.0, 10.0, 10.0, 80.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<u32> = live.iter()
                .filter(|(_, (_, p))| { let dx = p.x - cx; let dy = p.y - cy; let dz = p.z - cz; dx * dx + dy * dy + dz * dz <= r * r })
                .map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = tree.cull(&sphere).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "handle-churn cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }

        // ItemRefs survive a serialize round-trip: a kept handle still addresses
        // its item in the loaded tree. Pick one, teleport it via the loaded
        // tree's update_ref, and confirm the cull sees it at the new spot only.
        let (&id0, &(r0, p0)) = live.iter().next().unwrap();
        let mut buf = Vec::new();
        tree.serialize(&mut buf, |w, it| {
            w.write_all(&it.id.to_le_bytes())?;
            w.write_all(&it.p.x.to_le_bytes())?; w.write_all(&it.p.y.to_le_bytes())?; w.write_all(&it.p.z.to_le_bytes())
        }).unwrap();
        let mut loaded = Tree3::<M>::deserialize(&mut Cursor::new(&buf), |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?; let id = u32::from_le_bytes(a);
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?; let px = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let py = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let pz = f64::from_le_bytes(b);
            Ok(M { id, p: Point3::new(px, py, pz) })
        }).unwrap();
        // r0 must still point at id0 in the loaded tree.
        let mut seen = None;
        loaded.update_ref(r0, |m| seen = Some(m.id));
        assert_eq!(seen, Some(id0), "ItemRef did not survive the round-trip");
        let _ = p0;
    }

    #[test]
    fn stale_item_ref_is_inert_not_corrupting() {
        // A handle is FREED when its item leaves the root (update_ref → false) or is
        // removed. The caller may still hold that ItemRef. Before the DEAD_HANDLE
        // marker its stale location still named a live (node, slot) that swap_remove
        // had refilled with a DIFFERENT item, so reusing it silently mutated/removed
        // the wrong item — or panicked on a shrunk leaf. It must now be inert.
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        let world = Aabb::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
        let mut t = Tree3::<M>::new(world, 4);
        // Enough items to force splits, all in one corner region so they share leaves.
        let refs: Vec<ItemRef> = (0..40u32)
            .map(|id| t.insert_ref(M { id, p: Point3::new(1.0 + (id % 8) as f64, 1.0 + (id / 8) as f64, 1.0) }).unwrap())
            .collect();
        let before = t.item_count();

        // 1) Push item 0 out of the world: the update fails and the handle is freed.
        assert!(!t.update_ref(refs[0], |m| m.p = Point3::new(-50.0, -50.0, -50.0)), "leaving the root must report false");
        assert_eq!(t.item_count(), before - 1, "the item that left is dropped");

        // 2) Reusing the stale handle must be INERT — no panic, no aliasing.
        let mut touched = None;
        assert!(!t.update_ref(refs[0], |m| touched = Some(m.id)), "a stale handle must report false");
        assert_eq!(touched, None, "a stale handle must not reach ANY item (it aliased one before the fix)");
        assert!(t.remove_ref(refs[0]).is_none(), "a stale handle must remove nothing");
        assert!(matches!(t.update_ref_tracked(refs[0], |_| {}), Crossing::Left), "a stale handle reports Left");
        assert_eq!(t.item_count(), before - 1, "no extra item may vanish");

        // 3) Every surviving item is untouched and still reachable by its own handle.
        for (id, &r) in refs.iter().enumerate().skip(1) {
            let mut got = None;
            assert!(t.update_ref(r, |m| got = Some(m.id)), "live handle {id} broke");
            assert_eq!(got, Some(id as u32), "handle {id} now points at the wrong item");
        }

        // 4) remove_ref frees too — the second use is equally inert.
        assert!(t.remove_ref(refs[1]).is_some());
        assert!(t.remove_ref(refs[1]).is_none(), "double remove_ref must be a no-op");
        assert!(!t.update_ref(refs[1], |_| panic!("must not reach an item")));
        assert_eq!(t.item_count(), before - 2);

        // 5) A handle from a DIFFERENT tree (out of range here) is inert, not a panic.
        let mut other = Tree3::<M>::new(world, 4);
        let far = other.insert_ref(M { id: 999, p: Point3::new(9.0, 9.0, 9.0) }).unwrap();
        let mut empty = Tree3::<M>::new(world, 4);
        assert!(!empty.update_ref(far, |_| panic!("must not reach an item")), "foreign handle must be inert");
        assert!(empty.remove_ref(far).is_none());
    }

    #[test]
    fn update_ref_tracked_reports_leaf_crossings() {
        // The leaf-crossing signal: a tiny move inside a leaf reports Stayed; a
        // jump across the tree reports Moved{from,to} with a genuinely different
        // destination leaf; leaving the world reports Left and frees the handle.
        // And it must stay consistent with `update_ref`.
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut x = 0xC0FFEEu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let mut tree = Tree3::<P>::new(world, 4);
        // enough points to force real subdivision (many leaves to cross)
        let mut handles = Vec::new();
        for _ in 0..4000 { let p = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0); handles.push(tree.insert_ref(P(p)).unwrap()); }

        // Leaving the position unchanged keeps the item in its own leaf → Stayed.
        let r = handles[0];
        let from0 = tree.ref_leaf(r);
        let stayed = tree.update_ref_tracked(r, |_it| { /* no move */ });
        assert_eq!(stayed, Crossing::Stayed(from0), "a no-op move must be Stayed in the same leaf");

        // A teleport across the world almost always lands in a different leaf.
        let mut moved = 0;
        for &h in handles.iter().take(500) {
            let from = tree.ref_leaf(h);
            let np = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
            match tree.update_ref_tracked(h, |it| it.0 = np) {
                Crossing::Moved { from: f, to } => { assert_eq!(f, from, "reported `from` must be the pre-move leaf"); assert_ne!(f, to, "Moved must have distinct leaves"); moved += 1; }
                Crossing::Stayed(l) => assert_eq!(l, from, "Stayed must keep the same leaf"),
                Crossing::Left => panic!("in-bounds teleport should not leave the world"),
            }
        }
        assert!(moved > 400, "most cross-world teleports should cross a leaf ({moved}/500)");

        // Leaving the world frees the handle → Left, and matches update_ref=false.
        let hlast = *handles.last().unwrap();
        let left = tree.update_ref_tracked(hlast, |it| it.0 = Point3::new(1e6, 1e6, 1e6));
        assert_eq!(left, Crossing::Left, "out-of-world move must report Left");
    }
}
