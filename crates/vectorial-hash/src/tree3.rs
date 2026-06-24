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

    pub fn with_raster(mut self) -> Self {
        // Borrow-checker: build against a raster-less copy of self.
        let probe = Polyhedron3 { planes: self.planes.clone(), bbox: self.bbox, raster: None };
        self.raster = Some(VoxelRaster::for_shape(&probe));
        self
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

pub struct Node3<T> {
    pub bbox: Aabb,
    pub parent: Option<Node3Id>,
    pub children: Option<[Node3Id; 2]>,
    pub items: Vec<T>,
}

pub struct Tree3<T: Positioned3> {
    nodes: Vec<Node3<T>>,
    /// Slots freed by merge-ups, reused before the arena grows — see
    /// [`crate::Tree`]'s free-list for the rationale.
    free: Vec<Node3Id>,
    pub item_limit: usize,
    pub merge_limit: usize,
    min_cell: f64,
    pub root: Node3Id,
}

impl<T: Positioned3> Tree3<T> {
    pub fn new(bbox: Aabb, item_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let min_cell = bbox.w.max(bbox.h).max(bbox.d) * 1e-12;
        Self {
            nodes: vec![Node3 { bbox, parent: None, children: None, items: Vec::new() }],
            free: Vec::new(),
            item_limit,
            merge_limit: item_limit,
            min_cell,
            root: Node3Id(0),
        }
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
    /// Arena capacity (high-water-mark). [`Tree3::live_node_count`] is the
    /// reachable count.
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn live_node_count(&self) -> usize { self.nodes.len() - self.free.len() }

    pub fn insert(&mut self, item: T) -> bool {
        let p = item.position();
        if !self.get(self.root).bbox.contains(p) { return false; }
        let leaf = self.locate(p);
        self.get_mut(leaf).items.push(item);
        if self.get(leaf).items.len() > self.item_limit { self.divide(leaf); }
        true
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

    fn divide(&mut self, id: Node3Id) {
        let (bbox, items) = {
            let n = self.get_mut(id);
            (n.bbox, std::mem::take(&mut n.items))
        };
        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.w.max(bbox.h).max(bbox.d) <= self.min_cell {
            self.get_mut(id).items = items;
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
        let a = self.alloc(Node3 { bbox: a_box, parent: Some(id), children: None, items: Vec::new() });
        let b = self.alloc(Node3 { bbox: b_box, parent: Some(id), children: None, items: Vec::new() });
        for item in items {
            let p = item.position();
            if self.get(a).bbox.contains(p) { self.get_mut(a).items.push(item); }
            else { self.get_mut(b).items.push(item); }
        }
        self.get_mut(id).children = Some([a, b]);
        if self.get(a).items.len() > self.item_limit { self.divide(a); }
        if self.get(b).items.len() > self.item_limit { self.divide(b); }
    }

    /// Relocate via ascend-to-LCA (the 2D winner): mutate in place, and if
    /// the item leaves its leaf, ascend to the lowest ancestor containing
    /// the new position and descend from there. Returns `false` if not
    /// found or pushed out of the root.
    pub fn update<F, M>(&mut self, old: Point3, predicate: F, mutator: M) -> bool
    where F: Fn(&T) -> bool, M: FnOnce(&mut T) {
        if !self.get(self.root).bbox.contains(old) { return false; }
        let leaf = self.locate(old);
        let idx = match self.get(leaf).items.iter().position(|it| predicate(it)) {
            Some(i) => i, None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);
        let np = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(np) { return true; }
        // ascend to LCA
        let mut anc = self.get(leaf).parent;
        let lca = loop {
            match anc {
                Some(a) if self.get(a).bbox.contains(np) => break a,
                Some(a) => anc = self.get(a).parent,
                None => { // out of bounds: drop + merge
                    let _ = self.get_mut(leaf).items.remove(idx);
                    self.try_merge_up(leaf);
                    return false;
                }
            }
        };
        let item = self.get_mut(leaf).items.remove(idx);
        let dest = self.locate_from(lca, np);
        self.get_mut(dest).items.push(item);
        if self.get(dest).items.len() > self.item_limit { self.divide(dest); }
        self.try_merge_up(leaf);
        true
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
        let removed = {
            let items = &mut self.get_mut(leaf).items;
            let idx = items.iter().position(|it| predicate(it))?;
            items.remove(idx)
        };
        self.try_merge_up(leaf);
        Some(removed)
    }

    fn try_merge_up(&mut self, mut node: Node3Id) {
        loop {
            let parent = match self.get(node).parent { Some(p) => p, None => return };
            let [a, b] = self.get(parent).children.expect("parent has children");
            if self.get(a).children.is_some() || self.get(b).children.is_some() { return; }
            let combined = self.get(a).items.len() + self.get(b).items.len();
            if combined > self.merge_limit { return; }
            let mut ia = std::mem::take(&mut self.get_mut(a).items);
            let mut ib = std::mem::take(&mut self.get_mut(b).items);
            ia.append(&mut ib);
            let pnode = self.get_mut(parent);
            pnode.items = ia;
            pnode.children = None;
            self.free.push(a);
            self.free.push(b);
            node = parent;
        }
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

/// Squared distance from `q` to the nearest point of box `b` (0 if inside).
#[inline]
pub(crate) fn aabb_min_dist2(b: &Aabb, q: Point3) -> f64 {
    let dx = if q.x < b.x { b.x - q.x } else if q.x > b.x_max() { q.x - b.x_max() } else { 0.0 };
    let dy = if q.y < b.y { b.y - q.y } else if q.y > b.y_max() { q.y - b.y_max() } else { 0.0 };
    let dz = if q.z < b.z { b.z - q.z } else if q.z > b.z_max() { q.z - b.z_max() } else { 0.0 };
    dx * dx + dy * dy + dz * dz
}

// ------------------------------------------------------------- serialization
// Little-endian primitive read/write helpers (no external dependency).

fn w_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn r_u32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn r_u64<R: Read>(r: &mut R) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn r_f64<R: Read>(r: &mut R) -> io::Result<f64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b)) }
fn r_u8<R: Read>(r: &mut R) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }

fn w_aabb<W: Write>(w: &mut W, b: &Aabb) -> io::Result<()> {
    w_f64(w, b.x)?; w_f64(w, b.y)?; w_f64(w, b.z)?;
    w_f64(w, b.w)?; w_f64(w, b.h)?; w_f64(w, b.d)
}
fn r_aabb<R: Read>(r: &mut R) -> io::Result<Aabb> {
    Ok(Aabb::new(r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?))
}

const TREE3_MAGIC: &[u8; 4] = b"VHT3";
const TREE3_VERSION: u8 = 1;

fn corrupt(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, msg) }

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
            nodes.push(Node3 { bbox, parent, children, items });
        }
        if root.0 as usize >= nnodes { return Err(corrupt("root index out of range")); }
        Ok(Tree3 { nodes, free, item_limit, merge_limit, min_cell, root })
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
}
