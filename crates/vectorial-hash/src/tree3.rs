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
}
