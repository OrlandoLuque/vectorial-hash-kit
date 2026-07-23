//! `LinearOctree3` — a sparse, **adaptive** octree stored the *linear* way: no
//! child pointers, just a hash map of leaf buckets keyed by a self-describing
//! **location code** (Morton path + level in one `u64`). It sits between the two
//! existing 3D structures:
//!
//! - [`MortonGrid3`](crate::MortonGrid3) is a *single-level* uniform Z-order grid —
//!   one cell size everywhere. Cheapest to build and query when the density is
//!   roughly uniform, but a fixed cell can't be both fine in a cluster and coarse
//!   in the void.
//! - [`Octree3`](crate::Octree3) is a *pointer* octree — fully adaptive, but each
//!   node is a heap cell reached by chasing `ONodeId`s.
//!
//! `LinearOctree3` keeps the octree's **adaptivity** (a leaf subdivides into 8 only
//! where the points pile up, down to `max_depth`) while keeping Morton's **pointer
//! free** layout: a node's 8 children are its key shifted left 3 bits with the octant
//! OR'd in, so traversal is arithmetic, not indirection. Leaves live in a
//! `HashMap<u64, Vec<T>>`; a companion set marks the internal (subdivided) keys so an
//! empty subtree prunes in O(1) instead of being walked to `max_depth`.
//!
//! ## Location code
//! The root is `1`. The octant `o∈0..8` child of key `K` is `(K << 3) | o`. The
//! leading `1` is a sentinel three bits above the deepest octant, so the key encodes
//! its own level unambiguously (`level = (63 − key.leading_zeros()) / 3`) and two
//! cells at different depths never collide — the classic linear-octree trick, in one
//! `u64` (so ≤ 21 levels of the 10-bit-per-axis Morton range).
//!
//! Octant bits match [`morton3`](crate::morton3::morton3): x = bit 0, y = bit 1,
//! z = bit 2 (0 = low half, 1 = high half of the parent box on that axis).

use crate::template::CellState;
use crate::tree3::{aabb_min_dist2, knn_offer, knn_worst, Aabb, KnnEntry, Point3, Positioned3, Shape3};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Level of a location code (root = 0). `key` is always ≥ 1 (the sentinel).
#[inline]
fn level_of(key: u64) -> u32 { (63 - key.leading_zeros()) / 3 }

/// The `o`-th octant sub-box of `b` (halved on every axis; bits: x=1, y=2, z=4).
#[inline]
fn child_box(b: &Aabb, o: u8) -> Aabb {
    let (hw, hh, hd) = (b.w * 0.5, b.h * 0.5, b.d * 0.5);
    Aabb::new(
        b.x + if o & 1 != 0 { hw } else { 0.0 },
        b.y + if o & 2 != 0 { hh } else { 0.0 },
        b.z + if o & 4 != 0 { hd } else { 0.0 },
        hw, hh, hd,
    )
}

/// Which octant of `b` the point `p` falls in (each axis: high half → bit set).
#[inline]
fn octant_of(b: &Aabb, p: Point3) -> u8 {
    (if p.x >= b.x + b.w * 0.5 { 1 } else { 0 })
        | (if p.y >= b.y + b.h * 0.5 { 2 } else { 0 })
        | (if p.z >= b.z + b.d * 0.5 { 4 } else { 0 })
}

/// A sparse adaptive linear octree over `Positioned3` items. See the module docs.
pub struct LinearOctree3<T: Positioned3> {
    world: Aabb,
    capacity: usize,
    max_depth: u8,
    len: usize,
    leaves: HashMap<u64, Vec<T>>,
    internal: HashSet<u64>,
}

impl<T: Positioned3> LinearOctree3<T> {
    /// Empty tree over `world`; a leaf holds up to `capacity` items before it
    /// subdivides, stopping at `max_depth` (so coincident points can't recurse
    /// forever). `capacity` is floored at 1, `max_depth` capped at 21 (u64 room).
    pub fn new(world: Aabb, capacity: usize, max_depth: u8) -> Self {
        Self { world, capacity: capacity.max(1), max_depth: max_depth.min(21), len: 0,
               leaves: HashMap::new(), internal: HashSet::new() }
    }

    /// Bulk build: one top-down subdivision of all items. Cheaper than repeated
    /// [`insert`](Self::insert) (each item is bucketed once per level it descends).
    pub fn from_items(world: Aabb, capacity: usize, max_depth: u8, items: Vec<T>) -> Self {
        let mut t = Self::new(world, capacity, max_depth);
        t.len = items.len();
        if !items.is_empty() { t.subdivide(1, world, items); }
        t
    }

    #[inline] pub fn item_count(&self) -> usize { self.len }
    #[inline] pub fn leaf_count(&self) -> usize { self.leaves.len() }
    #[inline] pub fn world(&self) -> Aabb { self.world }
    /// The deepest occupied level (0 = a single root leaf). A proxy for how far
    /// the densest cluster forced the tree to refine.
    pub fn depth(&self) -> u32 { self.leaves.keys().map(|&k| level_of(k)).max().unwrap_or(0) }

    /// Visit every leaf as `(box, item_count)` — for debug / rendering overlays.
    pub fn visit_leaves<F: FnMut(&Aabb, usize)>(&self, mut f: F) {
        for (&key, items) in &self.leaves { f(&box_of(self.world, key), items.len()); }
    }

    /// Drop everything, keep the world/params.
    pub fn clear(&mut self) { self.leaves.clear(); self.internal.clear(); self.len = 0; }

    /// Insert one item, subdividing its target leaf if it now overflows.
    pub fn insert(&mut self, item: T) {
        self.len += 1;
        let (key, bx) = self.leaf_for(item.position());
        let bucket = self.leaves.entry(key).or_default();
        bucket.push(item);
        if bucket.len() > self.capacity && level_of(key) < self.max_depth as u32 {
            let items = self.leaves.remove(&key).expect("just inserted");
            self.subdivide(key, bx, items);
        }
    }

    /// Cull to a query volume — the same green/white/yellow descent as the other
    /// structures: a fully-contained subtree accepts wholesale, a disjoint one
    /// prunes, a straddling leaf does per-point tests.
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        if self.len > 0 { self.cull_node(1, self.world, shape, false, &mut out); }
        out
    }

    /// k nearest neighbours by Euclidean distance — best-first octant descent,
    /// pruning by the current k-th distance (same helper heap as the other trees).
    pub fn knn(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        if k == 0 || self.len == 0 { return Vec::new(); }
        let mut heap: BinaryHeap<KnnEntry<T>> = BinaryHeap::new();
        self.knn_node(1, self.world, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    // ---- internals ----------------------------------------------------------

    /// Descend from the root through internal nodes to the leaf key (present or
    /// not) whose box contains `p`.
    fn leaf_for(&self, p: Point3) -> (u64, Aabb) {
        let (mut key, mut bx) = (1u64, self.world);
        while self.internal.contains(&key) {
            let o = octant_of(&bx, p);
            bx = child_box(&bx, o);
            key = (key << 3) | o as u64;
        }
        (key, bx)
    }

    /// Place `items` under `key` (box `bx`): a leaf if it fits or we've hit
    /// `max_depth`, else mark internal and recurse into the 8 octants.
    fn subdivide(&mut self, key: u64, bx: Aabb, items: Vec<T>) {
        if items.len() <= self.capacity || level_of(key) >= self.max_depth as u32 {
            if !items.is_empty() { self.leaves.insert(key, items); }
            return;
        }
        self.internal.insert(key);
        let mut buckets: [Vec<T>; 8] = std::array::from_fn(|_| Vec::new());
        for it in items { buckets[octant_of(&bx, it.position()) as usize].push(it); }
        for (o, bucket) in buckets.into_iter().enumerate() {
            if !bucket.is_empty() { self.subdivide((key << 3) | o as u64, child_box(&bx, o as u8), bucket); }
        }
    }

    fn cull_node<'a, S: Shape3>(&'a self, key: u64, bx: Aabb, shape: &S, fully_inside: bool, out: &mut Vec<&'a T>) {
        if let Some(items) = self.leaves.get(&key) {
            if fully_inside {
                out.extend(items.iter());
            } else {
                let raster = shape.voxel_raster();
                for it in items {
                    let p = it.position();
                    match raster.map(|g| g.cell_at_world(p)) {
                        Some(CellState::In) => out.push(it),
                        Some(CellState::Out) => {}
                        _ => if shape.contains_point(p) { out.push(it); },
                    }
                }
            }
            return;
        }
        if !self.internal.contains(&key) { return; } // empty region
        for o in 0..8u8 {
            let cb = child_box(&bx, o);
            let ck = (key << 3) | o as u64;
            if fully_inside {
                self.cull_node(ck, cb, shape, true, out);
                continue;
            }
            match shape.classify_aabb(&cb) {
                CellState::Out => {}
                CellState::In => self.cull_node(ck, cb, shape, true, out),
                CellState::Maybe => self.cull_node(ck, cb, shape, false, out),
            }
        }
    }

    fn knn_node<'a>(&'a self, key: u64, bx: Aabb, q: Point3, k: usize, heap: &mut BinaryHeap<KnnEntry<'a, T>>) {
        if let Some(items) = self.leaves.get(&key) {
            for it in items { knn_offer(heap, k, it, q); }
            return;
        }
        if !self.internal.contains(&key) { return; }
        // Order the 8 octants by nearest-point distance, descend nearest-first,
        // prune a child once its box is farther than the current k-th neighbour.
        let mut order: [(f64, u8); 8] = std::array::from_fn(|o| (aabb_min_dist2(&child_box(&bx, o as u8), q), o as u8));
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d, o) in order {
            if d < knn_worst(heap, k) { self.knn_node((key << 3) | o as u64, child_box(&bx, o), q, k, heap); }
        }
    }
}

/// The world-space box of a location code, decoded by replaying its octant bits
/// from the root down (used by [`LinearOctree3::visit_leaves`]).
fn box_of(world: Aabb, key: u64) -> Aabb {
    let level = level_of(key);
    let mut bx = world;
    for d in (0..level).rev() {
        let o = ((key >> (3 * d)) & 7) as u8; // octant applied at this depth
        bx = child_box(&bx, o);
    }
    bx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree3::Sphere3;

    #[derive(Clone, Copy)]
    struct P { id: u32, p: Point3 }
    impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

    // Cheap deterministic LCG so the tests need no rand dep.
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
        fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }

    fn scatter(n: u32, seed: u64) -> (Aabb, Vec<P>) {
        let world = Aabb::new(0.0, 0.0, 0.0, 100.0, 60.0, 100.0);
        let mut r = Lcg(seed);
        // Half uniform, half in a tight cluster → forces adaptive depth.
        let items = (0..n).map(|id| {
            let p = if id % 2 == 0 {
                Point3::new(r.range(0.0, 100.0), r.range(0.0, 60.0), r.range(0.0, 100.0))
            } else {
                Point3::new(r.range(10.0, 14.0), r.range(10.0, 14.0), r.range(10.0, 14.0))
            };
            P { id, p }
        }).collect();
        (world, items)
    }

    #[test]
    fn linear_octree3_cull_matches_brute_force() {
        let (world, items) = scatter(4000, 7);
        let t = LinearOctree3::from_items(world, 16, 12, items.clone());
        assert_eq!(t.item_count(), 4000);
        for (i, &(cx, cy, cz, r)) in [(50.0, 30.0, 50.0, 25.0), (12.0, 12.0, 12.0, 4.0), (0.0, 0.0, 0.0, 40.0), (90.0, 55.0, 90.0, 30.0)].iter().enumerate() {
            let s = Sphere3::new(cx, cy, cz, r);
            let mut got: Vec<u32> = t.cull(&s).iter().map(|p| p.id).collect();
            let mut want: Vec<u32> = items.iter().filter(|p| { let d = p.p; (d.x - cx).powi(2) + (d.y - cy).powi(2) + (d.z - cz).powi(2) <= r * r }).map(|p| p.id).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "cull != brute for probe {i}");
        }
    }

    #[test]
    fn linear_octree3_is_adaptive_and_complete() {
        let (world, items) = scatter(4000, 11);
        let t = LinearOctree3::from_items(world, 16, 12, items.clone());
        // The dense cluster forces real depth; the void stays coarse → the tree
        // is genuinely multi-level, not a uniform grid.
        assert!(t.depth() >= 4, "the cluster should force depth, got {}", t.depth());
        // Every item is recovered by a world-covering cull, and once each.
        let all = t.cull(&Sphere3::new(50.0, 30.0, 50.0, 1000.0));
        assert_eq!(all.len(), 4000, "a world-covering cull must return every item");
        let mut ids: Vec<u32> = all.iter().map(|p| p.id).collect();
        ids.sort(); ids.dedup();
        assert_eq!(ids.len(), 4000, "no item may be dropped or duplicated across leaves");
    }

    #[test]
    fn linear_octree3_knn_matches_brute_force() {
        let (world, items) = scatter(3000, 5);
        let t = LinearOctree3::from_items(world, 12, 12, items.clone());
        for &(qx, qy, qz) in &[(50.0, 30.0, 50.0), (12.0, 12.0, 12.0), (5.0, 5.0, 95.0)] {
            let q = Point3::new(qx, qy, qz);
            let got: Vec<f64> = t.knn(q, 10).iter().map(|(d, _)| *d).collect();
            let mut want: Vec<f64> = items.iter().map(|p| ((p.p.x - qx).powi(2) + (p.p.y - qy).powi(2) + (p.p.z - qz).powi(2)).sqrt()).collect();
            want.sort_by(|a, b| a.total_cmp(b));
            want.truncate(10);
            // Compare DISTANCES (ties make identities ambiguous), like the other trees.
            assert_eq!(got.len(), 10);
            for (a, b) in got.iter().zip(want.iter()) { assert!((a - b).abs() < 1e-9, "knn dist {a} != brute {b}"); }
        }
    }

    #[test]
    fn linear_octree3_insert_matches_bulk() {
        let (world, items) = scatter(2000, 3);
        let bulk = LinearOctree3::from_items(world, 8, 12, items.clone());
        let mut inc = LinearOctree3::new(world, 8, 12);
        for it in &items { inc.insert(*it); }
        assert_eq!(inc.item_count(), bulk.item_count());
        // Incremental and bulk builds must answer culls identically.
        for &(cx, cy, cz, r) in &[(50.0, 30.0, 50.0, 20.0), (12.0, 12.0, 12.0, 5.0), (80.0, 40.0, 20.0, 35.0)] {
            let s = Sphere3::new(cx, cy, cz, r);
            let mut a: Vec<u32> = inc.cull(&s).iter().map(|p| p.id).collect();
            let mut b: Vec<u32> = bulk.cull(&s).iter().map(|p| p.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "incremental vs bulk cull diverged");
        }
    }
}
