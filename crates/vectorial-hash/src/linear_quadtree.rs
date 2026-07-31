//! `LinearQuadTree` — the 2D companion to [`LinearOctree3`](crate::LinearOctree3):
//! a sparse, **adaptive** quadtree stored the *linear* way (no child pointers, a
//! hash map of leaf buckets keyed by a self-describing location code). It fills the
//! same gap in 2D that the linear octree fills in 3D:
//!
//! - [`MortonGrid`](crate::MortonGrid) is a *single-level* uniform Z-order grid.
//! - [`QuadTree`](crate::QuadTree) is a *pointer* quadtree — adaptive, but a node is
//!   a heap cell reached by chasing `QNodeId`s.
//! - `LinearQuadTree` keeps the quadtree's **adaptivity** (a leaf subdivides into 4
//!   only where points pile up, to `max_depth`) with a **pointer-free** layout: a
//!   node's 4 children are its key shifted left 2 bits with the quadrant OR'd in.
//!
//! ## Location code
//! Root = `1`; the quadrant `q∈0..4` child of key `K` is `(K << 2) | q`. The leading
//! `1` sentinel sits two bits above the deepest quadrant, so the key encodes its own
//! level (`level = (63 − key.leading_zeros()) / 2`, ≤ 31) and two cells at different
//! depths never collide — one `u64` per node.
//!
//! Quadrant bits match [`morton2`](crate::morton::morton2): x = bit 0, y = bit 1
//! (0 = low half, 1 = high half of the parent rect on that axis).
//!
//! Cull uses the shape's analytic [`classify_box`](crate::Shape::classify_box) when
//! it has one (circle, capsule) — pruning subtrees tightly — and falls back to a
//! bounding-box overlap broadphase + per-point test otherwise. (The precomputed cull
//! *template* path the pointer `QuadTree` offers is not wired here; the analytic /
//! bbox path is exact regardless.)

use crate::morton3::Crossed;
use crate::culling::{Capsule, Shape};
use crate::geom::{Point, Rect};
use crate::serde_io::{corrupt, r_f64, r_u32, r_u64, r_u8, w_f64, w_u32, w_u64, w_u8};
use crate::template::CellState;
use crate::tree::{knn_offer2, rect_min_dist2, Positioned};
use crate::tree3::{knn_worst, KnnEntry};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{self, Read, Write};

const LQT2_MAGIC: &[u8; 4] = b"LQT2";
const LQT2_VERSION: u8 = 1;

/// Level of a location code (root = 0). `key` is always ≥ 1 (the sentinel).
#[inline]
fn level_of(key: u64) -> u32 { (63 - key.leading_zeros()) / 2 }

/// The `q`-th quadrant sub-rect of `r` (halved on both axes; bits: x=1, y=2).
#[inline]
fn child_rect(r: &Rect, q: u8) -> Rect {
    let (hw, hh) = (r.width * 0.5, r.height * 0.5);
    Rect::new(r.x + if q & 1 != 0 { hw } else { 0.0 }, r.y + if q & 2 != 0 { hh } else { 0.0 }, hw, hh)
}

/// Which quadrant of `r` the point `p` falls in (each axis: high half → bit set).
#[inline]
fn quadrant_of(r: &Rect, p: Point) -> u8 {
    (if p.x >= r.x + r.width * 0.5 { 1 } else { 0 }) | (if p.y >= r.y + r.height * 0.5 { 2 } else { 0 })
}

/// Do two rectangles overlap (half-open)? The bbox broadphase for shapes with no
/// analytic `classify_box`.
#[inline]
fn rects_overlap(a: &Rect, b: &Rect) -> bool {
    a.x < b.x_max() && b.x < a.x_max() && a.y < b.y_max() && b.y < a.y_max()
}

/// A sparse adaptive linear quadtree over `Positioned` items. See the module docs.
pub struct LinearQuadTree<T: Positioned> {
    world: Rect,
    capacity: usize,
    max_depth: u8,
    len: usize,
    leaves: HashMap<u64, Vec<T>>,
    internal: HashSet<u64>,
}

impl<T: Positioned> LinearQuadTree<T> {
    /// Empty tree over `world`; a leaf holds up to `capacity` items before it
    /// subdivides, stopping at `max_depth` (so coincident points can't recurse
    /// forever). `capacity` floored at 1, `max_depth` capped at 31 (u64 room).
    pub fn new(world: Rect, capacity: usize, max_depth: u8) -> Self {
        Self { world, capacity: capacity.max(1), max_depth: max_depth.min(31), len: 0,
               leaves: HashMap::new(), internal: HashSet::new() }
    }

    /// Bulk build: one top-down subdivision of all items.
    pub fn from_items(world: Rect, capacity: usize, max_depth: u8, items: Vec<T>) -> Self {
        let mut t = Self::new(world, capacity, max_depth);
        t.len = items.len();
        if !items.is_empty() { t.subdivide(1, world, items); }
        t
    }

    #[inline] pub fn item_count(&self) -> usize { self.len }
    #[inline] pub fn leaf_count(&self) -> usize { self.leaves.len() }
    #[inline] pub fn world(&self) -> Rect { self.world }
    /// The deepest occupied level (0 = a single root leaf) — how far the densest
    /// cluster forced the tree to refine.
    pub fn depth(&self) -> u32 { self.leaves.keys().map(|&k| level_of(k)).max().unwrap_or(0) }

    /// Visit every leaf as `(rect, item_count)` — for debug / rendering overlays.
    pub fn visit_leaves<F: FnMut(&Rect, usize)>(&self, mut f: F) {
        for (&key, items) in &self.leaves { f(&rect_of(self.world, key), items.len()); }
    }

    /// Drop everything, keep the world/params.
    pub fn clear(&mut self) { self.leaves.clear(); self.internal.clear(); self.len = 0; }

    /// Insert one item, subdividing its target leaf if it now overflows.
    pub fn insert(&mut self, item: T) {
        self.len += 1;
        let (key, rc) = self.leaf_for(item.position());
        let bucket = self.leaves.entry(key).or_default();
        bucket.push(item);
        if bucket.len() > self.capacity && level_of(key) < self.max_depth as u32 {
            let items = self.leaves.remove(&key).expect("just inserted");
            self.subdivide(key, rc, items);
        }
    }

    /// **Move an item that is already in the tree, in place.**
    ///
    /// Same bargain as [`crate::MortonGrid::update`]: the caller says where the item *was*,
    /// so only that one leaf is scanned. If the item has not left its leaf there is nothing to
    /// do at all; if it has, it is re-inserted through the normal path, which subdivides the
    /// destination if that tips it over `capacity`.
    ///
    /// **This structure drifts in SHAPE, and the grids do not.** A flat grid maintained in
    /// place is byte-identical to a rebuilt one, because its cells are fixed. This one is
    /// adaptive: it keeps splits made for a distribution the points have since left, and it
    /// never merges a leaf that has emptied out. What is guaranteed is that it keeps giving
    /// the *same answers* as a rebuild — the tests check cull sets and k-NN distances, not
    /// leaf counts. What is not guaranteed is that it keeps answering them as *fast*: on a
    /// workload that migrates across the world, rebuild periodically or accept the drift.
    pub fn update<P, M>(&mut self, old: Point, predicate: P, mutate: M) -> Crossed
    where
        P: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        if !self.world.contains(old) { return Crossed::Missing; }
        let (from, _) = self.leaf_for(old);
        let (idx, p) = {
            let Some(bucket) = self.leaves.get_mut(&from) else { return Crossed::Missing };
            let Some(idx) = bucket.iter().position(&predicate) else { return Crossed::Missing };
            mutate(&mut bucket[idx]);
            (idx, bucket[idx].position())
        };
        if !self.world.contains(p) {
            self.take_at(from, idx);
            return Crossed::Left;
        }
        let (to, _) = self.leaf_for(p);
        if to == from { return Crossed::Stayed; }
        let item = self.take_at(from, idx);
        self.insert(item);
        Crossed::Moved
    }

    /// **Remove an item, given where it was** — the companion to [`Self::update`].
    pub fn remove<P: Fn(&T) -> bool>(&mut self, old: Point, predicate: P) -> Option<T> {
        if !self.world.contains(old) { return None; }
        let (key, _) = self.leaf_for(old);
        let idx = self.leaves.get(&key)?.iter().position(&predicate)?;
        Some(self.take_at(key, idx))
    }

    /// Pull item `idx` out of leaf `key`, dropping the leaf if that empties it.
    fn take_at(&mut self, key: u64, idx: usize) -> T {
        let bucket = self.leaves.get_mut(&key).expect("caller located this leaf");
        let item = bucket.swap_remove(idx);
        if bucket.is_empty() { self.leaves.remove(&key); }
        self.len -= 1;
        item
    }

    /// Cull to a query volume: analytic `classify_box` (green/white/yellow) when the
    /// shape has one, else a bbox broadphase + per-point test.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        if self.len > 0 { self.cull_node(1, self.world, shape, &shape.bounding_box(), false, &mut out); }
        out
    }

    /// k nearest neighbours by Euclidean distance — best-first quadrant descent,
    /// pruning by the current k-th distance.
    pub fn knn(&self, q: Point, k: usize) -> Vec<(f64, &T)> {
        if k == 0 || self.len == 0 { return Vec::new(); }
        let mut heap: BinaryHeap<KnnEntry<T>> = BinaryHeap::new();
        self.knn_node(1, self.world, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    // ---- internals ----------------------------------------------------------

    fn leaf_for(&self, p: Point) -> (u64, Rect) {
        let (mut key, mut rc) = (1u64, self.world);
        while self.internal.contains(&key) {
            let q = quadrant_of(&rc, p);
            rc = child_rect(&rc, q);
            key = (key << 2) | q as u64;
        }
        (key, rc)
    }

    fn subdivide(&mut self, key: u64, rc: Rect, items: Vec<T>) {
        if items.len() <= self.capacity || level_of(key) >= self.max_depth as u32 {
            if !items.is_empty() { self.leaves.insert(key, items); }
            return;
        }
        self.internal.insert(key);
        let mut buckets: [Vec<T>; 4] = std::array::from_fn(|_| Vec::new());
        for it in items { buckets[quadrant_of(&rc, it.position()) as usize].push(it); }
        for (q, bucket) in buckets.into_iter().enumerate() {
            if !bucket.is_empty() { self.subdivide((key << 2) | q as u64, child_rect(&rc, q as u8), bucket); }
        }
    }

    fn cull_node<'a, S: Shape>(&'a self, key: u64, rc: Rect, shape: &S, bb: &Rect, fully_inside: bool, out: &mut Vec<&'a T>) {
        if let Some(items) = self.leaves.get(&key) {
            if fully_inside { out.extend(items.iter()); }
            else { for it in items { if shape.contains_point(it.position()) { out.push(it); } } }
            return;
        }
        if !self.internal.contains(&key) { return; } // empty region
        for q in 0..4u8 {
            let cr = child_rect(&rc, q);
            let ck = (key << 2) | q as u64;
            if fully_inside { self.cull_node(ck, cr, shape, bb, true, out); continue; }
            // analytic classify if the shape offers one, else the bbox broadphase.
            let state = shape.classify_box(&cr).unwrap_or(if rects_overlap(&cr, bb) { CellState::Maybe } else { CellState::Out });
            match state {
                CellState::Out => {}
                CellState::In => self.cull_node(ck, cr, shape, bb, true, out),
                CellState::Maybe => self.cull_node(ck, cr, shape, bb, false, out),
            }
        }
    }

    fn knn_node<'a>(&'a self, key: u64, rc: Rect, q: Point, k: usize, heap: &mut BinaryHeap<KnnEntry<'a, T>>) {
        if let Some(items) = self.leaves.get(&key) {
            for it in items { knn_offer2(heap, k, it, q); }
            return;
        }
        if !self.internal.contains(&key) { return; }
        let mut order: [(f64, u8); 4] = std::array::from_fn(|c| (rect_min_dist2(&child_rect(&rc, c as u8), q), c as u8));
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d, c) in order {
            if d < knn_worst(heap, k) { self.knn_node((key << 2) | c as u64, child_rect(&rc, c), q, k, heap); }
        }
    }
}

impl<T: Positioned> LinearQuadTree<T> {
    /// Batch cull — one result list per shape (serial).
    pub fn cull_many<'a, S: Shape>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }
    /// Parallel batch cull: rayon fans the independent reads over the query set (the
    /// tree is immutable for reads). Native only — rayon isn't in the wasm build.
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where T: Sync {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
    }
    /// Batch k-NN — one result list per query (serial).
    pub fn knn_many(&self, queries: &[Point], k: usize) -> Vec<Vec<(f64, &T)>> {
        queries.iter().map(|&q| self.knn(q, k)).collect()
    }
    /// Parallel batch k-NN (rayon over the query set).
    #[cfg(feature = "parallel")]
    pub fn knn_many_par(&self, queries: &[Point], k: usize) -> Vec<Vec<(f64, &T)>>
    where T: Sync {
        use rayon::prelude::*;
        queries.par_iter().map(|&q| self.knn(q, k)).collect()
    }

    /// "Thick ray-cast": every item within `radius` of the ray `origin + t·normalize(dir)`,
    /// `t ∈ [0, max_dist]`, as `(t, &item)` sorted by `t`. Built on the [`Capsule`] + `cull`.
    pub fn raycast(&self, origin: Point, dir: Point, max_dist: f64, radius: f64) -> Vec<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y).sqrt();
        if m == 0.0 { return Vec::new(); }
        let (ux, uy) = (dir.x / m, dir.y / m);
        let end = Point::new(origin.x + ux * max_dist, origin.y + uy * max_dist);
        let mut hits: Vec<(f64, &T)> = self.cull(&Capsule::new(origin, end, radius)).into_iter().map(|it| {
            let p = it.position();
            let t = ((p.x - origin.x) * ux + (p.y - origin.y) * uy).clamp(0.0, max_dist);
            (t, it)
        }).collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits
    }
    /// The nearest item along the ray (smallest `t`), if any.
    pub fn raycast_first(&self, origin: Point, dir: Point, max_dist: f64, radius: f64) -> Option<(f64, &T)> {
        self.raycast(origin, dir, max_dist, radius).into_iter().next()
    }
}

impl<T: Positioned> LinearQuadTree<T> {
    /// Serialize the built tree to any `Write` — dependency-free, items via a caller
    /// closure. Only the leaf buckets are stored; the internal-node set is rebuilt on
    /// load from each leaf key's ancestors (exact and smaller).
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(LQT2_MAGIC)?;
        w_u8(w, LQT2_VERSION)?;
        w_f64(w, self.world.x)?; w_f64(w, self.world.y)?; w_f64(w, self.world.width)?; w_f64(w, self.world.height)?;
        w_u32(w, self.capacity as u32)?;
        w_u8(w, self.max_depth)?;
        w_u32(w, self.leaves.len() as u32)?;
        for (&key, bucket) in &self.leaves {
            w_u64(w, key)?;
            w_u32(w, bucket.len() as u32)?;
            for it in bucket { write_item(w, it)?; }
        }
        Ok(())
    }

    /// Reload a serialized tree — no rebuild; the leaf map is restored and the
    /// internal set reconstructed from the leaf keys. Rejects corrupt input.
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != LQT2_MAGIC { return Err(corrupt("bad LinearQuadTree magic")); }
        if r_u8(r)? != LQT2_VERSION { return Err(corrupt("unsupported LinearQuadTree version")); }
        let world = Rect::new(r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?);
        let capacity = r_u32(r)? as usize;
        let max_depth = r_u8(r)?;
        if max_depth > 31 { return Err(corrupt("LinearQuadTree max_depth out of 0..=31")); }
        let mut t = Self::new(world, capacity, max_depth);
        let nleaves = r_u32(r)? as usize;
        t.leaves.reserve(nleaves);
        for _ in 0..nleaves {
            let key = r_u64(r)?;
            if key == 0 { return Err(corrupt("LinearQuadTree leaf key 0 (missing sentinel)")); }
            let n = r_u32(r)? as usize;
            let mut bucket = Vec::with_capacity(n);
            for _ in 0..n { bucket.push(read_item(r)?); }
            t.len += bucket.len();
            let mut key_up = key; // every proper ancestor of a leaf is an internal node
            while key_up > 1 { key_up >>= 2; t.internal.insert(key_up); }
            t.leaves.insert(key, bucket);
        }
        Ok(t)
    }
}

/// The world-space rect of a location code, decoded by replaying its quadrant bits
/// from the root down (used by [`LinearQuadTree::visit_leaves`]).
fn rect_of(world: Rect, key: u64) -> Rect {
    let level = level_of(key);
    let mut rc = world;
    for d in (0..level).rev() {
        let q = ((key >> (2 * d)) & 3) as u8;
        rc = child_rect(&rc, q);
    }
    rc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::culling::Circle;

    #[derive(Clone, Copy)]
    struct P { id: u32, p: Point }
    impl Positioned for P { fn position(&self) -> Point { self.p } }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
        fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }

    fn scatter(n: u32, seed: u64) -> (Rect, Vec<P>) {
        let world = Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut r = Lcg(seed);
        // Half uniform, half in a tight cluster → forces adaptive depth.
        let items = (0..n).map(|id| {
            let p = if id % 2 == 0 { Point::new(r.range(0.0, 100.0), r.range(0.0, 100.0)) }
                    else { Point::new(r.range(10.0, 14.0), r.range(10.0, 14.0)) };
            P { id, p }
        }).collect();
        (world, items)
    }

    #[test]
    fn linear_quadtree_cull_matches_brute_force() {
        let (world, items) = scatter(4000, 7);
        let t = LinearQuadTree::from_items(world, 16, 16, items.clone());
        assert_eq!(t.item_count(), 4000);
        for (i, &(cx, cy, rr)) in [(50.0, 50.0, 25.0), (12.0, 12.0, 4.0), (0.0, 0.0, 40.0), (90.0, 90.0, 30.0)].iter().enumerate() {
            let s = Circle::new(Point::new(cx, cy), rr);
            let mut got: Vec<u32> = t.cull(&s).iter().map(|p| p.id).collect();
            let mut want: Vec<u32> = items.iter().filter(|p| (p.p.x - cx).powi(2) + (p.p.y - cy).powi(2) <= rr * rr).map(|p| p.id).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "cull != brute for probe {i}");
        }
    }

    #[test]
    fn linear_quadtree_is_adaptive_and_complete() {
        let (world, items) = scatter(4000, 11);
        let t = LinearQuadTree::from_items(world, 16, 16, items.clone());
        assert!(t.depth() >= 4, "the cluster should force depth, got {}", t.depth());
        let all = t.cull(&Circle::new(Point::new(50.0, 50.0), 1000.0));
        assert_eq!(all.len(), 4000, "a world-covering cull must return every item");
        let mut ids: Vec<u32> = all.iter().map(|p| p.id).collect();
        ids.sort(); ids.dedup();
        assert_eq!(ids.len(), 4000, "no item may be dropped or duplicated across leaves");
    }

    #[test]
    fn linear_quadtree_knn_matches_brute_force() {
        let (world, items) = scatter(3000, 5);
        let t = LinearQuadTree::from_items(world, 12, 16, items.clone());
        for &(qx, qy) in &[(50.0, 50.0), (12.0, 12.0), (5.0, 95.0)] {
            let q = Point::new(qx, qy);
            let got: Vec<f64> = t.knn(q, 10).iter().map(|(d, _)| *d).collect();
            let mut want: Vec<f64> = items.iter().map(|p| ((p.p.x - qx).powi(2) + (p.p.y - qy).powi(2)).sqrt()).collect();
            want.sort_by(|a, b| a.total_cmp(b));
            want.truncate(10);
            assert_eq!(got.len(), 10);
            for (a, b) in got.iter().zip(want.iter()) { assert!((a - b).abs() < 1e-9, "knn dist {a} != brute {b}"); }
        }
    }

    #[test]
    fn linear_quadtree_insert_matches_bulk() {
        let (world, items) = scatter(2000, 3);
        let bulk = LinearQuadTree::from_items(world, 8, 16, items.clone());
        let mut inc = LinearQuadTree::new(world, 8, 16);
        for it in &items { inc.insert(*it); }
        assert_eq!(inc.item_count(), bulk.item_count());
        for &(cx, cy, rr) in &[(50.0, 50.0, 20.0), (12.0, 12.0, 5.0), (80.0, 20.0, 35.0)] {
            let s = Circle::new(Point::new(cx, cy), rr);
            let mut a: Vec<u32> = inc.cull(&s).iter().map(|p| p.id).collect();
            let mut b: Vec<u32> = bulk.cull(&s).iter().map(|p| p.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "incremental vs bulk cull diverged");
        }
    }

    #[test]
    fn linear_quadtree_serialize_roundtrip() {
        let (world, items) = scatter(3000, 9);
        let t = LinearQuadTree::from_items(world, 16, 16, items);
        let mut buf = Vec::new();
        t.serialize(&mut buf, |w: &mut Vec<u8>, p: &P| {
            w.write_all(&p.id.to_le_bytes())?;
            w.write_all(&p.p.x.to_le_bytes())?;
            w.write_all(&p.p.y.to_le_bytes())
        }).unwrap();
        let mut rd = &buf[..];
        let back = LinearQuadTree::<P>::deserialize(&mut rd, |r: &mut &[u8]| {
            let mut b4 = [0u8; 4]; r.read_exact(&mut b4)?; let id = u32::from_le_bytes(b4);
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b8)?; let x = f64::from_le_bytes(b8);
            r.read_exact(&mut b8)?; let y = f64::from_le_bytes(b8);
            Ok(P { id, p: Point::new(x, y) })
        }).unwrap();
        assert_eq!(back.item_count(), t.item_count());
        assert_eq!(back.leaf_count(), t.leaf_count());
        assert_eq!(back.depth(), t.depth());
        for &(cx, cy, rr) in &[(50.0, 50.0, 20.0), (12.0, 12.0, 5.0)] {
            let s = Circle::new(Point::new(cx, cy), rr);
            let mut a: Vec<u32> = t.cull(&s).iter().map(|p| p.id).collect();
            let mut b: Vec<u32> = back.cull(&s).iter().map(|p| p.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "cull diverged after round-trip");
        }
        assert!(LinearQuadTree::<P>::deserialize(&mut &b"XXXX"[..], |_r: &mut &[u8]| -> std::io::Result<P> { unreachable!() }).is_err());
    }

    #[test]
    fn linear_quadtree_cull_many_matches_singles() {
        let (world, items) = scatter(2000, 13);
        let t = LinearQuadTree::from_items(world, 16, 16, items);
        let shapes: Vec<Circle> = (0..20).map(|i| Circle::new(Point::new(10.0 + i as f64 * 4.0, 50.0), 12.0)).collect();
        for (s, m) in shapes.iter().zip(t.cull_many(&shapes).iter()) {
            let single: Vec<u32> = t.cull(s).iter().map(|p| p.id).collect();
            let batch: Vec<u32> = m.iter().map(|p| p.id).collect();
            assert_eq!(single, batch, "cull_many != individual cull");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn linear_quadtree_batch_par_matches_serial() {
        let (world, items) = scatter(3000, 17);
        let t = LinearQuadTree::from_items(world, 16, 16, items);
        let shapes: Vec<Circle> = (0..40).map(|i| Circle::new(Point::new((i % 10) as f64 * 10.0, (i / 10) as f64 * 25.0), 18.0)).collect();
        let sort = |vs: Vec<Vec<&P>>| -> Vec<Vec<u32>> { vs.iter().map(|v| { let mut ids: Vec<u32> = v.iter().map(|p| p.id).collect(); ids.sort(); ids }).collect() };
        assert_eq!(sort(t.cull_many(&shapes)), sort(t.cull_many_par(&shapes)), "cull_many_par != cull_many");
        let qs: Vec<Point> = (0..20).map(|i| Point::new(i as f64 * 5.0, 50.0)).collect();
        let kd = |vs: Vec<Vec<(f64, &P)>>| -> Vec<Vec<f64>> { vs.iter().map(|v| v.iter().map(|(d, _)| *d).collect()).collect() };
        assert_eq!(kd(t.knn_many(&qs, 8)), kd(t.knn_many_par(&qs, 8)), "knn_many_par != knn_many");
    }

    #[test]
    fn linear_quadtree_raycast_matches_brute() {
        let (world, items) = scatter(3000, 23);
        let t = LinearQuadTree::from_items(world, 16, 16, items.clone());
        let origin = Point::new(5.0, 5.0);
        let dir = Point::new(1.0, 0.6);
        let (max_dist, radius) = (120.0, 6.0);
        let hits = t.raycast(origin, dir, max_dist, radius);
        for w in hits.windows(2) { assert!(w[0].0 <= w[1].0 + 1e-9, "raycast hits not sorted by t"); }
        let m = (dir.x * dir.x + dir.y * dir.y).sqrt();
        let (ux, uy) = (dir.x / m, dir.y / m);
        let d2seg = |p: Point| -> f64 {
            let s = ((p.x - origin.x) * ux + (p.y - origin.y) * uy).clamp(0.0, max_dist);
            (p.x - origin.x - ux * s).powi(2) + (p.y - origin.y - uy * s).powi(2)
        };
        let mut got: Vec<u32> = hits.iter().map(|(_, p)| p.id).collect();
        let mut want: Vec<u32> = items.iter().filter(|p| d2seg(p.p) <= radius * radius).map(|p| p.id).collect();
        got.sort(); want.sort();
        assert_eq!(got, want, "raycast set != brute capsule");
        assert_eq!(t.raycast_first(origin, dir, max_dist, radius).map(|(_, p)| p.id), hits.first().map(|(_, p)| p.id));
    }
}

#[cfg(test)]
mod keep_tests {
    use super::*;
    use crate::Circle;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct M { id: u32, p: Point }
    impl Positioned for M { fn position(&self) -> Point { self.p } }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f64) / (1u64 << 24) as f64 }
        fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }

    /// A tree MAINTAINED through thousands of moves must keep giving the same ANSWERS as one
    /// rebuilt from the same positions.
    ///
    /// Note what is deliberately not asserted: the leaf count. This structure is adaptive, so
    /// a maintained copy keeps splits made for a distribution the points have since left, and
    /// never merges an emptied leaf — its shape legitimately drifts from a rebuild's. The
    /// grids can be held to byte-identity because their cells are fixed; this one can only be
    /// held to answering identically, and asserting shape would be a test that fails for a
    /// reason which is not a bug.
    #[test]
    fn maintained_answers_match_a_rebuild() {
        let world = Rect::new(0.0, 0.0, 256.0, 256.0);
        let mut rng = Rng(0x5EED_1234);
        let n = 600usize;
        let mut pts: Vec<M> = (0..n).map(|i| M { id: i as u32, p: Point::new(rng.r(0.0, 255.9), rng.r(0.0, 255.9)) }).collect();
        let mut kept = LinearQuadTree::from_items(world, 8, 12, pts.clone());

        let (mut stayed, mut moved) = (0u32, 0u32);
        for round in 0..30 {
            for (i, pt) in pts.iter_mut().enumerate() {
                let old = pt.p;
                let step = if (i + round) % 2 == 0 { 1.0 } else { 80.0 };
                let np = Point::new((old.x + rng.r(-step, step)).clamp(0.0, 255.9), (old.y + rng.r(-step, step)).clamp(0.0, 255.9));
                let id = pt.id;
                match kept.update(old, |it| it.id == id, |it| it.p = np) {
                    Crossed::Stayed => stayed += 1,
                    Crossed::Moved => moved += 1,
                    other => panic!("update failed on {id}: {other:?}"),
                }
                pt.p = np;
            }
            let fresh = LinearQuadTree::from_items(world, 8, 12, pts.clone());
            assert_eq!(kept.item_count(), fresh.item_count(), "count drifted at round {round}");
            for s in [Circle::new(Point::new(40.0, 40.0), 30.0), Circle::new(Point::new(128.0, 128.0), 55.0), Circle::new(Point::new(210.0, 60.0), 25.0)] {
                let mut a: Vec<u32> = kept.cull(&s).iter().map(|m| m.id).collect();
                let mut b: Vec<u32> = fresh.cull(&s).iter().map(|m| m.id).collect();
                a.sort_unstable(); b.sort_unstable();
                assert_eq!(a, b, "maintained != rebuilt at round {round}");
            }
            let q = Point::new(90.0, 110.0);
            let ka: Vec<u64> = kept.knn(q, 10).iter().map(|(d, _)| d.to_bits()).collect();
            let kb: Vec<u64> = fresh.knn(q, 10).iter().map(|(d, _)| d.to_bits()).collect();
            assert_eq!(ka, kb, "knn diverged at round {round}");
        }
        assert!(stayed > 500 && moved > 500, "both paths must be exercised: stayed {stayed}, moved {moved}");
    }

    #[test]
    fn remove_and_the_edges() {
        let world = Rect::new(0.0, 0.0, 256.0, 256.0);
        // Capacity 1, so the two points end up in DIFFERENT leaves. At capacity 8 they share
        // the root leaf, and then a "wrong" old position still locates the right bucket — the
        // first version of this test asserted Missing and got Stayed, correctly.
        let mut t = LinearQuadTree::new(world, 1, 12);
        t.insert(M { id: 1, p: Point::new(10.0, 10.0) });
        t.insert(M { id: 2, p: Point::new(200.0, 200.0) });
        assert_eq!(t.item_count(), 2);
        // a wrong `old` finds nothing rather than corrupting anything
        assert_eq!(t.update(Point::new(200.0, 200.0), |it| it.id == 1, |_| {}), Crossed::Missing);
        assert!(t.remove(Point::new(200.0, 200.0), |it| it.id == 1).is_none());
        // and a real removal is gone from the ANSWERS, not just the count
        let got = t.remove(Point::new(10.0, 10.0), |it| it.id == 1).expect("was there");
        assert_eq!(got.id, 1);
        assert_eq!(t.item_count(), 1);
        assert!(t.cull(&Circle::new(Point::new(10.0, 10.0), 2.0)).is_empty(), "the vacated leaf still answers");
    }
}
