//! `KdTree3` — a static **median-split** k-d tree over 3D points, with tight
//! per-node AABBs (a "k-d BVH"). It exists to answer one measured question the
//! kit's other trees leave open: the binary [`Tree3`](crate::Tree3) and the octrees
//! split at the **spatial midpoint**, so a *clustered* point set makes them keep
//! halving empty space — deep, thin subtrees. A k-d tree splits at the **median**
//! (equal point counts each side), so its depth is `⌈log₂(n/leaf)⌉` regardless of
//! how the points clump — balanced, shallow, fewer nodes to visit on a query. The
//! cost is a pricier build (an O(n) median selection per level).
//!
//! Build: recursively take the current slice's tight AABB, split its **longest**
//! axis at the point-count median (`select_nth_unstable_by`), recurse until a leaf
//! holds ≤ `capacity` points; each node's box is the union of its children's (so
//! `cull` prunes with tight boxes through the shared [`Shape3`] classify path).
//! `from_items` / `cull` / `knn`, brute-force-gated.

use crate::template::CellState;
use crate::tree3::{aabb_min_dist2, knn_offer, knn_worst, Aabb, KnnEntry, Point3, Positioned3, Shape3};
use std::collections::BinaryHeap;

enum Split { Leaf { start: u32, len: u32 }, Internal { left: u32, right: u32 } }
struct KdNode { bbox: Aabb, split: Split }

/// A static median-split k-d tree over `Positioned3` items. See the module docs.
pub struct KdTree3<T: Positioned3> {
    nodes: Vec<KdNode>,
    items: Vec<T>,
    capacity: usize,
    root: u32,
}

/// Tight AABB of `items` (assumes non-empty).
fn tight_box<T: Positioned3>(items: &[T]) -> Aabb {
    let p0 = items[0].position();
    let (mut lo, mut hi) = ([p0.x, p0.y, p0.z], [p0.x, p0.y, p0.z]);
    for it in &items[1..] {
        let p = it.position();
        lo[0] = lo[0].min(p.x); lo[1] = lo[1].min(p.y); lo[2] = lo[2].min(p.z);
        hi[0] = hi[0].max(p.x); hi[1] = hi[1].max(p.y); hi[2] = hi[2].max(p.z);
    }
    // pad to a non-degenerate box so classify's half-open tests behave
    Aabb::new(lo[0], lo[1], lo[2], (hi[0] - lo[0]).max(1e-9), (hi[1] - lo[1]).max(1e-9), (hi[2] - lo[2]).max(1e-9))
}

fn axis_of<T: Positioned3>(it: &T, a: usize) -> f64 { let p = it.position(); [p.x, p.y, p.z][a] }

impl<T: Positioned3> KdTree3<T> {
    /// Build from a point set: one top-down median partition. `capacity` is the max
    /// points a leaf holds (floored at 1).
    pub fn from_items(capacity: usize, mut items: Vec<T>) -> Self {
        let mut t = KdTree3 { nodes: Vec::new(), items: Vec::new(), capacity: capacity.max(1), root: 0 };
        if !items.is_empty() {
            let n = items.len();
            t.build(&mut items, 0, n);
        }
        t.items = items;
        t
    }

    #[inline] pub fn item_count(&self) -> usize { self.items.len() }
    #[inline] pub fn node_count(&self) -> usize { self.nodes.len() }
    /// Deepest leaf level (0 = a single root leaf) — a k-d tree stays ~balanced.
    pub fn depth(&self) -> u32 { if self.nodes.is_empty() { 0 } else { self.depth_of(self.root) } }
    fn depth_of(&self, id: u32) -> u32 {
        match self.nodes[id as usize].split {
            Split::Leaf { .. } => 0,
            Split::Internal { left, right } => 1 + self.depth_of(left).max(self.depth_of(right)),
        }
    }

    /// Recursively build over `items[lo..hi]`; returns the node id. Partitions the
    /// slice in place so each leaf owns a contiguous range.
    fn build(&mut self, items: &mut [T], lo: usize, hi: usize) -> u32 {
        let bbox = tight_box(&items[lo..hi]);
        let n = hi - lo;
        if n <= self.capacity {
            let id = self.nodes.len() as u32;
            self.nodes.push(KdNode { bbox, split: Split::Leaf { start: lo as u32, len: n as u32 } });
            return id;
        }
        // longest axis of the tight box → median partition on it.
        let a = if bbox.w >= bbox.h && bbox.w >= bbox.d { 0 } else if bbox.h >= bbox.d { 1 } else { 2 };
        let mid = lo + n / 2;
        items[lo..hi].select_nth_unstable_by(n / 2, |x, y| axis_of(x, a).total_cmp(&axis_of(y, a)));
        let id = self.nodes.len() as u32;
        self.nodes.push(KdNode { bbox, split: Split::Leaf { start: 0, len: 0 } }); // placeholder, fixed below
        let left = self.build(items, lo, mid);
        let right = self.build(items, mid, hi);
        self.nodes[id as usize].split = Split::Internal { left, right };
        id
    }

    /// Cull to a query volume — green/white/yellow descent over the tight boxes.
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        if !self.nodes.is_empty() { self.cull_node(self.root, shape, false, &mut out); }
        out
    }
    fn cull_node<'a, S: Shape3>(&'a self, id: u32, shape: &S, fully_inside: bool, out: &mut Vec<&'a T>) {
        match self.nodes[id as usize].split {
            Split::Leaf { start, len } => {
                let items = &self.items[start as usize..(start + len) as usize];
                if fully_inside { out.extend(items.iter()); }
                else {
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
            }
            Split::Internal { left, right } => {
                for child in [left, right] {
                    if fully_inside { self.cull_node(child, shape, true, out); continue; }
                    match shape.classify_aabb(&self.nodes[child as usize].bbox) {
                        CellState::Out => {}
                        CellState::In => self.cull_node(child, shape, true, out),
                        CellState::Maybe => self.cull_node(child, shape, false, out),
                    }
                }
            }
        }
    }

    /// k nearest neighbours — best-first descent, nearer child first, pruned by the
    /// current k-th distance.
    pub fn knn(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        if k == 0 || self.items.is_empty() { return Vec::new(); }
        let mut heap: BinaryHeap<KnnEntry<T>> = BinaryHeap::new();
        self.knn_node(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }
    fn knn_node<'a>(&'a self, id: u32, q: Point3, k: usize, heap: &mut BinaryHeap<KnnEntry<'a, T>>) {
        match self.nodes[id as usize].split {
            Split::Leaf { start, len } => {
                for it in &self.items[start as usize..(start + len) as usize] { knn_offer(heap, k, it, q); }
            }
            Split::Internal { left, right } => {
                let (dl, dr) = (aabb_min_dist2(&self.nodes[left as usize].bbox, q), aabb_min_dist2(&self.nodes[right as usize].bbox, q));
                let (near, far, dfar) = if dl <= dr { (left, right, dr) } else { (right, left, dl) };
                self.knn_node(near, q, k, heap);
                if dfar < knn_worst(heap, k) { self.knn_node(far, q, k, heap); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree3::Sphere3;

    #[derive(Clone, Copy)]
    struct P { id: u32, p: Point3 }
    impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
        fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }
    fn scatter(n: u32, seed: u64) -> Vec<P> {
        let mut r = Lcg(seed);
        (0..n).map(|id| {
            let p = if id % 2 == 0 { Point3::new(r.range(0.0, 100.0), r.range(0.0, 60.0), r.range(0.0, 100.0)) }
                    else { Point3::new(r.range(10.0, 14.0), r.range(10.0, 14.0), r.range(10.0, 14.0)) };
            P { id, p }
        }).collect()
    }

    #[test]
    fn kdtree3_cull_matches_brute_force() {
        let items = scatter(4000, 7);
        let t = KdTree3::from_items(16, items.clone());
        assert_eq!(t.item_count(), 4000);
        for (i, &(cx, cy, cz, r)) in [(50.0, 30.0, 50.0, 25.0), (12.0, 12.0, 12.0, 4.0), (0.0, 0.0, 0.0, 40.0), (90.0, 55.0, 90.0, 30.0)].iter().enumerate() {
            let s = Sphere3::new(cx, cy, cz, r);
            let mut got: Vec<u32> = t.cull(&s).iter().map(|p| p.id).collect();
            let mut want: Vec<u32> = items.iter().filter(|p| (p.p.x - cx).powi(2) + (p.p.y - cy).powi(2) + (p.p.z - cz).powi(2) <= r * r).map(|p| p.id).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "cull != brute for probe {i}");
        }
    }

    #[test]
    fn kdtree3_balanced_and_complete() {
        let items = scatter(4000, 11);
        let t = KdTree3::from_items(16, items);
        // median split → near-perfect balance: depth ≈ ceil(log2(n/leaf)) even though
        // half the points are in a tight cluster (a midpoint tree would go far deeper).
        let ideal = (4000f64 / 16.0).log2().ceil() as u32;
        assert!(t.depth() <= ideal + 1, "median-split depth {} exceeds ~ideal {}", t.depth(), ideal);
        let all = t.cull(&Sphere3::new(50.0, 30.0, 50.0, 1000.0));
        assert_eq!(all.len(), 4000, "a world-covering cull must return every item");
        let mut ids: Vec<u32> = all.iter().map(|p| p.id).collect();
        ids.sort(); ids.dedup();
        assert_eq!(ids.len(), 4000, "no item dropped or duplicated across leaves");
    }

    #[test]
    fn kdtree3_knn_matches_brute_force() {
        let items = scatter(3000, 5);
        let t = KdTree3::from_items(12, items.clone());
        for &(qx, qy, qz) in &[(50.0, 30.0, 50.0), (12.0, 12.0, 12.0), (5.0, 5.0, 95.0)] {
            let q = Point3::new(qx, qy, qz);
            let got: Vec<f64> = t.knn(q, 10).iter().map(|(d, _)| *d).collect();
            let mut want: Vec<f64> = items.iter().map(|p| ((p.p.x - qx).powi(2) + (p.p.y - qy).powi(2) + (p.p.z - qz).powi(2)).sqrt()).collect();
            want.sort_by(|a, b| a.total_cmp(b));
            want.truncate(10);
            assert_eq!(got.len(), 10);
            for (a, b) in got.iter().zip(want.iter()) { assert!((a - b).abs() < 1e-9, "knn dist {a} != brute {b}"); }
        }
    }
}
