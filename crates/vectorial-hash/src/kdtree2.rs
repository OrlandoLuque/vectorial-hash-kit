//! `KdTree2` — the 2D twin of [`KdTree3`](crate::KdTree3): a static **median-split**
//! k-d tree over 2D points with tight per-node rects (a "k-d BVH" in the plane).
//!
//! It completes the family. The kit's other 2D indexes all split at the spatial
//! **midpoint** ([`Tree`](crate::Tree) binary, [`QuadTree`](crate::QuadTree) 4-way,
//! [`LinearQuadTree`](crate::LinearQuadTree) adaptive-but-midpoint) or not at all
//! ([`MortonGrid`](crate::MortonGrid), one flat level). A midpoint split keeps halving
//! empty space when the points clump; a **median** split gives each side an equal point
//! *count*, so depth stays `⌈log₂(n/leaf)⌉` however the data clusters — shallower tree,
//! tighter boxes, fewer nodes visited per query. The cost is a pricier build (an O(n)
//! median selection per level), which is why this is a **build-once** structure: no
//! handle layer, no `remove`, no `update`. If your points move every frame you want
//! `Tree` + `ItemRef`; see `docs/CHOOSING.md`.
//!
//! Build: recursively take the slice's tight rect, split its **longer** axis at the
//! point-count median (`select_nth_unstable_by`), recurse until a leaf holds ≤
//! `capacity` points. `from_items` / `from_items_par` / `cull` / `knn` / `raycast`,
//! all brute-force-gated.

use crate::culling::{Capsule, Shape};
use crate::serde_io::{corrupt, r_rect, r_u32, r_u8, w_rect, w_u32, w_u8};
use std::io::{self, Read, Write};
use crate::geom::{Point, Rect};
use crate::template::CellState;
use crate::tree::{knn_offer2, rect_min_dist2, Positioned};
use crate::tree3::{knn_worst, KnnEntry};
use std::collections::BinaryHeap;

const KD2_MAGIC: &[u8; 4] = b"KDT2";
const KD2_VERSION: u8 = 1;

enum Split { Leaf { start: u32, len: u32 }, Internal { left: u32, right: u32 } }
struct KdNode { bbox: Rect, split: Split }

/// A static median-split k-d tree over `Positioned` items. See the module docs.
pub struct KdTree2<T: Positioned> {
    nodes: Vec<KdNode>,
    items: Vec<T>,
    capacity: usize,
    root: u32,
}

/// Tight rect of `items` (assumes non-empty).
fn tight_rect<T: Positioned>(items: &[T]) -> Rect {
    let p0 = items[0].position();
    let (mut lo, mut hi) = ([p0.x, p0.y], [p0.x, p0.y]);
    for it in &items[1..] {
        let p = it.position();
        lo[0] = lo[0].min(p.x); lo[1] = lo[1].min(p.y);
        hi[0] = hi[0].max(p.x); hi[1] = hi[1].max(p.y);
    }
    // pad to a non-degenerate rect so the half-open classify tests behave
    Rect::new(lo[0], lo[1], (hi[0] - lo[0]).max(1e-9), (hi[1] - lo[1]).max(1e-9))
}

fn axis_of<T: Positioned>(it: &T, a: usize) -> f64 { let p = it.position(); if a == 0 { p.x } else { p.y } }

/// Below this many points a split is built serially: `rayon::join` costs more than the
/// median selection it would overlap.
#[cfg(feature = "parallel")]
const PAR_CUTOFF: usize = 4096;

/// Build the subtree over `items` (whose first element is global index `base`) into a
/// fresh node vector whose **root is index 0** and whose child ids are local to it.
#[cfg(feature = "parallel")]
fn build_par<T: Positioned + Send>(items: &mut [T], base: usize, capacity: usize) -> Vec<KdNode> {
    let n = items.len();
    let bbox = tight_rect(items);
    if n <= capacity {
        return vec![KdNode { bbox, split: Split::Leaf { start: base as u32, len: n as u32 } }];
    }
    let a = if bbox.width >= bbox.height { 0 } else { 1 };
    let mid = n / 2;
    items.select_nth_unstable_by(mid, |x, y| axis_of(x, a).total_cmp(&axis_of(y, a)));
    let (lo, hi) = items.split_at_mut(mid);
    let (mut left, mut right) = if n >= PAR_CUTOFF {
        rayon::join(|| build_par(lo, base, capacity), || build_par(hi, base + mid, capacity))
    } else {
        (build_par(lo, base, capacity), build_par(hi, base + mid, capacity))
    };
    let (loff, roff) = (1u32, 1 + left.len() as u32);
    for nd in left.iter_mut() { if let Split::Internal { left: l, right: r } = &mut nd.split { *l += loff; *r += loff; } }
    for nd in right.iter_mut() { if let Split::Internal { left: l, right: r } = &mut nd.split { *l += roff; *r += roff; } }
    let mut out = Vec::with_capacity(1 + left.len() + right.len());
    out.push(KdNode { bbox, split: Split::Internal { left: loff, right: roff } });
    out.append(&mut left);
    out.append(&mut right);
    out
}

impl<T: Positioned> KdTree2<T> {
    /// Build from a point set: one top-down median partition. `capacity` is the max
    /// points a leaf holds (floored at 1).
    pub fn from_items(capacity: usize, mut items: Vec<T>) -> Self {
        let mut t = KdTree2 { nodes: Vec::new(), items: Vec::new(), capacity: capacity.max(1), root: 0 };
        if !items.is_empty() { let n = items.len(); t.build(&mut items, 0, n); }
        t.items = items;
        t
    }

    /// Same tree, built with rayon — **node-for-node identical** to `from_items`
    /// (tested), because the serial build already emits *parent, then left subtree, then
    /// right subtree*, so a subtree can be built separately and spliced in with an id
    /// shift. See [`KdTree3::from_items_par`](crate::KdTree3::from_items_par) and
    /// `docs/PARALLEL.md` for why the median split fans out better than a midpoint one.
    #[cfg(feature = "parallel")]
    pub fn from_items_par(capacity: usize, mut items: Vec<T>) -> Self
    where T: Send {
        let capacity = capacity.max(1);
        let nodes = if items.is_empty() { Vec::new() } else { build_par(&mut items, 0, capacity) };
        KdTree2 { nodes, items, capacity, root: 0 }
    }

    #[inline] pub fn item_count(&self) -> usize { self.items.len() }
    #[inline] pub fn node_count(&self) -> usize { self.nodes.len() }
    /// Visit every leaf as `(rect, item_count)` — for debug / rendering overlays. The
    /// rects are **tight**, so unlike the midpoint trees' cells they don't tile the
    /// world: the gaps are exactly the empty space a median split refuses to index.
    pub fn visit_leaves<F: FnMut(&Rect, usize)>(&self, mut f: F) {
        for nd in &self.nodes {
            if let Split::Leaf { len, .. } = nd.split { f(&nd.bbox, len as usize); }
        }
    }
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
        let bbox = tight_rect(&items[lo..hi]);
        let n = hi - lo;
        if n <= self.capacity {
            let id = self.nodes.len() as u32;
            self.nodes.push(KdNode { bbox, split: Split::Leaf { start: lo as u32, len: n as u32 } });
            return id;
        }
        let a = if bbox.width >= bbox.height { 0 } else { 1 };
        let mid = lo + n / 2;
        items[lo..hi].select_nth_unstable_by(n / 2, |x, y| axis_of(x, a).total_cmp(&axis_of(y, a)));
        let id = self.nodes.len() as u32;
        self.nodes.push(KdNode { bbox, split: Split::Leaf { start: 0, len: 0 } }); // placeholder
        let left = self.build(items, lo, mid);
        let right = self.build(items, mid, hi);
        self.nodes[id as usize].split = Split::Internal { left, right };
        id
    }

    /// Cull to a query shape — analytic `classify_box` (green/white/yellow) when the
    /// shape offers one, else a bbox broadphase, over the tight node rects.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        if !self.nodes.is_empty() {
            let bb = shape.bounding_box();
            self.cull_node(self.root, shape, &bb, false, &mut out);
        }
        out
    }
    fn cull_node<'a, S: Shape>(&'a self, id: u32, shape: &S, bb: &Rect, fully_inside: bool, out: &mut Vec<&'a T>) {
        match self.nodes[id as usize].split {
            Split::Leaf { start, len } => {
                let items = &self.items[start as usize..(start + len) as usize];
                if fully_inside { out.extend(items.iter()); }
                else { for it in items { if shape.contains_point(it.position()) { out.push(it); } } }
            }
            Split::Internal { left, right } => {
                for child in [left, right] {
                    if fully_inside { self.cull_node(child, shape, bb, true, out); continue; }
                    let cr = self.nodes[child as usize].bbox;
                    let state = shape.classify_box(&cr).unwrap_or(if rects_overlap(&cr, bb) { CellState::Maybe } else { CellState::Out });
                    match state {
                        CellState::Out => {}
                        CellState::In => self.cull_node(child, shape, bb, true, out),
                        CellState::Maybe => self.cull_node(child, shape, bb, false, out),
                    }
                }
            }
        }
    }

    /// k nearest neighbours — best-first descent, nearer child first, pruned by the
    /// current k-th distance.
    pub fn knn(&self, q: Point, k: usize) -> Vec<(f64, &T)> {
        if k == 0 || self.items.is_empty() { return Vec::new(); }
        let mut heap: BinaryHeap<KnnEntry<T>> = BinaryHeap::new();
        self.knn_node(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }
    fn knn_node<'a>(&'a self, id: u32, q: Point, k: usize, heap: &mut BinaryHeap<KnnEntry<'a, T>>) {
        match self.nodes[id as usize].split {
            Split::Leaf { start, len } => {
                for it in &self.items[start as usize..(start + len) as usize] { knn_offer2(heap, k, it, q); }
            }
            Split::Internal { left, right } => {
                let (dl, dr) = (rect_min_dist2(&self.nodes[left as usize].bbox, q), rect_min_dist2(&self.nodes[right as usize].bbox, q));
                let (near, far, dfar) = if dl <= dr { (left, right, dr) } else { (right, left, dl) };
                self.knn_node(near, q, k, heap);
                if dfar < knn_worst(heap, k) { self.knn_node(far, q, k, heap); }
            }
        }
    }

    /// "Thick ray-cast": every item within `radius` of the ray `origin + t·normalize
    /// (dir)`, `t ∈ [0, max_dist]`, as `(t, &item)` sorted by `t`. Built on the
    /// [`Capsule`] + `cull`.
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

    /// Serialize the built tree (the node arena + the point-ordered items) to any `Write`.
    /// Dependency-free; items via a caller closure. Same shape as [`KdTree3`](crate::KdTree3).
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(KD2_MAGIC)?;
        w_u8(w, KD2_VERSION)?;
        w_u32(w, self.capacity as u32)?;
        w_u32(w, self.root)?;
        w_u32(w, self.nodes.len() as u32)?;
        for nd in &self.nodes {
            w_rect(w, &nd.bbox)?;
            match nd.split {
                Split::Leaf { start, len } => { w_u8(w, 0)?; w_u32(w, start)?; w_u32(w, len)?; }
                Split::Internal { left, right } => { w_u8(w, 1)?; w_u32(w, left)?; w_u32(w, right)?; }
            }
        }
        w_u32(w, self.items.len() as u32)?;
        for it in &self.items { write_item(w, it)?; }
        Ok(())
    }

    /// Reload a serialized tree — no rebuild. Rejects corrupt input.
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != KD2_MAGIC { return Err(corrupt("bad KdTree2 magic")); }
        if r_u8(r)? != KD2_VERSION { return Err(corrupt("unsupported KdTree2 version")); }
        let capacity = r_u32(r)? as usize;
        let root = r_u32(r)?;
        let nn = r_u32(r)? as usize;
        let mut nodes = Vec::with_capacity(nn);
        for _ in 0..nn {
            let bbox = r_rect(r)?;
            let split = match r_u8(r)? {
                0 => Split::Leaf { start: r_u32(r)?, len: r_u32(r)? },
                1 => Split::Internal { left: r_u32(r)?, right: r_u32(r)? },
                _ => return Err(corrupt("bad KdTree2 node tag")),
            };
            nodes.push(KdNode { bbox, split });
        }
        let ni = r_u32(r)? as usize;
        let mut items = Vec::with_capacity(ni);
        for _ in 0..ni { items.push(read_item(r)?); }
        if !nodes.is_empty() && root as usize >= nodes.len() { return Err(corrupt("KdTree2 root out of range")); }
        Ok(KdTree2 { nodes, items, capacity: capacity.max(1), root })
    }
}

#[inline]
fn rects_overlap(a: &Rect, b: &Rect) -> bool {
    a.x < b.x_max() && b.x < a.x_max() && a.y < b.y_max() && b.y < a.y_max()
}


impl<T: Positioned> KdTree2<T> {
    /// Batch cull — one result list per shape (`out[i]` for `shapes[i]`). Serial; see
    /// [`Self::cull_many_par`]. Identical to calling [`Self::cull`] in a loop, which is exactly
    /// why it is worth having: it is the shape a parallel version can be swapped into.
    pub fn cull_many<'a, S: Shape>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// Parallel batch cull (feature `parallel`) — the independent queries fan out over rayon.
    /// Reads only, so there is nothing to synchronise; the crossover is in `docs/PARALLEL.md`.
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where T: Sync {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
    }

    /// Batch k-NN — one result list per query point (`out[i]` for `queries[i]`). Serial; see
    /// [`Self::knn_many_par`].
    pub fn knn_many(&self, queries: &[Point], k: usize) -> Vec<Vec<(f64, &T)>> {
        queries.iter().map(|&q| self.knn(q, k)).collect()
    }

    /// Parallel batch k-NN (feature `parallel`) — the independent queries fan out over rayon.
    #[cfg(feature = "parallel")]
    pub fn knn_many_par(&self, queries: &[Point], k: usize) -> Vec<Vec<(f64, &T)>>
    where T: Sync {
        use rayon::prelude::*;
        queries.par_iter().map(|&q| self.knn(q, k)).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circle;
    use std::io::{Read, Write};

    #[derive(Clone, Copy)]
    struct P { p: Point }
    impl Positioned for P { fn position(&self) -> Point { self.p } }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f64) / (1u64 << 24) as f64 }
        fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
    }
    /// Clustered on purpose: five tight blobs in a big empty world — the case a median
    /// split exists for, and the case a midpoint split handles worst.
    fn cloud(n: usize) -> Vec<P> {
        let mut r = Rng(0x2468_ACE1);
        let blobs = [(60.0, 60.0), (400.0, 120.0), (200.0, 380.0), (470.0, 470.0), (120.0, 250.0)];
        (0..n).map(|i| { let b = blobs[i % blobs.len()]; P { p: Point::new(b.0 + r.r(-18.0, 18.0), b.1 + r.r(-18.0, 18.0)) } }).collect()
    }

    #[test]
    fn serialize_round_trips_and_rejects_corruption() {
        let pts = cloud(2000);
        let t = KdTree2::from_items(8, pts.clone());
        let mut buf = Vec::new();
        t.serialize(&mut buf, |w, it| {
            w.write_all(&it.p.x.to_le_bytes())?;
            w.write_all(&it.p.y.to_le_bytes())
        }).expect("serialize");
        let back = KdTree2::<P>::deserialize(&mut buf.as_slice(), |r| {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?; let x = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let y = f64::from_le_bytes(b);
            Ok(P { p: Point::new(x, y) })
        }).expect("deserialize");

        assert_eq!(back.item_count(), t.item_count());
        assert_eq!(back.node_count(), t.node_count());
        assert_eq!(back.depth(), t.depth());
        // the reloaded tree must answer identically, not merely hold the same points
        for (cx, cy, rr) in [(60.0, 60.0, 25.0), (250.0, 250.0, 120.0)] {
            let c = Circle::new(Point::new(cx, cy), rr);
            let (mut a, mut b): (Vec<_>, Vec<_>) = (
                t.cull(&c).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect(),
                back.cull(&c).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect());
            a.sort(); b.sort();
            assert_eq!(a, b, "reloaded tree answers differently");
        }

        // corruption must be rejected rather than silently mis-parsed
        let mut bad = buf.clone(); bad[1] = b'X';
        assert!(KdTree2::<P>::deserialize(&mut bad.as_slice(), |_| unreachable!()).is_err(), "bad magic accepted");
        let mut old = buf.clone(); old[4] = 99;
        assert!(KdTree2::<P>::deserialize(&mut old.as_slice(), |_| unreachable!()).is_err(), "bad version accepted");
    }

    #[test]
    fn cull_matches_brute_force() {
        for n in [0usize, 1, 9, 500, 5000] {
            for cap in [1usize, 8, 24] {
                let pts = cloud(n);
                let t = KdTree2::from_items(cap, pts.clone());
                assert_eq!(t.item_count(), n);
                for (cx, cy, rr) in [(60.0, 60.0, 25.0), (250.0, 250.0, 200.0), (0.0, 0.0, 5.0), (200.0, 380.0, 40.0)] {
                    let c = Circle::new(Point::new(cx, cy), rr);
                    let mut want: Vec<(u64, u64)> = pts.iter().filter(|q| { let (dx, dy) = (q.p.x - cx, q.p.y - cy); dx * dx + dy * dy <= rr * rr }).map(|q| (q.p.x.to_bits(), q.p.y.to_bits())).collect();
                    let mut got: Vec<(u64, u64)> = t.cull(&c).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
                    want.sort(); got.sort();
                    assert_eq!(want, got, "cull != brute n={n} cap={cap} circle ({cx},{cy}) r={rr}");
                }
            }
        }
    }

    #[test]
    fn knn_matches_brute_force() {
        let pts = cloud(3000);
        let t = KdTree2::from_items(8, pts.clone());
        for q in [Point::new(60.0, 60.0), Point::new(300.0, 300.0), Point::new(-50.0, 600.0)] {
            for k in [1usize, 5, 40] {
                let mut brute: Vec<f64> = pts.iter().map(|p| { let (dx, dy) = (p.p.x - q.x, p.p.y - q.y); (dx * dx + dy * dy).sqrt() }).collect();
                brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                let got: Vec<f64> = t.knn(q, k).iter().map(|(d, _)| *d).collect();
                assert_eq!(got.len(), brute.len(), "knn count k={k}");
                // compare DISTANCES, not identity: exact ties must not fail the test
                for (a, b) in got.iter().zip(brute.iter()) { assert!((a - b).abs() <= 1e-9 * (1.0 + b), "knn dist {a} != brute {b}"); }
            }
        }
    }

    #[test]
    fn raycast_matches_brute_force() {
        let pts = cloud(2000);
        let t = KdTree2::from_items(8, pts.clone());
        let (o, d, md, rad) = (Point::new(0.0, 0.0), Point::new(1.0, 1.0), 700.0, 20.0);
        let (ux, uy) = (1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt());
        let mut want: Vec<(u64, u64)> = pts.iter().filter(|p| {
            let t0 = ((p.p.x - o.x) * ux + (p.p.y - o.y) * uy).clamp(0.0, md);
            let (cx, cy) = (o.x + ux * t0, o.y + uy * t0);
            let (dx, dy) = (p.p.x - cx, p.p.y - cy);
            dx * dx + dy * dy <= rad * rad
        }).map(|p| (p.p.x.to_bits(), p.p.y.to_bits())).collect();
        let mut got: Vec<(u64, u64)> = t.raycast(o, d, md, rad).iter().map(|(_, m)| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "raycast != brute force");
        // and the sort order is by t
        let ts: Vec<f64> = t.raycast(o, d, md, rad).iter().map(|(t, _)| *t).collect();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "raycast hits not sorted by t");
    }

    #[test]
    fn depth_beats_a_midpoint_split_on_clusters() {
        // The whole point of the median split: depth stays ~log2(n/leaf) whatever the
        // clumping. 5000 points, leaf 8 → ceil(log2(625)) = 10.
        let t = KdTree2::from_items(8, cloud(5000));
        assert!(t.depth() <= 11, "median split went deep on clusters: {}", t.depth());
        let mut leaves = 0usize;
        let mut counted = 0usize;
        t.visit_leaves(|_, c| { leaves += 1; counted += c; });
        assert_eq!(counted, 5000, "leaves must partition the item set");
        assert!(leaves >= 5000 / 8, "too few leaves: {leaves}");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_build_is_identical_to_serial() {
        for n in [0usize, 1, 7, 100, 5000, 20000] {
            for cap in [1usize, 8, 32] {
                let a = KdTree2::from_items(cap, cloud(n));
                let b = KdTree2::from_items_par(cap, cloud(n));
                assert_eq!(a.node_count(), b.node_count(), "node count n={n} cap={cap}");
                assert_eq!(a.depth(), b.depth());
                let mut la = Vec::new(); a.visit_leaves(|r, c| la.push((r.x.to_bits(), r.y.to_bits(), c)));
                let mut lb = Vec::new(); b.visit_leaves(|r, c| lb.push((r.x.to_bits(), r.y.to_bits(), c)));
                assert_eq!(la, lb, "leaf layout differs n={n} cap={cap}");
                let c = Circle::new(Point::new(200.0, 380.0), 60.0);
                let (mut ga, mut gb): (Vec<_>, Vec<_>) = (
                    a.cull(&c).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect(),
                    b.cull(&c).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect());
                ga.sort(); gb.sort();
                assert_eq!(ga, gb, "cull differs n={n} cap={cap}");
            }
        }
    }
}
