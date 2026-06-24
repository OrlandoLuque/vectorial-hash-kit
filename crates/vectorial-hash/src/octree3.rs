//! Octree — the 8-way (2×2×2) 3D tree, the 3D analogue of [`crate::QuadTree`]
//! (just as [`crate::Tree3`] is the analogue of the binary [`crate::Tree`]).
//! Kept beside `Tree3` for the head-to-head comparison the 2D quad-vs-binary
//! result raised: at a tuned `item_limit` the two were within ~2% in 2D, and
//! this lets the same question be measured in 3D.
//!
//! Shares the [`Shape3`] machinery (sphere/polyhedron classification, voxel
//! raster) with `Tree3`, so any cull-speed difference is the structure, not
//! the plumbing. Splits a leaf into eight equal octants; the free-list
//! reclaims merged-out slots.

use crate::template::CellState;
use crate::tree3::{aabb_min_dist2, knn_offer, knn_worst, Aabb, KnnEntry, Point3, Positioned3, Shape3};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ONodeId(pub u32);

pub struct ONode<T> {
    pub bbox: Aabb,
    pub parent: Option<ONodeId>,
    pub children: Option<[ONodeId; 8]>,
    pub items: Vec<T>,
}

pub struct Octree3<T: Positioned3> {
    nodes: Vec<ONode<T>>,
    free: Vec<ONodeId>,
    pub item_limit: usize,
    pub merge_limit: usize,
    min_cell: f64,
    pub root: ONodeId,
}

impl<T: Positioned3> Octree3<T> {
    pub fn new(bbox: Aabb, item_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let min_cell = bbox.w.max(bbox.h).max(bbox.d) * 1e-12;
        Self {
            nodes: vec![ONode { bbox, parent: None, children: None, items: Vec::new() }],
            free: Vec::new(),
            item_limit,
            merge_limit: item_limit,
            min_cell,
            root: ONodeId(0),
        }
    }

    #[inline] pub fn get(&self, id: ONodeId) -> &ONode<T> { &self.nodes[id.0 as usize] }
    #[inline] fn get_mut(&mut self, id: ONodeId) -> &mut ONode<T> { &mut self.nodes[id.0 as usize] }
    fn alloc(&mut self, n: ONode<T>) -> ONodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0 as usize] = n;
            id
        } else {
            let id = ONodeId(self.nodes.len() as u32);
            self.nodes.push(n);
            id
        }
    }
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

    pub fn locate(&self, p: Point3) -> ONodeId {
        let mut cur = self.root;
        loop {
            match self.get(cur).children {
                None => return cur,
                Some(kids) => {
                    cur = *kids.iter().find(|&&k| self.get(k).bbox.contains(p))
                        .expect("octants tile the parent");
                }
            }
        }
    }

    fn divide(&mut self, id: ONodeId) {
        let (b, items) = {
            let n = self.get_mut(id);
            (n.bbox, std::mem::take(&mut n.items))
        };
        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || b.w.max(b.h).max(b.d) <= self.min_cell {
            self.get_mut(id).items = items;
            return;
        }
        let (hw, hh, hd) = (b.w / 2.0, b.h / 2.0, b.d / 2.0);
        let mut kids = [ONodeId(0); 8];
        let mut oi = 0;
        for sz in 0..2 {
            for sy in 0..2 {
                for sx in 0..2 {
                    let oct = Aabb::new(
                        b.x + sx as f64 * hw, b.y + sy as f64 * hh, b.z + sz as f64 * hd,
                        hw, hh, hd,
                    );
                    kids[oi] = self.alloc(ONode { bbox: oct, parent: Some(id), children: None, items: Vec::new() });
                    oi += 1;
                }
            }
        }
        for item in items {
            let p = item.position();
            let k = *kids.iter().find(|&&k| self.get(k).bbox.contains(p)).expect("octants tile the parent");
            self.get_mut(k).items.push(item);
        }
        self.get_mut(id).children = Some(kids);
        for k in kids {
            if self.get(k).items.len() > self.item_limit { self.divide(k); }
        }
    }

    /// Relocate via ascend-to-LCA — the 3D-binary [`crate::Tree3::update`]
    /// strategy, ported to the 8-way split so the *dynamic* octree can be
    /// measured against the binary tree (instead of a full per-frame rebuild).
    /// Mutate in place; if the item leaves its leaf, ascend to the lowest
    /// ancestor containing the new position and descend from there. Returns
    /// `false` if not found or pushed out of the root.
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

    fn locate_from(&self, start: ONodeId, p: Point3) -> ONodeId {
        let mut cur = start;
        loop {
            match self.get(cur).children {
                None => return cur,
                Some(kids) => {
                    cur = *kids.iter().find(|&&k| self.get(k).bbox.contains(p))
                        .expect("octants tile the parent");
                }
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

    fn try_merge_up(&mut self, mut node: ONodeId) {
        loop {
            let parent = match self.get(node).parent { Some(p) => p, None => return };
            let kids = self.get(parent).children.expect("parent has children");
            if kids.iter().any(|&k| self.get(k).children.is_some()) { return; }
            let combined: usize = kids.iter().map(|&k| self.get(k).items.len()).sum();
            if combined > self.merge_limit { return; }
            let mut merged: Vec<T> = Vec::with_capacity(combined);
            for &k in &kids { merged.append(&mut std::mem::take(&mut self.get_mut(k).items)); }
            let pnode = self.get_mut(parent);
            pnode.items = merged;
            pnode.children = None;
            for k in kids { self.free.push(k); }
            node = parent;
        }
    }

    pub fn visit_leaves<F: FnMut(&ONode<T>)>(&self, mut f: F) {
        self.visit_from(self.root, &mut f);
    }
    fn visit_from<F: FnMut(&ONode<T>)>(&self, id: ONodeId, f: &mut F) {
        match self.get(id).children {
            Some(kids) => { for k in kids { self.visit_from(k, f); } }
            None => f(self.get(id)),
        }
    }
    pub fn item_count(&self) -> usize { let mut n = 0; self.visit_leaves(|l| n += l.items.len()); n }
    pub fn leaf_count(&self) -> usize { let mut n = 0; self.visit_leaves(|_| n += 1); n }

    /// Same cull contract as [`crate::Tree3::cull`].
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        self.cull_recurse(self.root, shape, false, &mut out);
        out
    }

    fn cull_recurse<'a, S: Shape3>(&'a self, id: ONodeId, shape: &S, fully_inside: bool, out: &mut Vec<&'a T>) {
        let node = self.get(id);
        if fully_inside {
            match node.children {
                Some(kids) => { for k in kids { self.cull_recurse(k, shape, true, out); } }
                None => out.extend(node.items.iter()),
            }
            return;
        }
        match node.children {
            Some(kids) => {
                for k in kids {
                    let cb = self.get(k).bbox;
                    match shape.classify_aabb(&cb) {
                        CellState::Out => {}
                        CellState::In => self.cull_recurse(k, shape, true, out),
                        CellState::Maybe => self.cull_recurse(k, shape, false, out),
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

    /// The `k` nearest items to `q`, sorted ascending by distance — the same
    /// best-first, bbox-pruned search as [`crate::Tree3::knn`], over the 8-way
    /// split. The eight children are visited nearest-box-first so the pruning
    /// bound tightens as early as possible.
    pub fn knn(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        if k == 0 { return Vec::new(); }
        let mut heap: std::collections::BinaryHeap<KnnEntry<T>> = std::collections::BinaryHeap::new();
        self.knn_recurse(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    fn knn_recurse<'a>(&'a self, id: ONodeId, q: Point3, k: usize, heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>) {
        match self.get(id).children {
            None => {
                for it in &self.get(id).items {
                    knn_offer(heap, k, it, q);
                }
            }
            Some(kids) => {
                // Order the 8 octants by their box's nearest-point distance,
                // then descend nearest-first, pruning by the current k-th.
                let mut order: [(f64, ONodeId); 8] = [(0.0, ONodeId(0)); 8];
                for (i, &kid) in kids.iter().enumerate() {
                    order[i] = (aabb_min_dist2(&self.get(kid).bbox, q), kid);
                }
                order.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (d, kid) in order {
                    if d < knn_worst(heap, k) {
                        self.knn_recurse(kid, q, k, heap);
                    }
                }
            }
        }
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
    fn octree_knn_matches_brute_force() {
        // Octree3::knn must return the same k smallest distances as a brute sort.
        let mut x = 0x0C73_E33Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..4000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Octree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }
        let d2 = |a: Point3, q: Point3| { let (dx, dy, dz) = (a.x - q.x, a.y - q.y, a.z - q.z); dx * dx + dy * dy + dz * dz };
        for _ in 0..40 {
            let q = Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
            for k in [1usize, 5, 17] {
                let got = tree.knn(q, k);
                assert_eq!(got.len(), k.min(pts.len()));
                assert!(got.windows(2).all(|w| w[0].0 <= w[1].0), "octree knn not sorted");
                let mut bf: Vec<f64> = pts.iter().map(|p| d2(p.0, q)).collect();
                bf.sort_by(|a, b| a.total_cmp(b));
                for (i, (dist, _)) in got.iter().enumerate() {
                    assert!((dist * dist - bf[i]).abs() <= 1e-6 * (1.0 + bf[i]),
                        "octree knn dist #{i} mismatch: {} vs {}", dist * dist, bf[i]);
                }
            }
        }
    }

    #[test]
    fn octree_cull_matches_brute() {
        let mut x = 0x515E_3D3Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<P> = (0..3000).map(|_| P(Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = Octree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8);
        for p in &pts { tree.insert(*p); }
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 40.0), (40.0, 40.0, 40.0, 60.0), (0.0, 0.0, 0.0, 100.0)] {
            let s = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<(u64, u64, u64)> = pts.iter().filter(|p| {
                let dx = p.0.x - cx; let dy = p.0.y - cy; let dz = p.0.z - cz; dx*dx+dy*dy+dz*dz <= r*r
            }).map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = tree.cull(&s).iter()
                .map(|p| (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits())).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "octree cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }
    }

    #[test]
    fn octree_cull_matches_brute_after_churn() {
        // Mirror of Tree3's churn test: build, churn with update/remove/insert,
        // then verify the dynamic octree's cull still equals brute force and the
        // item count tracks the ground truth. Exercises `update`'s ascend-to-LCA
        // relocation and merge-up bookkeeping under heavy movement.
        let mut x = 0x0C7_0EEFu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };

        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Octree3::<M>::new(world, 6);
        let mut live: std::collections::HashMap<u32, Point3> = std::collections::HashMap::new();
        let mut next_id = 0u32;

        let rp = |rng: &mut dyn FnMut() -> f64| Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);

        for _ in 0..2000 {
            let p = rp(&mut rng);
            tree.insert(M { id: next_id, p });
            live.insert(next_id, p);
            next_id += 1;
        }

        for _ in 0..6000 {
            let roll = rng();
            if roll < 0.6 && !live.is_empty() {
                let ids: Vec<u32> = live.keys().copied().collect();
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id];
                let np = rp(&mut rng);
                let ok = tree.update(old, |m| m.id == id, |m| m.p = np);
                if ok { live.insert(id, np); } else { live.remove(&id); }
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

        assert_eq!(tree.item_count(), live.len(), "octree item count drifted after churn");

        for (cx, cy, cz, r) in [(128.0,128.0,128.0,30.0),(60.0,200.0,90.0,50.0),(10.0,10.0,10.0,80.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<u32> = live.iter()
                .filter(|(_, p)| { let dx=p.x-cx; let dy=p.y-cy; let dz=p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
                .map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = tree.cull(&sphere).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "post-churn octree cull != brute for sphere ({cx},{cy},{cz}) r={r}");
        }
    }
}
