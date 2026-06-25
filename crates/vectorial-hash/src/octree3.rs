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
use crate::tree3::{aabb_min_dist2, knn_offer, knn_worst, Aabb, ItemRef, KnnEntry, Point3, Positioned3, Shape3};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ONodeId(pub u32);

pub struct ONode<T> {
    pub bbox: Aabb,
    pub parent: Option<ONodeId>,
    pub children: Option<[ONodeId; 8]>,
    pub items: Vec<T>,
    /// Parallel to `items`: the stable handle of each item (see the
    /// [`crate::ItemRef`] layer on [`crate::Tree3`]). Empty on internal nodes.
    hs: Vec<u32>,
}

/// Where a handle's item currently lives (leaf node + slot).
#[derive(Copy, Clone)]
struct OItemLoc { node: ONodeId, slot: u32 }

pub struct Octree3<T: Positioned3> {
    nodes: Vec<ONode<T>>,
    free: Vec<ONodeId>,
    locs: Vec<OItemLoc>,
    free_handles: Vec<u32>,
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
            nodes: vec![ONode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() }],
            free: Vec::new(),
            locs: Vec::new(),
            free_handles: Vec::new(),
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

    // ---- stable ItemRef handle layer (mirrors Tree3) ----
    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() { h }
        else { let h = self.locs.len() as u32; self.locs.push(OItemLoc { node: ONodeId(0), slot: 0 }); h }
    }
    fn push_h(&mut self, node: ONodeId, item: T, h: u32) {
        let slot = self.get(node).items.len() as u32;
        let n = self.get_mut(node);
        n.items.push(item);
        n.hs.push(h);
        self.locs[h as usize] = OItemLoc { node, slot };
    }
    fn swap_remove_h(&mut self, node: ONodeId, slot: usize) -> (T, u32) {
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

    pub fn insert(&mut self, item: T) -> bool {
        self.insert_ref(item).is_some()
    }

    /// Insert and return a stable [`crate::ItemRef`] for O(1) `update_ref` /
    /// `remove_ref` (skips the predicate scan). `None` if outside the root.
    pub fn insert_ref(&mut self, item: T) -> Option<ItemRef> {
        let p = item.position();
        if !self.get(self.root).bbox.contains(p) { return None; }
        let leaf = self.locate(p);
        let h = self.alloc_handle();
        self.push_h(leaf, item, h);
        if self.get(leaf).items.len() > self.item_limit { self.divide(leaf); }
        Some(ItemRef(h))
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
        let (b, items, hs) = {
            let n = self.get_mut(id);
            (n.bbox, std::mem::take(&mut n.items), std::mem::take(&mut n.hs))
        };
        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || b.w.max(b.h).max(b.d) <= self.min_cell {
            let n = self.get_mut(id);
            n.items = items;
            n.hs = hs;
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
                    kids[oi] = self.alloc(ONode { bbox: oct, parent: Some(id), children: None, items: Vec::new(), hs: Vec::new() });
                    oi += 1;
                }
            }
        }
        for (item, h) in items.into_iter().zip(hs) {
            let p = item.position();
            let k = *kids.iter().find(|&&k| self.get(k).bbox.contains(p)).expect("octants tile the parent");
            self.push_h(k, item, h);
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
        let idx = match self.get(leaf).items.iter().position(&predicate) {
            Some(i) => i, None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);
        let np = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(np) { return true; }
        self.relocate(leaf, idx, np)
    }

    /// O(1) relocation via a stable [`crate::ItemRef`] — no locate, no scan.
    pub fn update_ref<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> bool {
        let loc = self.locs[r.0 as usize];
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) { return true; }
        self.relocate(node, slot, np)
    }

    /// Shared tail of `update`/`update_ref`: the item at `(leaf, slot)` left its
    /// leaf — ascend to the LCA, re-descend; drop (freeing its handle) if out.
    fn relocate(&mut self, leaf: ONodeId, slot: usize, np: Point3) -> bool {
        let mut anc = self.get(leaf).parent;
        let lca = loop {
            match anc {
                Some(a) if self.get(a).bbox.contains(np) => break a,
                Some(a) => anc = self.get(a).parent,
                None => {
                    let (_, h) = self.swap_remove_h(leaf, slot);
                    self.free_handles.push(h);
                    self.try_merge_up(leaf);
                    return false;
                }
            }
        };
        let (item, h) = self.swap_remove_h(leaf, slot);
        let dest = self.locate_from(lca, np);
        self.push_h(dest, item, h);
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
        let idx = self.get(leaf).items.iter().position(&predicate)?;
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.free_handles.push(h);
        self.try_merge_up(leaf);
        Some(item)
    }

    /// Remove the item behind a stable [`crate::ItemRef`] in O(1).
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.locs[r.0 as usize];
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handles.push(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    fn try_merge_up(&mut self, mut node: ONodeId) {
        loop {
            let parent = match self.get(node).parent { Some(p) => p, None => return };
            let kids = self.get(parent).children.expect("parent has children");
            if kids.iter().any(|&k| self.get(k).children.is_some()) { return; }
            let combined: usize = kids.iter().map(|&k| self.get(k).items.len()).sum();
            if combined > self.merge_limit { return; }
            let mut merged: Vec<T> = Vec::with_capacity(combined);
            let mut merged_hs: Vec<u32> = Vec::with_capacity(combined);
            for &k in &kids {
                merged.append(&mut std::mem::take(&mut self.get_mut(k).items));
                merged_hs.append(&mut std::mem::take(&mut self.get_mut(k).hs));
            }
            let pnode = self.get_mut(parent);
            pnode.items = merged;
            pnode.hs = merged_hs;
            pnode.children = None;
            let len = self.get(parent).hs.len();
            for slot in 0..len {
                let h = self.get(parent).hs[slot];
                self.locs[h as usize] = OItemLoc { node: parent, slot: slot as u32 };
            }
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

    #[test]
    fn octree_update_ref_churn_matches_brute() {
        // Build with insert_ref, churn with update_ref/remove_ref/insert_ref
        // (the O(1) stable-handle path), and verify cull == brute (by id) + count.
        use crate::ItemRef;
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point3 }
        impl Positioned3 for M { fn position(&self) -> Point3 { self.p } }

        let mut x = 0x0C7_EF00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let world = Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0);
        let mut tree = Octree3::<M>::new(world, 6);
        let rp = |rng: &mut dyn FnMut() -> f64| Point3::new(rng() * 256.0, rng() * 256.0, rng() * 256.0);
        let mut live: std::collections::HashMap<u32, (ItemRef, Point3)> = std::collections::HashMap::new();
        let mut next = 0u32;
        for _ in 0..2000 {
            let p = rp(&mut rng);
            let r = tree.insert_ref(M { id: next, p }).unwrap();
            live.insert(next, (r, p)); next += 1;
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
                tree.remove_ref(live[&id].0);
                live.remove(&id);
            } else {
                let p = rp(&mut rng);
                let r = tree.insert_ref(M { id: next, p }).unwrap();
                live.insert(next, (r, p)); next += 1;
            }
        }
        assert_eq!(tree.item_count(), live.len(), "octree handle-churn item count drift");
        for (cx, cy, cz, r) in [(128.0, 128.0, 128.0, 30.0), (60.0, 200.0, 90.0, 50.0), (10.0, 10.0, 10.0, 80.0)] {
            let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
            let mut want: Vec<u32> = live.iter()
                .filter(|(_, (_, p))| { let dx = p.x - cx; let dy = p.y - cy; let dz = p.z - cz; dx * dx + dy * dy + dz * dz <= r * r })
                .map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = tree.cull(&sphere).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "octree handle-churn cull != brute ({cx},{cy},{cz}) r={r}");
        }
    }
}
