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
use crate::tree3::{Aabb, Point3, Positioned3, Shape3};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy)]
    struct P(Point3);
    impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

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
}
