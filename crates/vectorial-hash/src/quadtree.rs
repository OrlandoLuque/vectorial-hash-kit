//! Reference quadtree, kept alongside the binary-split [`crate::Tree`] for
//! head-to-head comparisons (benchmarks, the critters demo's dual mode).
//!
//! Mirrors the tree's full dynamic contract — `insert`, `remove`, `update`
//! with the merge rule, `cull` with the per-cell-size template machinery —
//! but always splits into 4 equal quadrants. Shares the classification and
//! leaf-resolution helpers with `Tree::cull`, so any speed difference
//! between the two structures is the structure itself, not the plumbing.
//!
//! The merge rule is the quadtree-granularity analogue of the paper's
//! merge-up: a parent whose **four** children are all leaves re-absorbs them
//! when their combined items fit within `merge_limit`.

use crate::culling::{classify_child, collect_matching_items, SizeCache};
use crate::geom::{Point, Rect};
use crate::serde_io::{corrupt, r_f64, r_rect, r_u32, r_u64, r_u8, w_f64, w_rect, w_u32, w_u64, w_u8};
use crate::tree::{Positioned, UpdateStrategy};
use crate::tree3::ItemRef;
use crate::CellState;
use crate::Shape;
use std::io::{self, Read, Write};

/// Stable handle into the quadtree's node arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct QNodeId(pub u32);

pub struct QNode<T> {
    pub bbox: Rect,
    pub parent: Option<QNodeId>,
    pub children: Option<[QNodeId; 4]>,
    pub items: Vec<T>,
    /// Parallel to `items`: each item's stable handle (the [`ItemRef`] layer).
    hs: Vec<u32>,
}

/// Where a handle's item currently lives (leaf node + slot).
#[derive(Copy, Clone)]
struct QItemLoc { node: QNodeId, slot: u32 }

pub struct QuadTree<T: Positioned> {
    nodes: Vec<QNode<T>>,
    /// Slots freed by 4-way merge-ups, reused before the arena grows — see
    /// [`crate::Tree`]'s free-list for the rationale.
    free: Vec<QNodeId>,
    locs: Vec<QItemLoc>,
    free_handles: Vec<u32>,
    /// A leaf splits when it holds more than this many items.
    pub item_limit: usize,
    /// Four sibling leaves merge back into their parent when their combined
    /// items fit within this. Defaults to `item_limit`.
    pub merge_limit: usize,
    /// Leaves whose side is at or below this never split further.
    min_cell: f64,
    pub root: QNodeId,
}

impl<T: Positioned> QuadTree<T> {
    pub fn new(bbox: Rect, item_limit: usize) -> Self {
        Self::with_limits(bbox, item_limit, item_limit)
    }

    pub fn with_limits(bbox: Rect, item_limit: usize, merge_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        assert!(merge_limit <= item_limit, "merge_limit must be <= item_limit");
        let min_cell = bbox.width.max(bbox.height) * 1e-12;
        let root = QNode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() };
        Self { nodes: vec![root], free: Vec::new(), locs: Vec::new(), free_handles: Vec::new(), item_limit, merge_limit, min_cell, root: QNodeId(0) }
    }

    /// Empty the tree, retaining capacity — see [`crate::Tree3::clear`].
    pub fn clear(&mut self) {
        let bbox = self.get(self.root).bbox;
        self.nodes.clear();
        self.nodes.push(QNode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() });
        self.free.clear();
        self.locs.clear();
        self.free_handles.clear();
        self.root = QNodeId(0);
    }

    pub fn get(&self, id: QNodeId) -> &QNode<T> {
        &self.nodes[id.0 as usize]
    }

    fn get_mut(&mut self, id: QNodeId) -> &mut QNode<T> {
        &mut self.nodes[id.0 as usize]
    }

    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() { h }
        else { let h = self.locs.len() as u32; self.locs.push(QItemLoc { node: QNodeId(0), slot: 0 }); h }
    }
    fn push_h(&mut self, node: QNodeId, item: T, h: u32) {
        let slot = self.get(node).items.len() as u32;
        let n = self.get_mut(node);
        n.items.push(item);
        n.hs.push(h);
        self.locs[h as usize] = QItemLoc { node, slot };
    }
    fn swap_remove_h(&mut self, node: QNodeId, slot: usize) -> (T, u32) {
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

    fn alloc(&mut self, node: QNode<T>) -> QNodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0 as usize] = node;
            id
        } else {
            let id = QNodeId(self.nodes.len() as u32);
            self.nodes.push(node);
            id
        }
    }

    /// Insert an item. Returns `false` if its position falls outside the root bbox.
    pub fn insert(&mut self, item: T) -> bool {
        self.insert_ref(item).is_some()
    }

    /// Insert and return a stable [`ItemRef`] for O(1) `update_ref`/`remove_ref`.
    pub fn insert_ref(&mut self, item: T) -> Option<ItemRef> {
        let pos = item.position();
        if !self.get(self.root).bbox.contains(pos) {
            return None;
        }
        let leaf = self.locate(pos);
        let h = self.alloc_handle();
        self.push_h(leaf, item, h);
        if self.get(leaf).items.len() > self.item_limit {
            self.divide(leaf);
        }
        Some(ItemRef(h))
    }

    /// O(1) relocation through a stable [`ItemRef`] (no locate, no scan).
    pub fn update_ref<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> bool {
        let loc = self.locs[r.0 as usize];
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) {
            return true;
        }
        self.relocate(node, slot, np)
    }

    /// Remove the item behind a stable [`ItemRef`] in O(1).
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.locs[r.0 as usize];
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handles.push(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    /// Shared LCA relocate tail (predicate `update` and `update_ref`).
    fn relocate(&mut self, leaf: QNodeId, slot: usize, new_pos: Point) -> bool {
        let lca = match self.ascend_to_lca(leaf, new_pos) {
            Some(id) => id,
            None => {
                let (_, h) = self.swap_remove_h(leaf, slot);
                self.free_handles.push(h);
                self.try_merge_up(leaf);
                return false;
            }
        };
        let (item, h) = self.swap_remove_h(leaf, slot);
        let dest = self.locate_from(lca, new_pos);
        self.push_h(dest, item, h);
        if self.get(dest).items.len() > self.item_limit {
            self.divide(dest);
        }
        self.try_merge_up(leaf);
        true
    }

    /// Find the leaf that contains `point`. Caller must ensure `point` is in-bounds.
    pub fn locate(&self, point: Point) -> QNodeId {
        self.locate_from(self.root, point)
    }

    /// Like [`QuadTree::locate`] but starting the descent at an arbitrary node.
    pub fn locate_from(&self, start: QNodeId, point: Point) -> QNodeId {
        let mut current = start;
        loop {
            match self.get(current).children {
                None => return current,
                Some(kids) => {
                    current = *kids
                        .iter()
                        .find(|&&k| self.get(k).bbox.contains(point))
                        .expect("quadrants tile the parent");
                }
            }
        }
    }

    /// Walk parents up from `leaf` until one whose bbox contains `point`.
    /// Returns `None` if the search escapes the root.
    fn ascend_to_lca(&self, leaf: QNodeId, point: Point) -> Option<QNodeId> {
        let mut node = self.get(leaf).parent?;
        loop {
            if self.get(node).bbox.contains(point) {
                return Some(node);
            }
            node = self.get(node).parent?;
        }
    }

    /// Remove the first matching item in the leaf at `point`; same contract
    /// as [`crate::Tree::remove`], with the 4-way merge rule.
    pub fn remove<F: Fn(&T) -> bool>(&mut self, point: Point, predicate: F) -> Option<T> {
        if !self.get(self.root).bbox.contains(point) {
            return None;
        }
        let leaf = self.locate(point);
        let idx = self.get(leaf).items.iter().position(&predicate)?;
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.free_handles.push(h);
        self.try_merge_up(leaf);
        Some(item)
    }

    /// Same contract as [`crate::Tree::update`].
    pub fn update<F, M>(&mut self, old_position: Point, predicate: F, mutator: M) -> bool
    where
        F: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        self.update_with(UpdateStrategy::default(), old_position, predicate, mutator)
    }

    /// Same contract as [`crate::Tree::update_with`]. The `LcaRopes` variant
    /// falls back to `Lca` here — the quadtree has no rope lists.
    pub fn update_with<F, M>(
        &mut self,
        strategy: UpdateStrategy,
        old_position: Point,
        predicate: F,
        mutator: M,
    ) -> bool
    where
        F: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        if !self.get(self.root).bbox.contains(old_position) {
            return false;
        }
        let leaf = self.locate(old_position);
        let idx = match self.get(leaf).items.iter().position(&predicate) {
            Some(i) => i,
            None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);

        let new_pos = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(new_pos) {
            return true;
        }

        match strategy {
            UpdateStrategy::Legacy => {
                // remove + re-insert reassigns the handle; the Lca path preserves it.
                let (item, h) = self.swap_remove_h(leaf, idx);
                self.free_handles.push(h);
                self.try_merge_up(leaf);
                self.insert(item)
            }
            UpdateStrategy::Lca | UpdateStrategy::LcaRopes => self.relocate(leaf, idx, new_pos),
        }
    }

    /// 4-way merge rule: collapse parents whose four children are all leaves
    /// with combined items within `merge_limit`; cascade upward.
    fn try_merge_up(&mut self, mut node: QNodeId) {
        loop {
            let parent_id = match self.get(node).parent {
                Some(p) => p,
                None => return,
            };
            let kids = self.get(parent_id).children.expect("parent has children");
            if kids.iter().any(|&k| self.get(k).children.is_some()) {
                return;
            }
            let combined: usize = kids.iter().map(|&k| self.get(k).items.len()).sum();
            if combined > self.merge_limit {
                return;
            }
            let mut merged: Vec<T> = Vec::with_capacity(combined);
            let mut merged_hs: Vec<u32> = Vec::with_capacity(combined);
            for &k in &kids {
                merged.append(&mut std::mem::take(&mut self.get_mut(k).items));
                merged_hs.append(&mut std::mem::take(&mut self.get_mut(k).hs));
            }
            let parent = self.get_mut(parent_id);
            parent.items = merged;
            parent.hs = merged_hs;
            parent.children = None;
            let len = self.get(parent_id).hs.len();
            for slot in 0..len {
                let h = self.get(parent_id).hs[slot];
                self.locs[h as usize] = QItemLoc { node: parent_id, slot: slot as u32 };
            }
            for k in kids {
                self.free.push(k);
            }
            node = parent_id;
        }
    }

    /// Split a leaf into four quadrants; same degenerate-subdivision guards
    /// as the binary tree (identical positions, minimum cell size).
    fn divide(&mut self, id: QNodeId) {
        let (bbox, items, hs) = {
            let n = self.get_mut(id);
            let items = std::mem::take(&mut n.items);
            let hs = std::mem::take(&mut n.hs);
            (n.bbox, items, hs)
        };

        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.width.max(bbox.height) <= self.min_cell {
            let n = self.get_mut(id);
            n.items = items;
            n.hs = hs;
            return;
        }

        let hw = bbox.width / 2.0;
        let hh = bbox.height / 2.0;
        let quads = [
            Rect::new(bbox.x, bbox.y, hw, hh),
            Rect::new(bbox.x + hw, bbox.y, hw, hh),
            Rect::new(bbox.x, bbox.y + hh, hw, hh),
            Rect::new(bbox.x + hw, bbox.y + hh, hw, hh),
        ];
        let mut kids = [QNodeId(0); 4];
        for (i, q) in quads.iter().enumerate() {
            kids[i] = self.alloc(QNode {
                bbox: *q,
                parent: Some(id),
                children: None,
                items: Vec::new(),
                hs: Vec::new(),
            });
        }
        for (item, h) in items.into_iter().zip(hs) {
            let pos = item.position();
            let k = kids
                .iter()
                .copied()
                .find(|&k| self.get(k).bbox.contains(pos))
                .expect("quadrants tile the parent");
            self.push_h(k, item, h);
        }
        self.get_mut(id).children = Some(kids);
        for k in kids {
            if self.get(k).items.len() > self.item_limit {
                self.divide(k);
            }
        }
    }

    /// Arena capacity (high-water-mark; bounded under churn by the
    /// free-list). [`QuadTree::live_node_count`] is the reachable count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn live_node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// Visit every live leaf reachable from the root, depth-first.
    pub fn visit_leaves<F: FnMut(QNodeId, &QNode<T>)>(&self, mut f: F) {
        self.visit_leaves_from(self.root, &mut f);
    }

    fn visit_leaves_from<F: FnMut(QNodeId, &QNode<T>)>(&self, id: QNodeId, f: &mut F) {
        match self.get(id).children {
            Some(kids) => {
                for k in kids {
                    self.visit_leaves_from(k, f);
                }
            }
            None => f(id, self.get(id)),
        }
    }

    pub fn item_count(&self) -> usize {
        let mut n = 0;
        self.visit_leaves(|_, leaf| n += leaf.items.len());
        n
    }

    pub fn leaf_count(&self) -> usize {
        let mut n = 0;
        self.visit_leaves(|_, _| n += 1);
        n
    }

    /// Same culling contract as [`crate::Tree::cull`]: per-cell-size template
    /// selection with the per-execution size cache, single-grid fallback,
    /// and the shared leaf-resolution (bbox pre-filter + 1×1 raster).
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bbox = shape.bounding_box();
        let mut sizes = SizeCache::new();
        self.cull_recurse(self.root, shape, &bbox, false, &mut sizes, &mut out);
        out
    }

    /// Batch cull — see [`crate::Tree3::cull_many`].
    pub fn cull_many<'a, S: Shape>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>> {
        shapes.iter().map(|s| self.cull(s)).collect()
    }

    /// The `k` nearest items to `q` — see [`crate::Tree::knn`]. 4-way best-first
    /// descent: children are visited nearest-box-first and pruned once their box
    /// is farther than the current k-th nearest.
    pub fn knn(&self, q: Point, k: usize) -> Vec<(f64, &T)> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap = std::collections::BinaryHeap::new();
        self.knn_recurse(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    fn knn_recurse<'a>(&'a self, id: QNodeId, q: Point, k: usize, heap: &mut std::collections::BinaryHeap<crate::tree3::KnnEntry<'a, T>>) {
        let node = self.get(id);
        match node.children {
            None => {
                for it in &node.items {
                    crate::tree::knn_offer2(heap, k, it, q);
                }
            }
            Some(children) => {
                let mut kids: [(QNodeId, f64); 4] = [(children[0], 0.0); 4];
                for (slot, &c) in children.iter().enumerate() {
                    kids[slot] = (c, crate::tree::rect_min_dist2(&self.get(c).bbox, q));
                }
                kids.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (c, d) in kids {
                    if d < crate::tree3::knn_worst(heap, k) {
                        self.knn_recurse(c, q, k, heap);
                    }
                }
            }
        }
    }

    /// Parallel batch cull — see [`crate::Tree3::cull_many_par`].
    #[cfg(feature = "parallel")]
    pub fn cull_many_par<'a, S: Shape + Sync>(&'a self, shapes: &[S]) -> Vec<Vec<&'a T>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        shapes.par_iter().map(|s| self.cull(s)).collect()
    }

    fn cull_recurse<'a, S: Shape>(
        &'a self,
        id: QNodeId,
        shape: &S,
        shape_bbox: &Rect,
        fully_inside: bool,
        sizes: &mut SizeCache,
        out: &mut Vec<&'a T>,
    ) {
        let node = self.get(id);

        if fully_inside {
            match node.children {
                Some(kids) => {
                    for k in kids {
                        self.cull_recurse(k, shape, shape_bbox, true, sizes, out);
                    }
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }

        match node.children {
            Some(kids) => {
                for k in kids {
                    let child_bbox = self.get(k).bbox;
                    match classify_child(shape, shape_bbox, &child_bbox, sizes) {
                        CellState::Out => {}
                        CellState::In => self.cull_recurse(k, shape, shape_bbox, true, sizes, out),
                        CellState::Maybe => {
                            self.cull_recurse(k, shape, shape_bbox, false, sizes, out)
                        }
                    }
                }
            }
            None => collect_matching_items(&node.items, shape, shape_bbox, out),
        }
    }
}

// ------------------------------------------------------------- serialization

const QUADTREE_MAGIC: &[u8; 4] = b"VHQ2";
const QUADTREE_VERSION: u8 = 1;

impl<T: Positioned> QuadTree<T> {
    /// Serialize the **built** quadtree (exact arena, free-list, params — no
    /// rebuild on load) to `w`. Items are written by `write_item`. Mirrors
    /// [`crate::Tree3::serialize`] over the 2D 4-way split.
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(QUADTREE_MAGIC)?;
        w_u8(w, QUADTREE_VERSION)?;
        w_u64(w, self.item_limit as u64)?;
        w_u64(w, self.merge_limit as u64)?;
        w_f64(w, self.min_cell)?;
        w_u32(w, self.root.0)?;
        w_u32(w, self.free.len() as u32)?;
        for f in &self.free { w_u32(w, f.0)?; }
        w_u32(w, self.nodes.len() as u32)?;
        for n in &self.nodes {
            w_rect(w, &n.bbox)?;
            match n.parent {
                Some(p) => { w_u8(w, 1)?; w_u32(w, p.0)?; }
                None => w_u8(w, 0)?,
            }
            match n.children {
                Some(kids) => { w_u8(w, 1)?; for k in kids { w_u32(w, k.0)?; } }
                None => w_u8(w, 0)?,
            }
            w_u32(w, n.items.len() as u32)?;
            for it in &n.items { write_item(w, it)?; }
            for &h in &n.hs { w_u32(w, h)?; }
        }
        Ok(())
    }

    /// Inverse of [`QuadTree::serialize`]: rebuild the exact quadtree from `r`,
    /// reading each item with `read_item` (must mirror the writer's layout).
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != QUADTREE_MAGIC { return Err(corrupt("bad QuadTree magic")); }
        if r_u8(r)? != QUADTREE_VERSION { return Err(corrupt("unsupported QuadTree version")); }
        let item_limit = r_u64(r)? as usize;
        let merge_limit = r_u64(r)? as usize;
        let min_cell = r_f64(r)?;
        let root = QNodeId(r_u32(r)?);
        let nfree = r_u32(r)? as usize;
        let mut free = Vec::with_capacity(nfree);
        for _ in 0..nfree { free.push(QNodeId(r_u32(r)?)); }
        let nnodes = r_u32(r)? as usize;
        let mut nodes = Vec::with_capacity(nnodes);
        for _ in 0..nnodes {
            let bbox = r_rect(r)?;
            let parent = if r_u8(r)? == 1 { Some(QNodeId(r_u32(r)?)) } else { None };
            let children = if r_u8(r)? == 1 {
                let mut kids = [QNodeId(0); 4];
                for k in &mut kids { *k = QNodeId(r_u32(r)?); }
                Some(kids)
            } else { None };
            let nitems = r_u32(r)? as usize;
            let mut items = Vec::with_capacity(nitems);
            for _ in 0..nitems { items.push(read_item(r)?); }
            let mut hs = Vec::with_capacity(nitems);
            for _ in 0..nitems { hs.push(r_u32(r)?); }
            nodes.push(QNode { bbox, parent, children, items, hs });
        }
        if root.0 as usize >= nnodes { return Err(corrupt("root index out of range")); }
        let max_h = nodes.iter().flat_map(|n| n.hs.iter().copied()).max().map_or(0, |m| m + 1) as usize;
        let mut locs = vec![QItemLoc { node: QNodeId(0), slot: 0 }; max_h];
        let mut used = vec![false; max_h];
        for (ni, n) in nodes.iter().enumerate() {
            for (slot, &h) in n.hs.iter().enumerate() {
                locs[h as usize] = QItemLoc { node: QNodeId(ni as u32), slot: slot as u32 };
                used[h as usize] = true;
            }
        }
        let free_handles: Vec<u32> = (0..max_h as u32).filter(|&h| !used[h as usize]).collect();
        Ok(QuadTree { nodes, free, locs, free_handles, item_limit, merge_limit, min_cell, root })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Pt(Point);
    impl Positioned for Pt {
        fn position(&self) -> Point {
            self.0
        }
    }

    struct Disc { cx: f64, cy: f64, r: f64 }
    impl Shape for Disc {
        fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
        fn contains_point(&self, p: Point) -> bool { let (dx, dy) = (p.x - self.cx, p.y - self.cy); dx * dx + dy * dy <= self.r * self.r }
    }

    #[test]
    fn quadtree_serialize_roundtrip() {
        use std::io::{Cursor, Read, Write};
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point }
        impl Positioned for M { fn position(&self) -> Point { self.p } }
        let mut x = 0x0DAB_5E12u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut tree = QuadTree::<M>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 5);
        let mut live: std::collections::BTreeMap<u32, Point> = std::collections::BTreeMap::new();
        let mut next = 0u32;
        for _ in 0..1500 {
            let p = Point::new(rng() * 256.0, rng() * 256.0);
            tree.insert(M { id: next, p }); live.insert(next, p); next += 1;
        }
        for _ in 0..3000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.5 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id]; let np = Point::new(rng() * 256.0, rng() * 256.0);
                if tree.update(old, |m| m.id == id, |m| m.p = np) { live.insert(id, np); }
            } else if roll < 0.7 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                tree.remove(live[&id], |m| m.id == id); live.remove(&id);
            } else {
                let p = Point::new(rng() * 256.0, rng() * 256.0);
                tree.insert(M { id: next, p }); live.insert(next, p); next += 1;
            }
        }
        let doomed: Vec<u32> = live.keys().copied().take(live.len() * 2 / 5).collect();
        for id in doomed { tree.remove(live[&id], |m| m.id == id); live.remove(&id); }
        assert!(tree.node_count() > tree.live_node_count(), "expected dead slots");

        let mut buf = Vec::new();
        tree.serialize(&mut buf, |w, it| {
            w.write_all(&it.id.to_le_bytes())?;
            w.write_all(&it.p.x.to_le_bytes())?; w.write_all(&it.p.y.to_le_bytes())
        }).unwrap();
        let loaded = QuadTree::<M>::deserialize(&mut Cursor::new(&buf), |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?; let mut b = [0u8; 8];
            let id = u32::from_le_bytes(a);
            r.read_exact(&mut b)?; let px = f64::from_le_bytes(b);
            r.read_exact(&mut b)?; let py = f64::from_le_bytes(b);
            Ok(M { id, p: Point::new(px, py) })
        }).unwrap();
        assert_eq!(loaded.node_count(), tree.node_count(), "arena size");
        assert_eq!(loaded.live_node_count(), tree.live_node_count(), "live nodes");
        assert_eq!(loaded.leaf_count(), tree.leaf_count(), "leaves");
        assert_eq!(loaded.item_count(), tree.item_count(), "items");
        for (cx, cy, r) in [(128.0, 128.0, 30.0), (60.0, 200.0, 50.0), (10.0, 10.0, 80.0)] {
            let s = Disc { cx, cy, r };
            let mut a: Vec<u32> = tree.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = loaded.cull(&s).iter().map(|m| m.id).collect();
            a.sort(); b.sort();
            assert_eq!(a, b, "cull differs after round-trip ({cx},{cy}) r={r}");
        }
        let ka: Vec<f64> = tree.knn(Point::new(120.0, 120.0), 8).iter().map(|(d, _)| *d).collect();
        let kb: Vec<f64> = loaded.knn(Point::new(120.0, 120.0), 8).iter().map(|(d, _)| *d).collect();
        assert_eq!(ka, kb, "knn differs after round-trip");
        assert!(QuadTree::<M>::deserialize(&mut Cursor::new(&b"XXXXX"[..]), |_| unreachable!()).is_err());
    }

    #[test]
    fn knn_matches_brute() {
        let mut x = 0x1357_9BDFu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let pts: Vec<Pt> = (0..2000).map(|_| Pt(Point::new(rng() * 256.0, rng() * 256.0))).collect();
        let mut tree = QuadTree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 4);
        for p in &pts { tree.insert(*p); }
        for (qx, qy) in [(128.0, 128.0), (10.0, 240.0), (0.0, 0.0), (255.0, 255.0)] {
            let q = Point::new(qx, qy);
            for k in [1usize, 5, 25] {
                let mut brute: Vec<f64> = pts.iter().map(|p| { let (dx, dy) = (p.0.x - qx, p.0.y - qy); (dx * dx + dy * dy).sqrt() }).collect();
                brute.sort_by(|a, b| a.total_cmp(b));
                brute.truncate(k);
                let got: Vec<f64> = tree.knn(q, k).into_iter().map(|(d, _)| d).collect();
                assert_eq!(got.len(), brute.len(), "knn count k={k}");
                for (a, b) in got.iter().zip(brute.iter()) {
                    assert!((a - b).abs() < 1e-9, "knn dist != brute k={k}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn update_ref_churn_matches_brute() {
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point }
        impl Positioned for M { fn position(&self) -> Point { self.p } }
        struct Box2 { r: Rect }
        impl crate::Shape for Box2 {
            fn bounding_box(&self) -> Rect { self.r }
            fn contains_point(&self, p: Point) -> bool { self.r.contains(p) }
        }
        let mut x = 0x9D7_EF00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut q = QuadTree::<M>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 6);
        let rp = |rng: &mut dyn FnMut() -> f64| Point::new(rng() * 256.0, rng() * 256.0);
        let mut live: std::collections::HashMap<u32, (ItemRef, Point)> = std::collections::HashMap::new();
        let mut next = 0u32;
        for _ in 0..2000 { let p = rp(&mut rng); let r = q.insert_ref(M { id: next, p }).unwrap(); live.insert(next, (r, p)); next += 1; }
        for _ in 0..6000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.6 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let (r, _) = live[&id]; let np = rp(&mut rng);
                assert!(q.update_ref(r, |m| m.p = np)); live.get_mut(&id).unwrap().1 = np;
            } else if roll < 0.8 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                q.remove_ref(live[&id].0); live.remove(&id);
            } else { let p = rp(&mut rng); let r = q.insert_ref(M { id: next, p }).unwrap(); live.insert(next, (r, p)); next += 1; }
        }
        for r in [Rect::new(100.0, 100.0, 60.0, 60.0), Rect::new(0.0, 0.0, 128.0, 128.0), Rect::new(200.0, 10.0, 50.0, 200.0)] {
            let mut want: Vec<u32> = live.iter().filter(|(_, (_, p))| r.contains(*p)).map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = q.cull(&Box2 { r }).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "quadtree handle-churn cull != brute for {r:?}");
        }
    }

    #[test]
    fn insert_remove_update_roundtrip() {
        let mut q = QuadTree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
        for &(x, y) in &[(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0), (50.0, 50.0)] {
            assert!(q.insert(Pt(Point::new(x, y))));
        }
        assert_eq!(q.item_count(), 5);
        assert!(q.get(q.root).children.is_some(), "overflow must divide");

        // Update relocates across leaves.
        assert!(q.update(
            Point::new(10.0, 10.0),
            |it| it.0.x == 10.0 && it.0.y == 10.0,
            |it| it.0 = Point::new(85.0, 85.0),
        ));
        assert_eq!(q.item_count(), 5);
        let landed = q.locate(Point::new(85.0, 85.0));
        assert!(q.get(landed).items.iter().any(|p| p.0 == Point::new(85.0, 85.0)));

        // Remove down to merge territory.
        for &(x, y) in &[(85.0, 85.0), (90.0, 10.0), (10.0, 90.0)] {
            assert!(q.remove(Point::new(x, y), |it| it.0 == Point::new(x, y)).is_some());
        }
        assert_eq!(q.item_count(), 2);
        assert!(
            q.get(q.root).children.is_none(),
            "4-way merge rule must collapse the root once children fit",
        );
    }

    #[test]
    fn duplicate_positions_do_not_split_forever() {
        let mut q = QuadTree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        for _ in 0..5 {
            assert!(q.insert(Pt(Point::new(10.0, 10.0))));
        }
        assert_eq!(q.item_count(), 5);
    }

    #[test]
    fn merge_limit_hysteresis() {
        let mut q = QuadTree::<Pt>::with_limits(Rect::new(0.0, 0.0, 100.0, 100.0), 2, 1);
        q.insert(Pt(Point::new(10.0, 10.0)));
        q.insert(Pt(Point::new(90.0, 10.0)));
        q.insert(Pt(Point::new(10.0, 90.0)));
        assert!(q.get(q.root).children.is_some());
        q.remove(Point::new(90.0, 10.0), |it| it.0.x == 90.0);
        // 2 left > merge_limit 1: stays split.
        assert!(q.get(q.root).children.is_some());
        q.remove(Point::new(10.0, 90.0), |it| it.0.y == 90.0);
        assert!(q.get(q.root).children.is_none());
    }

    /// QuadTree::cull must agree with Tree::cull on identical data.
    #[test]
    fn cull_agrees_with_binary_tree() {
        use crate::{Shape, Tree};
        struct Circle {
            c: Point,
            r: f64,
        }
        impl Shape for Circle {
            fn bounding_box(&self) -> Rect {
                Rect::new(self.c.x - self.r, self.c.y - self.r, self.r * 2.0, self.r * 2.0)
            }
            fn contains_point(&self, p: Point) -> bool {
                let dx = p.x - self.c.x;
                let dy = p.y - self.c.y;
                dx * dx + dy * dy <= self.r * self.r
            }
        }
        let mut x = 0x1234ABCDu64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 3);
        let mut quad = QuadTree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 3);
        for _ in 0..300 {
            let p = Pt(Point::new(next() * 256.0, next() * 256.0));
            tree.insert(p);
            quad.insert(p);
        }
        let shape = Circle { c: Point::new(128.0, 128.0), r: 70.0 };
        let mut a: Vec<_> = tree.cull(&shape).iter().map(|p| p.0).collect();
        let mut b: Vec<_> = quad.cull(&shape).iter().map(|p| p.0).collect();
        let key = |p: &Point| (p.x.to_bits(), p.y.to_bits());
        a.sort_by_key(key);
        b.sort_by_key(key);
        assert_eq!(a, b);
    }
}
