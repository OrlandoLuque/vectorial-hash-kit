//! Integer-coordinate binary-split tree — same algorithm as [`crate::Tree`],
//! but with `i32` coordinates and a power-of-two root extent. The point of
//! existing alongside the float tree is the **bit-shift question** from the
//! paper: when divisions are by powers of two, can we measure a real
//! advantage on insert / locate / update over float math?
//!
//! Concretely:
//! - `IPoint` / `IRect` are `i32`-based and roughly half the cache footprint
//!   of the float versions.
//! - Splits are at midpoints reached via `>> 1` rather than `/ 2.0`.
//! - The locate descent picks a child by comparing `point.x` / `point.y` to
//!   the parent's midpoint — one comparison instead of a child-bbox call.
//! - Construction asserts `root.width == root.height == 1 << k` for some
//!   `k`. Sub-power-of-two cases (e.g. 768 × 768) are out of scope.
//!
//! Cull reuses the existing [`crate::Shape`] trait by converting `IRect`
//! to `Rect` at the descent boundary and `IPoint` to `Point` at per-item
//! checks. The conversions are essentially free on modern CPUs; the
//! bit-shift advantage lives in tree storage and locate, not at the
//! shape boundary.
//!
//! For the rationale and benchmark results, see `docs/UPDATE_STRATEGIES.md`.

use crate::culling::{classify_child, SizeCache};
use crate::geom::{Point, Rect};
use crate::serde_io::{corrupt, r_i32, r_u32, r_u64, r_u8, w_i32, w_u32, w_u64, w_u8};
use crate::template::CellState;
use crate::tree3::ItemRef;
use crate::Shape;
use std::io::{self, Read, Write};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IPoint {
    pub x: i32,
    pub y: i32,
}

impl IPoint {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Squared distance (`f64`) from integer point `q` to the nearest point of
/// `r` — the integer analogue of `tree::rect_min_dist2`. Coords are widened to
/// `f64` before subtraction so the difference can't overflow `i32`.
#[inline]
fn irect_min_dist2(r: &IRect, q: IPoint) -> f64 {
    let dx = if q.x < r.x { (r.x as f64) - q.x as f64 } else if q.x > r.x_max() { (q.x as f64) - r.x_max() as f64 } else { 0.0 };
    let dy = if q.y < r.y { (r.y as f64) - q.y as f64 } else if q.y > r.y_max() { (q.y as f64) - r.y_max() as f64 } else { 0.0 };
    dx * dx + dy * dy
}

/// Offer one integer item to the bounded k-NN heap. Mirrors `tree3::knn_offer`.
#[inline]
fn iknn_offer<'a, T: IPositioned>(heap: &mut std::collections::BinaryHeap<crate::tree3::KnnEntry<'a, T>>, k: usize, it: &'a T, q: IPoint) {
    let p = it.position();
    let (dx, dy) = (p.x as f64 - q.x as f64, p.y as f64 - q.y as f64);
    let d2 = dx * dx + dy * dy;
    if heap.len() < k {
        heap.push(crate::tree3::KnnEntry { d2, item: it });
    } else if d2 < heap.peek().unwrap().d2 {
        heap.pop();
        heap.push(crate::tree3::KnnEntry { d2, item: it });
    }
}

impl IRect {
    pub fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }
    #[inline]
    pub fn x_max(&self) -> i32 { self.x + self.w }
    #[inline]
    pub fn y_max(&self) -> i32 { self.y + self.h }
    #[inline]
    pub fn contains(&self, p: IPoint) -> bool {
        p.x >= self.x && p.x < self.x_max() && p.y >= self.y && p.y < self.y_max()
    }
}

pub trait IPositioned {
    fn position(&self) -> IPoint;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct INodeId(pub u32);

pub struct INode<T> {
    pub bbox: IRect,
    pub parent: Option<INodeId>,
    pub children: Option<[INodeId; 2]>,
    pub items: Vec<T>,
    /// Parallel to `items`: each item's stable handle (the [`crate::ItemRef`]
    /// layer).
    hs: Vec<u32>,
}

/// Where a handle's item currently lives (leaf node + slot).
#[derive(Copy, Clone)]
struct IItemLoc { node: INodeId, slot: u32 }

/// Same relocation strategies as [`crate::UpdateStrategy`]; `LcaRopes`
/// behaves like `Lca` here (no rope bookkeeping in this tree).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum IUpdateStrategy {
    Legacy,
    #[default]
    Lca,
}

pub struct IntegerTree<T: IPositioned> {
    nodes: Vec<INode<T>>,
    /// Slots freed by merge-ups, reused before the arena grows — see
    /// [`crate::Tree`]'s free-list for the rationale.
    free: Vec<INodeId>,
    locs: Vec<IItemLoc>,
    free_handles: Vec<u32>,
    pub item_limit: usize,
    pub merge_limit: usize,
    /// Smallest non-degenerate cell dimension. With integer coords this is 1.
    min_cell: i32,
    pub root: INodeId,
}

impl<T: IPositioned> IntegerTree<T> {
    /// Every item within `radius` of the ray, sorted by distance along it.
    ///
    /// A capsule cull plus a sort — the same shape as [`crate::LinearQuadTree::raycast`], and
    /// deliberately not a DDA walk. The binary `Tree` can walk neighbour-to-neighbour because it
    /// carries ropes (feature `neighbors`); a quadtree has none, and a capsule cull already
    /// prunes by the segment's own bound at every node.
    ///
    /// `t` is the distance along the ray of each item's closest approach, clamped to
    /// `[0, max_dist]`, so an item beside the origin reports 0 rather than a negative.
    pub fn raycast(&self, origin: Point, dir: Point, max_dist: f64, radius: f64) -> Vec<(f64, &T)> {
        let m = (dir.x * dir.x + dir.y * dir.y).sqrt();
        if m == 0.0 { return Vec::new(); }
        let (ux, uy) = (dir.x / m, dir.y / m);
        let end = Point::new(origin.x + ux * max_dist, origin.y + uy * max_dist);
        let mut hits: Vec<(f64, &T)> = self.cull(&crate::Capsule::new(origin, end, radius)).into_iter().map(|it| {
            let p = { let ip = it.position(); Point::new(ip.x as f64, ip.y as f64) };
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

    /// Read an item through its stable [`ItemRef`] — `None` if the handle has been retired by
    /// `remove_ref`.
    ///
    /// The handle layer could **move** an item and could **delete** it, but not look at it, so
    /// any caller wanting to read one had to keep a parallel copy or abuse `update_ref`'s
    /// mutator to smuggle a value out. O(1), no descent and no scan: the handle *is* the dense
    /// index into the location table.
    pub fn get_ref(&self, r: ItemRef) -> Option<&T> {
        let loc = self.live_loc(r)?;
        self.get(loc.node).items.get(loc.slot as usize)
    }

    /// `bbox.w` and `bbox.h` must each be `1 << k` for some `k ≥ 1`.
    pub fn new(bbox: IRect, item_limit: usize) -> Self {
        Self::with_limits(bbox, item_limit, item_limit)
    }

    pub fn with_limits(bbox: IRect, item_limit: usize, merge_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        assert!(merge_limit <= item_limit, "merge_limit must be <= item_limit");
        assert!(bbox.w > 0 && bbox.h > 0, "bbox extent must be positive");
        assert!((bbox.w as u32).is_power_of_two(), "bbox.w must be a power of two");
        assert!((bbox.h as u32).is_power_of_two(), "bbox.h must be a power of two");
        let root = INode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() };
        Self {
            nodes: vec![root],
            free: Vec::new(),
            locs: Vec::new(),
            free_handles: Vec::new(),
            item_limit,
            merge_limit,
            min_cell: 1,
            root: INodeId(0),
        }
    }

    /// Empty the tree, retaining capacity — see [`crate::Tree3::clear`].
    /// Build a tree from all the items at once: drop them into the root, then split **once**,
    /// instead of descending the tree N times. The integer twin of [`crate::QuadTree::bulk_load`].
    ///
    /// Items outside `bbox` are dropped, matching [`IntegerTree::insert`]'s contract.
    pub fn bulk_load(bbox: IRect, item_limit: usize, items: Vec<T>) -> Self {
        let mut t = Self::new(bbox, item_limit);
        let root = t.root;
        for item in items {
            if !t.get(root).bbox.contains(item.position()) { continue; }
            let h = t.alloc_handle();
            t.push_h(root, item, h);
        }
        if t.get(root).items.len() > item_limit { t.divide(root); }
        t
    }

    /// [`IntegerTree::bulk_load`] with the recursion fanned out over rayon — the integer twin of
    /// [`crate::Tree::bulk_load_par`]. Needs the `parallel` feature and `T: Send`.
    ///
    /// It produces the **same partition** as the serial build — identical node and leaf counts,
    /// every item in the same leaf box — because the split rule is a pure function of a node's own
    /// items: long axis for rectangles, better-balanced axis for squares, always at the half
    /// boundary (exact, thanks to the pow2 invariant). It does **not** produce the same arena
    /// order: [`IntegerTree::bulk_load`] goes through `divide`, which allocates both children
    /// before recursing, while this flattens depth-first. So node ids differ and a byte-level
    /// comparison would fail for a reason no caller can observe. `tests/filled_capabilities.rs`
    /// asserts the property that is real.
    ///
    /// Measured **1.8-2.9x** on 16 threads (`examples/itree_bulk_load`, 10k-500k, uniform and
    /// clustered), best on clustered data where the chosen split axis has something to choose.
    /// That is well short of linear scaling: every node allocates, and partitioning items into
    /// fresh vectors at each level is memory traffic, which threads share rather than divide.
    #[cfg(feature = "parallel")]
    pub fn bulk_load_par(bbox: IRect, item_limit: usize, items: Vec<T>) -> Self
    where T: Send {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        // Drop out-of-bounds items FIRST, then number what survives: handles index `locs`
        // densely, and numbering before the filter would leave holes that no handle points at.
        let kept: Vec<(u32, T)> = items.into_iter().filter(|it| bbox.contains(it.position()))
            .enumerate().map(|(i, it)| (i as u32, it)).collect();
        let n = kept.len();
        let build = ibuild_par(item_limit, bbox, kept);
        let mut nodes: Vec<INode<T>> = Vec::new();
        let mut locs = vec![IItemLoc { node: INodeId(0), slot: 0 }; n];
        iflatten(&mut nodes, &mut locs, bbox, None, build);
        IntegerTree { nodes, free: Vec::new(), locs, free_handles: Vec::new(),
                      item_limit, merge_limit: item_limit, min_cell: 1, root: INodeId(0) }
    }

    pub fn clear(&mut self) {
        let bbox = self.get(self.root).bbox;
        self.nodes.clear();
        self.nodes.push(INode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() });
        self.free.clear();
        self.locs.clear();
        self.free_handles.clear();
        self.root = INodeId(0);
    }

    pub fn get(&self, id: INodeId) -> &INode<T> { &self.nodes[id.0 as usize] }
    fn get_mut(&mut self, id: INodeId) -> &mut INode<T> { &mut self.nodes[id.0 as usize] }

    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() { h }
        else { let h = self.locs.len() as u32; self.locs.push(IItemLoc { node: INodeId(0), slot: 0 }); h }
    }
    /// Retire a handle: mark its location [`crate::tree::DEAD_HANDLE`] before recycling
    /// the id, so a stale `ItemRef` can't alias whatever item later lands in that slot.
    fn free_handle(&mut self, h: u32) {
        self.locs[h as usize] = IItemLoc { node: INodeId(crate::tree::DEAD_HANDLE), slot: 0 };
        self.free_handles.push(h);
    }
    /// The live location behind a handle, or `None` if it was freed (item removed or
    /// dropped out of the root) or never belonged to this tree.
    fn live_loc(&self, r: ItemRef) -> Option<IItemLoc> {
        let loc = *self.locs.get(r.0 as usize)?;
        (loc.node.0 != crate::tree::DEAD_HANDLE).then_some(loc)
    }
    fn push_h(&mut self, node: INodeId, item: T, h: u32) {
        let slot = self.get(node).items.len() as u32;
        let n = self.get_mut(node);
        n.items.push(item);
        n.hs.push(h);
        self.locs[h as usize] = IItemLoc { node, slot };
    }
    fn swap_remove_h(&mut self, node: INodeId, slot: usize) -> (T, u32) {
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

    fn alloc(&mut self, node: INode<T>) -> INodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0 as usize] = node;
            id
        } else {
            let id = INodeId(self.nodes.len() as u32);
            self.nodes.push(node);
            id
        }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn live_node_count(&self) -> usize { self.nodes.len() - self.free.len() }

    /// Reorder the node arena into DFS pre-order and drop freed slots — the
    /// [`Tree3::compact`](crate::Tree3::compact) cache-locality pass for the
    /// integer 2D tree. Pure layout: shape, items, bboxes, `ItemRef` handles and
    /// every query result are unchanged; only the internal `INodeId`s move
    /// (handles remapped, raw `INodeId`s not). O(live nodes), one pass.
    pub fn compact(&mut self) {
        let mut old2new = vec![u32::MAX; self.nodes.len()];
        let mut order: Vec<INodeId> = Vec::with_capacity(self.live_node_count());
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            old2new[id.0 as usize] = order.len() as u32;
            order.push(id);
            if let Some([a, b]) = self.get(id).children { stack.push(b); stack.push(a); }
        }
        let remap = |id: INodeId| INodeId(old2new[id.0 as usize]);
        let mut new_nodes: Vec<INode<T>> = Vec::with_capacity(order.len());
        for &old in &order {
            let bbox = self.nodes[old.0 as usize].bbox;
            let mut node = std::mem::replace(&mut self.nodes[old.0 as usize],
                INode { bbox, parent: None, children: None, items: Vec::new(), hs: Vec::new() });
            node.parent = node.parent.map(remap);
            node.children = node.children.map(|[a, b]| [remap(a), remap(b)]);
            new_nodes.push(node);
        }
        for loc in self.locs.iter_mut() {
            if loc.node.0 == crate::tree::DEAD_HANDLE { continue; } // freed handle: no live node to remap
            let nn = old2new[loc.node.0 as usize];
            if nn != u32::MAX { loc.node = INodeId(nn); }
        }
        self.root = remap(self.root);
        self.nodes = new_nodes;
        self.free.clear();
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

    pub fn visit_leaves<F: FnMut(INodeId, &INode<T>)>(&self, mut f: F) {
        self.visit_leaves_from(self.root, &mut f);
    }
    fn visit_leaves_from<F: FnMut(INodeId, &INode<T>)>(&self, id: INodeId, f: &mut F) {
        match self.get(id).children {
            Some([a, b]) => {
                self.visit_leaves_from(a, f);
                self.visit_leaves_from(b, f);
            }
            None => f(id, self.get(id)),
        }
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.insert_ref(item).is_some()
    }

    /// Insert and return a stable [`crate::ItemRef`] for O(1) `update_ref`/`remove_ref`.
    pub fn insert_ref(&mut self, item: T) -> Option<ItemRef> {
        let pos = item.position();
        if !self.get(self.root).bbox.contains(pos) { return None; }
        let leaf = self.locate(pos);
        let h = self.alloc_handle();
        self.push_h(leaf, item, h);
        if self.get(leaf).items.len() > self.item_limit { self.divide(leaf); }
        Some(ItemRef(h))
    }

    /// O(1) relocation through a stable [`crate::ItemRef`] (no locate, no scan).
    pub fn update_ref<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> bool {
        let Some(loc) = self.live_loc(r) else { return false }; // stale handle: item already gone
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) { return true; }
        self.relocate(node, slot, np)
    }

    /// Remove the item behind a stable [`crate::ItemRef`] in O(1).
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.live_loc(r)?; // stale handle: already removed
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handle(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    /// Shared LCA relocate tail (predicate `update` and `update_ref`).
    fn relocate(&mut self, leaf: INodeId, slot: usize, new_pos: IPoint) -> bool {
        let lca = match self.ascend_to_lca(leaf, new_pos) {
            Some(id) => id,
            None => {
                let (_, h) = self.swap_remove_h(leaf, slot);
                self.free_handle(h);
                self.try_merge_up(leaf);
                return false;
            }
        };
        let (item, h) = self.swap_remove_h(leaf, slot);
        let dest = self.locate_from(lca, new_pos);
        self.push_h(dest, item, h);
        if self.get(dest).items.len() > self.item_limit { self.divide(dest); }
        self.try_merge_up(leaf);
        true
    }

    pub fn locate(&self, point: IPoint) -> INodeId {
        self.locate_from(self.root, point)
    }

    /// Same descent shape as [`crate::Tree::locate_from`]: read child A's
    /// bbox, ask if it contains the point, descend accordingly. Integer
    /// `contains` short-circuits on the first failing comparison just like
    /// the float version; the actual `>> 1` bit-shift wins live in `divide`,
    /// not here.
    pub fn locate_from(&self, start: INodeId, point: IPoint) -> INodeId {
        let mut current = start;
        loop {
            match self.get(current).children {
                None => return current,
                Some([a, b]) => {
                    current = if self.get(a).bbox.contains(point) { a } else { b };
                }
            }
        }
    }

    fn ascend_to_lca(&self, leaf: INodeId, point: IPoint) -> Option<INodeId> {
        let mut node = self.get(leaf).parent?;
        loop {
            if self.get(node).bbox.contains(point) { return Some(node); }
            node = self.get(node).parent?;
        }
    }

    pub fn remove<F: Fn(&T) -> bool>(&mut self, point: IPoint, predicate: F) -> Option<T> {
        if !self.get(self.root).bbox.contains(point) { return None; }
        let leaf = self.locate(point);
        let idx = self.get(leaf).items.iter().position(&predicate)?;
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.free_handle(h);
        self.try_merge_up(leaf);
        Some(item)
    }

    pub fn update<F, M>(&mut self, old_position: IPoint, predicate: F, mutator: M) -> bool
    where F: Fn(&T) -> bool, M: FnOnce(&mut T) {
        self.update_with(IUpdateStrategy::default(), old_position, predicate, mutator)
    }

    pub fn update_with<F, M>(
        &mut self,
        strategy: IUpdateStrategy,
        old_position: IPoint,
        predicate: F,
        mutator: M,
    ) -> bool
    where F: Fn(&T) -> bool, M: FnOnce(&mut T) {
        if !self.get(self.root).bbox.contains(old_position) { return false; }
        let leaf = self.locate(old_position);
        let idx = match self.get(leaf).items.iter().position(&predicate) {
            Some(i) => i,
            None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);

        let new_pos = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(new_pos) { return true; }

        match strategy {
            IUpdateStrategy::Legacy => {
                // remove + re-insert reassigns the handle; the Lca path preserves it.
                let (item, h) = self.swap_remove_h(leaf, idx);
                self.free_handle(h);
                self.try_merge_up(leaf);
                self.insert(item)
            }
            IUpdateStrategy::Lca => self.relocate(leaf, idx, new_pos),
        }
    }

    // ----- cull -----

    /// Return references to every item inside `shape`. Mirrors
    /// [`crate::Tree::cull`] semantically; internally converts `IRect` ↔
    /// `Rect` and `IPoint` ↔ `Point` at the shape boundary so the existing
    /// float-based [`Shape`] machinery (templates, raster, contains_point)
    /// works unchanged.
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

    /// The `k` nearest items to integer point `q`, sorted ascending by distance
    /// (distances are `f64`) — the integer analogue of [`crate::Tree::knn`].
    pub fn knn(&self, q: IPoint, k: usize) -> Vec<(f64, &T)> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap = std::collections::BinaryHeap::new();
        self.knn_recurse(self.root, q, k, &mut heap);
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    fn knn_recurse<'a>(&'a self, id: INodeId, q: IPoint, k: usize, heap: &mut std::collections::BinaryHeap<crate::tree3::KnnEntry<'a, T>>) {
        let node = self.get(id);
        match node.children {
            None => {
                for it in &node.items {
                    iknn_offer(heap, k, it, q);
                }
            }
            Some([a, b]) => {
                let da = irect_min_dist2(&self.get(a).bbox, q);
                let db = irect_min_dist2(&self.get(b).bbox, q);
                let (first, dfirst, second, dsecond) = if da <= db { (a, da, b, db) } else { (b, db, a, da) };
                if dfirst < crate::tree3::knn_worst(heap, k) {
                    self.knn_recurse(first, q, k, heap);
                }
                if dsecond < crate::tree3::knn_worst(heap, k) {
                    self.knn_recurse(second, q, k, heap);
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
        node_id: INodeId,
        shape: &S,
        shape_bbox: &Rect,
        fully_inside: bool,
        sizes: &mut SizeCache,
        out: &mut Vec<&'a T>,
    ) {
        let node = self.get(node_id);
        if fully_inside {
            match node.children {
                Some([a, b]) => {
                    self.cull_recurse(a, shape, shape_bbox, true, sizes, out);
                    self.cull_recurse(b, shape, shape_bbox, true, sizes, out);
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }
        match node.children {
            Some([a, b]) => {
                for child_id in [a, b] {
                    let cb = irect_to_rect(self.get(child_id).bbox);
                    match classify_child(shape, shape_bbox, &cb, sizes) {
                        CellState::Out => {}
                        CellState::In => {
                            self.cull_recurse(child_id, shape, shape_bbox, true, sizes, out);
                        }
                        CellState::Maybe => {
                            self.cull_recurse(child_id, shape, shape_bbox, false, sizes, out);
                        }
                    }
                }
            }
            None => collect_matching_items_i(&node.items, shape, shape_bbox, out),
        }
    }

    fn try_merge_up(&mut self, mut node: INodeId) {
        loop {
            let parent_id = match self.get(node).parent {
                Some(p) => p,
                None => return,
            };
            let [a, b] = self.get(parent_id).children.expect("parent has children");
            if self.get(a).children.is_some() || self.get(b).children.is_some() { return; }
            let combined = self.get(a).items.len() + self.get(b).items.len();
            if combined > self.merge_limit { return; }
            let mut items_a = std::mem::take(&mut self.get_mut(a).items);
            let mut hs_a = std::mem::take(&mut self.get_mut(a).hs);
            let mut items_b = std::mem::take(&mut self.get_mut(b).items);
            let mut hs_b = std::mem::take(&mut self.get_mut(b).hs);
            items_a.append(&mut items_b);
            hs_a.append(&mut hs_b);
            let parent = self.get_mut(parent_id);
            parent.items = items_a;
            parent.hs = hs_a;
            parent.children = None;
            let len = self.get(parent_id).hs.len();
            for slot in 0..len {
                let h = self.get(parent_id).hs[slot];
                self.locs[h as usize] = IItemLoc { node: parent_id, slot: slot as u32 };
            }
            self.free.push(a);
            self.free.push(b);
            node = parent_id;
        }
    }

    fn divide(&mut self, id: INodeId) {
        let (bbox, items, hs) = {
            let n = self.get_mut(id);
            let items = std::mem::take(&mut n.items);
            let hs = std::mem::take(&mut n.hs);
            (n.bbox, items, hs)
        };
        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.w.max(bbox.h) <= self.min_cell {
            let n = self.get_mut(id);
            n.items = items;
            n.hs = hs;
            return;
        }
        // Split policy mirrors the float tree's `pick_split`:
        // - rectangles split along the long axis,
        // - squares pick whichever axis distributes the items more evenly.
        // Always at the half boundary; exact thanks to the pow2 invariant.
        let (a_bbox, b_bbox) = if bbox.w > bbox.h {
            let half = bbox.w >> 1;
            (IRect::new(bbox.x, bbox.y, half, bbox.h),
             IRect::new(bbox.x + half, bbox.y, half, bbox.h))
        } else if bbox.h > bbox.w {
            let half = bbox.h >> 1;
            (IRect::new(bbox.x, bbox.y, bbox.w, half),
             IRect::new(bbox.x, bbox.y + half, bbox.w, half))
        } else {
            let mid_x = bbox.x + (bbox.w >> 1);
            let mid_y = bbox.y + (bbox.h >> 1);
            let left = items.iter().filter(|it| it.position().x < mid_x).count();
            let top  = items.iter().filter(|it| it.position().y < mid_y).count();
            let n = items.len() as i64;
            let vert = (2 * left as i64 - n).abs();
            let horz = (2 * top as i64 - n).abs();
            if vert <= horz {
                let half = bbox.w >> 1;
                (IRect::new(bbox.x, bbox.y, half, bbox.h),
                 IRect::new(bbox.x + half, bbox.y, half, bbox.h))
            } else {
                let half = bbox.h >> 1;
                (IRect::new(bbox.x, bbox.y, bbox.w, half),
                 IRect::new(bbox.x, bbox.y + half, bbox.w, half))
            }
        };
        let a = self.alloc(INode { bbox: a_bbox, parent: Some(id), children: None, items: Vec::new(), hs: Vec::new() });
        let b = self.alloc(INode { bbox: b_bbox, parent: Some(id), children: None, items: Vec::new(), hs: Vec::new() });
        for (item, h) in items.into_iter().zip(hs) {
            let pos = item.position();
            let dest = if self.get(a).bbox.contains(pos) { a } else { b };
            self.push_h(dest, item, h);
        }
        self.get_mut(id).children = Some([a, b]);
        if self.get(a).items.len() > self.item_limit { self.divide(a); }
        if self.get(b).items.len() > self.item_limit { self.divide(b); }
    }
}

// ---------------------------------------------------------------- parallel bulk build
//
// `bulk_load` fills the root and calls `divide`, which recurses in place. That cannot fan out —
// every level borrows the arena mutably. So the parallel path builds the subtree structure
// OFF-arena (no shared state, so rayon can split it), and flattens into the arena serially
// afterwards, which is O(nodes) pointer work.

/// Mirrors `divide`'s refusal to split: overflow, a floor on cell size, and the inseparable case
/// where every item sits on the same point and no boundary could ever separate them.
#[cfg(feature = "parallel")]
fn isplittable<T: IPositioned>(items: &[(u32, T)], item_limit: usize, bbox: IRect) -> bool {
    items.len() > item_limit
        && bbox.w.max(bbox.h) > 1
        && { let first = items[0].1.position(); !items.iter().all(|(_, it)| it.position() == first) }
}

/// `divide`'s split policy, lifted out so both paths cannot drift apart.
#[cfg(feature = "parallel")]
fn ipick_split<T: IPositioned>(bbox: IRect, items: &[(u32, T)]) -> (IRect, IRect) {
    if bbox.w > bbox.h {
        let half = bbox.w >> 1;
        (IRect::new(bbox.x, bbox.y, half, bbox.h), IRect::new(bbox.x + half, bbox.y, half, bbox.h))
    } else if bbox.h > bbox.w {
        let half = bbox.h >> 1;
        (IRect::new(bbox.x, bbox.y, bbox.w, half), IRect::new(bbox.x, bbox.y + half, bbox.w, half))
    } else {
        let mid_x = bbox.x + (bbox.w >> 1);
        let mid_y = bbox.y + (bbox.h >> 1);
        let left = items.iter().filter(|(_, it)| it.position().x < mid_x).count();
        let top = items.iter().filter(|(_, it)| it.position().y < mid_y).count();
        let n = items.len() as i64;
        if (2 * left as i64 - n).abs() <= (2 * top as i64 - n).abs() {
            let half = bbox.w >> 1;
            (IRect::new(bbox.x, bbox.y, half, bbox.h), IRect::new(bbox.x + half, bbox.y, half, bbox.h))
        } else {
            let half = bbox.h >> 1;
            (IRect::new(bbox.x, bbox.y, bbox.w, half), IRect::new(bbox.x, bbox.y + half, bbox.w, half))
        }
    }
}

/// A subtree built off-arena. The split rects ride along rather than being recomputed on flatten:
/// the square case depends on how the items happened to balance, so it is not a function of the
/// box alone.
#[cfg(feature = "parallel")]
enum IBuild<T> { Leaf(Vec<(u32, T)>), Split(IRect, IRect, Box<IBuild<T>>, Box<IBuild<T>>) }

#[cfg(feature = "parallel")]
fn ibuild_par<T: IPositioned + Send>(item_limit: usize, bbox: IRect, items: Vec<(u32, T)>) -> IBuild<T> {
    if !isplittable(&items, item_limit, bbox) { return IBuild::Leaf(items); }
    let (ab, bb) = ipick_split(bbox, &items);
    let (mut ai, mut bi) = (Vec::new(), Vec::new());
    for (h, it) in items { if ab.contains(it.position()) { ai.push((h, it)); } else { bi.push((h, it)); } }
    let (a, b) = rayon::join(|| ibuild_par(item_limit, ab, ai), || ibuild_par(item_limit, bb, bi));
    IBuild::Split(ab, bb, Box::new(a), Box::new(b))
}

/// Serial DFS flatten into the arena, recording each handle's leaf and slot as it goes.
#[cfg(feature = "parallel")]
fn iflatten<T: IPositioned>(nodes: &mut Vec<INode<T>>, locs: &mut [IItemLoc], bbox: IRect, parent: Option<INodeId>, build: IBuild<T>) -> INodeId {
    let id = INodeId(nodes.len() as u32);
    nodes.push(INode { bbox, parent, children: None, items: Vec::new(), hs: Vec::new() });
    match build {
        IBuild::Leaf(items) => {
            let node = &mut nodes[id.0 as usize];
            for (h, it) in items {
                let slot = node.items.len() as u32;
                locs[h as usize] = IItemLoc { node: id, slot };
                node.items.push(it); node.hs.push(h);
            }
        }
        IBuild::Split(ab, bb, a, b) => {
            let ca = iflatten(nodes, locs, ab, Some(id), *a);
            let cb = iflatten(nodes, locs, bb, Some(id), *b);
            nodes[id.0 as usize].children = Some([ca, cb]);
        }
    }
    id
}

#[inline]
fn irect_to_rect(r: IRect) -> Rect {
    Rect::new(r.x as f64, r.y as f64, r.w as f64, r.h as f64)
}

/// Integer-side analogue of `culling::collect_matching_items`: bbox
/// prefilter (closed bounds), then the 1×1 raster when available, exact
/// geometry only on boundary pixels. Converts `IPoint` to `Point` at the
/// per-item check.
fn collect_matching_items_i<'a, T: IPositioned, S: Shape>(
    items: &'a [T],
    shape: &S,
    shape_bbox: &Rect,
    out: &mut Vec<&'a T>,
) {
    let point_grid = shape.point_template();
    for it in items {
        let ip = it.position();
        let p = Point::new(ip.x as f64, ip.y as f64);
        if !shape_bbox.contains_closed(p) {
            continue;
        }
        match point_grid.map(|g| g.cell_at_world(p)) {
            Some(CellState::In) => out.push(it),
            Some(CellState::Out) => {}
            _ => {
                if shape.contains_point(p) {
                    out.push(it);
                }
            }
        }
    }
}

// ------------------------------------------------------------- serialization

const ITREE_MAGIC: &[u8; 4] = b"VHI2";
const ITREE_VERSION: u8 = 1;

impl<T: IPositioned> IntegerTree<T> {
    /// Serialize the **built** integer tree (exact arena, free-list, params — no
    /// rebuild on load) to `w`. Items are written by `write_item`. Mirrors
    /// [`crate::Tree3::serialize`] over the i32 binary split.
    pub fn serialize<W: Write>(&self, w: &mut W, write_item: impl Fn(&mut W, &T) -> io::Result<()>) -> io::Result<()> {
        w.write_all(ITREE_MAGIC)?;
        w_u8(w, ITREE_VERSION)?;
        w_u64(w, self.item_limit as u64)?;
        w_u64(w, self.merge_limit as u64)?;
        w_i32(w, self.min_cell)?;
        w_u32(w, self.root.0)?;
        w_u32(w, self.free.len() as u32)?;
        for f in &self.free { w_u32(w, f.0)?; }
        w_u32(w, self.nodes.len() as u32)?;
        for n in &self.nodes {
            w_i32(w, n.bbox.x)?; w_i32(w, n.bbox.y)?; w_i32(w, n.bbox.w)?; w_i32(w, n.bbox.h)?;
            match n.parent {
                Some(p) => { w_u8(w, 1)?; w_u32(w, p.0)?; }
                None => w_u8(w, 0)?,
            }
            match n.children {
                Some([a, b]) => { w_u8(w, 1)?; w_u32(w, a.0)?; w_u32(w, b.0)?; }
                None => w_u8(w, 0)?,
            }
            w_u32(w, n.items.len() as u32)?;
            for it in &n.items { write_item(w, it)?; }
            for &h in &n.hs { w_u32(w, h)?; }
        }
        Ok(())
    }

    /// Inverse of [`IntegerTree::serialize`]: rebuild the exact tree from `r`,
    /// reading each item with `read_item` (must mirror the writer's layout).
    pub fn deserialize<R: Read>(r: &mut R, read_item: impl Fn(&mut R) -> io::Result<T>) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != ITREE_MAGIC { return Err(corrupt("bad IntegerTree magic")); }
        if r_u8(r)? != ITREE_VERSION { return Err(corrupt("unsupported IntegerTree version")); }
        let item_limit = r_u64(r)? as usize;
        let merge_limit = r_u64(r)? as usize;
        let min_cell = r_i32(r)?;
        let root = INodeId(r_u32(r)?);
        let nfree = r_u32(r)? as usize;
        let mut free = Vec::with_capacity(nfree);
        for _ in 0..nfree { free.push(INodeId(r_u32(r)?)); }
        let nnodes = r_u32(r)? as usize;
        let mut nodes = Vec::with_capacity(nnodes);
        for _ in 0..nnodes {
            let bbox = IRect::new(r_i32(r)?, r_i32(r)?, r_i32(r)?, r_i32(r)?);
            let parent = if r_u8(r)? == 1 { Some(INodeId(r_u32(r)?)) } else { None };
            let children = if r_u8(r)? == 1 { Some([INodeId(r_u32(r)?), INodeId(r_u32(r)?)]) } else { None };
            let nitems = r_u32(r)? as usize;
            let mut items = Vec::with_capacity(nitems);
            for _ in 0..nitems { items.push(read_item(r)?); }
            let mut hs = Vec::with_capacity(nitems);
            for _ in 0..nitems { hs.push(r_u32(r)?); }
            nodes.push(INode { bbox, parent, children, items, hs });
        }
        if root.0 as usize >= nnodes { return Err(corrupt("root index out of range")); }
        let max_h = nodes.iter().flat_map(|n| n.hs.iter().copied()).max().map_or(0, |m| m + 1) as usize;
        let mut locs = vec![IItemLoc { node: INodeId(0), slot: 0 }; max_h];
        let mut used = vec![false; max_h];
        for (ni, n) in nodes.iter().enumerate() {
            for (slot, &h) in n.hs.iter().enumerate() {
                locs[h as usize] = IItemLoc { node: INodeId(ni as u32), slot: slot as u32 };
                used[h as usize] = true;
            }
        }
        let free_handles: Vec<u32> = (0..max_h as u32).filter(|&h| !used[h as usize]).collect();
        Ok(IntegerTree { nodes, free, locs, free_handles, item_limit, merge_limit, min_cell, root })
    }
}


impl<T: IPositioned> IntegerTree<T> {
    /// Batch k-NN — one result list per query point (`out[i]` for `queries[i]`). Serial; see
    /// [`Self::knn_many_par`].
    pub fn knn_many(&self, queries: &[IPoint], k: usize) -> Vec<Vec<(f64, &T)>> {
        queries.iter().map(|&q| self.knn(q, k)).collect()
    }

    /// Parallel batch k-NN (feature `parallel`) — the independent queries fan out over rayon.
    #[cfg(feature = "parallel")]
    pub fn knn_many_par(&self, queries: &[IPoint], k: usize) -> Vec<Vec<(f64, &T)>>
    where T: Sync {
        use rayon::prelude::*;
        queries.par_iter().map(|&q| self.knn(q, k)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct IP(IPoint);
    impl IPositioned for IP { fn position(&self) -> IPoint { self.0 } }

    struct Disc { cx: f64, cy: f64, r: f64 }
    impl Shape for Disc {
        fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
        fn contains_point(&self, p: Point) -> bool { let (dx, dy) = (p.x - self.cx, p.y - self.cy); dx * dx + dy * dy <= self.r * self.r }
    }

    #[test]
    fn compact_preserves_queries_and_handles() {
        // IntegerTree twin of the Tree3 compact test: churn scrambles the arena,
        // then compact() must change no cull/knn result, reclaim free slots, and
        // keep every live handle resolving to its own item.
        let mut x = 0x1D7C_0DE9u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let ri = |r: &mut dyn FnMut() -> f64| (r() * 256.0) as i32;
        let mut tree = IntegerTree::<IP>::new(IRect::new(0, 0, 256, 256), 8);
        let mut refs: Vec<Option<ItemRef>> = Vec::new();
        let mut expected: Vec<Option<IPoint>> = Vec::new();
        for _ in 0..3000 {
            let p = IPoint::new(ri(&mut rng), ri(&mut rng));
            refs.push(tree.insert_ref(IP(p)));
            expected.push(Some(p));
        }
        for i in 0..3000 {
            if i % 3 == 0 { if let Some(r) = refs[i] {
                let np = IPoint::new(ri(&mut rng), ri(&mut rng));
                tree.update_ref(r, |q| q.0 = np); expected[i] = Some(np);
            } }
            if i % 5 == 0 { if let Some(r) = refs[i].take() { tree.remove_ref(r); expected[i] = None; } }
        }
        assert!(tree.node_count() > tree.live_node_count(), "churn should leave free slots to reclaim");
        let discs = [(128.0,128.0,40.0),(40.0,40.0,60.0),(250.0,250.0,30.0),(0.0,0.0,100.0)];
        let snap_cull = |t: &IntegerTree<IP>| -> Vec<Vec<(i32,i32)>> {
            discs.iter().map(|&(cx,cy,r)| {
                let mut v: Vec<(i32,i32)> = t.cull(&Disc{cx,cy,r}).iter().map(|p| (p.0.x, p.0.y)).collect();
                v.sort(); v
            }).collect()
        };
        let queries = [IPoint::new(50,60), IPoint::new(200,10), IPoint::new(128,128)];
        let snap_knn = |t: &IntegerTree<IP>| -> Vec<Vec<u64>> {
            queries.iter().map(|&q| { let mut d: Vec<u64> = t.knn(q, 12).iter().map(|(dist,_)| dist.to_bits()).collect(); d.sort(); d }).collect()
        };
        let cull_before = snap_cull(&tree);
        let knn_before = snap_knn(&tree);
        tree.compact();
        assert_eq!(tree.node_count(), tree.live_node_count(), "compact must reclaim every free slot");
        assert_eq!(cull_before, snap_cull(&tree), "compact changed a cull result");
        assert_eq!(knn_before, snap_knn(&tree), "compact changed a knn result");
        for i in 0..3000 {
            if let Some(r) = refs[i] {
                let got = tree.remove_ref(r).expect("live handle must resolve after compact").0;
                let want = expected[i].unwrap();
                assert_eq!((got.x, got.y), (want.x, want.y), "handle {i} resolved to the wrong item after compact");
            }
        }
    }

    #[test]
    fn itree_serialize_roundtrip() {
        use std::io::{Cursor, Read, Write};
        #[derive(Clone, Copy)]
        struct M { id: u32, p: IPoint }
        impl IPositioned for M { fn position(&self) -> IPoint { self.p } }
        let mut x = 0x01EE_5E12u64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let rint = |r: &mut dyn FnMut() -> f64| (r() * 256.0) as i32;
        let mut tree = IntegerTree::<M>::new(IRect::new(0, 0, 256, 256), 5);
        let mut live: std::collections::BTreeMap<u32, IPoint> = std::collections::BTreeMap::new();
        let mut next = 0u32;
        for _ in 0..1500 {
            let p = IPoint::new(rint(&mut rng), rint(&mut rng));
            tree.insert(M { id: next, p }); live.insert(next, p); next += 1;
        }
        for _ in 0..3000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.5 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let old = live[&id]; let np = IPoint::new(rint(&mut rng), rint(&mut rng));
                if tree.update(old, |m| m.id == id, |m| m.p = np) { live.insert(id, np); }
            } else if roll < 0.7 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                tree.remove(live[&id], |m| m.id == id); live.remove(&id);
            } else {
                let p = IPoint::new(rint(&mut rng), rint(&mut rng));
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
        let loaded = IntegerTree::<M>::deserialize(&mut Cursor::new(&buf), |r| {
            let mut a = [0u8; 4]; r.read_exact(&mut a)?; let id = u32::from_le_bytes(a);
            r.read_exact(&mut a)?; let px = i32::from_le_bytes(a);
            r.read_exact(&mut a)?; let py = i32::from_le_bytes(a);
            Ok(M { id, p: IPoint::new(px, py) })
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
        let ka: Vec<f64> = tree.knn(IPoint::new(120, 120), 8).iter().map(|(d, _)| *d).collect();
        let kb: Vec<f64> = loaded.knn(IPoint::new(120, 120), 8).iter().map(|(d, _)| *d).collect();
        assert_eq!(ka, kb, "knn differs after round-trip");
        assert!(IntegerTree::<M>::deserialize(&mut Cursor::new(&b"XXXXX"[..]), |_| unreachable!()).is_err());
    }

    #[test]
    fn knn_matches_brute() {
        let mut x = 0x0F1E_2D3Cu64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) % 256) as i32 };
        let pts: Vec<IP> = (0..2000).map(|_| IP(IPoint::new(rng(), rng()))).collect();
        let mut tree = IntegerTree::<IP>::new(IRect::new(0, 0, 256, 256), 4);
        for p in &pts { tree.insert(*p); }
        for (qx, qy) in [(128, 128), (10, 240), (0, 0), (255, 255)] {
            let q = IPoint::new(qx, qy);
            for k in [1usize, 5, 25] {
                let mut brute: Vec<f64> = pts.iter().map(|p| { let (dx, dy) = (p.0.x as f64 - qx as f64, p.0.y as f64 - qy as f64); (dx * dx + dy * dy).sqrt() }).collect();
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
        // Build with insert_ref, churn with update_ref/remove_ref/insert_ref;
        // verify a whole-world cull returns exactly the live id set (no loss /
        // dup / corruption from the handle bookkeeping) + item count.
        #[derive(Clone, Copy)]
        struct M { id: u32, p: IPoint }
        impl IPositioned for M { fn position(&self) -> IPoint { self.p } }
        struct Box2 { r: Rect }
        impl crate::Shape for Box2 {
            fn bounding_box(&self) -> Rect { self.r }
            fn contains_point(&self, p: Point) -> bool { self.r.contains(p) }
        }
        let mut x = 0x17_7EF00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut t = IntegerTree::<M>::new(IRect::new(0, 0, 1024, 1024), 6);
        let rp = |rng: &mut dyn FnMut() -> f64| IPoint::new((rng() * 1024.0) as i32, (rng() * 1024.0) as i32);
        let mut live: std::collections::HashMap<u32, ItemRef> = std::collections::HashMap::new();
        let mut next = 0u32;
        for _ in 0..2000 { let p = rp(&mut rng); let r = t.insert_ref(M { id: next, p }).unwrap(); live.insert(next, r); next += 1; }
        for _ in 0..6000 {
            let roll = rng();
            let ids: Vec<u32> = live.keys().copied().collect();
            if roll < 0.6 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                let np = rp(&mut rng);
                assert!(t.update_ref(live[&id], |m| m.p = np));
            } else if roll < 0.8 && !ids.is_empty() {
                let id = ids[(rng() * ids.len() as f64) as usize % ids.len()];
                t.remove_ref(live[&id]); live.remove(&id);
            } else { let p = rp(&mut rng); let r = t.insert_ref(M { id: next, p }).unwrap(); live.insert(next, r); next += 1; }
        }
        assert_eq!(t.item_count(), live.len(), "itree handle-churn item count drift");
        let mut want: Vec<u32> = live.keys().copied().collect();
        let mut got: Vec<u32> = t.cull(&Box2 { r: Rect::new(0.0, 0.0, 1024.0, 1024.0) }).iter().map(|m| m.id).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "itree handle-churn whole-world cull != live set");
    }

    #[test]
    #[should_panic(expected = "must be a power of two")]
    fn rejects_non_pow2_extent() {
        let _ = IntegerTree::<IP>::new(IRect::new(0, 0, 100, 128), 4);
    }

    #[test]
    fn insert_locate_smoke() {
        let mut t = IntegerTree::<IP>::new(IRect::new(0, 0, 1024, 1024), 2);
        t.insert(IP(IPoint::new(10, 10)));
        t.insert(IP(IPoint::new(500, 500)));
        t.insert(IP(IPoint::new(900, 900)));
        let n = t.locate(IPoint::new(900, 900));
        assert!(t.get(n).items.iter().any(|p| p.0 == IPoint::new(900, 900)));
    }

    #[test]
    fn update_relocates_lca() {
        let mut t = IntegerTree::<IP>::new(IRect::new(0, 0, 1024, 1024), 1);
        t.insert(IP(IPoint::new(10, 10)));
        t.insert(IP(IPoint::new(900, 900)));
        let ok = t.update(IPoint::new(10, 10), |it| it.0.x == 10,
            |it| it.0 = IPoint::new(800, 800));
        assert!(ok);
        let n = t.locate(IPoint::new(800, 800));
        assert!(t.get(n).items.iter().any(|p| p.0 == IPoint::new(800, 800)));
    }

    /// Cross-tree equality: seed the same integer-valued points into
    /// `Tree<T>` and `IntegerTree<T>`, then verify `cull` returns the same
    /// items on a few shapes. This catches drift between the integer-side
    /// descent (with IRect↔Rect conversion) and the float-side one.
    #[test]
    fn cull_matches_float_tree_on_integer_points() {
        use crate::tree::Positioned;
        use crate::Tree;

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Both { id: u32, x: i32, y: i32 }
        impl Positioned for Both {
            fn position(&self) -> Point { Point::new(self.x as f64, self.y as f64) }
        }
        impl IPositioned for Both {
            fn position(&self) -> IPoint { IPoint::new(self.x, self.y) }
        }

        struct Circle { cx: f64, cy: f64, r: f64 }
        impl Shape for Circle {
            fn bounding_box(&self) -> Rect {
                Rect::new(self.cx - self.r, self.cy - self.r, self.r * 2.0, self.r * 2.0)
            }
            fn contains_point(&self, p: Point) -> bool {
                let dx = p.x - self.cx; let dy = p.y - self.cy;
                dx*dx + dy*dy <= self.r*self.r
            }
        }

        let pts: Vec<Both> = (0..200u32).map(|i| {
            // Deterministic pseudo-random integer positions in 1024².
            let s = (i.wrapping_mul(2654435761)) as i32;
            Both { id: i, x: (s as i64 & 1023) as i32, y: ((s >> 10) as i64 & 1023) as i32 }
        }).collect();

        let mut tree = Tree::<Both>::new(Rect::new(0.0, 0.0, 1024.0, 1024.0), 4);
        let mut itree = IntegerTree::<Both>::new(IRect::new(0, 0, 1024, 1024), 4);
        for p in &pts { tree.insert(*p); itree.insert(*p); }

        for circle in [
            Circle { cx: 256.0, cy: 256.0, r: 100.0 },
            Circle { cx: 512.0, cy: 512.0, r: 300.0 },
            Circle { cx: 0.0,   cy: 0.0,   r: 50.0  },
            Circle { cx: 1000.0, cy: 1000.0, r: 200.0 },
        ] {
            let mut tree_ids: Vec<u32> = tree.cull(&circle).iter().map(|b| b.id).collect();
            let mut itree_ids: Vec<u32> = itree.cull(&circle).iter().map(|b| b.id).collect();
            tree_ids.sort();
            itree_ids.sort();
            assert_eq!(tree_ids, itree_ids,
                "cull mismatch for circle ({}, {}) r={}", circle.cx, circle.cy, circle.r);
        }
    }

    #[test]
    fn lca_state_matches_legacy() {
        fn run(strategy: IUpdateStrategy) -> Vec<IPoint> {
            let mut t = IntegerTree::<IP>::new(IRect::new(0, 0, 256, 256), 2);
            let pts = [(10,10),(30,30),(80,50),(120,200),(200,220),(50,150),(180,30),(220,90),(5,250)];
            for (x, y) in pts { t.insert(IP(IPoint::new(x, y))); }
            t.update_with(strategy, IPoint::new(10, 10), |it| it.0.x == 10,
                |it| it.0 = IPoint::new(15, 15));
            t.update_with(strategy, IPoint::new(200, 220), |it| it.0.x == 200,
                |it| it.0 = IPoint::new(5, 5));
            t.update_with(strategy, IPoint::new(180, 30), |it| it.0.x == 180,
                |it| it.0 = IPoint::new(190, 240));
            t.update_with(strategy, IPoint::new(50, 150), |it| it.0.x == 50,
                |it| it.0 = IPoint::new(9999, 9999));
            let mut out = Vec::new();
            t.visit_leaves(|_, leaf| for it in &leaf.items { out.push(it.0) });
            out.sort_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)));
            out
        }
        assert_eq!(run(IUpdateStrategy::Lca), run(IUpdateStrategy::Legacy));
    }
}
