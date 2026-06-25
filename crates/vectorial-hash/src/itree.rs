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
use crate::template::CellState;
use crate::tree3::ItemRef;
use crate::Shape;

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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IUpdateStrategy {
    Legacy,
    Lca,
}

impl Default for IUpdateStrategy {
    fn default() -> Self {
        IUpdateStrategy::Lca
    }
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

    pub fn get(&self, id: INodeId) -> &INode<T> { &self.nodes[id.0 as usize] }
    fn get_mut(&mut self, id: INodeId) -> &mut INode<T> { &mut self.nodes[id.0 as usize] }

    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() { h }
        else { let h = self.locs.len() as u32; self.locs.push(IItemLoc { node: INodeId(0), slot: 0 }); h }
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
        let loc = self.locs[r.0 as usize];
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) { return true; }
        self.relocate(node, slot, np)
    }

    /// Remove the item behind a stable [`crate::ItemRef`] in O(1).
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.locs[r.0 as usize];
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handles.push(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    /// Shared LCA relocate tail (predicate `update` and `update_ref`).
    fn relocate(&mut self, leaf: INodeId, slot: usize, new_pos: IPoint) -> bool {
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
        let idx = self.get(leaf).items.iter().position(|it| predicate(it))?;
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.free_handles.push(h);
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
        let idx = match self.get(leaf).items.iter().position(|it| predicate(it)) {
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
                self.free_handles.push(h);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct IP(IPoint);
    impl IPositioned for IP { fn position(&self) -> IPoint { self.0 } }

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
