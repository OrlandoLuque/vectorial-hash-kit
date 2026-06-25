//! Spatial tree: items live in leaf cells; cells split when they overflow.

use crate::geom::{Point, Rect};
use crate::tree3::ItemRef;

/// Stable handle into the tree's node arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Anything the tree can index. The position determines which cell it lives in.
pub trait Positioned {
    fn position(&self) -> Point;
}

/// One of a cell's four sides, used by the neighbour-finding APIs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    West,
    East,
    North,
    South,
}

impl Side {
    pub const ALL: [Side; 4] = [Side::West, Side::East, Side::North, Side::South];

    pub fn opposite(self) -> Side {
        match self {
            Side::West => Side::East,
            Side::East => Side::West,
            Side::North => Side::South,
            Side::South => Side::North,
        }
    }

    #[cfg_attr(not(feature = "neighbors"), allow(dead_code))]
    fn index(self) -> usize {
        match self {
            Side::West => 0,
            Side::East => 1,
            Side::North => 2,
            Side::South => 3,
        }
    }
}

/// Probe offset used by [`Tree::neighbors_probe`]; must be smaller than any
/// cell extent the tree can produce.
const PROBE_EPS: f64 = 1e-6;

/// Strategy used by [`Tree::update_with`] (and the matching `QuadTree` API)
/// to relocate an item that walked out of its leaf.
///
/// `update` itself picks the best path available at compile time
/// ([`UpdateStrategy::default`]); benchmarks call `update_with` to force a
/// specific path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UpdateStrategy {
    /// Remove the item and re-insert from the root. The pre-2026 path.
    Legacy,
    /// Ascend to the lowest common ancestor of the old leaf and the new
    /// position, then descend only within that subtree.
    Lca,
    /// First scan the old leaf's rope neighbours — if one of them contains
    /// the new position, move directly. Falls back to [`Self::Lca`] on miss.
    /// Without the `neighbors` feature the rope lists are empty, so this
    /// behaves like [`Self::Lca`] but pays for one cheap empty-list scan.
    LcaRopes,
}

impl Default for UpdateStrategy {
    fn default() -> Self {
        if cfg!(feature = "neighbors") {
            UpdateStrategy::LcaRopes
        } else {
            UpdateStrategy::Lca
        }
    }
}

pub struct Node<T> {
    pub bbox: Rect,
    pub parent: Option<NodeId>,
    pub children: Option<[NodeId; 2]>,
    pub items: Vec<T>,
    /// Parallel to `items`: the stable handle of each item (the [`ItemRef`]
    /// layer — orthogonal to the ropes below). Empty on internal nodes.
    hs: Vec<u32>,
    /// Leaf neighbour lists ("ropes") per side (W, E, N, S). Maintained on
    /// every split and merge; only meaningful for leaves.
    #[cfg(feature = "neighbors")]
    pub ropes: [Vec<NodeId>; 4],
}

impl<T> Node<T> {
    fn new_leaf(bbox: Rect, parent: Option<NodeId>) -> Self {
        Node {
            bbox,
            parent,
            children: None,
            items: Vec::new(),
            hs: Vec::new(),
            #[cfg(feature = "neighbors")]
            ropes: Default::default(),
        }
    }
}

/// Where a handle's item currently lives (leaf node + slot in its vectors).
#[derive(Copy, Clone)]
struct ItemLoc { node: NodeId, slot: u32 }

pub struct Tree<T: Positioned> {
    nodes: Vec<Node<T>>,
    /// Slots in `nodes` freed by merge-ups, reused by the next `alloc`
    /// before the arena grows. Without this, every merge orphaned its two
    /// child slots forever (unbounded arena growth under churn); with it,
    /// `nodes.len()` stabilises at the high-water-mark of live nodes.
    /// `NodeId`s of freed nodes are NOT stable — a freed id may be handed
    /// to a different node by a later `alloc`.
    free: Vec<NodeId>,
    /// Handle → current location (the stable [`ItemRef`] layer). Indexed by the
    /// handle's u32; freed handles sit on `free_handles`.
    locs: Vec<ItemLoc>,
    free_handles: Vec<u32>,
    /// A leaf splits when it holds more than this many items.
    pub item_limit: usize,
    /// Two sibling leaves merge back into their parent when their combined
    /// items fit within this. Defaults to `item_limit`; setting it lower adds
    /// hysteresis so cells don't flap between split and merged.
    pub merge_limit: usize,
    /// Leaves whose longest side is at or below this never split further,
    /// even over `item_limit` (safety net against degenerate subdivision).
    min_cell: f64,
    pub root: NodeId,
}

impl<T: Positioned> Tree<T> {
    pub fn new(bbox: Rect, item_limit: usize) -> Self {
        Self::with_limits(bbox, item_limit, item_limit)
    }

    /// Like [`Tree::new`] but with a separate merge-up threshold.
    /// `merge_limit` must not exceed `item_limit` (a parent holding more than
    /// `item_limit` items would immediately split again).
    pub fn with_limits(bbox: Rect, item_limit: usize, merge_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        assert!(
            merge_limit <= item_limit,
            "merge_limit must be <= item_limit",
        );
        let min_cell = bbox.width.max(bbox.height) * 1e-12;
        let root = Node::new_leaf(bbox, None);
        Self { nodes: vec![root], free: Vec::new(), locs: Vec::new(), free_handles: Vec::new(), item_limit, merge_limit, min_cell, root: NodeId(0) }
    }

    pub fn get(&self, id: NodeId) -> &Node<T> {
        &self.nodes[id.0 as usize]
    }

    fn get_mut(&mut self, id: NodeId) -> &mut Node<T> {
        &mut self.nodes[id.0 as usize]
    }

    // ---- stable ItemRef handle layer (mirrors Tree3) ----
    fn alloc_handle(&mut self) -> u32 {
        if let Some(h) = self.free_handles.pop() { h }
        else { let h = self.locs.len() as u32; self.locs.push(ItemLoc { node: NodeId(0), slot: 0 }); h }
    }
    fn push_h(&mut self, node: NodeId, item: T, h: u32) {
        let slot = self.get(node).items.len() as u32;
        let n = self.get_mut(node);
        n.items.push(item);
        n.hs.push(h);
        self.locs[h as usize] = ItemLoc { node, slot };
    }
    fn swap_remove_h(&mut self, node: NodeId, slot: usize) -> (T, u32) {
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

    fn alloc(&mut self, node: Node<T>) -> NodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id.0 as usize] = node;
            id
        } else {
            let id = NodeId(self.nodes.len() as u32);
            self.nodes.push(node);
            id
        }
    }

    /// Insert an item. Returns `false` if its position falls outside the root bbox.
    pub fn insert(&mut self, item: T) -> bool {
        self.insert_ref(item).is_some()
    }

    /// Insert and return a stable [`ItemRef`] for O(1) `update_ref`/`remove_ref`
    /// (skips `update`'s O(item_limit) predicate scan). `None` if out of bounds.
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

    /// Find the leaf that contains `point`. Caller must ensure `point` is in-bounds.
    pub fn locate(&self, point: Point) -> NodeId {
        self.locate_from(self.root, point)
    }

    /// Like [`Tree::locate`] but starting the descent at an arbitrary node.
    /// Used by the ascend-to-LCA `update` path so re-locating a moved item
    /// only walks the subtree rooted at the LCA.
    pub fn locate_from(&self, start: NodeId, point: Point) -> NodeId {
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

    /// Walk parents up from `leaf` until one whose bbox contains `point`.
    /// Returns `None` if the search escapes the root (`point` out of bounds).
    fn ascend_to_lca(&self, leaf: NodeId, point: Point) -> Option<NodeId> {
        let mut node = self.get(leaf).parent?;
        loop {
            if self.get(node).bbox.contains(point) {
                return Some(node);
            }
            node = self.get(node).parent?;
        }
    }

    /// Remove the first item in the leaf at `point` matching `predicate` and
    /// return it. Triggers the merge-up rule from the paper: when a leaf's
    /// parent ends up with two leaf children whose combined items would fit
    /// in `merge_limit`, the parent re-absorbs them and becomes a leaf again.
    /// The collapse cascades upward as long as the rule keeps applying.
    ///
    /// Returns `None` if `point` is outside the tree or no item in the leaf
    /// matches.
    ///
    /// Orphaned child nodes stay in the arena (their `NodeId`s remain valid
    /// but unreachable); the arena does not currently reclaim them.
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

    /// O(1) relocation through a stable [`ItemRef`] — no locate walk, no
    /// predicate scan. Mutate in place; relocate (ascend-to-LCA) only if the
    /// item leaves its leaf. `false` (and the handle freed) if it left the root.
    pub fn update_ref<M: FnOnce(&mut T)>(&mut self, r: ItemRef, mutator: M) -> bool {
        let loc = self.locs[r.0 as usize];
        let (node, slot) = (loc.node, loc.slot as usize);
        mutator(&mut self.get_mut(node).items[slot]);
        let np = self.get(node).items[slot].position();
        if self.get(node).bbox.contains(np) {
            return true;
        }
        self.relocate_via_lca(node, slot, np)
    }

    /// Remove the item behind a stable [`ItemRef`] in O(1) (no scan).
    pub fn remove_ref(&mut self, r: ItemRef) -> Option<T> {
        let loc = self.locs[r.0 as usize];
        let (item, h) = self.swap_remove_h(loc.node, loc.slot as usize);
        self.free_handles.push(h);
        self.try_merge_up(loc.node);
        Some(item)
    }

    /// Find the first item in the leaf at `old_position` matching `predicate`,
    /// apply `mutator`, and relocate it if its new position falls in a
    /// different leaf.
    ///
    /// Returns `true` when the item is found and updated. Returns `false` if:
    /// - `old_position` is outside the tree, or
    /// - no item in the leaf at `old_position` matches the predicate.
    ///
    /// If the mutator pushes the item outside the tree's root bbox, the
    /// item is removed and dropped, and the function returns `false`.
    ///
    /// Uses [`UpdateStrategy::default`] for the relocation path. Call
    /// [`Tree::update_with`] to force a specific strategy.
    pub fn update<F, M>(&mut self, old_position: Point, predicate: F, mutator: M) -> bool
    where
        F: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        self.update_with(UpdateStrategy::default(), old_position, predicate, mutator)
    }

    /// Like [`Tree::update`] but with an explicit relocation strategy.
    /// Exists for benchmarking; production callers should use `update`.
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
                // remove + re-insert reassigns the item's handle (any held
                // ItemRef to it is invalidated) — the default Lca path below
                // preserves it. Legacy is a benchmarking-only strategy.
                let (item, h) = self.swap_remove_h(leaf, idx);
                self.free_handles.push(h);
                self.try_merge_up(leaf);
                self.insert(item)
            }
            UpdateStrategy::Lca => self.relocate_via_lca(leaf, idx, new_pos),
            UpdateStrategy::LcaRopes => {
                #[cfg(feature = "neighbors")]
                if let Some(nbr) = self.find_rope_neighbour(leaf, new_pos) {
                    return self.relocate_to_neighbour(leaf, idx, nbr);
                }
                self.relocate_via_lca(leaf, idx, new_pos)
            }
        }
    }

    /// LCA path: ascend from `leaf` until an ancestor contains `new_pos`,
    /// then descend into that subtree to find the destination leaf. Returns
    /// `false` if `new_pos` is out of root bounds (item dropped).
    fn relocate_via_lca(&mut self, leaf: NodeId, idx: usize, new_pos: Point) -> bool {
        let lca = match self.ascend_to_lca(leaf, new_pos) {
            Some(id) => id,
            None => {
                // Out of bounds: drop the item (freeing its handle), then merge.
                let (_, h) = self.swap_remove_h(leaf, idx);
                self.free_handles.push(h);
                self.try_merge_up(leaf);
                return false;
            }
        };
        let (item, h) = self.swap_remove_h(leaf, idx);
        let dest = self.locate_from(lca, new_pos);
        self.push_h(dest, item, h);
        if self.get(dest).items.len() > self.item_limit {
            self.divide(dest);
        }
        self.try_merge_up(leaf);
        true
    }

    /// Direct neighbour-leaf move: pop the item from `leaf` and push to `nbr`.
    /// Triggers the usual divide/merge-up bookkeeping on each side.
    #[cfg(feature = "neighbors")]
    fn relocate_to_neighbour(&mut self, leaf: NodeId, idx: usize, nbr: NodeId) -> bool {
        let (item, h) = self.swap_remove_h(leaf, idx);
        self.push_h(nbr, item, h);
        if self.get(nbr).items.len() > self.item_limit {
            self.divide(nbr);
        }
        self.try_merge_up(leaf);
        true
    }

    /// Scan the leaf's four rope lists for a neighbour leaf whose bbox
    /// contains `point`. `O(rope sum)` — for a balanced tree, a small
    /// constant.
    #[cfg(feature = "neighbors")]
    fn find_rope_neighbour(&self, leaf: NodeId, point: Point) -> Option<NodeId> {
        for side in Side::ALL {
            for &nbr in &self.get(leaf).ropes[side.index()] {
                if self.get(nbr).bbox.contains(point) {
                    return Some(nbr);
                }
            }
        }
        None
    }

    /// Walk upward from `node` collapsing parents that satisfy the merge-up
    /// rule: both children are leaves and their combined items fit in
    /// `merge_limit`.
    fn try_merge_up(&mut self, mut node: NodeId) {
        loop {
            let parent_id = match self.get(node).parent {
                Some(p) => p,
                None => return,
            };
            let [a, b] = self
                .get(parent_id)
                .children
                .expect("parent must have children");
            if self.get(a).children.is_some() || self.get(b).children.is_some() {
                return;
            }
            let combined = self.get(a).items.len() + self.get(b).items.len();
            if combined > self.merge_limit {
                return;
            }
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
            // The merged items now live in `parent` (a leaf again) — re-point
            // their handle locations.
            let len = self.get(parent_id).hs.len();
            for slot in 0..len {
                let h = self.get(parent_id).hs[slot];
                self.locs[h as usize] = ItemLoc { node: parent_id, slot: slot as u32 };
            }
            #[cfg(feature = "neighbors")]
            self.update_ropes_on_merge(parent_id, a, b);
            // The two child slots are now unreachable — return them to the
            // free-list so the next split reuses them instead of growing the
            // arena. (Their `items` were taken above; ropes, if any, were
            // cleared by `update_ropes_on_merge`.)
            self.free.push(a);
            self.free.push(b);
            node = parent_id;
        }
    }

    /// Split a leaf into two children, redistribute its items, and recurse if
    /// needed. A leaf over `item_limit` stays unsplit (soft limit) when no
    /// split can make progress: all its items share one position, or the
    /// cell has already shrunk to the minimum size.
    fn divide(&mut self, id: NodeId) {
        let (bbox, items, hs) = {
            let n = self.get_mut(id);
            let items = std::mem::take(&mut n.items);
            let hs = std::mem::take(&mut n.hs);
            (n.bbox, items, hs)
        };

        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.width.max(bbox.height) <= self.min_cell {
            // Put them back unchanged — same slots, so `locs` stays valid.
            let n = self.get_mut(id);
            n.items = items;
            n.hs = hs;
            return;
        }

        let (a_bbox, b_bbox) = pick_split(bbox, &items);

        let a = self.alloc(Node::new_leaf(a_bbox, Some(id)));
        let b = self.alloc(Node::new_leaf(b_bbox, Some(id)));

        for (item, h) in items.into_iter().zip(hs) {
            let pos = item.position();
            let dest = if self.get(a).bbox.contains(pos) { a } else { b };
            self.push_h(dest, item, h);
        }

        self.get_mut(id).children = Some([a, b]);

        #[cfg(feature = "neighbors")]
        self.update_ropes_on_split(id, a, b, a_bbox.width < bbox.width);

        if self.get(a).items.len() > self.item_limit { self.divide(a); }
        if self.get(b).items.len() > self.item_limit { self.divide(b); }
    }

    /// Arena capacity: the high-water-mark of simultaneously-live nodes
    /// (the `Vec` never shrinks). With the free-list this stays bounded
    /// under churn instead of growing without limit. Use
    /// [`Tree::live_node_count`] for the currently-reachable count.
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Currently-reachable nodes (arena capacity minus free-list slots).
    pub fn live_node_count(&self) -> usize { self.nodes.len() - self.free.len() }

    /// Visit every live leaf reachable from the root, in depth-first order.
    ///
    /// Orphaned nodes left behind by `remove`/`update` merges are not visited.
    /// This is the intended way to enumerate current regions (e.g. for
    /// rendering the tree's subdivision) or to snapshot all stored items.
    pub fn visit_leaves<F: FnMut(NodeId, &Node<T>)>(&self, mut f: F) {
        self.visit_leaves_from(self.root, &mut f);
    }

    fn visit_leaves_from<F: FnMut(NodeId, &Node<T>)>(&self, id: NodeId, f: &mut F) {
        match self.get(id).children {
            Some([a, b]) => {
                self.visit_leaves_from(a, f);
                self.visit_leaves_from(b, f);
            }
            None => f(id, self.get(id)),
        }
    }

    /// Number of items currently stored (live leaves only).
    pub fn item_count(&self) -> usize {
        let mut n = 0;
        self.visit_leaves(|_, leaf| n += leaf.items.len());
        n
    }

    /// Number of live leaves reachable from the root.
    pub fn leaf_count(&self) -> usize {
        let mut n = 0;
        self.visit_leaves(|_, _| n += 1);
        n
    }

    // ----- neighbour finding -----

    /// Samet-style neighbour finding: ascend from `leaf` until the first
    /// ancestor whose split crosses `side`, then descend the sibling along
    /// the shared edge, collecting every adjacent leaf. Uses the parent
    /// pointers the arena already has — zero extra storage; O(1) amortized,
    /// O(depth) worst case.
    ///
    /// Reference: H. Samet, "Neighbor Finding Techniques for Images
    /// Represented by Quadtrees" (1982), adapted to a binary-split tree.
    pub fn neighbors_samet(&self, leaf: NodeId, side: Side, out: &mut Vec<NodeId>) {
        let mut node = leaf;
        let target = loop {
            let Some(parent) = self.get(node).parent else {
                return; // reached the root: `side` is the map border
            };
            let [a, b] = self.get(parent).children.expect("parent has children");
            let vertical = self.get(a).bbox.width < self.get(parent).bbox.width;
            // `a` is always the west (vertical) / north (horizontal) child.
            let crossing = match (side, vertical) {
                (Side::East, true) if node == a => Some(b),
                (Side::West, true) if node == b => Some(a),
                (Side::South, false) if node == a => Some(b),
                (Side::North, false) if node == b => Some(a),
                _ => None,
            };
            match crossing {
                Some(sibling) => break sibling,
                None => node = parent,
            }
        };
        let lb = self.get(leaf).bbox;
        self.collect_edge_leaves(target, side, &lb, out);
    }

    /// Descend `node`, keeping only subtrees that touch the `side` edge of
    /// `lb` and overlap its perpendicular extent; push the reached leaves.
    fn collect_edge_leaves(&self, node: NodeId, side: Side, lb: &Rect, out: &mut Vec<NodeId>) {
        match self.get(node).children {
            None => out.push(node),
            Some([a, b]) => {
                for child in [a, b] {
                    let cb = self.get(child).bbox;
                    let touches = match side {
                        Side::East => cb.x == lb.x_max(),
                        Side::West => cb.x_max() == lb.x,
                        Side::South => cb.y == lb.y_max(),
                        Side::North => cb.y_max() == lb.y,
                    };
                    let overlaps = match side {
                        Side::East | Side::West => cb.y < lb.y_max() && lb.y < cb.y_max(),
                        Side::North | Side::South => cb.x < lb.x_max() && lb.x < cb.x_max(),
                    };
                    if touches && overlaps {
                        self.collect_edge_leaves(child, side, lb, out);
                    }
                }
            }
        }
    }

    /// Pointerless neighbour finding by probing: take points just across the
    /// `side` edge and `locate` each adjacent leaf from the root, advancing
    /// along the edge by each found leaf's extent. Zero storage; O(depth)
    /// per adjacent leaf.
    pub fn neighbors_probe(&self, leaf: NodeId, side: Side, out: &mut Vec<NodeId>) {
        let b = self.get(leaf).bbox;
        let root = self.get(self.root).bbox;
        match side {
            Side::East | Side::West => {
                let x = if side == Side::East { b.x_max() + PROBE_EPS } else { b.x - PROBE_EPS };
                if x < root.x || x >= root.x_max() {
                    return;
                }
                let mut y = b.y + PROBE_EPS;
                while y < b.y_max() {
                    let n = self.locate(Point::new(x, y));
                    out.push(n);
                    y = self.get(n).bbox.y_max() + PROBE_EPS;
                }
            }
            Side::North | Side::South => {
                let y = if side == Side::South { b.y_max() + PROBE_EPS } else { b.y - PROBE_EPS };
                if y < root.y || y >= root.y_max() {
                    return;
                }
                let mut x = b.x + PROBE_EPS;
                while x < b.x_max() {
                    let n = self.locate(Point::new(x, y));
                    out.push(n);
                    x = self.get(n).bbox.x_max() + PROBE_EPS;
                }
            }
        }
    }

    /// Stored neighbour lists ("ropes"): O(1) access, maintained on every
    /// split and merge. Only available with the `neighbors` feature.
    #[cfg(feature = "neighbors")]
    pub fn neighbors_ropes(&self, leaf: NodeId, side: Side) -> &[NodeId] {
        &self.get(leaf).ropes[side.index()]
    }

    /// Rewire ropes when leaf `p` splits into `a` (west/north) and `b`.
    #[cfg(feature = "neighbors")]
    fn update_ropes_on_split(&mut self, p: NodeId, a: NodeId, b: NodeId, vertical: bool) {
        let p_ropes = std::mem::take(&mut self.get_mut(p).ropes);
        let (wi, ei, ni, si) = (
            Side::West.index(),
            Side::East.index(),
            Side::North.index(),
            Side::South.index(),
        );
        // Sides parallel to the split: one child inherits the whole list.
        // Sides perpendicular: the list is divided by overlap (a neighbour
        // can border both children).
        let (full_a, full_b, splits) = if vertical {
            // a west, b east; N/S lists divide by x-overlap.
            ((wi, ei), (ei, wi), [ni, si])
        } else {
            // a north, b south; W/E lists divide by y-overlap.
            ((ni, si), (si, ni), [wi, ei])
        };

        // Inherited full sides.
        let (a_side, a_opp) = full_a;
        for &n in &p_ropes[a_side] {
            Self::replace_rope(&mut self.get_mut(n).ropes[a_opp], p, &[a]);
        }
        self.get_mut(a).ropes[a_side] = p_ropes[a_side].clone();
        let (b_side, b_opp) = full_b;
        for &n in &p_ropes[b_side] {
            Self::replace_rope(&mut self.get_mut(n).ropes[b_opp], p, &[b]);
        }
        self.get_mut(b).ropes[b_side] = p_ropes[b_side].clone();

        // The new internal edge.
        self.get_mut(a).ropes[b_side] = vec![b];
        self.get_mut(b).ropes[a_side] = vec![a];

        // Perpendicular sides: distribute by overlap.
        let (ab_a, ab_b) = (self.get(a).bbox, self.get(b).bbox);
        for side_idx in splits {
            let opp = match side_idx {
                x if x == ni => si,
                x if x == si => ni,
                x if x == wi => ei,
                _ => wi,
            };
            let mut list_a = Vec::new();
            let mut list_b = Vec::new();
            for &n in &p_ropes[side_idx] {
                let nb = self.get(n).bbox;
                let overlap = |cb: &Rect| {
                    if vertical {
                        nb.x < cb.x_max() && cb.x < nb.x_max()
                    } else {
                        nb.y < cb.y_max() && cb.y < nb.y_max()
                    }
                };
                let mut subs: Vec<NodeId> = Vec::with_capacity(2);
                if overlap(&ab_a) {
                    list_a.push(n);
                    subs.push(a);
                }
                if overlap(&ab_b) {
                    list_b.push(n);
                    subs.push(b);
                }
                Self::replace_rope(&mut self.get_mut(n).ropes[opp], p, &subs);
            }
            self.get_mut(a).ropes[side_idx] = list_a;
            self.get_mut(b).ropes[side_idx] = list_b;
        }
    }

    /// Rewire ropes when `p` re-absorbs its leaf children `a` and `b`.
    #[cfg(feature = "neighbors")]
    fn update_ropes_on_merge(&mut self, p: NodeId, a: NodeId, b: NodeId) {
        let ra = std::mem::take(&mut self.get_mut(a).ropes);
        let rb = std::mem::take(&mut self.get_mut(b).ropes);
        for side in Side::ALL {
            let i = side.index();
            let opp = side.opposite().index();
            let mut merged: Vec<NodeId> = Vec::with_capacity(ra[i].len() + rb[i].len());
            for &n in ra[i].iter().chain(rb[i].iter()) {
                if n != a && n != b && !merged.contains(&n) {
                    merged.push(n);
                }
            }
            for &n in &merged {
                let list = &mut self.get_mut(n).ropes[opp];
                list.retain(|&x| x != a && x != b);
                if !list.contains(&p) {
                    list.push(p);
                }
            }
            self.get_mut(p).ropes[i] = merged;
        }
    }

    /// Remove `old` from `list` and append `subs` (no duplicates expected).
    #[cfg(feature = "neighbors")]
    fn replace_rope(list: &mut Vec<NodeId>, old: NodeId, subs: &[NodeId]) {
        list.retain(|&x| x != old);
        for &s in subs {
            if !list.contains(&s) {
                list.push(s);
            }
        }
    }
}

/// Pick how to split a cell.
///
/// - Rectangles split along the long axis so both children are closer to square.
/// - Squares pick the axis that distributes items most evenly.
fn pick_split<T: Positioned>(bbox: Rect, items: &[T]) -> (Rect, Rect) {
    if bbox.width > bbox.height {
        let half = bbox.width / 2.0;
        (
            Rect::new(bbox.x, bbox.y, half, bbox.height),
            Rect::new(bbox.x + half, bbox.y, half, bbox.height),
        )
    } else if bbox.height > bbox.width {
        let half = bbox.height / 2.0;
        (
            Rect::new(bbox.x, bbox.y, bbox.width, half),
            Rect::new(bbox.x, bbox.y + half, bbox.width, half),
        )
    } else {
        let mid_x = bbox.x + bbox.width / 2.0;
        let mid_y = bbox.y + bbox.height / 2.0;
        let left = items.iter().filter(|it| it.position().x < mid_x).count();
        let top = items.iter().filter(|it| it.position().y < mid_y).count();
        let n = items.len() as i64;
        let vert_balance = (2 * left as i64 - n).abs();
        let horz_balance = (2 * top as i64 - n).abs();
        if vert_balance <= horz_balance {
            let half = bbox.width / 2.0;
            (
                Rect::new(bbox.x, bbox.y, half, bbox.height),
                Rect::new(bbox.x + half, bbox.y, half, bbox.height),
            )
        } else {
            let half = bbox.height / 2.0;
            (
                Rect::new(bbox.x, bbox.y, bbox.width, half),
                Rect::new(bbox.x, bbox.y + half, bbox.width, half),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Pt(Point);
    impl Positioned for Pt {
        fn position(&self) -> Point { self.0 }
    }

    #[test]
    fn insert_lands_in_root_when_under_limit() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        assert!(tree.insert(Pt(Point::new(10.0, 10.0))));
        assert!(tree.insert(Pt(Point::new(20.0, 80.0))));
        assert_eq!(tree.get(tree.root).items.len(), 2);
        assert!(tree.get(tree.root).children.is_none());
    }

    #[test]
    fn update_ref_churn_matches_brute() {
        // Build with insert_ref, churn with update_ref/remove_ref/insert_ref
        // (the O(1) stable-handle path); verify cull == brute (by id) under the
        // splits/merges that move handles between leaves.
        #[derive(Clone, Copy)]
        struct M { id: u32, p: Point }
        impl Positioned for M { fn position(&self) -> Point { self.p } }
        struct Box2 { r: Rect }
        impl crate::Shape for Box2 {
            fn bounding_box(&self) -> Rect { self.r }
            fn contains_point(&self, p: Point) -> bool { self.r.contains(p) }
        }

        let mut x = 0x2D7_EF00Du64;
        let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
        let mut tree = Tree::<M>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 6);
        let rp = |rng: &mut dyn FnMut() -> f64| Point::new(rng() * 256.0, rng() * 256.0);
        let mut live: std::collections::HashMap<u32, (ItemRef, Point)> = std::collections::HashMap::new();
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
                assert!(tree.update_ref(r, |m| m.p = np));
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
        for r in [Rect::new(100.0, 100.0, 60.0, 60.0), Rect::new(0.0, 0.0, 128.0, 128.0), Rect::new(200.0, 10.0, 50.0, 200.0)] {
            let mut want: Vec<u32> = live.iter()
                .filter(|(_, (_, p))| r.contains(*p)).map(|(id, _)| *id).collect();
            let mut got: Vec<u32> = tree.cull(&Box2 { r }).iter().map(|m| m.id).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "2D handle-churn cull != brute for {r:?}");
        }
    }

    #[test]
    fn insert_rejects_out_of_bounds() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 10.0, 10.0), 4);
        assert!(!tree.insert(Pt(Point::new(15.0, 5.0))));
    }

    #[test]
    fn overflow_triggers_division() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 10.0)));
        assert!(tree.get(tree.root).children.is_none());
        tree.insert(Pt(Point::new(80.0, 80.0)));
        assert!(tree.get(tree.root).children.is_some());
    }

    #[test]
    fn rectangle_splits_along_long_axis() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 200.0, 100.0), 1);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(150.0, 50.0)));
        let [a, b] = tree.get(tree.root).children.unwrap();
        // both children should be 100x100 (the long axis was halved)
        assert_eq!(tree.get(a).bbox, Rect::new(0.0, 0.0, 100.0, 100.0));
        assert_eq!(tree.get(b).bbox, Rect::new(100.0, 0.0, 100.0, 100.0));
    }

    #[test]
    fn remove_returns_item_and_leaves_tree_consistent() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        let removed = tree.remove(Point::new(10.0, 10.0), |it| it.0.x == 10.0);
        assert_eq!(removed.map(|p| p.0), Some(Point::new(10.0, 10.0)));
        assert_eq!(tree.get(tree.root).items.len(), 1);
    }

    #[test]
    fn remove_returns_none_when_no_match() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        assert!(tree.remove(Point::new(10.0, 10.0), |it| it.0.x == 99.0).is_none());
        assert!(tree.remove(Point::new(500.0, 500.0), |_| true).is_none()); // out of bounds
    }

    #[test]
    fn remove_collapses_parent_via_merge_up_rule() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
        // Trigger a split: 3 items, item_limit = 2.
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        tree.insert(Pt(Point::new(80.0, 80.0)));
        assert!(tree.get(tree.root).children.is_some());

        // Remove one item; now combined count = 2 == item_limit → merge up.
        let removed = tree.remove(Point::new(80.0, 80.0), |it| it.0.x == 80.0);
        assert!(removed.is_some());
        assert!(tree.get(tree.root).children.is_none(), "parent should have absorbed children");
        assert_eq!(tree.get(tree.root).items.len(), 2);
    }

    #[test]
    fn remove_does_not_collapse_when_combined_exceeds_limit() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        // item_limit = 1, three items: must split deeper.
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        tree.insert(Pt(Point::new(80.0, 80.0)));
        let nodes_before = tree.node_count();
        tree.remove(Point::new(20.0, 20.0), |it| it.0.x == 20.0);
        // After removing one item we have two left; limit = 1, so the two
        // surviving leaves should NOT merge into their parent.
        assert!(tree.get(tree.root).children.is_some());
        assert_eq!(tree.node_count(), nodes_before, "no merge happened, so no nodes allocated or freed");
    }

    #[test]
    fn free_list_reuses_orphaned_slots_under_churn() {
        // Repeatedly split then merge the same region: the arena must NOT
        // grow each cycle — the merged children's slots get reused.
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        // First split.
        tree.insert(Pt(Point::new(80.0, 80.0)));
        assert!(tree.get(tree.root).children.is_some());
        let cap_after_first_split = tree.node_count();
        for _ in 0..50 {
            // Merge: remove the far item → combined fits → collapse.
            tree.remove(Point::new(80.0, 80.0), |it| it.0.x == 80.0);
            assert!(tree.get(tree.root).children.is_none());
            // Split again: reinsert the far item.
            tree.insert(Pt(Point::new(80.0, 80.0)));
            assert!(tree.get(tree.root).children.is_some());
        }
        // Arena capacity stayed put (slots reused), and the live count is
        // exactly the 3 reachable nodes (root + 2 leaves).
        assert_eq!(tree.node_count(), cap_after_first_split, "arena grew despite the free-list");
        assert_eq!(tree.live_node_count(), 3);
        assert_eq!(tree.item_count(), 3);
    }

    #[test]
    fn update_in_place_when_new_position_stays_in_leaf() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        let ok = tree.update(
            Point::new(10.0, 10.0),
            |it| it.0.x == 10.0,
            |it| it.0 = Point::new(15.0, 15.0),
        );
        assert!(ok);
        let positions: Vec<_> = tree.get(tree.root).items.iter().map(|p| p.0).collect();
        assert!(positions.contains(&Point::new(15.0, 15.0)));
    }

    #[test]
    fn update_relocates_when_new_position_lands_in_another_leaf() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(80.0, 80.0)));
        let [left, right] = tree.get(tree.root).children.unwrap();
        // Move the left-half item to the right half.
        let ok = tree.update(
            Point::new(10.0, 10.0),
            |it| it.0.x == 10.0,
            |it| it.0 = Point::new(70.0, 70.0),
        );
        assert!(ok);
        // The point at (70, 70) now sits with the existing (80, 80) on the
        // right child. (item_limit = 1, so the right leaf may have split again
        // to accommodate both — what matters is the left leaf is now empty.)
        assert!(tree.get(left).items.is_empty());
        // Locating (70, 70) should hit a leaf containing it.
        let landed = tree.locate(Point::new(70.0, 70.0));
        assert!(tree.get(landed).items.iter().any(|p| p.0 == Point::new(70.0, 70.0)),
            "moved item should live under the right subtree (root child = {:?})", right);
    }

    #[test]
    fn duplicate_positions_do_not_split_forever() {
        // 5 identical positions with item_limit 1: an unguarded divide would
        // recurse forever (identical points can never be separated).
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        for _ in 0..5 {
            assert!(tree.insert(Pt(Point::new(10.0, 10.0))));
        }
        assert_eq!(tree.item_count(), 5);
        // Mixed case: duplicates plus a separable point still divides sanely.
        assert!(tree.insert(Pt(Point::new(90.0, 90.0))));
        assert_eq!(tree.item_count(), 6);
        let leaf = tree.locate(Point::new(10.0, 10.0));
        assert_eq!(tree.get(leaf).items.len(), 5, "duplicates stay in one leaf");
    }

    #[test]
    fn merge_limit_adds_hysteresis() {
        // Split above 2, but only merge back when 1 item remains.
        let mut tree = Tree::<Pt>::with_limits(Rect::new(0.0, 0.0, 100.0, 100.0), 2, 1);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(20.0, 20.0)));
        tree.insert(Pt(Point::new(80.0, 80.0)));
        assert!(tree.get(tree.root).children.is_some());

        // 2 items remain: with merge_limit = item_limit this would collapse,
        // but merge_limit = 1 keeps the split.
        tree.remove(Point::new(80.0, 80.0), |it| it.0.x == 80.0);
        assert!(tree.get(tree.root).children.is_some(), "hysteresis must keep the split");

        // 1 item remains: now it collapses.
        tree.remove(Point::new(20.0, 20.0), |it| it.0.x == 20.0);
        assert!(tree.get(tree.root).children.is_none());
        assert_eq!(tree.get(tree.root).items.len(), 1);
    }

    #[test]
    #[should_panic(expected = "merge_limit must be <= item_limit")]
    fn merge_limit_above_item_limit_panics() {
        let _ = Tree::<Pt>::with_limits(Rect::new(0.0, 0.0, 10.0, 10.0), 2, 3);
    }

    #[test]
    fn visit_leaves_covers_live_leaves_only() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 1);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        tree.insert(Pt(Point::new(80.0, 80.0)));
        tree.insert(Pt(Point::new(90.0, 10.0)));
        assert_eq!(tree.item_count(), 3);
        let leaves = tree.leaf_count();
        assert!(leaves >= 2);

        // Remove one item; merges may shrink the leaf set, but counts stay consistent.
        tree.remove(Point::new(90.0, 10.0), |it| it.0.x == 90.0);
        assert_eq!(tree.item_count(), 2);
        assert!(tree.leaf_count() <= leaves);

        // Every visited node must actually be a leaf.
        tree.visit_leaves(|_, leaf| assert!(leaf.children.is_none()));
    }

    /// Brute-force adjacency: leaves sharing an edge segment on `side`.
    fn neighbors_brute(tree: &Tree<Pt>, leaf: NodeId, side: Side) -> Vec<NodeId> {
        let lb = tree.get(leaf).bbox;
        let mut out = Vec::new();
        tree.visit_leaves(|id, n| {
            if id == leaf {
                return;
            }
            let cb = n.bbox;
            let touches = match side {
                Side::East => cb.x == lb.x_max(),
                Side::West => cb.x_max() == lb.x,
                Side::South => cb.y == lb.y_max(),
                Side::North => cb.y_max() == lb.y,
            };
            let overlaps = match side {
                Side::East | Side::West => cb.y < lb.y_max() && lb.y < cb.y_max(),
                Side::North | Side::South => cb.x < lb.x_max() && lb.x < cb.x_max(),
            };
            if touches && overlaps {
                out.push(id);
            }
        });
        out
    }

    fn scattered_tree() -> Tree<Pt> {
        // Deterministic pseudo-random points, with removals to force merges.
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 2);
        let mut x = 0xDEADBEEFu64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut pts = Vec::new();
        for _ in 0..120 {
            let p = Point::new(next() * 256.0, next() * 256.0);
            pts.push(p);
            tree.insert(Pt(p));
        }
        for p in pts.iter().step_by(3) {
            tree.remove(*p, |it| it.0 == *p);
        }
        tree
    }

    #[test]
    fn samet_and_probe_neighbors_match_brute_force() {
        let tree = scattered_tree();
        let mut leaves = Vec::new();
        tree.visit_leaves(|id, _| leaves.push(id));
        for &leaf in &leaves {
            for side in Side::ALL {
                let mut expected = neighbors_brute(&tree, leaf, side);
                expected.sort_by_key(|n| n.0);

                let mut samet = Vec::new();
                tree.neighbors_samet(leaf, side, &mut samet);
                samet.sort_by_key(|n| n.0);
                assert_eq!(samet, expected, "samet {side:?} of {leaf:?}");

                let mut probe = Vec::new();
                tree.neighbors_probe(leaf, side, &mut probe);
                probe.sort_by_key(|n| n.0);
                assert_eq!(probe, expected, "probe {side:?} of {leaf:?}");
            }
        }
    }

    #[cfg(feature = "neighbors")]
    #[test]
    fn ropes_match_brute_force_after_churn() {
        let tree = scattered_tree();
        let mut leaves = Vec::new();
        tree.visit_leaves(|id, _| leaves.push(id));
        for &leaf in &leaves {
            for side in Side::ALL {
                let mut expected = neighbors_brute(&tree, leaf, side);
                expected.sort_by_key(|n| n.0);
                let mut ropes: Vec<NodeId> = tree.neighbors_ropes(leaf, side).to_vec();
                ropes.sort_by_key(|n| n.0);
                assert_eq!(ropes, expected, "ropes {side:?} of {leaf:?}");
            }
        }
    }

    #[test]
    fn update_returns_false_when_not_found() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        assert!(!tree.update(Point::new(10.0, 10.0), |it| it.0.x == 99.0, |_| {}));
        assert!(!tree.update(Point::new(500.0, 500.0), |_| true, |_| {})); // out of bounds
    }

    /// Build a tree with the same seeded points and return the sorted leaf
    /// positions — used to confirm two strategies converge on identical state.
    fn relocate_under(strategy: UpdateStrategy) -> Vec<Point> {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 256.0, 256.0), 2);
        let pts = [
            (10.0, 10.0), (30.0, 30.0), (80.0, 50.0),
            (120.0, 200.0), (200.0, 220.0), (50.0, 150.0),
            (180.0, 30.0), (220.0, 90.0), (5.0, 250.0),
        ];
        for (x, y) in pts { tree.insert(Pt(Point::new(x, y))); }
        // A handful of moves: in-leaf, sibling, cross-LCA-up-the-tree, out of bounds.
        tree.update_with(strategy, Point::new(10.0, 10.0), |it| it.0.x == 10.0,
            |it| it.0 = Point::new(15.0, 15.0));
        tree.update_with(strategy, Point::new(200.0, 220.0), |it| it.0.x == 200.0,
            |it| it.0 = Point::new(5.0, 5.0));
        tree.update_with(strategy, Point::new(180.0, 30.0), |it| it.0.x == 180.0,
            |it| it.0 = Point::new(190.0, 240.0));
        tree.update_with(strategy, Point::new(50.0, 150.0), |it| it.0.x == 50.0,
            |it| it.0 = Point::new(1000.0, 1000.0)); // out of bounds: dropped
        let mut out: Vec<Point> = Vec::new();
        tree.visit_leaves(|_, leaf| { for it in &leaf.items { out.push(it.0); } });
        out.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
        out
    }

    #[test]
    fn lca_strategy_matches_legacy_state() {
        assert_eq!(relocate_under(UpdateStrategy::Lca), relocate_under(UpdateStrategy::Legacy));
    }

    #[cfg(feature = "neighbors")]
    #[test]
    fn lca_ropes_strategy_matches_legacy_state() {
        assert_eq!(relocate_under(UpdateStrategy::LcaRopes), relocate_under(UpdateStrategy::Legacy));
    }
}
