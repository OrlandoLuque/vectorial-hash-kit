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
use crate::tree::Positioned;
use crate::CellState;
use crate::Shape;

/// Stable handle into the quadtree's node arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct QNodeId(pub u32);

pub struct QNode<T> {
    pub bbox: Rect,
    pub parent: Option<QNodeId>,
    pub children: Option<[QNodeId; 4]>,
    pub items: Vec<T>,
}

pub struct QuadTree<T: Positioned> {
    nodes: Vec<QNode<T>>,
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
        let root = QNode { bbox, parent: None, children: None, items: Vec::new() };
        Self { nodes: vec![root], item_limit, merge_limit, min_cell, root: QNodeId(0) }
    }

    pub fn get(&self, id: QNodeId) -> &QNode<T> {
        &self.nodes[id.0 as usize]
    }

    fn get_mut(&mut self, id: QNodeId) -> &mut QNode<T> {
        &mut self.nodes[id.0 as usize]
    }

    fn alloc(&mut self, node: QNode<T>) -> QNodeId {
        let id = QNodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Insert an item. Returns `false` if its position falls outside the root bbox.
    pub fn insert(&mut self, item: T) -> bool {
        let pos = item.position();
        if !self.get(self.root).bbox.contains(pos) {
            return false;
        }
        let leaf = self.locate(pos);
        self.get_mut(leaf).items.push(item);
        if self.get(leaf).items.len() > self.item_limit {
            self.divide(leaf);
        }
        true
    }

    /// Find the leaf that contains `point`. Caller must ensure `point` is in-bounds.
    pub fn locate(&self, point: Point) -> QNodeId {
        let mut current = self.root;
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

    /// Remove the first matching item in the leaf at `point`; same contract
    /// as [`crate::Tree::remove`], with the 4-way merge rule.
    pub fn remove<F: Fn(&T) -> bool>(&mut self, point: Point, predicate: F) -> Option<T> {
        if !self.get(self.root).bbox.contains(point) {
            return None;
        }
        let leaf = self.locate(point);
        let removed = {
            let items = &mut self.get_mut(leaf).items;
            let idx = items.iter().position(|it| predicate(it))?;
            items.remove(idx)
        };
        self.try_merge_up(leaf);
        Some(removed)
    }

    /// Same contract as [`crate::Tree::update`].
    pub fn update<F, M>(&mut self, old_position: Point, predicate: F, mutator: M) -> bool
    where
        F: Fn(&T) -> bool,
        M: FnOnce(&mut T),
    {
        if !self.get(self.root).bbox.contains(old_position) {
            return false;
        }
        let leaf = self.locate(old_position);
        let idx = match self.get(leaf).items.iter().position(|it| predicate(it)) {
            Some(i) => i,
            None => return false,
        };
        mutator(&mut self.get_mut(leaf).items[idx]);

        let new_pos = self.get(leaf).items[idx].position();
        if self.get(leaf).bbox.contains(new_pos) {
            return true;
        }
        let item = self.get_mut(leaf).items.remove(idx);
        self.try_merge_up(leaf);
        self.insert(item)
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
            for &k in &kids {
                merged.append(&mut std::mem::take(&mut self.get_mut(k).items));
            }
            let parent = self.get_mut(parent_id);
            parent.items = merged;
            parent.children = None;
            node = parent_id;
        }
    }

    /// Split a leaf into four quadrants; same degenerate-subdivision guards
    /// as the binary tree (identical positions, minimum cell size).
    fn divide(&mut self, id: QNodeId) {
        let (bbox, items) = {
            let n = self.get_mut(id);
            let items = std::mem::take(&mut n.items);
            (n.bbox, items)
        };

        let first = items[0].position();
        let inseparable = items.iter().all(|it| it.position() == first);
        if inseparable || bbox.width.max(bbox.height) <= self.min_cell {
            self.get_mut(id).items = items;
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
            });
        }
        for item in items {
            let pos = item.position();
            let k = kids
                .iter()
                .copied()
                .find(|&k| self.get(k).bbox.contains(pos))
                .expect("quadrants tile the parent");
            self.get_mut(k).items.push(item);
        }
        self.get_mut(id).children = Some(kids);
        for k in kids {
            if self.get(k).items.len() > self.item_limit {
                self.divide(k);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
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
