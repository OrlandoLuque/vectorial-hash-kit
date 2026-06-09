//! Spatial tree: items live in leaf cells; cells split when they overflow.

use crate::geom::{Point, Rect};

/// Stable handle into the tree's node arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Anything the tree can index. The position determines which cell it lives in.
pub trait Positioned {
    fn position(&self) -> Point;
}

pub struct Node<T> {
    pub bbox: Rect,
    pub parent: Option<NodeId>,
    pub children: Option<[NodeId; 2]>,
    pub items: Vec<T>,
}

pub struct Tree<T: Positioned> {
    nodes: Vec<Node<T>>,
    pub item_limit: usize,
    pub root: NodeId,
}

impl<T: Positioned> Tree<T> {
    pub fn new(bbox: Rect, item_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        let root = Node { bbox, parent: None, children: None, items: Vec::new() };
        Self { nodes: vec![root], item_limit, root: NodeId(0) }
    }

    pub fn get(&self, id: NodeId) -> &Node<T> {
        &self.nodes[id.0 as usize]
    }

    fn get_mut(&mut self, id: NodeId) -> &mut Node<T> {
        &mut self.nodes[id.0 as usize]
    }

    fn alloc(&mut self, node: Node<T>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
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
    pub fn locate(&self, point: Point) -> NodeId {
        let mut current = self.root;
        loop {
            match self.get(current).children {
                None => return current,
                Some([a, b]) => {
                    current = if self.get(a).bbox.contains(point) { a } else { b };
                }
            }
        }
    }

    /// Remove the first item in the leaf at `point` matching `predicate` and
    /// return it. Triggers the merge-up rule from the paper: when a leaf's
    /// parent ends up with two leaf children whose combined items would fit
    /// in `item_limit`, the parent re-absorbs them and becomes a leaf again.
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
        let removed = {
            let items = &mut self.get_mut(leaf).items;
            let idx = items.iter().position(|it| predicate(it))?;
            items.remove(idx)
        };
        self.try_merge_up(leaf);
        Some(removed)
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

        // Item walked out of its leaf: remove and reinsert at the new position.
        let item = self.get_mut(leaf).items.remove(idx);
        self.try_merge_up(leaf);
        self.insert(item)
    }

    /// Walk upward from `node` collapsing parents that satisfy the merge-up
    /// rule: both children are leaves and their combined items fit in
    /// `item_limit`.
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
            if combined > self.item_limit {
                return;
            }
            let mut items_a = std::mem::take(&mut self.get_mut(a).items);
            let mut items_b = std::mem::take(&mut self.get_mut(b).items);
            items_a.append(&mut items_b);
            let parent = self.get_mut(parent_id);
            parent.items = items_a;
            parent.children = None;
            node = parent_id;
        }
    }

    /// Split a leaf into two children, redistribute its items, and recurse if needed.
    fn divide(&mut self, id: NodeId) {
        let (bbox, items) = {
            let n = self.get_mut(id);
            let items = std::mem::take(&mut n.items);
            (n.bbox, items)
        };

        let (a_bbox, b_bbox) = pick_split(bbox, &items);

        let a = self.alloc(Node { bbox: a_bbox, parent: Some(id), children: None, items: Vec::new() });
        let b = self.alloc(Node { bbox: b_bbox, parent: Some(id), children: None, items: Vec::new() });

        for item in items {
            let pos = item.position();
            if self.get(a).bbox.contains(pos) {
                self.get_mut(a).items.push(item);
            } else {
                self.get_mut(b).items.push(item);
            }
        }

        self.get_mut(id).children = Some([a, b]);

        if self.get(a).items.len() > self.item_limit { self.divide(a); }
        if self.get(b).items.len() > self.item_limit { self.divide(b); }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

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
        assert_eq!(tree.node_count(), nodes_before, "no nodes allocated; orphans not reclaimed");
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

    #[test]
    fn update_returns_false_when_not_found() {
        let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        tree.insert(Pt(Point::new(10.0, 10.0)));
        assert!(!tree.update(Point::new(10.0, 10.0), |it| it.0.x == 99.0, |_| {}));
        assert!(!tree.update(Point::new(500.0, 500.0), |_| true, |_| {})); // out of bounds
    }
}
