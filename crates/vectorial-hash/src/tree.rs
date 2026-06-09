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
}
