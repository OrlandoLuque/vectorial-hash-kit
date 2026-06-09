//! Reference quadtree used by `vh bench` to compare against the
//! binary-split tree in `vectorial-hash`.
//!
//! Mirrors the tree's semantics (items in leaves, overflow splits, same
//! `Shape` trait with the green/yellow/white template short-circuit) but
//! always splits into 4 equal quadrants — the classic quadtree layout.

use vectorial_hash::{CellState, Positioned, Rect, Shape};

/// Leaves smaller than this never split further; guards against degenerate
/// recursion when many items share a position.
const MIN_SIDE: f64 = 1e-6;

struct QNode<T> {
    bbox: Rect,
    children: Option<[usize; 4]>,
    items: Vec<T>,
}

pub struct QuadTree<T: Positioned> {
    nodes: Vec<QNode<T>>,
    pub item_limit: usize,
}

impl<T: Positioned> QuadTree<T> {
    pub fn new(bbox: Rect, item_limit: usize) -> Self {
        assert!(item_limit >= 1, "item_limit must be >= 1");
        Self {
            nodes: vec![QNode { bbox, children: None, items: Vec::new() }],
            item_limit,
        }
    }

    pub fn insert(&mut self, item: T) -> bool {
        let pos = item.position();
        if !self.nodes[0].bbox.contains(pos) {
            return false;
        }
        let leaf = self.locate(pos);
        self.nodes[leaf].items.push(item);
        if self.nodes[leaf].items.len() > self.item_limit {
            self.divide(leaf);
        }
        true
    }

    fn locate(&self, pos: vectorial_hash::Point) -> usize {
        let mut current = 0;
        loop {
            match self.nodes[current].children {
                None => return current,
                Some(kids) => {
                    current = *kids
                        .iter()
                        .find(|&&k| self.nodes[k].bbox.contains(pos))
                        .expect("quadrants tile the parent");
                }
            }
        }
    }

    fn divide(&mut self, id: usize) {
        let bbox = self.nodes[id].bbox;
        if bbox.width / 2.0 < MIN_SIDE || bbox.height / 2.0 < MIN_SIDE {
            return;
        }
        let items = std::mem::take(&mut self.nodes[id].items);
        let hw = bbox.width / 2.0;
        let hh = bbox.height / 2.0;
        let quads = [
            Rect::new(bbox.x, bbox.y, hw, hh),
            Rect::new(bbox.x + hw, bbox.y, hw, hh),
            Rect::new(bbox.x, bbox.y + hh, hw, hh),
            Rect::new(bbox.x + hw, bbox.y + hh, hw, hh),
        ];
        let mut kids = [0usize; 4];
        for (i, q) in quads.iter().enumerate() {
            kids[i] = self.nodes.len();
            self.nodes.push(QNode { bbox: *q, children: None, items: Vec::new() });
        }
        for item in items {
            let pos = item.position();
            let k = kids
                .iter()
                .copied()
                .find(|&k| self.nodes[k].bbox.contains(pos))
                .expect("quadrants tile the parent");
            self.nodes[k].items.push(item);
        }
        self.nodes[id].children = Some(kids);
        for k in kids {
            if self.nodes[k].items.len() > self.item_limit {
                self.divide(k);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Same culling contract as `vectorial_hash::Tree::cull`, including the
    /// template short-circuit when the shape provides a `TemplateGrid`.
    pub fn cull<'a, S: Shape>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        let bbox = shape.bounding_box();
        self.cull_recurse(0, shape, &bbox, false, &mut out);
        out
    }

    fn cull_recurse<'a, S: Shape>(
        &'a self,
        id: usize,
        shape: &S,
        shape_bbox: &Rect,
        fully_inside: bool,
        out: &mut Vec<&'a T>,
    ) {
        let node = &self.nodes[id];

        if fully_inside {
            match node.children {
                Some(kids) => {
                    for k in kids {
                        self.cull_recurse(k, shape, shape_bbox, true, out);
                    }
                }
                None => out.extend(node.items.iter()),
            }
            return;
        }

        match node.children {
            Some(kids) => {
                for k in kids {
                    let child_bbox = self.nodes[k].bbox;
                    match classify(shape, shape_bbox, &child_bbox) {
                        CellState::Out => {}
                        CellState::In => self.cull_recurse(k, shape, shape_bbox, true, out),
                        CellState::Maybe => self.cull_recurse(k, shape, shape_bbox, false, out),
                    }
                }
            }
            None => {
                for it in &node.items {
                    if shape.contains_point(it.position()) {
                        out.push(it);
                    }
                }
            }
        }
    }
}

fn classify<S: Shape>(shape: &S, shape_bbox: &Rect, child_bbox: &Rect) -> CellState {
    if let Some(grid) = shape.template_grid() {
        grid.classify_region(child_bbox)
    } else if child_bbox.intersects(shape_bbox) {
        CellState::Maybe
    } else {
        CellState::Out
    }
}
