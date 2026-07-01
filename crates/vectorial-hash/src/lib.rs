//! Index and hash algorithms for vectorial spaces.
//!
//! The core type is [`Tree`], a binary-split spatial index where items live in
//! leaf cells and cells subdivide when they overflow `item_limit`. Queries are
//! answered by [`Tree::cull`], which walks the tree against a [`Shape`].
//!
//! ```
//! use vectorial_hash::{Point, Rect, Tree, Positioned, Shape};
//!
//! #[derive(Clone, Copy)]
//! struct Pt(Point);
//! impl Positioned for Pt {
//!     fn position(&self) -> Point { self.0 }
//! }
//!
//! struct Box2 { rect: Rect }
//! impl Shape for Box2 {
//!     fn bounding_box(&self) -> Rect { self.rect }
//!     fn contains_point(&self, p: Point) -> bool { self.rect.contains(p) }
//! }
//!
//! let mut tree = Tree::<Pt>::new(Rect::new(0.0, 0.0, 100.0, 100.0), 2);
//! tree.insert(Pt(Point::new(10.0, 10.0)));
//! tree.insert(Pt(Point::new(50.0, 50.0)));
//! tree.insert(Pt(Point::new(90.0, 90.0)));
//!
//! let hits = tree.cull(&Box2 { rect: Rect::new(0.0, 0.0, 60.0, 60.0) });
//! assert_eq!(hits.len(), 2);
//! ```

pub mod geom;
mod serde_io;
pub mod template;
pub mod tree;
pub mod culling;
pub mod quadtree;
pub mod itree;
pub mod tree3;
pub mod octree3;
pub mod morton3;
pub mod morton;

pub use geom::{Point, Rect};
pub use template::{CellState, PlacedTemplate, TemplateGrid};
pub use tree::{Node, NodeId, Positioned, RaycastOut, Side, Tree, UpdateStrategy};
pub use culling::{Capsule, Circle, Shape, WalkNeighbors};
pub use quadtree::{QNode, QNodeId, QuadTree};
pub use itree::{INode, INodeId, IPoint, IPositioned, IRect, IUpdateStrategy, IntegerTree};
pub use tree3::{Aabb, ItemRef, Node3, Node3Id, Point3, Polyhedron3, Positioned3, Segment3, Shape3, Sphere3, Tree3, VoxelRaster};
pub use octree3::{ONode, ONodeId, Octree3};
pub use morton3::MortonGrid3;
pub use morton::MortonGrid;
