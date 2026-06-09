//! Template generation: polygon × scale × angle vs. grid intersections.
//!
//! Port of the PHP project `multiDimensionalIndexTemplateCreation`.

pub mod vertex;
pub mod intersector;
pub mod polygon;
pub mod matrix;
pub mod templates;
pub mod task;
pub mod comparison_test;

#[cfg(feature = "redis-store")]
pub mod redis_store;
