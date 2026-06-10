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
pub mod adapter;
pub mod bank;

#[cfg(feature = "redis-store")]
pub mod redis_store;

pub use adapter::{
    apply_inverse_op, decode_binary, matrix_to_template_grid, DecodeError, TemplateIndex,
    TemplateKey,
};
pub use bank::{FigureKey, TemplateBank};
