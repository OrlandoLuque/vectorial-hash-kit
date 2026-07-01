//! Template generation: polygon × scale × angle vs. grid intersections.
//!
//! Port of the PHP project `multiDimensionalIndexTemplateCreation`.

// This crate is dense 2D-grid code: the template matrices are indexed by
// explicit `(x, y)` in nested `for x in 0..w { for y in 0..h { m[x][y] } }`
// loops, where the row index is the actual coordinate used elsewhere in the
// body. Rewriting those into `enumerate`/`zip` reads worse and obscures the
// coordinate maths, so the range-loop and (grid-of-grids) type-complexity lints
// are allowed crate-wide rather than contorted at each site.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

pub mod vertex;
pub mod intersector;
pub mod polygon;
pub mod matrix;
pub mod templates;
pub mod task;
pub mod comparison_test;
pub mod adapter;
pub mod bank;
pub mod fingerprint;

#[cfg(feature = "redis-store")]
pub mod redis_store;

pub use adapter::{
    apply_inverse_op, decode_binary, matrix_to_template_grid, DecodeError, TemplateIndex,
    TemplateKey,
};
pub use bank::{FigureKey, TemplateBank};
