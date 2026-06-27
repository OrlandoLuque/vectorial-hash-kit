//! Shared core for the critters demos.
//!
//! The graphical binary (`critters`) renders it with macroquad; the
//! `critters_headless` binary runs it at full CPU speed and reports
//! per-operation statistics. Everything in [`sim`] is deterministic for a
//! given seed and free of any graphics dependency.
//!
//! [`instanced3d`] is the only graphics-bearing module here: a GPU-instanced
//! point renderer (raw miniquad) used by the `critters3d` demo.

pub mod instanced3d;
pub mod model;
pub mod sim;
pub mod time;
