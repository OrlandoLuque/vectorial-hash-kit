//! Shared, window-free simulation core for the critters demos.
//!
//! The graphical binary (`critters`) renders it with macroquad; the
//! `critters_headless` binary runs it at full CPU speed and reports
//! per-operation statistics. Everything in [`sim`] is deterministic for a
//! given seed and free of any graphics dependency.

pub mod sim;
