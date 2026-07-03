//! Shared core for the critters demos.
//!
//! The graphical binary (`critters`) renders it with macroquad; the
//! `critters_headless` binary runs it at full CPU speed and reports
//! per-operation statistics. Everything in [`sim`] is deterministic for a
//! given seed and free of any graphics dependency.
//!
//! [`instanced3d`] is the only graphics-bearing module here: a GPU-instanced
//! point renderer (raw miniquad) used by the `critters3d` demo.

// The macroquad/miniquad-bearing modules are excluded from the `web-wgpu` build
// so the wgpu wasm binary links no macroquad — otherwise miniquad's `env`
// file-loading imports (fs_get_buffer_size / fs_take_buffer) leak in and break
// the wasm-bindgen module load. `model` + `siege_sim` are macroquad-free (model
// uses glam directly), so the wgpu binary uses only those.
#[cfg(not(feature = "web-wgpu"))]
pub mod instanced3d;
pub mod model;
#[cfg(not(feature = "web-wgpu"))]
pub mod sim;
/// Shared, graphics-free simulation for the `siege` demo — used by both the
/// macroquad and wgpu binaries so the two renderers can't drift apart.
pub mod siege_sim;
/// Shared, graphics-free simulation for the `horde` demo (They Are
/// Billions-style zombie assault; wgpu renderer) — see `docs/HORDE_DESIGN.md`.
pub mod horde_sim;
#[cfg(not(feature = "web-wgpu"))]
pub mod time;
