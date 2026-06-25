//! A tiny drop-in `Instant` that works on **native and WebAssembly**.
//!
//! `std::time::Instant::now()` panics on `wasm32-unknown-unknown` (no system
//! clock). macroquad's wasm backend doesn't use `wasm-bindgen`, so the `instant`
//! crate doesn't fit either. miniquad — which macroquad re-exports — has a
//! cross-platform `date::now()` (SystemTime on native, `Date.now()` via its own
//! JS glue on wasm) that needs no window/context. We wrap it so the call sites
//! keep using `Instant::now()` / `.elapsed().as_secs_f64()` unchanged: only the
//! `use` line changes from `std::time::Instant` to `vectorial_hash_demos::time::Instant`.
//!
//! Resolution is whatever the platform clock gives (milliseconds in a browser),
//! coarser than `std::time::Instant` — fine for the demos' on-screen readouts.

use std::time::Duration;

#[derive(Copy, Clone, Debug)]
pub struct Instant(f64);

impl Instant {
    #[inline]
    pub fn now() -> Self {
        Instant(macroquad::miniquad::date::now())
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        Duration::from_secs_f64((macroquad::miniquad::date::now() - self.0).max(0.0))
    }
}
