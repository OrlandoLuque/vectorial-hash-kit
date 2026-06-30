#!/usr/bin/env bash
# Build `siege_wgpu` for the WEB (WebGPU via wasm-bindgen) → docs/wgpu/ + the
# docs/siege_wgpu.html shell. WebGPU-only (the GPU skinning needs storage buffers
# WebGL2 lacks), so it runs in a recent Chrome / Edge.
#
# Needs: `rustup target add wasm32-unknown-unknown` and `wasm-bindgen-cli` whose
# version MATCHES the `wasm-bindgen` crate pin in crates/vectorial-hash-demos/Cargo.toml
# (currently =0.2.126 — `cargo install wasm-bindgen-cli --version 0.2.126`).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -p vectorial-hash-demos --bin siege_wgpu \
  --target wasm32-unknown-unknown --features web-wgpu --release
wasm-bindgen target/wasm32-unknown-unknown/release/siege_wgpu.wasm \
  --out-dir docs/wgpu --target web --no-typescript
echo "done -> docs/wgpu/{siege_wgpu.js,siege_wgpu_bg.wasm} + docs/siege_wgpu.html (commit + push to deploy)"
