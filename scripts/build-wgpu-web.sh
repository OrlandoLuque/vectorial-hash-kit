#!/usr/bin/env bash
# Build the wgpu demos for the WEB (WebGPU via wasm-bindgen) → docs/<dir>/ + their
# docs/<bin>.html shells. WebGPU-only (the GPU skinning needs storage buffers WebGL2
# lacks), so they run in a recent Chrome / Edge.
#
# Needs: `rustup target add wasm32-unknown-unknown` and `wasm-bindgen-cli` whose
# version MATCHES the `wasm-bindgen` crate pin in crates/vectorial-hash-demos/Cargo.toml
# (currently =0.2.126 — `cargo install wasm-bindgen-cli --version 0.2.126`).
#
# Usage:
#   scripts/build-wgpu-web.sh                        # every wgpu demo
#   scripts/build-wgpu-web.sh horde_wgpu fluid_wgpu  # just these
#
# This used to build `siege_wgpu` and nothing else, while six other demos were
# published by hand — which is exactly how one of them ended up still serving old
# code after its Rust changed. If a demo is on the site, it belongs in this list.
set -euo pipefail
cd "$(dirname "$0")/.."

# bin -> output directory under docs/ (must match the <bin>.html shell's import path)
declare -A OUT=(
  [siege_wgpu]=wgpu
  [horde_wgpu]=horde
  [formations_wgpu]=formations
  [fluid_wgpu]=fluid
  [pointcloud_wgpu]=pointcloud
  [stealth_wgpu]=stealth
  [gpu_storm]=gpu
)

BINS=("$@")
if [ ${#BINS[@]} -eq 0 ]; then BINS=("${!OUT[@]}"); fi

for bin in "${BINS[@]}"; do
  dir="${OUT[$bin]:-}"
  if [ -z "$dir" ]; then
    echo "unknown wgpu demo '$bin' — known: ${!OUT[*]}" >&2
    exit 2
  fi
  echo "==> $bin -> docs/$dir"
  cargo build -p vectorial-hash-demos --bin "$bin" \
    --target wasm32-unknown-unknown --features web-wgpu --release
  wasm-bindgen "target/wasm32-unknown-unknown/release/$bin.wasm" \
    --out-dir "docs/$dir" --target web --no-typescript
done

echo
echo "done. Commit docs/ and push to deploy (Pages serves main:/docs)."
