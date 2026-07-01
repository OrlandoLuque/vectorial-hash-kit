#!/usr/bin/env bash
# Rebuild the WebAssembly demos and drop them into docs/ for GitHub Pages.
#
# One-time setup:  rustup target add wasm32-unknown-unknown
# Pages is served from the `main` branch, `/docs` folder (Settings -> Pages).
# The wasm linker flags live in .cargo/config.toml; the JS glue (mq_js_bundle.js)
# ships with macroquad and is already committed in docs/ (re-copy it only when
# you bump the macroquad version).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --target wasm32-unknown-unknown -p vectorial-hash-demos \
  --bin critters --bin critters3d --bin siege --release

cp target/wasm32-unknown-unknown/release/critters.wasm   docs/critters.wasm
cp target/wasm32-unknown-unknown/release/critters3d.wasm docs/critters3d.wasm
cp target/wasm32-unknown-unknown/release/siege.wasm      docs/siege.wasm

# The siege demo fetches its models at runtime (they're NOT baked into the wasm —
# keeps siege.wasm ~1.6 MB instead of ~9.5). They must be served from docs/models/;
# copy exactly the set `siege_sim::SIEGE_MODEL_FILES` lists.
mkdir -p docs/models
for m in anne sharky pirate_captain witch henry zombie skeleton_a skeleton_sword \
         slime bat dragon cannon horse castle; do
  cp "crates/vectorial-hash-demos/assets/siege/models/$m.glb" "docs/models/$m.glb"
done

echo "done -> docs/{critters,critters3d,siege}.wasm + docs/models/*.glb  (commit + push to deploy)"
# To refresh the JS bundle after a macroquad upgrade:
#   cp ~/.cargo/registry/src/*/macroquad-*/js/mq_js_bundle.js docs/mq_js_bundle.js
