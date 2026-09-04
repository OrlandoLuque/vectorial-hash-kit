#!/usr/bin/env bash
# Is anything on the published site older than the code it was built from?
#
# On 2026-08-03 the mobile overlay was fixed and shipped as static HTML/JS while the two demos
# whose RUST had changed kept serving the previous day's wasm — so on a phone the slime still
# faced backwards and the walls still broke at every tower. Nothing was broken; the deploy was
# simply incomplete, and nothing could tell.
#
# This compares each published artefact's mtime against the newest source it depends on. It is
# a *staleness* check, not a correctness one: a false alarm costs one rebuild, a miss costs a
# user reporting a bug that was fixed days ago.
#
#   scripts/check-web-fresh.sh          # report, exit 1 if anything is stale
#   scripts/check-web-fresh.sh --list   # just print the table
#
# Note the deliberate limitation: mtimes are not content hashes, and a fresh `git clone` gives
# every file the same checkout time. So this is meant for a working copy and for the commit
# that publishes — in CI it is advisory (see ci.yml), because there the timestamps mean nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

# published artefact : the sources whose change should invalidate it
declare -A DEPS=(
  [docs/critters.wasm]="crates/vectorial-hash-demos/src/bin/critters.rs crates/vectorial-hash-demos/src/sim.rs"
  [docs/critters3d.wasm]="crates/vectorial-hash-demos/src/bin/critters3d.rs crates/vectorial-hash-demos/src/sim3.rs"
  [docs/siege.wasm]="crates/vectorial-hash-demos/src/bin/siege.rs crates/vectorial-hash-demos/src/siege_sim.rs"
  [docs/wgpu/siege_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/siege_wgpu.rs crates/vectorial-hash-demos/src/siege_sim.rs"
  [docs/horde/horde_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/horde_wgpu.rs crates/vectorial-hash-demos/src/horde_sim.rs"
  [docs/adaptive/adaptive_lab_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/adaptive_lab_wgpu.rs crates/vectorial-hash-demos/src/adaptive_lab.rs crates/vectorial-hash-demos/src/ui2d.rs"
  [docs/formations/formations_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/formations_wgpu.rs crates/vectorial-hash-demos/src/formations_sim.rs"
  [docs/fluid/fluid_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/fluid_wgpu.rs"
  [docs/pointcloud/pointcloud_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/pointcloud_wgpu.rs"
  [docs/stealth/stealth_wgpu_bg.wasm]="crates/vectorial-hash-demos/src/bin/stealth_wgpu.rs"
  [docs/gpu/gpu_storm_bg.wasm]="crates/vectorial-hash-demos/src/bin/gpu_storm.rs"
)

# Every demo also links the library, so a library change can stale all of them.
LIB_NEWEST=$(find crates/vectorial-hash/src -name '*.rs' -newer /dev/null -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 || true)
LIB_T=${LIB_NEWEST%% *}

list_only=0
[ "${1:-}" = "--list" ] && list_only=1

stale=0
printf "%-42s %-10s %s\n" "published" "state" "newest source"
for art in "${!DEPS[@]}"; do
  if [ ! -f "$art" ]; then
    printf "%-42s %-10s %s\n" "$art" "MISSING" "-"
    stale=1; continue
  fi
  at=$(stat -c %Y "$art")
  newest=""; newest_t=0
  for src in ${DEPS[$art]}; do
    [ -f "$src" ] || continue
    st=$(stat -c %Y "$src")
    if [ "$st" -gt "$newest_t" ]; then newest_t=$st; newest=$src; fi
  done
  # the library counts too
  if [ -n "${LIB_T:-}" ] && [ "${LIB_T%.*}" -gt "$newest_t" ] 2>/dev/null; then
    newest_t=${LIB_T%.*}; newest="crates/vectorial-hash/src (library)"
  fi
  if [ "$at" -lt "$newest_t" ]; then
    printf "%-42s %-10s %s\n" "$art" "STALE" "$newest"
    stale=1
  else
    printf "%-42s %-10s %s\n" "$art" "ok" "$newest"
  fi
done

if [ "$list_only" = "1" ]; then exit 0; fi
if [ "$stale" = "1" ]; then
  echo
  echo "Something on the site is older than the code it came from."
  echo "  scripts/build-web.sh                 # the macroquad demos"
  echo "  scripts/build-wgpu-web.sh [bin...]   # the wgpu demos"
  echo "then commit docs/ — Pages serves main:/docs, so an unbuilt fix is an unshipped fix."
  exit 1
fi
echo
echo "every published demo is at least as new as its source."
