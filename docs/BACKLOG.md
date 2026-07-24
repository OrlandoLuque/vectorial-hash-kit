# Backlog (to triage)

Future work queued for review. Nothing here is committed scope — prune,
reprioritise, or drop freely. Items graduate into the per-crate roadmaps
(e.g. `crates/vectorial-hash/README.md`) once picked up.

## ★ 2026-07-24 (cooperative night) — landed + follow-ups

**Landed (all on `main`, CI green):** horde **awake-front cap + stagger** (100k
playable — was a wave-1 instant loss), **garrison flattened** (higher pop now
*harder* via reserve depth, not trivially easier — the inversion fixed), **forests
= line-of-sight cover** (towers can't shoot through canopy), **`LinearOctree3`** (a
new adaptive pointer-free 3D linear-octree library structure, benched + THREE_D.md)
and it's **selectable in `critters3d`** (`M`). Plus a pre-existing templates clippy
fix (`for_kv_map`) to restore green CI. Full write-up in the local `CLAUDE.md`.

**Follow-ups queued (autonomous-safe):**
- `LinearOctree3::serialize`/`deserialize` (kit convention — Tree3 + MortonGrid3
  have it; Octree3 doesn't yet either, so low priority).
- Add `LinearOctree3` to the **headless decision-map sweep**
  (`critters3d_headless --sweep`) so it appears in the *measured* decision map, not
  just the interactive demo.
- Horde: a HUD line for the awake-front cap / reserve (so the cap is legible on
  screen); maybe a slider like the pop/thread ones.

**Eyes-dependent (do NOT silently finish — need the user):**
- `building_tweak` per-model scale/orientation for the Quaternius RTS buildings in
  `horde_wgpu` — tune live.
- **Mobile UI** on a real phone: confirm the synthetic-`KeyboardEvent` overlay
  (`docs/mobile-controls.js`) actually drives winit/miniquad; fix in `send()` if not.

## ⚠️ MORNING REMINDER (user asked, 2026-07-11)

**Test the battle fix in the browser** — `formations_wgpu` was black on web
(canvas had no CSS size + no `ControlFlow::Poll`); fixed + republished
(`docs/formations/`). Hard-refresh (`Ctrl+Shift+R`) the Pages site / local serve
and confirm pirates-vs-undead renders. *(And remove the `ClaudeNightSuspend`
scheduled task if it's still around.)*

## ★ 2026-07-11 (morning, with the user) — landed + new ask

**Landed this session (all pushed, CI green):**
- ~~gpu_storm / gpu_lbvh_demo **web crash**~~ — WebGPU `poll(Wait)` doesn't block →
  the synchronous timestamp readback threw every frame; gated to native.
- ~~**Demo screenshots → visual index**~~ (was §9) — `SHOT` capture mode in every
  demo (macroquad self-export via `get_screen_data`; wgpu title-freeze + `ffmpeg
  gdigrab`), 8 JPEG thumbnails wired into `docs/index.html`. Plus `$SIEGE_POP`
  boot override → recaptured siege/formations at max population.
- ~~horde_wgpu **web crash**~~ — impostor `bb-shader` had an integral vertex output
  without `@interpolate(flat)` (Tint-strict; naga tolerated it).
- ~~3D critters **replay overlay**~~ — the vision-cull sphere/sight-lines are now
  recorded into the history `Frame`, so they survive replay scrubbing.

### ★ NEW ASK (user, 2026-07-11): auto-boot every demo in its highest-FPS mode

With everything now measured (keep-index vs rebuild, voxel vs smooth terrain,
impostors vs skinned LOD, boids maths vs table, Tree3 vs Morton, thread counts,
frustum cull `K`, and — once done — decision-buckets), **pick the fastest default
boot configuration for each demo** and make that the out-of-the-box mode. Method:
**launch each demo for ~X seconds per candidate mode, print/collect the FPS, and
compare** — then set the winner as the default (env still overrides).
- Reuse the existing smoke/telemetry hooks that already print fps: `SIEGE_MAX_
  FRAMES`, `HORDE_MAX_FRAMES` (prints fps/slp/act per ~second), `CRITTERS3D_MAX_
  FRAMES`, plus the offscreen benches (`siege_cpu_bench`, `horde_bench`). Extend
  where a knob isn't yet togglable by env.
- Candidate axes per demo: index (Tree3 / Octree3 / Morton / projection), terrain
  (voxel/smooth), LOD/impostor distance, thread count (native), decision-bucket
  rate (horde), frustum cull on/off, population.
- Deliverable: a short FPS-comparison table (per demo, per mode) + the chosen
  defaults committed, noted in the relevant demo docs. Keep it honest — report
  the measured deltas, don't guess. **Native vs web differ** (no threads/rayon on
  web; WebGL1 vs wgpu) → pick the best mode *per target*, not one global default.

**Measured so far (2026-07-17 night):**
- **horde decide rate** → 15 Hz→7.5 Hz default, **+22 %** at 53k active (item 1 above).
- **critters3d index** → the headless decision map (`--sweep`) says `binR`
  (Tree3 + `ItemRef`) wins **maintain 15/16** configs and cull 9/16 (Morton only
  wins cull in sparse 1024³ worlds). The demo already boots Binary3 + ItemRef →
  **current default validated, no change.** And at 10k pop the index is ~0.1–0.3
  ms of a 16 ms frame → the demo is **render-bound**; its FPS lever is the render
  path (already instanced = default), not the index.
- **siege** → its defaults are *intentional*, not accidental: **voxel terrain**
  is the requested headline look (smooth is cheaper geometry but the user asked
  for voxel — an aesthetic default, not an FPS one; PERF_NOTES §"Voxel vs
  smooth" has the delta), the **thread slider boots at all-cores** (fastest), and
  the index is the **keep-index Tree3** (validated fastest, same as critters3d).
  Nothing to flip.
- **conclusion:** across the demos the boot defaults are already at/near optimal
  — the one concrete win was **horde decide-buckets** (done, +22 %). Honest
  outcome of the "auto-boot fastest mode" ask: *measured, mostly validated.*
  (Remaining low-value knobs: horde frustum-cull `K` default — a quality/latency
  call, left as a toggle.) **DONE enough to close.**

Note: **`Tree::bulk_load` / `bulk_load_par` already exist** (2D) with tests — the
old "2D Tree bulk-load (3D done)" backlog line was stale.

## ★ OVERNIGHT — round 5 (user, 2026-07-11): autonomous, safety-net 02:00

Full autonomous mandate again (commit + doc per task; suspend when done or at the
safety net). Ordered — verifiable/high-value first, `[eyes]` last:

1. **Publish `gpu_storm` + `gpu_lbvh_demo` to the web** — now cheap: the web
   gotcha is known (canvas CSS size + `Poll`, just fixed for formations) + the
   `build-wgpu-web.sh` pattern. Add a `requestDevice` limit shim if Chrome balks.
2. **`gpu_storm` polish**: colour by **local density** (smooth — collision mode
   is stuck blue because hard contacts are sparse) + **on-screen GPU meter bars**
   (a UI-quad overlay, not just the title).
3. **GPU line-of-sight shader** — the clean static-occluder offload (CPU measured;
   `gpu_lbvh_demo` traversal with the leaf test sphere→segment).
4. **Full 3-projection vs expensive narrowphase** (THREE_D open item) — measure.
5. **Nudge-free 3D walk** (Samet) — careful (the subset test can't catch under-collection).
6. **Demos**: `MortonGrid` (2D) in the `critters` toggle; batch sight-lines to meshes.
7. ~~**GPU-side LBVH build**~~ — **DONE** (`examples/gpu_lbvh_build_bench`,
   2026-07-17): the WHOLE build is GPU-resident — Morton → stable key-value radix
   → **Karras** hierarchy → atomic bottom-up **AABB refit** — verified by
   traversing the GPU-built tree on the CPU vs brute force (⇒ hierarchy + AABBs
   correct). **1 M-point BVH rebuilds in ~8.4 ms/frame** (262k 1.69 · 4M 36.3, all
   verified). The radix itself is ~2× past `sort_unstable` (16-way scan; the
   serial-scan first cut was only at parity; the bitonic was ~2× *slower*). Open
   headroom: Onesweep multi-workgroup scan; compressed wide nodes; wiring the
   build into a moving demo to push the crossover. See GPU.md + OPTIMIZATION_RESEARCH.md.
8. `[eyes]` (leave for the user): colour-by-leaf-depth viz · 2D k-NN/LoS demo ·
   Quaternius RTS wall models · classic-Reynolds boids · balance · web touch+HUD.

**LANDED (2026-07-11 night, all pushed, CI green):**
- ~~**(1) Web publish**~~ — `gpu_lbvh_demo` + `gpu_storm` now build for wasm32
  (web-wgpu), wasm-bindgen'd to `docs/gpu/` with HTML shells (requestDevice shim)
  + index cards. Web gotcha solved (canvas CSS size + `Poll`); gpu_storm uses a
  smaller grid on web (96³) to fit WebGPU's 128 MiB buffer limit. `[eyes: confirm
  in a WebGPU browser.]`
- ~~**(2) gpu_storm colour + bars**~~ — colour by local density (2.6r) so collision
  filaments light up (not stuck blue); on-screen per-phase GPU-load bars
  (grid/collide/integrate/render, scaled to a 60 fps budget). Same bars on
  `gpu_lbvh_demo` (query+render). Native builds re-verified after the web port.
- ~~**(3) GPU line-of-sight**~~ — `gpu_visibility_bench`: segment-vs-AABB LBVH
  traversal over STATIC occluders, verified == CPU `segment_hit` (Δ 0), ~1380× the
  serial CPU with a one-time 1 ms build (the *clean* GPU offload). Documented in
  PERF_NOTES + the local occlusion notes.
- ~~**(7) GPU-side build — attempted, honest negative**~~ — `gpu_sort_bench`: a GPU
  bitonic sort of Morton codes (verified == CPU sort). But naive bitonic is
  **slower** than the CPU sort (log² work + dispatch-per-pass). **The GPU-side LBVH
  build needs a RADIX sort** (or a workgroup-shared-memory bitonic) — a focused
  session, not a blind attempt. Blocker identified.

**New asks (user, 2026-07-11 — queued):**
- **(9) Demo screenshots → a more visual index.** Launch each demo, wait ~20 s,
  capture, use the images as thumbnails in `docs/index.html`. *Blocker: no CLI
  screenshot on Windows + PowerShell deny-listed → the clean path is an offscreen
  "render 1 frame to PNG" mode in the wgpu demos (needs an image-encode dep).*
- ~~**(10) Research more optimisation strategies**~~ — **DONE** via `deep-research`
  (23 sources, 25 claims verified, 17 confirmed) → `docs/OPTIMIZATION_RESEARCH.md`.
  Headline: the kit is near-SOTA; concrete next steps = **Onesweep GPU radix sort**
  (unblocks the GPU-side build; keep 32-bit Morton), **cache-oblivious arena
  layout** (algorithm-free, high upside), **compressed wide-BVH nodes**,
  **refit+rotations** if a persistent dynamic BVH is added. Our LBVH + keep-index
  choices *validated* by the literature. *(The run hit the monthly spend limit
  mid-verification — a few CPU/SIMD claims left unconfirmed; noted in the doc.)*
- **(9) Demo screenshots** — still queued (eyes + offscreen-render-to-PNG mechanism).

**Night ended on the monthly SPEND LIMIT** (the safety-net suspend's raison d'être).
13 commits, all pushed, CI green. Suspended.

## ★ NEXT DEMO (user, 2026-07-07): a GPU-RESIDENT collision storm

Agreed after the "why don't the sims benefit from GPU LBVH?" discussion. The
answer: the game sims are branchy-`decide`-bound (GPU-hostile) and pay a per-
frame CPU↔GPU round-trip; the real GPU win is a sim where **the whole hot loop
is GPU-resident** — a uniform-rule sim with no round-trip. So:

1. **`gpu_storm` — GPU-resident collision storm** (the vehicle for "how much MORE
   can we accelerate?"). Whole pipeline in compute shaders: Morton encode → LBVH
   build → broad-phase pair query → collision resolution → integration, all
   resident on the GPU, only the framebuffer read back. The **switch compares
   *whole sim on GPU* vs *on CPU*** (not just the query) + a GPU-load meter. Most
   visual: 100k–1M particles, collisions flash. Scales to 1M with the GPU-side
   build (item 3). Native wgpu-direct (no macroquad twin, per the demo policy).
2. **Influence-field mode** (a toggle on `gpu_storm`, ~90% shared code): thousands
   of moving emitters, each lighting everything within a radius — a large-scale
   perception / proximity-glow visual. Cheap add once (1) exists.
3. **GPU-side LBVH build** (parallel radix sort + Karras in WGSL) — the enabler:
   erases the per-frame rebuild tax measured in `gpu_spatial_bench`, pushing the
   moving-broad-phase crossover past 1M. Also unlocks (1) at 1M+.

Companion (measured tonight, separate track): **GPU line-of-sight shader** — the
`gpu_lbvh_demo` traversal with the leaf test swapped point-in-sphere → segment-
vs-AABB, over a STATIC occluder BVH (build once, no rebuild → the *clean* GPU
case). CPU baseline + `Polyhedron3::segment_hit` + `visibility_cull_bench` landed
2026-07-07 (CPU-parallel holds ~110–216k viewer×target pairs/frame @60 Hz).

## ★ OVERNIGHT — round 4 (user, 2026-07-07): "do it all, as much as possible"

Autonomous night. Full mandate: complete as much as possible, commit + doc each,
don't leave anything doable for tomorrow. Ordered (autonomously-verifiable first;
`[eyes]` = wants a human glance, lower priority). Safety-net suspend armed 02:00.

1. **`gpu_storm`** — GPU-resident collision storm (see NEXT DEMO §1) + **§2
   influence-field mode** + RustRover config + web publish.
2. **GPU-side LBVH build** (radix sort + Karras, WGSL) — enables (1) at 1M; folds
   into `gpu_spatial_bench` (re-measure the moving crossover with a GPU build).
3. **GPU line-of-sight shader** — the clean static-occluder offload (companion above).
4. **Nudge-free 3D walk** (Samet ascend-to-LCA) — raycast §A.2; removes the
   `locate`+epsilon nudge on `Tree3`/`Octree3` DDA. Brute-gated (⊆ exact capsule).
5. **Morton multi-level linear octree** (§C.8) — coarse occupancy levels so a
   big-radius cull skips empty blocks (2D + 3D). Brute-gated.
6. **3D dilation (Minkowski)** — agent body radius for 3D (the 2D
   `inflated_convex`/`within_dilation` analogue).
7. **Full 3-projection vs expensive narrowphase** (Index/algorithms) — the
   `THREE_D.md` open item: many-faced `Polyhedron3`, find where tight broadphase pays.
8. **Docs**: sync the lib README (raycast, MortonGrid 2D, `Circle`, `segment_hit`,
   the seven structures) + a performance guide (`target-cpu=native`, SIMD, threads).
9. **Demos**: colour-by-leaf-depth tree viz; 2D k-NN / line-of-sight demo; add
   `MortonGrid` (2D) to the `critters` toggle; batch sight-lines/effects to meshes.
10. **Siege/horde**: Quaternius RTS wall/tower/house models (horde #6, pack on
    disk); `siege_wgpu` web `requestDevice` shim; bridge decks + path-to-bridge AI.
11. `[eyes]` classic-Reynolds boids flag · balance pass · web touch controls + HUD.
12. **Held (user)**: crates.io publish · template not-found diagnostics rethink ·
    threading tutorial — leave for the user unless everything else is done.

**LANDED so far (2026-07-07 night, all pushed, CI green):**
- ~~**(1) `gpu_storm` milestone 1**~~ — GPU-resident collision storm: clear→build
  grid→collide (DEM)→integrate, four compute passes, pos/vel resident; CPU/GPU
  switch + meter; smoke-tested (0 out-of-bounds/NaN). **GPU sim ~0.33 ms vs CPU
  grid ~16.4 ms at 150k (~50×)**. RustRover config added. *Remaining: influence-
  field mode + web publish + the GPU-side build (item 3) to push it to 1M.*
- ~~**(5) Morton multi-level**~~ — `MortonGrid3::cull_layered` (coarse occupancy
  skip, Z-order prefix = hierarchy), brute-gated (== cull == brute on a
  clumps-in-a-void world). Non-invasive (separate method, no build-cost hit).
- ~~**(6) 3D dilation**~~ — `Polyhedron3::inflated(r)` (Minkowski-flavoured face
  offset), brute-gated vs exact L2 box distance. + `segment_hit` line-of-sight
  (occlusion) + `visibility_cull_bench` landed earlier tonight.
- ~~**(1b) `gpu_storm` influence-field mode**~~ — `F` toggles collision ↔ a
  proximity-glow/perception field (bigger radius, multi-ring, no forces); CPU
  path honours it too. GPU ~3.35 ms vs CPU ~128 ms at 150k (~38×). *Remaining:
  web publish + the GPU-side build (item 3) for 1M.*
- ~~**(6b) Octree3 boundary-crossing event**~~ — `update_ref_tracked`/`OCrossing`
  + `ref_leaf` (parity with Tree3's `Crossing`), brute-gated.
- ~~**(8) README sync + `docs/PERFORMANCE.md`**~~ — seven structures / raycast /
  knn / ItemRef / shapes / advisor; the practical speed guide (build flags,
  keep-index, threads, analytic-vs-raster, GPU) is written + cross-linked.
- ~~**(5b) Morton multi-level 2D**~~ — `MortonGrid::cull_layered` (the 2D twin;
  2D+3D coverage complete), brute-gated + fuzzed. `layered_cull_bench` **measures**
  it: 1.2× (≈uniform) → 42.5× (sparse 3%) on a big query over clumps-in-a-void.
- ~~**(8b) docs**~~ — demos README gains the GPU section (gpu_lbvh_demo /
  gpu_storm / gpu_spatial_bench). Proptest campaign fuzzes `cull_layered == cull`.
- Sanity: full lib suite (104 tests) + parallel clippy gate + all-targets build
  green after the whole night.

**Night total: 15 commits, all pushed, CI green.** Library: `segment_hit` (LoS),
`inflated` (3D dilation), `cull_layered` (2D+3D multi-level), Octree3 crossing
event, README/PERFORMANCE docs. Demos: `gpu_storm` (collision + influence,
switch, meter). Benches: `visibility_cull_bench`, `layered_cull_bench`, the
parametrised `gpu_spatial_bench`.

**Still open — WHY each waits (not blind-doable well tonight):**
- **GPU-side LBVH build** (radix sort + Karras, WGSL) — a focused session; a blind
  GPU radix sort is high-risk to get right + verify. Highest-value next item
  (unlocks `gpu_storm` at 1M + the moving broad-phase crossover).
- **`gpu_storm` / `siege_wgpu` web publish** — needs the `requestDevice` limit
  shim + a *browser* to verify; can't confirm WebGPU-compute works headless.
- **nudge-free 3D walk** (Samet, §A.2) — the "subset of capsule" test can't catch
  a subtle under-collection, so this one needs careful non-blind verification.
- **Demos §9 + `[eyes]` §11** (leaf-depth viz, 2D k-NN/LoS demo, balance,
  classic-Reynolds) — **need the user's eyes** to judge.
- **3-projection vs narrowphase** (§Index) · **siege/horde §10** (Quaternius
  models, bridges) — straightforward, just not reached tonight.

## Overnight queue — round 3 (set 2026-06-26)

The big **ray-cast thread graduated** (DDA leaf-walk 2D + 3D, capsule shapes
with analytic `classify_box`/`classify_aabb`, `raycast_first` early-exit, exact
thick band, ropes-maintenance ledger, SoA batch narrowphase, 2D `MortonGrid`,
3D `Tree3::raycast_dda`, and 2D + 3D decision maps) — all in `docs/RAYCAST.md`.
What's left, grouped. `[review]` = needs a human visual glance; rest is
autonomously verifiable. User picked **"Todo (A–E)"** + the headline below.

**ORDER (user, 2026-06-27): multithreading FIRST, then `siege`, then the rest.**

**★ NEXT DEMOS (user, 2026-07-02): two new flagship demos, "las dos" — wgpu-direct.**
- **HORDE — BUILT through the render scale pass (2026-07-02/06)**: `horde_sim`
  (noise-wake culls, waves + manual `N` trigger, towers, flow field, breaches,
  infection, Commander/defenders/sorties via gates, works economy + breach
  recall, contact trampling — 17 brute-force-gated tests) + `horde_wgpu`
  (**impostor billboards**: startup-photographed atlas, 8 yaws × 3 elevation
  bands × walk/idle/death clips; skinned models only in the camera bubble;
  static carpet + append-only corpses; playtest-driven fixes). Headlines:
  **100k dormant = 0.42 ms/step (sim) and ~740 fps rendered**; wave-time FPS
  fixed by closing 3 feedback loops (carpet rebuilds, contact percolation,
  walking noise). Shipped doc: [HORDE.md](HORDE.md).
  **HORDE queue (priority order):**
  1. ~~**Decision buckets 4–8 Hz**~~ — **DONE + tuned to the 4–8 Hz target**
     (2026-07-17): far-from-walls actives re-decide every `decide_n` frames
     (staggered by id; cached vel walks between), full rate inside BASE_R+60.
     Was 15 Hz (`4`); swept `$HORDE_DECIDE_N` on the 100k mass assault (~53k
     active): 15 Hz 5.41 ms (185 fps) → **7.5 Hz (`8`, new default) 4.44 ms
     (225 fps, +22 %)** → 4 Hz 4.09 ms (244 fps, +32 %). Chose 8 — banks most of
     the win with coherent steering; 8→15 buys only +8 % for coarser paths.
     Table in [HORDE.md](HORDE.md#decision-buckets).
  2. Impostor **elevation-band blending** — only if the band switch is visible
     when flying the camera up/down (user to confirm).
  3. **Scenario presets** (mountain pass / river crossing / forest paths) —
     IN PROGRESS (2026-07-06 night): sim side landed (terrain_h per scenario,
     passability grid, flow field skips blocked cells = the horde's min-paths,
     A* sortie routing = the defenders', movement slide, reachable wave
     landings, `M` Tree3↔Morton toggle); renderer keys/visuals pending.
     - **PENDING DISCUSSION (user, next session): their own idea for the
       horde's minimum paths** — tonight's flow-field-over-blocked-cells is
       the baseline to compare against.
     - ~~**research how They Are Billions GENERATES its maps**~~ — **DONE**
       (2026-07-07): researched (TAB = seeded procedural, 6 themes, maps built
       from weighted blob patches of forest/rock/water as natural barriers;
       the walkable network is the residual space between blobs — gorges,
       pockets, chokepoints). None of Pass/River/Forest was equivalent (those
       are single hand-authored features or the *inverse* solid-woods layout),
       so added the **PATCHES** scenario: three independent value-noise masks
       → blob mosaic, a flood-from-CC connectivity pass (corridor to each big
       unreached pocket, small ones stay dead-end recovecos) for TAB's
       playability guarantee. `HORDE_SCENARIO=PATCHES`; new brute-force test.
  4. ~~**Molón round**~~ — **DONE** (2026-07-06): `L` night + ≤64 flickering
     torch point-lights in every shader (a fallen tower's light dies with
     it), blood-pool + kill-ring ground decals, trauma camera (rotational,
     ∝ trauma²; breaches/waves/run-end charge it), night clear colour.
  5. ~~**Tree3 ↔ Morton live toggle**~~ — **DONE** (2026-07-06): `M`, the
     `ZQuery` borrow enum over every query site; Morton = honest
     clear+reinsert per frame at levels=5 (levels=7 shredded the 68-wu y
     axis: one wake blast touched 270k cells). Morton==Tree==brute gated.
     (The 2D leg of "ambos conmutables" would need a projection layer —
     queued behind the TAB-map research.)
  6. ~~**Quaternius Ultimate Fantasy RTS** wall/tower/house models~~ — **WIRED
     (first pass, 2026-07-18) — needs the user's eyes for scale/orientation.**
     Extracted `Stone Wall`→`wall.glb`, `Castle Gate`→`gate.glb`, `Stone
     Tower`→`tower.glb`, `House`→`house.glb`, `Storage House`→`storehouse.glb`
     into `assets/siege/models/` (native `include_bytes`) **and** `docs/models/`
     (web fetch). `horde_wgpu` loads one static `GpuModel` per `SKind`
     (`BUILDING_FILES`, via `build_gpu_model` like the castle) and draws them in
     **per-kind instance ranges** from a shared buffer (replacing the boxes;
     CommandCenter still the castle, porter bundles still boxes). HP damage tints
     the model toward alarm-red (`tint.a = dmg`), rubble sinks + darkens; tower
     cannon rides on top. First-pass scales in **`building_tweak(SKind) ->
     (scale, yaw, y)`** — trivial to tune. Compiles + runs (no panic, 58 fps).
     **PENDING (needs eyes): tune `building_tweak` scale/yaw/y per model, then
     rebuild + republish the horde wasm.** (wasm NOT yet rebuilt — a blind visual
     publish would risk shipping wrong scales.)
  7. ~~**Multi-goal flow field**~~ — **DONE** (2026-07-07): the user's
     multi-source idea. `O` toggle seeds 0 at every live building, one flood
     (multi-source Dijkstra); re-routes to survivors as buildings fall
     (brute-force test). Measured: 31 goals ≈ the single-CC rebuild cost
     (one Dijkstra either way). Docs: HORDE.md § "The multi-goal flow field".
     Follow-ups (queued): make it the DEFAULT if it reads better in play;
     a per-building "who's nearest" tint; D\* Lite only if moving goals ever
     want a field.
  8. ~~**"Wake all" button + 100k-active FPS**~~ — **DONE** (2026-07-07): `A`
     key + ALL button (`wake_all`) rouse every sleeper at once. Measured:
     **100k ALL active = ~79 fps rendered** (RTX 4080 SUPER, no vsync) — the
     honest worst-case ceiling. `HORDE_WAKE_ALL=1` boots straight into it.
  9. ~~**GPU LBVH broad-phase — support + a switchable demo + measure the
     sims**~~ — **DONE** (2026-07-07): `gpu_lbvh_demo` (a live three-way switch
     CPU `Tree3` ↔ GPU brute ↔ **GPU LBVH** compute traversal, N moving points
     heat-mapped by neighbour count, GPU-timer meter in the title, `1/2/3`
     keys + `[ ]` point count) and a parametrised `gpu_spatial_bench`
     (GPU_N/M/R/CLUSTER + the per-frame **rebuild-vs-keep** verdict). Raw query
     kernel is ~393× the serial CPU cull, **but** — measured — for *moving* data
     the per-frame BVH rebuild eats it: vs the **parallel keep-index** the sims
     already run, GPU LBVH wins 1.66–4.5× at ≤100k yet **loses 1.30× at 1M**
     (the *N log N* rebuild overtakes the linear keep-maintain). **Verdict: sims
     keep the CPU parallel keep-index** (right for moving data; their bottleneck
     is `decide`, not the cull); GPU LBVH is for static / rebuild-anyway loads,
     and a GPU-side build would move the crossover past 1M. Full write-up:
     `PERF_NOTES.md` § "GPU LBVH broad-phase". RustRover config added.
- **FORMATIONS — BUILT (2026-07-06 night)**: `formations_sim` (graphics-free,
  11 brute-force-gated tests: k-NN melee pairing, sector flank/rear vs brute
  angles, the kill-roll curve, charge-corridor raycast, volleys with honest
  friendly fire, morale/chain routs, kept-index == brute, bit-identical
  replay) + `formations_wgpu` (GPU-skinned move/attack clips per (faction,
  kind), horses under cavalry/generals, banner poles+flags with state colour,
  arrow lines, flattened fallen, HUD bars/sliders; ~7k soldiers ≈ 100 fps).
  Docs: [FORMATIONS.md](FORMATIONS.md); design [FORMATIONS_DESIGN.md](FORMATIONS_DESIGN.md).
  Web published + release binary. Remaining polish queued: KayKit banner/
  weapon props, tactical terrain (hill charge bonus, forest cover), a
  formations-vs-horde crossover scenario.

**★ PHASE 0 — Multithreading (before the demo).** The `siege` demo is the heavy
parallel consumer (1k+ units, each doing read queries on the shared index per
frame). Lay the parallel infra first so the demo is multithreaded from the start:
- **Parallel per-unit AI pattern** — `units.par_iter_mut().for_each(|u| { … read
  index … })`: reads are `&self` + `Sync`, units mutated disjointly → safe, no new
  API needed. Confirm + document it (this is the demo's lever).
- **Parallel batch `knn` / `raycast`** (the read-fan-out, like `cull_many_par`).
- **Parallel tree bulk-load** — **DONE** (2026-07-01): `Tree3::bulk_load` /
  `bulk_load_par` (top-down partition; par fans out over rayon), brute-force
  gated. A parallel *rebuild* (~1.14× CPU-fps at high threads). **But** the
  follow-up measurement (2026-07-02) showed **keeping** the tree with `update_ref`
  in place beats rebuilding it at all (~1.06–1.4×), so both siege binaries now use
  `siege_sim::sync_index` (keep) instead; `bulk_load_par` stays the library
  primitive for from-scratch static builds (PARALLEL.md § "rebuild vs keep").
- **Properly benchmarked + tested** (user, 2026-06-27): interleaved min-of-N vs
  the serial path (the `raycast_compare` anti-contamination methodology — rotate
  order, min, `noise` column, two runs), and correctness tests (parallel result
  == serial). Document the crossover (when threads pay vs the fork overhead).

**★ PHASE 1 — `siege` demo (procedural 3D medieval battlefield)** `[review]`
The flagship showcase: nearly every battle mechanic *is* a spatial query we
built. New bin `vectorial-hash-demos/src/bin/siege.rs`.
- **Procedural terrain** — heightfield (noise) + streams/rivers with **bridges**
  (index choke-points), **forests** (block line-of-sight), a central **volcano**
  (hazard + landmark), **two castles in opposite corners** (per-faction spawns).
- **Two factions** sally from the castles and clash in the middle; density
  evolves sparse-corners → dense-melee (the decision map, live).
- **Troop roster → library feature**: foot soldier = k-NN nearest enemy + short
  cull; knight (cavalry) = **thick raycast** sweep along the charge; archer =
  **`raycast_first`** (LoS + first hit), volleys via **`cull_many_par`**; ballista
  = **all-hits raycast** (pierces a line); catapult = **sphere cull** AoE on
  impact; mage = sphere cull + **chained k-NN** lightning; **dragon** (flies, 3D)
  = **`Polyhedron3` cone / capsule** fire breath; healer = friendly k-NN. Plus
  1k+ units relocating per frame = the `update_ref`/rebuild maintenance stress.
- **Unit AI** (user, 2026-06-27) — a per-unit state machine: advance → acquire
  target (k-NN nearest enemy in range) → attack (the type's query) → morale /
  flee / regroup. Emergent but simple; runs in the parallel per-unit loop.
- **Formations = boids** (user: "el de las abejas" = Reynolds **flocking**:
  separation + alignment + cohesion) — *another* index showcase, each boid
  queries its neighbours via **k-NN / radius cull**. Cavalry & soldiers flock;
  archers hold rigid ranks.
- **Smoke = dynamic line-of-sight blockers** (user) — smoke clouds (catapult
  impact, mage spell) in a separate index; an archer's `raycast` that hits smoke
  before the target → no LoS, temporary (the cloud dissipates). Dynamic
  obstacles + raycast.
- **Render**: reuse `critters3d`'s GPU instancing; one shape+colour per
  type/faction. Orbital camera.
- **Publish to the web** (user, 2026-06-27) — also build `siege` for wasm and
  deploy to the GitHub Pages site (like `critters`/`critters3d`): a card on
  `docs/index.html` + `docs/siege.html` + `docs/siege.wasm`. wasm has no threads,
  so the AI runs **serial** under `cfg(target_arch = "wasm32")` (the parallel
  `par_iter` path is native-only). Build via `scripts/build-web.sh`.
- **Live thread-count slider** (user, 2026-06-27) — the demo runs the per-unit
  AI inside a resizable `rayon::ThreadPool`; a screen-space slider sets
  `num_threads` (1..=cores) live, so the fps response *shows* the parallel
  scaling. Native only — hidden on wasm (no threads → serial). **Done** in the
  foundation.
- **Phased (overnight = the foundation)**: terrain+camera+render → factions+spawn
  +advance → troop types+attacks → combat (damage/death/clash). Visual polish +
  balance is iterated with the user (the artistic side needs their eyes).
  **Foundation built (2026-06-27):** value-noise terrain + volcano, two castles,
  Red/Blue armies (soldier/archer/knight/dragon) spawning + advancing + clashing,
  per-unit AI on the parallel decide→serial-apply split, deaths + respawns,
  instanced render, orbit camera, thread slider. The decide pass now runs three
  library queries per unit: **`knn`** (nearest enemy *and* nearby friends in one
  pass), **boids** separation+cohesion flocking for ground melee (from those
  friends), the dragon's **`Sphere3` `cull`** AoE, and the archer's thick
  **`raycast`** line-of-fire (first unit struck blocks the shot). Seven troop
  types now, each a distinct query: + catapult (**wide `Sphere3` cull** AoE),
  ballista (**all-hits `raycast`** pierce — doesn't stop at the first hit, vs the
  archer), mage (**chained `knn`** lightning, 4 links). Smoke = dynamic LoS
  blockers **done**: catapult/dragon strikes spawn puffs into their own `Tree3`
  (capped, aging out ~3.5s); archer/ballista `raycast_dda_first` it and a puff
  in the line blocks the shot (same parallel-safe emit pattern). Healer **done**:
  carries unit health in the index item, seeks its most-wounded comrade
  (friendly `knn`), heals it as *negative* damage (capped at full HP in apply).
  **Eight troop types, all eight a distinct query.** **Published to the web**
  (2026-06-27): `docs/siege.html` + `docs/siege.wasm` (626 K) + index card; the
  wasm AI runs serial (no threads).

**Siege — status (2026-06-28, after the rebuild night).** The 2026-06-27 visual
feedback is all addressed and several more rounds landed. **Done:**
- **Look.** Smooth lit heightfield → now a **voxel (blocky) terrain** by default
  (flat-top cells + cliff walls + baked corner AO + the quad-flip fix;
  `$SIEGE_SMOOTH=1` for the old mesh). Glowing crater + lava flow carried over.
- **Visible combat.** Transient `Fx` effects (arrow/bolt streaks, AoE rings,
  lightning, heal sparks) + a **projectile system** (arcing cannonballs with
  travel time; volcano lava bombs) — combat reads now.
- **Real models + animation.** Every (faction, kind) is a **Quaternius glTF
  model** (pirates Red vs undead Blue), with **baked skeletal animation**
  (walk / attack / idle-for-riders), cavalry, and a **Castle model** per keep.
- **World.** Procedural **rivers** carved into the height (units wade at 0.4×),
  **bridges** at the crossings, lava burns ground units, the volcano erupts.
- **Controls.** A **population slider** (army size per side) beside the thread
  slider. Random map each run.
- **wgpu twin.** A second binary (`siege_wgpu`) renders the battle with wgpu and
  **real GPU skeletal skinning** (the thing macroquad's WebGL1 can't) — see
  SIEGE.md. Native only so far.

**Siege — 2026-06-29 parity night (landed).** The wgpu twin went from a skeleton
to a real parity demo, plus several queued items: the whole sim **lifted into a
shared `siege_sim` lib module** (both binaries can't drift); the **cannon-facing
fix** (artillery aims forward while kiting); the wgpu binary now runs the shared
sim with **real per-(faction,kind) GPU-skinned models** (static fallback for the
cannon/castle), **combat effects + projectile markers** (LineList), **castles**,
the **voxel terrain + `V` smooth/voxel switch**, `[ ]` **population**, **pause**,
and **HUD stats** in the title. macroquad got **boids alignment**, a "who's
leading" HUD line, and **alterable voxel terrain** (impact craters + the live `V`
switch). A 2D **`Circle` shape** was promoted to the lib (analytic `classify_box`
+ SoA batch, brute-force gated). Web republished.

**Siege — 2026-06-30 fixes night (landed, after the user ran the wgpu build).**
All five reported bugs fixed: wgpu terrain now alters on impact, the projectile
got a real sphere model (was line crosses), on-screen counters + a draggable
population slider in wgpu, macroquad **smooth** mode deforms too, and units no
longer float over craters — root cause was that craters were render-only, so they
were **lifted into the shared sim** (`Craters`, one source of truth for unit feet
+ both meshes). Plus: a **live thread-count slider** in wgpu, the **slime tweak**
applied in wgpu, **per-pixel lighting** (terrain + units — no more Gouraud facets
on the volcano / big models), a **smooth elevation colour ramp** (no contour steps
/ patchy sand along the river), and the slider/HUD overlay. Perf: a precomputed
boids **separation-force table** behind `$SIEGE_BOID_TABLE=1` — *measured slower*
than the live maths (the memory wall; default stays maths) — and two free
heightfield-eval dedups. See **docs/PERF_NOTES.md** (the force-table finding + the
FPS review: free wins applied, quality-trade-off levers reported).

**Siege — still remaining:**
- ~~**slim `siege.wasm`**~~ — **done.** The macroquad siege web build fetches its
  glTF models from `docs/models/<name>.glb` at startup instead of `include_bytes!`,
  so the ~9 MB of models aren't baked in: **siege.wasm 9.5 MB → 1.6 MB** (the
  unreferenced `model_for` + prop bytes are dead-stripped by wasm-ld gc). Native
  still embeds them (zero behaviour change). Split hidden behind `unit_bytes` /
  `prop_bytes` + `siege_sim::{model_file, SIEGE_MODEL_FILES}`; the build script
  copies the set to `docs/models/`. *Runtime fetch to be eyeballed in-browser.*
- **`siege_wgpu` on the web** — publishes, but recent Chrome rejects `requestDevice`
  because wgpu 0.20 sends the now-removed `maxInterStageShaderComponents` limit.
  Fix queued: a small `GPUAdapter.prototype.requestDevice` shim in the shell that
  strips the dropped limit (no wasm rebuild); or upgrade wgpu.
- **Optional "classic Reynolds" boids** behind a flag (inverse-square separation,
  arrival, inertia) — offered; needs the user's eyes to judge "more organic".
- **Frustum-cull units** in the render (free quality-wise, situational; PERF_NOTES).
- **wgpu world polish**: bridge decks + path-to-bridge AI, optional unit-feet snap
  to voxel tops (user chose smooth-float for now; keep it a switch).
- ~~**Forests**~~ — **done** (first pass). Seeded copses (`siege_sim::forest_*`)
  that **slow ground movement** (up to ½ speed; flyers pass over) and give
  **ranged cover** (arrows/bolts do ½ damage to a target in the trees). Canopies
  drawn in both binaries (green sphere blobs). Copse count/size + factors are
  easy knobs; **balance pass** still wants the user's eyes on the HUD.
- ~~**2D critters thread slider (#56 opt A)**~~ — **done.** `Sims::cull_attack`
  split into a read-only `cull_attack_ro` (returns timing/mismatch) + `fold_atk_stat`;
  `Sim::step`'s firing loop restructured into decide (serial, rng) → cull
  (parallel over a live-sized pool `Sim::set_threads`, native / serial wasm) →
  apply (serial, order-preserving kill resolution). `critters` gained a **threads**
  slider. Verified via the headless bench in `--mode both`: *all culls still
  agree* (the dual-index mismatch check stays 0) and kills resolve — the parallel
  path is result-identical to serial. (One accepted change: a critter that dies
  this frame still decided/culled, discarded on apply, so the rng draw order
  shifts slightly — no determinism test pins it.)

**Thread-slider retrofit for the critters demos** (user, 2026-06-27) —
**`critters3d` done**; **2D `critters` deferred.** The 3D combat wave now runs
decide (serial, `rng`) → cull (parallel) → apply: each firing critter's attack
volume (a new `AttackVolume` enum unifying the Pulsar sphere + the Hunter/Drifter
drop) is culled over a live-sized rayon pool, with a panel **threads** slider
(combat, native, multicore only) and the count in the HUD. Culls are read-only,
so the kills are identical to the serial path — the slider just makes the
crossover visible (`PARALLEL.md`), like siege. wasm runs the wave serially.
- **2D remaining**: the 2D combat lives in the shared `sim.rs`, where
  `Sim::cull_attack` is `&mut self` (accumulates per-structure timing + the
  dual-"Both"-mode hit-set mismatch check). Parallelising it means refactoring
  that bench-depended-on method to a read-only cull + post-hoc stat aggregation
  and restructuring `Sim::step`'s firing loop — a deliberate change to shared,
  tested infrastructure for a small payoff (2D culls are cheap), so it's left for
  a focused pass rather than an autonomous one.

**A. Ray-cast / structure follow-ups (continue the thread)**
1. ~~**`Octree3` DDA**~~ — **done.** `raycast` (thick capsule), `raycast_dda`,
   `raycast_dda_first` on the 8-way octree (Probe-style, ported from `Tree3`).
   Brute-force gated (DDA ⊆ exact capsule + sorted + first==nearest; thick ==
   brute). Still uses the `locate`+epsilon nudge → item 2 (nudge-free walk).
2. ~~**Nudge-free 3D walk**~~ — **DONE** (2026-07-18). Both `Tree3` and `Octree3`
   DDA now step **without** the `locate`+epsilon nudge: the exact face-neighbour is
   found by **ascending to the LCA** whose sibling is across the exit face, then
   descending to the leaf at the exit point (Samet's rope-free neighbour). Tree3
   reads the split axis back from the child boxes; Octree3 flips the exit-axis octant
   bit. Verified by the existing gates (DDA ⊆ capsule, sorted, first == nearest)
   **plus a new completeness test** (sample the ray, `locate` each point, assert every
   crossed leaf was visited — catches the under-collection the subset test couldn't).
   `raycast_start_leaf` keeps one `locate` for the entry only. See RAYCAST.md.
3. **Promote a 2D `Circle` `Shape` to the lib** (analytic `classify_box`, like
   `Capsule`) — the examples each redefine a `Disc`; one lib primitive dedups
   them and gives users a tight 2D circle cull.
4. **Raycast in the Criterion suite + regression baseline** — add `raycast` /
   `raycast_first` / capsule ops so perf regressions are tracked.

**B. Correctness / robustness**
5. ~~**Property / fuzz tests** (proptest)~~ — **done.**
   `tests/proptest_structures.rs`: a randomized op sequence (insert / remove /
   update via the O(1) `ItemRef` path) driven against a brute-force model, then
   `cull` (3 volumes) + `knn` (2 probes × k∈{1,5,12}) asserted equal, with
   shrinking. All seven structures (`Tree`, `QuadTree`, `IntegerTree`, `Tree3`,
   `Octree3`, `MortonGrid` 2D + `MortonGrid3` insert-only). Also checks
   `item_count()` tracks the model so the handle bookkeeping can't silently
   drift. proptest is a dev-dependency only (not in the shipped crate / wasm).
6. ~~**Serialization for all structures**~~ — **done.** `serialize`/`deserialize`
   now on `Tree`, `QuadTree`, `IntegerTree`, `Octree3`, `MortonGrid` (2D),
   `MortonGrid3` (mirroring `Tree3`), sharing the LE byte-IO helpers in
   `serde_io.rs`. Each round-trips the exact arena + free-list + `ItemRef`
   handles (grids: world+levels+buckets), gated by a per-structure round-trip
   test (arena counts + cull + knn identical + corruption rejected). `Tree`'s
   `neighbors` ropes are *rebuilt geometrically on load* (not stored), so the
   format is feature-independent.
7. ~~**Templates CI cleanup**~~ — **done.** The ~21 clippy lints are gone
   (grid-indexing `needless_range_loop` + one `type_complexity` allowed
   crate-wide with justification — 2D matrices index by explicit `(x,y)`; the
   rest fixed: `Default` impls, collapsed ifs, `div_ceil`, redundant closure,
   module doc). The fingerprint test is now portable: it hard-gates
   **determinism** (generate twice) + **row structure** (count + per-row labels)
   everywhere, exact **bytes** on the reference platform (`VH_FINGERPRINT_STRICT=1`,
   where libm noise is zero), and on other runners fails only on a *wholesale*
   break (>25 % rows) — a subtle real regression and cross-platform libm ULPs are
   the same tiny magnitude, so subtle byte review belongs on the reference box.
   CI now **hard-gates** clippy + tests for templates + cli (was advisory).

**C. Performance**
8. **Morton multi-level linear octree** — coarse occupancy levels so a big-radius
   cull skips empty blocks instead of scanning every fine cell (2D + 3D).
9. **Parallel tree bulk-load** — **DONE** (2026-07-01, `Tree3`): `bulk_load` +
   `bulk_load_par` (top-down partition, par via rayon `join`), brute-force gated.
   Superseded in the demos by keep (`sync_index`); stays the from-scratch static
   build (horde's structure index uses it).
   *Remaining — bulk-load PARITY (user, 2026-07-04):* **DONE** (2026-07-06):
   `Octree3::bulk_load`/`bulk_load_par` (8 octants per level, nested `join`s
   fan the children) and the 2D `Tree` (its `pick_split` rule, ropes rebuilt
   under `neighbors`); same `ItemRef(i) == items[i]` contract, Tree3's
   brute-force + par==serial tests mirrored on both.
10. **SoA *permanent* leaf storage** — *low priority*: the batch kernel is done
    and measured marginal (~1.0–1.26× end-to-end, descent-dominated); permanent
    storage would only save the materialisation copy. Park unless a workload needs it.

**D. Docs / DX**
11. **Sync the lib README** — it predates the ray-cast surface, `MortonGrid` (2D),
    `Capsule`, and now lists fewer structures than exist (seven). Refresh the
    public-surface table + add a ray-cast paragraph.
12. **Performance guides** — `target-cpu=native`, AoS/SoA + SIMD, threading, the
    opt-in API story (deferred section below has the source material).

**E. Demos / web [review]**
13. **[review] `MortonGrid` (2D) in the `critters` demo** — add the flat grid to
    the 2D structure toggle (+ optionally `IntegerTree` to `decision2d`).
14. **[review] 2D k-NN / line-of-sight demo** — visualise k-NN and the ray-cast
    first-hit (the DDA corridor) in the 2D `critters` demo.
15. **[review] Web touch controls + stats HUD** — orbit/zoom by touch on the 3D
    web demo; an FPS/stats overlay in the browser.
16. **[review] 2D demo panel** — figure-size slider + port the 3D `Panel` into a
    shared `src/panel.rs`.

**Deferred (design / user-held)** — template not-found diagnostics + next-size-up
fallback (rethink); crates.io publish (until more complete). See sections below.

## Overnight queue (set 2026-06-25) — ✅ all done

1. ~~Morton / Z-order (linear octree)~~ — done (also now selectable in the
   `critters3d` `M` toggle).
2. ~~k-NN queries~~ — done.
3. ~~Index serialization~~ — done (`Tree3`).
4. ~~World-size 2D stepper~~ — done (visual scaling review → see active queue).
5. ~~Instancing stress test~~ — done.

## Overnight queue — set 2026-06-25 (expanded round 2)

Worked top-down. Commit + push per task. Items marked **[review]** need a human
visual glance the next morning; everything else is autonomously verifiable.

> **Local handoff:** a gitignored `CLAUDE.md` at the repo root summarises the
> project state, conventions, and what's done vs. open — for future sessions on
> this machine. (User asked to add it + gitignore it.)

**Done**
1. ~~**rayon / multithreading**~~ — **done.** `parallel` feature → rayon-backed
   `cull_many_par` on all five structures (serial `cull_many` always available).
   Finding: *reads fan out, writes don't* — batch culls parallelise cleanly
   (the relocation pass mutates and stays serial; its lever is `ItemRef`).
   Measured crossover in `critters3d_headless --parallel`: ≤4 queries never pays
   (fork ~15–30 µs), 16 pays at ≥20k points, 64+ pays always (up to ~8× at 1024
   queries / 100k points). Full table + guidance in `docs/PARALLEL.md`.

**Priority block (user bumped regression baseline to the front, 2026-06-25)**
2. ~~**Criterion benches + CI + regression baseline**~~ — **done.**
   - `benches/spatial.rs` (Criterion: build/cull/update/knn across structures).
   - `examples/regression_gate.rs` — deterministic gate vs a **committed**
     `benches/baseline.tsv`, exits 1 on regression. **min-of-N + a calibration
     loop** (compare op/`_calib` ratios) cut back-to-back variance ±60% → ±6%,
     so it can actually gate. `benches/README.md` documents both tools.
   - `.github/workflows/ci.yml` — clippy (`-D warnings`, gating on the flagship)
     + tests (lib/templates/cli, ±parallel) + bench/example compile + wasm build
     of the web demos. fmt is advisory (the dense hand-style isn't rustfmt-clean).
   - Cleaned 11 pre-existing clippy lints in `vectorial-hash` to make the gate
     pass.
   - *Follow-up:* templates + cli still carry ~21 pre-existing clippy lints
     (advisory in CI for now) — a focused cleanup pass should make them gate too
     (careful: some are `needless_range_loop` where the index is genuinely used).
   - *Follow-up:* `vectorial-hash-templates`' `template_fingerprint_matches_fixture`
     test is **not cross-platform** — the committed fixture was generated on the
     Windows dev box and the fingerprint differs on the Linux CI runner (float
     formatting / iteration order). Advisory in CI for now. Fix: make the
     fingerprint platform-independent (deterministic float formatting + sorted
     iteration), then move templates/cli tests back into the hard gate.

**Original remaining**
3. **Morton extras** — `MortonGrid3::knn` (ring-by-ring expanding shell, like
   the tree k-NN but over Z-order cells) + a **multi-level linear octree** layer
   (one hashed level per cell size) so Morton can answer big-radius culls without
   scanning every fine cell. Brute-force gated tests.
4. ~~**[review] Web demo responsive + hide-UI button**~~ — **done (needs visual
   review).** Pure `docs/` HTML/JS, no wasm rebuild. The 2D demo (fixed layout,
   no `screen_width` use) is centred and **CSS scale-to-fit** (letterbox) so the
   whole thing shows on any screen; mouse stays aligned (miniquad maps through
   the transformed bounding rect, framebuffer size unchanged). The 3D demo
   already lays out from `screen_width`, so it keeps filling the viewport. Both
   pages gain a **⤢ toggle** that hides the chrome and enters fullscreen (tap
   again to restore). **Morning check:** in-browser look on a phone, esp. that
   the 2D panel sliders are still usable when scaled (touch-friendly 2D panel is
   the deferred follow-up).

**Newly added (round 2 — autonomously verifiable)**
5. ~~**Bulk-build parallel**~~ — **done (Morton).** `MortonGrid3::extend_par`
   (feature `parallel`): per-item quantise+Morton-encode on rayon, then a serial
   bucket group. The grouping is the serial tail (Amdahl), so the win is the
   encode for large `N`; pair with `clear()` for a cheap parallel
   rebuild-per-frame. Tested identical to serial insert (count/cells/cull).
   *Remaining:* ~~a parallel **tree** bulk-load~~ — **also done** (2026-07-01):
   `Tree3::bulk_load_par` (top-down partition, rayon `join`), in the siege rebuild.
6. **k-NN parity** — `knn` exists on `Tree3`/`Octree3`; add it to `Tree`,
   `QuadTree`, `IntegerTree`, and `MortonGrid3` (the latter ring-by-ring, shared
   with item 3). Brute-force gated.
7. ~~**Ray-cast / segment query**~~ — **done, both approaches + comparison.**
   - **Capsule** (`Segment3` `Shape3` + `Tree3::raycast`, 3D): the ray as a
     thickened segment over `cull`. Exact "items within r", reuses culling.
   - **DDA leaf-walk** (`Tree::raycast(.., walk)`, 2D — the user's `TestDraw`
     idea adapted to variable cells): Amanatides–Woo over the tree, walking only
     the cells the centre ray crosses, neighbour step via selectable
     `WalkNeighbors` (Samet/Probe/Ropes). Returns sorted hits + traversal stats.
     The 3 methods agree (test); `examples/raycast_compare.rs` benchmarks them
     vs the capsule. Findings (`docs/RAYCAST.md`): DDA ~8–15× faster on the thin
     corridor; coverage falls with radius (97%→10%) so DDA=thin rays /
     capsule=thick; ropes < samet < probe on query time (same walk, just
     neighbour cost). **Open:** the maintenance side of ropes (build/update with
     vs without) to settle the overall trade-off; and a thick-corridor exact
     DDA + a hard `raycast_first` with early-exit.
8. ~~**Frustum as a first-class `Shape3`**~~ — **done.** A view frustum is just
   6 half-spaces, i.e. a `Polyhedron3`; added `Polyhedron3::from_corners([Point3;
   8])` (near face then far face, bl/br/tr/tl) that derives the six inward face
   planes (oriented against the corner centroid, so winding need not be exact).
   `tree.cull(&Polyhedron3::from_corners(corners))` culls a camera frustum.
   Tested: an axis-aligned box's corners recover its faces exactly.
9. ~~**`clear()` retaining capacity**~~ — **done.** `clear()` on all five trees
   + `MortonGrid3`: resets to an empty root leaf / clears the bucket table while
   keeping the arena/hash-map capacity (the cheap path for per-frame rebuilds).
   Invalidates outstanding `ItemRef`s. Tested on Tree3 + MortonGrid3.
10. **Property / fuzz tests** (proptest) — randomized insert/update/remove/cull
    vs brute force with shrinking, across all structures. Stronger than the
    hand-rolled churn tests.
11. **Serialization for all structures** — only `Tree3` has versioned serialize;
    extend to `Tree`/`QuadTree`/`IntegerTree`/`Octree3`/`MortonGrid3`.
12. **SoA leaf storage + SIMD narrowphase** — store leaf positions contiguously
    for cache-friendly `contains_point` sweeps; explore SIMD on the sphere/point
    test. Measure vs current AoS.
13. ~~**"Choosing a structure" flowchart**~~ — **done.** `docs/CHOOSING.md` (a
    2D/3D decision flowchart + summary table + rules of thumb), linked from the
    lib README. *(User: "interesante, sí.")*

**Newly added (round 2 — [review] / design)**
14. **[review] 2D k-NN / line-of-sight demo** — visualise nearest-neighbour and
    LoS in the 2D critters demo.
15. **[review] Web touch controls + stats HUD** — orbit/zoom by touch on the 3D
    web demo; an FPS/stats overlay in the browser.
16. ~~**CHANGELOG.md**~~ — **done.** Root `CHANGELOG.md` (Keep-a-Changelog),
    `[Unreleased]` capturing this round + a `[0.1.0]` baseline.

### Deferred — explicitly held back
- **Publish to crates.io** — *deferred until the crate is more complete* (user,
  2026-06-25). The crate is already publishable; cut a 0.1.0 when ready.
- **Performance guides** — when written, cover: how a library user opts into the
  demo's optimisations (mostly good defaults + a few opt-in API choices, not
  manual threading); **`-C target-cpu=native`** (the cheapest perf knob — unlocks
  AVX / wider auto-vectorisation, ~2–10× on number-crunching, but non-portable so
  local/bench/self-hosted only — see `docs/RAYCAST.md` § Concepts); and the
  AoS/SoA + SIMD story (the narrowphase-ceiling microbench). *Deferred to "when we
  stop adding so many new things" (user, 2026-06-25).* Source: the chat answers +
  `THREE_D.md` + `RAYCAST.md` (Concepts).
- **2D demo panel: figure-size slider + 3D-style sliders** — figure-size slider
  (rebuild arsenal via `build_arsenal_scaled`, step on release) + port the 3D
  `Panel` into a shared `src/panel.rs`. *(Lower priority; not selected.)*

### Templates — needs a rethink (design, not yet built)
- **Next-size-up fallback** — when no template exists for a cell size, take the
  next size up with special handling. *(See memory `project_template_application_design`.)*
- **Template-not-found diagnostics / debug mode** — report which cases fell
  through to the per-item point test (coverage gaps), and **bound the cell-size
  range** where the next-size-up fallback fires (the largest cells need not have
  templates). User flagged "other interesting things" here too — wants to give
  the whole template-coverage story a repensada. *(2026-06-25.)*
- **3D / voxel template bank** — extend the 2D template machinery to voxel
  rasters.

Everything else in this file is **future** — left to triage later.

## Render / demos
- ~~**Instancing stress test**~~ — **done.** Swept `critters3d` to 1M critters
  (`CRITTERS3D_POP`/`WORLD`/`RENDER`/`FREEZE`/`MAX_FRAMES`, one-line `STRESS`
  summary per run). Finding: the **live** demo is CPU-bound through 200k+ (the
  per-frame `Tree3::update` is the ceiling; rendering 200k instanced spheres
  adds only ~3 ms). **Frozen** (render-only), the GPU path is upload/transform
  bound and linear: square billboards ~14 ns/instance (1M @ 72 fps), spheres
  +~50% (1M @ 47 fps), not fill-bound. So the renderer has huge headroom; the
  next win is parallel index maintenance, not the GPU. See `THREE_D.md` §
  "GPU instancing stress test".
- ~~**World-size 2D stepper**~~ — **done (visual scaling wants a human glance).**
  `Sim` gained a runtime `world_size` (default `MAP_W`) + `set_world_size`
  (re-bounds critters, rebuilds the index); `random_pos`/movement/`world_rect`
  use it. The `critters` panel has a **world** button stepping pow-2 256→4096
  (`WORLD_STEPS`), with the render scale = `MAP_PX / world_size` so the map fills
  the same on-screen square at any size. Headless `--world N` + `CRITTERS_WORLD=N`
  env. **Verified headlessly**: cull agreement holds at 256/512/1024/2048 and
  the visual runs clean at each. **TODO (human):** eyeball the on-screen scaling
  at the extremes (256 zoomed-in, 4096 zoomed-out) — sprite/figure sizes are in
  world units, so attack templates shrink/grow with the world; confirm it reads
  well or cap the scale.
- **Colour by leaf depth / structure** — visualise the tree's interior (which
  cells are deep, how the split differs binary vs octree).
- **Batch the sight-lines / combat effects** — the per-line `draw_line_3d` and
  per-attack `draw_sphere_wires` calls are immediate; batch them into meshes
  when many are on screen at once.

## Index / algorithms
- ~~**Structure decision-map sweep**~~ — **done.** `critters3d_headless` now
  drives all four structures on one deterministic sim (per-structure maintain +
  cull) and `--sweep` prints the winner-per-cell map over world × pop ×
  item_limit × churn. **Surprise result that corrected the by-eye synthesis:**
  Morton wins *maintain* in every config (its flat re-bucket beats the trees'
  `update`, which pays an O(item_limit) predicate scan per point); the trees win
  *cull* only in the deep/dense corner. → promotes **Stable `ItemRef`** (O(1)
  item access) as the highest-leverage follow-up. See `THREE_D.md` § "Synthesis".
- ~~**Stable `ItemRef`**~~ — **done for `Tree3`** (`insert_ref` / `update_ref` /
  `remove_ref`; parallel per-leaf handle vec + handle→location map, churn-tested,
  survives `serialize`). Confirmed the decision map's prediction: O(1) handle
  updates flip the maintain winner from Morton back to the binary tree in 15/16
  configs, **~5–11× faster** (10× at item_limit 64). Follow-ups: port to
  `Octree3`; wire the live `critters3d` persistent tree to it. See `THREE_D.md`
  § "The fix: Stable ItemRef".
- ~~**`Octree3::update` (ascend-to-LCA)**~~ — **done.** `Octree3` now has the
  same ascend-to-LCA `update` as `Tree3` (churn-tested), and `critters3d_headless`
  compares the *dynamic* octree vs the binary tree on one deterministic run
  (octree's update is ~5–15% faster; cull ≈ equal; id sets identical) — though
  the decision map shows both trees lose *maintain* to Morton. The live demo's
  `M` toggle keeps a persistent octree too. See `THREE_D.md` § "Dynamic octree
  vs binary".
- ~~**Full 3-projection vs expensive narrowphase**~~ — **MEASURED** (2026-07-19,
  `examples/broadphase_tightness_bench`): tested whether a tight broadphase pays as
  the narrowphase gets expensive (a `faceted_ball` `Polyhedron3`, N faces = N
  dot-products/point). **Honest negative:** culling by the tight *volume*
  (`cull(&poly)`) LOSES to a cheap 6-plane box broadphase + N-plane narrowphase at
  every face count (1.0×→1.6×, gap widens with N) — the tight prune runs the N-plane
  `classify_box` at *every node* and nodes ≫ candidates, so the per-node cost
  dominates the candidate savings. Validates the kit's cheap-broadphase +
  exact-narrowphase design (and the `VoxelRaster` short-circuit for the narrowphase).
  THREE_D.md § "Does a tight broadphase pay…".
- ~~**Morton / Z-order keys (linear octree)**~~ — **done.** `MortonGrid3`
  (pointer-free Z-order hash grid) added + churn-free brute-force test, wired
  into `tree3d_bench` as the fourth structure. Fastest index on uniform data
  (14.5× vs brute, beating both trees) with the cheapest build; single fixed
  resolution is the catch (octree wins on stacked/non-uniform). See `THREE_D.md`
  § "Morton / Z-order linear grid". Follow-up ideas: a *multi-level* linear
  octree (mixed-depth codes) to recover adaptivity, and an `update` path for
  the dynamic workload.
- **Multi-query cull + multithreading** — cull many spheres at once; parallel
  build/cull (rayon) for large N.
- ~~**k-NN queries**~~ — **done.** `Tree3::knn` and `Octree3::knn` (best-first
  descent, bounded max-heap, bbox pruning, nearest-child-first), brute-force
  gated. 2–13 µs/query at 50k for k=1..50; octree ≈ binary. `tree3d_bench --knn
  K`. See `THREE_D.md` § "k-nearest-neighbour". Follow-up: `MortonGrid3::knn`
  (ring-by-ring outward from the query cell).
- ~~**Index serialization**~~ — **done for all structures.** `serialize` /
  `deserialize` round-trip the built index (exact arena + free-list + `ItemRef`
  handles, no rebuild) to any `std::io::Write`/`Read`, dependency-free, items via
  a caller closure (works for any `T`). Now on `Tree`, `QuadTree`, `IntegerTree`,
  `Tree3`, `Octree3`, and the Morton grids (`MortonGrid` 2D + `MortonGrid3` store
  world+levels+occupied buckets, no arena). Shared LE byte-IO in `serde_io.rs`;
  each a versioned format (magic `VHT2`/`VHQ2`/`VHI2`/`VHT3`/`VHO3`/`VHM2`/`VHM3`
  + version byte). Per-structure round-trip test: arena counts + cull + knn
  identical + corruption rejected. `Tree`'s `neighbors` ropes are derived state,
  rebuilt geometrically on load, so the format doesn't depend on the feature.

## Dilation
- **3D dilation (Minkowski)** — agent body radius for the 3D critters, the way
  `inflated_convex` / `within_dilation` does it in 2D.
- **Non-convex dilation** — `inflated_convex` is convex-only today.
- **Dilation LUT per radius** — more aggressive precache of inflated shapes.

## Templates / infra
- **Next-size-up fallback** in template application (already flagged as pending
  design: select the template, never move the figure).
- **3D voxel templates** cached (the N³ analogue of the 2D template grid).
- **Criterion benches + CI** — stable numbers instead of ad-hoc bins; run the
  fingerprint/verify_88/exhaustive battery in CI.
- **Sub-block dedup** (from the design doc) — investigate and summarise what it
  was; decide if it earns its place.
