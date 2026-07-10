# vectorial-hash-demos

Runnable demos for the [`vectorial-hash-kit`](../../README.md) workspace. Not published to crates.io (`publish = false`).

## Demos

`cargo run -p vectorial-hash-demos` runs two demos back-to-back.

### Demo 1 — 8-symmetry dedup

In-memory generation over a "drop" polygon at four angles (0°, 45°, 90°, 135°) on a 16-cell grid, showing how the 8-symmetry dedup collapses rotations onto canonical templates.

```
  angle   0.0deg -> id 1 via eq (new: true)
  angle  45.0deg -> id 2 via eq (new: true)
  angle  90.0deg -> id 1 via rCC (new: false)
  angle 135.0deg -> id 2 via rC  (new: false)
Unique templates: 2
```

### Demo 2 — end-to-end polygon → template → Tree::cull

Walks the full pipeline:

1. Build a drop polygon (scaled, translated into the root rect).
2. Generate its `Matrix` template via `get_template_grid_fast`.
3. Convert to a runtime `TemplateGrid` via `adapter::matrix_to_template_grid`.
4. Wrap polygon + grid in a `Shape` (template short-circuit on internal nodes, point-in-polygon fallback at leaves).
5. Insert a 40×40 point lattice into a `Tree` and call `Tree::cull`.
6. Cross-check the cull result against a brute-force `polygon::is_inside` sweep — they must match.

```
  polygon bbox: x[87.2,112.8] y[30.0,94.0]
  template grid: 5x9 cells of 8x8 anchored at (80.0, 24.0)
  tree size: 1600 pts | cull hits: 25 | brute-force inside: 25
  cull == brute-force: OK
```

### Demo 3 — critters (visual, macroquad)

```bash
cargo run -p vectorial-hash-demos --bin critters --release
```

A live 2D map (1024×1024 world, drawn at 0.75×) indexed by a `vectorial_hash::Tree` (item limit 3 per cell). Every live leaf region is filled with its own colour, so you watch the map **divide and merge in real time** as items move.

- **Precise template application** (the paper's scheme): attacks are applied at their real integer origin — the figure is **never moved to fit a grid**. A hierarchical `TemplateBank` (shape → dims → cell width → cell height → angle → offset x → offset y) is generated at startup for cell sizes 8–32 px (squares and 2:1 rectangles) plus a 1×1 raster per angle; identical templates are deduplicated and shared (~65k combos collapse to ~8k unique grids in well under a second). During a cull, each tree-cell size resolves the template whose generation offset matches the figure origin's displacement within the global virtual grid of that size — template cells align 1:1 with map cells, so internal nodes classify with a single direct cell read (resolved once per size per attack and cached). Leaf items are answered by the 1×1 raster; only boundary (`Maybe`) pixels run exact geometry, pre-filtered by the bounding box.
- **Critters** of three kinds with distinct movement and attacks:
  - *drifter* (blue): random walk; fires a drop-shaped area in a random direction.
  - *hunter* (red): chases the nearest non-hunter; fires a drop aimed at it (angle snapped to the precomputed 15° set).
  - *pulsar* (gold): circles around; radial blast centred on itself.
- Attack resolution is a real `Tree::cull` with the green/yellow/white short-circuit; victims are `Tree::remove`d (watch regions merge) and **respawn** a few seconds later (watch regions split).
- Attack areas are drawn from the template cells themselves (bright = `In`, dim = `Maybe`) with the real attack polygon outlined on top (arcs flattened), so you can see exactly how the precomputed grid approximates the shape.
- Kill credit is tracked per attacker kind; hunters show a faint sightline to their current prey.
- **Hunters hunt with the tree too**: prey acquisition is a real cull over a vision circle (radius 280), so targeting scales with the index instead of a linear scan — populations go up to 1,200 per kind (4,000 total).
- **Four structure modes** (`M` cycles, or `CRITTERS_MODE=binary|quad|both|ibinary`): the binary-split tree, the reference quadtree, **both at once**, or the **integer binary tree** (`IntegerTree`, pow-2 integer coords). In dual mode every operation — insert, remove, update, attack cull, vision cull — runs on both structures with identical inputs; their cull results are compared live (an "agree" indicator turns red on any mismatch), the sliders rebuild both identically, and the quadtree's subdivision is overlaid as cyan outlines on top of the binary tree's coloured regions. The integer-tree mode draws its leaf regions (violet) and feeds the perf graphs (a violet `int` line) like the other modes.
- A third column plots **live performance graphs** (one polyline per structure where applicable): frame time, average attack-cull and vision-cull times, per-frame movement/update cost, and per-frame insert+remove cost.
- The simulation runs on a single thread (only the startup bank generation parallelizes), so the graphs show exactly where that thread's budget goes as you scale populations up.

### Demo 4 — critters headless (statistics)

```bash
cargo run -p vectorial-hash-demos --bin critters_headless --release -- \
    --mode both --frames 600 --drifters 400 --hunters 400 --pulsars 400 \
    [--world 1024 --split 3 --merge 3 --dt 0.0167 --seed 42 --fire 1.0 --respawn 2.5 --csv out.csv]
```

The exact same simulation core (shared `sim` module, fully deterministic for a given seed) without a window or vsync: it runs at CPU speed and reports per-structure statistics — mean/p50/p95 of per-frame movement+update, attack-cull and vision-cull averages, and insert+remove cost — plus steps/s, final tree shapes and the live cull-agreement counter in `both` mode. `--csv` dumps per-frame rows for plotting. Determinism means a `binary` run and the binary half of a `both` run produce identical simulations (same kills, same final tree), so cross-structure numbers are directly comparable. `--world N` sets the (square) world size — a power of two; cull agreement holds at every size (256–4096 verified).

Controls: `1`/`2`/`3` select the spawn brush · left click (or hold-drag to paint) spawns at the cursor · right click removes · `+`/`-` add/remove five at random · `R` cycles region rendering · `[` `]` change simulation speed · `Space` pauses · `Esc` quits.

The **"tuning (live)" panel** adjusts everything while the simulation runs: the tree's **split threshold** (a leaf divides above it) and **merge threshold** (siblings collapse at/below it — set it lower than split for hysteresis; the tree is rebuilt on change), per-kind **population targets** (spawns/removes to match), respawn delay, simulation speed and fire rate. A **world button** steps the (square, power-of-two) world size 256→4096 live — it re-bounds the critters and rebuilds the index, and the map always fills the same on-screen square (the render scale follows the world size). Manual spawns/removals (clicks, `+`/`-`) update the population sliders so both mechanisms cooperate.

`CRITTERS_MAX_FRAMES=N` exits after N frames (CI/smoke runs); `CRITTERS_WORLD=N` starts at world size N.

### Demo 5 — critters 3D (visual + headless)

```bash
cargo run -p vectorial-hash-demos --bin critters3d --release            # visual (macroquad 3D)
cargo run -p vectorial-hash-demos --bin critters3d_headless --release \  # all 4 structures, one config
    -- --pop 20000 --item-limit 64 --vision 36 --frames 120 --seed 42
cargo run -p vectorial-hash-demos --bin critters3d_headless --release -- --sweep   # decision map
```

The 3D analogue. The index is a **persistent** `Tree3` (binary-split 3D tree,
sphere-vs-Aabb classification): built once, then each frame every critter is
moved in place with `update` (ascend-to-LCA) instead of rebuilding — the HUD's
`build µs` shows that win. The **visual** demo drifts critters inside a cube
and is two demos in one, switchable live (`T`):

- **observe**: an observer at the centre runs a sphere vision cull each frame;
  the seen critters light up with a sight-line.
- **combat**: the red critters become predators that attack nearby prey. Half
  throw an omnidirectional **sphere blast** (orange), half an **aimed flamer
  drop** — a teardrop cone pointed at the nearest prey, in any 3D direction
  (like a 40k flamer template). Prey inside (found by an *index cull*) burst,
  respawn, and flash. This is the "many queries per frame" workload (a cull
  per attacking predator, plus a targeting cull per flamer).

Most controls are clickable in the **on-screen panel** (top-right, with a
hover-help box) and shown live in **time-series graphs** (fps / cpu ms / cull
µs) below it. vsync is off so the fps counter is the real ceiling; the HUD
reports a CPU-bound vs GPU-bound verdict and the CPU-only fps ceiling.

- `M` — **index structure** (a *real* switch: the selected structure resolves
  *everything* — observe vision, combat attacks, persistence — and the others
  don't exist): `Tree3` (binary, **stable `ItemRef`** → O(1) `update_ref`) /
  `Octree3` (8-way, **`ItemRef`**) / `MortonGrid3` (Z-order hash grid, rebuilt
  each frame) / projection (a 2D `Tree` on xy, **`ItemRef`** + z-reject). All
  exact. Because the chosen structure does all the work, the **fps reflects the
  structure**: binary / octree / projection all relocate in place via the stable
  `ItemRef` (O(1) — no predicate scan); the Morton grid re-buckets each frame.
  The index is rebuilt only when the structure / world / population changes.
  `B` shows its cells/leaves.
- `G` — **render path**: GPU-instanced spheres / round billboards / square
  billboards (fastest) / NO RENDER (CPU only, to read the CPU's fps ceiling).
  The instanced paths (raw miniquad, `src/instanced3d.rs`) draw all critters
  in one call; the old per-frame `draw_sphere` path was dropped.
- `T` observe/combat · `R` attack desync vs synchronized-saturation · `O`
  separation (no overlap; one neighbour cull per critter) · `V` fps cap · `C`
  cull-timing reps · `B` leaf boxes · `+`/`-` population (snaps to 200) ·
  `[`/`]` radius (snaps to 5) · world-size **stepper** (pow-2) · `Space`/`Esc`.

Env: `CRITTERS3D_MAX_FRAMES=N` (smoke runs; also prints a one-line `STRESS`
summary — mean fps / frame ms / cpu ms / CPU-vs-GPU-bound),
`CRITTERS3D_RENDER=instanced|billboards|square|none`,
`CRITTERS3D_STRUCTURE=binary|octree|morton|projection` (initial index),
`CRITTERS3D_POP=N` / `CRITTERS3D_WORLD=N` / `CRITTERS3D_FREEZE=1` (the
instancing stress sweep — see `docs/THREE_D.md`: scales to 1M critters in one
draw call, GPU upload-bound; the live demo is CPU-bound on the index update),
`CRITTERS3D_COMBAT=1`, `CRITTERS3D_SEP=1`. The **headless** version drives all
four structures (binary `Tree3` / octree / Morton / projection) on one
deterministic sim, reporting per-structure **maintain** (update or rebuild) and
**cull** cost; `--sweep` prints the **structure decision map** (winner per cell
over world × pop × item_limit × churn). Headline: for this full-relocation
workload Morton wins maintain (flat re-bucket beats the trees' predicate-scan
`update`), the trees win cull in the deep/dense corner — see `docs/THREE_D.md`
§ "Synthesis".

### Demo 6 — GPU (wgpu compute): broad-phase + a resident collision storm

Native-only (winit + wgpu compute; wasm-gated out). These show where the GPU
wins — and, honestly measured, where it doesn't (`docs/PERF_NOTES.md`).

```bash
cargo run -p vectorial-hash-demos --bin gpu_lbvh_demo --release   # switchable broad-phase
cargo run -p vectorial-hash-demos --bin gpu_storm     --release   # GPU-resident sim
cargo run -p vectorial-hash-demos --example gpu_spatial_bench --release --features parallel
```

- **`gpu_lbvh_demo`** — the *same* neighbour-count query, three backends you flip
  live (`1` CPU `Tree3` · `2` GPU brute · `3` **GPU LBVH**, a BVH built from the
  Morton codes the kit computes, traversed in a compute shader), coloured by a
  density heat-map, GPU-time meter in the title. Identical picture, ~393× spread
  on the query kernel (50k pts: CPU 122 ms → GPU LBVH 0.31 ms).
- **`gpu_storm`** — a **GPU-resident** collision storm: the *whole* hot loop
  (grid build → contacts → integrate) lives on the GPU, no per-frame round-trip.
  Switch the whole sim CPU↔GPU (`1`/`2`), `F` toggles collision ↔ an influence
  field. ~50× the CPU sim at 150k. The answer to "how much more than the query
  can we accelerate?" — everything but the branchy `decide` a game sim needs.
- **`gpu_spatial_bench`** — the headless measurement behind both, with the honest
  per-frame *rebuild-vs-keep* verdict for moving data (`GPU_N/M/R/CLUSTER` env).
- **`gpu_visibility_bench`** — GPU **line-of-sight** over static occluders (the
  *clean* GPU case: build once, no rebuild). Segment-vs-AABB traversal in a compute
  shader, verified exactly against the CPU `segment_hit`; ~1380× the serial CPU LoS.

`gpu_lbvh_demo` and `gpu_storm` are also **published to the web** (WebGPU) — see the
[live demo index](https://orlandoluque.github.io/vectorial-hash-kit/).

## Try the demos

Three ways, easiest first:

### 1. In your browser (nothing to install)

The 2D and 3D critters run as a WebAssembly page — just open
**https://orlandoluque.github.io/vectorial-hash-kit/** and click a demo.
Click the canvas to give it keyboard focus. *(Rebuild with `scripts/build-web.sh`;
served from `main` `/docs` via GitHub Pages.)*

### 2. Download a prebuilt program (no Rust needed)

Grab a ready-to-run file from the
[**Releases**](https://github.com/OrlandoLuque/vectorial-hash-kit/releases)
page (Windows `.exe`), download it, and double-click. No build step.

### 3. Build from source (any OS)

You need **Rust**. If you don't have it, install it in one step from
**https://rustup.rs** (it gives you the `cargo` command used below). Then:

```bash
# 1. get the code
git clone https://github.com/OrlandoLuque/vectorial-hash-kit.git
cd vectorial-hash-kit

# 2. run a demo (the first build downloads dependencies + compiles — a few minutes)
cargo run -p vectorial-hash-demos --bin critters   --release   # 2D
cargo run -p vectorial-hash-demos --bin critters3d --release   # 3D
```

`--release` matters for the visual demos (they push a lot of geometry per
frame). `cargo` fetches every dependency and builds everything for you; there's
nothing else to install.

**Other binaries** (same pattern, swap the `--bin` name):

```bash
cargo run  -p vectorial-hash-demos                                   # console demos 1 + 2 (no window)
cargo run  -p vectorial-hash-demos --bin critters_headless   --release   # 2D stats (no window)
cargo run  -p vectorial-hash-demos --bin critters3d_headless --release -- --sweep   # 3D decision map
cargo build -p vectorial-hash-demos --release                        # just compile everything
```
