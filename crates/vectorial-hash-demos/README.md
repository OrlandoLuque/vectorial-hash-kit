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
- **Three structure modes** (`M` cycles): the binary-split tree, the reference quadtree, or **both at once**. In dual mode every operation — insert, remove, update, attack cull, vision cull — runs on both structures with identical inputs; their cull results are compared live (an "agree" indicator turns red on any mismatch), the sliders rebuild both identically, and the quadtree's subdivision is overlaid as cyan outlines on top of the binary tree's coloured regions.
- A third column plots **live performance graphs** (one polyline per structure where applicable): frame time, average attack-cull and vision-cull times, per-frame movement/update cost, and per-frame insert+remove cost.
- The simulation runs on a single thread (only the startup bank generation parallelizes), so the graphs show exactly where that thread's budget goes as you scale populations up.

### Demo 4 — critters headless (statistics)

```bash
cargo run -p vectorial-hash-demos --bin critters_headless --release -- \
    --mode both --frames 600 --drifters 400 --hunters 400 --pulsars 400 \
    [--split 3 --merge 3 --dt 0.0167 --seed 42 --fire 1.0 --respawn 2.5 --csv out.csv]
```

The exact same simulation core (shared `sim` module, fully deterministic for a given seed) without a window or vsync: it runs at CPU speed and reports per-structure statistics — mean/p50/p95 of per-frame movement+update, attack-cull and vision-cull averages, and insert+remove cost — plus steps/s, final tree shapes and the live cull-agreement counter in `both` mode. `--csv` dumps per-frame rows for plotting. Determinism means a `binary` run and the binary half of a `both` run produce identical simulations (same kills, same final tree), so cross-structure numbers are directly comparable.

Controls: `1`/`2`/`3` select the spawn brush · left click (or hold-drag to paint) spawns at the cursor · right click removes · `+`/`-` add/remove five at random · `R` cycles region rendering · `[` `]` change simulation speed · `Space` pauses · `Esc` quits.

The **"tuning (live)" panel** adjusts everything while the simulation runs: the tree's **split threshold** (a leaf divides above it) and **merge threshold** (siblings collapse at/below it — set it lower than split for hysteresis; the tree is rebuilt on change), per-kind **population targets** (spawns/removes to match), respawn delay, simulation speed and fire rate. Manual spawns/removals (clicks, `+`/`-`) update the population sliders so both mechanisms cooperate.

`CRITTERS_MAX_FRAMES=N` exits after N frames (CI/smoke runs).

## Run

```bash
cargo run -p vectorial-hash-demos            # demos 1 + 2 (console)
cargo run -p vectorial-hash-demos --bin critters --release   # visual demo
```
