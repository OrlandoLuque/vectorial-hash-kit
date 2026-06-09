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

A live 2D map indexed by a `vectorial_hash::Tree` (item limit 3 per cell). Every live leaf region is filled with its own colour, so you watch the map **divide and merge in real time** as items move.

- **Critters** of three kinds with distinct movement and attacks, all using **precomputed template areas** (a set generated at startup — drop at 24 angles × 16 sub-cell offsets + radial circle — deduplicated via the 8 symmetries and served from a hash-keyed `TemplateIndex`):
  - *drifter* (blue): random walk; fires a drop-shaped area in a random direction.
  - *hunter* (red): chases the nearest non-hunter; fires a drop aimed at it (angle snapped to the precomputed 15° set).
  - *pulsar* (gold): circles around; radial blast centred on itself.
- Attack resolution is a real `Tree::cull` with the template short-circuit; victims are `Tree::remove`d (watch regions merge) and **respawn** a few seconds later (watch regions split).
- Attack areas are drawn from the template cells themselves (bright = `In`, dim = `Maybe`) with the real attack polygon outlined on top (arcs flattened), so you can see exactly how the precomputed grid approximates the shape.
- Kill credit is tracked per attacker kind; hunters show a faint sightline to their current prey.

Controls: `1`/`2`/`3` select the spawn brush · left click (or hold-drag to paint) spawns at the cursor · right click removes · `+`/`-` add/remove five at random · `R` cycles region rendering · `[` `]` change simulation speed · `Space` pauses · `Esc` quits.

`CRITTERS_MAX_FRAMES=N` exits after N frames (CI/smoke runs).

## Run

```bash
cargo run -p vectorial-hash-demos            # demos 1 + 2 (console)
cargo run -p vectorial-hash-demos --bin critters --release   # visual demo
```
