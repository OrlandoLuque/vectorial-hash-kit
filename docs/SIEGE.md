# Siege — every battle mechanic is a spatial query

`siege` is the flagship 3D demo: a procedurally-generated battlefield (random
each run) where two factions — **🏴‍☠️ pirates vs 🧟 undead** — sally their armies
and clash, continuously (the fallen respawn from their keep). It exists to make
one point concrete — **almost every decision a unit makes is one query against a
single shared `Tree3`** — and to be the heavy parallel consumer that the per-unit
AI fan-out ([PARALLEL.md](PARALLEL.md) § "Per-unit AI") was built for. The units
are real **Quaternius (CC0) glTF models with baked skeletal animation** (see
[Rendering](#look--world) below and `assets/siege/CREDITS.md`).

```bash
cargo run -p vectorial-hash-demos --bin siege --release        # macroquad
cargo run -p vectorial-hash-demos --bin siege_wgpu --release   # wgpu (comparison)
```

Live (single-threaded wasm build): <https://orlandoluque.github.io/vectorial-hash-kit/siege.html>

- drag left mouse: orbit · scroll: zoom · `P`: pause · `[` / `]`: ± population ·
  `V`: voxel ↔ smooth terrain · `C`: collisions (boids separation) on/off — A/B
  the look and the perf of "no two bodies overlap" vs. letting them pile
  (see `PARALLEL.md`) (both binaries)
- `siege_wgpu` extras: `F` free-fly camera (WASD move, Q/E down/up, drag to
  look) ↔ orbit · `K` frustum-culling on/off · an on-screen **pause button**
  (tap — works on mobile touch, where the keyboard shortcut can't) · headless
  FPS bench via `SIEGE_BENCH=1` (see `PARALLEL.md`)
- **population slider** (top-left): army size per side — the spatial-index stress
  lever (20…10000 each, i.e. up to 20k units).
- **thread slider** (native only): set the rayon pool size *live* and watch the
  fps response — the parallel scaling, on screen. wasm has no threads, so the web
  build hides it and runs the AI serially.

## The roster — eight troop types, eight queries

Each type is a different library call, so the army reads as a catalogue of the
index's read API:

| Troop | Behaviour | Library query |
| --- | --- | --- |
| **Soldier** | melee the nearest enemy | `knn` (nearest enemy) |
| **Knight** | fast cavalry melee | `knn` + boids steering |
| **Archer** | ranged, one arrow, blocked by anyone in the line | thick `raycast` (**first** hit) |
| **Ballista** | piercing bolt — skewers *everyone* on the line | thick `raycast` (**all** hits) |
| **Catapult** | lobbed boulder, splash damage | wide `Sphere3` `cull` (AoE) |
| **Dragon** | flies; fire-breath AoE | `Sphere3` `cull` (AoE) |
| **Mage** | chain lightning, arcs to 4 enemies | `knn` from the strike point |
| **Healer** | mends the most-wounded comrade | friendly `knn` (heals as negative damage) |

The archer-vs-ballista pair is deliberate: same `Tree3::raycast`, but the archer
stops at the first unit struck (a friend in the way blocks the shot) while the
ballista damages every enemy along the ray — the two raycast modes side by side.

## The AI loop — read in parallel, resolve in serial

A frame is two passes:

1. **decide** (read-only, parallel). Every unit runs its queries on the shared
   index and writes the result — desired velocity, attack intents, a smoke
   emission — into *its own* fields only. Because nothing reads or writes another
   unit, the whole pass fans out with `units.par_iter_mut()` inside a sized
   `rayon::ThreadPool`, with **no new API and no contention**. ~11–12× on 16
   threads (PARALLEL.md).
2. **apply** (serial). Move units, resolve accumulated damage (negative = a
   healer's mend, capped at full HP), kill the fallen, schedule respawns, and
   turn this frame's smoke emissions into puffs.

This decide→apply split is the general pattern for a parallel agent simulation:
the reads parallelise trivially, the cross-unit writes are quarantined into one
cheap serial pass. The index is **rebuilt** each frame from live positions — a
serial insert loop by default, or (native `--features parallel`) a parallel
`Tree3::bulk_load_par`, which fans the rebuild out over the same pool and lifts
the CPU-fps ceiling ~1.14× at high thread counts by attacking the serial-rebuild
Amdahl tail ([PARALLEL.md](PARALLEL.md) § "Parallel bulk-load"). Relocation-by-
handle is the other lever when the rebuild dominates ([THREE_D.md](THREE_D.md)
§ `ItemRef`).

## Two extra index showcases

- **Boids formations.** Ground melee (soldier/knight) reuse the *same* k-NN pass
  that finds their target to also gather nearby friends, and steer with Reynolds
  separation + cohesion — so they advance as a loose band instead of collapsing
  to a point. (Alignment is the next layer.)
- **Smoke = dynamic line-of-sight blockers.** Catapult and dragon strikes spawn
  smoke puffs into their *own* `Tree3` (capped, aging out over ~3.5 s). An archer
  or ballista shot `raycast`s that smoke index first; a puff between shooter and
  target blocks the shot. A second, churning index that exists only to be
  ray-tested — the dynamic-obstacle case.

## Terrain & world

A two-octave value-noise heightfield (**different every run** — a seeded offset;
`$SIEGE_SEED=N` reproduces one) with a central **volcano** cone + crater, rendered
as a smooth lit triangle mesh (lambert shading baked into the vertex colours so
the relief reads), coloured by elevation (water/sand/grass/rock/snow) with an
**emissive lava** crater + flow. Living hazards:

- **Lava** burns ground units that stand in it; the flying dragon is immune.
- **Rivers** are carved into the height field (a seeded meander); ground units
  **wade** at 0.4× speed in the water — a soft obstacle — unless on a **bridge**.
- The **volcano** breathes a constant smoke plume and **erupts** every ~9–16 s:
  a lava spray + smoke burst + real arcing **lava bombs** (the projectile system)
  that land on the slopes and scorch whoever's there.

## Look & world

- **Models.** Each (faction, kind) is a real **Quaternius glTF model** (CC0; the
  Witch is CC-BY — `assets/siege/CREDITS.md`): pirates = Anne / Sharky / Pirate
  Captain / Henry / Witch; undead = Zombie / Skeletons / Slime / Bat; shared
  Dragon + Cannon, told apart by a faction tint. The **knight is cavalry** (rider
  on a horse).
- **Skeletal animation, baked.** macroquad's WebGL1 can't do GPU skinning, so the
  loader **bakes** each clip into N static frames (CPU skinning once at load) and
  the render picks the frame per unit — walk while moving, the **attack** clip
  while striking, idle for mounted riders. Units split into a few phase groups so
  the draw-call count stays bounded regardless of army size.
- **Projectiles.** Catapults lob a visible arcing cannonball (travel time → AoE
  on impact); the volcano spits lava bombs through the same system.

## The wgpu twin (`siege_wgpu`)

A second binary renders the **same battle** with **wgpu** (modern low-level GPU
stack) instead of macroquad, to compare — and to reach the thing macroquad's
WebGL1 stack can't: **real GPU skeletal skinning**. Per distinct model the rest
mesh + a storage buffer of all the animation's per-frame bone matrices are
uploaded once; the WGSL vertex shader skins `Σ wᵢ · bone[frame·J + jointᵢ] · pos`
per instance, so thousands of units animate on the GPU with zero per-frame CPU
skinning. The model's own colours are mixed toward the faction tint in-shader.

Crucially, the whole **simulation is shared** — both binaries run the exact same
`siege_sim` (units, decide→apply, projectiles, the volcano), so they can't drift.
The wgpu side is at functional parity: the real per-(faction,kind) models (static
fallback for the cannon/castle), combat effects + projectile markers (a LineList
pipeline), the voxel terrain + `V` smooth/voxel switch, `[` `]` population, pause,
and HUD stats in the window title. Native only (wgpu/winit aren't in the wasm
demo build); a web build is queued ([BACKLOG.md](BACKLOG.md)).

## Still queued

Web-publishing the wgpu twin (wasm-bindgen + WebGPU) and slimming `siege.wasm`
(fetch-loaded models), bridge decks + path-to-bridge AI on the wgpu side, forests
as LoS cover, and balance. The voxel-terrain technique notes live in
[MAP_DESIGN.md](MAP_DESIGN.md); the live queue is [BACKLOG.md](BACKLOG.md).
