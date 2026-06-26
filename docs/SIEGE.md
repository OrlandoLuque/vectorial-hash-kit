# Siege — every battle mechanic is a spatial query

`siege` is the flagship 3D demo: a procedurally-generated medieval battlefield
where two castles sally their armies and clash in the middle, continuously
(the fallen respawn from their keep). It exists to make one point concrete —
**almost every decision a unit makes is one query against a single shared
`Tree3`** — and to be the heavy parallel consumer that the per-unit AI fan-out
([PARALLEL.md](PARALLEL.md) § "Per-unit AI") was built for.

```bash
cargo run -p vectorial-hash-demos --bin siege --release
```

Live (single-threaded wasm build): <https://orlandoluque.github.io/vectorial-hash-kit/siege.html>

- drag left mouse: orbit · scroll: zoom · `P`: pause · `[` / `]`: rebuild armies
- **thread slider** (top-left, native only): set the rayon pool size *live* and
  watch the fps response — the parallel scaling, on screen. wasm has no threads,
  so the web build hides it and runs the AI serially.

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
cheap serial pass. The index is **rebuilt** each frame from live positions (a
cheap serial insert loop); relocation-by-handle is the alternative lever when the
rebuild dominates ([THREE_D.md](THREE_D.md) § `ItemRef`).

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

## Terrain

A two-octave value-noise heightfield with a central volcano cone (and a crater),
coloured by elevation (water / sand / grass / rock / snow + lava). Two castles in
opposite corners anchor the spawns. Rendered as a coarse grid of cubes today; a
baked mesh, plus rivers/bridges/forests as real choke-points and LoS cover, are
queued in [BACKLOG.md](BACKLOG.md).
