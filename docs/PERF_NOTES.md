# Performance notes — siege demo

Measured findings + the FPS-optimisation review. **Applied** = a free win, already
in. **Reported** = a quality-for-FPS trade-off left for the user to decide.

## Boids separation: precomputed force table vs live maths (measured)

The separation force is a pure function of the relative offset `(Δx, Δz)`, so it's
tabulable — the project's template idea applied to steering. We built it (per
`(faction,kind)` grid, offset → `(Fx, Fz)`) behind `$SIEGE_BOID_TABLE=1` and
benchmarked it (`bench_boid_force_table`, ~2000 units, this machine):

| combo | per-pass |
| --- | --- |
| index (k-NN) + **maths** | **3.5 ms** |
| index (k-NN) + table | 4.8 ms |
| no-index O(N²) + **maths** (N=400) | **0.20 ms** |
| no-index O(N²) + table (N=400) | 0.42 ms |

**The table is slower.** Counterintuitive, but it's the *memory wall*: a few
divisions are cheap on a modern FPU (tens of cycles, pipelined / out-of-order),
while the lookup costs a memory load — and the big-unit grids (a dragon's
`sep_dist ≈ 48` → a ~75 KB grid) miss the cache. Compute got cheap; latency
didn't. **Verdict: default to the live maths** (kept). The table stays behind the
flag as a demonstration. (A precomputed table *does* win when the kernel is
expensive — many ops / transcendental — or the table is tiny and hot; this one is
neither.)

The two axes still compose, just not as hoped here: the **index** cuts the
*number* of force evals (N² → k·N — the big win), the **table** would cut the
*cost* of each (a loss here).

## Voxel vs smooth terrain: why `V` changes the FPS

Pressing `V` (voxel ↔ smooth heightfield) noticeably moves the frame rate — e.g.
~50 → ~55 fps switching **to** smooth. That's expected, and it's a pure **GPU
geometry** difference, not a simulation one (both read the same crater-aware
`ground_height`; the sim is byte-for-byte the same):

- **Voxel** (`build_voxel_chunks`, the default) — each cell is a flat-top quad
  **plus up to 4 cliff-wall quads** wherever a neighbour sits at a different
  height, plus baked corner AO. Blocky and readable, but **several times the
  triangles** (and vertices) of the smooth mesh.
- **Smooth** (`build_terrain_chunks`) — one quad per heightfield cell, shared
  vertices, no walls. Far less geometry.

So switching to smooth **sheds vertex + rasterisation work** → the ~10 % bump.
Two takeaways:

1. **It's normal** — the lighter mesh renders faster; nothing is wrong.
2. **It's a signal you're partly GPU-bound** at that population/zoom. If the frame
   were CPU-bound (the sim), swapping the *terrain mesh* wouldn't move the needle.
   (Both meshes are static — rebuilt only on a crater remesh — so this is per-frame
   *draw* cost, not build cost.) The terrain-mesh-resolution lever below trades the
   same way.

## FPS-optimisation review

The demos are already well-structured for FPS (parallel `decide`, draw-calls
bounded by phase-groups in macroquad, real GPU skinning in wgpu, static terrain
meshes). So there are no huge *free* wins lying around — but a few clean ones, and
a list of bigger levers that cost quality (left for you to decide, per your note).

### Applied (free — no quality cost)

1. **`apply`: dedup `terrain_height(nx,nz)`** — it was computed twice per ground
   unit (inside `ground_height` *and* the lava check). Compute once, reuse. Saves
   one heightfield eval per ground unit per frame.
2. **wgpu render: feet from `u.p.y − radius`, not a recompute.** The render was
   calling `ground_height` per unit — a heightfield eval **plus a loop over up to
   64 craters** — to find the feet, when the sim already put the (crater-aware)
   centre in `u.p.y`. Now it just drops the centre by the radius, like macroquad.
   Removes a full `terrain_height + crater-loop` per unit per frame from the
   render. (Biggest of the free wins, especially with many craters / big armies.)
3. **Keep the unit index instead of rebuilding it (`sync_index`).** We used to
   `clear()` + `insert` the whole `Tree3` from live positions every frame. Now we
   *keep* the tree and `update_ref` each unit in place — O(1) when it stayed in its
   leaf (units drift a fraction of a cell/frame, so most do), relocate only on a
   boundary cross, `remove_ref`/`insert_ref` on death/respawn. **Measured
   ~1.06× (1 thread) → ~1.4× (12–16 threads)** on the full CPU frame, verified
   byte-identical to the rebuild (`siege_cpu_bench`; full three-way write-up in
   [PARALLEL.md](PARALLEL.md) § "rebuild vs keep"). This **supersedes** the old
   "marginal, not applied" note below — the earlier guess that the rebuild wasn't
   the bottleneck was wrong: it's the serial Amdahl tail, and shrinking it lifts
   the whole thread-scaling curve. (A parallel `bulk_load_par` *rebuild* was an
   intermediate step, ~1.1–1.16×; keep beats it and needs no threads, so wasm wins
   too. `bulk_load` stays a library primitive for from-scratch static builds.)

### Reported (FPS for a quality / complexity trade-off — NOT applied)

- **Frustum-cull units in the render.** Skip building/drawing instances for units
  outside the view. *Free quality-wise* (off-screen = invisible), but: big win only
  when **zoomed in** (when zoomed out — your usual view — every unit is on screen,
  so it's pure overhead), and a too-tight margin **pops** units at the edges. ~30
  lines × both binaries + a safe margin. Worth it if you play zoomed in a lot.
- **Terrain mesh resolution.** Smooth RES is 250 (macroquad) / 260 (wgpu) — I
  raised it so crater bowls read round. Dropping to ~180 would shave vertex work
  (cost: slightly more faceting on craters). Could be adaptive (fine only near
  craters). Est. small (terrain isn't the bottleneck; it's static).
- **macroquad baked animation frames** (`ANIM_FRAMES = 12`). Fewer baked frames =
  fewer distinct GPU meshes + draws, but choppier walk cycles. e.g. 8 frames ≈ −⅓
  of the animation draw groups. wgpu is unaffected (one rest mesh, GPU-skinned).
- **k-NN neighbours `k = 16`** in `decide`. Lower `k` (e.g. 12) speeds every unit's
  query, but the boids see fewer neighbours (looser flocking). Behaviour change.
- **Smoke as instanced billboards.** Minor — smoke is already capped (240 puffs).

