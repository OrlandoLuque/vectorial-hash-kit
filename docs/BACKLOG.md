# Backlog (to triage)

Future work queued for review. Nothing here is committed scope — prune,
reprioritise, or drop freely. Items graduate into the per-crate roadmaps
(e.g. `crates/vectorial-hash/README.md`) once picked up.

## Overnight queue (set 2026-06-25) — ✅ all done

1. ~~Morton / Z-order (linear octree)~~ — done (also now selectable in the
   `critters3d` `M` toggle).
2. ~~k-NN queries~~ — done.
3. ~~Index serialization~~ — done (`Tree3`).
4. ~~World-size 2D stepper~~ — done (visual scaling review → see active queue).
5. ~~Instancing stress test~~ — done.

## Overnight queue — set 2026-06-25 (round 2, all four selected)

Worked in this order: autonomously-verifiable first, web polish last (its
mobile look needs a morning visual review). Commit + push per task.

1. ~~**rayon / multithreading**~~ — **done.** `parallel` feature → rayon-backed
   `cull_many_par` on all five structures (serial `cull_many` always available).
   Finding: *reads fan out, writes don't* — batch culls parallelise cleanly
   (the relocation pass mutates and stays serial; its lever is `ItemRef`).
   Measured crossover in `critters3d_headless --parallel`: ≤4 queries never pays
   (fork ~15–30 µs), 16 pays at ≥20k points, 64+ pays always (up to ~8× at 1024
   queries / 100k points). Full table + guidance in `docs/PARALLEL.md`.
2. **Morton extras** — `MortonGrid3::knn` (ring-by-ring expanding shell, like
   the tree k-NN but over Z-order cells) + a **multi-level linear octree** layer
   (one hashed level per cell size) so Morton can answer big-radius culls without
   scanning every fine cell. Brute-force gated tests.
3. **Criterion benches + CI** — a `benches/` Criterion suite (cull, update,
   update_ref, knn across the five structures) + a GitHub Actions workflow
   (fmt + clippy + test on push). Establishes regression tracking.
4. **Web demo responsive + hide-UI button** — 2D demo adapts to window size;
   a button that hides everything except the demo and itself (toggle to restore)
   for mobile. Build-verified here; **the on-device mobile look needs the user's
   morning review.**

### Deferred (not in the overnight queue)
- **2D demo panel: figure-size slider + 3D-style sliders** — give the 2D
  `critters` panel a slider for the **attack figure size** (rebuild the arsenal
  via `build_arsenal_scaled`, ~0.45 s, so step on release not on drag), and port
  the 3D demo's custom `Panel` into a shared `src/panel.rs`. *(Lower priority;
  the user didn't select it for this round.)*

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
- **Full 3-projection vs expensive narrowphase** — `THREE_D.md`'s open item:
  run the projection methods (not just 1-proj) against a many-faced
  `Polyhedron3` to find where the tight broadphase finally pays for itself.
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
- ~~**Index serialization**~~ — **done for `Tree3`.** `Tree3::serialize` /
  `deserialize` round-trip the built tree (exact arena + free-list, no rebuild)
  to any `std::io::Write`/`Read`, dependency-free, items via a caller closure
  (works for any `T`). Round-trip test preserves cull + knn + arena and rejects
  corruption. Follow-up: the same ~60-line pattern for `Octree3` and the 2D
  trees (`Tree`/`QuadTree`/`IntegerTree`); a versioned format already in place
  (magic `VHT3` + version byte).

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
