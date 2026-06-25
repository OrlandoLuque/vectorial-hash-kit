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

## Active queue (next)

1. **2D demo panel: figure-size slider + 3D-style sliders** — give the 2D
   `critters` panel a slider for the **attack figure size** (rebuild the arsenal
   via `build_arsenal_scaled`, ~0.45 s, so step on release not on drag), and
   port the 3D demo's custom `Panel` (sliders with `[-]`/`[+]` and right-click
   keyboard entry) to replace the basic `root_ui` sliders. Best done by
   extracting the 3D `Panel` into a shared `src/panel.rs` so both demos use it.
   *(Requested after the world-size stepper made fixed-world-unit figures look
   huge in small worlds / tiny in large ones — the slider lets the user
   compensate.)*
2. **Multi-query cull + multithreading (rayon)** — the next-batch experiment:
   parallelise the per-frame index maintenance (`update_many`) and/or batch
   culls with rayon, and **establish under what workload / N it pays vs. the
   thread overhead** (the stress test showed the live demo is CPU-bound on the
   single-threaded `update`, so this is the headline lever). Quantify the
   crossover and where to *avoid* rayon (small N). See Index/algorithms below.

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
- **Structure decision-map sweep** — extend `critters3d_headless` to all four
  structures (binary / octree / Morton / projection — it's binary-vs-octree
  today) and sweep world × population × vision × `item_limit`, reporting the
  winner per cell. Makes the "which wins, and when" synthesis in `THREE_D.md`
  quantitative (today it's by-eye from the live demo). The persistent-vs-rebuilt
  build-cost asymmetry (binary/octree `update` vs Morton/projection rebuild) is
  the dominant effect to capture; the `C` cull-rep readout isolates the cull.
- ~~**`Octree3::update` (ascend-to-LCA)**~~ — **done.** `Octree3` now has the
  same ascend-to-LCA `update` as `Tree3` (churn-tested), and
  `critters3d_headless --structure both` compares the *dynamic* octree vs the
  binary tree on one deterministic run (octree's update is ~5–15% faster; cull
  ≈ equal; id sets identical). The live demo's `M` toggle keeps a persistent
  octree too. See `THREE_D.md` § "Dynamic octree vs binary".
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
