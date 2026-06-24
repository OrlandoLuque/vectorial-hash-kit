# Backlog (to triage)

Future work queued for review. Nothing here is committed scope — prune,
reprioritise, or drop freely. Items graduate into the per-crate roadmaps
(e.g. `crates/vectorial-hash/README.md`) once picked up.

## Overnight queue (set 2026-06-25)

Autonomous session, in order (prioritised by what can be finished + self-verified
without the user — brute-force / round-trip gated first; visual-feedback last):

1. **Morton / Z-order (linear octree)** — Index/algorithms below.
2. **k-NN queries** — Index/algorithms below.
3. **Index serialization** — Index/algorithms below.
4. **World-size 2D stepper** — Render/demos below (visual scaling needs review).
5. **Instancing stress test** — Render/demos below (GPU profiling semi-manual).

Everything else in this file is **future / not in the night queue** — left here
to triage later.

## Render / demos
- **Instancing stress test** *(queued tonight, #5)* — GPU instancing + billboards
  now ship in `critters3d` (`G` toggles immediate / instanced spheres /
  billboards, raw miniquad in `src/instanced3d.rs`). Still TODO: push to
  100k–1M critters and profile where it becomes GPU-bound (fill vs vertex vs
  instance upload).
- **World-size 2D stepper** *(queued tonight, #4)* — the 2D `critters` demo has a
  fixed 1024×1024 world (`MAP_W`/`MAP_H` consts in `sim.rs`); give it the same
  stepped pow-2 world-size control the 3D demo has. Refactor of the shared sim
  core: const → runtime field, render scale follows, `IntegerTree` re-bound to
  the new pow-2 size, index rebuilt on change. Largest/riskiest of the queue —
  the headless path (`critters_headless`) verifies determinism/correctness, but
  the on-screen scaling wants a human glance.
- **Colour by leaf depth / structure** — visualise the tree's interior (which
  cells are deep, how the split differs binary vs octree).
- **Batch the sight-lines / combat effects** — the per-line `draw_line_3d` and
  per-attack `draw_sphere_wires` calls are immediate; batch them into meshes
  when many are on screen at once.

## Index / algorithms
- ~~**`Octree3::update` (ascend-to-LCA)**~~ — **done.** `Octree3` now has the
  same ascend-to-LCA `update` as `Tree3` (churn-tested), and
  `critters3d_headless --structure both` compares the *dynamic* octree vs the
  binary tree on one deterministic run (octree's update is ~5–15% faster; cull
  ≈ equal; id sets identical). The live demo's `M` toggle keeps a persistent
  octree too. See `THREE_D.md` § "Dynamic octree vs binary".
- **Full 3-projection vs expensive narrowphase** — `THREE_D.md`'s open item:
  run the projection methods (not just 1-proj) against a many-faced
  `Polyhedron3` to find where the tight broadphase finally pays for itself.
- **Morton / Z-order keys (linear octree)** *(queued tonight, #1)* — a fourth
  structure to compare against `Tree3` / `Octree3` / projection: hashed/sorted
  Morton codes. Added to `tree3d_bench`, gated against brute force.
- **Multi-query cull + multithreading** — cull many spheres at once; parallel
  build/cull (rayon) for large N.
- **k-NN queries** *(queued tonight, #2)* — nearest-neighbour (best-first with a
  bounded heap), not only range cull. Gated against brute-force k-NN.
- **Index serialization** *(queued tonight, #3)* — save/load a built tree.
  Round-trip test (serialize → deserialize → `cull` identical).

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
