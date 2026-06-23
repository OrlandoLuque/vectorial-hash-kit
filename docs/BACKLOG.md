# Backlog (to triage)

Future work queued for review. Nothing here is committed scope — prune,
reprioritise, or drop freely. Items graduate into the per-crate roadmaps
(e.g. `crates/vectorial-hash/README.md`) once picked up.

## Render / demos
- **Instancing stress test** — push the instanced 3D demo to 100k–1M critters,
  profile where it becomes GPU-bound (fill vs vertex vs instance upload).
- **Colour by leaf depth / structure** — visualise the tree's interior (which
  cells are deep, how the split differs binary vs octree).
- **Batch the sight-lines** — when many critters are "seen", the per-line
  immediate `draw_line_3d` calls add up; batch them into a line mesh.

## Index / algorithms
- **`Octree3::update` (ascend-to-LCA)** — the octree has insert/remove/cull but
  no dynamic relocation. Add `update` and compare the *dynamic* octree vs the
  binary `Tree3` (mirrors the 2D update-strategy study).
- **Full 3-projection vs expensive narrowphase** — `THREE_D.md`'s open item:
  run the projection methods (not just 1-proj) against a many-faced
  `Polyhedron3` to find where the tight broadphase finally pays for itself.
- **Morton / Z-order keys (linear octree)** — a fourth structure to compare
  against `Tree3` / `Octree3` / projection: hashed/sorted Morton codes.
- **Multi-query cull + multithreading** — cull many spheres at once; parallel
  build/cull (rayon) for large N.
- **k-NN queries** — nearest-neighbour, not only range cull.
- **Index serialization** — save/load a built tree.

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
