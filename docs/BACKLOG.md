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

## Overnight queue — set 2026-06-25 (expanded round 2)

Worked top-down. Commit + push per task. Items marked **[review]** need a human
visual glance the next morning; everything else is autonomously verifiable.

**Done**
1. ~~**rayon / multithreading**~~ — **done.** `parallel` feature → rayon-backed
   `cull_many_par` on all five structures (serial `cull_many` always available).
   Finding: *reads fan out, writes don't* — batch culls parallelise cleanly
   (the relocation pass mutates and stays serial; its lever is `ItemRef`).
   Measured crossover in `critters3d_headless --parallel`: ≤4 queries never pays
   (fork ~15–30 µs), 16 pays at ≥20k points, 64+ pays always (up to ~8× at 1024
   queries / 100k points). Full table + guidance in `docs/PARALLEL.md`.

**Priority block (user bumped regression baseline to the front, 2026-06-25)**
2. ~~**Criterion benches + CI + regression baseline**~~ — **done.**
   - `benches/spatial.rs` (Criterion: build/cull/update/knn across structures).
   - `examples/regression_gate.rs` — deterministic gate vs a **committed**
     `benches/baseline.tsv`, exits 1 on regression. **min-of-N + a calibration
     loop** (compare op/`_calib` ratios) cut back-to-back variance ±60% → ±6%,
     so it can actually gate. `benches/README.md` documents both tools.
   - `.github/workflows/ci.yml` — clippy (`-D warnings`, gating on the flagship)
     + tests (lib/templates/cli, ±parallel) + bench/example compile + wasm build
     of the web demos. fmt is advisory (the dense hand-style isn't rustfmt-clean).
   - Cleaned 11 pre-existing clippy lints in `vectorial-hash` to make the gate
     pass.
   - *Follow-up:* templates + cli still carry ~21 pre-existing clippy lints
     (advisory in CI for now) — a focused cleanup pass should make them gate too
     (careful: some are `needless_range_loop` where the index is genuinely used).
   - *Follow-up:* `vectorial-hash-templates`' `template_fingerprint_matches_fixture`
     test is **not cross-platform** — the committed fixture was generated on the
     Windows dev box and the fingerprint differs on the Linux CI runner (float
     formatting / iteration order). Advisory in CI for now. Fix: make the
     fingerprint platform-independent (deterministic float formatting + sorted
     iteration), then move templates/cli tests back into the hard gate.

**Original remaining**
3. **Morton extras** — `MortonGrid3::knn` (ring-by-ring expanding shell, like
   the tree k-NN but over Z-order cells) + a **multi-level linear octree** layer
   (one hashed level per cell size) so Morton can answer big-radius culls without
   scanning every fine cell. Brute-force gated tests.
4. ~~**[review] Web demo responsive + hide-UI button**~~ — **done (needs visual
   review).** Pure `docs/` HTML/JS, no wasm rebuild. The 2D demo (fixed layout,
   no `screen_width` use) is centred and **CSS scale-to-fit** (letterbox) so the
   whole thing shows on any screen; mouse stays aligned (miniquad maps through
   the transformed bounding rect, framebuffer size unchanged). The 3D demo
   already lays out from `screen_width`, so it keeps filling the viewport. Both
   pages gain a **⤢ toggle** that hides the chrome and enters fullscreen (tap
   again to restore). **Morning check:** in-browser look on a phone, esp. that
   the 2D panel sliders are still usable when scaled (touch-friendly 2D panel is
   the deferred follow-up).

**Newly added (round 2 — autonomously verifiable)**
5. **Bulk-build parallel** (`from_positions_par`) — the natural follow-up to
   rayon: parallel sort-then-link bulk load for static datasets + parallel Morton
   code computation (then serial group). Already flagged in `docs/PARALLEL.md`.
6. **k-NN parity** — `knn` exists on `Tree3`/`Octree3`; add it to `Tree`,
   `QuadTree`, `IntegerTree`, and `MortonGrid3` (the latter ring-by-ring, shared
   with item 3). Brute-force gated.
7. **Ray-cast / segment query** — visit cells/leaves along a ray (line-of-sight,
   picking). *The user has their own approach (to be shared tomorrow); implement
   a reasonable one now and **compare designs** the next day.*
8. **Frustum as a first-class `Shape3`** — a camera frustum shape so the cull is
   not only spheres (the demo's vision cull could use it).
9. ~~**`clear()` retaining capacity**~~ — **done.** `clear()` on all five trees
   + `MortonGrid3`: resets to an empty root leaf / clears the bucket table while
   keeping the arena/hash-map capacity (the cheap path for per-frame rebuilds).
   Invalidates outstanding `ItemRef`s. Tested on Tree3 + MortonGrid3.
10. **Property / fuzz tests** (proptest) — randomized insert/update/remove/cull
    vs brute force with shrinking, across all structures. Stronger than the
    hand-rolled churn tests.
11. **Serialization for all structures** — only `Tree3` has versioned serialize;
    extend to `Tree`/`QuadTree`/`IntegerTree`/`Octree3`/`MortonGrid3`.
12. **SoA leaf storage + SIMD narrowphase** — store leaf positions contiguously
    for cache-friendly `contains_point` sweeps; explore SIMD on the sphere/point
    test. Measure vs current AoS.
13. ~~**"Choosing a structure" flowchart**~~ — **done.** `docs/CHOOSING.md` (a
    2D/3D decision flowchart + summary table + rules of thumb), linked from the
    lib README. *(User: "interesante, sí.")*

**Newly added (round 2 — [review] / design)**
14. **[review] 2D k-NN / line-of-sight demo** — visualise nearest-neighbour and
    LoS in the 2D critters demo.
15. **[review] Web touch controls + stats HUD** — orbit/zoom by touch on the 3D
    web demo; an FPS/stats overlay in the browser.
16. ~~**CHANGELOG.md**~~ — **done.** Root `CHANGELOG.md` (Keep-a-Changelog),
    `[Unreleased]` capturing this round + a `[0.1.0]` baseline.

### Deferred — explicitly held back
- **Publish to crates.io** — *deferred until the crate is more complete* (user,
  2026-06-25). The crate is already publishable; cut a 0.1.0 when ready.
- **Threading vs no-threading tutorial/guide** — document how a library user
  opts into the demo's optimisations (it's mostly good defaults + a few opt-in
  API choices, not manual threading). *Deferred to "when we stop adding so many
  new things" (user, 2026-06-25).* Source material: the chat answer + `THREE_D.md`.
- **2D demo panel: figure-size slider + 3D-style sliders** — figure-size slider
  (rebuild arsenal via `build_arsenal_scaled`, step on release) + port the 3D
  `Panel` into a shared `src/panel.rs`. *(Lower priority; not selected.)*

### Templates — needs a rethink (design, not yet built)
- **Next-size-up fallback** — when no template exists for a cell size, take the
  next size up with special handling. *(See memory `project_template_application_design`.)*
- **Template-not-found diagnostics / debug mode** — report which cases fell
  through to the per-item point test (coverage gaps), and **bound the cell-size
  range** where the next-size-up fallback fires (the largest cells need not have
  templates). User flagged "other interesting things" here too — wants to give
  the whole template-coverage story a repensada. *(2026-06-25.)*
- **3D / voxel template bank** — extend the 2D template machinery to voxel
  rasters.

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
