# Backlog (to triage)

Future work queued for review. Nothing here is committed scope — prune,
reprioritise, or drop freely. Items graduate into the per-crate roadmaps
(e.g. `crates/vectorial-hash/README.md`) once picked up.

## Overnight queue — round 3 (set 2026-06-26)

The big **ray-cast thread graduated** (DDA leaf-walk 2D + 3D, capsule shapes
with analytic `classify_box`/`classify_aabb`, `raycast_first` early-exit, exact
thick band, ropes-maintenance ledger, SoA batch narrowphase, 2D `MortonGrid`,
3D `Tree3::raycast_dda`, and 2D + 3D decision maps) — all in `docs/RAYCAST.md`.
What's left, grouped. `[review]` = needs a human visual glance; rest is
autonomously verifiable. User picked **"Todo (A–E)"** + the headline below.

**ORDER (user, 2026-06-27): multithreading FIRST, then `siege`, then the rest.**

**★ PHASE 0 — Multithreading (before the demo).** The `siege` demo is the heavy
parallel consumer (1k+ units, each doing read queries on the shared index per
frame). Lay the parallel infra first so the demo is multithreaded from the start:
- **Parallel per-unit AI pattern** — `units.par_iter_mut().for_each(|u| { … read
  index … })`: reads are `&self` + `Sync`, units mutated disjointly → safe, no new
  API needed. Confirm + document it (this is the demo's lever).
- **Parallel batch `knn` / `raycast`** (the read-fan-out, like `cull_many_par`).
- **Parallel tree bulk-load** (sort-then-link) for fast per-frame rebuilds (C9).
- **Properly benchmarked + tested** (user, 2026-06-27): interleaved min-of-N vs
  the serial path (the `raycast_compare` anti-contamination methodology — rotate
  order, min, `noise` column, two runs), and correctness tests (parallel result
  == serial). Document the crossover (when threads pay vs the fork overhead).

**★ PHASE 1 — `siege` demo (procedural 3D medieval battlefield)** `[review]`
The flagship showcase: nearly every battle mechanic *is* a spatial query we
built. New bin `vectorial-hash-demos/src/bin/siege.rs`.
- **Procedural terrain** — heightfield (noise) + streams/rivers with **bridges**
  (index choke-points), **forests** (block line-of-sight), a central **volcano**
  (hazard + landmark), **two castles in opposite corners** (per-faction spawns).
- **Two factions** sally from the castles and clash in the middle; density
  evolves sparse-corners → dense-melee (the decision map, live).
- **Troop roster → library feature**: foot soldier = k-NN nearest enemy + short
  cull; knight (cavalry) = **thick raycast** sweep along the charge; archer =
  **`raycast_first`** (LoS + first hit), volleys via **`cull_many_par`**; ballista
  = **all-hits raycast** (pierces a line); catapult = **sphere cull** AoE on
  impact; mage = sphere cull + **chained k-NN** lightning; **dragon** (flies, 3D)
  = **`Polyhedron3` cone / capsule** fire breath; healer = friendly k-NN. Plus
  1k+ units relocating per frame = the `update_ref`/rebuild maintenance stress.
- **Unit AI** (user, 2026-06-27) — a per-unit state machine: advance → acquire
  target (k-NN nearest enemy in range) → attack (the type's query) → morale /
  flee / regroup. Emergent but simple; runs in the parallel per-unit loop.
- **Formations = boids** (user: "el de las abejas" = Reynolds **flocking**:
  separation + alignment + cohesion) — *another* index showcase, each boid
  queries its neighbours via **k-NN / radius cull**. Cavalry & soldiers flock;
  archers hold rigid ranks.
- **Smoke = dynamic line-of-sight blockers** (user) — smoke clouds (catapult
  impact, mage spell) in a separate index; an archer's `raycast` that hits smoke
  before the target → no LoS, temporary (the cloud dissipates). Dynamic
  obstacles + raycast.
- **Render**: reuse `critters3d`'s GPU instancing; one shape+colour per
  type/faction. Orbital camera.
- **Publish to the web** (user, 2026-06-27) — also build `siege` for wasm and
  deploy to the GitHub Pages site (like `critters`/`critters3d`): a card on
  `docs/index.html` + `docs/siege.html` + `docs/siege.wasm`. wasm has no threads,
  so the AI runs **serial** under `cfg(target_arch = "wasm32")` (the parallel
  `par_iter` path is native-only). Build via `scripts/build-web.sh`.
- **Live thread-count slider** (user, 2026-06-27) — the demo runs the per-unit
  AI inside a resizable `rayon::ThreadPool`; a screen-space slider sets
  `num_threads` (1..=cores) live, so the fps response *shows* the parallel
  scaling. Native only — hidden on wasm (no threads → serial). **Done** in the
  foundation.
- **Phased (overnight = the foundation)**: terrain+camera+render → factions+spawn
  +advance → troop types+attacks → combat (damage/death/clash). Visual polish +
  balance is iterated with the user (the artistic side needs their eyes).
  **Foundation built (2026-06-27):** value-noise terrain + volcano, two castles,
  Red/Blue armies (soldier/archer/knight/dragon) spawning + advancing + clashing,
  per-unit AI on the parallel decide→serial-apply split, deaths + respawns,
  instanced render, orbit camera, thread slider. The decide pass now runs three
  library queries per unit: **`knn`** (nearest enemy *and* nearby friends in one
  pass), **boids** separation+cohesion flocking for ground melee (from those
  friends), the dragon's **`Sphere3` `cull`** AoE, and the archer's thick
  **`raycast`** line-of-fire (first unit struck blocks the shot). Seven troop
  types now, each a distinct query: + catapult (**wide `Sphere3` cull** AoE),
  ballista (**all-hits `raycast`** pierce — doesn't stop at the first hit, vs the
  archer), mage (**chained `knn`** lightning, 4 links). Smoke = dynamic LoS
  blockers **done**: catapult/dragon strikes spawn puffs into their own `Tree3`
  (capped, aging out ~3.5s); archer/ballista `raycast_dda_first` it and a puff
  in the line blocks the shot (same parallel-safe emit pattern). Healer **done**:
  carries unit health in the index item, seeks its most-wounded comrade
  (friendly `knn`), heals it as *negative* damage (capped at full HP in apply).
  **Eight troop types, all eight a distinct query.** **Published to the web**
  (2026-06-27): `docs/siege.html` + `docs/siege.wasm` (626 K) + index card; the
  wasm AI runs serial (no threads).

**Siege — status (2026-06-28, after the rebuild night).** The 2026-06-27 visual
feedback is all addressed and several more rounds landed. **Done:**
- **Look.** Smooth lit heightfield → now a **voxel (blocky) terrain** by default
  (flat-top cells + cliff walls + baked corner AO + the quad-flip fix;
  `$SIEGE_SMOOTH=1` for the old mesh). Glowing crater + lava flow carried over.
- **Visible combat.** Transient `Fx` effects (arrow/bolt streaks, AoE rings,
  lightning, heal sparks) + a **projectile system** (arcing cannonballs with
  travel time; volcano lava bombs) — combat reads now.
- **Real models + animation.** Every (faction, kind) is a **Quaternius glTF
  model** (pirates Red vs undead Blue), with **baked skeletal animation**
  (walk / attack / idle-for-riders), cavalry, and a **Castle model** per keep.
- **World.** Procedural **rivers** carved into the height (units wade at 0.4×),
  **bridges** at the crossings, lava burns ground units, the volcano erupts.
- **Controls.** A **population slider** (army size per side) beside the thread
  slider. Random map each run.
- **wgpu twin.** A second binary (`siege_wgpu`) renders the battle with wgpu and
  **real GPU skeletal skinning** (the thing macroquad's WebGL1 can't) — see
  SIEGE.md. Native only so far.

**Siege — 2026-06-29 parity night (landed).** The wgpu twin went from a skeleton
to a real parity demo, plus several queued items: the whole sim **lifted into a
shared `siege_sim` lib module** (both binaries can't drift); the **cannon-facing
fix** (artillery aims forward while kiting); the wgpu binary now runs the shared
sim with **real per-(faction,kind) GPU-skinned models** (static fallback for the
cannon/castle), **combat effects + projectile markers** (LineList), **castles**,
the **voxel terrain + `V` smooth/voxel switch**, `[ ]` **population**, **pause**,
and **HUD stats** in the title. macroquad got **boids alignment**, a "who's
leading" HUD line, and **alterable voxel terrain** (impact craters + the live `V`
switch). A 2D **`Circle` shape** was promoted to the lib (analytic `classify_box`
+ SoA batch, brute-force gated). Web republished.

**Siege — 2026-06-30 fixes night (landed, after the user ran the wgpu build).**
All five reported bugs fixed: wgpu terrain now alters on impact, the projectile
got a real sphere model (was line crosses), on-screen counters + a draggable
population slider in wgpu, macroquad **smooth** mode deforms too, and units no
longer float over craters — root cause was that craters were render-only, so they
were **lifted into the shared sim** (`Craters`, one source of truth for unit feet
+ both meshes). Plus: a **live thread-count slider** in wgpu, the **slime tweak**
applied in wgpu, **per-pixel lighting** (terrain + units — no more Gouraud facets
on the volcano / big models), a **smooth elevation colour ramp** (no contour steps
/ patchy sand along the river), and the slider/HUD overlay. Perf: a precomputed
boids **separation-force table** behind `$SIEGE_BOID_TABLE=1` — *measured slower*
than the live maths (the memory wall; default stays maths) — and two free
heightfield-eval dedups. See **docs/PERF_NOTES.md** (the force-table finding + the
FPS review: free wins applied, quality-trade-off levers reported).

**Siege — still remaining:**
- **`siege_wgpu` on the web** (wasm-bindgen + WebGPU/WebGL2 — toolchain work, user
  asked twice) and **slim `siege.wasm`** (fetch-load the glTF models).
- **Optional "classic Reynolds" boids** behind a flag (inverse-square separation,
  arrival, inertia) — offered; needs the user's eyes to judge "more organic".
- **Frustum-cull units** in the render (free quality-wise, situational; PERF_NOTES).
- **wgpu world polish**: bridge decks + path-to-bridge AI, optional unit-feet snap
  to voxel tops (user chose smooth-float for now; keep it a switch).
- **Forests as LoS cover** (trees block arrows/bolts via a raycast index, like
  smoke) + a **balance** pass (use the HUD).

**Thread-slider retrofit for the critters demos** (user, 2026-06-27) — the same
live `num_threads` slider in `critters` (2D) and `critters3d` (3D), but it only
*moves* in the **combat** mode (one `cull` per critter = many queries/frame →
`cull_many_par` inside the sized pool). In *observe* mode (a single vision cull)
threads can't help — and that contrast is the point: the slider makes the
measured crossover (`PARALLEL.md`) visible. Native only; hidden on wasm.

**A. Ray-cast / structure follow-ups (continue the thread)**
1. **`Octree3` DDA** — `raycast_dda` + `raycast_dda_first` on the 8-way octree
   (Probe-style, like `Tree3`). Brute-force gated (⊆ capsule + first==nearest).
2. **Nudge-free 3D walk** — real 3D neighbour-finding (Samet ascend-to-LCA, or
   ropes) so the `Tree3`/`Octree3` DDA steps without the `locate`+epsilon nudge.
   (The 2D ledger says ropes rarely pay upkeep → Samet first.)
3. **Promote a 2D `Circle` `Shape` to the lib** (analytic `classify_box`, like
   `Capsule`) — the examples each redefine a `Disc`; one lib primitive dedups
   them and gives users a tight 2D circle cull.
4. **Raycast in the Criterion suite + regression baseline** — add `raycast` /
   `raycast_first` / capsule ops so perf regressions are tracked.

**B. Correctness / robustness**
5. **Property / fuzz tests** (proptest) — randomized insert/update/remove/cull/knn
   vs brute force with shrinking, across all seven structures.
6. ~~**Serialization for all structures**~~ — **done.** `serialize`/`deserialize`
   now on `Tree`, `QuadTree`, `IntegerTree`, `Octree3`, `MortonGrid` (2D),
   `MortonGrid3` (mirroring `Tree3`), sharing the LE byte-IO helpers in
   `serde_io.rs`. Each round-trips the exact arena + free-list + `ItemRef`
   handles (grids: world+levels+buckets), gated by a per-structure round-trip
   test (arena counts + cull + knn identical + corruption rejected). `Tree`'s
   `neighbors` ropes are *rebuilt geometrically on load* (not stored), so the
   format is feature-independent.
7. **Templates CI cleanup** — fix the ~21 clippy lints + make the fingerprint
   test cross-platform (deterministic float formatting + sorted iteration), then
   move templates/cli back into the **hard** CI gate (currently advisory).

**C. Performance**
8. **Morton multi-level linear octree** — coarse occupancy levels so a big-radius
   cull skips empty blocks instead of scanning every fine cell (2D + 3D).
9. **Parallel tree bulk-load** — sort-then-link build for static datasets
   (`Tree`/`Tree3`), the parallel build the trees lack (Morton has `extend_par`).
10. **SoA *permanent* leaf storage** — *low priority*: the batch kernel is done
    and measured marginal (~1.0–1.26× end-to-end, descent-dominated); permanent
    storage would only save the materialisation copy. Park unless a workload needs it.

**D. Docs / DX**
11. **Sync the lib README** — it predates the ray-cast surface, `MortonGrid` (2D),
    `Capsule`, and now lists fewer structures than exist (seven). Refresh the
    public-surface table + add a ray-cast paragraph.
12. **Performance guides** — `target-cpu=native`, AoS/SoA + SIMD, threading, the
    opt-in API story (deferred section below has the source material).

**E. Demos / web [review]**
13. **[review] `MortonGrid` (2D) in the `critters` demo** — add the flat grid to
    the 2D structure toggle (+ optionally `IntegerTree` to `decision2d`).
14. **[review] 2D k-NN / line-of-sight demo** — visualise k-NN and the ray-cast
    first-hit (the DDA corridor) in the 2D `critters` demo.
15. **[review] Web touch controls + stats HUD** — orbit/zoom by touch on the 3D
    web demo; an FPS/stats overlay in the browser.
16. **[review] 2D demo panel** — figure-size slider + port the 3D `Panel` into a
    shared `src/panel.rs`.

**Deferred (design / user-held)** — template not-found diagnostics + next-size-up
fallback (rethink); crates.io publish (until more complete). See sections below.

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

> **Local handoff:** a gitignored `CLAUDE.md` at the repo root summarises the
> project state, conventions, and what's done vs. open — for future sessions on
> this machine. (User asked to add it + gitignore it.)

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
5. ~~**Bulk-build parallel**~~ — **done (Morton).** `MortonGrid3::extend_par`
   (feature `parallel`): per-item quantise+Morton-encode on rayon, then a serial
   bucket group. The grouping is the serial tail (Amdahl), so the win is the
   encode for large `N`; pair with `clear()` for a cheap parallel
   rebuild-per-frame. Tested identical to serial insert (count/cells/cull).
   *Remaining:* a parallel **tree** bulk-load (sort-then-link) is a different
   algorithm and is left as future work.
6. **k-NN parity** — `knn` exists on `Tree3`/`Octree3`; add it to `Tree`,
   `QuadTree`, `IntegerTree`, and `MortonGrid3` (the latter ring-by-ring, shared
   with item 3). Brute-force gated.
7. ~~**Ray-cast / segment query**~~ — **done, both approaches + comparison.**
   - **Capsule** (`Segment3` `Shape3` + `Tree3::raycast`, 3D): the ray as a
     thickened segment over `cull`. Exact "items within r", reuses culling.
   - **DDA leaf-walk** (`Tree::raycast(.., walk)`, 2D — the user's `TestDraw`
     idea adapted to variable cells): Amanatides–Woo over the tree, walking only
     the cells the centre ray crosses, neighbour step via selectable
     `WalkNeighbors` (Samet/Probe/Ropes). Returns sorted hits + traversal stats.
     The 3 methods agree (test); `examples/raycast_compare.rs` benchmarks them
     vs the capsule. Findings (`docs/RAYCAST.md`): DDA ~8–15× faster on the thin
     corridor; coverage falls with radius (97%→10%) so DDA=thin rays /
     capsule=thick; ropes < samet < probe on query time (same walk, just
     neighbour cost). **Open:** the maintenance side of ropes (build/update with
     vs without) to settle the overall trade-off; and a thick-corridor exact
     DDA + a hard `raycast_first` with early-exit.
8. ~~**Frustum as a first-class `Shape3`**~~ — **done.** A view frustum is just
   6 half-spaces, i.e. a `Polyhedron3`; added `Polyhedron3::from_corners([Point3;
   8])` (near face then far face, bl/br/tr/tl) that derives the six inward face
   planes (oriented against the corner centroid, so winding need not be exact).
   `tree.cull(&Polyhedron3::from_corners(corners))` culls a camera frustum.
   Tested: an axis-aligned box's corners recover its faces exactly.
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
- **Performance guides** — when written, cover: how a library user opts into the
  demo's optimisations (mostly good defaults + a few opt-in API choices, not
  manual threading); **`-C target-cpu=native`** (the cheapest perf knob — unlocks
  AVX / wider auto-vectorisation, ~2–10× on number-crunching, but non-portable so
  local/bench/self-hosted only — see `docs/RAYCAST.md` § Concepts); and the
  AoS/SoA + SIMD story (the narrowphase-ceiling microbench). *Deferred to "when we
  stop adding so many new things" (user, 2026-06-25).* Source: the chat answers +
  `THREE_D.md` + `RAYCAST.md` (Concepts).
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
- ~~**Index serialization**~~ — **done for all structures.** `serialize` /
  `deserialize` round-trip the built index (exact arena + free-list + `ItemRef`
  handles, no rebuild) to any `std::io::Write`/`Read`, dependency-free, items via
  a caller closure (works for any `T`). Now on `Tree`, `QuadTree`, `IntegerTree`,
  `Tree3`, `Octree3`, and the Morton grids (`MortonGrid` 2D + `MortonGrid3` store
  world+levels+occupied buckets, no arena). Shared LE byte-IO in `serde_io.rs`;
  each a versioned format (magic `VHT2`/`VHQ2`/`VHI2`/`VHT3`/`VHO3`/`VHM2`/`VHM3`
  + version byte). Per-structure round-trip test: arena counts + cull + knn
  identical + corruption rejected. `Tree`'s `neighbors` ropes are derived state,
  rebuilt geometrically on load, so the format doesn't depend on the feature.

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
