# 3D indexing: true 3D tree vs projection indexing (2026-06-23)

Two strategies for "which 3D points lie inside this query volume?", measured
on time AND precision against a brute-force ground truth. Both are exact
(gated against brute force); they differ in cost and in what infrastructure
they need.

Code:
- `crates/vectorial-hash/src/tree3.rs` — `Tree3<T>` (binary-split 3D tree),
  `Aabb`, `Point3`, `Shape3`, `Sphere3` (analytic AABB classification),
  `VoxelRaster` (the 1×1×1 voxel raster — 3D analogue of the 2D 1×1 raster).
- `crates/vectorial-hash-demos/src/bin/tree3d_bench.rs` — the comparison.

## The two strategies

1. **True 3D tree** (`Tree3`): a binary split in 3D (split the longest of
   the 3 axes), the direct analogue of the 2D `Tree`. The query volume
   classifies each node box green/white/yellow; green takes the whole
   subtree, white skips it, yellow recurses. For a sphere the
   classification is analytic and exact (nearest/farthest box corner vs
   `r`), so no template bank is needed — this is the **best case for the
   3D tree** (a non-analytic shape would need an N³ voxel template, the
   memory wall of true 3D). Leaf items resolve through the 1×1×1
   `VoxelRaster` (In/Out by lookup, Maybe → exact).

2. **Projection indexing** (the author's idea): keep three 2D `Tree`s on
   the (x,y), (x,z), (y,z) projections. Cull each with the sphere's
   circular shadow, intersect the candidate id sets, then run the exact 3D
   test on survivors. Reuses *all* the optimized 2D machinery and has no
   N³ memory — but the intersection is a **broadphase** (a superset; the
   corners of the three-cylinder intersection that stick out of the sphere
   are false positives the exact test drops).

   A **1-projection** variant (added during measurement): cull just *one*
   plane and exact-test its shadow in 3D. Looser broadphase (a bigger
   candidate set) but no second/third cull and no set intersection.

## Results

50k–200k uniform points in a 512³ world, item_limit 8, 300 queries, sphere
radii as noted. All three index methods returned EXACTLY the brute-force
set every time.

### Large query spheres (r 10–80, mean 233 hits/query, low selectivity)

| method | mean ns/query | vs brute | broadphase candidate/true |
| --- | ---: | ---: | ---: |
| brute force | 40,113 | 1.0× | — |
| true 3D tree | 26,245 | 1.5× | (exact) |
| 3-projection (intersect+exact) | 180,303 | 0.22× | **1.12×** |
| 1-projection (+exact) | 38,570 | 1.04× | 6.35× |

### Small query spheres (r 5–20, high selectivity — the realistic case)

| method | mean ns/query (50k) | vs brute | mean ns/query (200k) | vs brute |
| --- | ---: | ---: | ---: | ---: |
| brute force | 39,111 | 1.0× | 145,329 | 1.0× |
| true 3D tree | 2,464 | **15.9×** | 10,391 | **14.0×** |
| 3-projection | 22,827 | 1.71× | 94,818 | 1.53× |
| 1-projection | 5,353 | **7.31×** | 19,678 | **7.39×** |

## Findings

1. **Query size dominates the headline.** With large spheres (low
   selectivity) every method is near brute force — there's little to cull
   when a third of the world is "near" the result. With small, realistic
   spheres the trees pull far ahead. Benchmark 3D culling with
   *representative* query sizes or the numbers mislead (the initial r 10–80
   run made the 3D tree look like a 1.1× win; at r 5–20 it's 16×).

2. **The true 3D tree wins on speed — 14–16× over brute** at realistic
   selectivity, and scales (14× at 200k). Its cost is the structure plus,
   for non-analytic shapes, an N³ voxel template. For analytic volumes
   (spheres, boxes, ellipsoids) no template is needed and it's pure win.

3. **The author's 3-projection idea has the best *precision* but the worst
   *speed* among the index methods.** Its broadphase candidate/true ratio
   is 1.12× — essentially an exact broadphase — yet it runs 0.22–1.7×
   because computing the 3-way intersection of three *dense shadow* sets is
   expensive: each 2D projection of 50k points is dense, so each plane's
   cull returns far more than the final 3D result, and hashing+intersecting
   those large sets dominates. **Tight precision is wasted effort when the
   narrowphase is cheap.**

4. **1-projection is the sweet spot for cheap narrowphase.** Culling one
   plane and exact-testing its shadow gives a looser broadphase (6.35×
   candidates) but runs 7.3× faster than brute — far better than the
   3-projection's 1.5×, because a cheap exact 3D test (a sphere is one
   distance compare) makes the bigger candidate set irrelevant while
   avoiding two culls and two hash sets. It reuses **all** the 2D
   machinery and has no N³ memory.

5. **Practical guidance:**
   - Cheap exact 3D test (sphere, box) + want minimal new code / no N³
     memory → **1-projection** (7×, reuses 2D index).
   - Want maximum speed and can afford the 3D structure (analytic shape, or
     the N³ template for general shapes) → **true 3D tree** (14–16×).
   - Expensive exact 3D test (complex polyhedron, where the narrowphase
     cost is what hurts) → the intuition was that a **tight** broadphase would
     finally pay by minimising the number of exact tests. **Measured (2026-07-19,
     `examples/broadphase_tightness_bench`) — it does NOT, for a tight *volume*
     cull.** See below.

### Does a tight broadphase pay when the narrowphase is expensive? (measured — no)

The open question above, tested directly: query = a convex `Polyhedron3`
(`faceted_ball`, N faces) whose point test costs one dot-product **per face** — so N
*is* the narrowphase cost. Two exact ways to get the points inside it, both
brute-force-verified: **TIGHT** = `tree.cull(&poly)` (the tree prunes each node by
the poly's N half-spaces via `classify_box`, tests survivors by the N-plane
`contains`); **BOX+NP** = `tree.cull(&box6)` by the poly's 6-face bounding box (cheap
prune, over-collects ~82 %) then an exact N-plane narrowphase on the candidates.
Measured (200k points, 120 queries, RTX-class box, ns/query):

| faces | TIGHT poly-cull | BOX6 + narrowphase | winner |
| --- | --- | --- | --- |
| 8 | 90 230 | 89 738 | BOX+NP 1.01× |
| 32 | 171 600 | 126 047 | BOX+NP 1.36× |
| 64 | 251 750 | 178 091 | BOX+NP 1.41× |
| 128 | 414 328 | 263 191 | BOX+NP 1.57× |
| 256 | 942 982 | 718 300 | BOX+NP 1.31× |

**BOX+NP wins at every face count, and the gap *widens* with N** — the opposite of
the intuition. The reason is structural: culling by the tight volume runs the N-plane
`classify_box` **at every visited node**, and a query visits **far more nodes than it
keeps candidates** (~6 k candidates vs a whole descent of nodes). So the tight cull
pays N-per-node while only saving the narrowphase on the ~45 % surplus the box
over-collects — and node count dominates. The cheap 6-plane box prune + an exact
narrowphase on the few survivors is simply less total plane-arithmetic. Lesson: for a
convex query with an expensive point test, **broadphase with a cheap bound and pay
the exact test on the candidates** — don't push the expensive test into the node
prune. (The `VoxelRaster` short-circuit — a 1×1×1 O(1) lookup replacing the N-plane
`contains` — is the further accelerator for that narrowphase, exactly as the
`Polyhedron3` doc-comment notes; it attacks the surviving-candidate cost, not the
node prune.) The author's *2D 3-projection* broadphase is a different point on the
tightness spectrum (cheap 2D shadows, no per-node N-plane cost) — its 1.12× edge in
the earlier table stands; what does *not* pay is buying tightness with a per-node
N-plane volume test.

## 1-projection refinements: z-reject, raster, quadtree (2026-06-23)

Three follow-up experiments on the 1-projection path (`tree3d_bench` gained
`--stack` for a dense-xy "things stacked in height" distribution, plus the
extra methods). All exact vs brute force.

### A z-slab reject between the cull and the exact test is a clear win

The xy-cull returns the *cylinder* (every point in the query circle's
column). A cheap 1D reject — `|z - cz| <= r` — drops the column points
outside the sphere's z-extent *before* the full distance test:

| config | 1-proj (+exact) | 1-proj **+z-reject** +exact | true 3D tree |
| --- | ---: | ---: | ---: |
| uniform 50k, r 5–20, il 8 | 7.7× | **9.3×** | 15.7× |
| stacked 50k, r 5–20, il 48 | 11.3× | **14.9×** | 13.5× |

The z-reject lifts the 1-projection ~20–30%. **At a tuned item_limit
(48) on the stacked distribution it reaches 14.9× — matching/beating the
true 3D tree (13.5×)** while reusing the entire 2D stack and carrying no
N³ memory. This is the strongest case for the projection approach: with
the cheap z-reject and a tuned 2D index it is competitive with the full
3D tree on speed, at a fraction of the code and memory.

### The voxel raster does NOT help the narrowphase for a cheap shape

Using `VoxelRaster::cell_at_world` for the 1-projection narrowphase
instead of the analytic distance test ran at **0.7–0.8×** (slower than
brute): the lookup (floor + index + bounds) plus rebuilding the raster
costs more than one `dx²+dy²+dz² ≤ r²`. This confirms the 2D lesson in
3D — **the raster only pays when `contains_point` is expensive** (a
complex polyhedron, a heavy predicate). For a sphere, analytic wins.

### Keep the quadtree for "stacked in height" worlds

A `QuadTree` projection (4-way split) is retained as an option for worlds
that stack many items in height — the xy-shadow is dense there, the regime
where the quadtree's depth advantage showed up in 2D (BENCHMARKS Results
6). In these tests the binary-tree projection with z-reject still edged it
(stacked il 48: binary 14.9× vs quad 9.8×), so the quadtree did not win at
this density, but it stays available (and likely pulls ahead at far higher
per-column density). The structure choice is a knob, not a fixed decision.

### The 2.5D sweet spot

Pick the projection plane **perpendicular to the world's thinnest axis**
so the unpruned column is as short as possible. For a "2.5D" world (large
in x,y, shallow in z — a great many games), projecting onto x,y makes z
the thin unpruned axis → short cylinders → near-2D performance with full
3D correctness, reusing the whole 2D stack. A bonus property: **pure-z
motion is free in the index** (an item changing only its height doesn't
move in the xy-tree). For such worlds the 1-projection (+z-reject) is
likely the best engineering choice, ahead of building a full 3D tree.

## Micro-optimization noted for the 2D exact test (`make_attack`)

Unrelated to 3D but found while tracing the exact-test path: in the 2D
critters, `make_attack` clones the precomputed rotated polygon and
`move_by(origin)` to translate it into world space, then tests
`poly.is_inside(p)`. The rotation is precomputed per (shape, angle) — only
24 drop rotations — so runtime pays just a clone + O(V) translate per
attack. That clone+translate can be avoided by testing in the polygon's
**local frame**: translate the *query point* by `-origin` and test against
the un-translated precomputed rotated polygon. For an attack that tests N
points (a cull touching N `Maybe` items) this is O(N) point shifts with no
polygon clone, vs O(V) + clone per attack. A small, safe win worth taking
when the dilation / extended-item work touches that path.

## The 1×1×1 voxel raster

`VoxelRaster::for_sphere` builds the 3D analogue of the 2D 1×1 raster:
one In/Out/Maybe classification per unit voxel over the shape's bounding
box, each voxel classified exactly by nearest/farthest-corner distance.
At leaves the per-point test is a voxel lookup; In/Out resolve with no
geometry and only Maybe (boundary) voxels run the exact test. For a sphere
the exact test is already trivial so the raster is mostly a demonstration
of the mechanism; for an expensive `contains_point` (a real use case) it
saves the geometry on all non-boundary leaf items, exactly as in 2D.

Memory note: the voxel raster is N³ for an N-wide shape — the same memory
wall as 3D templates. For large query volumes a parametric voxel scheme
(store, per voxel, the min radius that makes it Maybe / In — the 3D version
of the parametric-circle roadmap item) would answer every radius from one
grid, the way the 2D parametric templates do.

## 3D dynamic workload — `critters3d_headless`

The 3D analogue of the 2D critters movement+cull loop (no combat — just the
index workload): N points move in a 512³ cube, each frame every one is
relocated (`Tree3::update`, ascend-to-LCA) and a sample run a sphere vision
cull. `Tree3` gained `update` (LCA), `remove`, and a merge-up rule for this;
a deep churn test (`cull_matches_brute_after_churn`: 6000 mixed
move/remove/insert ops, item count + cull both gated against ground truth)
validates the dynamic path.

20k critters, 120 measured frames, 100 vision culls/frame, vision r=36:

| item_limit | move+update (µs/frame) | vision cull (µs/cull) |
| ---: | ---: | ---: |
| 8 | 2,883 | 2.25 |
| 32 | 2,150 | 1.55 |
| 64 | 1,945 | 1.43 |
| 128 | 1,820 | 1.39 |

**The 2D `item_limit` lesson carries to 3D**: both update and cull keep
improving with larger leaves through 128 (and the crossover is likely
higher than 2D's ~100 — a 3D cube spreads points over more leaves, so each
boundary leaf holds fewer items at a given item_limit). The exact 3D
optimum is unmeasured; the monotone trend is the finding.

Note (superseded): `Tree3` originally had no arena free-list and churn
left zombie nodes (98k arena nodes vs 3.6k live leaves at il=8). The
free-list now reclaims merged-out slots, so the arena stabilises near the
live count (~7.3k) — same fix as the 2D trees.

## GPU instancing stress test (`critters3d` visual)

How far does the raw-miniquad instanced renderer scale, and where does it
become GPU-bound? Swept via `CRITTERS3D_POP` / `CRITTERS3D_WORLD` /
`CRITTERS3D_RENDER` / `CRITTERS3D_MAX_FRAMES` (each run prints a one-line
`STRESS` summary: mean fps, frame ms, cpu ms, bound). World 1024³, observe mode.

**Live (sim running — per-frame `Tree3::update` of every critter):**

| pop | instanced spheres | square billboards | NO RENDER (CPU ceiling) | bound |
| ---: | ---: | ---: | ---: | --- |
| 50k | 116 fps (8.5 ms) | 117 fps | 128 fps (7.8 ms) | CPU |
| 100k | 51 fps (19.4 ms) | 51 fps | 55 fps (18.2 ms) | CPU |
| 200k | 22 fps (46 ms) | 22 fps | 23 fps (43 ms) | CPU |

**The live demo is CPU-bound through 200k+** — the per-frame index update is the
ceiling (43 ms at 200k), and *all* the rendering of 200k instanced spheres adds
only ~3 ms on top. Instancing is never the bottleneck while the sim runs; the
render mode barely moves the fps (spheres ≈ square ≈ none). So "where does it
become GPU-bound?" has a surprising answer for a dynamic scene: **it doesn't —
the index maintenance dominates first.**

**Frozen (`CRITTERS3D_FREEZE=1`, render-only — CPU≈0, pure GPU throughput):**

| pop | spheres fps (ms) | square fps (ms) | none fps (ms) |
| ---: | ---: | ---: | ---: |
| 200k | 219 (4.6) | 383 (2.6) | 2791 (0.4) |
| 500k | 90 (11.2) | 160 (6.2) | 1788 (0.6) |
| 1M | 47 (21.4) | 72 (13.9) | 1024 (1.0) |

Freezing the sim removes the CPU bottleneck and exposes the GPU breakdown:

1. **Instance upload + transform dominates** (square billboards, 2 tris each, so
   negligible vertex/fill): cost is **linear in instance count** — 2.6 ms at
   200k → 13.9 ms at 1M (~5× for 5×), ≈ **14 ns/instance**. A single instanced
   draw call eats 1M instances at ~72 fps.
2. **Sphere geometry adds ~50%** on top (spheres vs square): +1.9 ms at 200k →
   +7.5 ms at 1M, the per-instance vertex processing of the sphere mesh. Still
   linear, so **vertex-bound, not fill-bound** — the critters are small on
   screen, so overdraw/fill never becomes the limiter at these sizes.
3. The `none` baseline (no draw, vsync off) is 1–2.8k fps — loop + present
   overhead only.

**Takeaway:** the instanced path scales to **1M critters in one draw call** (72
fps frozen for square, 47 fps for spheres); the GPU is upload/transform bound,
not fill bound. For the *live* dynamic demo the GPU has enormous headroom — the
single-thread per-frame `update` is what caps it, so the next win is
parallelising the index maintenance (rayon `update_many`), not the renderer.

## When the 1×1×1 voxel raster pays — crossover by shape cost

The earlier sphere result (raster 0.7–0.8×, slower than analytic) was the
*cheap-shape* regime: a sphere's `contains_point` is one distance compare,
so a memory lookup can't beat it. To find where the raster turns the corner,
`voxel_raster_bench` culls a **`Polyhedron3::faceted_ball`** — a convex
polyhedron of N tangent half-spaces, clipped to its bounding cube — whose
`contains_point` costs one dot product **per face**. Both paths go through
`Tree3::cull`; the only difference is the leaf narrowphase (raster lookup
vs analytic). The raster is precomputed once per query shape (the realistic
model — like the 2D attack templates) and amortised over repeated culls.

60k points, item_limit 64, 150 queries:

| faces | analytic ns/cull | raster ns/cull | speedup |
| ---: | ---: | ---: | ---: |
| 8 | 7,675 | 10,773 | 0.71× |
| 24 | 13,755 | 14,205 | 0.97× |
| 48 | 15,660 | 12,649 | **1.24×** |
| 96 | 26,096 | 18,518 | **1.41×** |
| 192 | 42,697 | 30,208 | **1.41×** |

**The crossover sits around 24–48 faces.** Below it, analytic wins (cheap
`contains_point`, the lookup's floor + index + memory load is overhead);
above it, the raster wins because one memory lookup beats N plane
evaluations. A sphere behaves like the bottom of this curve (≈3-face cost),
which is exactly why the raster lost for it — and a many-faced or otherwise
expensive shape is where the 1×1×1 raster earns its keep, the 3D mirror of
the 2D 1×1 raster's role for complex polygons. (Reusable pieces:
`VoxelRaster::for_shape::<S: Shape3>` builds the grid from any shape's
`classify_aabb`; `Polyhedron3` is a convex N-face shape with an exact
`classify_aabb` and an expensive `contains_point`.)

## Octree (8-way) vs binary-3D tree

`Octree3` (`src/octree3.rs`) is the 8-way 2×2×2 split — the 3D analogue of
the 2D `QuadTree`, where `Tree3` is the analogue of the binary `Tree`. Both
go through the same `Shape3` machinery, so `tree3d_bench` (now culling both)
isolates the structure. Sphere queries, r 5–20, 50k points:

| item_limit | binary-3D ns/cull | octree ns/cull | octree vs binary | nodes (bin / oct) |
| ---: | ---: | ---: | ---: | --- |
| 8 | 5,622 | 4,842 | **−14%** | 18,043 / 32,945 |
| 48 | 3,543 | 2,864 | **−19%** | 3,095 / 4,681 |

**The octree is faster on cull (14–19%)**, the same direction as the 2D
quad-vs-binary result and for the same reason: one 8-way level does the work
of three binary levels, so the descent walks fewer levels. It costs more
nodes (an 8-way split allocates more children), so it trades memory for
descent depth. (The 2D quad-vs-binary gap was only ~2% at a tuned
item_limit; in 3D the depth advantage compounds — three binary levels per
octree level vs two quad levels per binary level in 2D — so the gap is
wider.) Both remain exact vs brute force.

### Dynamic octree vs binary (`update`, ascend-to-LCA)

`Octree3` gained `update` — the same ascend-to-LCA relocation as
`Tree3::update`, ported to the 8-way split (a churn test,
`octree_cull_matches_brute_after_churn`, gates it against ground truth the
way `Tree3`'s does). `critters3d_headless` runs the **same deterministic
simulation** against all the indexes, times each separately, and cross-checks
every vision cull for agreement; this table is the **binary-vs-octree pair**
(the full four-way picture, where Morton beats both on *maintain*, is in
§ Synthesis below). 20k critters moving in a 512³ cube, seed 42:

| item_limit | update bin / oct (µs/frame) | cull bin / oct (µs/cull) | agreement |
| ---: | ---: | ---: | --- |
| 8 | 3,097 / **2,650** (−14%) | 2.6 / 2.5 | 0 mismatches / 10.8k |
| 16 | 2,628 / **2,474** (−6%) | 2.1 / 2.3 | 0 mismatches / 9k |
| 64 | 2,060 / **1,833** (−11%) | 1.7 / 1.5 | 0 mismatches / 9k |

**The dynamic octree's `update` is consistently faster than the binary
tree's (~5–15%)** — same cause as the cull result: the octree is shallower,
so a relocation ascends/descends fewer levels even though each level touches
8 children instead of 2. Fewer levels wins. The cull numbers match the
static bench (octree ≈ binary, within noise at this density), and the two
structures returned **identical** id sets on every sampled cull across all
item_limits.

These numbers are not a cache-contention artifact: isolated single-structure
runs (in the earlier two-structure version of the tool, run in separate
processes) reproduced them within noise — at item_limit 8, isolated update was
3,066 µs (binary) vs 2,645 µs (octree), matching the 3,097 / 2,650 measured in
one combined run. The two index footprints are similar (binary
~7.3k arena nodes / 3.6k leaves vs octree ~7.0k / 5.7k at il 8), so neither
thrashes the other's cache enough to skew the comparison. So in 3D the 8-way split is a small, free win on both the cull
and the dynamic-relocation paths — the binary tree stays the reference (it
mirrors the primary 2D structure and needs less memory), but the octree is
the faster option when memory is cheap. The live demo's `M` toggle now keeps
a **persistent** octree (built on entry, `update` per frame) so this is
visible interactively, not only headless.

## Morton / Z-order linear grid (the pointer-free fourth structure)

`MortonGrid3` (`src/morton3.rs`) drops the pointers entirely. Quantise each
point to an integer cell `(ix,iy,iz)`, interleave the cell's bits into a single
64-bit **Morton code** (the Z-order curve), and bucket points by code in a hash
map. The spatial hierarchy is implicit in the bit layout — no nodes, no splits,
no merges, no rebalancing. A cull visits only the cells overlapping the query
bbox, with the same green/white/yellow short-circuit per cell. One **fixed
resolution** (`2^levels` cells/axis), so the cell size is the only knob; the
bench sets it ≈ the mean query radius (`levels_for_cell_size`).

50k points in 512³, 300 queries, all methods EXACT vs brute force:

| distribution / query | binary | octree | **morton grid** | 1-proj +z |
| --- | ---: | ---: | ---: | ---: |
| uniform, r 5–20 (cell 16, il 8) | 11.4× | 12.6× | **14.5×** | 9.1× |
| uniform, r 10–80 (cell 64, il 8) | 1.4× | 1.3× | **2.1×** | 1.3× |
| stacked, r 5–20 (cell 16, il 48) | 13.3× | **17.4×** | 15.5× | 16.0× |

(× = speedup over brute force; higher is better.)

**On uniform data the pointer-free grid is the fastest index** — 14.5× at
realistic small-sphere selectivity, beating both trees — *and* has the cheapest
build (4 ms vs the binary tree's 11 ms: bucket-and-go, no rebalancing). The win
is structural: when the cell is sized to the query radius, a query touches a
handful of cells via O(1) hash lookups, each holding ~2 points — no descent
through ~15 tree levels of pointer-chasing. The trees' adaptivity is wasted
effort on uniform data where one resolution already fits.

**The single fixed resolution is the catch.** On the *stacked* distribution
(dense columns — non-uniform density) the adaptive octree retakes the lead
(17.4× vs the grid's 15.5×): the grid can't subdivide the hot columns the way
the trees do. And a query much larger or smaller than the chosen cell degrades
it (too-fine → many empty cells visited; too-coarse → many points per cell to
test). So: **grid for uniform-ish data with a known query scale (fastest, cheap
build, trivial code); a tree when density varies or query sizes span a wide
range (adaptive depth earns its keep).** A multi-level linear octree (codes at
mixed depths) would recover the adaptivity, at the cost of the grid's
simplicity — noted as a possible follow-up. `MortonGrid3` is build-and-cull
only (no `update`); in the live `critters3d` demo it's selectable via the `M`
toggle (`CRITTERS3D_STRUCTURE=morton`) and **rebuilt from scratch each frame**
— which is fine precisely because its build is so cheap (the dynamic workload
where the trees use `update`, the grid just re-buckets). `visit_cells` exposes
the occupied cells for the `B` box overlay; `demorton3` decodes a code back to
its `(x,y,z)` cell.

## Adaptive linear octree (`LinearOctree3`) — grid build, octree queries (measured 2026-07-24)

`MortonGrid3` has *one* cell size everywhere — great on uniform data, poor when the
density varies (a fine-enough cell for the clusters is far too fine for the void, and
a coarse one dumps a whole cluster in one bucket). `Octree3` is fully adaptive but
each node is a heap cell reached by chasing pointers. `LinearOctree3` is the middle:
a **sparse, adaptive** octree stored the *linear* way — leaf buckets in a
`HashMap<u64, Vec<T>>` keyed by a self-describing **location code** (Morton path +
level in one `u64`: the root is `1`, octant `o` of key `K` is `(K<<3)|o`, so the
leading sentinel bit encodes the level). A companion `HashSet` marks the subdivided
keys so an empty subtree prunes in O(1). A leaf splits into 8 only where points pile
up, down to `max_depth`. `from_items` (bulk) / `insert` / `cull` / `knn`, all
brute-force-gated.

Measured (`examples/linear_octree3_bench`, 200k points, 67 % in 8 dense blobs, 2 000
sphere culls r=40 and k-NN k=8, release):

| structure | build (ms) | cull 2000q (ms) | knn 2000q (ms) | leaves/cells |
|-----------|-----------:|----------------:|---------------:|-------------:|
| **LinearOctree3** | **18.2** | 9.98 | 3.52 | 12 397 (depth 8) |
| `Octree3` (pointer) | 38.4 | **7.56** | **2.85** | — |
| `MortonGrid3` (uniform) | 16.3 | 11.6 | 17.8 | 28 589 (flat) |

Honest read: the tuned **pointer `Octree3` still wins the queries** (cull 1.3×, knn
1.2× over the linear octree) — the arena already captures the adaptivity with better
locality, the same result the wide-BVH probe reached. But `LinearOctree3` has two
real wins: it **builds ~2.1× faster than `Octree3`** (pointer-free, one bucket pass
per level — near the uniform grid's build cost) *and* on clustered data its **k-NN is
~5× faster than `MortonGrid3`** (the uniform grid's ring-shell expansion wades through
thousands of empty cells; the adaptive tree descends straight to the cluster). So it's
the pick when the data is **skewed** *and* you **rebuild often** — grid-cheap builds
without the grid's clustered-kNN cliff. On uniform data or a keep-maintained index,
`MortonGrid3` / `Tree3`+`ItemRef` remain the calls (see the decision map below).
It's selectable live in the `critters3d` demo — `M` cycles to it (after Morton, before KdTree3),
or `CRITTERS3D_STRUCTURE=linear`; the HUD shows its leaf count and adaptive depth,
`B` draws its (variably-sized) leaf boxes.

**In the measured decision-map sweep** (`critters3d_headless --sweep`, added as a 6th
structure, culls **exact** vs the other five) it wins **0 of 16 configs** — and that's
the honest, *expected* result: the critters workload is bouncing balls that fill the
box roughly **uniformly**, so there's no skew for the adaptivity to exploit. The flat
`MortonGrid3` is already ideal there (cheap uniform rebuild, comparable cull) and
`Tree3`+`ItemRef` wins maintain outright by never rebuilding (its O(1) handle). But the
number that *is* interesting: its per-frame **rebuild beats the flat grid's** at coarser
leaf capacities — at 50 000 pop / 1024³ / cap 32 it rebuilds in **2 499 µs vs Morton's
3 154 µs** (~1.26×) and lands **2nd overall** on total frame cost (behind only the
no-rebuild `binary-ref`), because its adaptive buckets skip the fine grid cells the
uniform grid spends on empty space. So even where it doesn't *win*, the pointer-free
adaptive build is a real, measured edge over the grid; its outright win needs the
skewed-data case the bench above isolates.

The **`KdTree3`** (median split) was added as the 7th structure — the surprise of the
sweep: it **ties `Tree3`+`ItemRef` for the most cull wins, 6 of 16 each** (Morton 4),
*even on this uniform data*. The median split keeps the tree perfectly balanced and its
tight per-node boxes prune harder than the midpoint trees', so it's a top-tier cull
structure whenever a per-frame rebuild is affordable — it does not win *maintain* (still
`binary-ref` at 15/16; a rebuild can't beat the O(1) handle). On the *clustered* data of
`examples/kdtree3_bench` the cull gap over `Tree3` widens to **≈2.0–2.3×**. It is quoted as
a range on purpose: even measured the careful way, it still moves ~20% between separate
processes, and `bench-runner` flags it for exactly that. It was the noisiest figure in the repo — quoted from separately-taken
millisecond timings it swung 1.57–3.28 between runs — until the bench started measuring
the pair **interleaved (A/B/B/A) in CPU cycles and aggregating the median of per-round
ratios**. Same code, same machine; the number was an artefact of measuring one structure
fully before the other. Culls **exact** vs all
six others, every config. It's selectable live in `critters3d` too (`M` cycles to it
after the linear octree, or `CRITTERS3D_STRUCTURE=kd`): the `B` overlay shows its
**tight** leaf boxes — they don't tile the world like the midpoint trees' cells, the
gaps are exactly the empty space a median split refuses to index.

Its one real cost, the O(n) median selection per level, is also **the part that recovers
best from threads**: `from_items_par` (feature `parallel`) is **3.4× on 16 threads**
(200k points, 20.6 → 6.0 ms, median of 3 passes) against `Tree3::bulk_load_par`'s
1.6–1.9×, because a median
split hands each fork exactly n/2 points while a midpoint split can hand one side almost
everything. Threaded, the k-d build is *faster than every serial build in the table* —
see [PARALLEL.md](PARALLEL.md). The parallel build is node-for-node identical to the
serial one (tested, not just equivalent).

## k-nearest-neighbour (`knn`) — a different query than range cull

Range cull answers "which points are inside this volume?" (you supply the
volume). **k-NN** answers "what are the `k` closest points to `q`?" — no radius,
just the k nearest whatever the distance ("nearest enemy", "5 neighbours for a
cohesion force"). `Tree3::knn` and `Octree3::knn` implement the classic
**best-first descent with bounding-box pruning**:

- a bounded **max-heap** holds the current k best, so its top is the k-th
  nearest found so far — the pruning bound;
- at each node, descend the **nearer child first** (by box nearest-point
  distance) to tighten that bound early;
- **skip any subtree** whose box's nearest point is already farther than the
  current k-th nearest. The octree orders its 8 octants nearest-box-first.

Both are gated against a brute-force sort (`knn_matches_brute_force`): the k
smallest distances must match (unique even under boundary ties). 50k uniform
points in 512³, 300 queries:

| k | binary knn | octree knn | vs brute (full sort) |
| ---: | ---: | ---: | ---: |
| 1 | 2.3 µs | 2.4 µs | ~370× |
| 10 | 6.2 µs | 5.6 µs | ~140× |
| 50 | 13.4 µs | 14.3 µs | ~60× |

**The tree visits a tiny fraction of nodes** — a k=1 query is ~2 µs because the
pruning kills almost everything after the first leaf tightens the bound. Cost
grows sub-linearly in k (the heap stays small and the bound stays tight).
Octree ≈ binary again, the same near-tie as the cull/update paths. Caveat on the
headline ratio: the brute baseline is a *full sort* (~850 µs); an O(n)
bounded-heap brute would be faster but still scans all 50k points (tens of µs),
so the tree is still ~10–20× ahead even against an optimised brute — the real
story is the **absolute** 2–13 µs, not the inflated ×. (`tree3d_bench --knn K`.)
`MortonGrid3` has no `knn` yet — a grid would spiral outward ring-by-ring from
`q`'s cell; noted as a follow-up.

## Synthesis — which structure wins, and when

Playing with the live `critters3d` demo (the `M` toggle), the ranking visibly
*changes with the scenario*: sometimes the binary `Tree3` is fastest, and it's
near the top almost everywhere else. That is the headline of the whole 3D
study — **there is no universal winner; the binary tree is the robust default
and each alternative has a sweet spot.** Two layers explain the observation.

### The build cost: persistent `update` vs full rebuild — and the surprise

The four structures differ in how they track the moving points:

| structure | per frame | kind |
| --- | --- | --- |
| Tree3 (binary) | incremental `update` (ascend-to-LCA) | persistent |
| Octree3 | incremental `update` | persistent |
| MortonGrid3 | full rebuild (re-bucket all) | rebuilt |
| projection | rebuild the 2D tree | rebuilt |

The intuition — and the first draft of this section — was that the *persistent*
structures must beat the *rebuilt* ones on build: an incremental `update` sounds
cheaper than re-inserting everything. **The decision map below shows the
opposite, and that is the most important finding here.**

### Decision map (`critters3d_headless --sweep`)

The same deterministic simulation drives all four structures; each frame every
point is relocated (update or rebuild) and a sample run a sphere cull. The sweep
varies world × population × `item_limit` × churn (vision radius fixed), two
winners per config — **maintain** (per-frame relocate cost) and **cull**
(per-cull) — over 16 configs:

| metric | binary | octree | morton | projection |
| --- | ---: | ---: | ---: | ---: |
| **maintain** wins | 0 | 0 | **16** | 0 |
| **cull** wins | 2 | 3 | **11** | 0 |

(All four returned identical id sets in every config — exact.)

**Morton wins `maintain` in *every* config** — its flat re-bucket (N hash
inserts) is cheaper than the trees' `update`, at every population from 1k to 50k,
slow or fast churn. Why do the trees lose the relocate race? Each `update(old,
predicate, …)` must **find the item in its old leaf by predicate — an
O(item_limit) scan** — then walk to locate it, then maybe split/merge. For a
workload that moves *every* point *every* frame, that per-point lookup costs more
than rebuilding flat. The trees' incremental `update` only pays off when movement
is **sparse** (it touches few points; a rebuild still does all N) or with a
**stable item handle** (O(1) access instead of the predicate scan). Without it,
Morton wins the build side outright.

### The fix: Stable `ItemRef` (and it flips the result)

`Tree3` gained `insert_ref` → [`ItemRef`], a **stable handle** that stays valid
as the item moves between leaves (a parallel handle vector per leaf + a
handle→location map, maintained through splits/merges; round-trips through
`serialize`; churn-tested as `update_ref_churn_matches_brute`). `update_ref` /
`remove_ref` then reach the item in **O(1)** — no locate walk, no predicate scan.
Re-running the sweep with a fifth competitor, `binary-ref` (the binary tree
maintained via `update_ref`):

| metric | binary | octree | morton | projection | **binary-ref** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **maintain** wins | 0 | 0 | 1 | 0 | **15** |
| **cull** wins | 0 | 0 | 6 | 0 | 10 |

**The handle path flips the maintain winner from Morton back to the binary tree
in 15 of 16 configs** — and not marginally: per-frame relocate drops **~5–11×**
(e.g. at world 512, pop 20k, item_limit 64, churn 55: binary `update` 2,166 µs
→ `update_ref` 217 µs, a **10× speedup**, because the predicate scan was an
O(item_limit)=O(64) cost per point). So the earlier conclusion stands corrected
*again*: the trees' incremental `update` **is** the cheaper relocate — the
predicate scan was hiding it. The single Morton maintain win left is a 1.07×
near-tie. (`binary-ref`'s *cull* is the same tree as `binary`; its higher cull
win-count is a cache-ordering artefact of being timed right after it, not a real
structural edge.)

The lesson: **for a relocate-heavy dynamic index, hold a stable handle.** The
predicate `update`/`remove` stay for callers without one (sparse edits, ad-hoc
lookups); the handle path is the one to reach for in a per-frame movement loop.

The trees keep the **cull** crown in the dense/deep corner (binary + octree take
5 of 16): a small, dense world with tight queries is where a tree's descent
prunes hardest; elsewhere Morton's cell ≈ query also wins the cull. Projection
never wins here — its rebuild is a full 2D tree plus a disc-cull + z-reject +
exact narrowphase; its sweet spot (2.5D thin-z worlds, expensive narrowphase) is
a different sweep.

So the by-eye demo impression that "the binary tree wins" is the **low-N
near-tie**: at pop ~1–2k the maintain gap is only ~1.1× (a few µs), and the HUD's
rolling average jitters between them. The data shows Morton edging it even there.
What *is* true is that the binary tree has **no pathological weakness** and is
the cull champion in the deep/dense corner.

### The knobs that move the ranking

- **Churn (movement speed) → relocate cost.** High churn = more cross-leaf moves
  = more split/merge for the trees; Morton re-buckets flat regardless. It widens
  Morton's maintain lead but does not flip it here.
- **World size / density → tree depth.** Small/dense = deep tree → the octree's
  fewer levels win the *cull* (it allocates 8 children even when 1–2 are used, so
  it thrashes cache at low density).
- **Vision radius → cull selectivity.** A large query is low selectivity
  (structure barely matters); a small query is where the trees separate.
- **`item_limit` → leaf granularity *and* the per-`update` scan length** — bigger
  leaves mean fewer, shallower nodes but a longer predicate scan per relocate.

### The takeaway

- **MortonGrid3** — for *this* workload (uniform-ish density, *every* point
  relocated each frame, sphere queries) it is the all-round winner: cheapest
  maintain (no per-item lookup) and usually cheapest cull. Caveats: one fixed
  resolution (loses to the octree on *stacked*/non-uniform density), and it
  re-buckets the whole set each frame.
- **Binary `Tree3` — the robust default, *and* the maintain champion with a
  handle.** Adapts to any density, no N³ memory, no fixed resolution. Its
  predicate-`update` maintain loss was an artefact of the O(item_limit) leaf
  scan: with the new **stable `ItemRef`** (`update_ref`, O(1)) it wins maintain
  15/16 at ~5–11×, and stays competitive on cull. The all-round pick for a
  relocate-heavy dynamic index — *if* you hold the handle.
- **Octree3** — when the tree is *deep* (small/dense world, tight queries): the
  shallower descent wins the cull, at more nodes/memory.
- **Projection** — a *2.5D* world (large in xy, thin in z) or an *expensive*
  narrowphase, reusing the whole 2D stack with no N³ memory; not competitive on
  this uniform-cube sphere sweep.

## `MortonGrid3::knn` and the shape of the world

`levels` is one number for all three axes, so a world that is not a cube does not have cubic
cells: the siege/horde worlds (1000 x 300 x 1000 at levels 5) get cells of 31.25 x 9.375 x
31.25. Any k-NN expansion that grows one radius for all three axes is then isotropic in *cell*
space and anisotropic in *world* space — to reach 30 units along the short axis it has to drag
the wide axes out to +-125 and scan everything in between. `knn` therefore grows **each axis
separately**, always the one whose scanned world half-extent is currently smallest, which
keeps the scanned region near-cubic in world units. The stopping rule is unchanged and still
exact: the scanned region is still a box, so the nearest unscanned point is still at least
`safe` away.

Measured (`examples/morton_knn_axis_bench`, 50k clustered points, k=8, levels=5):

| world aspect (h/w) | points tested/query | ms/query | before → after |
| ---: | ---: | ---: | ---: |
| 1.00 (cubic cells) | 14 946 → 14 823 | 0.2012 → 0.0883 | **2.3x** |
| 0.50 | 33 846 → 14 595 | 0.5713 → 0.0889 | **6.4x** |
| 0.30 | 41 529 → 12 933 | 1.0779 → 0.0867 | **12.4x** |
| 0.15 | 45 510 → 11 452 | 1.3533 → 0.0784 | **17.3x** |
| 0.05 | 48 219 → 10 591 | 1.6112 → 0.0761 | **21.2x** |

Two things worth reading off that table. The flatter the world the bigger the win, which is
the effect being fixed. And the **cubic** row still improves 2.3x while its point count barely
moves — that saving is cells never visited, which the point counter cannot see (the old
enumeration walked the full Chebyshev shell rejecting out-of-grid cells one at a time; the new
one clamps its loops to the grid). It does not change the verdict for clustered data: a k-d
tree or `LinearOctree3` still answers a clustered k-NN with far less work, because a blob
inside one grid cell has to be scanned whole however you reach it.

**The 2D `MortonGrid` had the identical problem on a non-square `Rect`** and got the identical
fix, measured by the same bench — one fewer axis to over-scan, so the effect is smaller but the
shape is the same:

| world aspect (h/w) | points tested/query | ms/query | before → after |
| ---: | ---: | ---: | ---: |
| 1.00 (square cells) | 11 395 → 11 266 | 0.0261 → 0.0135 | **1.9x** |
| 0.50 | 19 088 → 9 649 | 0.0373 → 0.0113 | **3.3x** |
| 0.30 | 28 431 → 9 901 | 0.0592 → 0.0116 | **5.1x** |
| 0.15 | 34 525 → 9 199 | 0.0660 → 0.0108 | **6.1x** |
| 0.05 | 40 438 → 8 489 | 0.0912 → 0.0106 | **8.6x** |

The exact gate saw only 5% of that, because its world is square — which is the point of having
both: the ratchet proves nothing changed for the common case, and the sweep shows what changed
for the case the ratchet does not cover.

## Still open
- ~~**Stable `ItemRef`**~~ — **done** (`Tree3::insert_ref`/`update_ref`/
  `remove_ref`). The sweep predicted it and confirmed it: O(1) handle updates
  flip the maintain winner from Morton back to the binary tree in 15/16 configs,
  ~5–11× faster (see § "The fix: Stable ItemRef"). Follow-ups: port it to
  `Octree3`, and wire the live `critters3d` demo's persistent tree to it.
- **Non-analytic 3D shapes** in the *projection* comparison: the
  static bench used a sphere (analytic); the polyhedron crossover above
  lives in the voxel-raster bench. Running the projection methods against a
  many-faced polyhedron would show where the 3-projection's tight broadphase
  finally pays (expensive narrowphase), worth a follow-up.
- **Exact 3D `item_limit` optimum** — both the binary and octree update/cull
  curves keep improving with larger leaves through the range measured; the
  precise optimum is unmeasured (the monotone trend is the finding).
