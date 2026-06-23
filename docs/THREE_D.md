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
     cost is what hurts) → **3-projection**'s tight 1.12× broadphase finally
     pays, because it minimises the number of exact tests. Not our case
     here, but the regime where it wins.

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

## Still open
- **Non-analytic 3D shapes**: the comparison used a sphere (analytic). A
  general polyhedron needs the N³ voxel template for the 3D tree, and a
  more expensive narrowphase — the regime where 3-projection's precision
  starts to pay. Worth a follow-up with, say, an inflated 3D drop.
- **Octree (8-way) vs binary-3D**: this used the binary split (mirror of
  the 2D primary `Tree`). The octree is the analogue of `QuadTree`; the 2D
  lesson was the two are within ~2% at a tuned item_limit, so the binary-3D
  choice is unlikely to matter much, but unmeasured in 3D.
