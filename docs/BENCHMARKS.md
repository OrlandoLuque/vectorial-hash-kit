# Benchmarks: template-driven culling

Performance analysis of the vectorial-hash culling pipeline. All numbers are
reproducible with the commands below; rerun them after any change to the cull
path or the template bank, and refresh the tables.

## Environment

| | |
| --- | --- |
| CPU | AMD Ryzen 7 7800X3D (8 cores / 16 threads) |
| OS | Windows 10 Pro (build 19045) |
| Toolchain | rustc 1.96.0, `--release` (opt-level 3, LTO) |
| Date of the numbers below | 2026-06-10 |

## Methodology

- Deterministic point cloud (xorshift64\*, fixed seed) — every config sees the
  same data; flags `--points`, `--culls`, `--item-limit`, `--seed` vary the
  scenario.
- **Correctness gate before timing**: every configuration must return exactly
  the same hit count, or the bench aborts. Speed without agreement is noise.
- Wall-clock `std::time::Instant` over N repeated culls, single-threaded
  queries, results consumed through `std::hint::black_box`. All structures
  answer the same contract (collect a `Vec` of item references), so no config
  gets a cheaper job.
- Template/bank generation happens before timing and is reported separately —
  it is the offline precomputation cost.

## How to reproduce

```bash
# 4-way: binary-split tree vs quadtree, single fixed template on/off
cargo run -p vectorial-hash-cli --release -- bench

# per-cell-size selection study (the paper's scheme), incl. the old
# snap-to-offset method and the industry uniform-grid baseline
cargo run -p vectorial-hash-cli --release -- bench-sizes

# both accept: --points N --culls N --item-limit N --seed N
```

## Results 1 — `vh bench`: single fixed template, tree vs quadtree

200k uniform points in a 4096² world, item_limit 16, query = drop polygon at
scale 1400 rotated 30°, one 64px-cell template classified per node via
`classify_region`. 50 culls/config, 5246 hits (all configs agree).

| config | avg/cull (ms) | speedup |
| --- | ---: | ---: |
| vectorial (no templates) | 2.275 | 1.0x |
| vectorial + template | 0.587 | 3.9x |
| quadtree (no templates) | 2.234 | 1.0x |
| quadtree + template | 0.533 | 4.2x |

Conclusions:

- A single precomputed template already cuts ~4x off either tree, mostly by
  proving subtrees fully-inside (taken wholesale) or fully-outside (skipped).
- With uniformly distributed points, the binary-split tree and the quadtree
  are equivalent (±10%). The binary tree's edge is structural (cheap
  incremental `remove`/`update` with the merge-up rule), not raw cull speed.
- **Clustered data doesn't change the picture** (`--clusters N` aims the
  query at the first cluster): with 12 clusters and ~27k hits, or 4 clusters
  and ~80k hits, both trees stay within ±1% of each other, with or without
  templates. Cull cost is dominated by classification work that is identical
  in both. Where they do differ: the quadtree *builds* ~1.8x faster (one
  4-way redistribution per level vs two 2-way ones) and only needs square
  template sets, while the binary split owes its place to the dynamic
  mechanics (merge-up with hysteresis, data-aware split axis, non-square
  worlds) rather than to query speed.

## Results 2 — `vh bench-sizes`: per-cell-size selection (the paper's scheme)

200k uniform points in a 4096² world, item_limit 16, query = drop polygon at
scale 350 rotated 30° applied at a **real integer origin — the figure is
never moved to fit any grid**. The bank resolves, per tree-cell size, the
template whose generation offset matches the origin's displacement within the
global virtual grid of that size (one resolution per size per cull, cached,
zero cell-data clones via `PlacedTemplate`). Leaf items use the 1×1 raster:
only boundary (`Maybe`) pixels run exact geometry. 50 culls/config, all
configs agree on the hit count.

Bank generation (offline, 16 threads): 1×1 raster 0.05s; sizes ≤16 in 0.19s
(577 combos → 410 unique); ≤32 in 0.19s (2,625 → 852); ≤64 in 0.14s
(10,817 → 1,081). Content dedup shares identical grids behind `Arc`s — at
≤64, **90% of index leaves point at a shared template**.

Memory (measured via `TemplateBank::memory_usage`, demo-sized bank — 2
figures, 24+1 angles, sizes 8–32 px + 1×1 rasters, 65,625 combos → 7,797
unique grids): **5.84 MB total**, of which the deduplicated template data is
only **1.06 MB** (~136 B per unique grid), the hierarchical lookup index is
**3.20 MB** (one ~41 B entry per key combination plus hash-map overhead —
the explicit price of O(1) lookup per key), and **1.58 MB** are flat cell
copies retained as dedup-map keys, a generation-time aid that could be
dropped after building.

| config | avg/cull (ms) | speedup |
| --- | ---: | ---: |
| no templates (bbox + exact geometry) | 0.138 | 1.0x |
| single 16px grid, `classify_region` (≈ old snap method) | 0.057 | 2.4x |
| bank ≤16 + raster | 0.014 | 9.8x |
| bank ≤32 + raster | 0.012 | 11.9x |
| bank ≤64 + raster | 0.011 | 12.4x |
| bank ≤64, **no** raster | 0.056 | 2.5x |
| quadtree, no templates | 0.145 | 1.0x |
| quadtree, bank ≤64 + raster | 0.008 | 17.1x |
| uniform grid 32px (industry baseline) | 0.136 | 1.0x |
| uniform grid 32px + raster | 0.007 | 19.2x |

### Conclusions

1. **The precise method beats the "easy" method it replaced.** Selecting the
   matching template (figure stays put) is ~4–5x faster than the old
   move-the-figure + single-grid `classify_region` approach — *and* it is
   exact. Per-node classification drops from a region scan to one array read.
2. **Gains saturate once template sizes cover the band where the tree
   actually lives** (leaf cells were 16–32px here). Each extra size family
   keeps helping slightly (≤64 > ≤32 > ≤16) now that resolution is
   zero-clone, but the increments shrink; cells much larger than the figure
   can never classify `In`, so their sets mostly buy corner `Out`s.
   The cost of over-generating is precompute time and RAM, not query time —
   the per-cull size cache caps lookups at one per distinct size.
3. **The 1×1 raster is half the win.** Without it the bank stalls at ~2.5x;
   with it, 12–19x. Replacing exact point-in-polygon (arcs!) with a raster
   read, reserving geometry for boundary pixels, dominates leaf cost.
4. **The technique composes with any spatial structure.** Quadtree + bank
   (17x) and even the flat uniform grid + raster (19x) beat the binary tree +
   bank (12x) *on static uniform data*, because their traversals are simpler
   and the raster equalizes the per-item cost everywhere. The trees' green
   short-circuit matters more as queries grow relative to cell sizes and as
   item density rises; the binary tree additionally keeps its dynamic
   `remove`/`update` merge-up behaviour, which none of the static baselines
   offer.
5. Caveats: single machine, uniform random points, one query shape per run,
   wall-clock timing. Scenarios still to measure: clustered/skewed point
   distributions, many simultaneous queries, larger worlds, mixed query
   sizes, and the planned "granularity as fallback" aggregation.

## Results 3 — `vh bench-walk`: tree descent vs neighbour-walk (flood fill)

Same world and bank setup as Results 2 (bank ≤64 + raster everywhere, so the
comparison isolates the **traversal strategy**). The walk starts at the leaf
containing a seed point inside the figure and expands through leaf
neighbours, stopping at `Out` leaves. Three neighbour sources: Samet-style
ascent/descent over the existing parent pointers (zero storage), point
probing + `locate` from the root (zero storage), and stored per-leaf
neighbour lists — *ropes* — behind the `neighbors` cargo feature (compiled
out entirely when off). All strategies pass the equality gate against
`Tree::cull`.

| config | scale 350 (ms/cull) | scale 1400 (ms/cull) |
| --- | ---: | ---: |
| descent (`Tree::cull`) | **0.012** | **0.076** |
| walk + Samet ascent | 0.024 (0.49x) | 0.213 (0.35x) |
| walk + locate probe | 0.026 (0.45x) | 0.318 (0.24x) |
| walk + ropes (stored) | 0.017 (0.70x) | 0.108 (0.70x) |

Rope maintenance cost: building the 200k-point tree takes 49.9 ms without
the feature vs 78.0 ms with it (~+56% on inserts; splits/merges rewire the
neighbour lists). Descent cull times are unaffected by the feature.

### Conclusions

1. **Descent wins, and the reason is structural**: with per-cell-size
   templates, an internal node classified green/white takes or skips its
   *entire subtree* without visiting leaves. The flood-fill walk has no
   subtree short-circuit — it must touch every leaf in the region, pay a
   visited-set membership check per leaf, and run 4 neighbour queries per
   leaf. The gap *widens* with query size (0.70x → 0.35x for the zero-storage
   variants) precisely because larger queries contain more wholesale-takeable
   subtrees.
2. **Among walk variants, ropes > Samet > probe**, as expected: O(1) stored
   lists beat O(1)-amortized ascent, which beats O(depth) root descents. But
   even ropes lose ~30% to descent while making every insert/split/merge
   ~56% more expensive — a bad trade for this workload.
3. Where a neighbour walk could still win: queries *without* useful templates
   (no green short-circuit to exploit), incremental queries that slide
   between frames (reuse the previous frontier), or region operations that
   are inherently neighbour-based (e.g. connected-component analysis,
   contour extraction). The mechanism stays available (`Tree::cull_walk`,
   `neighbors_samet`/`neighbors_probe`/`neighbors_ropes`) for those cases.
4. References for the strategies: H. Samet, *Neighbor Finding Techniques for
   Images Represented by Quadtrees* (1982); "ropes" as used in stackless
   spatial traversal (Popov et al., *Stackless KD-Tree Traversal for High
   Performance GPU Ray Tracing*, 2007); flood fill / region growing from
   image processing.

```bash
cargo run -p vectorial-hash-cli --release -- bench-walk            # ropes included
cargo run -p vectorial-hash-cli --release -- bench-walk --scale 1400
cargo run -p vectorial-hash-cli --release --no-default-features -- bench-walk  # no rope bookkeeping
```

## Results 4 — `vh bench-fallback`: granularity-as-fallback aggregation

Same scenario as Results 2. The .docx design notes record an exact property:
a template generated for a small cell size can stand in for *any* larger
cell size whose dimensions are an integer multiple, by aggregating blocks
with the rule `all-In → In, all-Out → Out, otherwise → Maybe`. This is
**not an approximation** — a cell is fully inside the figure iff every
sub-cell is, etc. So an aggregated template carries exactly the
classification a directly-generated one would have, only paid at query
time instead of at precomputation.

`PlacedTemplate::aggregated(fx, fy)` realizes the property (aligning the
output to the world grid of the new cell size), and
`TemplateBank::placed_for_or_aggregated(...)` automates the fallback: it
serves the directly-precomputed set when available, otherwise picks the
largest stored sub-size that divides the request and aggregates.

| config | bank size | gen time | avg/cull (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| no templates | — | — | 0.155 | 1.0x |
| bank ≤16 + aggregated fallback | 0.59 MB | 0.20 s | 0.092 | 1.7x |
| bank full (every size precomputed) | 1.70 MB | 0.66 s | 0.011 | 14.2x |

Conclusions:

- **Correctness is preserved exactly.** The campaign (Results elsewhere)
  runs 2,000 random scenarios with the aggregating shape in the cull path
  and they all match brute force. The aggregated and the directly-generated
  templates return the same In/Out/Maybe classification for every aligned
  cell — verified by a unit test that picks the same shape and angle and
  asserts cell-by-cell equality.
- **The fallback is a memory/precompute knob, not a precision knob.** It
  costs ~3× the no-template baseline because aggregation builds a fresh
  template per cull (the per-execution `SizeCache` reuses it within a
  single cull, but not across culls); precomputing the full family pays
  itself back at runtime. Use the fallback when: (a) memory or generation
  time matters more than query speed, (b) only a few cell sizes carry the
  load (the fallback covers the rest), or (c) you are prototyping and
  haven't decided which sizes to precompute.

## Industry context

What games/physics engines typically use for this class of query (and what we
benchmarked against):

- **Uniform grids / spatial hashing** — the classic broadphase; simple and
  extremely fast for roughly uniform object sizes. Covered in depth in
  Christer Ericson, *Real-Time Collision Detection*, ch. 7 "Spatial
  Partitioning" (cell-size tradeoffs, hashed storage)
  ([book](https://www.routledge.com/Real-Time-Collision-Detection/Ericson/p/book/9781558607323),
  [chapter contents](https://www.oreilly.com/library/view/real-time-collision-detection/9781558607323/xhtml/c07.xhtml)).
  Our `UniformGrid` baseline implements exactly this.
- **Quadtrees/octrees**, including Thatcher Ulrich's **loose octrees**
  (Game Programming Gems 1, 2000) which relax cell bounds to avoid small
  objects landing in huge nodes
  ([Ulrich's write-up](https://www.tulrich.com/geekstuff/partitioning.html)).
  Our reference quadtree is the strict variant; loose bounds matter for
  objects with extent, less for point items.
- **Dynamic AABB trees (BVHs)** — the broadphase in Box2D (`b2DynamicTree`,
  inspired by Bullet's `btDbvt`), Bullet and others; binary bounding-volume
  hierarchies rebalanced incrementally
  ([Box2D docs](https://box2d.org/documentation/group__tree.html)).
  For point items an AABB tree degenerates to roughly what our binary-split
  tree already is; the comparison would become meaningful with area items.
- General surveys of broadphase choices and their tradeoffs:
  [Build New Games — broad phase collision detection](http://buildnewgames.com/broad-phase-collision-detection/),
  [GameDev.net spatial partitioning discussion](https://www.gamedev.net/forums/topic/598183-spatial-partitioning/).

None of these precompute shape-vs-grid classification templates; they all run
exact (or conservative AABB) per-object tests after the broadphase. The
template bank + 1×1 raster is orthogonal to the choice of structure — as the
numbers show, it accelerates the industry baselines too.
