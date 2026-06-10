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
