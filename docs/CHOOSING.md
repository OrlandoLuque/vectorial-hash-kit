# Choosing a structure

`vectorial-hash` ships **twelve** spatial indexes. They all answer the same two queries —
`cull` (everything inside a shape) and `knn` (k nearest neighbours) — so picking one is
about *your data and access pattern*, not features. This is the one-glance guide; the
quantitative backing is the decision map in [`THREE_D.md`](THREE_D.md), the parallelism
crossovers in [`PARALLEL.md`](PARALLEL.md), and the demo write-ups
([`FLUID.md`](FLUID.md), [`POINTCLOUD.md`](POINTCLOUD.md), [`STEALTH.md`](STEALTH.md))
where each one is measured on a real workload rather than a synthetic one.

## Before any of this: are your objects POINTS?

All twelve index **points**, and it is worth checking first because getting it wrong produces a
wrong *answer* rather than a slow one. Index a 3 km ship by its centre and a query for "everything
within 500 m" will **miss** it when its centre is 1 km away, even though its hull is 100 m from
you. No tuning fixes that — the index is answering a different question.

If your objects have extent, the known routes, cheapest first:

1. **Enlarge the query** by the largest object's radius, then test exactly. Correct; the price is
   the enlargement. With a 3 km object every query grows by 1.5 km, so a 500 m query sweeps
   `(2000/500)³` = **64×** the volume. Fine while the biggest thing is small next to your queries.
2. **Size tiers** — one index per size class, enlarging by *that tier's* maximum rather than the
   global one. The cheap fix for (1)'s pathology, and what most physics broadphases do.
3. **A loose octree** (Ulrich): expand each node's box ~2× so each object lives in exactly one node
   chosen by its **size**. No duplication, and big objects stop sinking to the root.
4. **Duplicate into every overlapping cell**, dedupe on query. Correct, usually hopeless: a 3 km
   object against 100 m cells is 30×30×30 = 27 000 entries.
5. **A BVH or R-tree** — group by data so each box adapts to its contents (see the
   space-partitioning-versus-data-grouping note further down). No duplication, no size/level
   mismatch; the cost is maintenance under motion, which is why engines use BVHs for static
   geometry and grids or sweep-and-prune for dynamic.

**Count the large objects before reaching for (5).** Size distributions are violently skewed —
thousands of capital ships against millions of small things — and a linear scan over a thousand big
ones is microseconds. The practical answer is usually **hybrid**: the small and numerous in a
structure from this crate, the large and few in a list or a small BVH beside it, enlarging each
query only by its own tier's maximum.

## The first question is not "which tree"

**Do the points move?** That splits the whole family in two, and it matters more than any
other property:

- **They move every tick (a simulation).** You want a structure you can *maintain*:
  `Tree` / `QuadTree` / `IntegerTree` / `Tree3` / `Octree3`, held across frames with the
  `ItemRef` handle (`insert_ref` → `update_ref`, O(1) relocation). Measured on the siege
  demo, keeping the index beats a per-frame rebuild ~1.06× (1 thread) → ~1.4× (12–16),
  and it needs no threads at all.
- **They're static, or you rebuild wholesale anyway.** Then maintenance is worth nothing
  and *build + query* is the whole cost: `KdTree2`, `KdTree3`, `LinearOctree3`, `LinearQuadTree`,
  `MortonGrid` / `MortonGrid3`. These have no handle or remove surface by design.

**Second question: is the density even?** Uniform data suits a flat grid (Morton) —
nothing to adapt to. Skewed data (points on surfaces, crowds, clusters in empty space) is
where the adaptive structures earn their keep, and where the *median* split (`KdTree2` /
`KdTree3`) beats the *midpoint* splits.

## Flowchart

```
                        ┌──────────────────────────────────┐
                        │ Do the points MOVE every tick?   │
                        └──────────────┬───────────────────┘
             yes (simulation) ─────────┴───────── no (static / rebuilt wholesale)
                      │                                        │
   ┌──────────────────┴──────────────┐        ┌────────────────┴─────────────────┐
   │ 2D or 3D?                       │        │ density even, or skewed?         │
   └──────┬───────────────────┬──────┘        └──────┬────────────────────┬──────┘
        2D│                   │3D                even│                    │skewed
   ┌──────┴─────────┐   ┌─────┴──────────┐    ┌──────┴───────┐   ┌────────┴─────────┐
   │ integer coords?│   │ Tree3 + ItemRef│    │ MortonGrid3  │   │ KdTree3 (median  │
   │ yes→IntegerTree│   │ (the default)  │    │ / MortonGrid │   │ split: balanced  │
   │ no →Tree       │   │ Octree3 if the │    │ (cheapest    │   │ whatever the     │
   │    (QuadTree   │   │ density varies │    │  build+cull  │   │ clumping)        │
   │     if uniform)│   │ a lot locally  │    │  when dense  │   │ …or LinearOctree3│
   └────────────────┘   └────────────────┘    │  & uniform)  │   │ /LinearQuadTree  │
                                              └──────────────┘   │ if you rebuild   │
                                                                 │ far more often   │
                                                                 │ than you query   │
                                                                 └──────────────────┘
```

Three cross-cutting choices that sit on top of the above:

- **Points that live on a plane** (a heightfield, units on terrain)? A 3D query can be
  answered by a **2D `Tree` on xy + a z-slab reject + exact 3D narrowphase** (the
  "projection" path). Wins when the z-extent is thin relative to xy.
- **Many independent queries at once** (one cull per attacker, a batch of frustums)?
  `cull_many` / `cull_many_par` (feature `parallel`). The crossover is in
  [`PARALLEL.md`](PARALLEL.md).
- **Building once, with cores to spare?** `Tree3::bulk_load_par` and
  `KdTree3::from_items_par`. The k-d tree recovers the most from threads (**3.4× on 16**
  vs the binary tree's 1.6–1.9×) because a median split hands each fork exactly half the
  points; a midpoint split can hand one side almost everything.

## Summary table

| Structure | Dim | Points move? | Best when | Build | Query |
| --- | --- | --- | --- | --- | --- |
| **`Tree`** | 2D | yes | general-purpose 2D, the default | adaptive | 14–16× brute |
| **`QuadTree`** | 2D | yes | uniform density, simple 4-way | adaptive | ≈ `Tree` |
| **`IntegerTree`** | 2D | yes | integer coords, no float fuzz | adaptive | ≈ `Tree` |
| **`Tree3`** | 3D | yes | dynamic 3D, the 3D default | adaptive | strong; **`update_ref` O(1)** |
| **`Octree3`** | 3D | yes | 3D with locally varying density | adaptive (8-way) | strong |
| **`MortonGrid`** | 2D | rebuild | dense + uniform 2D, refilled each frame | **cheapest** | **cheapest** when uniform |
| **`MortonGrid3`** | 3D | rebuild | dense + uniform 3D, refilled each frame | **cheapest** | cheap; loses on skew |
| **`KdTree3`** | 3D | static | **skewed/clustered, query-heavy** | median select (**≈3.4× on 16 threads**) | **cull ≈2.0–2.3× `Tree3` on clusters**, k-NN 1.67× |
| **`LinearOctree3`** | 3D | static | skewed data you **rebuild often** | ~2.1× faster than `Octree3` | loses cull ~1.3× to `Octree3`; **ties a cubic `MortonGrid3`** on cull, keeps ~1.3–1.8× on k-NN |
| **`RadixTrie3`** | 3D | static | you address data **by key**, not by search | 1.8–4.4× the naive trie; **1.5–4.2× `Octree3` on clustered** | **loses every query** to `Octree3`/`Tree3`/`MortonGrid3`/`KdTree3` — pick it for `region()`, not for speed |
| **`LinearQuadTree`** | 2D | static | skewed 2D you rebuild often | fast | **won the fluid's neighbour query** |
| **`KdTree2`** | 2D | static | **skewed/clustered 2D, query-heavy** | **fastest of the 2D builds** (2.8× on 16 threads) | **cull ~1.62× the pointer quadtree** |

The three headline measurements behind the right-hand column — every one the **median of
repeated passes** on an idle machine, via `cargo run -p bench-runner --release`:

- **Point cloud** (120k static points, one k-NN per point): both trees answer k-NN **~1.5×
  faster than the flat grid**, and `KdTree3` **builds 1.7× faster than `Octree3`** while
  tying it on query; `MortonGrid3` still builds fastest of all.
  → [`POINTCLOUD.md`](POINTCLOUD.md)
- **Fluid** (every particle relocates every step): kept `Tree`+`ItemRef` maintains **3.5–
  3.9× cheaper** than either rebuild — and gives more than that back on query (+22%), so
  on *this* workload the rebuild wins the frame. → [`FLUID.md`](FLUID.md)
- **Stealth** (frustum culls per guard): an index only beats a linear scan **above ~1000
  agents** — 6.7× by 40 000, but honestly *slower* at 40. → [`STEALTH.md`](STEALTH.md)

## Rules of thumb

- **Start with `Tree` (2D) or `Tree3` (3D).** They adapt leaf size to local density and
  carry the full dynamic contract. Only move off the default for a concrete reason below.
- **Check whether you need an index at all — and note the answer is not a population.**
  A scan costs per **query**; an index costs per **move**. Measured across both axes
  (`examples/brute_edge`, 500³, a quarter moving each frame):

  | population | 1 cull/frame | n/16 | n/4 | n culls/frame |
  | ---: | --- | --- | --- | --- |
  | 64 | scan 1.96× | scan 1.41× | scan 1.10× | scan 1.06× |
  | 128 | scan 2.07× | scan 1.12× | **keep 1.10×** | **grid 1.28×** |
  | 512 | scan 3.80× | keep 1.45× | keep 1.80× | grid 1.97× |
  | 2 048 | **scan 7.00×** | keep 2.82× | grid 4.60× | grid 5.33× |

  Read along a row: the winner changes with query load alone. A scan still wins **7× at 2 048
  items** if you barely query, and loses at 128 if you query hard. The old "below ~500–1000 a
  scan wins" was the middle of that table mistaken for its conclusion.
  `AdaptiveIndex` splits the two questions accordingly: `brute_max` (64) is an unconditional
  floor set from the case least favourable to a scan, and `scan_budget` handles the rest
  because it can see the load. Don't index 40 guards; do index 2 000 if you ask them
  something every frame.
- **Relocating everything every frame?** Hold the `ItemRef` that `insert_ref` returns and
  call `update_ref` — it skips the predicate's leaf scan and is the single biggest
  **maintain** win (the decision map flipped on it). One extra field per entity.
  **But maintain is not the frame.** Measured on two workloads that both relocate
  everything:

  | | maintain | query | verdict |
  | --- | --- | --- | --- |
  | siege (20k units, modest culls) | keep 3–5× cheaper | ~unchanged | **keep wins 1.05×→1.50×** (1→16 threads) |
  | fluid (2.2k particles, one neighbour query *per particle*) | keep 3.5–3.9× cheaper | keep **+22%** | **rebuild wins the frame by 16%** |

  The difference is how far items move *relative to their leaf* and how query-heavy the
  frame is. A kept tree drifts from the ideal partition as the data sloshes; a rebuild is
  always perfectly fitted. When the query dominates (SPH), that drift costs more than the
  relocation saves — which is exactly what `advisor::HIGH_RELOCATION` exists to flag.
  See [`FLUID.md`](FLUID.md) and [`PARALLEL.md`](PARALLEL.md) § the per-frame index.
- **Rebuilding from scratch each frame** (no persistent handles, uniform dense field)?
  `MortonGrid3` has the cheapest build and cull — there's nothing to maintain, you refill
  it. If the field is *skewed* rather than uniform, try `LinearOctree3` /
  `LinearQuadTree`: same rebuild-friendly shape, adaptive where the points actually are.
- **Static and query-heavy, especially clustered?** `KdTree3` in 3D, `KdTree2` in 2D. The median split keeps
  depth at ~log₂(n/leaf) however the points clump, and its tight per-node boxes prune
  harder. It's also the structure that gains most from a parallel build. Measured on a
  clustered 2D set (200k points, 2000 circle culls, median of 3): `KdTree2` build **7.96
  ms** and cull **7.42 ms** — the fastest of both columns, against `QuadTree` 33.50/11.45,
  `LinearQuadTree` 16.17/12.76 and `MortonGrid` 8.96/15.23 — while k-NN is a hair behind
  the pointer quadtree (1.47 vs 1.39 ms). The cull ratio is **~1.62×** measured paired
  (A/B/B/A); taken as two separate measurements it reads anywhere from 1.50 to 1.73.
  Reproduce: `cargo run -p bench-runner --release -- --group kd --repeat 3`.
- **Integer world (tiles, pixels)?** `IntegerTree` avoids float boundary fuzz entirely.
- **`item_limit` / `capacity`** is the main tuning knob: smaller = deeper tree, fewer
  per-leaf tests, more nodes; larger = shallower, more brute per leaf. 8–16 is a good
  default; profile with the Criterion suite / regression gate (`benches/README.md`).
- **Measure your workload, and at YOUR population.** The decision maps rank the structures
  head-to-head on a moving-points sim: `examples/decision2d.rs` (2D, knobs `D2_POP` etc.)
  and `critters3d_headless --sweep` (3D). The winner moves with population, and the two
  dimensions do not agree:

  | moving points, per-frame total | 500 | 2 000 | 10 000 | 50 000 |
  | --- | --- | --- | --- | --- |
  | **2D** winner | QuadTree 1.06x | QuadTree 1.04x | QuadTree 1.09x | **MortonGrid 1.32x** |
  | **3D** winner | `Tree3`+`ItemRef` 4.0x | `Tree3`+`ItemRef` 3.6x | `Tree3`+`ItemRef` ~4x | `Tree3`+`ItemRef` 2.2-7.6x |

  In **3D the kept binary tree dominates maintain** (15 of 16 sweep configs, 1.6-7.6x over
  the runner-up) and ties for best cull. In **2D it is consistently 4-10% behind the
  QuadTree** — the 4-way split halves the depth, so `locate` is cheaper — and at 50k the
  Morton rebuild takes both columns. Same handle layer, opposite verdict, purely because
  of dimension.

  And the reason the k-d trees win where they win is countable, with no clock involved
  (`examples/work_counters.rs`): on **clustered** points a `KdTree3` k-NN query tests 219
  points to `Tree3`'s 404, but on **uniform** points it is 86.6 to 92.1. The median split is
  not a faster tree; it is the tree that does not care how the points are distributed. If
  your data is uniform, the cheaper build wins and the k-d tree has nothing to sell you.

  **`KdTree2` is in that map too, and loses it.** On moving 2D data (50k, leaf 8) its
  per-frame rebuild costs 5 628 µs against Morton's 3 551 and QuadTree's 4 895 maintain,
  and its second-best cull (3.65 µs) does not repay the difference. That is the median
  split's build cost showing up exactly where the 3D twin never has to pay it — the k-d
  trees are for data that stops moving.
- **Or don't choose at all.** `AdaptiveIndex` (3D) and `AdaptiveIndex2` (2D) own the items
  and hold whichever structure currently fits, migrating when the workload genuinely
  changes: a brute scan while the population is small, `Tree3`/`Tree` + `ItemRef` while
  things move, a rebuilt Morton grid when queries per item get high enough to pay for the
  rebuild, `KdTree3`/`KdTree2` once nothing has moved for a while. Handles (`Slot`) survive
  every migration *and* every removal — the item list is a slot table with a free list, not
  a `Vec` that gets swap-removed, because compacting would silently repoint somebody else's
  handle. The hysteresis is the hard part and it is deliberate: a candidate must win
  `hold_ticks` consecutive ticks, boundaries widen by `margin` in the direction of travel,
  and there is a `cooldown` after each switch — without those it flaps at the boundary and
  loses to *both* candidates. Thresholds come from `VH_CALIBRATION` if you point it at a
  file the `calibrate` example wrote, because the defaults are one machine's measurements.

  Worth knowing: **in 2D the margins are thinner**, so it has less to win. The kept binary
  tree leads by 1.6-7.6x on 3D maintain but trails a `QuadTree` by 4-10% in 2D. The two
  policies are held identical by a test that runs one script of work through both and
  compares the sequence of backends they pick.

  **And it is insurance, not optimisation — measured both ways.** On a *stationary* workload
  (`fluid_wgpu`, one neighbour query per particle per frame) it reaches **parity with the best
  fixed choice** — 347-360 fps against 352 — having found that choice itself, and beats the
  other two by 8-15%. On a workload that *changes character* four times
  (`examples/adaptive_vs_pinned`) it runs at **0.70× the best pinned backend**, while turning
  the catastrophic guess into a survivable one: a pinned brute scan takes ~22 000 ms on that
  script where the adaptive index takes ~1 200. Reach for it when you cannot know the workload
  in advance; pin the structure when you can.
- **"Grids rebuild, trees keep" was an API limit, not a law.** `MortonGrid3::update` /
  `MortonGrid::update` move an item in place: told where it *was*, the grid finds it in that
  one cell, and if it has not left the cell there is nothing to do at all. A rebuild costs the
  same however few items moved, so the win tracks the moving fraction (50k points, cells
  holding ~1.1 items):

  | fraction moving per frame | keep | rebuild | speed-up |
  | ---: | ---: | ---: | ---: |
  | 100 % | 8.02 ms | 6.08 ms | **0.76×** (rebuild wins) |
  | 50 % | 3.95 ms | 6.22 ms | 1.57× |
  | 10 % | 0.78 ms | 6.19 ms | 7.98× |
  | 1 % | 0.067 ms | 6.17 ms | 91.9× |
  | 0.1 % | 0.006 ms | 5.96 ms | **938×** |

  **Confirmed independently by the 2D decision map**, which moves *every* point every frame —
  the far side of that crossover. There the kept grid loses maintain exactly as predicted
  (50k: 3 715 µs against the rebuild's 2 200) while *winning* the cull by 1.11–1.20×, most
  likely because its buckets keep their addresses while a rebuild re-allocates every one of
  them each frame. Net, the rebuild still takes the frame. Two different benchmarks, one
  crossover, and neither was fitted to the other.

  **The adaptive linear trees got the same API and a much worse verdict.** `LinearOctree3`
  and `LinearQuadTree` are the same bucket hash and had the same omission, but their keep path
  loses far earlier — and it degrades the queries, which the flat grid's does not:

  | fraction moving | keep | rebuild | speed-up | leaves after 300 frames |
  | ---: | ---: | ---: | ---: | ---: |
  | 100 % | 20.4 ms | 2.40 ms | **0.12×** | 10 375 |
  | 10 % | 2.07 ms | 2.20 ms | 1.06× | 7 236 |
  | 1 % | 0.237 ms | 2.29 ms | **9.67×** | 6 990 |

  The keep path costs more here than on a flat grid and wins over a narrower band, because
  `from_items` is a fast bulk build while `update` pays a leaf descent and possibly a
  subdivision.

  **The leaf column is why these trees also needed `try_merge_up`,** which the four pointer
  trees have always had and these never did — they had no removal, so nothing ever left a leaf
  and there was nothing to collapse. Without it, over the same 300 frames the same workload
  reaches 23 385 / 18 822 / 10 756 leaves — **2.25× / 2.60× / 1.54×** what a rebuild produces,
  and still climbing when the run ended. (Those ratios are the previously-measured no-merge
  counts divided by the corrected baseline, not a fresh run: deleting `try_merge_up` cannot
  change `from_items`, which never merges, so the two measured quantities compose. The
  no-merge configuration has not been re-run since.) That is a slow leak, not a fixed tax, and a 20-frame
  window showed only 13 914 leaves at 10 % churn and hid the trend entirely (see
  [`MEASURING.md`](MEASURING.md) § 8c).

  > **Correction, 2026-09-12.** This paragraph used to divide those counts by **6 939** and to
  > say a rebuild "ends with 6 939 leaves". It does not: 6 939 is the leaf count of the
  > *starting* distribution, and the workload is a **clamped** random walk, which piles points
  > against the walls — so after 300 frames the points genuinely need more leaves than they did
  > at the start, and that has nothing to do with the structure. Measured against the right
  > baseline (`from_items` on the **current** contents), the kept tree's shape is not merely
  > close, it is **exactly equal**: 10 375 / 7 236 / 6 990 against 10 375 / 7 236 / 6 990.
  > **With the merge, the drift is zero**, and `grid_keep_bench` now asserts it row by row.
  >
  > The same paragraph also claimed culls ran **1.37×** a fresh tree's. That column was two
  > independent `wall_ms` calls, not a pair: two runs of the same binary read 1.115× and 1.50×
  > for one row. Paired through `compare2` it reads 0.94–1.01× with a 17–47 % spread — no
  > measurable difference, which is what equal shapes must produce. See § 8j of
  > [`MEASURING.md`](MEASURING.md).

### When is `RadixTrie3` actually the right pick?

Short answer: **when your access pattern is a lookup, not a search.** Its query is the slowest of
the 3D family and `radix_trie_bench` says so at every radius — if you are culling spheres, use
`Octree3`. What it has instead is that *the key is the identity*, and that buys four things nothing
else here offers:

1. **`region(prefix, digits)` returns a cell as a borrowed contiguous slice** — no allocation, no
   copy, no geometry, O(digits) to find. Every other structure answers a region by descending with
   box tests and pushing survivors into a fresh `Vec`. That is right for an arbitrary sphere and
   wrong for *"give me cell 0o5273"*. Fetching a tile/chunk by address, streaming a region to a
   peer, or iterating the world cell by cell are all this verb.

   > **Correction, 2026-09-19 — and the honest win is much smaller than first published.** This
   > point read *"0.085 µs against 22–197 µs for a box cull: **265–1531×**"*, and that comparison
   > raced a **lookup against a search**. `MortonGrid3` had no cell lookup at all — not because a
   > hash of buckets cannot do one, but because nobody had asked — so the strongest alternative was
   > missing from its own comparison. The same omission the tree-partition arm had, found by
   > applying the same question to the bench next door. `MortonGrid3::cell` exists now.
   >
   > Re-measured against it, asking for the identical cell with the item counts asserted equal, by
   > prefix length `d` against a grid at `levels = 5`:
   >
   > | `d` | buckets the grid unions | `region` | `MortonGrid3::cell` | grid / region |
   > | ---: | ---: | ---: | ---: | ---: |
   > | 1 | 4 096 | 0.18 µs | 175 µs | **967×** |
   > | 2 | 512 | 0.18 µs | 21.2 µs | 115× |
   > | 3 | 64 | 0.15 µs | 2.56 µs | 17× |
   > | 4 | 8 | 0.09 µs | 0.19 µs | 2.0× |
   > | **5** | **1** | 0.18 µs | **0.037 µs** | **0.20× — the grid wins ~5×** |
   >
   > **At the grid's own resolution the grid wins**: one hash lookup beats descending five levels
   > of trie. `region`'s real property is that it is **flat in `d`** — O(depth), resolution
   > independent — so the trie's advantage is strictly the **multi-resolution** case and grows as
   > `8^(levels−d)`, which is simply how many buckets a fixed-resolution index must union to answer
   > a coarser question.
   >
   > **So: one cell size → `MortonGrid3::cell`. A hierarchy of cell sizes (LOD, tiles at several
   > zooms, streaming at varying granularity) → `RadixTrie3::region`.**
2. **`cell_of(point, digits)` needs no index at all.** A peer, a client, or a file format can
   compute which cell an object belongs to from the point alone. Addresses become portable.
3. **Any contiguous run of keys is a coherent shard.** `key_partition_bench` measures a query
   reaching **~2 % of shards** under a curve key against **~47 %** under a balanced-but-unordered
   one, and 100 % under clustering.

   > **Correction, 2026-09-19.** This point used to continue: *"a pointer tree's subtree
   > populations are whatever the data made them, so 'give me K equal parts' has no answer in the
   > tree"*. **That is false**, and the user said so. Partitioning by tree structure is the
   > standard approach with a decade of literature behind it — SpatialHadoop ships Quadtree,
   > KD-tree and STR partitioners beside its Z-curve and Hilbert ones, and R\*-Grove describes the
   > family as *"reuse existing index search trees as-is … use its leaf nodes as partition
   > boundaries"*.
   >
   > Measured (`key_partition_bench` now has the arm): a tree partition **grouped the obvious
   > way**, largest leaf into the emptiest shard, has fine balance and its spatial quality
   > collapses — R\*-Grove's Q2 overlap reads **288 and 13 381** against Morton's 4.5 and 6.8,
   > because greedy packing groups leaves by *size* and a shard ends up a union of compact cubes
   > scattered over the whole world. Grouped **in curve order** with leaves much finer than a
   > shard, it lands on Morton's numbers in **every** column, balance included.
   >
   > So the honest statement is the opposite of the old one: **a tree partition done properly is
   > not worse, it is the same thing** — a tree whose leaves are finer than a shard, walked in
   > curve order, *is* a key sort at leaf granularity. What the key adds is that it can cut
   > **anywhere**, where a tree cuts only at node boundaries; that quantisation is what forces a
   > choice between balance and locality (coarse leaves in curve order read **30× max/mean** at
   > K = 512). And what survives outside the table: which shard owns a point is two comparisons on
   > a number its holder computes, where a tree partition is a node→shard directory somebody has
   > to ship, agree on and keep in step.
4. **The shape is canonical.** Six build orders, including reversed and already-sorted, produce
   one structure (asserted). Useful when two machines must agree on a layout without exchanging it.

And the build is genuinely competitive: sorting by Morton and building bottom-up beats `Octree3`'s
insert path by **1.5–4.2× on clustered data** (it ties or loses on uniform, and these were measured
on a noisy laptop — read the range, not the digits).

So: **`RadixTrie3` if you address by key or shard by range; `Octree3` if you query by shape.** If
you find yourself wanting both, you want `Octree3` plus a Morton sort of your own ids, and that is
a fine answer too.

### Index quality across all twelve — R\*-Grove's metrics, and what they cannot say

`key_partition_bench` borrowed three of R\*-Grove's five quality metrics to compare *partitioners*.
They were defined for spatial **indexes**, so `examples/index_quality` asks them of the kit's own
structures — measuring the **tight bounding box of what each leaf holds**, not the leaf's own box,
because the first asks how much dead space the index claims and the second only asks whether it
tiles the world. (Six structures exposed `(box, count)` and not their items, which is why this
table did not exist; `visit_leaf_items` is now uniform across all twelve.)

The five, and what each is actually asking:

| | sums | wants | why |
| --- | --- | --- | --- |
| **Q1** volume | each leaf box's volume | small | **dead space** — a leaf claiming a cube whose points sit in one corner gets descended into by every query that grazes the cube |
| **Q2** overlap | pairwise intersection of leaf boxes | zero | a query landing in the shared part must walk both subtrees |
| **Q3** margin | each box's side lengths | small | separates **shapes at equal volume**: a cube and a long sliver can measure the same, but the sliver has far more surface, so more queries touch it for the same contents (the criterion the R\*-tree adds over the R-tree) |
| **Q4** utilization | fill against capacity | high | a half-empty leaf pays a header, a pointer and a descent for few items |
| **Q5** balance | stddev of leaf sizes | small | one fat leaf ruins the worst case |

Normalised below: Q1 as a fraction of the world, Q3 in multiples of the world's side, Q4 as mean
fill over capacity, Q5 as stddev over mean. **"Knob" is the granularity dial**, and it is a
different quantity per structure — leaf `capacity` for the trees, `levels` for the grids, `bits`
for `RadixTrie3` — which is why the matched table has to *search* for it rather than compute it.

20 000 points, leaf capacity 16 where the structure has one, `levels 4` for the grids:

| | leaves | Q1 dead vol | Q3 margin | Q4 fill | **Q5 balance** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **uniform** | | | | | |
| `Tree3` | 1 831 | 0.573 | **386.4** | **0.68** | 0.255 |
| `Octree3` | 4 058 | 0.269 | 462.1 | 0.31 | 0.444 |
| `LinearOctree3` | 4 058 | 0.269 | 462.1 | 0.31 | 0.444 |
| `MortonGrid3` | 4 058 | 0.269 | 462.1 | – | 0.444 |
| `KdTree3` | 2 048 | 0.598 | 418.9 | 0.61 | **0.043** |
| `RadixTrie3` | 19 989 | *0.000* | *0.0* | – | 0.023 |
| **clustered** | | | | | |
| `Octree3` / `LinearOctree3` | 4 051 | 0.000 | 49.1 | 0.31 | 0.614 |
| `MortonGrid3` | **44** | 0.001 | 4.1 | – | **1.175** |
| `KdTree3` | 2 048 | 0.016 | 51.7 | 0.61 | **0.043** |

Three findings, and the first is against the metrics themselves.

**★ Q1 and Q3 reward degeneracy, so they are only readable at a comparable leaf count.**
`RadixTrie3` scores a perfect 0.000 volume and 0.0 margin on uniform data — because it has 19 989
leaves for 20 000 points, and a box around *one* point has no volume and no margin by definition.
Minimising either metric alone drives you to one item per leaf: an index that prunes nothing and
costs a descent per point. This is the **second** time this week an R\*-Grove metric read in
isolation selected the worst candidate present; the first was an x-stripe partitioner with the best
Q1/Q2 of any arm and by far the worst query fan-out.

**★ On uniform data `Octree3`, `LinearOctree3` and `MortonGrid3` are the same partition** — identical
leaf counts and identical Q1/Q3/Q5 to three decimals. An item limit applied to evenly spread points
subdivides evenly, and that is a grid; the adaptivity has nothing to adapt to. Under clustering they
separate immediately and enormously: the grid collapses to **44** non-empty cells at Q5 **1.175**
while the octrees hold ~4 000 at 0.614. (`Octree3` and `LinearOctree3` agree in *both* columns, which
is the cross-check you want — same algorithm, different storage, so disagreement would be a bug.)

**★★ And tuning every structure toward a matched leaf count reverses the first table**, which is the
degeneracy above caught in the act. At its default capacity `Octree3` reads Q1 = 0.269 against
`Tree3`'s 0.573 and looks twice as tight — it had 4 058 leaves against 1 831. Matched, the order
flips: `Tree3` 0.573 at 1 831 leaves, `Octree3` 0.822 at 750. A binary longest-axis split claims
*less* dead space than octants once it stops being paid in resolution. **A metric that moves with the
knob cannot be read at two different knobs.**

**★★ `RadixTrie3` at `bits b` *is* `MortonGrid3` at `levels b`** — exactly: same leaf count, same Q1,
same Q3, same Q5, in both distributions (512 / 0.856 / 182.3 / 0.160 uniform; 689 / 0.001 / 23.0 /
0.855 clustered). A Morton trie descending to a fixed depth with no item limit partitions space into
precisely the `8^b` cells of a grid at that resolution: same partition, different storage — a trie
descent against a hash lookup. That is the fourth time this family has turned out to be one structure
in different clothes, and the first time it is an *identity* rather than a resemblance.

The matching is deliberately loose and the `leaves` column says so: capacity is a near-continuous
knob, but a grid or a fixed-depth trie can only have `8^b` cells — 512 or 4 096, nothing between. An
exactly matched comparison across both families does not exist.

> **And these are NOT each structure's sweet spot — that is a third, different table.** Three
> comparisons are possible and they answer different questions:
>
> 1. **Same knob value** (`cap 16` everywhere) — compares nothing. `cap 16` means different things
>    to a binary tree and an eight-way one: it gave `Octree3` 4 058 leaves against `Tree3`'s 1 831.
>    That is the error the matched table exists to fix.
> 2. **Matched granularity** — the table above. Answers *"at equal resolution, whose boxes are
>    tighter"*, which is the only way Q1 and Q3 can be read at all.
> 3. **Each at its own optimum** — answers *"who wins when properly tuned"*, requires saying what
>    you are optimising (query time? memory? build?), and is not measured by `index_quality`, which
>    never starts a clock. It is measured by **`examples/sweet_spot`**, and the section below is
>    its answer.
>
> The cost of choosing (2) is worth stating plainly: matching the granularity **pushed some
> structures to settings nobody would ship**. `Octree3` landed on `cap 48`, well above the usual
> 8–16, and its utilisation fell to 0.56. You cannot have both — a knob-sensitive metric is only
> comparable with the knob pinned, and pinning it takes structures away from their sensible
> settings. For sweet spots, use the thing that does hold a clock: `examples/sweet_spot`, below.

### The sweet spot, measured — and there is no such thing as *the* sweet spot

`examples/sweet_spot` sweeps each structure's own knob (`item_limit` / `capacity` / `levels` /
`bits`) against six objectives — build time, cull time, k-NN time, bytes, and `build + q·cull` at
a light (`N/10`) and a heavy (`N`) query load — across **48 cells** (12 structures × uniform and
clustered × a small and a large query radius, N = 50 000). Every knob within **5 %** of the best is
counted as tied for best, so what is reported is a band and not an argmin.

**In 44 of those 48 cells, no single setting is within 5 % on all six objectives.** The four
exceptions are `RadixTrie3` on uniform data and `LinearOctree3` on clustered data, both radii.

That is not bad luck, it is forced, and the exact counts say why. Sweeping the knob from fine to
coarse, **boxes classified per query falls monotonically** (fewer, bigger leaves ⇒ a shallower
descent) while **points tested per query rises monotonically** (a bigger leaf is scanned linearly).
Two costs moving in opposite directions produce an interior optimum for queries. Build and memory
have no such tension — both simply get cheaper as the structure gets coarser — so they sit at the
end of the ladder. Hence the gradient:

| objective | how often the ladder's COARSEST setting is within 5 % of best |
| --- | ---: |
| `bytes` | **46 / 48** |
| `build` | **40 / 48** |
| `cull` | 21 / 48 |
| `knn` | **4 / 48** |

So the practical rule is to name the objective before the knob:

- **Minimising memory or build (a structure you rebuild every frame, or a big static one you must
  fit):** go coarse. The coarsest setting on the ladder is essentially always within 5 % of the best
  either objective can do.
- **Minimising cull:** a middle setting — `item_limit` 32–96 for the trees, and for the grids it
  depends on the data (see below).
- **Minimising k-NN: go noticeably finer than you would for cull.** In **35 of 36** tree cells the
  k-NN band's centre sits below the cull band's — typically 16–32 against 32–64. A k-NN descent
  tightens its own radius bound as it goes, so more and smaller leaves buy pruning that a
  fixed-shape cull cannot use.

**★ And the grid's ladder points the other way depending on the data.** On uniform points
`MortonGrid3` wants its coarsest (`levels 3`) on every objective. On clustered points it wants
`levels 5–7`: at `levels 3` a whole blob lands in one cell and a query tests **2 620 points**
against **66** at `levels 7`. Its memory barely moves there (**1.17×** across the whole ladder,
because clustered data occupies few cells at any resolution), so on clustered data the grid has no
build-versus-query conflict at all and you should simply refine it. On uniform data it does have
one, and refining costs you.

#### How much the knob can cost you, by structure

Worst-axis spread across the ladder, median over the cells — i.e. how badly a careless setting can
hurt:

| structure | worst axis | build | cull | k-NN | bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `KdTree3` / `KdTree2` | **2.2×** | 1.7–1.8× | 1.4–1.6× | 1.4–1.6× | 2.2× |
| `RadixTrie3` | 2.9× | 2.0× | 2.7× | 2.7× | 2.0× |
| `Tree` / `LinearQuadTree` / `Tree3` / `QuadTree` | 3.6–3.8× | 2.9–3.6× | 1.6–2.2× | 1.4–1.8× | 1.9–3.8× |
| `LinearOctree3` / `IntegerTree` / `Octree3` | 4.1–4.6× | 3.2–3.6× | 1.6–3.0× | 1.8–2.0× | 2.1–4.4× |
| `MortonGrid` | 5.7× | 2.3× | 3.7× | 5.6× | 2.2× |
| `MortonGrid3` | **10.7×** | 2.6× | **7.0×** | 7.4× | 2.5× |

**The k-d trees are the hardest to misconfigure and the Morton grids by far the easiest**, a factor
of about five apart. A median split derives the partition from the points, so `capacity` only sets
how early the recursion stops; a grid's `levels` sets the cell size outright, and getting that wrong
is the pathology `grid_min_hits` exists to veto. The single worst reading in the whole sweep is a
uniform `MortonGrid3` cull at radius 60: **44.7×** between its best and worst `levels`.

Caveat on the microsecond columns, as ever: they are one laptop on one night, and the leverage
ratios in particular are the kind of quantity MEASURING.md § 8e says moves between runs. The
**counts** (boxes and points per query), the monotonicity directions, and the 44/48 verdict are
arithmetic over a fixed point set and say the same thing anywhere.

#### What this says about the knob `AdaptiveIndex` picks for you

Worth checking, since the sweep is the first thing able to price it. `AdaptiveIndex` takes one
`leaf` from its caller and gives it to *both* the `Tree3` backend (as `item_limit`) and the
`KdTree3` backend (as `capacity`); the grid sizes itself with
`MortonGrid3::levels_for_cell_size(world, q_extent)`.

**Sharing one number between the tree and the k-d tree is cheap**, and the leverage table is why:
`KdTree3` is the least knob-sensitive structure in the kit (2.2× worst-axis) while `Tree3` is 3.7×,
so a caller who picks for the tree loses almost nothing on the k-d side. Pick for the tree.

**The grid's self-sizing lands 1–2 levels off the per-level optimum in all four cells tested** —
never on it — because it is geometry only and cannot see clustering. The penalties are 1.05–1.09×
on `cull` throughout, and on k-NN `1.00× / 1.00× / 1.83× / 12.46×`.

**And the 12.46× is unreachable, which is the interesting part.** It is the sparse-uniform
small-query case, where queries return **1.01 items** against a `grid_min_hits` default of 9 — so
the threshold that decides *whether* to hold a grid rejects that workload before cell sizing is ever
consulted. The 6.77-hit cell is rejected too. Of the two reachable cells one is clean and the other
pays **1.83× on k-NN**. No default is worth changing on that evidence (the reachable `cull`
penalties sit inside the 5 % tie band), but the k-NN gap is real and data-aware cell sizing is
queued as #183 with that figure as its prize.

**★ The k-d trees' Q5 is 0.043 in every row, uniform and clustered alike.** A median split puts half
the points either side by construction, so balance stops being a property of the data and becomes a
property of the algorithm. Nothing else is within 2.5× on uniform data and the grids are 20–27×
worse under clustering. That column is the argument for the k-d trees, and this repo had never
measured it — the metric came from a partitioning paper rather than an index one.

**And Q2 (overlap) is identically zero for all twelve, asserted rather than printed.** The reason is
worth spelling out, because it is the one structural division in spatial indexing this document
never states outright:

- **Partitioning space** — everything in this kit. You decide the regions *before* looking at the
  objects: an octree cuts the cube into eight octants, a grid into cells, a k-d tree with a plane
  into left and right. The regions come from geometry, so they **cannot overlap by construction**,
  and every point falls in exactly one.
- **Grouping by data** — the R-tree family. You decide the *groups* first ("these eight objects are
  near each other, they go together") and each group's box is whatever encloses its members. Because
  the box is *derived* from the membership rather than carved out beforehand, two groups' boxes are
  free to sit on top of each other.

Overlap is the price of the second: if two boxes intersect and a query lands in the shared part, both
subtrees must be walked, where disjoint regions let a point query follow exactly one path. Half the
R-tree literature (R\*-tree and successors) is about minimising it. So a zero here is not "we are
good at this", it is "the question does not arise" — and a metric that cannot separate twelve
candidates is the wrong instrument, not a weak signal. It is asserted so that a structure which ever
*does* group by data fails loudly, where a column of noughts would quietly gain an entry.

(R-trees earn that cost honestly: they index **extended** objects — rectangles, polygons, a building
— where partitioning space forces you to file one object in several cells. This kit indexes points.)

**Q4 likewise needs a capacity to be a fraction of**, and the three unbounded-bucket structures print
`–` rather than a fabricated number. That is not a gap in the library, it is the two designs
inverting the same trade: a tree **fixes occupancy** (at most N per leaf) and lets resolution adapt,
while a grid **fixes resolution** (`levels`) and lets occupancy adapt. Give a grid a capacity and it
must subdivide when full — which makes it an adaptive tree. What a grid has instead is
[`Occupancy`](../crates/vectorial-hash/src/morton3.rs), the same concern expressed as items per
non-empty cell rather than as a fraction of a ceiling.

  **Why zero, and which structures actually get it.** `merge_limit == item_limit`, and the split
  planes are **positional** (fixed octants, not a data-dependent median). So "this node is
  subdivided" is equivalent to "this node holds more than the limit" — the same predicate a fresh
  build evaluates, with no hysteresis band between splitting and merging for history to hide in.
  A structure that split on a **median** could not have this property, which is the same reason
  `KdTree2`/`KdTree3` cannot maintain at all.

  `tests/shape_is_history_free.rs` sweeps **12 seeds** across all nine structures that can
  maintain, and the answer is not uniform:

  | | seeds that drifted | worst | why |
  | --- | ---: | ---: | --- |
  | `QuadTree`, `Tree3`, `Octree3` | 0 / 12 | 1.0000× | position **and** axis chosen from the box |
  | `LinearQuadTree`, `LinearOctree3` | 0 / 12 | 1.0000× | same, over a hash |
  | `MortonGrid`, `MortonGrid3` | 0 / 12 | 1.0000× | no shape to change at all |
  | **`Tree`** | **12 / 12** | 1.0143× | square node picks its **axis** by counting items |
  | **`IntegerTree`** | **11 / 12** | 1.0340× | the same policy, transcribed to integers |

  The two binary 2D trees ask the data a question — for a **square** node, `pick_split_by` counts
  which axis distributes the items more evenly — and that count is taken on whatever the node held
  at the moment it split. **A split that asks the data a question remembers the answer.** `Tree3`
  is binary as well and is exempt only because it splits the *longest* axis on a `>=` tie-break,
  which is pure geometry. The effect is a couple of leaves in ~740; the point is that it is not
  zero, and that one seed said the opposite (the first seed tried had `IntegerTree` drifting and
  `Tree` not, which reads as "the integer tree is the odd one out" — a conclusion about the wrong
  thing, since the two share the policy verbatim).

  **A second, rarer mechanism, in all five pointer trees.** `divide` refuses to split a node whose
  items are **all at one point**; `try_merge_up` collapses children only when they *fit in one
  leaf*. Different predicates, so "spread out, split, then became coincident" is a one-way door:
  the maintained node stays subdivided where a rebuild would refuse to split at all. It needs
  exact coincidence to bite, which is why it showed up in 2D (a clamped walk pins escapees onto
  four exactly-coincident corners) and not in 3D (a point must clamp on all three axes at once).
  Left as it is deliberately — closing it means an O(combined) scan on the branch a *rejected*
  merge takes, which is the common branch on the relocation hot path. Pinned by its own test.

  The crossover sits near **70 % moving**, and note the movement here is deliberately harsh —
  40-unit steps against 15.6-unit cells, so 99 % of updates actually re-bucket. A workload
  whose items mostly stay in their cell does even better. It costs O(occupancy of the old
  cell), which is one more reason the tuning knob matters ([`Occupancy`]). If everything moves
  every frame, keep the rebuild.
- **Don't reach for threads first.** Reads parallelise (`cull_many_par`), writes don't —
  the lever for write-heavy loops is `update_ref` + the right structure, not rayon. See
  [`PARALLEL.md`](PARALLEL.md).
- **An index only knows what it holds.** Items outside its world box are dropped at
  insert/bulk-load time, so an index and a linear scan will legitimately disagree about
  anything that escaped. If you compare the two (and you should), make sure both are
  looking at the same set — see the bug in [`STEALTH.md`](STEALTH.md).
