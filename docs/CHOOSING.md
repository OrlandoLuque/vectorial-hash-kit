# Choosing a structure

`vectorial-hash` ships **eleven** spatial indexes. They all answer the same two queries —
`cull` (everything inside a shape) and `knn` (k nearest neighbours) — so picking one is
about *your data and access pattern*, not features. This is the one-glance guide; the
quantitative backing is the decision map in [`THREE_D.md`](THREE_D.md), the parallelism
crossovers in [`PARALLEL.md`](PARALLEL.md), and the demo write-ups
([`FLUID.md`](FLUID.md), [`POINTCLOUD.md`](POINTCLOUD.md), [`STEALTH.md`](STEALTH.md))
where each one is measured on a real workload rather than a synthetic one.

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
| **`LinearOctree3`** | 3D | static | skewed data you **rebuild often** | ~2.1× faster than `Octree3` | loses cull ~1.3× to `Octree3` |
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
- **Check whether you need an index at all.** Below ~500–1000 items a contiguous scan
  wins — no descent, no allocation, perfect cache behaviour. The kit says so out loud:
  `advisor::BRUTE_FORCE_MAX`, the `formations` regiment level, and the stealth HUD all
  report the scan winning at small N. Don't index 40 guards.
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

  | fraction moving | keep | rebuild | speed-up | leaves | cull vs a fresh tree |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 100 % | 13.70 ms | 3.00 ms | **0.22×** | 23 279 | **1.30× slower** |
  | 10 % | 1.10 ms | 2.62 ms | 2.38× | 13 914 | 1.20× slower |
  | 1 % | 0.098 ms | 2.20 ms | **22.4×** | 8 670 | 0.99× (none) |

  The extra column is the cost of keeping an *adaptive* structure: it holds splits made for a
  distribution the points have left and never merges an emptied leaf, so a hard-churned tree
  ends with **3.4× the leaves** a rebuild would produce and answers 30 % slower. The answers
  stay exact — that is tested — but the shape does not stay good. Use the keep path on the
  linear trees when churn is low; rebuild periodically if it is not.

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
