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
| **`KdTree3`** | 3D | static | **skewed/clustered, query-heavy** | median select (**3.4× on 16 threads**) | **cull 2.2–2.5× `Tree3` on clusters**, k-NN 1.67× |
| **`LinearOctree3`** | 3D | static | skewed data you **rebuild often** | ~2.1× faster than `Octree3` | loses cull ~1.3× to `Octree3` |
| **`LinearQuadTree`** | 2D | static | skewed 2D you rebuild often | fast | **won the fluid's neighbour query** |
| **`KdTree2`** | 2D | static | **skewed/clustered 2D, query-heavy** | **fastest of the 2D builds** (2.8× on 16 threads) | **cull 1.54× the pointer quadtree** |

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
  maintain win (the decision map flipped on it). One extra field per entity.
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
  the pointer quadtree (1.47 vs 1.39 ms).
  Reproduce: `cargo run -p bench-runner --release -- --group kd --repeat 3`.
- **Integer world (tiles, pixels)?** `IntegerTree` avoids float boundary fuzz entirely.
- **`item_limit` / `capacity`** is the main tuning knob: smaller = deeper tree, fewer
  per-leaf tests, more nodes; larger = shallower, more brute per leaf. 8–16 is a good
  default; profile with the Criterion suite / regression gate (`benches/README.md`).
- **Measure your workload.** The decision maps rank the structures head-to-head on a
  moving-points sim: `examples/decision2d.rs` (2D) and `critters3d_headless --sweep` (3D).
  Rough 2D read: **QuadTree** is the all-rounder; **MortonGrid** (rebuilt each frame) wins
  **dense + high-churn** scenes where the trees' `update` split/merge churn dominates.
- **Don't reach for threads first.** Reads parallelise (`cull_many_par`), writes don't —
  the lever for write-heavy loops is `update_ref` + the right structure, not rayon. See
  [`PARALLEL.md`](PARALLEL.md).
- **An index only knows what it holds.** Items outside its world box are dropped at
  insert/bulk-load time, so an index and a linear scan will legitimately disagree about
  anything that escaped. If you compare the two (and you should), make sure both are
  looking at the same set — see the bug in [`STEALTH.md`](STEALTH.md).
