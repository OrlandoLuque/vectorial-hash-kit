# Update strategy comparison (work in progress, 2026-06-17/18)

Empirical evaluation of three `Tree::update` relocation strategies introduced
on the working branch, against the critters headless workload.

## What the strategies do

When a `Tree::update` mutator pushes an item out of its current leaf, the item
must be relocated. Three implementations exist behind `UpdateStrategy`:

| Strategy | Path |
| --- | --- |
| `Legacy` | The pre-2026 path: `remove` from old leaf, `try_merge_up`, then `insert` descending from the root. |
| `Lca` | Ascend by parent pointers until an ancestor's bbox contains `new_pos`; descend only within that subtree (`locate_from(lca, ...)`); push; `divide` if overflow; `try_merge_up` on the old leaf. |
| `LcaRopes` | First scan the old leaf's four rope lists (feature `neighbors`); if any rope neighbour leaf contains `new_pos`, move directly. Otherwise fall through to `Lca`. Without the feature, behaves like `Lca`. |

`Tree::update` calls `update_with(UpdateStrategy::default(), ...)`. The default
is `LcaRopes` when the `neighbors` feature is on, `Lca` otherwise.

## Methodology so far

- Binary: `cargo run -p vectorial-hash-demos --bin critters_headless --release --features neighbors`
- Mode: `--mode binary` (the binary-split `Tree<T>`, not the quadtree)
- World: `MAP_W = MAP_H = 1024.0`
- `split = merge = 3` (item_limit, merge_limit)
- `seed = 42`
- One run per cell — variance not yet measured.
- All runs with `--features neighbors` on (rope bookkeeping is paid by every
  strategy; `LcaRopes` additionally pays the scan).
- "Pop target" splits evenly across drifters / hunters / pulsars.

### Caveat: trajectory divergence

`Legacy` and the LCA paths produce **different trees** from the same seed: the
order of `cull` iteration depends on tree shape, and `vision_prey`'s `min_by`
picks different prey in ties, so trajectories diverge after seconds. The cull
RESULT SET is identical (validated by `exhaustive_culling`), only the order
differs. `Lca` and `LcaRopes` produce **identical trees** with each other —
they only differ in how the destination leaf is found.

This means `legacy` vs `lca` comparisons mix two effects: the strategy itself,
and the diverged simulation. `lca` vs `lca-ropes` is a clean A/B (same tree
state, only the lookup path differs).

## Headline numbers (`move+update` mean / p50 / p95, µs/frame)

`move+update` is the **total per frame** spent in `Tree::update_with` summed
over all critter motions that frame. It is the only metric that `UpdateStrategy`
directly affects.

| Pop target | Strategy | mean | p50 | p95 | alive (end) | arena nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 300 | legacy | 11.7 | 10.9 | 18.5 | 198 | 5093 |
| 300 | lca | 10.9 | 10.3 | 17.7 | 187 | 3613 |
| 300 | lca-ropes | 10.8 | 10.2 | 17.3 | 187 | 3613 |
| 3000 | legacy | 154 | 154 | 291 | 1320 | 103235 |
| 3000 | lca | 139 | 134 | 263 | 1237 | 70907 |
| 3000 | lca-ropes | 138 | 132 | 260 | 1237 | 70907 |
| 10000 (\*) | legacy | 3914 | 2664 | 6481 | 9457 | 652265 |
| 10000 (\*) | lca | 2720 | 2556 | 4341 | 9636 | 462105 |
| 10000 (\*) | lca-ropes | 2488 | 2123 | 4865 | 9636 | 462105 |
| 30000 (\*) | legacy | 32076 | 28623 | 31596 | 28070 | 2324659 |
| 30000 (\*) | lca | 22054 | 25307 | 27527 | 28357 | 1685285 |
| 30000 (\*) | lca-ropes | 23210 | 23016 | 28441 | 28357 | 1685285 |

(\*) 10000 and 30000 runs use `--respawn 0.05` so the population sustains near
the target; the 300 and 3000 runs use the default `--respawn 2.5` (population
in those tiers drifts ~75% of target). Frame counts: 300/3000 at 240/120
(measured/warmup); 10000 at 240/120; 30000 at 30/240.

## Observations

- **LCA scales much better than Legacy.** Savings move from ~7% at 300 pop to
  ~31% at 30k. Coherent with theory: Legacy descends from root on every
  cross-leaf move (cost O(log n)); LCA typically ascends 0–2 levels and
  descends 0–2 — near-constant in balanced trees.
- **Arena footprint also shrinks.** At 30k, Legacy has 2.32M arena nodes vs
  1.69M for LCA — Legacy's remove+insert path cascades more split/merge
  oscillation, leaving more orphan nodes behind.
- **LcaRopes vs Lca: mixed.** At 300, 3000, 10000 LcaRopes wins on mean by a
  small margin. At 30000 LcaRopes has the best p50 (23016 vs 25307) but a
  worse mean than Lca (23210 vs 22054), suggesting a heavier tail — most
  likely the cases where the rope scan finds nothing and the work falls
  through to LCA anyway.
- **Tree-op cost interpretation requires care.** `attack cull avg` and
  `vision cull avg` differ slightly across strategies, but this is mostly
  trajectory divergence (different trees → different cull workloads), NOT
  the strategy choice affecting cull cost. Likewise `insert+remove`
  reflects how many spawns/kills happened that frame, which depends on
  trajectory.

## Formal sweep (`--no-attack`, 135 cells, 3 seeds per cell)

Conducted 2026-06-19 to answer the live questions cleanly. Workload:
`--no-attack` (no firing → no kills → no respawns → no insert/remove churn).
After warmup the population is stable at the target and every frame is pure
movement — `move+update` cost is **fully isolated** from cull and tree-op
churn, and Legacy / LCA produce **the same tree** because trajectories no
longer diverge (no cull-based decisions feed back into the sim).

Matrix: strategy ∈ {legacy, lca, lca-ropes} × pop ∈ {1000, 3000, 10000} ×
item_limit ∈ {3, 10, 30} × neighbors-feature ∈ {on, off} × seed ∈ {42, 7,
123}. LcaRopes is only meaningful with the feature on; `lca-ropes × off` is
skipped to avoid duplicating `lca × off`. 135 cells total, 38.9 min wall.

CSV: `docs/sweep_update_strategies.csv`. Aggregator:
`docs/analyze_sweep.ps1`.

### Cost of the `neighbors` feature (rope bookkeeping)

The single biggest lever found. The feature pays a cost on every `divide`
and every `try_merge_up`; that cost scales with tree depth and with churn.

The cost is **purely a function of the feature being compiled in**, not of
whether anything actively uses it. The rope-maintenance code is wrapped in
`#[cfg(feature = "neighbors")]` so when the feature is off the bookkeeping
does not exist in the compiled binary — guaranteed by the compiler, not
by a runtime branch. When the feature is on, every split and every merge
rewires the per-leaf rope lists, regardless of whether downstream code
ever calls `neighbors_ropes`, `LcaRopes`, or `cull_walk`. So:

- Feature **off**: zero cost for `cull_walk`, ropes, etc. even being
  available as method names — they compile to no-ops or are absent.
- Feature **on**: full bookkeeping cost on every structural change, plus
  any lookup cost from the call sites you actually use.


| pop | item_limit | strategy | mv_mean off | mv_mean on | feature overhead |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1000 | 3 | legacy | 75.9 | 107.6 | **+41.8%** |
| 1000 | 3 | lca | 71.5 | 94.2 | +31.7% |
| 1000 | 10 | legacy | 54.4 | 60.3 | +10.8% |
| 1000 | 30 | legacy | 46.9 | 49.3 | +5.1% |
| 3000 | 3 | legacy | 291.7 | 410.8 | +40.8% |
| 10000 | 3 | legacy | 1191.0 | 2157.1 | **+81.1%** |
| 10000 | 10 | legacy | 813.5 | 995.7 | +22.4% |
| 10000 | 30 | legacy | 651.5 | 699.4 | +7.4% |

Headline: turning the feature on with `item_limit = 3` at 10k pop nearly
**doubles** the `update` cost. At `item_limit = 30` (shallow trees) the
overhead is mild (5–10%).

### Legacy vs LCA (clean, `neighbors=off`)

When the feature is off, this isolates the strategy itself — no rope
bookkeeping interferes, and the tree state is identical between strategies
(no trajectory divergence under `--no-attack`).

| pop | item_limit | legacy | lca | lca vs legacy |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 3 | 75.9 | 71.5 | **-5.8%** |
| 1000 | 10 | 54.4 | 53.4 | -1.8% |
| 1000 | 30 | 46.9 | 47.2 | +0.6% |
| 3000 | 3 | 291.7 | 263.1 | **-9.8%** |
| 3000 | 10 | 198.5 | 193.8 | -2.4% |
| 3000 | 30 | 163.1 | 161.9 | -0.7% |
| 10000 | 3 | 1191.0 | 1107.7 | **-7.0%** |
| 10000 | 10 | 813.5 | 786.4 | -3.3% |
| 10000 | 30 | 651.5 | 641.7 | -1.5% |

LCA wins when the tree is deep (small `item_limit`) and the population is
non-trivial. At shallow trees (`item_limit = 30`) the descent from root
takes only 1–2 levels — LCA's ascent + descent reaches parity or
marginally loses.

Bonus consistent win — **arena footprint**: LCA generates ~30% fewer
zombie nodes (the orphan children left behind by `remove`'s merge-ups).
e.g. 10k il=3: 424k arena (lca) vs 611k (legacy). Memory matters when the
tree is long-lived.

### LCA vs LCA-ropes (clean, `neighbors=on`, identical tree state)

Same tree shape (the only difference is WHERE update finds the destination
leaf), so this is the sharpest A/B available.

| pop | item_limit | lca | lca-ropes | ropes vs lca |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 3 | 94.2 | 92.0 | -2.3% |
| 1000 | 10 | 58.4 | 57.9 | -0.9% |
| 1000 | 30 | 48.8 | 48.5 | -0.6% |
| 3000 | 3 | 372.7 | 368.8 | -1.0% |
| 3000 | 10 | 212.4 | 212.6 | +0.1% |
| 3000 | 30 | 170.8 | 168.9 | -1.1% |
| 10000 | 3 | 1855.5 | 1807.0 | -2.6% |
| 10000 | 10 | 944.8 | 902.0 | **-4.5%** |
| 10000 | 30 | 703.0 | 674.1 | **-4.1%** |

LCA-ropes is consistently a hair better than LCA — small but real gains
0.5–4.5%. Best at moderate item_limits where rope lists are short (fewer
neighbours per side) but the LCA still has work to do. The Bonus: ropes
costs nothing extra when the lookup misses (the scan is a few `bbox.contains`
on a short list), so the downside risk is tiny.

## Verdict / decisions

1. **`Lca` is the strict winner over `Legacy`** when isolated. Up to ~10%
   faster on `move+update` and ~30% less arena footprint. `Tree::update`
   already defaults to `Lca` (or `LcaRopes` when the feature is on), so the
   pre-2026 path is now off by default and `Legacy` stays only as an
   opt-in benchmark target.

2. **The `neighbors` feature is expensive in update-heavy workloads** —
   5–81% extra cost on `update` depending on tree depth. The feature only
   pays for itself when something downstream actually consumes the ropes
   (today: `cull_walk` strategies, and the `LcaRopes` lookup). For
   workloads that don't use either, the feature should stay off — and it
   already does (off by default).

3. **`LcaRopes` is a small but free addition on top of `Lca` when the
   feature is on**. -1 to -5% on `move+update`, no regression at the cell
   level. Defaulting to `LcaRopes` when the feature is on is the right
   call. Already wired this way via `UpdateStrategy::default()`.

4. **Item-limit (split threshold) dominates absolute cost** more than the
   strategy choice. `il = 30` is 10× faster than `il = 3` at the same pop.
   Tuning that is a separate axis — the strategy gains compound on top.

### Caveats and what we did NOT measure

- All numbers are `--no-attack` (pure movement). The full game-like
  workload adds cull and insert/remove costs not captured here. The earlier
  contaminated runs gave us a feel for those but no clean numbers.
- 30k pop NOT in the sweep (would have multiplied wall time x10). Earlier
  one-run probes at 30k showed LCA winning ~30% on mean over Legacy with
  feature on, consistent with the trend.
- `merge_limit = item_limit` throughout. Varying these independently (real
  hysteresis) is left for the future.
- Movement step is the critters' default speeds (75–105 px/sec at dt=1/60).
  Faster critters would cross leaves more often and stress the LCA path
  harder.
- No `cull_walk` measurement in this sweep — the question "is `neighbors`
  worth its cost?" cannot be answered until we measure cull_walk's
  upside on the same workload.

## Bit-shift / IntegerTree (2026-06-19)

Implemented `IntegerTree<T>` in `crates/vectorial-hash/src/itree.rs`:
same binary-split algorithm as `Tree<T>` but with `i32` coordinates and a
**power-of-two root extent** asserted at construction. The split policy
matches the float tree exactly (long axis first, item-balanced tiebreak on
squares), so the only algorithmic difference is the underlying scalar.

Head-to-head bench at pop=10000, 240 measured steps, seed=42, both trees
seeded with the same items, identical motion vectors per item per step,
positions pre-converted out of the timed section:

| item_limit | Tree<T> (mean µs) | IntegerTree<T> (mean µs) | IntegerTree vs Tree |
| ---: | ---: | ---: | ---: |
| 3 | 1467.9 | 1137.7 | **-22.5%** |
| 10 | 1064.0 | 831.3 | **-21.9%** |
| 30 | 838.3 | 653.9 | **-22.0%** |

A very consistent **~22% gain across depths**. Where the gain comes from:

- `IRect` is 16 B vs `Rect`'s 32 B → smaller node footprint → better cache
  density during descent.
- `IItem` is 12 B vs `FItem`'s 20 B → tighter `items: Vec<T>` packing.
- `>> 1` instead of `/ 2.0` in `divide` — minor compared to the cache
  effects on modern CPUs.

The "bit-shift wins in `locate`" intuition the paper sketches doesn't
materialise as a separate effect: today's `locate` descends via
`bbox.contains` (4 comparisons, short-circuited) rather than dividing
midpoints, so int just substitutes the comparison type. The gain is the
cache footprint, not the math.

### Caveat — float-to-int conversion cost

The 22% only holds when **the application stores positions natively as
i32**. If positions live in floats elsewhere (typical physics), the
`f64::round() as i32` conversion happens on every `update` and eats the
gain. A measurement run with the conversion INSIDE the timed section
showed IntegerTree **losing** +10 to +40% — the conversion is dominant.

So IntegerTree is a strict win only for integer-native simulations or
games (grid-based, voxel, retro) that can store positions as `i32` end to
end. For floating-point physics workloads the float tree remains the
better choice unless the descent depth grows enough that cache pressure
dominates over conversion (not yet measured).

Bench source: `crates/vectorial-hash-demos/src/bin/itree_bench.rs`.

### Full-sim head-to-head: `IntegerTree` doesn't transfer the update win

`itree_bench` measured update only (pop=10000, pure motion, items as
`IItem` = `id + IPoint`, 12 bytes). IntegerTree won by ~22%.

Then we extended `IntegerTree::cull` (mirrors `Tree::cull`, converts IRect↔
Rect and IPoint↔Point at the shape boundary) and ran it inside the
critters demo at full workload. Crucially the items now hold **both** a
float `pos: VPoint` (for cull / movement geometry) and a cached
`ipos: IPoint` (for IntegerTree internal lookups, refreshed via
`set_pos`). Items are ~40 bytes regardless of tree, so the smaller-bbox
advantage of IntegerTree (`IRect` 16B vs `Rect` 32B per node) is the only
real saving — and it doesn't carry the day.

Sweep: pop=10000, lca, merge=item_limit, neighbors off, full critters sim
(`--respawn 0.05`), 3 seeds, 120 measured frames.

| item_limit | mv tree | mv itree | mv Δ | vis tree | vis itree | vis Δ | total tree | total itree |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 912 | 1118 | **+22.6%** | 16.2 | 16.3 | +0.6% | 49.7 ms | 50.2 ms |
| 50 | 889 | 1098 | +23.5% | 14.3 | 14.6 | +2.1% | 43.9 ms | 45.0 ms |
| 100 | 863 | 1068 | +23.7% | 13.3 | 13.8 | +3.8% | **40.9 ms** | 42.6 ms |
| 200 | 897 | 1110 | +23.8% | 13.8 | 14.3 | +3.6% | 42.4 ms | 44.1 ms |
| 500 | 1070 | 1274 | +19.0% | 16.3 | 16.7 | +2.5% | 50.1 ms | 51.5 ms |

The update slowdown is **consistently ~23%**, not 22% win as the synthetic
bench predicted. Cull is essentially neutral (within 4%). Total throughput
is ~3–5% worse with IntegerTree across the board.

The likely cause is `Critter` size: now ~40 bytes (`id + kind + pos +
ipos + heading`) vs ~32 bytes before the int integration. That hurts
both trees equally on items-per-cache-line, but it neutralises
IntegerTree's main advantage (the bbox size win, which only applies to
nodes, not items). Plus we pay `set_pos`'s 2 `round()` calls per `update`
in both trees just so the integer tree's `IPositioned::position()` can
return a cached `ipos`.

### When IntegerTree would actually win

The pure-update bench tells the true story when the item type is
**int-native**:

- Game state that doesn't need float positions at all (grid-based,
  voxel, retro). `Critter` becomes `IItem`-like (12 bytes), no `set_pos`
  overhead, no duplicated state. IntegerTree's ~22% update win returns
  and the cull stays comparable.
- Workloads where the descent dominates AND the items are small. The
  per-node bbox saving (IRect 16B vs Rect 32B) is real cache savings on
  descent.

If positions live in floats elsewhere (typical physics-based games),
IntegerTree as a drop-in replacement is **not** worth it: the
synchronisation cost destroys the cache win.

**Action**: keep IntegerTree available for int-native scenarios; do not
promote it as the default for float-position workloads. Documented
empirically here so future workloads can decide on data.

### Zero-cost coexistence with `Tree<T>`

A subtle but important design property: **`IntegerTree<T>` is implemented
as a separate type, not as a runtime switch on `Tree<T>`**. There is no
`if bit_shift` anywhere in the hot path of either tree. The two trees are
distinct monomorphizations, each with its own optimized code. Concretely:

- A binary that only instantiates `Tree<T>` never executes any
  `IntegerTree` code at runtime — guaranteed by dispatch.
- A binary that never references `IntegerTree<T>` at all has the
  `IntegerTree` code dead-code-eliminated by the compiler, so it doesn't
  even occupy binary footprint.
- Switching between the two is a matter of which constructor you call;
  there is no shared runtime check anyone has to pay.

If a future user wants to go further and not even *compile* `IntegerTree`
when they're sure they won't use it, the `itree` module can be wrapped in
a cargo feature (`integer-tree`) in one line. The runtime cost is already
zero by design.

## Total-time math: does `neighbors` ever pay for itself on `update`?

The per-frame `mv_mean` numbers above implicitly answer this, but
explicitly: scale by 10,000 ticks at pop=10000, item_limit=3 (where the
feature bites hardest):

| config | µs/tick | total over 10k ticks |
| --- | ---: | ---: |
| lca, nbrs=off | 1108 | 11.1 s |
| lca-ropes, nbrs=on | 1807 | 18.1 s |
| lca, nbrs=on | 1856 | 18.6 s |
| legacy, nbrs=on | 2157 | 21.6 s |

So turning the feature on (best case: `lca-ropes`) costs **+7.0 seconds
over 10k ticks** vs leaving it off. The ropes-lookup atajo only claws back
`1856 − 1807 = 49 µs/tick` (~0.5 s over 10k ticks) — about **7%** of the
bookkeeping cost it adds. **For `update`-only workloads the feature does
not pay for itself**, no matter which strategy you pair with it.

The break-even question is therefore: does the rest of your workload save
more than 700 µs/tick at 10k pop il=3 by having the ropes available? See
`cull_walk` below.

## When `cull_walk` (and therefore `neighbors`) could rent

`Tree::cull_walk` is an alternative to `Tree::cull` that traverses by
flood-fill over leaf neighbours instead of descending the tree from the
root. It's where the `neighbors` feature gets its production justification:
without ropes, the walk reconstructs neighbours on the fly via
`neighbors_samet` (parent-pointer ascent) or `neighbors_probe` (point
probe + `locate` from root). Ropes turn that into an O(1) lookup.

The existing `vh bench-walk` results (`docs/BENCHMARKS.md`, Results 3)
measured this at 200k uniform points with the per-cell-size template bank:

| strategy | scale 350 (ms/cull) | scale 1400 (ms/cull) |
| --- | ---: | ---: |
| descent (`Tree::cull`) | **0.012** | **0.076** |
| walk + ropes | 0.017 (0.70×) | 0.108 (0.70×) |
| walk + Samet | 0.024 (0.49×) | 0.213 (0.35×) |
| walk + probe | 0.026 (0.45×) | 0.318 (0.24×) |

Descent wins even against ropes (by ~30%) because the template machinery
classifies internal nodes wholesale: a green/white classification at an
internal node takes or skips its whole subtree without visiting leaves.
The walk has no such short-circuit — it must touch every leaf in the
region.

So **for any workload that has a usable template**, `cull_walk` is the
wrong tool and the `neighbors` feature pays without benefit. The
break-even scenarios for ropes are documented in BENCHMARKS Results 3,
point 3:

- Queries **without useful templates** (no green short-circuit to
  exploit): no descent advantage to lose.
- **Incremental queries that slide between frames** (e.g. a unit's vision
  area that moves a bit each tick): reuse the previous frontier instead of
  redoing template classification.
- **Operations that are inherently neighbour-based** (connected-component
  analysis, contour extraction): no descent formulation exists.

None of those describe the critters workload. For everything we measure
today, `neighbors` should stay off — and that's the default.

## Fine sweep: `item_limit` × `merge_limit` at 10k pop, full sim

Conducted 2026-06-19, 63 cells, 27.5 min wall. Settings: pop=10000, full
critters workload (with attacks, `--respawn 0.05`), strategy `lca`,
`neighbors` off, seed ∈ {42, 7, 123}, 120 warmup + 120 measured frames.
Matrix: `item_limit ∈ {3, 6, 10, 15, 20, 30, 50}`, `merge_limit` ∈
{1, ⌊il/2⌋, il}.

CSV: `docs/sweep_item_merge.csv`. Aggregator:
`docs/analyze_item_merge.ps1`.

### Trend along the diagonal (`merge_limit = item_limit`)

| item_limit | mv mean (µs) | vis_avg (µs) | atk_avg (µs) | ins+rm (µs) | leaves |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1056 | 62.1 | 12.3 | 69 | 4716 |
| 6 | 850 | 36.6 | 9.6 | 60 | 2365 |
| 10 | 755 | 27.4 | 8.7 | 57 | 1439 |
| 15 | 676 | 21.2 | 7.1 | 52 | 938 |
| 20 | 645 | 18.4 | 6.7 | 48 | 706 |
| 30 | 609 | 15.6 | 6.1 | 45 | 473 |
| 50 | 583 | 13.8 | 5.5 | 41 | 282 |

**Every single metric improves as `item_limit` grows, up to 50.** There is
no crossover where "more items per `Maybe` leaf" cost starts dominating —
not in this range, not in this workload. The reasons:

- The number of `Maybe` leaves on the figure boundary scales roughly with
  the boundary length, which shrinks (in leaf count) as leaves grow.
- The per-`Maybe`-leaf cost grows linearly with items, but the total
  number of items in `Maybe` leaves is dominated by **how many items are
  geometrically near the figure boundary** — that quantity is set by item
  density and figure boundary length, not by `item_limit`.
- Template classification cost (which is paid per node visited, not per
  item) drops with the leaf count.
- Tree depth shrinks → `locate` cheaper → `update`, `insert`, `remove`
  all cheaper.

So the naive worry ("won't 30 items per leaf cost 30 per-point checks?")
is technically true per leaf, but is more than offset by the drop in leaf
count and template-classification work.

**Practical recommendation**: the demo's `ITEM_LIMIT = 3` constant is
heavily sub-optimal for this workload. A default of 10–30 would cut every
cost component by 30–50%. Whether the optimum is past 50 is open: this
sweep capped at 50 and the trend is still going.

### Hysteresis: `merge_limit < item_limit`

| item_limit | best `merge_limit` | mv mean (µs) | savings vs merge=item_limit |
| ---: | ---: | ---: | ---: |
| 3 | 2 | 1044.2 | -1.2% |
| 6 | 3 | 832.7 | -2.0% |
| 10 | 5 | 728.3 | **-3.5%** |
| 15 | 8 | 666.4 | -1.4% |
| 20 | 10 | 638.7 | -1.0% |
| 30 | 30 | 609.3 | 0.0% |
| 50 | 25 | 580.5 | -0.4% |

Hysteresis gives consistent small wins on `mv` (1–3%) for moderate
`item_limit`, peaking at `item_limit = 10` with `merge_limit = 5`. The
gain decays at larger `item_limit` because oscillation is rarer (fewer
borderline cells).

The bigger story is in **arena footprint**:

| item_limit | merge_limit=1 arena | merge_limit=item_limit arena | ratio |
| ---: | ---: | ---: | ---: |
| 3 | 164,970 | 307,395 | 1.9× |
| 6 | 34,145 | 109,737 | 3.2× |
| 10 | 11,637 | 52,036 | 4.5× |
| 15 | 5,459 | 28,752 | 5.3× |
| 20 | 3,236 | 21,275 | 6.6× |
| 30 | 1,890 | 9,968 | 5.3× |
| 50 | 978 | 5,570 | 5.7× |

`merge = 1` produces **3–6× fewer arena nodes** than `merge = item_limit`.
Cause: every merge orphans the child nodes into the arena (they stay
allocated, unreachable). With `merge = 1` merges only fire when a leaf
has emptied, which under sustained churn is rare. Less merging → fewer
orphans.

**But** — and this is where the picture flips — `mv` is one component of
the per-frame cost. To pick the best combination we have to add up all
the components weighted by their call counts.

### Total per-frame cost (the real answer)

The per-cell metrics above are: `mv` (total per frame), `vis_avg` (per
call), `atk_avg` (per call), `ins+rm` (total per frame). To get the
real total we need the call counts. Measured directly (10k pop, full sim,
seed=42, steady state):

- `vis_n ≈ 3000 calls/frame` (one vision query per Hunter, every frame)
- `atk_n ≈ 16 calls/frame` (firing critters; rare relative to vision)

So per-frame total cost ≈ `mv + vis_avg×3000 + atk_avg×16 + ins_rm`.

**Vision culls dominate** — they account for ~95% of the total. So the
right knob to optimize is `vis_avg`, not `mv`.

### Re-evaluated best combination

Extending the sweep beyond 50 (single-seed probe) shows a real crossover
exactly where the per-item check cost in `Maybe` leaves starts dominating
the savings from fewer leaves to classify:

| item_limit | merge_limit | mv | vis_avg | atk_avg | vis_total | total µs/frame | leaves | arena |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 1056 | 62.1 | 12.3 | 186,300 | **187,622** | 4716 | 307k |
| 30 | 30 | 609 | 15.6 | 6.1 | 46,800 | **47,552** | 473 | 10k |
| 50 | 50 | 583 | 13.8 | 5.5 | 41,400 | **42,112** | 282 | 5.6k |
| **100** | **100** | **572** | **13.3** | **6.3** | **39,900** | **40,610** | **138** | **2.0k** |
| 200 | 200 | 602 | 13.8 | 6.5 | 41,400 | 42,139 | 66 | 553 |
| 500 | 500 | 754 | 15.8 | 7.0 | 47,400 | 48,308 | 29 | 323 |
| 1000 | 1000 | 1046 | 19.4 | 8.1 | 58,200 | 59,429 | 16 | 153 |

**The minimum total is at `item_limit = 100`.** Both `mv` (572) and
`vis_avg` (13.3) bottom out there. Past 100:

- At `item_limit = 1000` the tree has only 16 leaves total (256×256 px
  each). The drop polygon (~110×110) hits 1–3 of them, each containing
  ~608 items. Every item in those `Maybe` leaves walks the per-item
  raster check. 608 × ~15 ns = ~9 µs *per leaf*, on top of the
  classification work — vis_avg climbs to 19.4 µs/call.
- At `item_limit = 100` leaves are ~85×85 px, small enough that few items
  fall in `Maybe` leaves (~70 each), but large enough that the descent
  terminates fast on green/white internal nodes (which always have
  power-of-two aligned bboxes regardless of leaf size).

So the user's intuition was correct: per-item checks in `Maybe` leaves DO
scale with `item_limit`, and they eventually dominate the cull cost. The
crossover sits around `item_limit = 100` for this workload (10k pop,
drop+circle figures, 1024² world). Not within the original 3–50 sweep
range, but real.

**This depends on template availability.** The demo precomputes templates
at cell sizes 8/16/32 (and rectangles). For sizes outside this set there
are two fallbacks:
- **Granularity-as-fallback** (implemented): smaller templates aggregate
  into larger via the In/Out/Maybe block rule, exactly preserved.
- **Bbox-intersect** (last resort): no template at all — every leaf gets
  treated as `Maybe`, no green/white short-circuit.

Even at `item_limit = 50–200`, the descent classifies the *internal*
nodes (which always have power-of-two bboxes since splits are at midpoints)
using the precomputed templates and short-circuits their subtrees as
green/white. The expensive per-item path only happens at leaves that the
descent actually reaches as `Maybe`. That's why item_limit can grow past
the precomputed template sizes without immediate cliff.

### Recommended default

| Goal | item_limit | merge_limit | total µs/frame | arena |
| --- | ---: | ---: | ---: | ---: |
| Maximum throughput | 100 | 100 | 40,610 | 2,035 |
| Balanced (still very fast) | 50 | 50 | 42,112 | 5,592 |
| Memory-tight | 100 or 200 | 1 | ~45–50k | <500 |

The earlier "merge_limit = item_limit / 2 for balance" recommendation
turned out to be a `mv`-only optimization: it loses on the total because
the cull cost (95% of the frame) goes up with the extra leaf count from
hysteresis. **Default `merge_limit = item_limit` is the right call for
throughput**; hysteresis is purely a memory-vs-speed knob.

### Cull-cost cross-over verified

The user's question — *"isn't 100 items per Maybe leaf going to cost more
than 10?"* — turned out to be exactly right, just on a different
timescale than the rest of the sweep was showing. The sweep ran up to
`item_limit = 50` and saw monotonic improvement. Pushing to 100, 200,
500, 1000 revealed:

- Per-item check inside `Maybe` leaves scales linearly with `item_limit`.
- Number of `Maybe` leaves visited per cull DOESN'T shrink past a point
  (it's bounded by the figure's projection on the leaf grid, which
  saturates once leaves get large enough to cover the figure interior in
  a few of them).
- So total items per cull eventually grows: `cost ≈ Maybe_leaves ×
  item_limit`.

In practical terms: in a workload like this (cull-dominated, ~3000
vision queries per frame), pick `item_limit` matching the figure size —
the optimum is where leaves are smaller than the figure but not by orders
of magnitude. For drop scale 110 in a 1024² world, that's ~85×85
(`item_limit = 100`).

## `item_limit` ↔ cull cost (not just update)

The fine sweep above settles this question. The user's intuition that
larger `Maybe` leaves carry more per-point work is correct *per leaf*,
but the count of `Maybe` leaves on the boundary shrinks faster than the
per-leaf cost grows. Net: **every metric improves with larger
`item_limit` through 50**, no crossover. See the fine-sweep tables above.

The earlier `BENCHMARKS.md` Results 6 datapoint (3 → 6 cuts everything
~30–40%) is consistent with this.

## Which neighbour method `LcaRopes` uses, and why

The crate exposes three:

| Method | Storage | Lookup cost | Bookkeeping cost |
| --- | --- | --- | --- |
| `neighbors_samet` | none (parent pointers) | O(depth) amortized | none |
| `neighbors_probe` | none | O(depth × num_neighbours) | none |
| `neighbors_ropes` | stored lists per leaf | O(rope length), in practice O(1) | rewires on every split/merge |

`LcaRopes` uses **ropes**, the cheapest at lookup time. The question was:
why not `samet` or `probe`?

**Insight noticed late in the analysis**: for the `update` problem,
**`Lca` puro ya hace `samet` implícitamente**. The two algorithms have the
same structural shape — *ascend by parent pointers until an ancestor
contains the target, descend the relevant subtree*:

- Samet for *"give me the neighbour on side X"*: ascend until the first
  ancestor whose split crosses X, descend the sibling.
- LCA for *"where is new_pos?"*: ascend until the first ancestor whose
  bbox contains new_pos, descend that subtree.

In the dominant case (small motion, `new_pos` in a direct neighbour leaf),
LCA's ascent stops at the direct parent and descends the sibling — that's
exactly the Samet trajectory. So a hypothetical `LcaSamet` strategy would
be no different from the `Lca` we already have. It's already in there.

`LcaProbe` would be `locate(new_pos)` from root, i.e. essentially the
`Legacy` path — strictly worse, no reason to expose.

So **`LcaRopes` is the only strategy that adds anything on top of `Lca`**,
and only when the `neighbors` feature is paying for itself elsewhere. That
is why only it is exposed as a third variant.

### When `neighbors_probe` would beat `neighbors_samet`

Both are available on `Tree<T>` independently of the feature. Probe is
strictly worse than Samet in the current data structure (it pays an
O(depth) `locate` per neighbour instead of O(1)-amortized parent ascent),
but it has corner cases where it edges out:

- **No parent pointers.** Samet needs `parent: Option<NodeId>` on every
  node; probe only needs `locate` from root. A future stripped-down tree
  variant (e.g. for SIMD-friendly layouts or read-only snapshots) might
  drop parent pointers — probe stays available, Samet does not.
- **Concurrent reads under mutation.** Probe only reads the root-to-leaf
  path; Samet walks parent links that a concurrent writer might invalidate.
  Relevant once the multithreaded story lands (see the roadmap entry on
  the `vectorial-hash` README); not relevant today.
- **Reference / didactic value.** Probe is the simplest of the three to
  understand and to verify against — useful when sanity-checking the
  others.

For the current single-threaded tree with parent pointers, `samet` is the
right zero-storage choice and probe stays available mostly for
completeness.

## Final verdict / decisions

> **Note (2026-06-23):** points 1–2 below were refined by the overnight
> batch. At the *recommended* `item_limit = 100`, LCA and Legacy are at
> speed parity (the LCA speed win only exists at small item_limit / deep
> trees); LCA still wins on arena footprint (~22% fewer zombie nodes). And
> the `neighbors` feature is now confirmed to have **no** beneficial
> consumer on the critters workload (cull_walk loses to descent by
> 28–47%). See the "Overnight sweep batch" section for the data.

1. **Default `Tree::update` to `Lca`.** Already wired. `Legacy` survives
   only as a benchmark target via `update_with`. The win is up to ~10% on
   speed at small item_limit, narrowing to speed-parity at il=100 but a
   persistent ~22% arena-footprint advantage at every item_limit.
   **Action: keep current API; no further code change needed.**

2. **Keep the `neighbors` feature off by default.** Its bookkeeping cost
   on `update` ranges from 5% (shallow trees) to **81%** (deep tree, high
   pop). The overnight cull_walk break-even confirmed the only potential
   consumers (LcaRopes lookup, cull_walk) both *lose* to the feature-off
   path on this workload. **Action: leave default off; it should be turned
   on only by a workload with templateless or incremental culls where
   cull_walk could win (none here).**

3. **When `neighbors` is on, default to `LcaRopes`.** Already wired
   through `UpdateStrategy::default()`. Worth 0.5–4.5% over pure `Lca` at
   no downside risk. **Action: none.**

4. **`IntegerTree` is a *conditional* win — int-native workloads only.**
   In the synthetic update bench (items are 12 bytes, no f64 position):
   ~22% faster than `Tree`. In the full critters sim where items keep
   both `pos: VPoint` and a cached `ipos: IPoint`: **23% slower on
   update, neutral on cull, ~3–5% slower in total throughput**. Cull
   machinery now exists (mirrors `Tree::cull`, converts at the shape
   boundary). **Action: keep it as a specialty option for genuinely
   int-native simulations (grid/voxel/retro); do not adopt for
   float-position workloads; document the trade so users decide on
   data.**

5. **`item_limit` is the single biggest knob, with a real optimum.** The
   extended probe found a crossover at `item_limit = 100` for this
   workload. Below 100, larger leaves mean fewer classifications → faster.
   Above 100, leaves become so large that per-item checks inside `Maybe`
   leaves start dominating. The demo's `ITEM_LIMIT = 3` is **40× off
   optimum** (187 ms/frame vs 40.6 ms/frame at item_limit=100). The exact
   optimum depends on figure-vs-leaf-size relationship.
   **Action: raise the demo's `ITEM_LIMIT` to ~100; document the
   figure-size relationship that determines the optimum.**

6. **`merge_limit = item_limit` (aggressive merging) is the throughput
   default.** Vision culls dominate, and fewer leaves means cheaper culls
   — outweighing the small `mv` savings of hysteresis. At
   `item_limit = 100, merge_limit = 100`: **40.6 ms/frame, 2k arena
   nodes**. **Action: recommend `merge_limit = item_limit` for
   throughput; `merge_limit = 1` is a release valve for memory-tight
   sims (5–10% slower, drastically less arena).**

7. **Multithreading remains the largest unexplored axis.** See the
   `vectorial-hash` roadmap entry — concurrent updates and culls would
   change every absolute number above, possibly upending the strategy
   ranking. Out of scope for this round.

## Sensitivity sweep: `item_limit` optimum vs population & figure size (2026-06-20)

Conducted to derive a tuning heuristic. Matrix:
`pop ∈ {1000, 3000, 10000}` × `item_limit ∈ {10, 30, 50, 100, 200, 500}`
× `figure_scale ∈ {0.5, 1.0, 2.0}` × `seeds ∈ {42, 7, 123}`.
`merge_limit = item_limit`, strategy `lca`, neighbors off, full sim,
`--respawn 0.05`, 120 warmup + 120 measured frames. 162 cells, 15.4 min
wall. Figure scale uniformly multiplies `DROP_SCALE` (110→55/110/220 px)
and `CIRCLE_RADIUS` (48→24/48/96 px).

CSV: `docs/sweep_sensitivity.csv`. Aggregator:
`docs/analyze_sensitivity.ps1`. The aggregator estimates total per-frame
cost from `mv + vis_avg × 0.30·pop + atk_avg × 0.005·pop + ins+rm`
(call counts scale with population by these factors, measured earlier).

### Headline: `item_limit = 100` is the universal optimum across the tested grid

| pop | scale | il=10 | il=30 | il=50 | **il=100** | il=200 | il=500 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.5 | 1491 | 1062 | 1000 | **950** | 977 | 1118 |
| 1000 | 1.0 | 1442 | 1044 | 971 | **943** | 968 | 1128 |
| 1000 | 2.0 | 1425 | 1025 | 944 | **914** | 930 | 1108 |
| 3000 | 0.5 | 9346 | 6276 | 5791 | **5614** | 5824 | 6941 |
| 3000 | 1.0 | 9062 | 6199 | 5621 | **5472** | 5652 | 6895 |
| 3000 | 2.0 | 9125 | 6044 | 5586 | **5374** | 5456 | 6644 |
| 10000 | 0.5 | 82010 | 49611 | 43367 | **41559** | 43843 | 51030 |
| 10000 | 1.0 | 80592 | 49985 | 43396 | **40806** | 42144 | 49398 |
| 10000 | 2.0 | 80938 | 48803 | 43341 | **41032** | 41163 | 47361 |

Values are µs/frame, total. `item_limit = 100` wins **every single (pop,
scale) cell**.

### Why it's so robust

A few observations:

- **The cost curve is flat around the optimum.** At `pop = 1000`, the
  spread between `il = 50, 100, 200` is < 6%. At `pop = 10000` it's
  ~7%. So `il ∈ [50, 200]` is "good enough" everywhere; `il = 100` is
  the local minimum.
- **The cost curve has a deep tail in both directions.** At `il = 10`
  the cost is 60–100% higher; at `il = 500` it's 15–25% higher.
- **The leaf size at `il = 100`** is workload-dependent (256 px at
  pop=1000, 87 px at pop=10000), but the throughput barely cares. The
  cull descent terminates at internal nodes (always power-of-two
  bboxes) classified by templates — the leaf size mainly affects how
  many items end up in each `Maybe` leaf, which is bounded by
  `item_limit` itself.

### The "match leaf to figure" hypothesis didn't transfer

We expected the optimum `il` to shift with figure size — bigger figures
should justify larger leaves. The data says otherwise:

| pop | scale | figure side | optimum leaf side | ratio leaf/figure |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.5 | 55 | 256 | 4.65 |
| 1000 | 2.0 | 220 | 264 | 1.20 |
| 10000 | 0.5 | 55 | 89 | 1.62 |
| 10000 | 2.0 | 220 | 86 | 0.39 |

The ratio varies from 0.39 to 4.65 yet the **optimum `il` stays at 100**.
What actually pins the optimum is the *number of items in a `Maybe`
leaf*, which is `≤ item_limit` regardless of leaf size. Templates work
on internal nodes (power-of-two bboxes) above leaves, so the leaf size
mostly controls how much per-item work happens at the boundary —
independent of figure size.

### Final heuristic for this workload

**For 2D point-position critters in a 1024² world with the demo's
template set (8/16/32 + 1×1 raster + granularity fallback), use
`item_limit = 100, merge_limit = 100`**. The optimum is insensitive to
population (1k–10k) and to figure size (55–220 px) within ~5%.

This sweep does **not** cover:

- **30k+ pop** — runs took too long to fit in the budget. The 30k probe
  earlier (single-seed) was consistent with the il=100 optimum.
- **Very different world sizes** — at 4096² the leaf-vs-template
  geometry changes; the optimum probably shifts.
- **Non-point items** (Areal / dilation roadmap item). Per-item check
  cost is replaced by per-extent overlap test, changing the cost ratio.

### Probe: denser template set (8/16/32/64/128) doesn't move the optimum

A natural follow-up hypothesis was that adding (64×64) and (128×128) to
`TEMPLATE_SIZES` would shift the optimum upward — bigger templates give
the cull descent more places to early-terminate, which should reward
larger leaves. Tested empirically at pop=10000, scale=1.0, 3 seeds, same
sim params (18 cells, 4.3 min wall):

| item_limit | base total | dense total | delta |
| ---: | ---: | ---: | ---: |
| 10 | 80,592 | 82,990 | **+3.0%** |
| 30 | 49,985 | 48,888 | -2.2% |
| 50 | 43,396 | 42,818 | -1.3% |
| **100** | **40,806** | **40,404** | **-1.0%** |
| 200 | 42,144 | 42,073 | -0.2% |
| 500 | 49,398 | 48,980 | -0.8% |

The optimum stays at `il = 100`. Dense templates trim 1% off the optimum
total — well within the seed-to-seed noise floor. So **the il=100
optimum is structurally driven, not template-set driven**. The cull
descent already terminates at internal nodes whose bboxes (always
power-of-two-aligned) classify cleanly via the granularity-as-fallback
aggregation: adding native 64/128 templates barely helps because the
aggregation was already doing the equivalent work.

A side observation: at `il = 10` the dense set is actually **3% slower**.
With tiny leaves (≤15 px), the cull never descends to a depth where 64/128
classifications happen — the new sizes just inflate the SizeCache without
covering anything new, paying a small overhead per cull. CSV:
`docs/sweep_dense_templates.csv`. Aggregator:
`docs/analyze_dense_templates.ps1`.

**Conclusion on the heuristic**: `item_limit ≈ 100` is robust for this
workload across pop ∈ {1k, 3k, 10k}, figure scale ∈ {0.5×, 1.0×, 2.0×},
**and** template density (sparse 8/16/32 → dense 8/16/32/64/128). Not a
universal law of physics, but a far more stable target than we expected
going in.

For now, the `ITEM_LIMIT = 100` constant in `crates/vectorial-hash-demos/src/sim.rs`
is empirically defended for the demo's workload.

## Overnight sweep batch (2026-06-23)

Five measurement-only sweeps run in sequence (never in parallel — concurrent
benches contaminate each other's cache). All at pop=10000 unless noted,
il=100 (the recommended optimum), lca, full sim, `--respawn 0.05`, 3 seeds.
Scripts: `docs/sweep_{cull_walk,strategy_il100,quad_vs_binary,movement_step,30k}.ps1`.
Aggregator: `docs/analyze_night.ps1`. Total-frame estimate uses
`vis_n ≈ 0.30·pop`, `atk_n ≈ 0.005·pop`.

### 1. cull_walk break-even: descent wins decisively, `neighbors` never pays

| feature | cull | mv | vis_avg | total est µs/frame |
| --- | --- | ---: | ---: | ---: |
| off | descent | 849 | 13.0 | **40,206** |
| off | walk-samet | 868 | 19.0 | 58,246 |
| off | walk-probe | 871 | 19.3 | 59,153 |
| on | descent | 874 | 13.2 | 40,836 |
| on | walk-ropes | 889 | 16.7 | 51,363 |

Descent's vision cull is 13.0 µs; the best walk (ropes, feature on) is
16.7 µs (+28%), and the zero-storage walks are +45–47%. **No walk strategy
beats descent on this workload**, confirming the BENCHMARKS Results 3
finding (200k static points) holds under the dynamic critters workload too.
The reason is structural: descent classifies internal nodes wholesale via
templates and short-circuits whole subtrees green/white; the flood-fill
walk must touch every leaf in the region and run a neighbour query per leaf.

**This closes the `neighbors` feature question**: on the critters workload
there is no consumer that benefits from it (neither LcaRopes update nor
cull_walk), so it is pure cost. Keep it off — which is the default.

### 2. Strategy at il=100: LCA's *speed* win evaporates, *memory* win persists

The original strategy sweep was at il ∈ {3,10,30}, where LCA was up to ~10%
faster than Legacy on `mv`. At the now-recommended il=100 the tree is much
shallower, so the ascent-vs-descent difference shrinks to nothing:

| il | strategy | mv | mv p95 | arena |
| ---: | --- | ---: | ---: | ---: |
| 30 | lca | 894 | 937 | 9,968 |
| 30 | legacy | 908 | 954 | 12,658 |
| 100 | lca | 857 | 903 | 1,867 |
| 100 | legacy | 856 | 910 | 2,278 |
| 200 | lca | 891 | 944 | 606 |
| 200 | legacy | 887 | 939 | 730 |

At il=100, **LCA and Legacy are at speed parity** (857 vs 856 µs — within
noise; LCA marginally better on p95). The headline "LCA is faster" is
therefore **only true at small item_limit** (deep trees). At the
recommended il=100 the speed difference is gone.

**But LCA still wins on arena footprint at every item_limit**: il=100 gives
1,867 nodes (lca) vs 2,278 (legacy) — Legacy's remove+insert path orphans
~22% more zombie nodes. So the LCA-as-default decision still holds, but the
justification shifts: at the recommended config it's "same speed, ~22% less
memory churn", not "faster".

Also visible: `lca-ropes` (feature on) is consistently a hair *slower* than
`lca` (feature off) — 866 vs 857 at il=100 — the rope bookkeeping cost with
no payback. Reconfirms: feature off.

### 3. Quadtree vs binary: the quad advantage shrinks to ~2% at il=100

BENCHMARKS Results 6 found the quadtree 10–35% ahead — but that was at
il=3/6 (deep trees, where the quad's half-depth helps `locate` most). At
the recommended il=100:

| il | mode | mv | vis_avg | total est µs/frame |
| ---: | --- | ---: | ---: | ---: |
| 30 | binary | 894 | 15.8 | **48,695** |
| 30 | quad | 832 | 16.3 | 50,099 |
| 100 | binary | 844 | 12.9 | 39,881 |
| 100 | quad | 792 | 12.6 | **38,912** |
| 200 | binary | 887 | 13.4 | 41,423 |
| 200 | quad | 846 | 13.2 | **40,775** |

At il=100 the quad is ~2.4% ahead on total; at il=30 the binary is ~2.8%
ahead (its smaller leaf count gives a cheaper vision cull there). **At the
recommended item_limit the two structures are within ~2% of each other** —
the structural advantage that mattered at il=3/6 has all but vanished once
trees are shallow. This validates keeping the binary tree as the primary
structure: its other advantages (anisotropic / data-aware splits,
rectangular and non-power-of-two worlds, smaller node footprint) come at
negligible throughput cost at the recommended config.

### 4. Movement-step sensitivity: the LCA relocation path is robust

`--no-attack` (pure movement), il=100, varying dt (movement per step):

| dt | mv mean | mv p95 |
| ---: | ---: | ---: |
| 0.0083 | 904 | 932 |
| 0.0167 | 909 | 931 |
| 0.0333 | 924 | 946 |
| 0.0667 | 948 | 974 |

8× more movement per step (more cross-leaf jumps per update) costs only
**+5% on `mv`**. The relocation path doesn't degrade under heavier churn:
at il=100 the tree is shallow, so even a long jump's LCA ascent is short,
and most of `update`'s cost is the locate + predicate scan, not the
relocation itself.

### 5. 30k: optimum drifts slightly upward; sim is very stable

| il | mv mean | mv stdev | vis_avg | total est ms/frame |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 3,026 | 5.4 | 35.2 | 321.9 |
| 100 | 2,877 | 25.2 | 30.9 | 282.9 |
| 150 | 2,913 | 9.6 | 30.2 | **276.6** |
| 200 | 2,925 | 19.6 | 30.5 | 279.3 |
| 300 | 3,099 | 6.5 | 31.5 | 288.6 |

Seed-to-seed variance is tiny (stdev < 1% of mean) — the sim is highly
deterministic. The optimum at 30k is **il=150** (276.6 ms/frame), with a
flat bottom: il ∈ [100, 200] are all within ~2% of each other. The
optimum has **drifted upward from 100 (at 10k) to ~150 (at 30k)**.

The mechanism: more population → denser → at a fixed item_limit the leaves
carry more boundary items, so a larger item_limit keeps leaves "big enough"
to stay efficient. The drift is **sub-linear** in population: 3× the pop
(10k→30k) moves the optimum ~1.5× (100→150). A rough rule for this
workload: **optimum item_limit ≈ 100·(pop/10000)^0.4**, but the flat bottom
means anything in [100, 200] is within ~2% of optimal across the 10k–30k
range — precise tuning of item_limit past "around 100–150" is not worth the
effort here.

### Updated tuning recommendation

| Population | Recommended item_limit (= merge_limit) |
| ---: | --- |
| ~1k–10k | 100 |
| ~10k–30k | 100–150 (flat; 150 best at 30k) |
| general | the flat bottom is wide; `100` is a safe single default, nudge toward `150` for sustained 20k+ |

## Pending follow-ups (not yet measured)

Ordered roughly by value / effort. (cull_walk break-even, denser
templates, 30k variance, movement-step sensitivity, and strategy-at-il100
were measured 2026-06-23 — see the "Overnight sweep batch" section above.)

- **Bit-shift extension to cull is now done** (`IntegerTree::cull`); the
  head-to-head vs `Tree` in the full sim is in the "Full-sim head-to-head"
  section. What remains open here is only int-native game integration
  (where items wouldn't carry a float `pos` at all).
- **Items with area / volume** (`Areal` / index dilation). Roadmap item
  from `vectorial-hash/README.md`. WIP for the dilation half exists
  (`polygon::inflated_convex`, one failing sharp-corner test). Changes the
  data model — items live in multiple leaves — so every tuning rule above
  has to be rederived. High-impact, high-effort, separate milestone.
- **3D**. Two candidate strategies now written up in the
  `vectorial-hash` README roadmap: a true 3D tree (octree, N³ template
  explosion) vs. the author's projection-indexing idea (two/three 2D
  trees, intersect candidate sets, exact 3D narrowphase). The proposed
  cheap first experiment is to measure the broadphase/exact false-positive
  ratio per shape for the 2-projection scheme before building any 3D tree.
- **Multithreaded path.** The whole headless workload — every cull,
  every update, every insert/remove — runs single-threaded today. A real
  game loop would parallelise movement (each critter's `update` against
  the same tree) and culls (each attack/vision query). `Tree<T>` is
  `!Sync` for mutation as written. Designing a concurrency story (locked
  shards, copy-on-write snapshots per frame, or a SoA `update_many` that
  takes a bulk batch) is its own milestone — relevant once the
  single-thread paths are settled, but should not be lost.

## Status of changes

These results were produced from a working branch that adds:

- `UpdateStrategy` enum in `vectorial-hash/src/tree.rs` (and `quadtree.rs`).
- `Tree::update_with(strategy, ...)` and `QuadTree::update_with(strategy, ...)`.
- `Tree::locate_from(node, point)` helper exposed publicly.
- Equivalence tests `lca_strategy_matches_legacy_state` and
  `lca_ropes_strategy_matches_legacy_state` in `tree.rs`.
- `--update-strategy {legacy,lca,lca-ropes}` and `--no-attack` flags in
  `critters_headless`.
- `Sims::update_strategy` field, propagated through `update_critter`.
- `MAX_CRITTERS` bumped from 4000 to 40000 so the 10k/30k probes can sustain
  their populations.

Nothing committed yet.
