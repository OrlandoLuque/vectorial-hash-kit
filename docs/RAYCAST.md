# Ray-casting: capsule cull vs DDA leaf-walk

Two ways to ask "what does this ray touch?", with opposite trade-offs, plus the
distance-math optimisations applied to both. The harness is
`examples/raycast_compare.rs`.

## The two methods

**Capsule** (`Tree3::raycast` / `Segment3` in 3D; `Capsule2` over `Tree::cull`
in the 2D harness). Model the ray as a segment thickened by `radius` and run a
normal `cull`. Exact for "every item within `radius` of the ray". The key is an
analytic **`classify_box`** (a `Shape` hook): the tree recursion then visits only
the radius-`r` band — coarse nodes in the interior (`In` → take the whole
subtree), fine only at the boundary — handling "thickness" for free, no manual
neighbour chasing. *(Without `classify_box` the cull falls back to the segment's
fat AABB and is several× slower — the pitfall below.)*

**DDA leaf-walk** (`Tree::raycast(origin, dir, max_t, radius, walk)`, 2D — the
user's uniform-grid `TestDraw` idea adapted to variable cells). The variable-cell
**Amanatides–Woo** traversal: walk only the leaves the centre ray crosses,
front-to-back, stepping to the neighbour across each exit edge via the selected
`WalkNeighbors` strategy (Samet / Probe / Ropes). Returns hits sorted by `t` plus
traversal stats. The only change from a uniform grid is that the per-cell exit
`t` is recomputed from each leaf's actual bounds instead of a constant `tDelta`.

## Optimisations applied (and what they bought)

Distance/geometry is the hot path; both methods got the stable-Rust tricks:

**Capsule** (`cap-opt` in the harness, vs the `cap-naive` "before"):
- **Precomputed segment invariants** (`Capsule2::new`): `ab`, `len²`, `1/len²`,
  unit direction & normal, `r²`, bbox — computed once per ray, so the inner
  loops multiply instead of divide.
- **Perpendicular `contains_point`**: branch on the projection (`dot ≤ 0` → cap
  a, `dot ≥ len²` → cap b, else `|ap|² − dot²·inv_len²`) — no division, no
  projected-point construction.
- **Conservative slab `classify_box`**: project the node box onto the segment-
  aligned (u, n) axes; the whole capsule lives in `[−r, len+r] × [−r, r]`, so a
  non-overlapping interval ⟹ `Out` — a handful of compares, **no exact box
  distance**. Plus a centre-based conservative `In` (`spine_dist(centre) ≤
  r − half_diag`) instead of four corner distances. Exact work happens only
  per-item at the leaves.

**DDA** (`Tree::raycast`):
- **Reciprocal direction** precomputed (`1/dir`) so the per-cell slab test is a
  multiply, not a divide (the Amanatides–Woo trick).
- **Projection-based narrowphase**: the along-ray projection `proj` gives both
  the distance (`|ap|² − proj²`) and the hit `t` — no separate distance call, no
  division.

## Measured — exhaustive (`--features neighbors`)

**Methodology (contamination-resistant).** This machine runs other processes, so
the harness (a) times every method **interleaved** — each round times all methods
once, rotating the start order, so a background-load spike in any round is shared
across all of them rather than penalising whichever was being measured at the
time; (b) reports the **min** of `ROUNDS=40` rounds (noise only adds time, so the
minimum is the least-disturbed estimate); and (c) prints a **`noise`** column =
worst `median/min` ratio in that row (≈1.0 clean; ≫1 ⟹ the machine was busy →
re-run). The **speedup** is the stable headline — because naive and opt are
measured interleaved, any load hits both equally and the ratio holds; absolute ms
drift with machine load, the ratio does not. Two back-to-back runs agreed on the
speedups to ±0.1×.

world 1024², 64 rays (full-world length). Times = ms per whole batch.
`cap-naive`/`cap-opt` = same exact capsule before/after the tuning; `dda best` =
fastest neighbour strategy (ropes); `coverage` = fraction of cap-opt's exact hits
the thin DDA recovers.

```
 N=10000  il=8        N=50000  il=8        N=200000  il=8
r  naive  opt   ×  cov   naive  opt   ×  cov   naive   opt   ×  cov
2  0.252 0.170 1.5 99%   0.738 0.508 1.5 97%   1.945  1.381 1.4 91%
8  0.393 0.248 1.6 92%   1.140 0.771 1.5 74%   3.106  2.193 1.4 45%
32 0.647 0.439 1.5 48%   1.825 1.311 1.4 22%   4.831  3.834 1.3 11%
128 1.034 0.788 1.3 11%  3.328 2.941 1.1  5%  11.150 11.014 1.0  2%
```

(`il=16` is similar: capsule speedups 1.1–1.6×; coverage a few points higher
since fewer, fatter leaves.) DDA walk per ray: ~23/52/103 leaves and ~134/301/597
items tested for N=10k/50k/200k — flat in radius (the corridor), identical across
the three strategies.

**Capsule optimisation: a steady 1.3–1.7×** across density and leaf size,
shrinking toward 1.0× only at `radius` 128 where the answer is huge and the cost
is dominated by *building the result vector* (38–52 k hits), not distance math.

**DDA optimisation: ~10 % at scale, noise-limited when small.** Best DDA time at
N=200k went 0.997 → 0.856 ms (r=2, ≈14 %); at N=10k the deltas sit inside
run-to-run noise. The reason: the DDA is **walk-bound** — its cost is the
neighbour step (locate / Samet / Probe per leaf), and it narrowphases only a few
hundred items per ray, so there's little distance math to shave. The capsule, by
contrast, does a `classify_box` per node *and* a per-item test over a far larger
set, so the distance tricks pay much more there.

So the optimisations confirm the division of labour: **distance-math wins land on
the capsule (classify + narrowphase bound); the DDA is bound by the walk.**

## Method choice — the scorecard

The two methods came out **complementary, not rivals**: the DDA owns thin /
first-hit, the capsule owns thick / all-hits.

| Query | Use | Why |
| --- | --- | --- |
| **Thin ray, first hit** (LoS, picking) | **`Tree::raycast_first`** (DDA) | early-exit → **7–96×** the full corridor |
| Thin ray, all hits | `Tree::raycast` (thin DDA) or capsule | close; DDA's corridor is tight |
| **Thick ray, all within `r`** | **`cull(&Capsule)`** (descent) | exact; cost tracks the answer; recursion handles width |
| Thick ray, first hit | `cull(&Capsule)` + min projection `t` | a dedicated flood would lose to the descent |

- The thin DDA only *looks* fast on thick rays because it misses 70–98 % of the
  hits (coverage; its hit count plateaus as the corridor saturates).
- **Ropes < Samet < Probe** on query time (identical walk) — pure
  neighbour-finding cost: ropes O(1), Samet/Probe O(depth) — but ropes cost
  +45–53 % to maintain, so Samet is the default (see the ledger below).

## First-hit (early-exit) — `Tree::raycast_first`

The line-of-sight / picking query: the *nearest* item along the ray. Same DDA
walk, but it keeps the best hit and **stops as soon as the next leaf starts
beyond it** (entry `t` − `radius` slack > best `t`) — so it touches a handful of
leaves instead of the whole corridor. Exact for thin rays; for thick rays it's
the nearest hit *in the corridor* (the usual coverage caveat). Verified to return
the same nearest hit as `raycast`.

The early-exit payoff (N=200k, item_limit 8, full DDA vs first-hit, interleaved):

```
radius   full ms   first ms   speedup
2        1.341     0.014      95.8×
8        1.456     0.019      76.2×
32       1.459     0.053      27.5×
128      1.456     0.213       6.8×
```

For a thin ray in a dense scene the first hit is found almost immediately — two
orders of magnitude faster than gathering the whole corridor. (The first-hit
times are so small, ~14 µs, that their `noise` ratio is high — near timer
resolution; the min still holds across runs.) This is the regime where the DDA
decisively beats the capsule (which has no ordering and must gather everything).

## Exact thick band — descend, don't flood

The thin DDA's coverage gap is fixable by widening the corridor with the distance
test. That widened, exact band already exists in the library as **`cull_walk`** —
a neighbour flood seeded at the ray origin, pruned by the capsule's `classify_box`
(any `WalkNeighbors` strategy). It gives the same exact result as the descent
`cull`. Measured (N=200k, item_limit 8, interleaved):

```
radius   descent(cull)   flood-walk   thin DDA   walk coverage
2        1.306           2.170        1.378      100.0%
8        2.057           3.862        1.505      100.0%
32       3.696          10.161        1.546      100.0%
128     10.413          40.406        1.625      100.0%
```

The flood-walk is **exact (100%)** but **2–4× slower than the descent**, widening
with radius — it re-finds neighbours (O(depth) per step for Samet/Probe) and
carries visited-set bookkeeping, where the descent prunes hierarchically in one
pass. So the rule stands, now measured: **for "all within `r`", descend (the
capsule `cull`); the neighbour flood is exact but not the way to do it.** The thin
DDA stays fastest (flat ~1.5 ms) but only at partial coverage. The one place the
neighbour walk wins outright is ordered first-hit (`raycast_first`), where the
early-exit beats both.

## Do ropes pay off? (the maintenance ledger)

Stored ropes make the neighbour walk ~30–45 % faster per query — but they're
rewired on every split/merge. `examples/ropes_balance.rs` measures that upkeep,
build + churn **with vs without** the `neighbors` feature (N=50k, item_limit 8,
relocating every point per frame via `update_ref`):

```
                without ropes    with ropes    overhead
build (50k inserts)   13.6 ms       19.6 ms      +45 %
update / frame         3.70 ms       5.68 ms      +53 %
```

So ropes add **~45 % to build and ~53 % to per-frame relocation** — a real cost.
The break-even: the maintenance adds ≈ 2 ms to a frame that relocates 50k points,
while a neighbour-heavy query saves only single-digit µs each, so you need on the
order of **hundreds of neighbour-queries per frame** to offset the churn. The
verdict:

- **Static / low-churn, very query-heavy** (build the ropes once, then fire many
  rays / floods): ropes win — the upkeep is amortised.
- **Churn-heavy with few queries** (the common game-loop case): **Samet** — zero
  storage, O(depth), and you skip the 45–53 % maintenance tax entirely.

This is exactly why the library keeps `neighbors` **off by default**: the
zero-storage finders (`neighbors_samet` / `neighbors_probe`) are the right
default, and ropes are an opt-in for the query-bound, low-churn regime.

## Narrowphase ceiling — SoA + SIMD (`narrowphase_simd`)

How much could a **SoA leaf + branchless kernel** buy the per-item test? The tree
runs the AoS branch-on-projection distance today; the SoA form
(`t = clamp(dot·inv_len², 0, 1)`; `|ap − t·ab|²`) over `xs[]`/`ys[]` has no
data-dependent branches, so LLVM auto-vectorises it. Microbench, 4 M points, one
segment:

```
target            AoS ms   SoA ms   speedup
default (SSE2)     16.2      2.6      6.3×
target-cpu=native   1.57     1.22     1.3×
```

The microbench win is **real but build-dependent**: on the default target the
branchy AoS doesn't vectorise, so SoA gives ~6×; with `-C target-cpu=native`
(AVX2) the compiler vectorises *even the AoS branches*, so AoS drops ~10× and the
SoA edge shrinks to ~1.3×.

### Implemented + measured end-to-end (the reality check)

It's wired in: `Shape::wants_batch()` + `Shape::contains_batch()` (opt-in,
default off → zero change for every other shape), and `Capsule` overrides them
with the branchless kernel; the cull's leaf narrowphase runs it over a
thread-local SoA scratch. Measured on the **full** capsule cull (N=200k, vs the
same capsule with the per-point path):

```
              default target        target-cpu=native
item_limit  r=8   r=32  r=128     r=8   r=32  r=128
16          1.00  1.02  1.03      1.00  1.02  1.03
64          1.05  1.01  0.99      1.12  1.08  1.05
256         1.13  1.01  0.99      1.26  1.11  1.08
```

So end-to-end it's **~1.0–1.26×**, *far* below the 6.3× microbench. Two reasons:
the narrowphase is only **part** of the cull — the `classify_box` descent is the
other half (**Amdahl**); and the SoA materialisation (bbox pre-filter + copy into
scratch) is scalar overhead. It only helps with **big leaves** (high
`item_limit`, where per-item work dominates) and **thin radii** (where the
narrowphase out-weighs the descent), and even then modestly. Conclusion: the
batch kernel is kept (it's free when unused and a small win on big leaves), but
the **dominant lever is the descent, not the narrowphase** — and the cheapest
real win remains `-C target-cpu=native`. A *permanent* SoA leaf store (vs the
scratch) would only save the copy, which isn't the bottleneck, so it stays
deferred.

## 3D — `MortonGrid3` DDA

The ray-cast is in 3D too, on the uniform Z-order grid: `MortonGrid3::raycast`
(all hits) + `raycast_first` (nearest, early-exit). On a **uniform** grid the DDA
is the textbook **3D Amanatides–Woo** — `tMax`/`tDelta` are constant, so each
voxel step is one add + compare, with **no neighbour-finding** (the user's
`TestDraw` 3D, exactly). The ray is clipped to the world AABB first.

Measured (N=200k in 1024³, 64 cells/axis ≈ 16-unit cells, 64 rays × min-of-40,
interleaved; two back-to-back runs agreed):

```
radius   capsule ms   dda ms   first ms   cells  tested  coverage
4         11.3        0.094    0.028       34     26       90.2%
16        15.9        0.127    0.014       34     26       43.5%
64        35.4        0.110    0.022       34     26        2.7%
256      207.3        0.094    0.043       34     26        0.1%
```

- **The DDA's lead is huge in 3D — ~100–2000×** (vs 8–15× in 2D). The reason is
  structural: `MortonGrid3` is **flat** (no hierarchy), so the capsule
  `cull(&Segment3)` scans the whole bounding-box of cells — which grows as `r³`
  (207 ms at r=256!) — while the DDA walks only the **fixed thin corridor** (34
  cells, flat in radius). `raycast_first` early-exits to ~tens of µs.
- **But the coverage gap is also worse in 3D**: the radius-`r` band's volume
  grows as `r³` while the corridor stays fixed, so the thin DDA's coverage falls
  off a cliff (90 % → 0.1 %). For a thick 3D "all within `r`", the thin DDA is
  useless.
- **So for a thick 3D band, the flat Morton grid is the wrong structure** — its
  capsule cull has no hierarchy to prune with. The hierarchical **`Tree3` /
  `Octree3`** are the right home: `Tree3::raycast` (the `Segment3` capsule over
  the binary tree) now prunes tightly too, because `Segment3` gained an analytic
  `classify_aabb` (conservative sphere test: box bounding-sphere vs the spine).

### Variable-cell 3D DDA — `Tree3::raycast_dda`

The adaptive-cell version is in too: `Tree3::raycast_dda` / `raycast_dda_first`.
`Tree3` has no 3D neighbour links, so the step is **Probe-style** — the slab test
on the current leaf gives the exit, and `locate` finds the next leaf just across
the face (the 3D analogue of the 2D `WalkNeighbors::Probe`). Measured (N=200k,
item_limit 8):

```
radius   capsule ms   dda ms   first ms   leaves  tested  coverage
4         0.62        0.21     0.063       18      104      96.9%
16        1.13        0.31     0.039       18      104      74.4%
64        6.92        0.40     0.063       18      104      10.7%
256      86.6         0.41     0.130       18      104       0.6%
```

Two points stand out. **(1)** The variable-cell DDA works — 96.9 % coverage at a
thin radius, ~18 adaptive leaves per ray (vs Morton's 34 uniform cells). **(2)
The hierarchical `Tree3` capsule is far better than the flat Morton one for thick
bands** — 86.6 ms vs Morton's 207 ms at r=256 — because the tree *prunes*
(`classify_aabb` + the `In`/`Out` descent) where the flat grid just *scans* the
`r³` bbox. So the 3D scorecard: thin ray / first-hit → either DDA (fast); thick
"all within `r`" → the **`Tree3` capsule**, not the Morton one.

The DDA step is now **nudge-free** (2026-07-18). The old Probe step nudged a hair
across the exit face and re-`locate`d from the root — an epsilon that could skip a
thin neighbour sliver on pathological cells (never a false positive — always a
subset). The step now finds the exact face-neighbour by **ascending to the
least-common-ancestor** whose sibling lies across the exit face, then descending to
the leaf touching the exit point — **Samet's rope-free neighbour**, no epsilon. On
`Tree3` the split axis is read back from the two child boxes (the upper child starts
higher on exactly the split axis); on `Octree3` the exit-axis **octant bit is
flipped** and the other two shared. Beyond the existing gates (DDA ⊆ exact capsule,
sorted, first == nearest) a **completeness** test now densely samples the ray,
`locate`s each interior point, and asserts every crossed leaf is one the walk
visited — a direct check the nudge couldn't guarantee (it could under-collect). Both
`Tree3` and `Octree3` carry the nudge-free walk; `raycast_start_leaf` still uses a
single `locate` for the *entry* leaf only (not a per-step nudge).

The **8-way `Octree3`** carries the same surface — `Octree3::raycast` (thick
capsule), `raycast_dda`, `raycast_dda_first`. So both adaptive 3D trees and the flat
Morton grid answer thin-ray / first-hit; the hierarchical capsule still wins the
thick band.

## Concepts (glossary)

**Descent vs. `cull_walk` — two ways to traverse the same tree.** Both return the
same items; they differ in *direction*.
- **Descent** (`cull`) is **top-down**: start at the root, recurse into children,
  classify each node's box (`In`/`Out`/`Maybe`), **prune** `Out` subtrees, take
  `In` subtrees *whole* (no per-item test), descend only `Maybe`. One pass over
  parent→child pointers. The win is hierarchical pruning: a fat band's interior
  is covered by a few **coarse `In` nodes**, and only the boundary needs fine
  nodes — work ∝ the band's *surface*.
- **`cull_walk`** is a **lateral flood**: start at the *leaf* containing a seed
  point, then spread to neighbour leaves (find-neighbour by Samet/Probe/Ropes),
  flood-filling outward, stopping at `Out` leaves, with a visited-set to avoid
  repeats. It always works at *leaf* granularity (no coarse `In` shortcut) and
  pays the neighbour-finding cost per leaf (O(depth) for Samet/Probe). That's why
  the descent beats it for a connected band — `cull_walk` is for when you only
  have a seed and want a local region, not a root-down query.

**Templates vs. analytic `classify_box` (why the capsule uses the latter).** The
library's *templates* precompute a cell's In/Out/Maybe for a shape, indexed by
cell size / angle / offset — they amortise across **many repeated queries of the
same shape** (the demo's attack figures, which have no cheap analytic test). A
capsule is the wrong fit: it's thin and **oriented**, so it would need a template
*per angle*, a slightly-wrong angle misclassifies cells along the *whole* ray
length, and "every hundredth of a degree" is a memory blow-up — while rays are
usually one-shot at arbitrary angles, so the bank rarely hits. For a shape with a
cheap exact test (circle, capsule) the right tool is the analytic `classify_box`
(segment↔box distance) — exact pruning, no template, no per-query rasterisation.
Templates are for the non-analytic, repeated-same-shape case.

**AoS vs. SoA — memory layout.** *Array of Structs* stores records interleaved:
`Vec<Point>` ⇒ `[x0,y0, x1,y1, …]`. *Struct of Arrays* stores each field
contiguously: `xs=[x0,x1,…]`, `ys=[y0,y1,…]`. SoA puts all the `x`s next to each
other.

**SIMD — Single Instruction, Multiple Data.** One CPU instruction that operates
on several values at once, using wide registers: SSE2 (128-bit) = 2×f64 or 4×f32
per op; **AVX/AVX2** (256-bit) = 4×f64 / 8×f32; AVX-512 = 8×f64. So a SIMD
multiply does 4 lanes in roughly the time of one scalar multiply.

**Vendor / architecture portability.** SSE2 / AVX / AVX2 are part of the **x86-64
ISA, shared by Intel *and* AMD** — both run them (AVX-512 is patchier: Intel
on/off by line, AMD since Zen 4, so AVX2 is the safe "modern x86-64" target).
Other architectures have *their own* SIMD — **ARM = NEON** (128-bit) + SVE/SVE2
(Apple Silicon, phones, AWS Graviton), RISC-V = the Vector extension. The
*instructions* differ, but the **technique is portable**: write the branchless
SoA loop once and the compiler **auto-vectorises to whatever the target has** —
SSE/AVX on x86, NEON/SVE on ARM. Only *hand-written* intrinsics are
architecture-specific; the auto-vectorised path (what we rely on) just needs a
rebuild per target, and `target-cpu=native` adapts to whatever chip you build on.

**Why SoA + branchless ⇒ SIMD.** To do `x0..x3` in one SIMD load they must be
contiguous — that's SoA (AoS would need a slow gather). And the loop must be
**branchless**: a data-dependent `if` (like the cap branches in the
point-segment distance) makes each lane take a different path, so the compiler
can't run them in lockstep. The branchless clamped form (`t = clamp(dot·inv,0,1)`,
no `if`; the hit test accumulated as a 0/1 mask) is what LLVM
**auto-vectorises** — it turns the scalar loop into SIMD by itself when these
conditions hold. That's the AoS-branch (16 ms) vs SoA-branchless (2.6 ms) gap.

**`-C target-cpu=native`.** By default `cargo --release` builds for a *generic*
x86-64 CPU, so the binary runs anywhere — but that baseline only guarantees SSE2,
**not** AVX (older CPUs lack it). `RUSTFLAGS="-C target-cpu=native"` says "compile
for *this* CPU", unlocking AVX2/AVX-512 and more aggressive auto-vectorisation —
often a free 2–10× on number-crunching loops (it's what shrank the AoS time
10×). The catch: the binary may crash on older CPUs (illegal instruction), so use
it for **locally-built / self-hosted** hot paths and benchmarks, **not** for
distributed releases or the wasm build — those keep a portable baseline (or do
runtime feature detection). *(For the future how-to guides: this is the cheapest
perf knob — worth a section, next to threading.)*

## Pitfall that nearly hid this

The very first measurement had the capsule at ~12 ms and "concluded" the DDA was
8–15× faster. The capsule's `Shape` had **no `classify_box`**, so the cull fell
back to the fat AABB. Adding the analytic classify dropped it to ~1.5 ms and
flipped the conclusion. Lesson: an analytic shape *must* implement `classify_box`
or the tree can't prune it.

## Resolved — the two "niche" follow-ups

- **Thick ordered first-hit** — *use `cull(&Capsule)` + pick the minimum
  projection `t`.* A dedicated front-to-back widened flood (±r perpendicular, with
  early-exit) is the same *flood* family that already lost 2–4× to the descent
  above, so it would lose here too; the early-exit only helps in the thin case,
  which `raycast_first` already covers. So the completion is the right tool, not a
  new walk — and the library now ships the **2D `Capsule` `Shape`** (with the
  optimised `classify_box`) so the pattern works:
  ```rust
  let cap = Capsule::new(a, b, r);
  let near = tree.cull(&cap).into_iter()
      .map(|it| { let p = it.position(); /* project onto the ray */ (t_of(p), it) })
      .min_by(|x, y| x.0.total_cmp(&y.0));
  ```
- **SoA narrowphase** — *done* as an opt-in batch kernel (`wants_batch` /
  `contains_batch`, used by `Capsule`); measured **~1.0–1.26× end-to-end**
  (above), the descent dominates. The *free* win for any hot consumer remains
  `-C target-cpu=native`. A permanent SoA leaf store stays deferred — it would
  only save the materialisation copy, which isn't the bottleneck.

*(Origin: the user's uniform-grid DDA ray-cast in `TestDraw`, adapted to the
variable-size tree.)*
