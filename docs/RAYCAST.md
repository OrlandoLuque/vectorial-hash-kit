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

## Method choice (unchanged by the tuning)

- **Thin ray / first-hit** (LoS, picking, `radius` ≲ leaf): DDA — ~2× the capsule
  *and* 90–100 % coverage, with front-to-back order enabling early-exit.
- **Thick "all within r"**: capsule — exact, and its cost now tracks the answer
  size. The thin DDA only *looks* fast there because it misses 70–98 % of the
  hits (coverage column; its hit count plateaus as the corridor saturates).
- **Ropes < Samet < Probe** on query time (identical walk) — the difference is
  pure neighbour-finding: ropes O(1), Samet/Probe O(depth).

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

## Pitfall that nearly hid this

The very first measurement had the capsule at ~12 ms and "concluded" the DDA was
8–15× faster. The capsule's `Shape` had **no `classify_box`**, so the cull fell
back to the fat AABB. Adding the analytic classify dropped it to ~1.5 ms and
flipped the conclusion. Lesson: an analytic shape *must* implement `classify_box`
or the tree can't prune it.

## Open

- **Does maintaining ropes pay off overall?** The query-side win (≈30–45 % vs
  Samet) is half the ledger; ropes are rewired on every split/merge. Next:
  measure build/`update` with vs without the `neighbors` feature and combine.
- **Exact thick DDA** — widen the corridor with the distance test (visit the
  perpendicular neighbours within `r`). For thin rays the widen is ±1 and keeps
  the ordering; for fat rays its cost converges to the capsule (full coverage of
  a radius-`r` band is inherently O(len·r·density) — no free lunch), so there use
  the capsule. *(`raycast_first` above is the thin early-exit; the thick widen is
  the remaining piece.)*
- **SoA + SIMD narrowphase** — needs the SoA leaf-storage backlog item; the
  segment-aligned `|y'| ≤ r` test vectorises cleanly and is the remaining ceiling
  for the capsule's per-item cost.

*(Origin: the user's uniform-grid DDA ray-cast in `TestDraw`, adapted to the
variable-size tree.)*
