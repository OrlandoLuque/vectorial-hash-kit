# Ray-casting: capsule cull vs DDA leaf-walk

Two ways to ask "what does this ray touch?", with opposite trade-offs. Both are
implemented; this note records the design and the measured comparison so we can
pick per use-case (and decide whether maintaining ropes pays off).

## The two methods

**Capsule** (`Tree3::raycast` / `Segment3` in 3D; `Capsule2` over `Tree::cull`
in the 2D prototype). Model the ray as a segment thickened by `radius` and run a
normal `cull`. Exact for "every item within `radius` of the ray", reuses all the
culling machinery. The key is an analytic **`classify_box`** (segment↔box
distance: `>r` → prune, farthest corner `<r` → take whole subtree, else
descend): the tree recursion then visits **only the radius-`r` band**, coarse
nodes in the interior and fine only at the boundary — it handles arbitrary
"thickness" for free, no manual neighbour chasing. *(Without `classify_box` the
cull falls back to the segment's fat AABB and is ~8× slower — see the pitfall
note below.)*

**DDA leaf-walk** (`Tree::raycast(origin, dir, max_t, radius, walk)`, 2D). The
variable-cell **Amanatides–Woo** traversal: walk only the leaves the centre ray
crosses, front-to-back, stepping to the neighbour across each exit edge. Per
leaf, a slab test gives the exit `t` and side; the neighbour is found by the
selected `WalkNeighbors` strategy (Samet / Probe / Ropes). Returns hits sorted
by distance along the ray, plus traversal stats.

The DDA is the variable-size generalisation of a uniform-grid DDA: the only
change is that the per-cell step `t` is recomputed from each leaf's *actual*
bounds instead of a constant `tDelta`, and "next cell" is the neighbour leaf
(found via the existing neighbour machinery — with ropes the step is O(1)).

## Measured (`examples/raycast_compare.rs`)

world 1024², N=50 000, item_limit 8, 128 rays × 60 reps, `--features neighbors`,
capsule **with** the analytic `classify_box`:

```
method      radius |  total ms     hits |  leaves   tested   coverage
capsule          2 |    1.54      10984 |       -        - 100% (ref)
DDA/samet        2 |    1.12      10631 |      52      308      96.8%
DDA/probe        2 |    1.43      10631 |      52      308      96.8%
DDA/ropes        2 |    0.71      10631 |      52      308      96.8%
capsule          8 |    2.38      44368 |       -        - 100% (ref)
DDA/ropes        8 |    1.09      32377 |      52      308      73.0%
capsule         24 |    3.72     137444 |       -        - 100% (ref)
DDA/ropes       24 |    1.09      39478 |      52      308      28.7%
capsule         64 |    5.13     392368 |       -        - 100% (ref)
DDA/ropes       64 |    1.07      39478 |      52      308      10.1%
```

Reading it:

- **The capsule scales with the answer, as it should.** With `classify_box` it
  visits only the band, so its cost tracks the hit count (1.5 ms @ 11 k hits →
  5 ms @ 392 k). It is **exact** at every radius.
- **The DDA's flat cost is a thin-ray win and a thick-ray mirage.** For a thin
  ray (`radius` 2) it's ~2× the capsule *and* 97 % coverage — a real choice
  (faster vs exact). For thick rays it only looks fast because it's **missing
  70–90 % of the hits** (coverage 29 %, 10 %; its hit count even plateaus as the
  corridor saturates). **So: DDA for thin rays / first-hit, capsule for thick
  "all within r".**
- **Ropes < Samet < Probe on query time** (≈0.7 / 1.1 / 1.4 ms), with an
  *identical* walk (same 52/308) — the difference is purely neighbour-finding:
  ropes O(1), Samet/Probe O(depth). On the query side ropes wins ~30–45%.

### Pitfall that nearly hid this

The first measurement had the capsule at **12 ms** and "concluded" the DDA was
8–15× faster. The capsule's `Shape` had **no `classify_box`**, so the cull fell
back to the segment's fat AABB and swept far more than the band. Adding the
analytic classify (segment↔box distance) dropped it to **1.5 ms** and flipped
the conclusion. Lesson: an analytic shape *must* implement `classify_box`, or
the tree can't prune it and the recursion can't do its job.

## Open: does maintaining ropes pay off *overall*?

The query-side win above is only half the ledger. Ropes are rewired on every
split/merge, so the verdict depends on the **query : mutation ratio** and how
churny the workload is. Next step: measure build/`update` cost with vs without
the `neighbors` feature and combine with the per-query numbers — for a
mostly-static, ray-heavy scene ropes likely win; under heavy churn with few
rays, Samet (zero storage, O(depth)) may be enough.

## Making the DDA exact for thick rays (if wanted)

Widen the corridor: at each step also visit the perpendicular neighbours whose
bbox comes within `radius` of the ray, and narrowphase their items. That keeps
the front-to-back early-exit while restoring coverage — a tighter alternative to
the capsule's fat AABB. Not yet built (the prototype is the thin corridor).

*(Origin: the user's uniform-grid DDA ray-cast in `TestDraw` — this is that idea
adapted to the variable-size tree. A hard surface-hit `raycast_first` with the
thick-corridor early-exit is the natural follow-up.)*
