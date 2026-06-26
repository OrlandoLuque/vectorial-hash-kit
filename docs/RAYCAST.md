# Ray-casting: capsule cull vs DDA leaf-walk

Two ways to ask "what does this ray touch?", with opposite trade-offs. Both are
implemented; this note records the design and the measured comparison so we can
pick per use-case (and decide whether maintaining ropes pays off).

## The two methods

**Capsule** (`Tree3::raycast` / `Segment3` in 3D; `Capsule2` over `Tree::cull`
in the 2D prototype). Model the ray as a segment thickened by `radius` and run a
normal `cull`. Exact for "every item within `radius` of the ray", reuses all the
culling machinery — but it descends the segment's **fat axis-aligned bounding
box**, so it visits cells well off the ray (worst at the ends and on diagonals)
and returns everything unordered.

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

world 1024², N=50 000, item_limit 8, 128 rays × 60 reps, `--features neighbors`:

```
method      radius |   total ms     hits |   leaves    tested   coverage
capsule          2 |    12.31      10984 |        -         - 100% (ref)
DDA/samet        2 |     1.17      10631 |       52       308      96.8%
DDA/probe        2 |     1.48      10631 |       52       308      96.8%
DDA/ropes        2 |     0.82      10631 |       52       308      96.8%
capsule          8 |    10.83      44368 |        -         - 100% (ref)
DDA/*            8 |   1.1–1.8     32377 |       52       308      73.0%
capsule         24 |    14.47     137444 |        -         - 100% (ref)
DDA/*           24 |   1.1–1.8     39478 |       52       308      28.7%
capsule         64 |    16.57     392368 |        -         - 100% (ref)
DDA/*           64 |   1.1–1.8     39478 |       52       308      10.1%
```

Reading it:

- **DDA is ~8–15× faster.** The thin corridor (52 leaves, 308 narrowphase tests)
  is a fraction of the capsule's fat-AABB descent. For "what does this ray hit
  first" (line-of-sight, picking) it's the clear winner — and front-to-back order
  gives a natural early-exit the capsule can't.
- **Coverage falls as the ray thickens.** At `radius` 2 the centre corridor
  recovers 96.8% of the capsule's exact hits; by `radius` 64 only 10%. Items
  within a fat radius live in cells the centre line never enters, so the thin
  walk misses them (its hit count even plateaus — the corridor saturates). The
  capsule stays exact. **So: DDA for thin rays, capsule for thick "all within r".**
- **Ropes < Samet < Probe on query time** (≈0.8 / 1.2 / 1.5 ms), with an
  *identical* walk (same 52/308) — the difference is purely neighbour-finding:
  ropes O(1), Samet/Probe O(depth). On the query side ropes wins ~30–45%.

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
