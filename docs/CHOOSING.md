# Choosing a structure

`vectorial-hash` ships six spatial indexes. They all answer the same two
queries — `cull` (everything inside a shape) and `knn` (k nearest neighbours) —
so picking one is about *your data and access pattern*, not features. This is
the one-glance guide; the quantitative backing is the decision map in
[`THREE_D.md`](THREE_D.md) and the parallelism crossover in
[`PARALLEL.md`](PARALLEL.md).

## Flowchart

```
                          ┌─────────────────────────────┐
                          │  2D or 3D?                  │
                          └──────────────┬──────────────┘
              2D ─────────────────────────┴───────────────────────── 3D
               │                                                       │
   ┌───────────┴────────────┐                          ┌──────────────┴───────────────┐
   │ integer coordinates    │                          │ do points relocate every     │
   │ (pixels / a grid)?     │                          │ frame (a live simulation)?   │
   └───────┬────────────┬───┘                          └───────┬───────────────┬──────┘
        yes│            │no                                 yes│               │no (mostly static,
           │            │                                      │               │     query-heavy)
   ┌───────┴──────┐  ┌──┴────────────────────┐        ┌────────┴─────────┐  ┌──┴───────────────────┐
   │ IntegerTree  │  │ uniform density and   │        │ keep a handle    │  │ density very uneven? │
   │ (i32, exact, │  │ you like simple 4-way │        │ per item →       │  └───┬──────────────┬───┘
   │  no float    │  │ recursion?            │        │ Tree3 +          │   yes│              │no
   │  fuzz)       │  └────┬─────────────┬────┘        │ insert_ref/      │  ┌───┴────────┐  ┌──┴─────────┐
   └──────────────┘    yes│             │no           │ update_ref       │  │ Tree3 or   │  │ MortonGrid3│
                    ┌─────┴─────┐  ┌────┴──────┐      │ (O(1) relocate,  │  │ Octree3    │  │ (flat grid:│
                    │ QuadTree  │  │ Tree (2D, │      │ ~5–10× the       │  │ (adaptive  │  │ cheapest   │
                    │ (4-way,   │  │ binary    │      │ predicate path)  │  │ leaf size) │  │ build+cull │
                    │ uniform)  │  │ split —   │      └──────────────────┘  └────────────┘  │ when dense │
                    └───────────┘  │ the       │                                            │ & uniform) │
                                   │ default)  │                                            └────────────┘
                                   └───────────┘
```

Two cross-cutting choices that sit on top of the above:

- **Points that live on a plane (e.g. a heightfield, units on terrain)?** A 3D
  query can be answered by a **2D `Tree` on xy + a z-slab reject + exact 3D
  narrowphase** (the "projection" path in the demo). Wins when the z-extent is
  thin relative to xy. See the decision map.
- **Many independent queries at once** (one cull per attacker, a batch of
  frustums)? Use `cull_many` / `cull_many_par` (feature `parallel`). The
  crossover — when threads pay — is in [`PARALLEL.md`](PARALLEL.md).

## Summary table

| Structure | Dim | Best when | Build | Cull | Relocate |
| --- | --- | --- | --- | --- | --- |
| **`Tree`** | 2D | general-purpose 2D, the default | adaptive | 14–16× brute | `update` / `update_ref` |
| **`QuadTree`** | 2D | uniform density, simple 4-way | adaptive | similar to `Tree` | same |
| **`IntegerTree`** | 2D | integer coords, no float fuzz | adaptive | similar | same |
| **`Tree3`** | 3D | dynamic 3D, the 3D default | adaptive | strong | **`update_ref` O(1)** |
| **`Octree3`** | 3D | 3D with locally varying density | adaptive (8-way) | strong | `update` / `update_ref` |
| **`MortonGrid3`** | 3D | dense + uniform, rebuilt per frame | **cheapest** | **cheapest** | rebuild (flat) |

## Rules of thumb

- **Start with `Tree` (2D) or `Tree3` (3D).** They adapt leaf size to local
  density and carry the full dynamic contract. Only move off the default for a
  concrete reason below.
- **Relocating everything every frame?** Hold the `ItemRef` that `insert_ref`
  returns and call `update_ref` — it skips the predicate's leaf scan and is the
  single biggest maintain win (the decision map flipped on it). One extra field
  per entity; see [`THREE_D.md`](THREE_D.md) § "The fix: Stable ItemRef".
- **Rebuilding from scratch each frame** (no persistent handles, uniform dense
  field)? `MortonGrid3` has the cheapest build and cull — there's nothing to
  maintain, you just refill it.
- **Integer world (tiles, pixels)?** `IntegerTree` avoids float boundary
  fuzz entirely.
- **`item_limit`** is the main tuning knob: smaller = deeper tree, fewer
  per-leaf tests, more nodes; larger = shallower, more brute per leaf. 8–16 is a
  good default; profile with the Criterion suite / regression gate
  (`benches/README.md`) if it matters.
- **Don't reach for threads first.** Reads parallelise (`cull_many_par`),
  writes don't — the lever for write-heavy loops is `update_ref` + the right
  structure, not rayon. See [`PARALLEL.md`](PARALLEL.md).
