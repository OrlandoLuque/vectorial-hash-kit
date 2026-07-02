# Update strategy comparison — conclusions

> **The full study — the 135-cell formal sweep, methodology, per-cell tables, and
> the IntegerTree head-to-head — lives in the research repo:
> [OrlandoLuque/vectorialHash → `research/UPDATE_STRATEGIES.md`](https://github.com/OrlandoLuque/vectorialHash/blob/master/research/UPDATE_STRATEGIES.md)**
> (kept with the paper so the investigation stays attributable). The raw sweep
> data + analysis scripts are in that repo's `research/benchmarks/`. This page
> keeps the verdicts that set the library's API defaults.

When a `Tree::update` mutator pushes an item out of its leaf, it must be
relocated. Three strategies live behind `UpdateStrategy`:

| Strategy | Path |
| --- | --- |
| `Legacy` | `remove` from the old leaf, `try_merge_up`, then `insert` descending from the root. |
| `Lca` | Ascend by parent pointers to the first ancestor whose bbox contains the new position; descend only within that subtree. **Default.** |
| `LcaRopes` | First scan the old leaf's rope neighbours (feature `neighbors`); move directly if one contains the new position, else fall through to `Lca`. |

## Verdicts (what set the defaults)

- **`Lca` is the default** (since 2026-06): up to **~10% faster** than `Legacy` on
  update-heavy workloads, and consistently **~30% lower arena footprint**.
- **`LcaRopes`** (with the `neighbors` feature) adds a further **0.5–4.5%** on top —
  worth it when updates dominate; the rope bookkeeping is the cost.
- **`IntegerTree<T>` (bit-shift, power-of-two worlds)** — a **conditional** win:
  **~22% faster** on `move+update` *only* when items already store positions as
  integers. If items need both float and integer positions, the duplicated state
  reverses it (~23% slower on update, ~3–5% slower overall). Use it only when your
  data is natively integer.
- **Tuning:** `item_limit` is the dominant knob; on uniform data the quadtree edges
  the binary tree; the `cull_walk` neighbour-walk only breaks even in narrow
  regimes (descent usually wins — see [BENCHMARKS.md](BENCHMARKS.md)).

Structure-choice guidance is in [CHOOSING.md](CHOOSING.md); the per-cell sweep and
the derivation are in the full study linked above.
