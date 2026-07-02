# Benchmarks: template-driven culling — conclusions

> **The full study — methodology, environment, and complete reproducible result
> tables — lives in the research repo:
> [OrlandoLuque/vectorialHash → `research/BENCHMARKS.md`](https://github.com/OrlandoLuque/vectorialHash/blob/master/research/BENCHMARKS.md)**
> (kept with the paper so the investigation stays attributable). The raw sweep
> data + analysis scripts are in that repo's `research/benchmarks/`. This page
> keeps the headline findings for quick reference from the implementation.

Performance analysis of the vectorial-hash culling pipeline. All numbers are
reproducible with the `vh bench-*` commands (see the full study); rerun them after
any change to the cull path or the template bank.

## Headline findings

| # | What | Headline |
| --- | --- | --- |
| 1 | `vh bench`: single fixed template, binary-split tree vs quadtree | ~4× speedup from a single template; both trees within 10% on uniform data. |
| 2 | `vh bench-sizes`: per-cell-size selection (the paper's scheme) | 12–19× over no-template baseline; the precise method beats the old "snap" shortcut by ~5×; ~88% of index leaves share storage via content dedup. |
| 3 | `vh bench-walk`: descent vs neighbour-walk flood fill | Hierarchical descent dominates; ropes is the best neighbour source but still 0.7× of descent and costs ~56% extra on inserts. |
| 4 | `vh bench-fallback`: granularity-as-fallback aggregation | The aggregated fallback is **exact**, costs 0.59 MB vs 1.70 MB of full precomputation, ~3× the no-template baseline. A memory/precompute knob. |
| 5 | `vh bench-scale`: figure↔grid scale equivalence | One canonical set serves many query scales: 25× less memory, 10× faster generation; cull cost equals direct at low factors, ~2.5× at factor 8. |
| 6 | `critters_headless`: full dynamic workload (updates + culls + churn) | Quadtree ahead 10–35% even on dynamic ops (depth halves `locate`); hysteresis helps the binary tree; `item_limit` is the dominant knob; deterministic cross-structure runs with zero cull mismatches. |

The takeaway for **using** the library: a single template already buys ~4×;
per-cell-size selection (the paper's scheme) is the big win at 12–19×; on uniform
data the quadtree edges the binary tree, and `item_limit` is the knob that moves
the needle most. Structure-choice guidance is in [CHOOSING.md](CHOOSING.md); the
*why* and the numbers are in the full study linked above.
