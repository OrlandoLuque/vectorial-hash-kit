# Performance notes — siege demo

Measured findings + the FPS-optimisation review. **Applied** = a free win, already
in. **Reported** = a quality-for-FPS trade-off left for the user to decide.

## Boids separation: precomputed force table vs live maths (measured)

The separation force is a pure function of the relative offset `(Δx, Δz)`, so it's
tabulable — the project's template idea applied to steering. We built it (per
`(faction,kind)` grid, offset → `(Fx, Fz)`) behind `$SIEGE_BOID_TABLE=1` and
benchmarked it (`bench_boid_force_table`, ~2000 units, this machine):

| combo | per-pass |
| --- | --- |
| index (k-NN) + **maths** | **3.5 ms** |
| index (k-NN) + table | 4.8 ms |
| no-index O(N²) + **maths** (N=400) | **0.20 ms** |
| no-index O(N²) + table (N=400) | 0.42 ms |

**The table is slower.** Counterintuitive, but it's the *memory wall*: a few
divisions are cheap on a modern FPU (tens of cycles, pipelined / out-of-order),
while the lookup costs a memory load — and the big-unit grids (a dragon's
`sep_dist ≈ 48` → a ~75 KB grid) miss the cache. Compute got cheap; latency
didn't. **Verdict: default to the live maths** (kept). The table stays behind the
flag as a demonstration. (A precomputed table *does* win when the kernel is
expensive — many ops / transcendental — or the table is tiny and hot; this one is
neither.)

The two axes still compose, just not as hoped here: the **index** cuts the
*number* of force evals (N² → k·N — the big win), the **table** would cut the
*cost* of each (a loss here).

## FPS-optimisation review

_(filled in as the review runs)_
