# Shipped calibrations

`Thresholds`' defaults are one machine's measurements. `examples/calibrate` measures another and
writes a file here; `VH_CALIBRATION=path` makes `AdaptiveIndex::new` read it.

| file | machine | notes |
| --- | --- | --- |
| *(defaults, in code)* | the repo's main box | what you get with no `VH_CALIBRATION` |
| `i7-laptop-2026-09-02.txt` | Intel i7 laptop | measured while the main machine was out of service |

**There is no auto-selection, and that is deliberate.** Picking a file by hardware needs a
hardware key, and with two data points any key would be a guess dressed as a lookup. Point
`VH_CALIBRATION` at the right one, or run `calibrate` on the machine you deploy to — that is the
whole intended workflow and it takes seconds.

## What two machines have shown — and which numbers you can believe

**`rebuild_query_ratio` travels.** 0.2 shipped, **0.2048** here, and *identical to four decimals
on five consecutive runs*. The grid-vs-keep-tree crossover looks close to hardware-independent,
which was not assumed and is worth knowing before anyone spends effort tuning it per box.

**`brute_max` does not, and worse, it does not even reproduce on ONE box.** Five runs:

| | run 1 | run 2 | run 3 | run 4 | run 5 |
| --- | --- | --- | --- | --- | --- |
| before the fix | 182 | 182 | **1** | 256 | 256 |
| each rung voted best-of-3 | 96 | 256 | 182 | 256 | 182 |

The `1` was the tell. The ladder reads rungs in ascending order and the first `index` reading
closes the search for good, so a single noisy flip on the *first* rung collapsed the answer by
two orders of magnitude — while a comment in the code claimed a flip cost "one rung's worth of
conservatism". Deciding each rung by a majority of three does not make the comparison less
noisy; it stops one unlucky read from ending the search. The spread is now bounded (96–256,
median 182) instead of unbounded, and `calibrate` prints the per-rung vote so a rung that splits
2–1 tells you the crossover is *there*.

**So read the file this way:** `rebuild_query_ratio` is a measurement. `brute_max` is a summary
of a genuinely fuzzy boundary — which is survivable, because it is an unconditional floor and
the load-aware `scan_budget` rule decides everything above it.

The general rule this suggests, stated as a hypothesis rather than a result: thresholds that
describe a **ratio between two structures** travel between machines; thresholds that describe an
**absolute population** do not.
