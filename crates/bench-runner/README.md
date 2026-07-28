# `bench-runner` — every measurement in this repo, reproducible

A dependency-free Rust binary that runs the kit's benchmarks back to back on a **quiet**
machine and writes the results out as tables.

```bash
cargo run -p bench-runner --release -- --list                 # what would run
cargo run -p bench-runner --release -- --group kd --repeat 3  # the median-split numbers
cargo run -p bench-runner --release -- --group all            # everything (~10 min)
cargo run -p bench-runner --release -- --group all --include-slow --repeat 3
```

Output lands in `bench-results/<unix-time>-<group>.md` (environment header, a per-pass
run table, a metric table, and the full raw output) plus `…-metrics.csv`.

The methodology behind all of this — and the wrong answers it replaced — is
[`docs/MEASURING.md`](../../docs/MEASURING.md).

## Why it exists

The numbers in `docs/` used to be taken by hand, one bench at a time, whenever the work
happened to finish — which makes them hostage to whatever else the machine was doing.
When the whole set was finally re-run this way, three published figures did not survive:

| claim | was | is |
| --- | ---: | ---: |
| `KdTree3` cull vs `Tree3`, clustered | 3.06× | ≈2.0–2.3× |
| fluid: which index wins the neighbour query | `LinearQuadTree` | `MortonGrid`, by 5% |
| point cloud: `KdTree3` k-NN vs `Octree3` | 1.12× | tied (within the k-d tree's own spread) |

None of those were lies; they were single readings taken while something else had the
CPU. That is the whole argument for this tool.

## What it does

- **Builds everything first**, so no compile time lands inside a measured run.
- **Waits for the machine to be free** before every pass. The load signal is a fixed
  calibration loop, not an OS performance counter: if the same arithmetic suddenly takes
  15% longer, someone else is using the CPU. No platform APIs, no localisation problems
  (a Spanish Windows has no counter called `\Processor(_Total)\% Processor Time`), and it
  measures the thing that actually matters — how much CPU *this* process can get.
- **Repeats** (`--repeat N`) and reports **min / median / max / spread** per metric.
  Spread is peak-to-peak over the median: how far a single reading could have been from
  the truth. Anything above ~10% is a number you should not quote without saying so.
- **Records the environment** — rustc version, logical CPUs, git commit, and whether the
  worktree was dirty — because a table of milliseconds without that is folklore.
- **Flags ratios that moved too much to quote.** Any metric whose *unit* is `x` and whose
  spread exceeds 15% is marked in the report and listed after the table; `--strict` exits
  non-zero. Judged by unit and not by name, because a metric called `..._ratio_spread` is
  reported in percent and is the diagnostic *about* a ratio — flagging it for varying is
  circular. Its first act was to fail this repo's own published figures.

## Adding a metric

A bench opts into the metric tables by printing one line per number:

```
#M <key> <value> [unit]
```

`bench-runner` aggregates them as `<bench>.<key>` across passes; everything else the
bench prints is passed through untouched. Adding a metric is one `println!` and needs no
change to the runner. See `kdtree3_bench`, `linear_quadtree_bench`, and the three
`*_wgpu` demos for examples.

## Groups

| group | what |
| --- | --- |
| `core` | structure comparisons: which index for which data |
| `query` | the query verbs: raycast, narrowphase, visibility, BVH variants |
| `gpu` | GPU sort / LBVH build / LBVH query / spatial / visibility |
| `sim` | whole-simulation workloads: siege, horde, critters, parallel AI |
| `demos` | the three application demos, swept over the structures they compare |
| `cli` | the template benchmarks behind the paper's early sections |
| `cold` | the on-disk cold store prototypes |
| `gate` | the committed-baseline regression gate |
| `kd` | just the two benches the median-split numbers come from |
| `all` | all of the above |

Benches marked `[slow]` (minutes each) are skipped unless `--include-slow`.

## Caveats worth knowing

- A **sub-millisecond** measurement is the noisiest thing in any of these tables. The
  clustered k-d cull is ~0.39 ms; treat its ratio as a range, not a figure.
- **Repeating is not the same as comparing.** Measuring A fully and then B fully lets the
  second inherit a machine the first warmed: that alone moved one ratio between 1.57 and
  3.28. Compare with `common::compare2`, which interleaves the pair (`A B B A`) and
  aggregates the median of per-round ratios. Even then, ~20% of between-process variation
  remains — hence the ranges.
- The runner cannot make a *bad* benchmark good. If a demo reports one frame's reading
  rather than a mean over the run, repeating it just gives you three unreliable numbers —
  which is exactly how the stealth demo was caught printing a plausible **zero** for a
  frame that never stepped. It reports means over the run now.
