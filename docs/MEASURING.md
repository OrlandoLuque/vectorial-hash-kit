# Measuring — how this repo takes a number, and what it got wrong first

Every rule here replaced a wrong answer that looked right. They are in the order they bite,
and each one names the measurement that produced it, because a methodology without its
failures is just an opinion with formatting.

The machinery lives in three places: `crates/vectorial-hash/examples/common/mod.rs` (the
clock and the comparison harness), `crates/bench-runner` (the runner, the idle gate and the
ratio gate), and `crates/vectorial-hash/examples/work_counters.rs` (comparing with no clock
at all).

---

## 1. Wall time answers the wrong question

Wall time says *how long you waited*. When a browser steals a core, it inflates, and the
benchmark reports a regression that does not exist. A processor clock only ticks while the
process actually runs.

**But the obvious processor clock is unusable.** `GetProcessTimes` (and `getrusage`) report
in **system timer ticks — ~15.6 ms on Windows**. A 0.3 ms cull measures as **zero**. That is
why microbenchmarks reach for the wall clock in the first place, and why "just use CPU time"
is not the answer it sounds like.

## 2. Scaling the repetition count is the classic fix, and it is not enough here

Repeat the operation until the interval spans hundreds of clock ticks, then divide. Sound,
and it makes a coarse clock usable — but on a 15.6 ms tick, "hundreds of ticks" is **seconds
per sample**. Measured: it turned one bench into a **>10-minute run**. It stays in the
harness for genuinely tiny operations, but it cannot carry the whole job.

## 3. Cycles, and the calibration that betrays you

`QueryProcessCycleTime` (Windows) counts **CPU cycles attributed to this process** at cycle
resolution; Linux's `CLOCK_PROCESS_CPUTIME_ID` gives nanoseconds. Cycles are also
**frequency-invariant**, so turbo and thermal drift cannot fake a regression.

Converting cycles to milliseconds needs a rate, and **the rate is the fragile part**.
Measured over a fixed *wall* interval on a 3×-oversubscribed machine, the process only gets
a fraction of that interval, the rate reads low, and every converted number inflates by the
reciprocal: **a 0.37 ms cull was reported as 5.5 ms**. Calibrating as the **best of several
short trials** fixes it — at least one slice runs on a full core — and ratios never touch
the rate at all.

## 4. The measurement is not free

Two clock reads plus the repetition loop cost cycles. Both are measured once against an
empty closure and subtracted. This matters exactly when the operation is small — which is
when a benchmark is most at risk of reporting its own timing code.

## 5. Cache is a choice, so it is two functions

Repeating an operation leaves its data hot. `measure` reports the **warm** cost (the right
question for something a frame does thousands of times); `measure_cold` evicts the caches
between calls and reports **first touch** (the right question for something a frame does
once), typically several times larger. They are separate names rather than a flag because
quoting one for the other is a real error, not a nuance.

## 6. Compare in pairs, aggregate the ratios

**The biggest single correction in this repo.** Measuring A fully and then B fully is not a
fair comparison at this scale: whoever runs second inherits a machine the first one warmed
or dirtied. The clustered `KdTree3` vs `Tree3` cull ratio, five consecutive runs of the same
binary on the same machine:

| how it was measured | range across runs |
| --- | --- |
| milliseconds, each structure separately | **1.571 – 3.283** |
| CPU cycles, each structure separately | 2.114 – 2.425 |
| **interleaved A/B/B/A, median of per-round ratios** | **2.019 – 2.281** |

`common::compare2` runs `A B B A` per round — A's samples straddle B's, so first-order drift
cancels *within* the round — and reports the **median of the per-round ratios**, plus how
much those ratios moved. The voxel span-vs-naive ratio behaved the same way: `1.129 – 2.654`
taken separately, `1.784 – 1.835` paired.

**Some benches get the pairing for free.** `examples/decision2d` maintains and culls all four
2D structures *inside the same frame loop*, so their samples already straddle each other and a
drift in machine speed lands on all of them alike — the property `compare2` manufactures, here
provided by the simulation. It also could not use `compare2` if it wanted to: maintain is
stateful, and running one structure's frame twice would mean moving the points twice, which is
a different workload. Interleaving is the goal; `compare2` is one way to get it.

## 7. Pairing does not remove between-process variation

Even paired, the same ratio moves ~20% between separate process invocations. So the honest
form is a **range**, not a point: the docs say `≈2.0–2.3×`, not `2.2×`.

`bench-runner` enforces this: any metric whose **unit** is `x` and whose spread across
passes exceeds 15% is flagged, and `--strict` exits non-zero. It is judged by unit, not by
name — a metric called `..._ratio_paired_spread` is reported in percent and is the
diagnostic *about* a ratio; flagging it for being variable is circular. (The gate's first
act was to fail this repo's own published figures. That is what it is for.)

## 8. Best of all: do not use a clock

What differs between a binary tree, an octree, a k-d tree and a grid is **how much work each
does**: node boxes classified, points tested. Those are integers, and
`examples/work_counters.rs` counts them by wrapping the query volume in a counter — no
library change, since `cull` accepts any `Shape`.

`cull` takes a `Shape`, so it is counted by wrapping the query. `knn` and `raycast` take a
point and a ray — nothing to wrap — so for those the counter goes in the **item**: every
traversal must ask an item where it is before testing it, so counting `position()` counts leaf
work across all three verbs with no library change. It is what showed that `KdTree3`'s k-NN
advantage over `Tree3` is **1.8x on clustered data and 1.06x on uniform** — the median split
does not beat the binary tree, it beats *skew* — and that a uniform grid's shell expansion
tests **280x** more points clustered than uniform.

It also checks a guarantee no test covered: the DDA ray walks promise to be a strict *subset*
of the exact capsule (they visit only leaves the centre ray crosses). Counted: zero invented
hits, 75% of the hits for 23% of the tests on `Tree3`, 50% for 11% on `Octree3`. A recall
number is the honest way to describe a cheap approximate query — "faster" alone is not.

**Proven, not asserted**: the whole report is byte-for-byte identical between an idle run and
one taken with 32 processes burning CPU.

And because the counts have no variance, they are the only thing here that can be gated with
`==` rather than a tolerance. `tests/work_counts.rs` is that ratchet: twenty traversal counts,
checked exactly, in CI, on every push. The timing gate has to pass anything within 25%, which
is right for a clock and also means a traversal change costing 15% more work sails through it
looking like noise. This one cannot miss it. Two lessons are baked into the file: the first
version blessed `tested = 0` for four of five 3D structures (uniform queries mostly land in
empty space — a ratchet holding nothing), so half the queries now aim at a cluster; and it was
checked against a deliberate perturbation (leaf 16 → 15) to confirm it actually fails. Use it for algorithmic claims; use time for the
constant factors it cannot see (the grid does 30× the point tests in 2D and still wins
several timed configs — that gap *is* the memory wall, quantified).

## 8b. A count only counts what you counted

The point counter counts `position()` calls — **leaf work**. For `cull` the descent is counted
too (`classify_aabb`), but `knn` has no query volume to classify, so **visiting a cell costs
nothing the counter can see**. That is not academic: rewriting `MortonGrid3::knn` to expand
per-axis left the points-tested count on a *cubic* world essentially unchanged (14 946 →
14 823) while the time fell **2.3x**. The saving was entirely in cells never iterated — the old
enumeration walked the whole Chebyshev shell and rejected out-of-grid cells one at a time,
the new one clamps its loops to the grid. Real work, invisible to that metric.

The rule: counts prove an algorithmic change; they do not bound one. When a count says
"nothing changed" and a clock disagrees, the clock may be measuring something you did not
think to count.

**And then count that too.** The blind spot was structural rather than fundamental, so the
grids gained a `grid-stats` feature: an opt-in, thread-local counter of cells looked up, routed
through a single `bucket_of` helper that every query path goes through so it cannot drift out
of step with the traversals. Off by default and genuinely absent when off — neither the counter
nor the increments are compiled. `work_counters` grows a `cells/query` column when built with
it, and the first thing that column said was that a clustered `MortonGrid3` k-NN visits **6 888
cells to return 8 neighbours** (2D: 203). That number had never appeared in any table here.

The `cull` tables carry it too now, and there the column exposed something sharper: on a sphere
cull `morton3` classifies **0.2 boxes** per query while looking up **56.8 cells**. The `boxes`
column — the descent counter — was reporting almost nothing for grids, because a grid barely
classifies anything; it looks things up. Two verbs, one structure, and the existing counter was
blind to its main cost in both.

Cell counts are as deterministic as point counts, so they are gated the same way: exactly, in
CI, by a second ratchet in `tests/work_counts.rs` and a CI job that builds with the feature.
A ratchet nobody's build enables is a ratchet checking nothing.

## 8c. A short window hides a slow leak

The linear trees' keep path was first measured over **20 frames**, and the shape drift looked
like a mild fixed tax: 1.20x slower culls at 10 % churn, 13 914 leaves. Re-run over **300**, the
same configuration reads 18 822 leaves and **1.37x, still climbing**. The effect was never a tax;
it was an accumulation — every frame added subdivisions and none were ever given back.

Twenty frames was plenty to measure a *rate* (maintain cost per frame, stable from the first
frame) and far too short to measure a *state* that integrates over time. Both numbers came out
of the same bench in the same run, which is exactly how this hides: the columns look equally
trustworthy and one of them silently depends on a parameter nobody was varying.

Ask of every column: **is this a rate or an accumulation?** If it accumulates, the run length is
part of the answer, and one run length is one point of a curve. (The same shape of mistake as
measuring a grid at 100 % churn and concluding grids must rebuild — § "Grids rebuild, trees
keep" in CHOOSING.md.)

## 8d. In an interleaved loop, position in the frame is a confound

A sweep that measures several structures inside one frame loop — maintain them all, then cull
them all — is comparing arms that run at **different points in the same frame**, and therefore
in different cache states. The arm timed last has had every other arm's working set evicted
out from under it, or warmed into it, depending on what they touched.

This is not hypothetical. The 3D decision map reported the kept Morton grid culling **1.18×
faster** than the rebuilt one at identical world, identical `levels`, identical contents and
identical probes, and the same ~15 % gap had shown up in a second, unrelated bench — two
sightings, which is normally when you start believing an effect.

Two checks killed it:

- **Swap the two arms' positions in the frame.** `morton`'s cull went 1.66 → 1.53 µs purely by
  being measured later, and the gap fell from ~1.11× to 1.06×.
- **Measure them in isolation** (`examples/kept_grid_query_edge`): same points, same levels,
  same probes, one grid kept across 150 frames and one rebuilt every frame. Result **1.082 vs
  1.084 µs — 1.00×**, with `grid-stats` reporting the identical **25 925 cells visited** for
  both. No traversal difference existed to explain, and none of the timing difference survived.

So: a difference under ~10 % between two arms of an interleaved sweep is not evidence. Reach
for cell/point counts first (they are position-independent), and when the question is really
about time, take the comparison **out** of the loop and pair it directly. The decision map's
*maintain* column is far less exposed — each maintain is a large block of work that dominates
whatever preceded it — but its cull column should be read as ordering-sensitive.

## 9. Publish from an idle machine

None of the above removes contention: another process still evicts your cache lines and eats
memory bandwidth, so under load the same work genuinely costs more cycles (measured: ~1.7×
under 3× oversubscription, for *both* clocks). `bench-runner` waits for the machine to be
free before every pass — using a calibration loop rather than OS performance counters, so it
needs no platform API and no localised counter names.

---

# Four traps that are not about clocks

## An index only knows what it holds

An index and a linear scan disagreed on ~77% of frames in the stealth demo. The library was
innocent — 400 random frustums × 4000 points, **0 disagreements**. Some agents had drifted
outside the index's world box, which `bulk_load` correctly drops and a scan still counts.
Neither side was wrong; they were answering questions about **different sets**. Before
attributing a difference to the algorithm, verify both sides see the same population.

## A reference that shares the assumption cannot catch the assumption

Golden data generated from the implementation under test validates nothing. The sphere
reference sets in `_dev/sphere_golden/` are computed by **two independent paths** and emitted
only when they agree — and they cover three centre-alignment conventions, because a half-block
shift produces a set that is self-consistent, symmetric, and wrong.

Related: **symmetry is not identity**. A sphere centred at a block centre and one centred at
a block corner are *both* symmetric and differ by 349 blocks at r=8.

## A set-comparison test cannot catch a performance defect

An implementation that walks the wrong axis *consistently* returns the correct set and is
merely slow. No golden file detects that. Convert the performance property into a
deterministic assertion instead — "consecutive blocks in the inner loop have consecutive
storage indices" — or put a floor under it (the fast path must beat the baseline by a margin).

## Summaries must aggregate over the run, not sample it

The stealth demo reported one frame's values. Across three passes, one landed on a frame that
had not stepped and printed a clean, plausible **zero**. It reports means over the run now,
and prints the stepped-frame count so an empty run cannot masquerade as a fast one.

**And check your test is not vacuous.** A set-comparison test that compares two empty sets
passes forever. The frustum properties were instrumented once to confirm they find 12.3 items
per cone on average (max 66) before being trusted.

---

# The checklist

1. Is this the bottleneck? Profile before optimising. "No" is a valid, valuable answer.
2. Are both sides answering the same question, over the same population?
3. Is the test non-vacuous — does it actually find things?
4. Warm or cold? Say which.
5. Comparing two things? Pair them in time and aggregate the ratios.
6. Repeat, and report the spread. Above ~15% on a ratio, quote a range.
7. Idle machine for anything you publish (`bench-runner` gates on it).
8. Can it be counted instead of timed? Then count it.
9. Quote ratios, not absolutes, unless the environment is pinned in the same table.
