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

## 8d. A ratio that is not a property — the anatomy of four wrong answers

The 3D decision map reported a kept Morton grid culling **1.09–1.17× faster** than a rebuilt one
holding the same points at identical world, identical `levels` and identical probes. It took four
wrong explanations to find out what that number is, and the sequence is more useful than the
answer.

**What it is not.** Six hypotheses, each refuted:

| hypothesis | test | result |
| --- | --- | --- |
| one grid traverses less | `grid-stats` cell counts | **identical** — nothing algorithmic |
| position in the frame | swap the two arms | 1.11× → 1.06×, so: *probably it* |
| position in the frame | **rotate all nine arms every frame** | ratio **unchanged** |
| a warm bench hides it | isolated, 4 000 culls ×5 | **1.00×** |
| a cold first-touch effect | isolated, 16 culls after the maintain | **0.97×** |
| the grids hold different items | assert equal populations per frame | **equal** |

**The seventh looked decisive and was not.** The only surviving idea was that the gap comes from
sharing the frame with seven other structures, whose working sets pass through the cache between
arms — something rotation equalises the *position* of but not the *presence* of. `$D3_ARMS`
runs the map with only chosen arms, skipped in both phases. Eight A/B/B/A-paired readings gave
**zero overlap**: alone 0.99–1.07, with company 1.15–1.22. Every reading separated cleanly.

Then one parameter was swept, and it fell apart. Six paired readings per population
(rebuilt ÷ kept — above 1.0 means the kept grid is faster):

| population | two arms alone | all nine arms |
| ---: | --- | --- |
| 5 000 | 0.978–1.021 (**≈1.00**) | 0.960–1.028 (**≈1.00**) |
| 20 000 | 0.976–1.008 (**≈1.00**) | 1.127–1.202 (**≈1.17**) |
| 50 000 | **1.147–1.218 (≈1.19)** | 0.993–1.077 (**≈1.03**) |

At 50 000 the advantage appears *without* company — the opposite of the mechanism the 20 000 rows
had just "proved". The distributions do not overlap within any population, so none of this is
noise; the effect is real, reproducible, and **conditional on the configuration in a way no single
mechanism tested predicts**.

### What can be said, and what should be quoted

The kept grid is never *slower*; it is equal or up to ~1.2× faster, and *when* that happens
depends on population and on what else shares the cache. Since the traversal counts are identical,
this lives entirely in the memory system — a footprint coincidence, not a difference between
keeping and rebuilding.

So: **do not quote the cull ratio between these two.** It is a property of the measurement
configuration, not of the strategies. The column that distinguishes them reliably is *maintain*,
where the difference is algorithmic and shows up at every population.

The microarchitectural cause — which cache level, which footprint crosses which boundary — is not
pinned down, and pinning it would need counters this kit does not have. That is a smaller gap than
it looks: the practical question ("is one of these a faster structure to query?") is answered, and
the answer is no.

### The four lessons, which are the real output

1. **One A/B on a noisy metric is not evidence for a cause any more than for an effect.** The swap
   experiment moved the number by less than the run-to-run spread and was believed for a day.
2. **Zero overlap across paired readings is not proof of a mechanism** — only that the two
   conditions differ *at that configuration*. It is exactly as convincing at a point where the
   conclusion happens to be wrong.
3. **Sweep one parameter before believing any of it.** A single parameter change refuted a result
   that eight clean paired readings had just established.
4. **Controls are cheap and beliefs are expensive.** Rotating arms, asserting equal populations,
   counting traversals and `$D3_ARMS` each cost under an hour; the four wrong explanations cost
   two days between them.

## 8e. Noise is episodic, so "wait for an idle machine" is not the control you think

The obvious defence against a noisy machine is to wait for it to go quiet, and both the gate and
`bench-runner` said exactly that in their own output. Measured on this desktop at 18–32 %
background load:

- **Ten identical runs of one op: 851–1 192 µs, a 1.40× spread.** Not two clusters — values
  spread through the whole range.
- **Two consecutive gate runs on an untouched op: ±3 %, then +74 %** — and the +74 % run had
  *lower* reported CPU load than the ±3 % ones.

So load percentage does not predict it. Whatever the source (turbo residency, a background task
bursting, thermals), it arrives in episodes an idle-*looking* machine does not rule out, and
`bench-runner`'s idle gate waits up to 300 s per bench for a threshold it cannot see through.

Two things do work, and both are this file's own estimator applied where it was missing:

1. **Minimum over repeats.** A minimum converges on the uncontended floor, which is the stable
   quantity; a mean or a single reading chases the episode. It is why the gate confirms a
   suspected regression over further passes — and now why `--save` does the same, a baseline
   being the thing every future run is judged against. Saving one pass would have recorded an op
   at **1.76× its floor**, and then never flagged a 70 % regression in it.
2. **Pairing inside one process.** The `$D3_ARMS` comparison in § 8d ran A/B/B/A and separated
   cleanly at 18–46 % load, because an episode that hits one arm hits its partner too.

The corollary for anything you intend to publish: **a number measured once is unmeasured**, and
"the machine looked quiet" is not a method.

## 8g. Setup inside the clock does not cancel — it inverted a result

`bulk_load` consumes its input, so the natural way to bench it is:

```rust
compare2(rounds, || build(items.clone()), || build_par(items.clone()))
```

Both arms clone. The clone is the same size, the same type, on the same thread. It reads like a
constant added to both sides, and the arithmetic seems to settle it: if the measured ratio is
`(C + t_par) / (C + t_ser)`, then a larger `C` drags the ratio toward 1.0 and can never carry it
across 1.0. A real speed-up still reads as a speed-up. It just reads smaller.

That argument is wrong, and the evidence is not subtle. `IntegerTree::bulk_load_par`, 500k items,
16 threads:

| harness | 10k | 100k | 500k |
| --- | ---: | ---: | ---: |
| clone **inside** the clock | 0.67× | 0.68× | 0.89× |
| clone **outside** (`common::abba`) | 1.97× | 2.01× | 2.33× |

Same code, same machine, same A B B A round structure, opposite conclusions — one says ship it,
the other says the parallel build is a pessimisation. The flaw in the arithmetic is the premise
that `C` is the same in both arms. It is not: the parallel arm frees the input from a different
allocator state than the serial one, and its 16 workers are contending for memory bandwidth with
whatever the clone left in flight. The setup and the work are not independent, so they do not
subtract.

**The distortion scales with how cheap the real work is**, which is the part worth carrying
forward. It flipped both binary trees and barely touched `QuadTree::bulk_load_par` (1.44–1.75×
measured either way), because a 4-way build is expensive enough to dominate its own clone. So a
harness of this shape lies *most* about the code that is *fastest* — precisely the code you are
most likely to be trying to prove is fast.

**How it was caught, which is the transferable part.** Not by inspection. The bench measured a
control it did not need: the float `Tree` twin, same binary split, same off-arena parallel path.
It read 0.68× as well — and `Tree3`'s own long-standing bench had that same algorithm winning
2.17× over serial. Two of the kit's own benches disagreeing about one algorithm is what moved
suspicion from the code to the harness. § 8d's lesson was that one A/B on a noisy metric is not
evidence of a cause; this is its companion: **when two of your own measurements disagree, the
harness is a suspect, not just the machine.**

The fix is `common::abba`: clone every input up front, outside the clock, and have each timed call
consume one. Use it for anything that takes ownership of its input.

## 8f. A number nobody can re-run is not a measurement, it is a memory

On 2026-07-29 a headline figure went stale without anyone touching the sentence that stated it.
THREE_D.md and the README said `LinearOctree3`'s clustered k-NN was ~5x `MortonGrid3`'s. That was
true when written; then the grid gained per-axis shell expansion (§ 8b's sibling finding) and got
3.6x faster on exactly that workload, leaving the published claim wrong by a factor of three.
Nothing broke. No test failed. The prose simply went on describing a world that had moved.

The general shape: **a comparative number has two operands, and it can be invalidated by a change
to the one it does not name.** No amount of care about the structure you are documenting protects
you, because the edit that falsifies the sentence happens somewhere else entirely.

There is no cheap gate for "is this number still true" — re-running every bench takes minutes, and
§ 8e says this machine's noise is episodic, so a gate that fails on a re-measurement would fail on
weather. What *is* cheap is enforcing the precondition:

```bash
scripts/check-docs-numbers.sh      # in CI, gating
```

It checks that every command a doc tells you to run exists, that every link and `#anchor`
resolves, and that **every section quoting a measurement names something you can actually run** —
where "something you can run" is decided against the real set of examples, bins and benches, not
against a phrasing convention. Sections inherit a source from their ancestors, because a `###`
table under a `##` heading that names the command is properly sourced.

Two design notes worth keeping:

- **The first version was wrong in the direction that destroys a check's credibility.** It
  demanded a literal `cargo run` invocation and flagged a dozen sections that named
  their source perfectly well as `` `critters3d_headless --parallel` `` or as a path to the file.
  The docs were right and the check was wrong. A checker that cries wolf gets switched off, so
  the bar is not "did the author phrase it my way" but "can a reader find the thing".
- **Dated log entries are exempt** (BACKLOG.md, CLAUDE.md). "115 lib tests green" under a
  2026-07-24 heading is a record of that day, and rewriting history to match the present is how a
  log stops being evidence.

And it was verified by breaking it: strip a source, break a link, break an anchor — all three
flag, and all three clear on restore. A green check that has never been shown to go red is not a
check.

## 8h. A search whose first step is noisy does not fail gracefully

`calibrate` finds `brute_max` — the population where an index stops losing to a linear scan — by
walking a ladder of populations and taking the largest one where the scan still wins. It used to
bisect; § 8's cousin, [#131], replaced that because a bisection assumes a monotone predicate and
near the crossover this one is a coin toss. The ladder came with a comment claiming that a single
noisy flip now cost "one rung's worth of conservatism".

It did not. Five runs on one idle machine:

```
brute_max = 182, 182, 1, 256, 256
```

The `1` is the whole lesson. The rungs are read in ascending order and the first `index` reading
ends the search for good, so a flip on the *first* rung — the smallest, cheapest, noisiest
comparison — collapses the answer by two orders of magnitude. **The bound the ladder was chosen
for applies to every rung except the one that can do the damage.**

Deciding each rung by a majority of three probes bounds it for real:

```
brute_max = 96, 256, 182, 256, 182      (median 182)
```

That is not a more precise measurement — the underlying comparison is exactly as noisy — it just
stops one unlucky read from being decisive. The per-rung vote is printed now, because a rung that
splits 2–1 is more informative than the number the run writes out: it says the crossover is
*there*, and that no single number describes it.

**The transferable part.** When a search's early steps are cheap and its late steps are expensive,
the cheap ones are the noisy ones, and a search that terminates on the first positive is a search
that terminates on noise. Either repeat the early steps, or arrange for termination to need
agreement.

Two things this run also settled, which is why it is worth writing down rather than just fixing:

- `rebuild_query_ratio` read **0.2048 on all five runs**, against 0.2 shipped from a very
  different machine. Some thresholds genuinely travel.
- `brute_max` does not travel *and does not even reproduce locally*. The rule of thumb this
  suggests — a ratio between two structures travels, an absolute population does not — is
  recorded in `calibrations/README.md` as a hypothesis, not a finding.

## 9. Publish from an idle machine — necessary, not sufficient

None of the above removes contention: another process still evicts your cache lines and eats
memory bandwidth, so under load the same work genuinely costs more cycles (measured: ~1.7×
under 3× oversubscription, for *both* clocks). `bench-runner` waits for the machine to be
free before every pass — using a calibration loop rather than OS performance counters, so it
needs no platform API and no localised counter names.

Read that together with § 8e, though: waiting for an idle machine lowers the *rate* of bad
episodes, it does not eliminate them, and the wait is charged per bench (up to 300 s each, which
on a loaded desktop is an hour before anything is measured). Prefer a design that survives an
episode — minimum over repeats, pairs inside one process — and treat idleness as the thing that
makes those cheaper rather than the thing that makes them unnecessary.

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

## 8i. A sweep is a slice, and the axis you held fixed is the one that decides

`grid_tree_frontier` sweeps the (churn × query-load) plane, 42 cells, five repetitions each,
both arms kept and neither rebuilt. At every one of those 42 cells the kept `MortonGrid3` beat
the keep-`Tree3`, by 1.5–2.2×. It is a careful table and its conclusion — *the grid wins here* —
is wrong in a way no amount of care inside the plane could have caught.

The `radius` was a `const`, 36, chosen because it is the cell size. Re-run at `radius=8`:

| | radius 8 | radius 36 |
| --- | --- | --- |
| grid wins | 8 of 42 cells | **42 of 42** |
| typical ratio | 0.36–1.5, no structure | 1.5–2.2 grid |

Same populations, same churn rows, same query-load columns, opposite answer. And at the horde's
own coordinates (`churn=0.056 queries=0.06`) `pick_a_structure` names **three** different winners
along that axis alone: the tree at radius 1–4, the grid from 12 to 90, a **brute scan** at 170.

The failure is not that a third axis existed — there is always a third axis. It is that the value
we froze it at was **derived from the structure under test**: radius = one cell width is the
extent a grid is built to serve in one lookup, so the sweep was conducted at the grid's best
point and reported as a property of the plane. A held-fixed parameter chosen *from* one arm is a
thumb on the scale, and it does not show up as noise, a wide spread, or an unstable ratio — every
cell was reproducible to a few percent.

**Two questions to ask of any sweep, including your own:**
1. What did I hold fixed, and would a reader guess that it was an axis?
2. Was that value derived from one of the arms? If so, sweep it too, or state the conclusion as
   *"at radius 36"* — which is a real result and a much smaller one.

This resolved a 7× disagreement between two of our own measurements (`docs/HORDE.md`), and it
demotes `Thresholds::rebuild_query_ratio` from "a threshold fitted at maximum churn" to "a
threshold fitted at maximum churn **and one query extent**" — the policy reads two axes of a
three-axis surface. See § 8d for the other shape of this mistake: there, a ratio that *did* move
turned out to depend on something not in the model either.

## 10. A measured file must say which machine measured it

Two files in this repo hold measured values that a later run is judged against: the regression
gate's `benches/baseline.tsv` and the adaptive index's `calibrations/*.txt`. Both are
hardware-specific, and until 2026-09-03 **neither recorded the hardware**. The gate's own header
said to treat cross-machine numbers as orientation only, and CI marked its run informational for
exactly that reason — so the knowledge existed, in prose, where the tool could not read it.

Running the gate on a second machine therefore printed a full table of confident verdicts about
nothing. Measured on the first run of the check: the calibration loop read **11.9 ms** here
against the committed baseline's **5.5 ms**, and seven k-NN ops came out **+41 % to +60 %**. All
of it would have been reported as regression.

**Clock normalisation is not provenance.** `_calib` is a fixed CPU loop, so dividing by it cancels
clock speed and nothing else. Cache sizes, memory bandwidth and core counts pass through the
division untouched, and on a spatial-index workload they are most of what separates two machines.
A normalised number from elsewhere is still a number from elsewhere.

The fix is `src/machine.rs`: a coarse, dependency-free fingerprint (`os/arch Nc HOST`) written
into both file formats, and three states rather than two —

| | what it means | may a verdict be issued? |
| --- | --- | --- |
| `SameMachine` | fingerprint matches | yes |
| `OtherMachine(id)` | measured elsewhere, and it says where | no |
| `Unknown` | file predates the fingerprint | **no** |

`Unknown` being distinct from `SameMachine` is the part worth copying. Every file written before
the check existed has no fingerprint, and treating "no evidence" as "matches" would have left the
whole problem in place for precisely the files most likely to be stale.

Refusing is only half a fix, though: a developer working for weeks on a second machine would
simply have no gate. So a machine keeps **its own baseline beside the committed one**
(`--save --local` → `baseline.<slug>.tsv`, auto-selected there afterwards), and `--local` is
explicit so a second machine cannot silently overwrite the first one's numbers.

The general form: **a number and the conditions it was taken under are one object.** Splitting
them — number in the file, conditions in a comment, in a README, or in a colleague's memory — is
how a measurement becomes a memory (§ 8f) and how a comparison becomes a fabrication.
