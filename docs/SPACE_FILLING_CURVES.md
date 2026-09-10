# Space-filling curves: Morton, Hilbert, and what a "geohash" actually is

Measured with `cargo run -p vectorial-hash --example cold_index_bench --release --features parallel`,
then checked against the published closed forms. The headline is that the kit's numbers **reproduce
the theory to within 3 %**, and that one intuition everybody has about these curves is wrong.

## A geohash is a Morton code

A geohash interleaves the bits of the coordinates and encodes the result so that **a prefix is a
coarser cell**. That is a Z-order (Morton) key with the prefix property made explicit. "3D geohash"
is the same construction over three axes — not a different curve.

So the kit already computes them, in two different shapes:

| | key | store | prefix property |
| --- | --- | --- | --- |
| `MortonGrid3` | Morton at a **fixed** level | `HashMap` | computed, never used |
| `LinearOctree3` | Morton path **+ level** in one `u64` | `HashMap` | used to find ancestors |

`LinearOctree3` is a variable-precision 3D geohash — a linear octree. What neither has is the other
half of what makes geohashes useful elsewhere: the keys in an **ordered** store, so that a prefix is
a *range* and a box query becomes range scans.

## The locality result, and it matches the closed form

Boxes of side `s` **in cells** (16 bits/axis), every cell's key computed, sorted, maximal runs of
consecutive keys counted, averaged over 200 random positions:

| box side | Morton runs | Hilbert runs | M/H | H/s² | M/s² |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 25 | 16 | 1.62× | 0.98 | 1.58 |
| 8 | 114 | 65 | 1.75× | 1.02 | 1.78 |
| 16 | 462 | 249 | 1.86× | 0.97 | 1.80 |
| 32 | 1 904 | 993 | 1.92× | 0.97 | 1.86 |

Moon, Jagadish, Faloutsos & Saltz (TKDE 2001) derive the Hilbert clustering number as the query's
**surface area divided by twice the dimensionality**. For a cube of side `s` in 3D that is
`6s² / 6 = s²`. The `H/s²` column is that prediction, and it reads 0.97–1.02. The bench now
**asserts** it: a wrong Hilbert encoder still prints a plausible table, so the check has to be a
check.

**The correction worth writing down.** The `M/H` column climbs — 1.62, 1.75, 1.86, 1.92 — and it is
tempting to read that as Hilbert's advantage *growing* with query size. It is not: it is
*converging*. Xu & Tirthapura (PODS 2012) proved the Z-order curve is within a **constant factor**
of optimal for any space-filling curve, and the `M/s²` column shows exactly that constant settling
in at ~1.9. The small-box ratios are below the asymptote, not on the way up to infinity. Choosing
Hilbert buys you a constant near 2, once, forever — worth having, and not a different complexity
class.

## …and if you scan the span, Hilbert is actually WORSE

The obvious way to use a sorted key store is to scan `[min_code, max_code]` over the box:

| N | Morton over-scan | Hilbert over-scan | avg hits |
| ---: | ---: | ---: | ---: |
| 100 000 | **102.0×** | 112.2× | 91 |
| 1 000 000 | **101.4×** | 111.3× | 913 |

Two orders of magnitude of waste, and the curve that wins on locality **loses here by 10 %**. That
is not a contradiction, it is the distinction the whole page turns on: Hilbert's advantage is in how
many *runs* the box decomposes into, and a span scan does not care about runs. It cares about the
distance from the lowest key in the box to the highest — and Hilbert, not being monotone in the
coordinates, can put those two further apart than Morton, whose extremes sit exactly on the box's
low and high corners.

So the summary is sharper than "the span scan is bad": **scanning the span throws away Hilbert's
advantage and then charges you 10 % for having chosen it.**

> **Correction, 2026-09-11.** This table first read 7 299× and 7 251× — "indistinguishable, four
> orders of magnitude" — and both figures were wrong by ~72×. Two bugs, found when the extremes
> were made exact. The cell mapping **masked** instead of clamping, so a query box overlapping the
> world edge wrapped round to the far side and became an enormous different box; and Hilbert's
> extremes were *sampled* from 8 corners and 6 face centres, with a comment admitting the sample
> under-counted its advantage. The sampling error was real but small; the masking error was the 72×,
> and it inflated both columns roughly equally, which is exactly why the wrong numbers looked
> plausible — they preserved the ratio and only broke the magnitude. Both are fixed
> (`box_extremes` walks the octree for the true min and max in O(8·depth), and `cell` clamps),
> and only after that did the ordering between the curves become visible at all.

This is the point at which the literature stops recommending the naive scan and reaches for
**BIGMIN / LITMAX** (Tropf & Herzog 1981, later the UB-tree line of work). They are a pair and both
are needed: given a key that has left the box, `BIGMIN` is the smallest in-box key above it — where
to resume — and `LITMAX` is the largest in-box key below it — where to close the run just abandoned.
Knowing only where to re-enter tells you nothing about where to cut.

## Scanning the runs: exact, and the knob has a floor

`box_ranges` in the same bench decomposes a query box into maximal contiguous key ranges by
descending the octree — both curves are hierarchical, so every subcube is a contiguous interval,
and only nodes straddling the boundary have to be opened. That is the same decomposition
BIGMIN/LITMAX produce as a cursor; the recursion is easier to verify, so it goes first.

| bits/axis | cells/box | Morton ranges | Hilbert ranges | over-scan vs cell box | over-scan vs the SPHERE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 7 | 82 | 45 | 1.00× | 2.86× |
| 8 | 25 | 1 117 | 610 | 1.00× | 2.08× |
| 10 | 99 | 17 733 | 9 358 | 1.00× | 1.90× |
| 12 | 392 | 291 701 | 153 282 | 1.00× | 1.86× |

Three things, and the third is the useful one.

**It is exact.** Over-scan against the cell box is 1.00 at every resolution — the scan reads
precisely the box, against the span scan's 102×. That also cross-checks the decomposition: the
range counts reproduce `s²` again (45 vs 49, 610 vs 625, 9 358 vs 9 801, 153 282 vs 153 664), which
is now the **third independent** confirmation of the same law, after A1's run counting and the
published closed form.

**Hilbert's constant is exactly what A1 said.** 82/45, 1 117/610, 17 733/9 358, 291 701/153 282 =
1.82, 1.83, 1.89, 1.90. Half the ranges for identical over-scan — the same answer for half the
cursor seeks.

**And the resolution knob has a floor, not a direction.** The cell-box column flatters itself: the
caller asked for a sphere. Against the real query, a coarse key costs 2.86× and a fine one settles
at 1.86× — and it settles there because **1.86 ≈ 6/π = 1.91**, the volume of a cube over its
inscribed sphere. That is not an artifact to tune away; it is the price of bounding a ball with an
axis-aligned box. Meanwhile the range count climbs as the square of the box in cells. Coarse: few
ranges, bad sphere. Fine: geometric floor, exploding ranges. At 16 bits, A2's bubble would need
~43 M ranges to fetch ~780 points.

So "use Hilbert and scan the runs" is only half an answer. The other half is choosing the key
resolution so that `s²` stays small — and nobody states that precondition in the same breath.

## BIGMIN and LITMAX, and why they are a pair

`cargo run -p vectorial-hash --example bigmin_litmax --release` implements the cursor form: the
thing you want when you are dragging a cursor across a B-tree or a trie and need to know where to
jump, rather than materialising every range up front.

- **`BIGMIN(qlo, qhi, z)`** — the smallest key ≥ `z` inside the box. *Where to resume* once the
  curve has wandered out.
- **`LITMAX(qlo, qhi, z)`** — the largest key ≤ `z` inside the box. *Where the run you just left
  actually ended.*

Knowing only where to re-enter tells you nothing about where to cut, which is why every treatment
introduces them together and why quoting one without the other is half an algorithm.

The file is mostly verification, deliberately. The bit-twiddling is notorious for being subtly
wrong in a way that still returns *a* key that is *usually* right — the failure hides at the box
boundaries and surfaces as a query that silently drops a few points. So both are checked two
independent ways: **exhaustively against brute force** (300 random boxes × all 32 768 keys at 5
bits/axis = 9 830 400 probes, every one exact), and by **walking a box end to end with the pair**
and comparing the runs against the enumerated truth (200 boxes, 50 126 runs, identical).

One note on the verification itself, because it bit: written the obvious way — a linear `find` over
the key space inside the probe loop — the check is quadratic in the key space, 3 × 10¹¹ operations,
and never returns. The oracle is now built once per box and answered by binary search. **A
verification that cannot finish verifies nothing**, and it fails silently by looking like a slow
test rather than a broken one.

## What actually answers the query today

Same bench, section B — an area-of-interest bubble over the same points:

Ranges over two runs, because one run of a timing is a memory and not a measurement
(`MEASURING.md` § 7) — and these were taken on a laptop with the project's disk attached
externally, so read the ORDER, not the digits:

| N | BTree naive span | BTree cell-probe | `MortonGrid3` | `Tree3` |
| ---: | ---: | ---: | ---: | ---: |
| 100 000 | ~361 µs | 6.2 µs | **5.6 µs** | 6.0 µs |
| 1 000 000 | 6 443–8 138 µs | **17.9–18.9 µs** | 23.2–23.3 µs | 32.5–35.5 µs |

The naive span arm swings 26 % between runs; the three usable arms swing 1–9 %. Two things still
fall out cleanly. The span scan is unusable, as A2 predicts. And at a million points the **ordered
store with per-cell probes beats both in-memory structures**, by a margin (17.9–18.9 against 23.2
and 32.5–35.5) wider than the spread. A sorted key store is not only a disk shape.

## Open

- **Promote the pair out of the example** if an ordered-store index is ever built here. Today
  nothing in the kit uses a sorted key store, so the algorithms have no caller and live where they
  were measured.
(The radix trie is answered below; the `radix` elsewhere in this repo is radix *sort*, for building
the GPU LBVH — unrelated.)

## The radix / PATRICIA trie: measured, and the question closes

`cargo run -p vectorial-hash --example radix_trie_bench --release` — 200 000 points, 10 bits/axis,
an 8-ary trie with path compression racing the kit's `Octree3` on the **same sphere**, with both
answers asserted equal to brute force and to each other.

| data | trie build | octree build | trie query | octree query |
| --- | ---: | ---: | ---: | ---: |
| uniform | 294 ms | **117 ms** | 43.6 µs | **12.2 µs** |
| clustered | 177 ms | **119 ms** | 25.6 µs | **4.9 µs** |

The hypothesis the bench was written to kill was that a radix trie over Morton keys at 3 bits per
digit simply *is* an octree with path compression. It is: `Octree3` wins 3.6–5.2× on the query and
1.5–2.5× on the build. The kit already has this structure, spelled better.

**The reason matters more than the verdict, because the famous lever turns out to be the small
one.** The trie pays **~1.4 nodes per item** — it descends to full depth for every point, so a leaf
holds almost nothing. Path compression attacks the *depth* of sparse single-child chains and it
does help exactly where predicted, clustered data keeping 263 k nodes against uniform's 288 k — but
that is **8 %**. What the octree has instead is an **item limit**: stop subdividing at 8 items and
the node count falls by nearly 8×. Adding that to the trie would not make it competitive; it would
make it an octree.

The ordered-store question is separate and stays open — that is what a `BTreeMap` (or a real
on-disk B-tree) answers, and it is measured properly in `cold_index_bench` with range scans rather
than per-cell probes. Deliberately not raced here: probing a 61-cell-wide box cell by cell is
230 000 lookups, which is not a rival, it is a straw man.

## Sources

- Moon, Jagadish, Faloutsos & Saltz, *Analysis of the Clustering Properties of the Hilbert
  Space-Filling Curve*, IEEE TKDE 13(1), 2001.
- Xu & Tirthapura, *On the Optimality of Clustering Through a Space Filling Curve*, PODS 2012.
- Tropf & Herzog, *Multidimensional Range Search in Dynamically Balanced Trees*, 1981 (BIGMIN /
  LITMAX).
