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

## …and it buys nothing at all if you scan the span

The obvious way to use a sorted key store is to scan `[min_code, max_code]` for the box's corners:

| N | Morton over-scan | Hilbert over-scan | avg hits |
| ---: | ---: | ---: | ---: |
| 100 000 | 7 299× | 7 251× | 78 |
| 1 000 000 | 72 342× | 71 582× | 782 |

Four orders of magnitude of waste, and **the two curves are indistinguishable** — 0.7 % apart. The
locality advantage measured above is entirely about how many *runs* the box decomposes into; the
*span* from the lowest to the highest key in the box covers nearly everything either way, because
the corners are far apart along the curve whatever the curve does in between.

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
precisely the box, against the span scan's 7 299×. That also cross-checks the decomposition: the
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

BIGMIN/LITMAX themselves are still to write, as the cursor form for a real ordered store; the
recursion above is what they should be checked against.

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

- **BIGMIN / LITMAX as a cursor.** The recursive decomposition above already realises the run
  count, so the remaining value is the streaming form: seeking a B-tree/trie cursor forward without
  materialising every range first. `box_ranges` is the oracle to check it against — two independent
  methods that must agree.
- **A radix / PATRICIA trie** over the same keys, next to `BTreeMap` — a prefix is a *subtree* there
  rather than a range, path compression eats the long shared prefixes clustered data produces, and
  in-order traversal is curve order for free. (The `radix` already in this repo is radix *sort*, for
  building the GPU LBVH. Unrelated.)

## Sources

- Moon, Jagadish, Faloutsos & Saltz, *Analysis of the Clustering Properties of the Hilbert
  Space-Filling Curve*, IEEE TKDE 13(1), 2001.
- Xu & Tirthapura, *On the Optimality of Clustering Through a Space Filling Curve*, PODS 2012.
- Tropf & Herzog, *Multidimensional Range Search in Dynamically Balanced Trees*, 1981 (BIGMIN /
  LITMAX).
