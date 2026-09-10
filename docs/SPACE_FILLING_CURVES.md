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

Not implemented here yet. The number above is what says it is worth doing.

## What actually answers the query today

Same bench, section B — an area-of-interest bubble over the same points:

| N | BTree naive span | BTree cell-probe | `MortonGrid3` | `Tree3` |
| ---: | ---: | ---: | ---: | ---: |
| 100 000 | 361.09 µs | 6.17 µs | **5.64 µs** | 6.00 µs |
| 1 000 000 | 6 443.06 µs | **17.90 µs** | 23.16 µs | 32.46 µs |

Two things fall out. The naive span scan is unusable, as A2 predicts. And at a million points the
**ordered store with per-cell probes beats both in-memory structures** — 17.90 µs against the hash
grid's 23.16 and the tree's 32.46. A sorted key store is not only a disk shape.

## Open

- **BIGMIN / LITMAX range decomposition**, so the run count measured in A1 is what a scan actually
  pays. Until then Hilbert's constant is unrealised.
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
