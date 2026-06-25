# Parallelism: where threads pay, and where they don't

The library is single-threaded by default and stays that way unless you opt in.
The `parallel` Cargo feature adds rayon-backed **batch culls** — and nothing
else, on purpose. This note explains *why* that is the only thing parallelised,
and the measured crossover for when it's worth it.

## What parallelises cleanly, and what doesn't

A frame of the critters demo is two kinds of work:

1. **Relocation** — move every item and update its slot in the index. This
   **mutates** the tree (nodes split, merge, items hop leaves). Parallelising
   mutation means either locking shared nodes (contention eats the win) or
   partitioning the world into disjoint sub-trees (cross-region moves break it).
   So relocation stays **serial** — and the right lever there is algorithmic, not
   threads: the stable [`ItemRef`](THREE_D.md) handle makes each update O(1)
   (no predicate leaf scan), which is the real maintain win. See
   `THREE_D.md` § "The fix: Stable ItemRef".

2. **Querying** — each cull is **read-only** against the shared index. Many
   independent culls (e.g. one attack volume per attacker in a combat sweep) are
   embarrassingly parallel: no contention, no locks, just fan the queries over a
   thread pool. **This** is what the `parallel` feature accelerates.

That asymmetry is the whole story: *reads fan out, writes don't.*

## The API

```rust
// Always available, serial — the ergonomic batch form:
let hits: Vec<Vec<&T>> = tree.cull_many(&shapes);

// Feature `parallel`, rayon-backed — same result, fanned over threads:
let hits: Vec<Vec<&T>> = tree.cull_many_par(&shapes);
```

`cull_many_par` exists on all five structures (`Tree`, `QuadTree`,
`IntegerTree`, `Tree3`, `Octree3`, `MortonGrid3`) and requires `T: Sync`. It is
gated behind the feature so rayon stays out of the dependency tree (and out of
the wasm build, which has no OS threads) unless you ask for it:

```toml
vectorial-hash = { version = "0.1", features = ["parallel"] }
```

## The crossover (measured)

`critters3d_headless --parallel` (16 hardware threads, world 512³, vision r=36,
binary `Tree3`, item_limit 8) times serial `cull_many` vs parallel
`cull_many_par` over a grid of index size × query count:

```
      pop  queries |  serial ms     par ms  speedup | verdict
     2000        4 |     0.0021     0.0166     0.13  | serial wins (fork cost)
     2000       16 |     0.0052     0.0244     0.21  | serial wins (fork cost)
     2000       64 |     0.0254     0.0235     1.08  | tie
     2000      256 |     0.0971     0.0723     1.34  | parallel wins
     2000     1024 |     0.4828     0.1583     3.05  | parallel wins
    20000        4 |     0.0048     0.0321     0.15  | serial wins (fork cost)
    20000       16 |     0.0197     0.0136     1.45  | parallel wins
    20000       64 |     0.0942     0.0610     1.55  | parallel wins
    20000      256 |     0.5690     0.1477     3.85  | parallel wins
    20000     1024 |     2.5152     0.3948     6.37  | parallel wins
   100000        4 |     0.0111     0.0376     0.30  | serial wins (fork cost)
   100000       16 |     0.0703     0.0396     1.78  | parallel wins
   100000       64 |     0.3823     0.1013     3.77  | parallel wins
   100000      256 |     1.7459     0.2636     6.62  | parallel wins
   100000     1024 |     7.2698     0.8878     8.19  | parallel wins
```

Reading it:

- **≤ 4 queries — never parallelise.** The fork/join costs ~15–30 µs; a handful
  of culls finish in less than that, so the threads are pure overhead (down to
  0.13× — an 8× *slowdown*). One frustum cull per frame? Stay serial.
- **16 queries — depends on the index.** It pays at 20k+ points (the per-cull
  work is big enough to amortise the fork) but not at 2k (too little work).
- **64+ queries — parallelise always.** The win grows with both axes, reaching
  ~8× at 1024 queries over 100k points (≈ the thread count, as expected).

Rule of thumb: **parallelise when `queries × per-cull-work` comfortably exceeds
the ~tens-of-µs fork cost** — practically, ≥ 64 independent queries, or ≥ 16 over
a large index. Below that, `cull_many` (serial) is the right call, and the demo's
per-frame single frustum cull would only be slowed down by threads.

## Why not parallelise the build / relocation too?

- **Build (`insert` loop):** a tree build is a sequence of dependent structural
  mutations; parallel insertion needs lock-free nodes or a parallel
  sort-then-link (bulk-load) pass. Worth exploring for static datasets, but it's
  a different algorithm, not a free `par_iter`. Out of scope for this feature.
- **Morton rebuild:** the per-point cell code *is* independent (computing
  `morton3(cell_of(p))` for every point parallelises), but the bucket inserts
  into the shared `HashMap` serialise. A parallel build would compute codes in
  parallel, then group — a future `from_positions_par`. Noted, not yet built.

The honest summary: the library hands you a clean parallel primitive for the one
shape of work that actually parallelises (batch reads), with the crossover
measured so you don't pay threads where they lose. Everything else is left
serial because that's genuinely faster — or because making it parallel means a
different algorithm, not a flag.
