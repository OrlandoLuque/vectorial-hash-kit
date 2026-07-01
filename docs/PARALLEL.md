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

## Thread-count scaling (measured)

The same bench also sweeps the **pool size** 1..=16 at a fixed workload
(`pop=20000`), so the diminishing returns are visible. `vs 1` is speedup over a
single thread (ideal = N); `eff` is that ÷ N (parallel efficiency). The pool's
worker threads are built **once** and reused every rep — only the *work* is
forked/joined each call — so this is steady-state per-frame cost, not thread
spin-up.

```
                 64 queries      256 queries      1024 queries
 threads   par ms  vs1  eff |  par ms  vs1  eff |  par ms  vs1  eff
       1   0.134  1.0x 100% |  0.678  1.0x 100% |  2.815  1.0x 100%
       2   0.107  1.3x  62% |  0.395  1.7x  86% |  1.506  1.9x  93%
       4   0.070  1.9x  48% |  0.186  3.6x  91% |  0.762  3.7x  92%
       6   0.064  2.1x  35% |  0.152  4.5x  75% |  0.541  5.2x  87%
       8   0.052  2.6x  33% |  0.183  3.7x  46% |  0.437  6.4x  80%
      11   0.053  2.5x  23% |  0.133  5.1x  46% |  0.381  7.4x  67%
      13   0.047  2.8x  22% |  0.154  4.4x  34% |  0.377  7.5x  57%
      16   0.039  3.5x  22% |  0.136  5.0x  31% |  0.400  7.0x  44%
```

Reading it:

- **Efficiency falls as threads rise.** On the heavy frame (1024 culls) 8 threads
  give 6.4× at **80 %** efficiency; pushing to 16 only reaches 7.0× and drops to
  **44 %** — you pay two more cores of fork/memory-bandwidth cost for ~10 % more
  speed. The sweet spot for realistic frames is **~half the cores** (here 6–8).
- **Light frames barely scale.** At 64 culls even 16 threads reach only 3.5×
  (~20 % efficiency) — the fork/join floor dominates.
- This is exactly why the demos expose a **thread slider** rather than always
  using every core: the best count depends on how busy the frame is, and "more"
  is not "better" past the knee.

## Collisions on/off — separation cost + its second-order effect

The siege demos have a live **collision toggle** (`C` key, `siege_sim::
set_separation`) that turns the boids **separation** (the "no two bodies share a
space" push) on/off, so you can A/B both the look and the cost. Two effects:

1. **Direct cost (small).** Separation reuses the *same* `knn(16)` the AI already
   runs for targeting, so turning it off saves only the per-neighbour force
   lookups — a few table reads per unit. On its own, negligible.
2. **Second-order (the interesting one).** With separation *off*, units pile into
   the same cells; the spatial index's leaves overflow and it **deepens**, so
   every `knn` / `cull` / `update_ref` that frame walks a taller tree — i.e.
   allowing collisions can *cost* FPS via index degeneracy, not save it. This is
   the boids/flocking-vs-hard-collision trade the demo was built to show: on the
   ground, flocking alone keeps a tidy distribution cheaply; the failure mode was
   the *dragons* (now fixed — separation is altitude-layered).

Compare live via the on-screen counters (wgpu: `FPS` + the green/red `COL` tag;
macroquad: the `fps`/`collisions` line). Fill in your machine's readings:

```
 wgpu · pop 20000 · <your GPU>          FPS
 collisions ON  (separation, default)   ____
 collisions OFF (overlaps allowed)      ____
```

Expectation on a GPU-bound frame (20k units): the two are close, because the
limiter is the render, not the AI — but watch whether OFF *drops* as the piled
units degenerate the index over a few seconds.

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

## Per-unit AI — the parallel pattern for thousands of agents

The biggest real-world win isn't a batch helper, it's the **per-unit fan-out**:
in a simulation each agent runs its own read queries on the *shared* index every
frame (target = `knn`, perception = `cull`, line-of-fire = `raycast`). Because
every query is `&self` + `Sync` and the agents are mutated disjointly, the whole
AI pass parallelises with one line and **no new API, no contention**:

```rust
units.par_iter_mut().for_each(|u| {
    let target = index.knn(u.pos, 1);          // read-only on the shared index
    let seen   = index.cull(&u.vision());      // …
    u.decide(target, seen);                     // mutate this unit only
});
```

Measured (`examples/parallel_ai.rs`, each unit doing `knn(4)` + a vision-sphere
cull, 16 threads, interleaved min-of-30):

```
 units    serial ms   par ms   speedup
 5 000      5.5        0.52      10.4×
 20 000    31         2.4       12–14×
 80 000   166        14.4       11.5×
 200 000  573        50         11.5×
```

**~11–12× on 16 threads, flat from 5k to 200k agents** — near-linear (the gap to
16× is memory bandwidth, not the algorithm). This is what lets the `siege` demo
run a thousand+ units. The relocation pass (writes) stays serial — its lever is
`update_ref`, not threads. `knn_many` / `knn_many_par` (and `cull_many_par`) are
the batch convenience form for a *homogeneous* set of queries; the per-unit
`par_iter` above is the general pattern when the queries differ per agent.

(On wasm there are no threads — the same code runs serial under
`cfg(target_arch = "wasm32")`; the web `siege` is single-threaded.)

The `siege` demo makes this tangible: its AI pass runs inside a resizable
`rayon::ThreadPool` and a **live thread-count slider** sets `num_threads` from 1
to the core count while it runs, so the fps response *shows* the ~11–12× scaling
(and that one query per frame wouldn't move it — the crossover, on screen). The
slider is native-only; the web build hides it and runs the AI serially.
