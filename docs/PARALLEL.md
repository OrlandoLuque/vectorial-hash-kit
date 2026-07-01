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

// Feature `parallel` — build a whole tree from all items at once, top-down
// partition fanned over threads (the per-frame-rebuild lever; see below):
let tree = Tree3::bulk_load_par(bounds, item_limit, items); // serial: bulk_load
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

Compare live via the on-screen counters (wgpu: `FPS`/`THR` + the green/red `COL`
tag; macroquad: the `fps`/`collisions` line).

**Observed (user, wgpu, ~20k units): turning collisions OFF *lowers* FPS.** This
confirms the second-order effect — the direct saving (a few force lookups) is
dwarfed by the cost of the units piling into shared cells and deepening the
index, so every `knn`/`cull`/`update` that frame walks a taller tree. The
counter-intuitive headline: **collision avoidance here is a net perf *win*, not a
cost**, because keeping bodies spread keeps the spatial index shallow and its
queries fast. It's a neat demonstration that *good spatial distribution is a
performance feature*, not just a visual one.

## Siege: CPU-only vs full-pipeline FPS (headless, by thread count)

Two headless benchmarks measure the real per-frame budget of the `siege_wgpu`
demo at **20 000 units, mid-clash** — both warm the sim to the clash first (the
armies start apart; the early spread frames are cheap and unrepresentative), then
sweep the thread count. Measured on an RTX 4080 SUPER at 1600×1000:

- `cargo run -p vectorial-hash-demos --example siege_cpu_bench --release` — the
  **CPU sim only** (index rebuild + parallel `decide` + serial `apply`), no GPU.
- `SIEGE_BENCH=1 cargo run -p vectorial-hash-demos --bin siege_wgpu --release` —
  the **full pipeline offscreen**: sim + build instances + render to a texture,
  **no window, no present/vsync**, blocking on the GPU each frame.

```
 threads   CPU-only fps   offscreen (sim+render) fps
       1        24.2                14.8
       2        39.8                21.0
       4        60.2                26.8
       6        71.5                29.5
       8        76.2            ~31  (peak)
      12        81.7                29.1
      16        91.8                29.5
```

Reading it:

- **CPU sim alone**: 24 fps (1 thread) → **92 fps (16 threads)**, scaling 3.8×.
  Only `decide` parallelises; the serial index-rebuild + `apply` cap it (Amdahl),
  so efficiency falls from 82 % (2 threads) to 24 % (16) — the sweet spot is
  ~half the cores.
- **Add the GPU render** and the ceiling drops to **~15–31 fps**, peaking at
  **~7–8 threads** then *plateauing*: the render is now a big **fixed** per-frame
  cost, so past ~8 threads the CPU `decide` is no longer the bottleneck and more
  cores do nothing. At 20 k, the geometry (skinned units + horses + forest) makes
  the GPU a real limiter.
- **This offscreen number is a conservative *lower bound*.** It blocks CPU↔GPU
  each frame (`device.poll(Wait)`), so it measures `CPU + GPU` *summed*. The real
  windowed demo **pipelines** them (the CPU builds frame N+1 while the GPU renders
  N), so on-screen FPS is closer to `1/max(CPU, GPU)` — higher. (That's why the
  demo shows ~48 fps at 4 threads where this blocked bench reads ~27.)

The takeaway for the thread slider: **more threads is not always better** — past
the point where the GPU (or the serial sim tail) dominates, extra cores are idle
overhead. The slider lets you find the knee for your machine + unit count.

## Parallel bulk-load — the rebuild lever (measured)

`decide` parallelises; the per-frame **index rebuild** doesn't — it's the serial
Amdahl tail above. `Tree3::bulk_load` / `bulk_load_par` attack exactly that: one
**top-down partition** (longest-axis-midpoint, the same rule `divide` uses
incrementally) instead of N root-descending `insert`s. `bulk_load_par` fans the
partition out over rayon (`join` per split), leaving a cheap serial arena flatten.

First, the rebuild step *in isolation* (`bulk_load_bench`, 15 542 units mid-clash,
16-core / RTX 4080 SUPER box):

```
                  strategy   us/rebuild   vs insert
  clear + insert (current)      3639         1.00x
        bulk_load (serial)      4433         0.82x   ← SLOWER
     bulk_load_par (2 thr)      3125         1.16x
     bulk_load_par (4 thr)      2339         1.56x
     bulk_load_par (8 thr)      2313         1.57x
    bulk_load_par (16 thr)      2300         1.58x
```

Two honest findings:

- **Serial `bulk_load` *loses* (0.82×)** to `clear()` + `insert`. The insert loop
  reuses the arena (`clear` keeps capacity) and touches memory linearly; the
  recursive partition allocates two fresh child `Vec`s at every internal node —
  that allocation churn costs more than the root-descents it saves. So the serial
  path is **not** worth it; the win is purely from parallelism.
- **`bulk_load_par` wins, then plateaus** at ~1.58× (4+ threads). The serial
  flatten tail + allocator contention cap it — more than 4 threads on *this* step
  buys nothing.

End-to-end, though, the frame is rebuild + `decide` + `apply`, and the rebuild's
share *grows* with thread count (decide gets cheaper, the serial tail doesn't). So
swapping the rebuild for `bulk_load_par` lifts the whole CPU-fps ceiling more the
more cores you have (`siege_cpu_bench`, full CPU frame, 20 k units):

```
 threads   insert fps   bulk_load fps   gain
       1       24.3          24.3        1.00x
       4       59.4          63.6        1.07x
       8       79.0          87.9        1.11x
      13       87.3         100.4        1.15x
      16       86.4          98.7        1.14x
```

**No benefit single-threaded** (nothing to parallelise; the serial `bulk_load` it
falls back on is a wash-to-slight-loss), rising to **~1.14–1.15× at 12–16
threads** — 87 → 100 fps. It's a lever for exactly the regime where the serial
rebuild had become the bottleneck.

Wiring: the siege binaries use `bulk_load_par` on **native builds with
`--features parallel`** (the perf build); the default native run and the wasm
build keep the serial `clear()` + `insert` loop (wasm has no threads, and the
serial `bulk_load` would only lose). Handle `i` still addresses `items[i]`, so
`ItemRef` stays valid across a bulk rebuild.

## Why not parallelise relocation / the Morton build too?

- **Relocation (`update_ref`):** a single item's move is an ascend-to-LCA +
  re-descend — a dependent structural mutation, not a `par_iter`. The lever there
  is doing *fewer* of them (`ItemRef`), not threading each one.
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
