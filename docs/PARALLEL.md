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
// partition fanned over threads (the per-frame-rebuild lever; see below).
// Exists on `Tree3`, `Octree3` (8-way fan) and the 2D `Tree`:
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
sweep the thread count. Measured on an RTX 4080 SUPER at 1600×1000.

> These numbers were taken on the old `clear()`+`insert` **rebuild** path (the
> `insert` column of the three-way table below). The demo now *keeps* the index
> (`sync_index`), which raises the CPU-only figures ~1.06–1.4× — see § "The
> per-frame index: rebuild vs keep". The scaling *shape* (and the GPU-bound
> plateau) is unchanged; keep just shifts the CPU curve up.

- `cargo run -p vectorial-hash-demos --example siege_cpu_bench --release` — the
  **CPU sim only** (index maintenance + parallel `decide` + serial `apply`), no
  GPU. Reports all three maintenance strategies (insert / bulk / keep).
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

## The per-frame index: rebuild vs keep (measured)

`decide` parallelises; the per-frame **index maintenance** doesn't — it's the
serial Amdahl tail above. So it's the lever worth chasing. Three strategies,
measured end-to-end on the full CPU frame (`siege_cpu_bench`, 20 k units
mid-clash, 16-core / RTX 4080 SUPER):

- **insert** — `clear()` + an `insert` per unit. What the demo used to do.
- **bulk** — `Tree3::bulk_load_par`: one **top-down partition** (longest-axis
  midpoint, the rule `divide` uses) fanned over rayon (`join` per split) instead
  of N root-descents. A parallel *rebuild*.
- **keep** — don't rebuild: keep the tree and `update_ref` each unit to its new
  spot (**O(1) if it stayed in its leaf**; relocate only on a boundary cross),
  `remove_ref` deaths, `insert_ref` respawns.

```
 threads   insert fps    bulk fps    keep fps  | bulk/ins  keep/ins
       1       24.4         24.3        25.9    |   0.99x     1.06x
       4       60.2         65.2        73.8    |   1.08x     1.23x
       8       78.7         90.8       105.0    |   1.15x     1.33x
      13       89.1        102.3       124.3    |   1.15x     1.39x
      16       91.7        101.0       127.0    |   1.10x     1.38x
```

**`keep` wins outright — 1.06× single-threaded, up to ~1.4× at 12–16 threads**
(91 → 127 fps). Two things make it win:

1. **Less work.** Units drift a *fraction of a cell* per frame, so most stay in
   their leaf and `update_ref` short-circuits to an O(1) position write — far
   cheaper than reinserting all N from the root. The tree still `divide`s/merges
   on the few boundary crossers, so it stays balanced enough that the `decide`
   queries **don't** slow down (the drift I worried about didn't materialise).
2. **It shrinks the serial tail.** Cheaper maintenance = smaller Amdahl serial
   fraction, so the parallel `decide` dominates more and thread-scaling improves
   — which is why the win *grows* with core count. No threads needed either, so
   **wasm gets the single-thread win for free**.

Verified a **byte-for-byte-identical index to `clear()`+`insert` every frame**
(`siege_cpu_bench` runs both in lockstep and asserts equal item counts; a handful
of units that sink into deep craters below `y=0` fall out of the root AABB and are
dropped by *both* — pre-existing, identical for each).

Both siege binaries now use `keep` (the shared `siege_sim::sync_index`) on every
build — native and wasm. The 3D `critters` demo already relocated in place
(`update_ref`), so this brings siege in line with it.

**Where `bulk_load` still fits.** `bulk` is the *middle* option — a parallel
rebuild beats a serial one (1.1–1.16×) but loses to not rebuilding at all. Keep it
for the case `keep` can't serve: a **from-scratch build of a static or
fully-churned set** (no prior tree to maintain, no stable handles) — e.g. a
one-shot spatial join, or the first build before a `keep` loop. The serial
`Tree3::bulk_load` is a wash-to-slight-loss vs `insert` (per-node `Vec`
allocations lose to the arena-reuse insert loop); the win is purely `bulk_load_par`
fanning the partition over threads (plateaus ~1.58× at 4+ threads — the serial
flatten tail + allocator contention cap it). Handle `i` addresses `items[i]`, so
`ItemRef` survives a bulk build.

**The GPU-side rebuild — and an adaptive keep↔GPU switch (measured 2026-07-17).**
"keep beats rebuild" holds *when a small fraction moves per frame* — the demos'
regime, where `keep` skips the unmoved. Push to a **mostly-moving** cloud and it
flips: a serial `update_ref` over *all* N loses to a parallel from-scratch build.
And that build can now be the **GPU** — `gpu_lbvh_build_bench` builds a whole LBVH
(Morton→radix→Karras→refit) **GPU-resident in ~8 ms/frame at 1 M**, verified by
traversal-vs-brute. Since keep skips the unmoved, keep-cost ≈ linear in the moving
fraction while the GPU rebuild is flat, so they cross at **f\* ≈ 16 % (262k) /
12–14 % (1 M)** moving (and f\* *drops* with N). An **adaptive** policy — keep below
f\*, GPU rebuild above, with a **hysteresis** dead-band to stop thrashing at the
boundary — beats *both* pure strategies on a varying load (1 M wave: adaptive 1020
ms vs pure-GPU 1099 vs pure-keep 5227) and roughly halves the mode switches when
the load hovers at f\* (110→64). Numbers + the GPU-vs-CPU verdict: [GPU.md](GPU.md).
So the full rule: **sparse motion → CPU keep · mostly-moving huge clouds → GPU
rebuild · varying → adaptive with hysteresis.**

## Parallel builds: why the k-d tree fans out better than the midpoint tree

`KdTree3::from_items_par` (feature `parallel`) splits with `rayon::join` while the slice
is above 4096 points. Because the serial build already emits *parent, then left subtree,
then right subtree*, a subtree can be built into its own node vector and spliced in with
an id shift — so the parallel build produces a **node-for-node identical** tree, which is
what the test asserts (same node count, same leaf boxes and ranges, same depth, same cull
and k-NN answers) rather than merely an equivalent one.

200 000 points, 16 threads, min-of-5 inside the bench, **median of 3 passes** of it:

| build | serial | parallel | speed-up |
| --- | ---: | ---: | ---: |
| `KdTree3` uniform | 20.13 ms | **6.03 ms** | **≈3.4×** |
| `KdTree3` clustered | 20.57 ms | **6.02 ms** | **≈3.3×** |
| `Tree3::bulk_load` uniform | 40.49 ms | 25.77 ms | 1.56× |
| `Tree3::bulk_load` clustered | 55.86 ms | 29.98 ms | 1.86× |
| `KdTree2` (2D, clustered) | 7.96 ms | **2.85 ms** | **2.79×** |

**The median split parallelises better than the midpoint split, and it's not an accident.**
`rayon::join` is only as fast as its slower half, so a fan-out is worth what its *load
balance* is worth:

- A **median** split hands each side exactly `n/2` points, by construction. Every fork is
  balanced, at every level, whatever the data looks like.
- A **midpoint** split halves the *box*, not the points. On clustered data one side can
  get almost everything, and that fork is then a serial tail with idle threads beside it.

So the structure whose build is the expensive one (the median selection) is also the one
that recovers the most from threads. Note the end state: the parallel k-d build (6.0 ms)
is **faster than `LinearOctree3`'s serial build** (13.6 ms), which is the fastest serial
build in the comparison — "the k-d tree builds slowly" stops being true once you have
cores to spend, while its query advantage on clustered data (cull **≈2.0–2.3×**, k-NN
1.67× vs `Tree3`) is unchanged. Reproduce with:

```bash
cargo run -p bench-runner --release -- --group kd --repeat 3   # both benches, 3 passes
cargo run -p vectorial-hash --example kdtree3_bench --release --features parallel
```

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
