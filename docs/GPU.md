# GPU — what runs on the GPU, and the honest verdict on when it wins

The kit has a full GPU-compute (wgpu / WGSL) strand: two live demos and three
headless benches. This page ties them together and states, from measurement, when
the GPU is worth it and when a parallel CPU path beats it. Numbers are min-of-N on
one box (RTX 4080 SUPER); reproduce commands are inline.

## The demos (native + published to the web)

Both are WebGPU-only (compute + storage buffers) and run in a recent Chrome/Edge —
see the [live index](https://orlandoluque.github.io/vectorial-hash-kit/).

- **`gpu_lbvh_demo`** — the *same* neighbour-count query, three backends flipped
  live: CPU `Tree3` cull · GPU brute · **GPU LBVH** (a BVH from the Morton codes
  the kit already computes, traversed in a compute shader). Identical picture,
  ~**393×** spread on the query kernel (50k pts: CPU 122 ms → GPU LBVH 0.31 ms).
  On-screen per-backend load bars.
- **`gpu_storm`** — a **GPU-resident** collision storm: the *whole* hot loop (grid
  build → contact resolution via a spring-dashpot DEM → integration) lives on the
  GPU, no per-frame CPU↔GPU round-trip. Switch the whole sim CPU↔GPU; `F` toggles
  collision ↔ an influence field; coloured by local density; per-phase GPU-load
  bars. ~**50×** the CPU sim at 150k particles.
- **`adaptive_broadphase`** *(native only)* — the moving-data **maintenance
  crossover**, live. A slider (`,` `.`) sets what **fraction** of the cloud moves;
  every frame it measures BOTH the **CPU keep-index** (`update_ref` only the movers)
  and a full **GPU LBVH rebuild** (Morton→radix→Karras→refit) and draws them as two
  bars. Keep is ~linear in the fraction (it skips the unmoved), the GPU rebuild is
  flat, so the bars **cross** at f\* — and the **ADAPTIVE** controller switches
  keep↔GPU with a **hysteresis** dead-band (`A` auto · `1`/`2` force a mode). The
  live face of `examples/gpu_lbvh_build_bench`.

```bash
cargo run -p vectorial-hash-demos --bin gpu_lbvh_demo       --release
cargo run -p vectorial-hash-demos --bin gpu_storm           --release
cargo run -p vectorial-hash-demos --bin adaptive_broadphase --release   # ADAPT_N
```

## The benches (headless, measured)

| bench | what | headline |
| --- | --- | --- |
| `gpu_spatial_bench` | GPU brute / GPU LBVH / CPU, + the per-frame **rebuild-vs-keep** verdict for *moving* data | kernel ~100–400×; moving data → **fraction-dependent**: keep wins sparse motion, GPU rebuild wins dense (f\* ≈ 30 %→2.8 % as N grows) |
| `gpu_visibility_bench` | GPU **line-of-sight** over STATIC occluders (segment-vs-AABB LBVH traversal), verified == CPU `segment_hit` (Δ 0) | **~1380×** the serial CPU; 1 ms one-time build — the *clean* GPU case |
| `gpu_sort_bench` | GPU **bitonic sort** of Morton codes, verified == CPU sort | **slower** than the CPU sort (log² work + dispatch/pass) — the honest negative that motivated the radix |
| `gpu_radix_bench` | GPU **stable LSD radix sort** of Morton codes (hierarchical scan), **4-bit×8 vs 8-bit×4** widths, verified == CPU sort | 8-bit/4-pass is **1.3–1.8× faster** than 4-bit/8-pass → **8–23× faster** than `sort_unstable` (1M **8.1×** · 4M 16.8× · **8M peak 22.6×** · 16M 17.2×) — fewer global passes, the portable Onesweep step |
| `gpu_lbvh_build_bench` | the **whole LBVH built on the GPU** — Morton → radix → Karras → refit — verified by traversal-vs-brute | **1 M-point BVH in ~4.4 ms/frame** (262k 2.28 · 4M 13.0), all GPU-resident — the on-GPU per-frame rebuild |

```bash
cargo run -p vectorial-hash-demos --example gpu_spatial_bench    --release --features parallel   # GPU_N/M/R/CLUSTER
cargo run -p vectorial-hash-demos --example gpu_visibility_bench  --release                        # VIS_OCC/VIS_SEG
cargo run -p vectorial-hash-demos --example gpu_sort_bench        --release                        # SORT_N (bitonic)
cargo run -p vectorial-hash-demos --example gpu_radix_bench       --release                        # SORT_N (radix)
cargo run -p vectorial-hash-demos --example gpu_lbvh_build_bench  --release                        # LBVH_N (full build)
```

## The verdict — when the GPU wins (measured, not assumed)

The GPU broad-phase *query kernel* is ~100–400× the serial CPU cull. But that
headline is the kernel only. The honest rule:

- **✅ Static / rebuild-anyway / query-dominated → GPU.** No per-frame rebuild to
  pay for, so the kernel win is real. **Line-of-sight over static occluders is the
  cleanest case** (~1380×; the occluder BVH is built once). Big interest bubbles /
  thousands of clients against a static world are the same shape.
- **✅ Whole-loop GPU-resident → GPU.** When positions/velocities never leave the
  GPU (a particle/DEM/flocking sim of one uniform rule), everything — grid, query,
  forces, integration — is on-GPU with no round-trip. `gpu_storm` = ~50× the CPU.
- **◑ Moving data → it depends on the moving FRACTION** (now that the GPU-side
  build is real — `gpu_lbvh_build_bench`, ~4.4 ms/frame at 1 M, all GPU-resident).
  - **Most of the cloud moves → GPU rebuild.** All-moving, per frame: GPU rebuild
    vs the CPU keep-index's *best case* (serial `update_ref` over all, ~no
    relocations): **262k 2.3 vs 5.7 ms · 1 M 4.4 vs 48.6 ms · 4 M 13 vs 452 ms** — a
    serial maintain-*all* pass loses to the parallel build (2.5× at 262k, **11× at
    1 M**, 35× at 4 M now the scan is hierarchical). (Before the radix, the GPU-side
    build didn't exist — the bitonic sort lost to the CPU.)
  - **Only a fraction moves → CPU keep-index.** Its real edge is *skipping* the
    items that didn't move (the horde demo's dormant carpet keeps for ~nothing);
    `ItemRef` + the moved-only sync + `cull_many_par` is still the lever there.
  - **Adaptive hybrid (measured, viable).** Keep below a crossover fraction **f\***,
    GPU rebuild above, with a **hysteresis** dead-band so it doesn't thrash. Since
    keep skips the unmoved, keep-cost ≈ linear in the moving fraction while GPU is
    flat, so they cross at a clean **f\* ≈ 30 % (262k) · 7.6 % (1 M) · 2.8 % (4 M)** —
    and f\* *drops sharply* with N (the serial keep pass scales worse). On a wave
    whose moving fraction ramps 0→1→0 the adaptive policy beats **both** pure
    strategies (1 M: **504 ms** vs pure-GPU 527 vs pure-keep 4163); a ±25 %-of-f\*
    dead-band roughly **halves the switch count** when the load hovers at f\* (110→64
    over 200 noisy frames). `gpu_lbvh_build_bench` prints the f\* sweep + the
    wave/noise comparison.
- **❌ The game sims stay on the CPU.** Their dominant cost is the *branchy*
  per-agent `decide` (FSM, targeting, morale) — warp-divergent, GPU-hostile — not
  the spatial query. See `PERF_NOTES.md`.
- **❌ Uniform-density collision → a grid, not a BVH.** `gpu_storm`'s hash grid is
  O(1)-per-cell and already the right broad-phase for a uniform storm; an LBVH buys
  nothing there. A BVH's edge is **non-uniform** density or **culling by shape**
  (ray / frustum / sphere), not uniform pair-finding — so we did *not* retrofit the
  new GPU LBVH build into `gpu_storm` (it would be a fancier tool for a worse fit).

## The GPU-side LBVH build — DONE (measured)

The one thing that would push the *moving* broad-phase crossover past 1 M: build
the BVH on the GPU each frame instead of the CPU. **`gpu_lbvh_build_bench` does the
whole build GPU-resident** — Morton → stable key-value radix → Karras hierarchy →
atomic bottom-up AABB refit, no CPU round-trip — and **verifies it** by traversing
the GPU-built tree on the CPU vs brute force (a pass ⇒ hierarchy *and* AABBs
correct). Measured (RTX 4080 SUPER, min-of-7, whole build per frame):

| points | build/frame (4-bit → 8-bit sort) | throughput |
| --- | --- | --- |
| 262 k | 2.28 → **1.58 ms** | 166 Mpts/s |
| 1 M | 4.40 → **3.7 ms** | 273 Mpts/s |
| 4 M | 12.98 → ~14 ms (flat) | 283 Mpts/s |

So a 1 M-point BVH **rebuilds on the GPU in ~3.7 ms/frame**, verified correct — down
from 8.4 ms (hierarchical scan) then 4.4 ms (see the sort row), now the build's own
key-value radix runs the **8-bit / 4-pass** width. The width helps where the *sort*
is a big slice of the build (262 k **1.4×**, 1 M **1.2×**); at 4 M the sort is a
smaller fraction (Karras + refit dominate) and the 256-bucket scan overhead makes it
a **wash** — an honest, size-dependent win. This
is what lets a *moving* broad-phase rebuild on the GPU each frame rather than lean
on the CPU keep-index — the rebuild-vs-keep crossover itself is in
`gpu_spatial_bench` / PARALLEL.md. Further headroom: the sort is **hierarchical**
(reduce/scan/add) **+ 8-bit/4-pass** in both the standalone `gpu_radix_bench`
(1.6–1.8× the 4-bit width, 8–17× the CPU) and now the build; a *true* single-pass
**Onesweep** (decoupled-lookback) is the remaining lever — **and now measured to be
un-implementable in portable WGSL** (`gpu_onesweep_scan_bench`): the look-back needs a
**device-scope** acquire/release between one spinning thread and other workgroups, but
WGSL's only ordering primitive, `storageBarrier()`, is **workgroup-scope and
uniform-only** (no device fence). On native NVIDIA the scan returns a **wrong** prefix
(a tile reads a predecessor's flag before its aggregate is visible) with **no**
spin-timeout — so it's a *memory-ordering* wall, not a forward-progress one, and the
hierarchical + 8-bit/4-pass radix is the honest portable ceiling. On the **node
layout** (both measured on the CPU, but they govern
the GPU buffer too): quantised **u16** boxes are **1.6× smaller and exact** (round
outward + test the exact leaf point — `compressed_bvh_bench`) but only a *footprint*
win for the binary tree; going **wide (8-ary)** with an SoA/SIMD 8-box test is the
real **~2× cull** win *over a naive binary BVH* (`wide_bvh_bench`), shrinking the arena
~5×. **But reference-checked against the kit's shipping `Tree3`/`Octree3` cull it is
only ~1.0×** (Tree3 is even a hair faster at 1 M) — the kit's tuned arena already sits
at the wide node, so a static wide BVH is **not worth promoting**; the SoA-wide layout
stays a **GPU-side** win (where the alternative is a naive kernel).

## Design notes

- The BVH is built from the **same Morton encoding** the in-memory structures use
  (`morton3`), so nothing is bespoke to the GPU.
- **Timestamps** (`TIMESTAMP_QUERY`) drive the per-phase meters natively; browsers
  gate the feature, so the on-web meters read 0 (handled, no crash).
- **Buffer limits on web:** `gpu_storm` uses a smaller grid (96³) on wasm to fit
  WebGPU's 128 MiB `maxStorageBufferBindingSize`; the HTML shells carry a
  `requestDevice` shim that strips limits the adapter doesn't advertise.
