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

```bash
cargo run -p vectorial-hash-demos --bin gpu_lbvh_demo --release
cargo run -p vectorial-hash-demos --bin gpu_storm     --release
```

## The benches (headless, measured)

| bench | what | headline |
| --- | --- | --- |
| `gpu_spatial_bench` | GPU brute / GPU LBVH / CPU, + the per-frame **rebuild-vs-keep** verdict for *moving* data | kernel ~100–400×; but moving data → **parallel CPU keep-index wins at 1 M** |
| `gpu_visibility_bench` | GPU **line-of-sight** over STATIC occluders (segment-vs-AABB LBVH traversal), verified == CPU `segment_hit` (Δ 0) | **~1380×** the serial CPU; 1 ms one-time build — the *clean* GPU case |
| `gpu_sort_bench` | GPU **bitonic sort** of Morton codes, verified == CPU sort | **slower** than the CPU sort (log² work + dispatch/pass) — the honest negative that motivated the radix |
| `gpu_radix_bench` | GPU **stable 4-bit LSD radix sort** of Morton codes, verified == CPU sort | **at parity** with `sort_unstable` (262k 1.01× · 4M 1.09×) — correct+stable, past the bitonic; the sort primitive for the GPU-side build |

```bash
cargo run -p vectorial-hash-demos --example gpu_spatial_bench   --release --features parallel   # GPU_N/M/R/CLUSTER
cargo run -p vectorial-hash-demos --example gpu_visibility_bench --release                        # VIS_OCC/VIS_SEG
cargo run -p vectorial-hash-demos --example gpu_sort_bench       --release                        # SORT_N (bitonic)
cargo run -p vectorial-hash-demos --example gpu_radix_bench      --release                        # SORT_N (radix)
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
- **❌ Moving data, one query slice offloaded → CPU keep-index.** A moving cloud
  forces the BVH to be **rebuilt every frame**; that rebuild (an *N log N* sort)
  overtakes the linear keep-index maintain, and at 1 M the parallel CPU keep-index
  **beats** GPU LBVH per frame (measured 1.30×). The lever for moving data is the
  `ItemRef` keep-index + `cull_many_par`, not the GPU.
- **❌ The game sims stay on the CPU.** Their dominant cost is the *branchy*
  per-agent `decide` (FSM, targeting, morale) — warp-divergent, GPU-hostile — not
  the spatial query. See `PERF_NOTES.md`.

## The GPU-side LBVH build (open)

The one thing that would push the *moving* broad-phase crossover past 1 M is a
GPU-side build (build the BVH on the GPU each frame instead of the CPU). The
**sort half is now done**: `gpu_radix_bench` is a stable all-GPU 4-bit LSD radix,
verified == CPU and **at parity** with `sort_unstable` (vs the bitonic's ~2×
loss). Remaining for the full build: an **Onesweep** decoupled-lookback scan (the
single-workgroup exclusive scan is the current radix bottleneck), then a GPU
**Karras** hierarchy + **AABB refit** on top — with the sort GPU-resident, the
whole build avoids the readback. Until that lands, GPU is for the static /
resident / query-dominated cases above.

## Design notes

- The BVH is built from the **same Morton encoding** the in-memory structures use
  (`morton3`), so nothing is bespoke to the GPU.
- **Timestamps** (`TIMESTAMP_QUERY`) drive the per-phase meters natively; browsers
  gate the feature, so the on-web meters read 0 (handled, no crash).
- **Buffer limits on web:** `gpu_storm` uses a smaller grid (96³) on wasm to fit
  WebGPU's 128 MiB `maxStorageBufferBindingSize`; the HTML shells carry a
  `requestDevice` shim that strips limits the adapter doesn't advertise.
