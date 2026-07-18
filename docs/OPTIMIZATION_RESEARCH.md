# Optimisation strategies — a literature scan (with sources)

A `deep-research` scan (2026-07-11) for techniques the kit might be missing, CPU
and GPU, favouring recent work with measured trade-offs. 5 search angles → 23
sources → 25 falsifiable claims adversarially verified (17 confirmed; 8 left
unverified — several because the run hit a spend limit mid-verification, not
because they were refuted). Auto-generated draft — **curate before treating as
gospel**; the confirmed items are load-bearing.

## Headline

The kit is **already close to the state of the art**; the highest-value gaps are
concrete. Our own measured results are *validated*, not contradicted, by the
literature — and the literature names the exact next steps.

## GPU

- **Onesweep radix sort** — the right primitive for the LBVH Morton sort (fixes
  our "bitonic loses to CPU" result). A single-pass LSD radix sort: ~2n vs ~3n
  global memory traffic per digit pass, **~1.5× over CUB** (the prior SOTA, same
  author); radix needs 4–8 passes vs bitonic's dozens. *Confirmed 3-0.*
  [Adinets & Merrill, 2022](https://arxiv.org/abs/2206.01784). Now folded into CUB.
  → **This is the GPU-side-build unblock** (`gpu_sort_bench` found bitonic wanting).
  Caveat: hierarchy gen + AABB fit are still ~40–50 % of an LBVH build (Karras).
- **Keep Morton codes 32-bit.** GPU radix sort is bandwidth-bound; **64-bit codes
  are ~3× slower** (128-bit ~9.7×). *Confirmed 3-0.* Measured on an RTX 4080 SUPER
  (our GPU class) by [NexusBVH](https://github.com/StokastX/NexusBVH); corroborated
  by [Stehle & Bast, 2016](https://arxiv.org/abs/1611.01137). → prefer 32-bit unless
  resolution demands more.
- **PLOC++ / H-PLOC (NexusBVH)** — higher-quality BVH, but **does NOT beat LBVH on
  build time** (reaches ~0.56–1.03× of a fast 4D LBVH; H-PLOC "within 15 % of LBVH
  build"), buying 5–18 % better *traversal*. *Confirmed 3-0.*
  [PLOC++, HPG 2022](https://dl.acm.org/doi/10.1145/3543867) ·
  [H-PLOC](https://gpuopen.com/download/HPLOC.pdf). → **our LBVH choice is validated
  for build-bound / rebuild-heavy** work; consider H-PLOC only if *traversal*
  throughput is the bottleneck.
- **Warp-cooperative traversal (CoopRT)** — 2.15× geomean, up to 5.11×. *Confirmed
  3-0.* But it leans on ray-tracing HW/warp intrinsics → **motivation, not directly
  portable** to a generic wgpu compute path.

## Dynamic BVH — the direct analogue of our keep-index finding

- **Refit + tree rotations** is the BVH's "keep-index": **~1.6–2× plain-refit cost,
  yet ~an order of magnitude cheaper than a parallel binned rebuild**, and it folds
  into the *same* parallel post-order pass "for free", keeping SAH quality near an
  ideal rebuild except in pathologically chaotic scenes. Refit alone: 200k tris
  **< 1 ms** vs **~40 ms** rebuild. *Confirmed 3-0.*
  [Kopta et al., I3D 2012 (Utah UUCS-11-002)](https://www.sci.utah.edu/~thiago/papers/BVH_rotations-tech_report_UUCS-11-002.pdf).
  → strongly reinforces **keep-and-repair over rebuild** — the same lesson our
  `ItemRef`/decision-map already taught for the trees, now for BVH quality. If the
  kit ever grows a persistent (non-rebuilt) LBVH for dynamic data, add rotations to
  the refit. (Plain refit-only degrades quality catastrophically over time —
  rotations are what keep it usable. *Confirmed.*)

## CPU / memory-wall levers (orthogonal, algorithm-agnostic)

- **Compressed / quantized wide-BVH nodes** — **35–60 % of the uncompressed
  footprint**, cutting bytes touched per traversal step (the memory wall we keep
  hitting). *Confirmed 3-0.* → **built + measured** (`compressed_bvh_bench`): u16
  nodes are **1.6× smaller and EXACT** (round outward + exact leaf test), but the
  cull-speed win needs a *wide* node — the binary layout's footprint drop doesn't
  move latency. See "Implemented + measured".
- **Cache-oblivious tree layout** (van-Emde-Boas-style) — **26 %–300 % typical, up
  to ~2600 % peak** speedup, **zero algorithm change** — purely reorder nodes in the
  arena for locality. *Confirmed 3-0.* → cheap, high-upside for the arena trees.
- **AVX-512 SIMD broad-phase** and **half-precision BVH descriptors** — claims were
  *not* verified (the run abstained/hit the spend limit), so treat as **unconfirmed
  leads**, not results. SIMD's win is workload-dependent (the memory wall can eat it
  — as our boid-table result showed).

## Not re-examined (already measured in the kit)

Hilbert-vs-Morton locality, BIGMIN/UB-tree range queries, layered coarse-skip,
keep-index vs rebuild, SoA vs AoS batch — the kit already has measured studies
(`STORAGE_AND_SCALE.md`, `THREE_D.md`, `PERF_NOTES.md`).

## Suggested next steps (ranked)

1. **GPU radix sort** → **DONE** (a stable LSD radix, see below): hierarchical scan
   **+ an 8-bit/4-pass width** (vs 4-bit/8-pass) now runs it at **8–17× the CPU** at
   scale. A *true* single-pass **Onesweep** (decoupled-lookback) is blocked by
   WebGPU's lack of an inter-workgroup forward-progress guarantee (it can deadlock) —
   8-bit/4-pass is the portable step in that direction. Keep 32-bit Morton codes.
2. ~~**Cache-oblivious arena layout**~~ → **DONE** as `Tree3::compact()` (see below).
3. ~~**Compressed wide-BVH nodes**~~ → **DONE + measured** as
   `examples/compressed_bvh_bench` (see below). *Correction (2026-07-18): an
   earlier note here claimed quantised AABBs "break exactness" — **that was wrong**.
   Round the box outward (min↓/max↑) and test the **exact point at the leaf**, and
   the answer is **identical to brute force** (conservative internal boxes only ever
   cost a few extra node visits, never a wrong result). Measured: **1.6× smaller
   nodes, exact, 0 % over-visit** at u16 — but a **footprint** win, not a cull-speed
   win for the binary layout (the dequant arithmetic offsets the smaller-node cache
   win; a wide/8-ary node is where the literature's speed win actually lives).* →
   **and that wide node is now built + measured** (`examples/wide_bvh_bench`): an
   8-ary SoA node with a vectorised 8-box test is **~2× faster cull** than the binary
   BVH *and* a smaller arena — the literature's latency win, confirmed. See below.
4. **Refit + rotations** — only if a persistent dynamic BVH is ever added.
5. Verify the **AVX-512 broad-phase** lead with our own bench before investing
   (the spend limit cut its verification short).

## Implemented + measured (from this list)

- **Wide (8-ary) SIMD BVH node → `examples/wide_bvh_bench`** (2026-07-18). The
  latency lever the compressed-node result pointed to, built and measured. An 8-ary
  BVH: each node holds up to 8 children with their boxes stored **SoA**
  (`lo[axis][8]`, `hi[axis][8]`), so the sphere-vs-8-boxes test is a fixed 8-wide
  loop LLVM **auto-vectorises to AVX** (`-C target-cpu=native`); leaves hold ≤8
  points, tested exactly ⇒ verified **== brute force** at every size. Three BVHs over
  the same clumpy cloud (RTX-class box, min-of-8):

  | N | bin-f32 | wide8-f32 | wide8-u16 | nodes/query (bin→wide) | arena (bin→wide-u16) |
  | --- | --- | --- | --- | --- | --- |
  | 200 k | 15.1 µs | **8.8 µs (1.71×)** | 9.8 µs (1.53×) | 1460 → 42 (35×↓) | 12.8 → 1.5 MB |
  | 1 M | 88.9 µs | **41.2 µs (2.16×)** | 45.1 µs (1.97×) | 6566 → 196 (34×↓) | 64 → 9.4 MB |
  | 4 M | 626 µs | **283 µs (2.21×)** | 298 µs (2.10×) | 24427 → 1117 (22×↓) | 256 → 59 MB |

  **The wide node is a real ~2× latency win** (and grows with N) — the binary u16
  node was only a wash, but going *wide* pays: the 8:1 fan-out visits **~30× fewer
  nodes** (shallow tree, fewer pointer-chases) and the 8-box test vectorises. Arena
  is also **far smaller** (points batch into ≤8-point leaves, ~64× fewer internal
  nodes): 1 M drops 64 → 13 MB (f32) / 9.4 MB (u16). Quantising the wide node to u16
  costs a little vs wide-f32 (the same dequantise offset as the binary case) yet
  stays ~2× over binary **and** ~1.4× smaller than wide-f32 — so **wide8-u16 is the
  best footprint-and-speed point**. This is the layout to reach for if a static /
  query-heavy BVH ever graduates into the kit (the GPU LBVH build is the natural
  producer). Follow-on: the same SoA-wide idea on `Tree3`/`Octree3` arenas.

- **Compressed / quantized BVH nodes → `examples/compressed_bvh_bench`**
  (2026-07-18). A binary BVH over N points stored two ways with the **same
  topology**: full f32 boxes (**32 B/node**) vs each box **quantised to u16 relative
  to the root** — min rounded **down**, max rounded **up**, so the dequantised box is
  a *superset* (**20 B/node**). The leaves test the **exact** point, so the quantised
  cull is **bit-for-bit == brute force** (verified over random spheres at every
  size). Measured (RTX-class box, min-of-8, clumpy 3D cloud):
  - **1.6× smaller nodes** (32→20 B; e.g. 1 M pts: **64 → 40 MB** arena) at
    **zero accuracy cost** and, at u16 resolution, **0 % extra node visits** (the
    ~0.016-unit outward rounding is far below the clump scale, so conservative boxes
    match the exact ones for traversal).
  - **Cull latency: a wash** — 200k **18.4 → 18.6 µs** (1.01× slower), 1 M **206.6 →
    204.4 µs** (1.01× faster), 4 M **778 → 847 µs** (1.09× slower). The per-node
    dequantise (a float mul-add × 3 axes) offsets the smaller-node cache win; this
    binary traversal is **pointer-/branch-bound, not bandwidth-bound**.
  - **Honest verdict:** compression here is a **footprint** lever (fit ~1.6× more
    BVH in cache/VRAM — matters for the GPU build's `maxStorageBufferBindingSize`
    and huge static worlds), **not** a cull-speed lever — *and it corrects the
    earlier "breaks exactness" worry: it does not*. The literature's *speed* win
    comes from **wide (8-ary) compressed nodes** (one cache line = one node, SIMD box
    tests), which is the real next step if latency (not footprint) is the goal.

- **Cache-oblivious arena layout → `Tree3::compact()`** (2026-07-17). A one-pass
  DFS pre-order reorder of the node arena (a node lands adjacent to its first
  child, so a root→leaf descent walks mostly-contiguous memory) that also
  reclaims freed slots. Pure layout — brute-force-gated identical cull/knn/handle
  results (`compact_preserves_queries_and_handles`). **Measured on a churned
  keep-index tree** (`examples/compact_bench`, 300 frames of `update_ref`
  relocations that scramble the arena):
  - N=100k: cull **3.66 → 3.13 µs/query = 1.17×** (36k nodes, 1.8 ms to compact).
  - N=200k: cull **5.50 → 4.74 µs/query = 1.16×** (72k nodes, 3.2 ms to compact).
  Honest read: a steady **~1.16× cull win**, not the literature's 26–300 % (that
  was pathological / other structures). One pass, amortises over many frames; best
  before a query-heavy phase after churn (`bulk_load` already lays out compactly,
  so a fresh build doesn't need it). Not wired into the demos — their per-frame
  cull is a small slice, so 15 % of it isn't worth the periodic compaction hitch.

- **GPU radix sort → `examples/gpu_radix_bench`** (2026-07-17). A **stable 4-bit
  LSD radix** of 32-bit Morton codes, fully on the GPU: 3 compute kernels per pass
  (per-tile histogram → single-workgroup exclusive scan → stable local-rank
  scatter), ping-pong buffers, 8 passes, no CPU in the loop. **Verified exactly ==
  a CPU sort** at every size. Measured (RTX 4080 SUPER, min-of-7 vs
  `sort_unstable`):
  - 262k keys: GPU 1.69 ms · CPU 2.56 ms — **1.52× faster**
  - 1 M keys:  GPU 2.25 ms · CPU 11.29 ms — **5.01× faster**
  - 4 M keys:  GPU 4.46 ms · CPU 48.74 ms — **10.93× faster**
  The headline vs the old `gpu_sort_bench`: the **bitonic was ~2× *slower* than the
  CPU; the radix is 5–11× *faster*** at scale and correct+stable — the primitive the
  research named. Two scan iterations got it there: a 16-way (one-thread-per-digit)
  scan lifted the first single-workgroup-*serial* version off parity, then a
  **hierarchical multi-workgroup scan** (reduce per block of tiles → scan the blocks
  → add) removed the real bottleneck at scale — **1 M went 6.3→2.3 ms, 4 M 24.5→4.5
  ms** (small N pays a little for the extra passes; the Onesweep the research named).
  This **unblocks the on-GPU LBVH build**: the sort now stays
  GPU-resident (no readback), so parity-in-isolation is a win in-pipeline — sort
  Morton codes here, then Karras split + AABB refit on top, all without a round
  trip. (32-bit keys; a full build emulates u64 as `vec2<u32>`.)
  **Update (2026-07-18): + an 8-bit/4-pass width.** Sorting 8 bits at a time (4
  passes) instead of 4 bits (8 passes) halves the global histogram+scatter
  round-trips (at the cost of a 256- vs 16-bucket histogram/scan per pass) — measured
  **1.6–1.8× faster** than 4-bit/8-pass (262k 1.70→0.93 · 1 M 2.20→1.30 · 4 M
  4.35→2.79 ms), so the radix is now **8–17× the CPU** (1 M **8.1×**, 4 M **16.8×**).
  A *true* single-pass **Onesweep** (decoupled-lookback) would go further but needs an
  inter-workgroup forward-progress guarantee WebGPU/WGSL doesn't give (it can
  deadlock) — 8-bit/4-pass is the portable ceiling. Both widths verified == CPU.

- **Full GPU-resident LBVH build → `examples/gpu_lbvh_build_bench`** (2026-07-17).
  The whole build on the GPU, no CPU round-trip: **Morton** (30-bit, GPU) → the
  **key-value radix** above → **Karras** hierarchy (common-prefix ranges + split,
  index as the tiebreaker for equal codes) → **atomic bottom-up AABB refit** (only
  the second child to reach a node unions the boxes). Verified two ways: the sort
  against GPU-produced data, and the **whole tree by traversing the GPU-built BVH
  on the CPU vs brute force** over random spheres (a pass ⇒ hierarchy *and* refit
  AABBs correct). Measured (RTX 4080 SUPER, min-of-7, whole build/frame): **262k
  1.69 ms · 1 M 8.39 ms · 4 M 36.3 ms** (≈120–155 Mpts/s). So a **1 M-point BVH
  rebuilds on the GPU in ~8 ms/frame** — the piece that lets a *moving* broad-phase
  rebuild on-GPU each frame instead of leaning on the CPU keep-index. (The atomic
  refit relies on the platform's atomic ordering for cross-workgroup visibility —
  correct on this NVIDIA hardware, as the brute-force verify confirms; a level-by-
  level refit would be spec-portable. Headroom: Onesweep scan, compressed nodes.)
