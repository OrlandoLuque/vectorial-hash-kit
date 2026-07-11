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
  hitting). *Confirmed 3-0.* → a candidate node layout for `Tree3`/`Octree3`/LBVH.
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

1. **Onesweep-style GPU radix sort** → unblocks the GPU-side LBVH build (the one
   thing that would push the moving-broad-phase crossover past 1 M). Keep 32-bit
   Morton codes.
2. **Cache-oblivious arena layout** for the trees — cheap, algorithm-free, high upside.
3. **Compressed wide-BVH nodes** — cut bytes-per-step against the memory wall.
4. **Refit + rotations** — only if a persistent dynamic BVH is ever added.
5. Verify the **AVX-512 broad-phase** lead with our own bench before investing
   (the spend limit cut its verification short).
