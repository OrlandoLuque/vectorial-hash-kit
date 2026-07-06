# Performance guide — making the chosen structure fast

[`CHOOSING.md`](CHOOSING.md) picks *which* structure; this is *how to get the
most out of it*. Every number here is measured on this repo (min-of-N, one box);
the deep derivations live in [`THREE_D.md`](THREE_D.md),
[`PARALLEL.md`](PARALLEL.md), [`RAYCAST.md`](RAYCAST.md) and
[`PERF_NOTES.md`](PERF_NOTES.md). Rules of thumb first, caveats after.

## 1. Build flags — free wins

- **`target-cpu=native`.** The hot loops (distance tests, bbox classify, Morton
  encode) autovectorise; let the compiler use your machine's SIMD. Add to
  `.cargo/config.toml`:
  ```toml
  [build]
  rustflags = ["-C", "target-cpu=native"]
  ```
  (The kit's own release bins already build this way; a downstream crate must opt
  in itself. Not portable across machines — build on the target class.)
- **`--release`.** Non-negotiable; the arena descent is a different structure
  debug vs release (often 20–50×).
- **LTO + one codegen unit** for the last few percent on a shipping binary
  (`[profile.release] lto = "thin"`, `codegen-units = 1`).

## 2. The single highest-leverage choice: keep the index, don't rebuild it

For **moving** data, the maintenance cost dwarfs the query cost. Two paths:

- **`ItemRef` keep-index** (`insert_ref`/`update_ref`/`remove_ref`) — O(1)
  relocation, no locate walk, no predicate scan. **~5–11× faster maintain** than
  the predicate `update` (10× at `item_limit` 64), and it's what flips the
  per-frame relocation winner from a flat grid back to the tree. If your items
  move and you can hold a handle, **use it** — this is the biggest lever in the kit.
- **Morton grid rebuild** (`clear` + `insert`, or `extend_par`) — for a full
  per-frame rebuild the pointer-free grid's flat re-bucket is cheapest.

The measured rule: **moving points → keep-index tree; rebuild-everything-per-frame
→ Morton grid.** See `THREE_D.md` § "The fix: Stable ItemRef".

## 3. Threads — when they pay

`--features parallel` enables `cull_many_par` (batch reads fan out over rayon) and
`bulk_load_par`. **Reads parallelise; writes don't** — the lever for writes is the
keep-index (§2), not threads.

- **Batch culls** (many queries per frame — interest sets, per-agent perception):
  `cull_many_par` scales near-linearly. ~1 µs/query at 16 threads on 50k–1M points.
- **Crossover:** threads win once the batch is large enough to amortise the pool
  hand-off (measured in `PARALLEL.md`); a handful of queries stays serial.
- **Not in wasm** (no threads) — keep it feature-gated; the serial `cull_many` is
  always available.

## 4. Brute force below ~1 000 items

A linear scan over a contiguous `Vec` is SIMD-perfect and cache-perfect. For a
single query it **beats any index up to ~1 000 points** (the tree's descent +
result-alloc is ~100 ns fixed; a scan is ~1 ns/point). The `formations` demo
showed the index losing at ~60 units per band. Below a few hundred, don't index —
and let the [`advisor`](../crates/vectorial-hash/src/advisor.rs) (`SpatialProfile`
+ `recommend()`) pick per region from the measured local rate.

## 5. Query radius / selectivity

- **Small radius** (perception, separation, contact): the tree/grid cull is
  sub-µs; the cost is the descent, not the result. Keep `item_limit` modest (8–16)
  so leaves are small.
- **Fat radius** (broad interest bubbles): more cells straddle the boundary → more
  exact tests. A coarser structure (bigger `item_limit`, or `MortonGrid3` at a
  cell ≈ the radius via `levels_for_cell_size`) wins. `MortonGrid3::cull_layered`
  skips empty coarse blocks for a big query over sparse space.

## 6. Analytic shapes beat voxel rasters (usually)

`Sphere3`/`Circle` classify a box analytically (one distance compare). A 1×1×1
`VoxelRaster` (`.with_raster()`) is a *memory lookup* — it **loses** to the
analytic test (compute is cheaper than a cache miss) and **only wins** for an
expensive `contains_point`, e.g. a `Polyhedron3` past ~24–48 faces. **Don't raster
a sphere.** (And a raster over a huge bbox is a memory bomb — a 4000³ raster is
64 GB; keep rasters for small, many-faced shapes.)

## 7. GPU offload — only for query-dominated or static loads

The GPU LBVH query kernel is ~100× the serial CPU cull, but for **moving** data
the per-frame BVH rebuild eats it — a **parallel keep-index CPU** beats it at 1 M
(`PERF_NOTES.md` § "GPU LBVH broad-phase"). Offload to the GPU only when the query
dominates (huge query counts / fat bubbles) or the data is **static / rebuilt
anyway**. When the *whole* hot loop is GPU-resident (no per-frame round-trip), the
GPU wins big — that's the `gpu_storm` demo (~50× the CPU sim). The branchy
per-agent decision logic in a game sim stays on the CPU (GPU-hostile).

## Cheat sheet

| Situation | Do this |
| --- | --- |
| Items move every frame, you hold a handle | `ItemRef` keep-index (§2) |
| Full rebuild every frame | `MortonGrid3` `clear`+`insert` / `extend_par` |
| Many queries per frame, native | `cull_many_par` (§3) |
| < ~1 000 items | brute-force `Vec` scan (§4) |
| Big query over sparse space | `MortonGrid3::cull_layered` (§5) |
| Sphere / circle query | analytic (no `.with_raster()`) (§6) |
| Static world, thousands of queries | GPU LBVH (§7) |
| Whole uniform sim, want max scale | GPU-resident (`gpu_storm`) (§7) |
