# Point cloud — where the median split earns its keep

`pointcloud_wgpu` views a large **static, strongly skewed** point cloud: a procedurally
"scanned" scene (rolling ground, building shells, tree canopies). Points sit *on
surfaces*, so the cloud is dense in thin sheets and empty in the air between them —
the density skew a uniform grid has to pick one cell size for.

```bash
cargo run -p vectorial-hash-demos --bin pointcloud_wgpu --release
```
Controls: drag to orbit · wheel to zoom · move the mouse to sweep the probe sphere ·
`M` structure · `K` k (8/16/32) · `C` colour (density ↔ height) · `[` `]` cloud size.
Env: `CLOUD_N`, `CLOUD_INDEX=kd|octree|morton`, `CLOUD_MAX_FRAMES` (headless smoke).

Every point is coloured by its **local density** — the mean distance to its `k` nearest
neighbours — which means painting the cloud is *N k-NN queries*. That's the measurement:
pressing `M` rebuilds on another structure and re-runs the whole pass.

## Measured (120 000 points, k = 16, release)

| structure | build | k-NN over **all** points | per query |
| --- | ---: | ---: | ---: |
| **`KdTree3`** (median split) | 11 ms | **195 ms** | **1.63 µs** |
| `Octree3` (midpoint split) | 19 ms | 218 ms | 1.82 µs |
| `MortonGrid3` (flat grid) | **6 ms** | 329 ms | 2.74 µs |

- **The k-d tree answers k-NN 1.68× faster than the flat grid** and 1.12× faster than the
  pointer octree. Balancing by point *count* keeps its depth at ~log₂(n/leaf) however the
  points clump, so a query descends straight to the neighbourhood; the grid's ring-shell
  expansion has to wade through cells that are empty because the scene is mostly air.
- **It also builds 1.7× faster than the octree** — the midpoint octree keeps subdividing
  empty space, the median split never does.
- **The grid still builds fastest** (6 ms): if you rebuild constantly and query rarely,
  that's the trade. Here you build once and query 120 000 times.

The probe sphere you sweep with the mouse is the same story at a single-query scale
(≈10 µs for ~1 000 hits). Its result is checked against brute force on every smoke run,
and switching structures cross-checks the k-NN distances against the previous one, so a
fast number is also a correct one.

## Gotcha worth recording

`Aabb` is half-open, so a point at exactly the world maximum (or below the floor) is
*outside* the box: `Octree3::bulk_load` panics with "octants tile the parent" and
`MortonGrid3::insert` silently rejects it. The generator now clamps into the box — the
smoke test prints the cloud's bounds precisely so that class of bug can't hide.
