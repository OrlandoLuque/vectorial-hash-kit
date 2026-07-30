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

Median of **7 passes**, machine-idle gated (`cargo run -p bench-runner --release --
--group demos --only pointcloud --repeat 7`):

| structure | build | k-NN over **all** points | per query | spread |
| --- | ---: | ---: | ---: | ---: |
| **`KdTree3`** (median split) | 11.0 ms | 214.0 ms | **1.78 µs** | 16% |
| `Octree3` (midpoint split) | 18.5 ms | 215.8 ms | 1.80 µs | 3% |
| `MortonGrid3` (flat grid) | **6.3 ms** | 322.3 ms | 2.69 µs | 2% |

- **Both trees answer k-NN ~1.5× faster than the flat grid.** Balancing by point *count*
  (or by space) keeps a query descending straight to the neighbourhood; the grid's
  ring-shell expansion wades through cells that are empty because the scene is mostly air.
- **The k-d tree and the pointer octree are tied on k-NN here** — 214.0 vs 215.8 ms, well
  inside the k-d tree's own 16% run-to-run spread. An earlier single run had the k-d tree
  1.12× ahead; repeated passes do not support that. Its clear win over `Octree3` on this
  data is the **build: 1.7× faster** (11.0 vs 18.5 ms), because the midpoint octree keeps
  subdividing empty space and the median split never does.
- **The grid still builds fastest** (6.3 ms): if you rebuild constantly and query rarely,
  that's the trade. Here you build once and query 120 000 times.

The probe sphere you sweep with the mouse is the same story at a single-query scale
(≈10 µs for ~1 000 hits). Its result is checked against brute force on every smoke run,
and switching structures cross-checks the k-NN distances against the previous one, so a
fast number is also a correct one.

## The grid's cells are slabs on purpose (measured 2026-07-31)

`MortonGrid3` takes one `levels` for all three axes, so this demo's 1000 × 400 × 1000 world
gives cells of 15.6 × 6.2 × 15.6. That looks like the defect fixed in
[`THREE_D.md`](THREE_D.md) § "Better still: do not build a non-cubic grid", and it is not.

Padding the index world to a cube — which is a 1.63–1.96× win in the horde — makes k-NN here
**1.2–1.4× slower** (261–276 ms over all points, up to 335–379 ms). Re-picking `levels` to
restore the same points-per-cell did not recover it either. The reason is that a scanned scene
genuinely *fills* its 400 units of height, so cells that are shorter in y than in x/z are
proportionate to the data; squaring them just makes each one swallow 2.5× more points.

The horde qualifies because its units sit in a ~25-unit band inside a 72-unit world. This one
does not. Left as it was, deliberately, with the numbers recorded so it does not get "fixed".

## Gotcha worth recording

`Aabb` is half-open, so a point at exactly the world maximum (or below the floor) is
*outside* the box: `Octree3::bulk_load` panics with "octants tile the parent" and
`MortonGrid3::insert` silently rejects it. The generator now clamps into the box — the
smoke test prints the cloud's bounds precisely so that class of bug can't hide.
