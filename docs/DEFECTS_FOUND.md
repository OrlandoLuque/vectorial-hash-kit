# Defects found during the validation campaign

Living log of bugs the test suite has caught — each entry: how the bug was
exposed, why it happened, how it was fixed, and how the fix is now pinned
against regressions. Every fix below ships with a permanent test that would
fail the same way if the bug returned.

## D-1 — `Tree::divide` recursed forever on duplicate items

**Exposed by**: the exhaustive culling campaign (Run 1, 2026-06-10), with a
stack overflow when `--ignored` ran. The reproducer was a scenario whose
random point cloud landed three items on the exact same lattice point with
`item_limit = 1`.

**Cause**: `divide(node)` recursed unconditionally as long as the leaf
overflowed `item_limit`. With ≥`item_limit + 1` items at one coordinate,
no split could ever separate them, and the cell side halved forever under
binary subdivision until float precision broke down (or the stack did).

**Fix** (`crates/vectorial-hash/src/tree.rs`):
- A leaf whose items all share one position stops dividing (soft limit).
- A leaf whose longest side has shrunk below `min_cell` (= 1e-12 × root
  side) stops dividing — final safety net against degenerate subdivision
  from arbitrary float jitter.

**Pinned by**: `tree::tests::duplicate_positions_do_not_split_forever` and
every long-campaign scenario that randomly seeds duplicates.

## D-2 — `Polygon::is_inside` was unstable on horizontal-tangent geometry

**Exposed by**: the exhaustive culling campaign (Run 2). Random box-at-135°
scenarios produced cull results that disagreed with brute force. Drilling
in: whole horizontal rows inside the figure came out `Out` in the
template, leaving "holes" in the middle of perfectly filled boxes.

**Cause**: the legacy `is_inside` cast a horizontal ray from `(-1e7, vy)`
toward the test point and counted polygon crossings. When the ray ran at
the exact height of a polygon vertex (a box at 135° puts two vertices at
the test height) or grazed an arc tangent (circle at 90°/270°, drop at
0°/30°), the per-edge "intersection on segment?" check became unstable
under float precision, and the crossing count flipped randomly. The
classification of every cell whose vertices happened to fall on those
heights inherited the instability.

**Fix** (`crates/vectorial-hash-templates/src/polygon.rs`):
- Detect ray degeneracy: a ray is degenerate if any polygon vertex lies
  within `2 × EPSILON` of the casting line, or if an arc is nearly
  tangent (`|distance from arc centre to line − radius| ≤ 2 × EPSILON`).
- When the horizontal ray is clean, use it exactly as before — k=0 keeps
  the original origin `(-1e7, vy)` byte-for-byte so non-degenerate cases
  produce **identical** results to the legacy code.
- When it is degenerate, rotate the ray by `k × 0.39996` rad for k =
  1..=7 until a clean one is found.

**Pinned by**:
- `crates/vectorial-hash-templates/tests/boundary_regressions.rs`
  (`rotated_box_raster_has_no_out_rows_through_the_middle`,
  `is_inside_is_stable_when_ray_would_graze_vertices`).
- `crates/vectorial-hash-templates/tests/fingerprint_regression.rs`:
  any drift in template bytes for the fixed reference set surfaces in
  `cargo test`, with the first eight differing lines printed.
- `crates/vectorial-hash-templates/tests/verify_88_ray_fix_templates.rs`:
  see D-4 for what this catches.

## D-3 — Leaf bbox pre-filter dropped on-boundary items

**Exposed by**: same campaign run. Items located exactly on the figure's
bounding-box edge (a box with integer side at an integer origin puts
several items there by construction) were filtered out before the
per-item exact test.

**Cause**: `Rect::contains` is half-open (`x < x_max && y < y_max`),
which is correct for cell-of-the-tree containment (cells must tile
disjointly) but wrong for the figure-bbox pre-filter, because the
figure's boundary belongs to the figure (`is_inside` returns true on the
contour).

**Fix** (`crates/vectorial-hash/src/geom.rs`,
`crates/vectorial-hash/src/culling.rs`,
`crates/vectorial-hash-cli/src/quadtree.rs`,
`crates/vectorial-hash-cli/src/bench.rs`):
- New `Rect::contains_closed` (right/bottom inclusive).
- Every leaf-fallback pre-filter now uses `contains_closed`.
- The original `contains` stays in use for tree-cell membership.

**Pinned by**: the exhaustive campaign — any scenario that seeds an
integer item on the figure's bbox edge exercises this path.

## D-4 — The ray-fix itself introduced an epsilon regression in k=0

**Exposed by**: a manual cell-by-cell verification of every template that
changed between pre-fix and post-fix versions of `is_inside` (D-2's fix).
Ground truth was built from explicit math (`ConvexQuad`, `Circle`,
`RotatedDrop` = triangle + arc cap), nothing of `Polygon::is_inside`
involved.

**Result of the first pass**: 81 templates differed; 940 of 941 changed
cells were strictly more correct under post-fix. One cell — `drop a30`
offset `(1, 5)`, world cell `[-16, -8] × [24, 32]` — went from `In`
(legacy, correct) to `Maybe` (post-fix, wrong). The cell sits entirely
inside the rotated drop; the geometry confirms 4096/4096 sample points
inside.

**Cause**: D-2's fix set the k=0 ray origin to `Vertex::new(vx - 1e7,
vy)` instead of the legacy `Vertex::new(-1e7, vy)`. Both rays are
geometrically identical (same horizontal direction, same length), but
the segment-parameter check `t = (int.x − l1.x) / (l2.x − l1.x)` lives
near `EPSILON = 1e-5`. Shifting `l1.x` by `vx` units re-arranges float
magnitudes enough that a near-tangent arc intersection at `t ≈ 1 +
8.4e-7` is now included in the segment, while under legacy it ended at
`t ≈ 1.00000084` and was filtered. The winding logic branches on
`int.len() == 1` vs `int.len() == 2` for arcs, so flipping which side
of the EPSILON boundary the intersection lands on flipped `inside` →
`outside`.

**Fix** (`crates/vectorial-hash-templates/src/polygon.rs`):
- The k=0 branch reverts to the legacy origin `Vertex::new(-1e7, vy)`.
- k≥1 keep the rotated rays — they are what actually handles
  degeneracy; clean rays never need to leave k=0 anyway.
- A hidden `Polygon::is_inside_legacy` is kept inside the crate so
  future investigations can A/B compare without resurrecting git history.

**Verification after fix**: 80 templates differ (down from 81 — the
regression is gone); the cell-by-cell check now reports **940/940
ground-truth matches in post-fix and 0/940 in pre-fix**. Every change
the ray-fix introduces is strictly a correction.

**Pinned by**:
- `verify_88_ray_fix_templates::every_changed_template_is_more_correct_post_fix`
  — regenerates the ground truth and compares cell-by-cell on every
  `cargo test`.
- `verify_88_ray_fix_templates::drop_a30_o1_5_cell_at_minus_16_32_is_inside`
  — focused regression for the specific cell, asserting current and
  legacy agree (because the ray is non-degenerate here).

---

## How regressions are kept out

| Layer | What it covers | Where |
| --- | --- | --- |
| Unit tests | Single operations on `Tree`, `TemplateGrid`, `TemplateBank` (split, merge-up, scale, aggregate, …) | `crates/*/src/**/*.rs` (`#[cfg(test)]`) |
| Boundary regressions | Geometric configurations known to have been broken (D-1 to D-4) | `crates/vectorial-hash-templates/tests/boundary_regressions.rs`, the focused tests in `verify_88_ray_fix_templates.rs` |
| Snapshot fingerprint | A deterministic fixed-set dump of generated templates compared byte-for-byte against a versioned fixture | `tests/fingerprint_regression.rs` + `tests/fixtures/template_fingerprint.txt` |
| Cell-by-cell verification | Every template that has ever differed from the legacy implementation, classified against pure-math ground truth | `tests/verify_88_ray_fix_templates.rs` + `tests/fixtures/fp_pre.txt` |
| Exhaustive culling campaign | Property/fuzz over random churned trees × random figures × random angles/origins, every cull config equality-gated against brute force | `tests/exhaustive_culling.rs` (40 scenarios in `cargo test`; `--ignored` runs 2,000) |

Together they form a defence in depth: a regression has to evade unit
tests, the boundary set, the byte-exact snapshot, the cell-by-cell
verification AND 2,000 randomized scenarios to stay unnoticed.
