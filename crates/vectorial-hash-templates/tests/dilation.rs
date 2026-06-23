//! Minkowski dilation: geometry verification, the distance-based
//! narrowphase, the precomputed inflated raster (production path), and the
//! `is_inside` agreement gates.
//!
//! ## Two ways to ask "is this point within distance r of figure F?"
//!
//! 1. **Distance (production narrowphase):** `F.within_dilation(r, p)` =
//!    `F.is_inside(p) || F.dist_to_boundary(p) <= r`. Pure distance math on
//!    the *original* polygon — the cheapest exact test, used on `Maybe`
//!    raster pixels.
//! 2. **Inflated polygon:** build `inflated_convex(F, r)` and call
//!    `is_inside` on it. The inflated drop has 6 mixed line/arc edges; the
//!    winding ray-casting used to double-count where a line edge met an arc
//!    edge and the ray grazed the shared vertex. Fixed 2026-06-23 (signed
//!    winding + widened degeneracy band); `dilation_is_inside_matches_
//!    distance_after_fix` and `dilation_matches_distance_ground_truth_via_
//!    is_inside` now gate it as exact. Production still prefers (1) because
//!    the distance test is cheaper, not because (2) is unreliable.
//!
//! Ground truth throughout is independent distance math (point-segment /
//! point-arc), never the polygon's own `is_inside`.

use vectorial_hash_templates::intersector;
use vectorial_hash_templates::polygon::{
    create_box, create_circle, create_drop, create_square, inflated_convex, rotated_copy,
    scaled_copy, Polygon,
};
use vectorial_hash_templates::templates::angle_to_radians;

const FUZZ: f64 = 2e-5;

fn dist_point_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return intersector::dist(px, py, ax, ay);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    intersector::dist(px, py, ax + t * dx, ay + t * dy)
}

/// Independent ground truth: distance from a point to the polygon's
/// boundary (edges + arcs). Deliberately NOT `Polygon::dist_to_boundary`
/// so the production helper is checked against a separate implementation.
fn dist_to_boundary(poly: &Polygon, px: f64, py: f64) -> f64 {
    let n = poly.vertices.len();
    let mut best = f64::MAX;
    for i in 0..n {
        let v = &poly.vertices[i];
        let w = &poly.vertices[(i + 1) % n];
        let d = if v.seg.d == 0 {
            dist_point_segment(px, py, v.x, v.y, w.x, w.y)
        } else {
            let (xc, yc) = (v.seg.xc, v.seg.yc);
            let radius = intersector::dist(xc, yc, v.x, v.y);
            let pa = intersector::angle(xc, yc, px, py);
            let mut a1 = intersector::angle(xc, yc, v.x, v.y);
            let mut a2 = intersector::angle(xc, yc, w.x, w.y);
            if v.seg.d == -1 {
                std::mem::swap(&mut a1, &mut a2);
            }
            let in_span = if a2 >= a1 { pa >= a1 && pa <= a2 } else { pa >= a1 || pa <= a2 };
            if in_span {
                (intersector::dist(xc, yc, px, py) - radius).abs()
            } else {
                intersector::dist(px, py, v.x, v.y).min(intersector::dist(px, py, w.x, w.y))
            }
        };
        best = best.min(d);
    }
    best
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn test_figures() -> Vec<(&'static str, Polygon)> {
    vec![
        ("drop", scaled_copy(&create_drop(0.2, 0.8), 40.0, 40.0)),
        ("box", create_box(36.0)),
        ("rect", scaled_copy(&create_square(0.0, 0.0, 1.0, 1.0), 30.0, 20.0)),
        ("circle", scaled_copy(&create_circle(1.0), 25.0, 25.0)),
    ]
}

/// 1. `dist_to_boundary` (the production helper) matches the independent
///    ground-truth distance everywhere.
#[test]
fn dist_to_boundary_matches_independent_ground_truth() {
    let mut seed = 500;
    for (name, base) in &test_figures() {
        for &angle in &[0.0, 30.0, 135.0, 250.0] {
            let fig = rotated_copy(base, angle_to_radians(angle));
            seed += 1;
            let mut rng = Rng(seed);
            let pad = 15.0;
            for _ in 0..3000 {
                let px = fig.x_min - pad + rng.unit() * (fig.x_max - fig.x_min + 2.0 * pad);
                let py = fig.y_min - pad + rng.unit() * (fig.y_max - fig.y_min + 2.0 * pad);
                let a = fig.dist_to_boundary(px, py);
                let b = dist_to_boundary(&fig, px, py);
                assert!(
                    (a - b).abs() < 1e-9,
                    "{name}@{angle}: dist_to_boundary mismatch at ({px:.4},{py:.4}): {a} vs {b}"
                );
            }
        }
    }
}

/// 2. The robust dilation narrowphase `within_dilation` matches distance
///    ground truth across shapes, angles, radii (outside the epsilon halo).
///    This is the production per-point test for an agent of body radius r.
#[test]
fn within_dilation_matches_distance_ground_truth() {
    let mut seed = 1000;
    let mut checked = 0u64;
    for (name, base) in &test_figures() {
        for &angle in &[0.0, 30.0, 135.0, 250.0] {
            let fig = rotated_copy(base, angle_to_radians(angle));
            for &r in &[3.0, 8.0, 15.0] {
                seed += 1;
                let mut rng = Rng(seed);
                let pad = r * 2.0 + 6.0;
                for _ in 0..4000 {
                    let px = fig.x_min - pad + rng.unit() * (fig.x_max - fig.x_min + 2.0 * pad);
                    let py = fig.y_min - pad + rng.unit() * (fig.y_max - fig.y_min + 2.0 * pad);
                    let d = dist_to_boundary(&fig, px, py);
                    // Skip the ambiguous halos: near r, and near the original
                    // boundary (where `is_inside` itself is epsilon-defined).
                    if (d - r).abs() <= FUZZ || d <= FUZZ {
                        continue;
                    }
                    let expected = fig.is_inside(px, py) || d <= r;
                    let actual = fig.within_dilation(r, px, py);
                    assert_eq!(
                        actual, expected,
                        "{name}@{angle} r={r}: within_dilation wrong at ({px:.4},{py:.4}) d={d:.5}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100_000, "too few unambiguous samples ({checked})");
}

/// 3. The inflated *raster* (the precomputed production lookup) classifies
///    pixels consistently with distance ground truth: a pixel strictly
///    inside the dilation is never `Out`, one strictly outside is never
///    `In`. This is the path the runtime actually uses, and it is robust
///    because raster cells are classified by polygon-polygon containment,
///    not by `is_inside` ray-casting.
#[test]
fn inflated_raster_matches_distance_ground_truth() {
    use vectorial_hash::{CellState, Point};
    use vectorial_hash_templates::bank::{FigureKey, TemplateBank};

    let mut seed = 99;
    let mut interior_hits = 0;
    for (name, base) in &test_figures() {
        for &angle in &[0.0, 135.0] {
            let fig = rotated_copy(base, angle_to_radians(angle));
            for &r in &[5.0, 10.0] {
                let inflated = inflated_convex(&fig, r);
                let dims: Vec<f64> = vec![fig.x_max - fig.x_min, fig.y_max - fig.y_min, r, angle];
                let figkey = FigureKey::new(7000, &dims);
                let mut bank = TemplateBank::new();
                bank.generate_size(&figkey, &inflated, &[0.0], 1, 1);

                let origin = (40i64, 64i64);
                let raster = bank.placed_raster(&figkey, 0.0, origin).unwrap();
                let mut moved = inflated.clone();
                moved.move_by(origin.0 as f64, origin.1 as f64);
                let mut moved_orig = fig.clone();
                moved_orig.move_by(origin.0 as f64, origin.1 as f64);

                seed += 1;
                let mut rng = Rng(seed);
                for _ in 0..3000 {
                    let px = moved.x_min - 5.0 + rng.unit() * (moved.x_max - moved.x_min + 10.0);
                    let py = moved.y_min - 5.0 + rng.unit() * (moved.y_max - moved.y_min + 10.0);
                    let d = dist_to_boundary(&moved_orig, px, py);
                    let inside_orig = moved_orig.is_inside(px, py);
                    let state = raster.cell_at_world(Point::new(px, py));
                    if inside_orig || d < r - 1.5 {
                        assert_ne!(state, CellState::Out,
                            "{name}@{angle} r={r}: ({px:.2},{py:.2}) d={d:.3} inside dilation but raster Out");
                        interior_hits += 1;
                    } else if !inside_orig && d > r + 1.5 {
                        assert_ne!(state, CellState::In,
                            "{name}@{angle} r={r}: ({px:.2},{py:.2}) d={d:.3} outside dilation but raster In");
                    }
                }
            }
        }
    }
    assert!(interior_hits > 1000, "too few interior probes ({interior_hits})");
}

/// 4. Sanity: an inflated circle is a bigger circle (closed form).
#[test]
fn inflated_circle_is_a_bigger_circle() {
    let circle = scaled_copy(&create_circle(1.0), 30.0, 30.0);
    let inflated = inflated_convex(&circle, 8.0);
    let mut rng = Rng(7);
    for _ in 0..2000 {
        let px = -50.0 + rng.unit() * 100.0;
        let py = -50.0 + rng.unit() * 100.0;
        let d = (px * px + py * py).sqrt();
        if (d - 38.0).abs() <= FUZZ {
            continue;
        }
        assert_eq!(inflated.is_inside(px, py), d <= 38.0, "({px:.3}, {py:.3}) |p|={d:.5}");
    }
}

/// 5. The inflated polygon's bounding box is the original grown by exactly
///    r per side — a property `inflated_convex` must hold regardless of the
///    `is_inside` issue (the geometry is correct; only ray-casting is not).
#[test]
fn inflated_bbox_grows_by_r() {
    for (name, base) in &test_figures() {
        for &angle in &[0.0, 30.0, 135.0] {
            let fig = rotated_copy(base, angle_to_radians(angle));
            for &r in &[3.0, 8.0] {
                let inf = inflated_convex(&fig, r);
                assert!((inf.x_min - (fig.x_min - r)).abs() < 1e-6, "{name}@{angle} r={r}: x_min");
                assert!((inf.x_max - (fig.x_max + r)).abs() < 1e-6, "{name}@{angle} r={r}: x_max");
                assert!((inf.y_min - (fig.y_min - r)).abs() < 1e-6, "{name}@{angle} r={r}: y_min");
                assert!((inf.y_max - (fig.y_max + r)).abs() < 1e-6, "{name}@{angle} r={r}: y_max");
            }
        }
    }
}

/// 6. REGRESSION (was a characterization of the bug; now a pass/fail gate):
///    `is_inside` ray-casting on the *inflated* polygon must agree with the
///    distance ground truth everywhere outside the epsilon halo. This grid
///    used to show ~0.028% disagreement clustered along lines that graze the
///    inflated arcs — the winding double-count at line/arc vertex junctions.
///    Since the 2026-06-23 fix (signed-winding rule + wider degeneracy band
///    so the multi-ray search steps off a grazing ray), it must be exact.
#[test]
fn dilation_is_inside_matches_distance_after_fix() {
    let base = scaled_copy(&create_drop(0.2, 0.8), 40.0, 40.0);
    let fig = rotated_copy(&base, angle_to_radians(135.0));
    let r = 3.0;
    let inf = inflated_convex(&fig, r);

    let (x0, x1) = (fig.x_min - 5.0, fig.x_max + 5.0);
    let (y0, y1) = (fig.y_min - 5.0, fig.y_max + 5.0);
    let step = 0.1;
    let mut total = 0u64;
    let mut disagree = 0u64;
    let mut y = y0;
    while y <= y1 {
        let mut x = x0;
        while x <= x1 {
            let d = dist_to_boundary(&fig, x, y);
            if (d - r).abs() > 1e-2 && d > 1e-2 {
                let expected = fig.within_dilation(r, x, y); // robust ground truth
                let raycast = inf.is_inside(x, y); // formerly fragile path
                total += 1;
                if raycast != expected {
                    disagree += 1;
                }
            }
            x += step;
        }
        y += step;
    }
    println!("inflated-drop@135 r=3: is_inside vs distance disagreed on {disagree}/{total}");
    assert_eq!(disagree, 0, "is_inside on the inflated polygon must match distance ground truth");
}

/// 8. PRECACHING: the production move for agents with body radius. For
///    each (shape, radius) we precompute an *inflated* template set keyed
///    by a FigureKey that includes the radius; at runtime we select the
///    template for the target's radius and the query becomes a raster
///    lookup — no geometry, no per-point distance, the dilation is baked
///    into the precomputed In/Out/Maybe cells. This is "precalculate the
///    dilation into the raster" from the design discussion.
#[test]
fn precached_dilated_templates_select_by_radius_at_runtime() {
    use vectorial_hash::{CellState, Point};
    use vectorial_hash_templates::bank::{FigureKey, TemplateBank};

    let base = scaled_copy(&create_drop(0.2, 0.8), 40.0, 40.0);
    let angles = [0.0, 90.0, 180.0, 270.0];
    let radii = [4.0, 9.0, 16.0];
    let sizes: [(u32, u32); 3] = [(8, 8), (16, 16), (1, 1)];

    // --- precache: one inflated template set per (shape, radius) ---
    let mut bank = TemplateBank::new();
    let mut keys = std::collections::HashMap::new();
    for &r in &radii {
        let inflated = inflated_convex(&base, r);
        // The radius is part of the figure identity, so different radii are
        // distinct precomputed sets (no collision in the bank index).
        let dims = vec![0.2 * 40.0, 0.8 * 40.0, r];
        let figkey = FigureKey::new(3, &dims);
        let ang: Vec<f64> = angles.to_vec();
        for &(cw, ch) in &sizes {
            bank.generate_size(&figkey, &inflated, &ang, cw, ch);
        }
        keys.insert(r as i64, figkey);
    }

    // --- runtime: pick the right precomputed set by the target's radius,
    //     and confirm the inflated raster answers "centre within r of the
    //     figure" consistently with distance ground truth. ---
    let mut seed = 4242;
    let mut interior_hits = 0;
    for &r in &radii {
        let figkey = keys.get(&(r as i64)).unwrap();
        let origin = (50i64, 70i64);
        let raster = bank.placed_raster(figkey, 0.0, origin).unwrap();
        let mut moved_orig = base.clone();
        moved_orig.move_by(origin.0 as f64, origin.1 as f64);

        seed += 1;
        let mut rng = Rng(seed);
        let pad = r + 6.0;
        for _ in 0..2500 {
            let px = moved_orig.x_min - pad + rng.unit() * (moved_orig.x_max - moved_orig.x_min + 2.0 * pad);
            let py = moved_orig.y_min - pad + rng.unit() * (moved_orig.y_max - moved_orig.y_min + 2.0 * pad);
            let d = dist_to_boundary(&moved_orig, px, py);
            let inside_orig = moved_orig.is_inside(px, py);
            let state = raster.cell_at_world(Point::new(px, py));
            if inside_orig || d < r - 1.5 {
                assert_ne!(state, CellState::Out,
                    "r={r}: agent centre ({px:.1},{py:.1}) is within r of the figure but raster Out");
                interior_hits += 1;
            } else if !inside_orig && d > r + 1.5 {
                assert_ne!(state, CellState::In,
                    "r={r}: agent centre ({px:.1},{py:.1}) is beyond r of the figure but raster In");
            }
        }
    }
    assert!(interior_hits > 1500, "too few interior probes ({interior_hits})");
}

/// 7. Direct contract: `inflated.is_inside(p)` matches distance ground truth
///    across shapes/angles/radii. This used to be `#[ignore]`d because the
///    winding ray-casting double-counted at line/arc vertex junctions on the
///    many-arc inflated polygons; the 2026-06-23 winding fix makes it pass,
///    so it is now an enabled gate. (The production narrowphase still prefers
///    `within_dilation` for the cheaper distance test, but the inflated
///    polygon's own `is_inside` is now correct too.)
#[test]
fn dilation_matches_distance_ground_truth_via_is_inside() {
    let figures = test_figures();
    let mut seed = 1000;
    for (name, base) in &figures {
        for &angle in &[0.0, 30.0, 135.0] {
            let fig = rotated_copy(base, angle_to_radians(angle));
            for &r in &[3.0, 8.0] {
                seed += 1;
                let inflated = inflated_convex(&fig, r);
                let mut rng = Rng(seed);
                let pad = r * 2.0 + 5.0;
                for _ in 0..4000 {
                    let px = fig.x_min - pad + rng.unit() * (fig.x_max - fig.x_min + 2.0 * pad);
                    let py = fig.y_min - pad + rng.unit() * (fig.y_max - fig.y_min + 2.0 * pad);
                    let d = dist_to_boundary(&fig, px, py);
                    if (d - r).abs() <= FUZZ || d <= FUZZ {
                        continue;
                    }
                    let expected = fig.is_inside(px, py) || d <= r;
                    assert_eq!(inflated.is_inside(px, py), expected,
                        "{name}@{angle} r={r}: ({px:.4},{py:.4}) d={d:.6}");
                }
            }
        }
    }
}
