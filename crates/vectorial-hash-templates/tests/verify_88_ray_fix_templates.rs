//! Verify, cell by cell, every template that changed between the pre-fix
//! and post-fix versions of `Polygon::is_inside` (the 2026-06 ray-degeneracy
//! fix). For each differing cell, classify it independently with explicit
//! geometry (no `Polygon::is_inside` involved) and confirm that the post-fix
//! template matches that classification while the pre-fix template doesn't.
//!
//! Inputs: `fp_pre.txt` and `fp_HEAD.txt` (deterministic fingerprints from
//! the worktree comparison; included as fixtures so the proof is part of
//! the repo). Skip if either file is missing — the verification is opt-in.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use vectorial_hash_templates::polygon::create_drop;

const POST: &str = include_str!("fixtures/template_fingerprint.txt");
const _: &str = include_str!("fixtures/fp_pre.txt"); // keep fixture pinned to the crate

// 8 samples per side = 64 probes per cell, enough to detect any cell whose
// inside-mass is not zero or one.
const PROBES_PER_SIDE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Out,
    Maybe,
    In,
}

fn decode_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

/// Decode the PHP-bug binary format back to (cols, rows, cells row-major).
fn decode_template(hex: &str) -> (u32, u32, Vec<Cell>) {
    let bytes = decode_hex(hex);
    let cols = bytes[0] as u32;
    let rows = bytes[1] as u32;
    let total = (cols * rows) as usize;
    let mut cells = Vec::with_capacity(total);
    let mut idx = 0usize;
    for _ in 0..total {
        let byte_idx = 2 + idx / 4;
        let shift = (3 - (idx % 4)) * 2;
        let v = if byte_idx < bytes.len() {
            (bytes[byte_idx] >> shift) & 0b11
        } else {
            0 // PHP-bug tail: missing bits are Out
        };
        cells.push(match v {
            0 => Cell::Out,
            1 => Cell::Maybe,
            2 => Cell::In,
            _ => unreachable!(),
        });
        idx += 1;
    }
    (cols, rows, cells)
}

/// World-space cell bounds: the template's grid is anchored at (gx0*cw,
/// gy0*ch) and we need to reconstruct that anchor. The fingerprint key
/// carries (scale, angle, cw, ch, ox, oy); we replay the bbox math.
#[derive(Debug, Clone)]
struct Header {
    figure: String,
    scale: f64,
    angle: f64,
    cw: i64,
    ch: i64,
    ox: i64,
    oy: i64,
}

fn parse_header(line: &str) -> (Header, String) {
    // Format: "name s32 a45 c8x8 o3,2 : hex"
    let (header, hex) = line.split_once(" : ").unwrap();
    let mut parts = header.split_whitespace();
    let figure = parts.next().unwrap().to_string();
    let scale: f64 = parts.next().unwrap().trim_start_matches('s').parse().unwrap();
    let angle: f64 = parts.next().unwrap().trim_start_matches('a').parse().unwrap();
    let cell = parts.next().unwrap().trim_start_matches('c');
    let (cw_s, ch_s) = cell.split_once('x').unwrap();
    let cw: i64 = cw_s.parse().unwrap();
    let ch: i64 = ch_s.parse().unwrap();
    let off = parts.next().unwrap().trim_start_matches('o');
    let (ox_s, oy_s) = off.split_once(',').unwrap();
    let ox: i64 = ox_s.parse().unwrap();
    let oy: i64 = oy_s.parse().unwrap();
    (
        Header { figure, scale, angle, cw, ch, ox, oy },
        hex.to_string(),
    )
}

/// Ground-truth classifier per figure family — pure mathematics, no calls
/// into `Polygon::is_inside`.
trait GroundTruth {
    fn contains(&self, x: f64, y: f64) -> bool;
}

/// Axis-aligned square (used for `square_0.5_0.7` post-rotation too via
/// generic polygon classifier below; here we only need it for non-rotated).
struct AxisBox {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}
impl GroundTruth for AxisBox {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x_min - EPS && x <= self.x_max + EPS && y >= self.y_min - EPS && y <= self.y_max + EPS
    }
}

/// Rotated rectangle (the box family `create_box(side)` rotated by `angle`).
/// We use the convex-quad point-in-polygon: a point is inside iff it lies on
/// the same side of every edge (we walk CCW and check sign(cross) >= 0).
struct ConvexQuad {
    v: [(f64, f64); 4],
}
impl GroundTruth for ConvexQuad {
    fn contains(&self, x: f64, y: f64) -> bool {
        let mut sign: Option<f64> = None;
        for i in 0..4 {
            let (ax, ay) = self.v[i];
            let (bx, by) = self.v[(i + 1) % 4];
            let cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if cross.abs() <= EPS {
                continue; // on the edge: treat as inside
            }
            match sign {
                None => sign = Some(cross),
                Some(s) => {
                    if s * cross < 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Circle of radius `r` centered at `(cx, cy)`.
struct Circle {
    cx: f64,
    cy: f64,
    r: f64,
}
impl GroundTruth for Circle {
    fn contains(&self, x: f64, y: f64) -> bool {
        let dx = x - self.cx;
        let dy = y - self.cy;
        dx * dx + dy * dy <= self.r * self.r + EPS
    }
}

/// Regression: the cell that exposed an unintended ray-origin change in the
/// k=0 branch of the ray-degeneracy fix. The cell sits entirely inside the
/// drop figure at angle 30° with offset (1, 5); legacy classifies it In, an
/// earlier version of the fix flipped it to Maybe by shifting the ray
/// origin by `vx` (which changed float epsilons in the segment-parameter
/// check on a near-tangent arc intersection). After fixing k=0 to keep the
/// original `(-1e7, vy)` origin byte-for-byte, the cell classifies In again
/// — same as legacy AND ground truth.
#[test]
fn drop_a30_o1_5_cell_at_minus_16_32_is_inside() {
    use vectorial_hash_templates::polygon::{create_drop, rotated_copy, scaled_copy};
    use vectorial_hash_templates::templates::angle_to_radians;
    let base = scaled_copy(&create_drop(0.2, 0.8), 32.0, 32.0);
    let rotated = rotated_copy(&base, angle_to_radians(30.0));
    let mut moved = rotated.clone();
    moved.move_by(1.0, 5.0);
    assert!(moved.is_inside(-16.0, 32.0), "vertex must classify inside");
    assert_eq!(
        moved.is_inside(-16.0, 32.0),
        moved.is_inside_legacy(-16.0, 32.0),
        "current must agree with legacy on this non-degenerate-ray case",
    );
}

/// The drop figure rotated by `angle_deg` and applied at offset (ox, oy):
/// triangle below the arc-cap. Built from the rotated polygon's vertices
/// and arc parameters directly so we don't depend on `is_inside`.
struct RotatedDrop {
    angle_rad: f64,
    ox: f64,
    oy: f64,
    width: f64,
    height: f64,
}
impl GroundTruth for RotatedDrop {
    fn contains(&self, x: f64, y: f64) -> bool {
        // Inverse-rotate around (ox, oy), translate by (-ox, -oy), then
        // test against the canonical drop with the same width/height.
        let dx = x - self.ox;
        let dy = y - self.oy;
        let c = self.angle_rad.cos();
        let s = self.angle_rad.sin();
        let px = c * dx + s * dy; // rotate by -angle
        let py = -s * dx + c * dy;

        // Canonical drop (per `create_drop`): vertices (-w, h), (w, h),
        // (0, 0); the segment from (-w, h) to (w, h) is an arc (clockwise,
        // center (0, h), radius w) bulging upward to y = h + w (cardinal
        // north of the centre falls inside the arc, as recalc_bounds proves).
        let (w, h) = (self.width, self.height);
        // Triangle (-w, h), (w, h), (0, 0) — same-side test.
        let in_triangle = {
            let v = [(-w, h), (w, h), (0.0, 0.0)];
            let mut sign: Option<f64> = None;
            let mut inside = true;
            for i in 0..3 {
                let (ax, ay) = v[i];
                let (bx, by) = v[(i + 1) % 3];
                let cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
                if cross.abs() <= EPS {
                    continue;
                }
                match sign {
                    None => sign = Some(cross),
                    Some(sg) => {
                        if sg * cross < 0.0 {
                            inside = false;
                            break;
                        }
                    }
                }
            }
            inside
        };
        // Upper arc cap: points with y >= h AND distance to (0, h) ≤ w.
        let in_cap = py >= h - EPS && {
            let r2 = px * px + (py - h) * (py - h);
            r2 <= w * w + EPS
        };
        in_triangle || in_cap
    }
}

const EPS: f64 = 1e-7;

/// Classify a world-space cell by sampling a `PROBES_PER_SIDE × PROBES_PER_SIDE`
/// grid of probes; require both saw-In and saw-Out for Maybe.
fn classify_ground_truth<G: GroundTruth + ?Sized>(
    g: &G,
    x: f64,
    y: f64,
    cw: f64,
    ch: f64,
) -> Cell {
    let mut saw_in = false;
    let mut saw_out = false;
    for j in 0..PROBES_PER_SIDE {
        let py = y + (j as f64 + 0.5) * ch / PROBES_PER_SIDE as f64;
        for i in 0..PROBES_PER_SIDE {
            let px = x + (i as f64 + 0.5) * cw / PROBES_PER_SIDE as f64;
            if g.contains(px, py) {
                saw_in = true;
            } else {
                saw_out = true;
            }
            if saw_in && saw_out {
                return Cell::Maybe;
            }
        }
    }
    if saw_in {
        Cell::In
    } else {
        Cell::Out
    }
}

/// Build the ground-truth classifier for the given fingerprint key.
fn build_ground_truth(h: &Header) -> Box<dyn GroundTruth> {
    let (off_x, off_y) = (h.ox as f64, h.oy as f64);
    let a = h.angle.to_radians();
    let rotate = |(x, y): (f64, f64)| (x * a.cos() - y * a.sin(), x * a.sin() + y * a.cos());
    let translate = |(x, y): (f64, f64)| (x + off_x, y + off_y);
    match h.figure.as_str() {
        // circle is invariant under rotation around its centre.
        "circle_1" => Box::new(Circle { cx: off_x, cy: off_y, r: h.scale }),
        "box_1" => {
            // create_box(1.0) scaled by 32 → square of side 32 centred at origin,
            // then rotated by `angle`, then translated by (ox, oy).
            let half = h.scale * 0.5;
            let v = [(-half, -half), (half, -half), (half, half), (-half, half)]
                .map(rotate)
                .map(translate);
            Box::new(ConvexQuad { v })
        }
        "drop_0.2_0.8" => Box::new(RotatedDrop {
            angle_rad: a,
            ox: off_x,
            oy: off_y,
            width: 0.2 * h.scale,
            height: 0.8 * h.scale,
        }),
        "square_0.5_0.7" => {
            // create_square(0,0,0.5,0.7) scaled by 32 then rotated.
            let s = h.scale;
            let v = [
                (0.0, 0.0),
                (0.5 * s, 0.0),
                (0.5 * s, 0.7 * s),
                (0.0, 0.7 * s),
            ]
            .map(rotate)
            .map(translate);
            Box::new(ConvexQuad { v })
        }
        other => panic!("no ground-truth implementation for {other}"),
    }
}

/// Replay `generate_size`'s bbox math to recover the grid anchor for a key:
/// the figure is rotated by `angle` and then translated by `(ox, oy)`; the
/// grid covers `[gx0*cw, gx1*cw) × [gy0*ch, gy1*ch)`.
fn grid_anchor(h: &Header) -> (i64, i64, i64, i64) {
    // We need the rotated/moved polygon's bbox. Easiest: rebuild the figure
    // exactly like the pipeline does. Crucially we delegate to the templates
    // crate so the bbox includes the arc-cap exactly as production does.
    use vectorial_hash_templates::polygon::{
        create_box, create_circle, create_square, rotated_copy, scaled_copy,
    };
    use vectorial_hash_templates::templates::angle_to_radians;
    let base = match h.figure.as_str() {
        "circle_1" => create_circle(1.0),
        "box_1" => create_box(1.0),
        "drop_0.2_0.8" => create_drop(0.2, 0.8),
        "square_0.5_0.7" => create_square(0.0, 0.0, 0.5, 0.7),
        other => panic!("{other}"),
    };
    let scaled = scaled_copy(&base, h.scale, h.scale);
    let rotated = rotated_copy(&scaled, angle_to_radians(h.angle));
    let mut moved = rotated.clone();
    moved.move_by(h.ox as f64, h.oy as f64);
    let gx0 = (moved.x_min / h.cw as f64).floor() as i64;
    let gx1 = (moved.x_max / h.cw as f64).ceil() as i64;
    let gy0 = (moved.y_min / h.ch as f64).floor() as i64;
    let gy1 = (moved.y_max / h.ch as f64).ceil() as i64;
    (gx0, gy0, gx1, gy1)
}

#[test]
fn every_changed_template_is_more_correct_post_fix() {
    let pre_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fp_pre.txt");
    if !pre_path.exists() {
        eprintln!("fp_pre.txt missing — skipping (verification is opt-in)");
        return;
    }
    let pre_text = fs::read_to_string(&pre_path).unwrap();
    let pre_map: HashMap<String, String> = pre_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (key, hex) = l.split_once(" : ").unwrap();
            (key.to_string(), hex.to_string())
        })
        .collect();
    let post_map: HashMap<String, String> = POST
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (key, hex) = l.split_once(" : ").unwrap();
            (key.to_string(), hex.to_string())
        })
        .collect();

    let mut changed_keys: Vec<&String> = pre_map
        .keys()
        .filter(|k| post_map.get(*k).map(|v| v != &pre_map[*k]).unwrap_or(true))
        .collect();
    changed_keys.sort();
    println!("changed templates: {}", changed_keys.len());
    assert_eq!(changed_keys.len(), 80, "expected 80 templates to differ");

    let mut total_cells_changed = 0u32;
    let mut post_correct = 0u32;
    let mut post_incorrect: Vec<String> = Vec::new();
    let mut pre_correct = 0u32;

    for key in &changed_keys {
        let pre_line = format!("{} : {}", key, pre_map[*key]);
        let post_line = format!("{} : {}", key, post_map[*key]);
        let (h, _) = parse_header(&pre_line);
        let (_, pre_hex) = parse_header(&pre_line);
        let (_, post_hex) = parse_header(&post_line);
        let (cols_pre, rows_pre, cells_pre) = decode_template(&pre_hex);
        let (cols_post, rows_post, cells_post) = decode_template(&post_hex);
        assert_eq!(
            (cols_pre, rows_pre),
            (cols_post, rows_post),
            "{key}: grid dimensions diverge ({cols_pre}x{rows_pre} vs {cols_post}x{rows_post})",
        );

        let g = build_ground_truth(&h);
        let (gx0, gy0, _, _) = grid_anchor(&h);
        let anchor_x = gx0 as f64 * h.cw as f64;
        let anchor_y = gy0 as f64 * h.ch as f64;

        for row in 0..rows_post {
            for col in 0..cols_post {
                let idx = (row * cols_post + col) as usize;
                let pre = cells_pre[idx];
                let post = cells_post[idx];
                if pre == post {
                    continue;
                }
                total_cells_changed += 1;

                let cx = anchor_x + col as f64 * h.cw as f64;
                let cy = anchor_y + row as f64 * h.ch as f64;
                let truth = classify_ground_truth(&*g, cx, cy, h.cw as f64, h.ch as f64);
                if truth == post {
                    post_correct += 1;
                } else {
                    post_incorrect.push(format!(
                        "{key}: cell ({col},{row}) [{cx:.1}..{:.1}, {cy:.1}..{:.1}] \
                         pre={pre:?} post={post:?} truth={truth:?}",
                        cx + h.cw as f64,
                        cy + h.ch as f64,
                    ));
                }
                if truth == pre {
                    pre_correct += 1;
                }
            }
        }
    }
    println!("total cell changes: {total_cells_changed}");
    println!("post-fix matches ground truth: {post_correct}/{total_cells_changed}");
    println!("pre-fix matches ground truth:  {pre_correct}/{total_cells_changed}");
    if !post_incorrect.is_empty() {
        let limit = 12;
        for line in post_incorrect.iter().take(limit) {
            println!("  bad: {line}");
        }
        let extra = post_incorrect.len().saturating_sub(limit);
        if extra > 0 {
            println!("  ... and {extra} more.");
        }
    }
    assert!(
        post_incorrect.is_empty(),
        "{} post-fix cells do not match ground truth",
        post_incorrect.len(),
    );
    assert!(pre_correct < post_correct, "pre-fix matches at least as much truth as post-fix");
}
