//! Deterministic template fingerprint used to detect regressions in the
//! template-generation pipeline.
//!
//! A fixed set (four figures × eight angles × four cell sizes × every
//! sub-cell offset) is generated and each template is encoded with
//! [`crate::matrix::bin_code`] (the PHP-compatible binary format). The
//! returned text is one line per template, ordered deterministically — feed
//! it to a snapshot test against `tests/fixtures/template_fingerprint.txt`.
//!
//! Updating the fixture is an explicit step: any change in the cell values
//! shows up as a diff in the test failure, so a legitimate change (e.g. the
//! 2026-06 ray-degeneracy fix that affected 88/14,361 cells around
//! horizontal-tangent configurations) requires regenerating the file via
//! `vh templates-fingerprint > crates/vectorial-hash-templates/tests/fixtures/template_fingerprint.txt`.

use std::fmt::Write;

use crate::matrix;
use crate::polygon::{
    create_box, create_circle, create_drop, create_square, rotated_copy, scaled_copy, Polygon,
};
use crate::templates::{angle_to_radians, get_template_grid_fast};

/// Generate the canonical template fingerprint as a string.
pub fn generate() -> String {
    let figures: Vec<(&str, f64, Polygon)> = vec![
        ("drop_0.2_0.8", 32.0, create_drop(0.2, 0.8)),
        ("circle_1", 32.0, create_circle(1.0)),
        ("box_1", 32.0, create_box(1.0)),
        ("square_0.5_0.7", 32.0, create_square(0.0, 0.0, 0.5, 0.7)),
    ];
    let angles = [0.0_f64, 15.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0];
    let cells: [(i64, i64); 4] = [(1, 1), (8, 8), (16, 16), (8, 16)];

    let mut out = String::new();
    for (name, scale, base) in &figures {
        let scaled = scaled_copy(base, *scale, *scale);
        for &angle in &angles {
            let rotated = rotated_copy(&scaled, angle_to_radians(angle));
            for &(cw, ch) in &cells {
                let cw_u = cw as u32;
                let ch_u = ch as u32;
                for ox in 0..cw_u {
                    for oy in 0..ch_u {
                        let mut moved = rotated.clone();
                        moved.move_by(ox as f64, oy as f64);
                        let gx0 = (moved.x_min / cw as f64).floor() as i64;
                        let gx1 = (moved.x_max / cw as f64).ceil() as i64;
                        let gy0 = (moved.y_min / ch as f64).floor() as i64;
                        let gy1 = (moved.y_max / ch as f64).ceil() as i64;
                        let m = get_template_grid_fast(gx0, gy0, gx1, gy1, cw, ch, &moved);
                        let bytes = matrix::bin_code(&m);
                        write!(
                            out,
                            "{name} s{scale} a{angle} c{cw}x{ch} o{ox},{oy} : ",
                        )
                        .unwrap();
                        for b in &bytes {
                            write!(out, "{b:02x}").unwrap();
                        }
                        out.push('\n');
                    }
                }
            }
        }
    }
    out
}
