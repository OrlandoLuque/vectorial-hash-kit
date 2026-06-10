//! Regression tests for numeric degeneracies found by the exhaustive
//! culling campaign (see `exhaustive_culling.rs`).

use vectorial_hash::{CellState, Point};
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{create_box, rotated_copy};
use vectorial_hash_templates::templates::angle_to_radians;

/// Campaign seed 543: a box rotated 135° puts two vertices at the test
/// point's exact height; the horizontal point-in-polygon ray then grazed
/// those vertices and produced unstable crossing counts — whole raster rows
/// through the middle of the figure came out `Out`. `is_inside` now picks a
/// ray that stays clear of vertices/arc tangencies.
#[test]
fn rotated_box_raster_has_no_out_rows_through_the_middle() {
    let side = 73.0;
    let angle = 135.0;
    let origin = (194i64, 181i64);
    let base = create_box(side);
    let figure = FigureKey::new(2, &[side]);

    let mut bank = TemplateBank::new();
    bank.generate_size(&figure, &base, &[angle], 1, 1);
    let raster = bank.placed_raster(&figure, angle, origin).unwrap();

    let mut poly = rotated_copy(&base, angle_to_radians(angle));
    poly.move_by(origin.0 as f64, origin.1 as f64);

    // Sweep a horizontal band through the figure's centre: every pixel that
    // the exact test says is inside (and not boundary-adjacent) must be In.
    for dy in -3i64..=3 {
        for dx in -30i64..=30 {
            let p = Point::new(origin.0 as f64 + dx as f64, origin.1 as f64 + dy as f64);
            if poly.is_inside(p.x, p.y) && poly.is_inside(p.x + 1.5, p.y) && poly.is_inside(p.x - 1.5, p.y)
                && poly.is_inside(p.x, p.y + 1.5) && poly.is_inside(p.x, p.y - 1.5)
            {
                assert_eq!(
                    raster.cell_at_world(p),
                    CellState::In,
                    "interior pixel at ({}, {}) must be In",
                    p.x, p.y,
                );
            }
        }
    }
}

/// The direct degeneracy: ray through two diamond vertices. Must be inside
/// regardless of the figure's float offset.
#[test]
fn is_inside_is_stable_when_ray_would_graze_vertices() {
    let base = create_box(73.0);
    let diamond = rotated_copy(&base, angle_to_radians(135.0));
    // Deep inside, exactly at the height of the left/right vertices.
    assert!(diamond.is_inside(-15.0, 0.0));
    assert!(diamond.is_inside(15.0, 0.0));
    let mut moved = diamond.clone();
    moved.move_by(194.0, 181.0);
    assert!(moved.is_inside(179.0, 181.0));
    assert!(moved.is_inside(209.0, 181.0));
}
