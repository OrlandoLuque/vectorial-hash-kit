use vectorial_hash_templates::polygon::{create_drop, scaled_copy, rotated_copy};
use vectorial_hash_templates::templates::{TemplateStore, angle_to_radians, get_template_grid_fast};

fn main() {
    println!("vectorial-hash demo: tiny in-memory template generation\n");

    let poly = create_drop(0.2, 0.8);
    let scaled = scaled_copy(&poly, 64.0, 64.0);
    let store = TemplateStore::new();
    let grid: i64 = 16;

    let angles = [0.0, 45.0, 90.0, 135.0];
    for angle in angles {
        let rotated = rotated_copy(&scaled, angle_to_radians(angle));
        let mut moved = rotated.clone();
        moved.move_by(0.0, 0.0);

        let gxr = [
            (moved.x_min / grid as f64).floor() as i64,
            (moved.x_max / grid as f64).ceil() as i64,
        ];
        let gyr = [
            (moved.y_min / grid as f64).floor() as i64,
            (moved.y_max / grid as f64).ceil() as i64,
        ];

        let tpl = get_template_grid_fast(gxr[0], gyr[0], gxr[1], gyr[1], grid, grid, &moved);
        let (id, op, is_new) = store.store_dedup(&tpl, &format!("drop-a{}", angle));
        println!("  angle {:>5.1}deg -> id {} via {} (new: {})", angle, id, op, is_new);
    }

    println!("\nUnique templates: {}", store.template_count());
}
