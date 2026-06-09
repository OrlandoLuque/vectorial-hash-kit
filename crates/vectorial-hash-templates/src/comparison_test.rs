use crate::polygon::*;
use crate::templates::{self, *};
use crate::task;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use rayon::prelude::*;

pub fn run_comparison() {
    run_bench("=== Parallel subtask test (grids 16+32) ===", vec![16, 32]);
}

pub fn run_heavy() {
    run_bench("=== HEAVY benchmark (grids 16+32+64+128) ===", vec![16, 32, 64, 128]);
}

fn run_bench(title: &str, grid_sizes: Vec<i64>) {
    let polys = vec![
        ("drop", create_drop(0.2, 0.8)),
        ("box", create_box(1.0)),
        ("circle", create_circle(1.0)),
    ];
    let scales: Vec<f64> = vec![128.0];
    let angles = get_angles(0.5);
    let max_per_task: u64 = 500_000;

    let subtasks = task::create_subtasks(&polys, &scales, &grid_sizes, angles.len(), max_per_task);
    let store = Arc::new(TemplateStore::new());
    let completed = Arc::new(AtomicUsize::new(0));
    let total = subtasks.len();

    println!("{}", title);
    task::print_summary(&subtasks);
    println!("Threads: {}\n", rayon::current_num_threads());

    let start = Instant::now();

    subtasks.par_iter().for_each(|st| {
        let t = Instant::now();
        let scaled = scaled_copy(&st.poly, st.scale, st.scale);
        let gx = st.grid_size;
        let gy = st.grid_size;
        let mut new_count = 0u32;

        for ai in st.angle_start..st.angle_end {
            let rotated = rotated_copy(&scaled, angle_to_radians(angles[ai]));
            for x in 0..gx {
                for y in 0..gy {
                    let mut moved = rotated.clone();
                    moved.move_by(x as f64, y as f64);
                    let gxr = [(moved.x_min / gx as f64).floor() as i64,
                               (moved.x_max / gx as f64).ceil() as i64];
                    let gyr = [(moved.y_min / gy as f64).floor() as i64,
                               (moved.y_max / gy as f64).ceil() as i64];
                    let tpl = templates::get_template_grid_fast(
                        gxr[0], gyr[0], gxr[1], gyr[1], gx, gy, &moved);
                    let (_, _, is_new) = store.store_dedup(&tpl, "");
                    if is_new { new_count += 1; }
                }
            }
        }

        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 5 == 0 || done == total {
            println!("  [{}/{}] {} s{} {}x{} a[{}..{}] | {} new | {:.2}s",
                done, total, st.poly_name, st.scale as i64,
                gx, gy, st.angle_start, st.angle_end,
                new_count, t.elapsed().as_secs_f64());
        }
    });

    println!("\n  Total: {:.2}s | {} unique templates | {} combinations",
        start.elapsed().as_secs_f64(),
        store.template_count(),
        store.generation_count());
}
