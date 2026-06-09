use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use rayon::prelude::*;

use vectorial_hash_templates::polygon::{create_drop, create_box, create_circle, scaled_copy, rotated_copy};
use vectorial_hash_templates::templates::{TemplateStore, angle_to_radians, get_angles, get_template_grid_fast};
use vectorial_hash_templates::{task, comparison_test};
#[cfg(feature = "redis-store")]
use vectorial_hash_templates::matrix;

#[derive(Parser)]
#[command(name = "vh", about = "vectorial-hash command-line tools")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate templates in-memory (single process).
    Generate,
    /// Parallel subtask benchmark (grids 16+32).
    Compare,
    /// Heavy benchmark (grids 16+32+64+128).
    Heavy,
    /// Generate templates and store them in Redis (multi-process).
    #[cfg(feature = "redis-store")]
    GenerateRedis {
        #[arg(long, default_value = "127.0.0.1")]
        redis_host: String,
        #[arg(long, default_value_t = 6379)]
        redis_port: u16,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Generate => run_in_memory(),
        Cmd::Compare => comparison_test::run_comparison(),
        Cmd::Heavy => comparison_test::run_heavy(),
        #[cfg(feature = "redis-store")]
        Cmd::GenerateRedis { redis_host, redis_port } => {
            run_with_redis(&redis_host, redis_port);
        }
    }
}

fn run_in_memory() {
    let polygons = vec![
        ("drop", create_drop(0.2, 0.8)),
        ("box", create_box(1.0)),
        ("circle", create_circle(1.0)),
    ];
    let scales: Vec<f64> = vec![128.0];
    let grid_sizes: Vec<i64> = vec![16];
    let angle_step = 0.5;
    let angles = get_angles(angle_step);
    let max_per_task: u64 = 500_000;

    let subtasks = task::create_subtasks(
        &polygons, &scales, &grid_sizes, angles.len(), max_per_task,
    );
    let total_tasks = subtasks.len();

    println!("Config: {} polygons, {} scales, {} grids, {} angles (step {}deg)",
        polygons.len(), scales.len(), grid_sizes.len(), angles.len(), angle_step);
    task::print_summary(&subtasks);
    println!("Threads: {} | Mode: In-memory\n", rayon::current_num_threads());

    let store = Arc::new(TemplateStore::new());
    let completed = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    subtasks.par_iter().for_each(|st| {
        process_subtask_mem(st, &angles, &store, &completed, total_tasks);
    });

    println!("\n=== COMPLETE ===");
    println!("Total time: {:.2}s", start.elapsed().as_secs_f64());
    println!("Unique templates: {}", store.template_count());
    println!("Total combinations: {}", store.generation_count());
}

fn process_subtask_mem(
    st: &task::SubTask,
    angles: &[f64],
    store: &Arc<TemplateStore>,
    completed: &Arc<AtomicUsize>,
    total_tasks: usize,
) {
    let task_start = Instant::now();
    let scaled = scaled_copy(&st.poly, st.scale, st.scale);
    let gx = st.grid_size;
    let gy = st.grid_size;
    let mut new_count = 0u32;

    for angle_idx in st.angle_start..st.angle_end {
        let angle = angles[angle_idx];
        let rotated = rotated_copy(&scaled, angle_to_radians(angle));

        for x in 0..gx {
            for y in 0..gy {
                let mut moved = rotated.clone();
                moved.move_by(x as f64, y as f64);

                let gxr = [
                    (moved.x_min / gx as f64).floor() as i64,
                    (moved.x_max / gx as f64).ceil() as i64,
                ];
                let gyr = [
                    (moved.y_min / gy as f64).floor() as i64,
                    (moved.y_max / gy as f64).ceil() as i64,
                ];

                let tpl = get_template_grid_fast(
                    gxr[0], gyr[0], gxr[1], gyr[1], gx, gy, &moved,
                );

                let gen_string = format!("{}-s{}-x{},y{}-a{}-dx{},dy{}",
                    st.poly_name, st.scale as i64, gx, gy, angle, x, y);

                let (_, _, is_new) = store.store_dedup(&tpl, &gen_string);
                if is_new { new_count += 1; }
            }
        }
    }

    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
    let elapsed = task_start.elapsed();
    println!("  [{}/{}] {} s{} {}x{} a[{}..{}] | {} new | {:.2}s",
        done, total_tasks, st.poly_name, st.scale as i64,
        gx, gy, st.angle_start, st.angle_end,
        new_count, elapsed.as_secs_f64());
}

#[cfg(feature = "redis-store")]
fn run_with_redis(host: &str, port: u16) {
    use vectorial_hash_templates::redis_store::RedisStore;

    let polygons = vec![
        ("drop", create_drop(0.2, 0.8)),
        ("box", create_box(1.0)),
        ("circle", create_circle(1.0)),
    ];
    let scales: Vec<f64> = vec![128.0];
    let grid_sizes: Vec<i64> = vec![16];
    let angle_step = 0.5;
    let angles = get_angles(angle_step);
    let max_per_task: u64 = 500_000;

    let subtasks = task::create_subtasks(
        &polygons, &scales, &grid_sizes, angles.len(), max_per_task,
    );
    let total_tasks = subtasks.len();

    let redis = match RedisStore::connect(host, port) {
        Ok(r) => Arc::new(r),
        Err(e) => { eprintln!("ERROR: {}", e); return; }
    };
    println!("Connected to Redis at {}:{}", host, port);
    task::print_summary(&subtasks);

    let completed = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    subtasks.par_iter().for_each(|st| {
        let task_key = format!("T{}-lock", st.id + 1);
        if !redis.try_lock_task(&task_key) { return; }

        let task_start = Instant::now();
        let scaled = scaled_copy(&st.poly, st.scale, st.scale);
        let gx = st.grid_size;
        let gy = st.grid_size;
        let mut new_count = 0u32;
        let mut iteration = 0u64;

        for angle_idx in st.angle_start..st.angle_end {
            let angle = angles[angle_idx];
            let rotated = rotated_copy(&scaled, angle_to_radians(angle));

            for x in 0..gx {
                for y in 0..gy {
                    let mut moved = rotated.clone();
                    moved.move_by(x as f64, y as f64);

                    let gxr = [
                        (moved.x_min / gx as f64).floor() as i64,
                        (moved.x_max / gx as f64).ceil() as i64,
                    ];
                    let gyr = [
                        (moved.y_min / gy as f64).floor() as i64,
                        (moved.y_max / gy as f64).ceil() as i64,
                    ];

                    let tpl = get_template_grid_fast(
                        gxr[0], gyr[0], gxr[1], gyr[1], gx, gy, &moved,
                    );
                    let gen_string = format!("{}-s{}-x{},y{}-a{}-dx{},dy{}",
                        st.poly_name, st.scale as i64, gx, gy, angle, x, y);

                    let transforms = matrix::all_transforms(&tpl);
                    let hashes: Vec<Vec<u8>> = transforms.iter()
                        .map(|m| matrix::bin_code(m)).collect();
                    let (_, is_new) = redis.store_template(
                        &hashes, &gen_string,
                        "templateCount", "templateList", "generatedSet",
                    );
                    if is_new { new_count += 1; }

                    iteration += 1;
                    if iteration % 1000 == 0 {
                        redis.keep_lock(&task_key);
                    }
                }
            }
        }

        redis.complete_task(&task_key);
        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = task_start.elapsed();
        println!("  [{}/{}] {} s{} {}x{} a[{}..{}] | {} new | {:.2}s",
            done, total_tasks, st.poly_name, st.scale as i64,
            gx, gy, st.angle_start, st.angle_end,
            new_count, elapsed.as_secs_f64());
    });

    let count = redis.get_template_count("templateCount");
    println!("\n=== COMPLETE ===");
    println!("Total time: {:.2}s", start.elapsed().as_secs_f64());
    println!("Templates in Redis: {}", count);
}
