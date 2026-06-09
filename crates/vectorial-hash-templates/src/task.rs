use crate::polygon::Polygon;

/// A subtask: one (polygon, scale, grid) combination with a range of angles
#[derive(Clone)]
pub struct SubTask {
    pub id: usize,
    pub poly_name: String,
    pub poly: Polygon,
    pub scale: f64,
    pub grid_size: i64,
    pub angle_start: usize, // index into angles array
    pub angle_end: usize,   // exclusive
    pub combinations: u64,
}

/// Create subtasks with approximately equal work, splitting by angle ranges
pub fn create_subtasks(
    polygons: &[(&str, Polygon)],
    scales: &[f64],
    grid_sizes: &[i64],
    angle_count: usize,
    max_combinations: u64,
) -> Vec<SubTask> {
    let mut tasks = Vec::new();
    let mut id = 0;

    for (name, poly) in polygons {
        for &scale in scales {
            for &grid_size in grid_sizes {
                let positions = (grid_size * grid_size) as u64;
                let angles_per_subtask = (max_combinations / positions).max(1) as usize;
                let mut angle_start = 0;

                while angle_start < angle_count {
                    let angle_end = (angle_start + angles_per_subtask).min(angle_count);
                    let combos = (angle_end - angle_start) as u64 * positions;

                    tasks.push(SubTask {
                        id,
                        poly_name: name.to_string(),
                        poly: poly.clone(),
                        scale,
                        grid_size,
                        angle_start,
                        angle_end,
                        combinations: combos,
                    });
                    id += 1;
                    angle_start = angle_end;
                }
            }
        }
    }
    tasks
}

/// Print task distribution summary
pub fn print_summary(tasks: &[SubTask]) {
    let total_combos: u64 = tasks.iter().map(|t| t.combinations).sum();
    let min_combos = tasks.iter().map(|t| t.combinations).min().unwrap_or(0);
    let max_combos = tasks.iter().map(|t| t.combinations).max().unwrap_or(0);

    println!("Tasks: {} | Total combinations: {} | Per task: {}-{} (ratio {:.1}x)",
        tasks.len(),
        total_combos,
        min_combos,
        max_combos,
        max_combos as f64 / min_combos.max(1) as f64,
    );
}
