//! `vh bench`: 4-way cull benchmark.
//!
//! Compares, over the same point set and the same query polygon:
//!
//! 1. vectorial-hash binary-split tree + template short-circuit
//! 2. vectorial-hash binary-split tree, per-point polygon test only
//! 3. quadtree + template short-circuit
//! 4. quadtree, per-point polygon test only
//!
//! All four must return the same hit count; the bench asserts it.

use std::hint::black_box;
use std::time::{Duration, Instant};

use vectorial_hash::{PlacedTemplate, Point, Positioned, Rect, Shape, TemplateGrid, Tree};
use vectorial_hash_templates::adapter::matrix_to_template_grid;
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::{angle_to_radians, get_template_grid_fast};

use vectorial_hash::QuadTree;

const WORLD: f64 = 4096.0;
const TEMPLATE_CELL: i64 = 64;

#[derive(Clone, Copy)]
struct Pt(Point);
impl Positioned for Pt {
    fn position(&self) -> Point {
        self.0
    }
}

/// Polygon query that exposes its precomputed template.
struct TplShape {
    poly: Polygon,
    bbox: Rect,
    grid: TemplateGrid,
}
impl Shape for TplShape {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_grid(&self) -> Option<&TemplateGrid> {
        Some(&self.grid)
    }
}

/// Same polygon query without the template (bbox + per-point fallback).
struct PlainShape {
    poly: Polygon,
    bbox: Rect,
}
impl Shape for PlainShape {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
}

/// xorshift64* — deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Deterministic point cloud: uniform, or `clusters` gaussian-ish blobs
/// (sum of uniforms) when `clusters > 0`. Returns the points and, in
/// clustered mode, the first cluster centre (so the query can be aimed at a
/// populated area instead of empty space).
fn make_points(rng: &mut Rng, points: usize, clusters: usize) -> (Vec<Pt>, Option<(f64, f64)>) {
    if clusters == 0 {
        let pts = (0..points)
            .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
            .collect();
        return (pts, None);
    }
    let centers: Vec<(f64, f64)> = (0..clusters)
        .map(|_| {
            (
                WORLD * (0.1 + 0.8 * rng.unit_f64()),
                WORLD * (0.1 + 0.8 * rng.unit_f64()),
            )
        })
        .collect();
    let sigma = WORLD / 40.0;
    let pts = (0..points)
        .map(|_| {
            let (cx, cy) = centers[(rng.next_u64() as usize) % clusters];
            let dx = (rng.unit_f64() + rng.unit_f64() + rng.unit_f64() + rng.unit_f64() - 2.0) * sigma;
            let dy = (rng.unit_f64() + rng.unit_f64() + rng.unit_f64() + rng.unit_f64() - 2.0) * sigma;
            Pt(Point::new(
                (cx + dx).clamp(0.0, WORLD - 1e-9),
                (cy + dy).clamp(0.0, WORLD - 1e-9),
            ))
        })
        .collect();
    (pts, Some(centers[0]))
}

pub fn run(points: usize, culls: usize, item_limit: usize, seed: u64, clusters: usize) {
    println!("=== Cull benchmark: binary-split tree vs quadtree, templates on/off ===");
    println!(
        "world {w}x{w} | {points} points ({dist}) | item_limit {item_limit} | {culls} culls/config | seed {seed}\n",
        w = WORLD as i64,
        dist = if clusters == 0 {
            "uniform".to_string()
        } else {
            format!("{clusters} clusters")
        },
    );

    // Points first: in clustered mode the query is aimed at the first
    // cluster so it actually covers populated space.
    let mut rng = Rng(seed.max(1));
    let (pts, query_center) = make_points(&mut rng, points, clusters);

    // Query polygon: a big rotated drop.
    let mut poly = rotated_copy(
        &scaled_copy(&create_drop(0.2, 0.8), 1400.0, 1400.0),
        angle_to_radians(30.0),
    );
    match query_center {
        Some((tx, ty)) => {
            let bcx = (poly.x_min + poly.x_max) / 2.0;
            let bcy = (poly.y_min + poly.y_max) / 2.0;
            poly.move_by(tx - bcx, ty - bcy);
        }
        None => poly.move_by(WORLD / 2.0, WORLD / 4.0),
    }
    let bbox = Rect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );

    // Precompute its template once (this is the "offline" step).
    let tpl_start = Instant::now();
    let gxr = [
        (poly.x_min / TEMPLATE_CELL as f64).floor() as i64,
        (poly.x_max / TEMPLATE_CELL as f64).ceil() as i64,
    ];
    let gyr = [
        (poly.y_min / TEMPLATE_CELL as f64).floor() as i64,
        (poly.y_max / TEMPLATE_CELL as f64).ceil() as i64,
    ];
    let matrix = get_template_grid_fast(
        gxr[0], gyr[0], gxr[1], gyr[1], TEMPLATE_CELL, TEMPLATE_CELL, &poly,
    );
    let anchor = Point::new(
        gxr[0] as f64 * TEMPLATE_CELL as f64,
        gyr[0] as f64 * TEMPLATE_CELL as f64,
    );
    let grid = matrix_to_template_grid(&matrix, anchor, TEMPLATE_CELL as f64, TEMPLATE_CELL as f64);
    println!(
        "query: drop polygon scale 1400 @30deg | template {}x{} cells of {}px (built in {:.1} ms)\n",
        grid.cols,
        grid.rows,
        TEMPLATE_CELL,
        tpl_start.elapsed().as_secs_f64() * 1000.0,
    );

    let tpl_shape = TplShape { poly: poly.clone(), bbox, grid };
    let plain_shape = PlainShape { poly, bbox };

    let world = Rect::new(0.0, 0.0, WORLD, WORLD);

    let t = Instant::now();
    let mut vtree: Tree<Pt> = Tree::new(world, item_limit);
    for p in &pts {
        vtree.insert(*p);
    }
    let vtree_build = t.elapsed();

    let t = Instant::now();
    let mut qtree: QuadTree<Pt> = QuadTree::new(world, item_limit);
    for p in &pts {
        qtree.insert(*p);
    }
    let qtree_build = t.elapsed();

    println!("build:");
    println!(
        "  vectorial-hash tree  {:>8.1} ms  ({} nodes)",
        vtree_build.as_secs_f64() * 1000.0,
        vtree.node_count(),
    );
    println!(
        "  quadtree             {:>8.1} ms  ({} nodes)\n",
        qtree_build.as_secs_f64() * 1000.0,
        qtree.node_count(),
    );

    // Correctness gate: the 4 configurations must agree.
    let h1 = vtree.cull(&tpl_shape).len();
    let h2 = vtree.cull(&plain_shape).len();
    let h3 = qtree.cull(&tpl_shape).len();
    let h4 = qtree.cull(&plain_shape).len();
    assert!(
        h1 == h2 && h2 == h3 && h3 == h4,
        "configs disagree: vh+tpl={h1} vh={h2} qt+tpl={h3} qt={h4}",
    );
    println!("correctness: all 4 configs return {h1} hits — OK\n");

    let mut results: Vec<(&str, Duration)> = Vec::new();

    let t = Instant::now();
    for _ in 0..culls {
        black_box(vtree.cull(&tpl_shape).len());
    }
    results.push(("vectorial + templates", t.elapsed()));

    let t = Instant::now();
    for _ in 0..culls {
        black_box(vtree.cull(&plain_shape).len());
    }
    results.push(("vectorial (no templates)", t.elapsed()));

    let t = Instant::now();
    for _ in 0..culls {
        black_box(qtree.cull(&tpl_shape).len());
    }
    results.push(("quadtree + templates", t.elapsed()));

    let t = Instant::now();
    for _ in 0..culls {
        black_box(qtree.cull(&plain_shape).len());
    }
    results.push(("quadtree (no templates)", t.elapsed()));

    println!("{:<26} {:>12} {:>12}", "config", "total (ms)", "avg/cull (ms)");
    for (name, d) in &results {
        println!(
            "{:<26} {:>12.2} {:>12.3}",
            name,
            d.as_secs_f64() * 1000.0,
            d.as_secs_f64() * 1000.0 / culls as f64,
        );
    }

    let ms = |i: usize| results[i].1.as_secs_f64() * 1000.0;
    println!("\nspeedups:");
    println!("  templates on vectorial tree : {:>6.2}x", ms(1) / ms(0));
    println!("  templates on quadtree       : {:>6.2}x", ms(3) / ms(2));
    println!("  vectorial vs quadtree (tpl) : {:>6.2}x", ms(2) / ms(0));
    println!("  vectorial vs quadtree (raw) : {:>6.2}x", ms(3) / ms(1));
}

/// `vh bench-fallback`: granularity-as-fallback study.
///
/// Compares, over the same tree and the same query, three culling configs:
///
/// 1. **no template**: the tree has nothing to short-circuit with.
/// 2. **fine set only + aggregate-on-the-fly**: the bank contains only one
///    small cell size (e.g. 8×8); the cull asks for every other size and
///    receives an aggregated grid. Lossless per the .docx property —
///    classification matches a directly-generated set exactly.
/// 3. **every size precomputed**: the bank already has sets for the full
///    family of cell sizes the tree uses; no aggregation needed.
///
/// All three configs must return the same hit count (asserted) before
/// timing. The interesting question is config #2's overhead vs #3, since #2
/// trades precomputation memory and time for per-cull aggregation work.
pub fn run_fallback(points: usize, culls: usize, item_limit: usize, seed: u64, scale: f64) {
    const ANGLE: f64 = 30.0;
    let origin = (2048i64, 1024i64);

    println!("=== Cull benchmark: granularity-as-fallback aggregation ===");
    println!(
        "world {w}x{w} | {points} points | item_limit {item_limit} | {culls} culls/config | seed {seed}",
        w = WORLD as i64,
    );
    println!("query: drop scale {scale} @{ANGLE}deg, origin {origin:?}\n");

    let base = scaled_copy(&create_drop(0.2, 0.8), scale, scale);
    let figure = FigureKey::new(0, &[0.2 * scale, 0.8 * scale]);
    let mut poly = rotated_copy(&base, angle_to_radians(ANGLE));
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = Rect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );

    let mut rng = Rng(seed.max(1));
    let pts: Vec<Pt> = (0..points)
        .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
        .collect();
    let world = Rect::new(0.0, 0.0, WORLD, WORLD);
    let mut tree: Tree<Pt> = Tree::new(world, item_limit);
    for p in &pts {
        tree.insert(*p);
    }

    // Three banks.
    let t = Instant::now();
    let mut bank_full = TemplateBank::new();
    for &(w, h) in &[
        (1u32, 1u32), (8, 8), (8, 16), (16, 8), (16, 16),
        (16, 32), (32, 16), (32, 32), (32, 64), (64, 32), (64, 64),
    ] {
        bank_full.generate_size(&figure, &base, &[ANGLE], w, h);
    }
    let full_gen = t.elapsed();
    let full_mem = bank_full.memory_usage();

    let t = Instant::now();
    let mut bank_small = TemplateBank::new();
    for &(w, h) in &[(1u32, 1u32), (8, 8), (8, 16), (16, 8)] {
        bank_small.generate_size(&figure, &base, &[ANGLE], w, h);
    }
    let small_gen = t.elapsed();
    let small_mem = bank_small.memory_usage();

    println!(
        "bank full   ({}us gen, {} combos, {} unique grids, {:.2} MB) — every cell size precomputed",
        full_gen.as_micros(),
        bank_full.entry_count(),
        bank_full.unique_count(),
        full_mem.total() as f64 / 1e6,
    );
    println!(
        "bank small  ({}us gen, {} combos, {} unique grids, {:.2} MB) — only sizes <=16; rest aggregated\n",
        small_gen.as_micros(),
        bank_small.entry_count(),
        bank_small.unique_count(),
        small_mem.total() as f64 / 1e6,
    );

    let raster_full = bank_full.placed_raster(&figure, ANGLE, origin);
    let raster_small = bank_small.placed_raster(&figure, ANGLE, origin);
    let plain = PlainShape { poly: poly.clone(), bbox };
    let shape_full = BankShape {
        bank: &bank_full,
        figure: figure.clone(),
        angle: ANGLE,
        origin,
        poly: poly.clone(),
        bbox,
        raster: raster_full,
    };
    let shape_small = FallbackShape {
        bank: &bank_small,
        figure: figure.clone(),
        angle: ANGLE,
        origin,
        poly: poly.clone(),
        bbox,
        raster: raster_small,
    };

    let expected = tree.cull(&plain).len();
    let got_full = tree.cull(&shape_full).len();
    let got_small = tree.cull(&shape_small).len();
    assert_eq!(got_full, expected, "full bank disagrees: {got_full} != {expected}");
    assert_eq!(got_small, expected, "small bank disagrees: {got_small} != {expected}");
    println!("correctness: all 3 configs return {expected} hits — OK\n");

    let mut results: Vec<(&str, Duration)> = Vec::new();

    let t = Instant::now();
    for _ in 0..culls {
        black_box(tree.cull(&plain).len());
    }
    results.push(("no templates", t.elapsed()));

    let t = Instant::now();
    for _ in 0..culls {
        black_box(tree.cull(&shape_small).len());
    }
    results.push(("bank <=16 + aggregated fallback", t.elapsed()));

    let t = Instant::now();
    for _ in 0..culls {
        black_box(tree.cull(&shape_full).len());
    }
    results.push(("bank full (every size precomputed)", t.elapsed()));

    println!("{:<38} {:>12} {:>14}", "config", "total (ms)", "avg/cull (ms)");
    let base_ms = results[0].1.as_secs_f64() * 1000.0;
    for (name, d) in &results {
        let ms = d.as_secs_f64() * 1000.0;
        println!(
            "{:<38} {:>12.2} {:>14.3}   ({:.2}x vs none)",
            name,
            ms,
            ms / culls as f64,
            base_ms / ms,
        );
    }
}

/// `vh bench-scale`: figure↔grid scale equivalence study.
///
/// One stored set covers many query scales for the same shape. Compare:
/// (a) bank with a separate set per query scale (the baseline);
/// (b) bank with **one** stored set, served at every query scale via
///     `placed_for_scaled` (no extra precomputation, no cell-data clones).
/// Both must return the same hit counts (asserted). Reports per-cull
/// timings and total memory.
pub fn run_scale(points: usize, culls: usize, item_limit: usize, seed: u64) {
    const ANGLE: f64 = 0.0; // box: angle is uninteresting here
    const BASE_DIM: f64 = 8.0;
    const FACTORS: [u32; 4] = [1, 2, 4, 8];
    let cells_at_factor = |f: u32| (f, f); // cell size scales with the figure

    println!("=== Cull benchmark: figure-grid scale equivalence ===");
    println!(
        "world {w}x{w} | {points} points | item_limit {item_limit} | {culls} culls/(config x query) | seed {seed}",
        w = WORLD as i64,
    );
    println!(
        "shapes: box side {BASE_DIM} × scales {:?} (one box per scale, queried at every factor)\n",
        FACTORS,
    );

    let mut rng = Rng(seed.max(1));
    let pts: Vec<Pt> = (0..points)
        .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
        .collect();
    let world = Rect::new(0.0, 0.0, WORLD, WORLD);
    let mut tree: Tree<Pt> = Tree::new(world, item_limit);
    for p in &pts {
        tree.insert(*p);
    }

    let base = vectorial_hash_templates::polygon::create_square(0.0, 0.0, BASE_DIM, BASE_DIM);
    let canonical_fig = FigureKey::new(0, &[BASE_DIM]);

    // Bank A: one stored set (canonical, factor 1), served via placed_for_scaled.
    let t = Instant::now();
    let mut bank_one = TemplateBank::new();
    bank_one.generate_size(&canonical_fig, &base, &[ANGLE], 1, 1);
    let (cw_canon, ch_canon) = cells_at_factor(1);
    bank_one.generate_size(&canonical_fig, &base, &[ANGLE], cw_canon, ch_canon);
    let one_gen = t.elapsed();
    let one_mem = bank_one.memory_usage();

    // Bank B: one stored set per factor.
    let t = Instant::now();
    let mut bank_per = TemplateBank::new();
    let mut figs_per = Vec::new();
    for &f in &FACTORS {
        let side = BASE_DIM * f as f64;
        let big_base =
            vectorial_hash_templates::polygon::create_square(0.0, 0.0, side, side);
        let fig = FigureKey::new(f, &[side]);
        bank_per.generate_size(&fig, &big_base, &[ANGLE], 1, 1);
        let (cw, ch) = cells_at_factor(f);
        bank_per.generate_size(&fig, &big_base, &[ANGLE], cw, ch);
        figs_per.push(fig);
    }
    let per_gen = t.elapsed();
    let per_mem = bank_per.memory_usage();

    println!(
        "bank A (one canonical set + placed_for_scaled): {} combos, {} unique, {:.2} MB, gen {:.0} us",
        bank_one.entry_count(),
        bank_one.unique_count(),
        one_mem.total() as f64 / 1e6,
        one_gen.as_micros(),
    );
    println!(
        "bank B (one set per scale):                     {} combos, {} unique, {:.2} MB, gen {:.0} us\n",
        bank_per.entry_count(),
        bank_per.unique_count(),
        per_mem.total() as f64 / 1e6,
        per_gen.as_micros(),
    );

    let mut results: Vec<(String, Duration)> = Vec::new();
    for (idx, &factor) in FACTORS.iter().enumerate() {
        let side = BASE_DIM * factor as f64;
        let mut poly =
            vectorial_hash_templates::polygon::create_square(0.0, 0.0, side, side);
        // Place the box somewhere away from the origin, aligned to its cell.
        let (cw, _) = cells_at_factor(factor);
        let origin = (
            (cw as i64) * 100i64,
            (cw as i64) * 60i64,
        );
        poly.move_by(origin.0 as f64, origin.1 as f64);
        let bbox = Rect::new(
            poly.x_min, poly.y_min,
            poly.x_max - poly.x_min, poly.y_max - poly.y_min,
        );

        let raster_one = bank_one.placed_for_scaled(
            &canonical_fig, factor as f64, 1, 1, ANGLE, origin,
        );
        let shape_one = ScaledShape {
            bank: &bank_one,
            base_figure: canonical_fig.clone(),
            scale_factor: factor as f64,
            angle: ANGLE,
            origin,
            poly: poly.clone(),
            bbox,
            raster: raster_one,
        };
        let raster_per = bank_per.placed_raster(&figs_per[idx], ANGLE, origin);
        let shape_per = BankShape {
            bank: &bank_per,
            figure: figs_per[idx].clone(),
            angle: ANGLE,
            origin,
            poly: poly.clone(),
            bbox,
            raster: raster_per,
        };

        let baseline = PlainShape { poly: poly.clone(), bbox };
        let expected = tree.cull(&baseline).len();
        let got_one = tree.cull(&shape_one).len();
        let got_per = tree.cull(&shape_per).len();
        assert_eq!(got_one, expected, "factor {factor}: scaled={got_one} != {expected}");
        assert_eq!(got_per, expected, "factor {factor}: per-scale={got_per} != {expected}");

        let t = Instant::now();
        for _ in 0..culls { black_box(tree.cull(&shape_one).len()); }
        results.push((format!("factor {factor} | scaled lookup"), t.elapsed()));

        let t = Instant::now();
        for _ in 0..culls { black_box(tree.cull(&shape_per).len()); }
        results.push((format!("factor {factor} | per-scale set"), t.elapsed()));
    }

    println!("{:<40} {:>12} {:>14}", "config", "total (ms)", "avg/cull (ms)");
    for (name, d) in &results {
        let ms = d.as_secs_f64() * 1000.0;
        println!(
            "{:<40} {:>12.2} {:>14.3}",
            name, ms, ms / culls as f64,
        );
    }
    println!(
        "\nmemory ratio (per-scale / one): {:.2}x; generation ratio: {:.2}x",
        per_mem.total() as f64 / one_mem.total() as f64,
        per_gen.as_secs_f64() / one_gen.as_secs_f64().max(1e-9),
    );
}

struct ScaledShape<'a> {
    bank: &'a TemplateBank,
    base_figure: FigureKey,
    scale_factor: f64,
    angle: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: Rect,
    raster: Option<PlacedTemplate>,
}

impl Shape for ScaledShape<'_> {
    fn bounding_box(&self) -> Rect { self.bbox }
    fn contains_point(&self, p: Point) -> bool { self.poly.is_inside(p.x, p.y) }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 { return None; }
        self.bank.placed_for_scaled(
            &self.base_figure,
            self.scale_factor,
            cell_w as u32, cell_h as u32,
            self.angle,
            self.origin,
        )
    }
    fn point_template(&self) -> Option<&PlacedTemplate> { self.raster.as_ref() }
}

/// Same as `BankShape` but goes through the aggregating fallback.
struct FallbackShape<'a> {
    bank: &'a TemplateBank,
    figure: FigureKey,
    angle: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: Rect,
    raster: Option<PlacedTemplate>,
}

impl Shape for FallbackShape<'_> {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None;
        }
        self.bank.placed_for_or_aggregated(
            &self.figure,
            cell_w as u32,
            cell_h as u32,
            self.angle,
            self.origin,
        )
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        self.raster.as_ref()
    }
}

/// Shape backed by the hierarchical `TemplateBank` (the paper's per-cell-size
/// selection): the figure sits at its real integer origin and each tree-cell
/// size resolves the template matching that origin's offset in the global
/// virtual grid of that size.
struct BankShape<'a> {
    bank: &'a TemplateBank,
    figure: FigureKey,
    angle: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: Rect,
    raster: Option<PlacedTemplate>,
}

impl Shape for BankShape<'_> {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None;
        }
        self.bank
            .placed_for(&self.figure, cell_w as u32, cell_h as u32, self.angle, self.origin)
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        self.raster.as_ref()
    }
}

/// `vh bench-sizes`: where do per-cell-size templates start paying off?
///
/// Compares, over the same tree and the same drop-shaped query applied at a
/// real integer origin:
///
/// 1. no templates (bbox + per-point geometry)
/// 2. one fixed fine grid + `classify_region` (≈ the old snap-to-offset
///    method's runtime cost)
/// 3. per-size template bank capped at ≤16, then ≤32, then ≤64 px cells
///    (cumulative), leaf items answered by the 1×1 raster
/// 4. the ≤64 bank again with the raster disabled (isolates its effect)
pub fn run_sizes(points: usize, culls: usize, item_limit: usize, seed: u64) {
    const SCALE: f64 = 350.0;
    const ANGLE: f64 = 30.0;
    let origin = (2048i64, 1024i64);

    println!("=== Cull benchmark: per-cell-size template selection ===");
    println!(
        "world {w}x{w} | {points} points | item_limit {item_limit} | {culls} culls/config | seed {seed}",
        w = WORLD as i64,
    );
    println!(
        "query: drop scale {SCALE} @{ANGLE}deg, origin {origin:?} (figure never moved)\n",
    );

    let base = scaled_copy(&create_drop(0.2, 0.8), SCALE, SCALE);
    let figure = FigureKey::new(0, &[0.2 * SCALE, 0.8 * SCALE]);
    let mut poly = rotated_copy(&base, angle_to_radians(ANGLE));
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = Rect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );

    // Point cloud + tree (binary-split only; quadtree comparison lives in `bench`).
    let mut rng = Rng(seed.max(1));
    let pts: Vec<Pt> = (0..points)
        .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
        .collect();
    let world = Rect::new(0.0, 0.0, WORLD, WORLD);
    let mut tree: Tree<Pt> = Tree::new(world, item_limit);
    for p in &pts {
        tree.insert(*p);
    }

    // Old-method stand-in: one fine fixed grid classified per node region.
    let single_cell: i64 = 16;
    let gxr = [
        (poly.x_min / single_cell as f64).floor() as i64,
        (poly.x_max / single_cell as f64).ceil() as i64,
    ];
    let gyr = [
        (poly.y_min / single_cell as f64).floor() as i64,
        (poly.y_max / single_cell as f64).ceil() as i64,
    ];
    let m = get_template_grid_fast(gxr[0], gyr[0], gxr[1], gyr[1], single_cell, single_cell, &poly);
    let single_grid = matrix_to_template_grid(
        &m,
        Point::new(
            gxr[0] as f64 * single_cell as f64,
            gyr[0] as f64 * single_cell as f64,
        ),
        single_cell as f64,
        single_cell as f64,
    );
    let plain = PlainShape { poly: poly.clone(), bbox };
    let single = TplShape { poly: poly.clone(), bbox, grid: single_grid };

    // Bank: 1x1 raster + cumulative size families.
    let mut bank = TemplateBank::new();
    let t = Instant::now();
    bank.generate_size(&figure, &base, &[ANGLE], 1, 1);
    let raster_gen = t.elapsed();
    let raster = bank.placed_raster(&figure, ANGLE, origin);
    println!(
        "generated 1x1 raster in {:.2}s ({} unique grids)",
        raster_gen.as_secs_f64(),
        bank.unique_count(),
    );

    let families: [(&str, &[(u32, u32)]); 3] = [
        ("<=16", &[(8, 8), (8, 16), (16, 8), (16, 16)]),
        ("<=32", &[(16, 32), (32, 16), (32, 32)]),
        ("<=64", &[(32, 64), (64, 32), (64, 64)]),
    ];

    // Correctness reference.
    let expected = tree.cull(&plain).len();

    let mut results: Vec<(String, Duration)> = Vec::new();
    let mut time_config = |label: String, shape: &dyn ErasedShape, tree: &Tree<Pt>| {
        let got = tree.cull_dyn(shape).len();
        assert_eq!(got, expected, "{label}: {got} hits != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(tree.cull_dyn(shape).len());
        }
        results.push((label, t.elapsed()));
    };

    time_config("no templates".into(), &plain, &tree);
    time_config(
        format!("single {single_cell}px grid (old snap method)"),
        &single,
        &tree,
    );

    for (label, sizes) in families {
        let t = Instant::now();
        for &(w, h) in sizes {
            bank.generate_size(&figure, &base, &[ANGLE], w, h);
        }
        let gen_time = t.elapsed();
        println!(
            "generated {label} family in {:.2}s (bank: {} combos, {} unique)",
            gen_time.as_secs_f64(),
            bank.entry_count(),
            bank.unique_count(),
        );
        let shape = BankShape {
            bank: &bank,
            figure: figure.clone(),
            angle: ANGLE,
            origin,
            poly: poly.clone(),
            bbox,
            raster: raster.clone(),
        };
        time_config(format!("bank {label} + raster"), &shape, &tree);
    }

    let no_raster = BankShape {
        bank: &bank,
        figure: figure.clone(),
        angle: ANGLE,
        origin,
        poly: poly.clone(),
        bbox,
        raster: None,
    };
    time_config("bank <=64, no raster".into(), &no_raster, &tree);

    // --- quadtree: same shapes, 4-way splits (square cells only) ---
    let mut qtree: QuadTree<Pt> = QuadTree::new(world, item_limit);
    for p in &pts {
        qtree.insert(*p);
    }
    let bank_shape = BankShape {
        bank: &bank,
        figure: figure.clone(),
        angle: ANGLE,
        origin,
        poly: poly.clone(),
        bbox,
        raster: raster.clone(),
    };
    {
        let got = qtree.cull(&plain).len();
        assert_eq!(got, expected, "quadtree plain: {got} != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(qtree.cull(&plain).len());
        }
        results.push(("quadtree, no templates".into(), t.elapsed()));

        let got = qtree.cull(&bank_shape).len();
        assert_eq!(got, expected, "quadtree bank: {got} != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(qtree.cull(&bank_shape).len());
        }
        results.push(("quadtree, bank <=64 + raster".into(), t.elapsed()));
    }

    // --- uniform grid (industry-standard broadphase baseline) ---
    let grid_cell = 32.0;
    let ugrid = UniformGrid::new(WORLD, grid_cell, &pts);
    {
        let got = ugrid.query(&bbox, None, &poly).len();
        assert_eq!(got, expected, "uniform grid: {got} != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(ugrid.query(&bbox, None, &poly).len());
        }
        results.push((format!("uniform grid {grid_cell}px (industry)"), t.elapsed()));

        let got = ugrid.query(&bbox, raster.as_ref(), &poly).len();
        assert_eq!(got, expected, "uniform grid + raster: {got} != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(ugrid.query(&bbox, raster.as_ref(), &poly).len());
        }
        results.push((format!("uniform grid {grid_cell}px + raster"), t.elapsed()));
    }

    println!("\n{:<38} {:>12} {:>14}", "config", "total (ms)", "avg/cull (ms)");
    let base_ms = results[0].1.as_secs_f64() * 1000.0;
    for (name, d) in &results {
        let ms = d.as_secs_f64() * 1000.0;
        println!(
            "{:<38} {:>12.2} {:>14.3}   ({:.2}x vs none)",
            name,
            ms,
            ms / culls as f64,
            base_ms / ms,
        );
    }
}

/// `vh bench-walk`: tree descent vs flood-fill traversal over leaf
/// neighbours, with every neighbour source (Samet ascent, locate probing,
/// and — with the `neighbors` feature — stored ropes). All configs use the
/// best per-size template setup (bank ≤64 + 1×1 raster) so the comparison
/// isolates the traversal strategy.
pub fn run_walk(points: usize, culls: usize, item_limit: usize, seed: u64, scale: f64) {
    use vectorial_hash::WalkNeighbors;

    const ANGLE: f64 = 30.0;
    let origin = (2048i64, 1024i64);

    println!("=== Cull benchmark: descent vs neighbour-walk (flood fill) ===");
    println!(
        "world {w}x{w} | {points} points | item_limit {item_limit} | {culls} culls/config | seed {seed}",
        w = WORLD as i64,
    );
    println!("query: drop scale {scale} @{ANGLE}deg, origin {origin:?}\n");

    let base = scaled_copy(&create_drop(0.2, 0.8), scale, scale);
    let figure = FigureKey::new(0, &[0.2 * scale, 0.8 * scale]);
    let mut poly = rotated_copy(&base, angle_to_radians(ANGLE));
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = Rect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );
    // Walk seed: the polygon's vertex centroid (inside — the drop is convex).
    let (mut cx, mut cy) = (0.0, 0.0);
    for v in &poly.vertices {
        cx += v.x;
        cy += v.y;
    }
    let n = poly.vertices.len() as f64;
    let seed_point = Point::new(cx / n, cy / n);
    assert!(poly.is_inside(seed_point.x, seed_point.y), "walk seed must be inside the figure");

    let mut bank = TemplateBank::new();
    let t = Instant::now();
    bank.generate_size(&figure, &base, &[ANGLE], 1, 1);
    for &(w, h) in &[
        (8u32, 8u32), (8, 16), (16, 8), (16, 16), (16, 32), (32, 16), (32, 32),
        (32, 64), (64, 32), (64, 64),
    ] {
        bank.generate_size(&figure, &base, &[ANGLE], w, h);
    }
    println!(
        "bank ready in {:.2}s ({} combos, {} unique)\n",
        t.elapsed().as_secs_f64(),
        bank.entry_count(),
        bank.unique_count(),
    );
    let raster = bank.placed_raster(&figure, ANGLE, origin);

    let mut rng = Rng(seed.max(1));
    let pts: Vec<Pt> = (0..points)
        .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
        .collect();
    let world = Rect::new(0.0, 0.0, WORLD, WORLD);
    let t = Instant::now();
    let mut tree: Tree<Pt> = Tree::new(world, item_limit);
    for p in &pts {
        tree.insert(*p);
    }
    println!(
        "tree built in {:.1} ms ({} nodes){}",
        t.elapsed().as_secs_f64() * 1000.0,
        tree.node_count(),
        if cfg!(feature = "neighbors") {
            "  [ropes maintained]"
        } else {
            "  [no rope bookkeeping compiled in]"
        },
    );

    let shape = BankShape {
        bank: &bank,
        figure: figure.clone(),
        angle: ANGLE,
        origin,
        poly: poly.clone(),
        bbox,
        raster,
    };

    let expected = tree.cull(&shape).len();
    println!("correctness reference: {expected} hits\n");

    let mut results: Vec<(String, Duration)> = Vec::new();

    let t = Instant::now();
    for _ in 0..culls {
        black_box(tree.cull(&shape).len());
    }
    results.push(("descent (Tree::cull)".into(), t.elapsed()));

    #[allow(unused_mut)] // mutated only with the `neighbors` feature
    let mut walk_configs: Vec<(&str, WalkNeighbors)> = vec![
        ("walk + Samet ascent", WalkNeighbors::Samet),
        ("walk + locate probe", WalkNeighbors::Probe),
    ];
    #[cfg(feature = "neighbors")]
    walk_configs.push(("walk + ropes (stored)", WalkNeighbors::Ropes));

    for (label, strategy) in walk_configs {
        let got = tree.cull_walk(&shape, seed_point, strategy).len();
        assert_eq!(got, expected, "{label}: {got} != {expected}");
        let t = Instant::now();
        for _ in 0..culls {
            black_box(tree.cull_walk(&shape, seed_point, strategy).len());
        }
        results.push((label.to_string(), t.elapsed()));
    }

    println!("{:<26} {:>12} {:>14}", "config", "total (ms)", "avg/cull (ms)");
    let base_ms = results[0].1.as_secs_f64() * 1000.0;
    for (name, d) in &results {
        let ms = d.as_secs_f64() * 1000.0;
        println!(
            "{:<26} {:>12.2} {:>14.3}   ({:.2}x vs descent)",
            name,
            ms,
            ms / culls as f64,
            base_ms / ms,
        );
    }
    #[cfg(not(feature = "neighbors"))]
    println!("\n(ropes config omitted: rebuild with --features neighbors)");
}

/// Flat uniform grid of buckets — the classic games-industry broadphase
/// (spatial hashing over a fixed cell size; see Ericson, *Real-Time
/// Collision Detection*, ch. 7). Query: iterate the buckets overlapped by
/// the query bbox and test each point.
struct UniformGrid {
    cell: f64,
    cols: usize,
    buckets: Vec<Vec<Point>>,
}

impl UniformGrid {
    fn new(world: f64, cell: f64, pts: &[Pt]) -> Self {
        let cols = (world / cell).ceil() as usize;
        let mut buckets = vec![Vec::new(); cols * cols];
        for p in pts {
            let cx = ((p.0.x / cell) as usize).min(cols - 1);
            let cy = ((p.0.y / cell) as usize).min(cols - 1);
            buckets[cy * cols + cx].push(p.0);
        }
        Self { cell, cols, buckets }
    }

    /// Collect references to points inside the polygon (same contract as the
    /// tree culls, for a fair comparison). `raster` optionally replaces the
    /// exact per-point test (boundary pixels still use geometry).
    fn query<'a>(
        &'a self,
        bbox: &Rect,
        raster: Option<&PlacedTemplate>,
        poly: &Polygon,
    ) -> Vec<&'a Point> {
        let c0 = ((bbox.x / self.cell).floor().max(0.0)) as usize;
        let r0 = ((bbox.y / self.cell).floor().max(0.0)) as usize;
        let c1 = (((bbox.x_max() / self.cell).ceil()) as usize).min(self.cols);
        let r1 = (((bbox.y_max() / self.cell).ceil()) as usize).min(self.cols);
        let mut out = Vec::new();
        for row in r0..r1 {
            for col in c0..c1 {
                for p in &self.buckets[row * self.cols + col] {
                    if !bbox.contains_closed(*p) {
                        continue;
                    }
                    match raster.map(|r| r.cell_at_world(*p)) {
                        Some(vectorial_hash::CellState::In) => out.push(p),
                        Some(vectorial_hash::CellState::Out) => {}
                        _ => {
                            if poly.is_inside(p.x, p.y) {
                                out.push(p);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// Object-safe shim so heterogeneous shapes can share one timing closure.
trait ErasedShape {
    fn bounding_box(&self) -> Rect;
    fn contains_point(&self, p: Point) -> bool;
    fn template_grid(&self) -> Option<&TemplateGrid>;
    fn template_for_cell(&self, w: f64, h: f64) -> Option<PlacedTemplate>;
    fn point_template(&self) -> Option<&PlacedTemplate>;
}

impl<S: Shape> ErasedShape for S {
    fn bounding_box(&self) -> Rect {
        Shape::bounding_box(self)
    }
    fn contains_point(&self, p: Point) -> bool {
        Shape::contains_point(self, p)
    }
    fn template_grid(&self) -> Option<&TemplateGrid> {
        Shape::template_grid(self)
    }
    fn template_for_cell(&self, w: f64, h: f64) -> Option<PlacedTemplate> {
        Shape::template_for_cell(self, w, h)
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        Shape::point_template(self)
    }
}

impl Shape for &dyn ErasedShape {
    fn bounding_box(&self) -> Rect {
        (**self).bounding_box()
    }
    fn contains_point(&self, p: Point) -> bool {
        (**self).contains_point(p)
    }
    fn template_grid(&self) -> Option<&TemplateGrid> {
        (**self).template_grid()
    }
    fn template_for_cell(&self, w: f64, h: f64) -> Option<PlacedTemplate> {
        (**self).template_for_cell(w, h)
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        (**self).point_template()
    }
}

trait CullDyn<T: Positioned> {
    fn cull_dyn<'a>(&'a self, shape: &dyn ErasedShape) -> Vec<&'a T>;
}

impl<T: Positioned> CullDyn<T> for Tree<T> {
    fn cull_dyn<'a>(&'a self, shape: &dyn ErasedShape) -> Vec<&'a T> {
        self.cull(&shape)
    }
}
