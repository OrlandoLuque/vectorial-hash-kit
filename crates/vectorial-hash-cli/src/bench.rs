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

use vectorial_hash::{Point, Positioned, Rect, Shape, TemplateGrid, Tree};
use vectorial_hash_templates::adapter::matrix_to_template_grid;
use vectorial_hash_templates::polygon::{create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::{angle_to_radians, get_template_grid_fast};

use crate::quadtree::QuadTree;

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

pub fn run(points: usize, culls: usize, item_limit: usize, seed: u64) {
    println!("=== Cull benchmark: binary-split tree vs quadtree, templates on/off ===");
    println!(
        "world {w}x{w} | {points} points | item_limit {item_limit} | {culls} culls/config | seed {seed}\n",
        w = WORLD as i64,
    );

    // Query polygon: a big rotated drop near the middle of the world.
    let mut poly = rotated_copy(
        &scaled_copy(&create_drop(0.2, 0.8), 1400.0, 1400.0),
        angle_to_radians(30.0),
    );
    poly.move_by(WORLD / 2.0, WORLD / 4.0);
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

    // Same points for both trees.
    let mut rng = Rng(seed.max(1));
    let pts: Vec<Pt> = (0..points)
        .map(|_| Pt(Point::new(rng.unit_f64() * WORLD, rng.unit_f64() * WORLD)))
        .collect();

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
