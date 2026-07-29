//! `morton_knn_axis_bench` — what a non-cubic world costs `MortonGrid3::knn`.
//!
//! `levels` is a single number for all three axes, so on a world that is not a cube the
//! cells are not cubes: a 1000x300x1000 world at levels=5 has cells 31.25 x 9.375 x 31.25.
//! Any expansion that grows one radius for all three axes is then isotropic in *cell* space
//! and wildly anisotropic in *world* space — reaching 30 units along the short axis drags the
//! wide axes out to +-125 and scans everything in between. `knn` grows each axis on its own
//! for exactly that reason; this bench is how that was measured and how it stays honest.
//!
//! It sweeps the aspect ratio and reports, per aspect, **points tested** (exact, no clock —
//! counted by wrapping the item's `position()`) and **time**. Both, because they answer
//! different questions and this repo has a case on record where less arithmetic ran slower
//! (the boids separation table, `docs/PERF_NOTES.md`): fewer point tests only pay if the
//! ones you skipped were not free.
//!
//! To compare against an older expansion, run it on both commits — the effect being measured
//! is several times larger than the ~20% between-process variation in `docs/MEASURING.md`.
//! There is deliberately no reimplementation of the old algorithm here to time against: the
//! obvious stand-in (re-cull the whole box each step) is quadratic where the real one visited
//! only the new shell, so it would have flattered the change by measuring a strawman.
//!
//! ```bash
//! cargo run -p vectorial-hash --example morton_knn_axis_bench --release
//! ```
//! Env: `MK_N`, `MK_Q`, `MK_K`, `MK_LEVELS`.

#[path = "common/mod.rs"]
mod common;

use std::cell::Cell;
use vectorial_hash::{Aabb, MortonGrid, MortonGrid3, Point, Point3, Positioned, Positioned3, Rect};

thread_local! { static TESTED: Cell<u64> = const { Cell::new(0) }; }

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P {
    fn position(&self) -> Point3 { TESTED.with(|c| c.set(c.get() + 1)); self.p }
}

#[derive(Clone, Copy)]
struct P2 { p: Point }
impl Positioned for P2 {
    fn position(&self) -> Point { TESTED.with(|c| c.set(c.get() + 1)); self.p }
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

fn main() {
    let n: usize = std::env::var("MK_N").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let nq: usize = std::env::var("MK_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let k: usize = std::env::var("MK_K").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let levels: u32 = std::env::var("MK_LEVELS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("morton3 k-NN vs world aspect | {n} points (clustered) | {nq} queries | k={k} | levels={levels}");
    println!("aspect = world height / width. 1.00 means cubic cells, where axis order cannot matter.\n");
    println!("{:<8} {:>10} {:>14} {:>12} {:>12}", "aspect", "cell y/x", "tested/query", "ms/query", "vs cubic");

    let mut base: Option<f64> = None;
    for aspect in [1.0f64, 0.5, 0.3, 0.15, 0.05] {
        let (w, h) = (1000.0, 1000.0 * aspect);
        let world = Aabb::new(0.0, 0.0, 0.0, w, h, w);
        let mut r = Lcg(0x5EED_1234);
        // Clustered: on a uniform cloud every structure looks the same and the effect hides.
        let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(0.1 * w, 0.9 * w), r.r(0.2 * h, 0.8 * h), r.r(0.1 * w, 0.9 * w))).collect();
        let items: Vec<P> = (0..n).map(|_| {
            let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
            P { p: Point3::new((b.0 + r.r(-0.014 * w, 0.014 * w)).clamp(0.0, w),
                (b.1 + r.r(-0.014 * h, 0.014 * h)).clamp(0.0, h),
                (b.2 + r.r(-0.014 * w, 0.014 * w)).clamp(0.0, w)) }
        }).collect();
        let queries: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, w), r.r(0.0, h), r.r(0.0, w))).collect();

        let mut g = MortonGrid3::new(world, levels);
        for it in &items { g.insert(*it); }

        // Counted first, and on its own: the count is exact and must not be inflated by the
        // repetitions the timing harness adds.
        TESTED.with(|c| c.set(0));
        let mut found = 0usize;
        for q in &queries { found += g.knn(*q, k).len(); }
        let tested = TESTED.with(|c| c.get()) as f64 / nq as f64;
        assert_eq!(found, nq * k.min(n), "k-NN did not return k neighbours — the bench would be measuring a failure");

        let s = common::measure(5, || { for q in &queries { std::hint::black_box(g.knn(*q, k).len()); } });
        let per = s.ms / nq as f64;
        let cells = (1u32 << levels) as f64;
        let rel = match base { None => { base = Some(per); 1.0 } Some(b) => per / b };
        println!("{aspect:<8.2} {:>10.3} {:>14.1} {:>12.4} {:>11.2}x", (h / cells) / (w / cells), tested, per, rel);
        println!("#M aspect{}.knn_tested_per_query {:.1} n", (aspect * 100.0) as u32, tested);
        println!("#M aspect{}.knn_ms_per_query {:.5} ms", (aspect * 100.0) as u32, per);
    }

    // The 2D grid has exactly the same shape of problem on a non-square Rect, and the same
    // fix. Swept here so the claim is not made for one dimension and assumed for the other.
    println!("\n2D - same sweep, MortonGrid over a Rect");
    println!("{:<8} {:>10} {:>14} {:>12} {:>12}", "aspect", "cell y/x", "tested/query", "ms/query", "vs square");
    let mut base2: Option<f64> = None;
    for aspect in [1.0f64, 0.5, 0.3, 0.15, 0.05] {
        let (w, h) = (1000.0, 1000.0 * aspect);
        let world = Rect::new(0.0, 0.0, w, h);
        let mut r = Lcg(0x5EED_1234);
        let blobs: Vec<(f64, f64)> = (0..6).map(|_| (r.r(0.1 * w, 0.9 * w), r.r(0.2 * h, 0.8 * h))).collect();
        let items: Vec<P2> = (0..n).map(|_| {
            let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
            P2 { p: Point::new((b.0 + r.r(-0.014 * w, 0.014 * w)).clamp(0.0, w), (b.1 + r.r(-0.014 * h, 0.014 * h)).clamp(0.0, h)) }
        }).collect();
        let queries: Vec<Point> = (0..nq).map(|_| Point::new(r.r(0.0, w), r.r(0.0, h))).collect();
        let mut g = MortonGrid::new(world, levels);
        for it in &items { g.insert(*it); }

        TESTED.with(|c| c.set(0));
        let mut found = 0usize;
        for q in &queries { found += g.knn(*q, k).len(); }
        let tested = TESTED.with(|c| c.get()) as f64 / nq as f64;
        assert_eq!(found, nq * k.min(n), "2D k-NN did not return k neighbours");

        let s = common::measure(5, || { for q in &queries { std::hint::black_box(g.knn(*q, k).len()); } });
        let per = s.ms / nq as f64;
        let cells = (1u32 << levels) as f64;
        let rel = match base2 { None => { base2 = Some(per); 1.0 } Some(b) => per / b };
        println!("{aspect:<8.2} {:>10.3} {:>14.1} {:>12.4} {:>11.2}x", (h / cells) / (w / cells), tested, per, rel);
        println!("#M d2aspect{}.knn_tested_per_query {:.1} n", (aspect * 100.0) as u32, tested);
        println!("#M d2aspect{}.knn_ms_per_query {:.5} ms", (aspect * 100.0) as u32, per);
    }

    // ---- cell SHAPE, at one fixed world shape ------------------------------------------
    // The expansion fix makes a non-cubic grid survivable. The other move is to not build one:
    // `cells_per_axis` is a single number, so the only way to get cubic CELLS out of a
    // non-cubic WORLD is to declare the world as a cube. The cell store is a sparse hash, so
    // the empty layers above the data cost nothing to hold - the question is whether the
    // resulting geometry is actually better, and at which cell size.
    println!("
cell shape - same 1000x300x1000 data, four ways to cut it up");
    println!("{:<28} {:>12} {:>11} {:>14} {:>12}", "grid", "cell (x,y,z)", "used cells", "tested/query", "ms/query");
    {
        let (w, h) = (1000.0f64, 300.0f64);
        let mut r = Lcg(0x5EED_1234);
        let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(0.1 * w, 0.9 * w), r.r(0.2 * h, 0.8 * h), r.r(0.1 * w, 0.9 * w))).collect();
        let items: Vec<P> = (0..n).map(|_| {
            let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
            P { p: Point3::new((b.0 + r.r(-0.014 * w, 0.014 * w)).clamp(0.0, w),
                (b.1 + r.r(-0.014 * h, 0.014 * h)).clamp(0.0, h),
                (b.2 + r.r(-0.014 * w, 0.014 * w)).clamp(0.0, w)) }
        }).collect();
        let queries: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, w), r.r(0.0, h), r.r(0.0, w))).collect();

      for clustered in [true, false] {
        let items: Vec<P> = if clustered { items.clone() } else {
            let mut u = Lcg(0xC0FFEE);
            (0..n).map(|_| P { p: Point3::new(u.r(0.0, w), u.r(0.0, h), u.r(0.0, w)) }).collect()
        };
        println!("  -- {} data", if clustered { "clustered (6 tight blobs)" } else { "uniform" });
        let mut reference: Option<Vec<f64>> = None;
        for (label, world, lv) in [
            ("anisotropic cells, L5", Aabb::new(0.0, 0.0, 0.0, w, h, w), 5u32),
            ("cubic (world padded), L5", Aabb::new(0.0, 0.0, 0.0, w, w, w), 5),
            ("cubic (world padded), L6", Aabb::new(0.0, 0.0, 0.0, w, w, w), 6),
            ("cubic (world padded), L7", Aabb::new(0.0, 0.0, 0.0, w, w, w), 7),
        ] {
            let mut g = MortonGrid3::new(world, lv);
            for it in &items { g.insert(*it); }
            let cells = (1u32 << lv) as f64;
            let (cx, cy, cz) = (world.w / cells, world.h / cells, world.d / cells);

            TESTED.with(|c| c.set(0));
            let mut all = Vec::with_capacity(nq * k);
            for q in &queries { all.extend(g.knn(*q, k).iter().map(|(d, _)| *d)); }
            let tested = TESTED.with(|c| c.get()) as f64 / nq as f64;
            // Different granularity must not mean different answers.
            match &reference {
                None => reference = Some(all),
                Some(rf) => assert!(rf.len() == all.len() && rf.iter().zip(&all).all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b.abs())),
                    "{label} returned different neighbours"),
            }
            let s = common::measure(5, || { for q in &queries { std::hint::black_box(g.knn(*q, k).len()); } });
            println!("{label:<28} {:>12} {:>11} {:>14.1} {:>12.4}",
                format!("{cx:.1},{cy:.1},{cz:.1}"), g.cell_count(), tested, s.ms / nq as f64);
            println!("#M shape_{}.knn_tested_per_query {:.1} n", label.replace([' ', ',', '(', ')'], "_"), tested);
            let tag = if clustered { "cl" } else { "un" };
            println!("#M shape_{tag}_{}.knn_ms_per_query {:.5} ms", label.replace([' ', ',', '(', ')'], "_"), s.ms / nq as f64);
        }
      }
    }

    println!("\nreading: 'tested/query' is exact and machine-independent — the algorithmic number.");
    println!("'vs cubic' is each aspect's time against the cubic world's, which is the control:");
    println!("a flatter world holds the same points in the same clusters, so a big rise there is");
    println!("the expansion shape paying for the cell shape, not the data getting harder.");
}
