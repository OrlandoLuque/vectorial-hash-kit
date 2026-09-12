//! `LinearOctree3` vs `Octree3` (pointer octree) vs `MortonGrid3` (uniform grid):
//! build, sphere-cull and k-NN over the same clustered 3D point set. The linear
//! octree's niche is *adaptive depth without pointers* — this measures whether that
//! actually pays against the two structures it sits between.
//!
//! **★ The grid arm was handicapped until 2026-09-13, and the cull ratio this bench publishes did
//! not survive fixing it.** The world is 1000 x 300 x 1000 and `levels` is one number for all
//! three axes, so `levels 5` means cells of 31.25 x 9.375 x 31.25 — slabs — and a radius-40 query
//! spans ~121 cells instead of the ~45 cubic cells would span. Declaring the *index* world a cube
//! costs nothing (sparse hash: the layers above y = 300 are never stored or traversed) and the
//! grid gets **1.45-1.96x faster**, which takes `LinearOctree3`'s cull advantage from 1.4-1.9x to
//! **1.04-1.09x — a tie**. The k-NN row survives, because `MortonGrid3::knn` already expands
//! per-axis; `cull` never needed that fix and so never got one.
//!
//! All three grid configurations are reported rather than one being swapped in, because the slab
//! row is now the evidence for two lessons: a good `Occupancy::mean` (7.0, in band) is **not** a
//! fast grid, and finer is not automatically better either — cube `levels 6` has the best mean of
//! the three (4.4) and is the slowest, at ~229 cells per query.
//!
//! ```bash
//! cargo run -p vectorial-hash --example linear_octree3_bench --release
//! ```
//! Env: `LO_N` (points), `LO_Q` (queries), `LO_R` (cull radius).

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;
use vectorial_hash::linear_octree3::LinearOctree3;
use vectorial_hash::{Aabb, MortonGrid3, Octree3, Point3, Positioned3, Sphere3};

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

fn best<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let mut lo = f64::INFINITY;
    for _ in 0..runs { let t = Instant::now(); f(); lo = lo.min(t.elapsed().as_secs_f64()); }
    lo * 1e3 // ms
}

fn main() {
    let n: usize = std::env::var("LO_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = std::env::var("LO_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let radius: f64 = std::env::var("LO_R").ok().and_then(|s| s.parse().ok()).unwrap_or(40.0);
    let world = Aabb::new(0.0, 0.0, 0.0, 1000.0, 300.0, 1000.0);

    // A few dense blobs in a sparse field — the case single-cell grids handle worst.
    let mut r = Lcg(0x1234_5678);
    let blobs: Vec<(f64, f64, f64)> = (0..8).map(|_| (r.r(80.0, 920.0), r.r(40.0, 260.0), r.r(80.0, 920.0))).collect();
    let items: Vec<P> = (0..n).map(|i| {
        let p = if i % 3 == 0 {
            Point3::new(r.r(0.0, 1000.0), r.r(0.0, 300.0), r.r(0.0, 1000.0))
        } else {
            let (bx, by, bz) = blobs[i % blobs.len()];
            Point3::new((bx + r.r(-30.0, 30.0)).clamp(0.0, 1000.0), (by + r.r(-15.0, 15.0)).clamp(0.0, 300.0), (bz + r.r(-30.0, 30.0)).clamp(0.0, 1000.0))
        };
        P { p }
    }).collect();
    let queries: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, 1000.0), r.r(0.0, 300.0), r.r(0.0, 1000.0))).collect();

    println!("LinearOctree3 bench — {n} points ({:.0}% clustered), {nq} queries, cull r={radius}\n", 200.0 / 3.0);

    // ---- build ----
    let t_build_lin = best(5, || { let _ = LinearOctree3::from_items(world, 32, 14, items.clone()); });
    let t_build_oct = best(5, || { let _ = Octree3::bulk_load(world, 32, items.clone()); });
    let t_build_mor = best(5, || { let mut g = MortonGrid3::new(world, 5); for it in &items { g.insert(*it); } });

    let lin = LinearOctree3::from_items(world, 32, 14, items.clone());
    let oct = Octree3::bulk_load(world, 32, items.clone());
    let mut mor = MortonGrid3::new(world, 5);
    for it in &items { mor.insert(*it); }

    // ---- the grid arm, with its cells made CUBIC ---------------------------------------------
    //
    // The world here is 1000 x 300 x 1000, and `levels` is ONE number for all three axes, so
    // `levels 5` gives cells of 31.25 x 9.375 x 31.25 — slabs, against a query radius of 40. That
    // is the pathology #118 fixed in the horde and #116 fixed inside `MortonGrid3::knn`, and this
    // bench publishes the grid ratios quoted in THREE_D.md and the README. It was found by
    // auditing every 3D bench for the signature "a fixed quantity chosen next to a parameter of
    // one arm" (MEASURING § 8i) after the radix bench turned out to have it.
    //
    // Declaring the index world a CUBE costs nothing: the backing store is a sparse hash, so the
    // layers above y = 300 are never stored and never traversed. Both resolutions are reported
    // rather than one being swapped in, so the size of the handicap is visible.
    let side = world.w.max(world.h).max(world.d);
    let cube = Aabb::new(world.x, world.y, world.z, side, side, side);
    let mut mor_c5 = MortonGrid3::new(cube, 5);
    for it in &items { mor_c5.insert(*it); }
    let mut mor_c6 = MortonGrid3::new(cube, 6);
    for it in &items { mor_c6.insert(*it); }
    let t_build_c5 = best(5, || { let mut g = MortonGrid3::new(cube, 5); for it in &items { g.insert(*it); } });
    let t_build_c6 = best(5, || { let mut g = MortonGrid3::new(cube, 6); for it in &items { g.insert(*it); } });
    println!("grid cells: slab L5 {:.2} x {:.2} x {:.2}  |  cube L5 {:.2}^3  |  cube L6 {:.2}^3",
             world.w / 32.0, world.h / 32.0, world.d / 32.0, side / 32.0, side / 64.0);
    println!("  cells a r={radius} query spans: slab ~{:.0} | cube L5 ~{:.0} | cube L6 ~{:.0}  \
              (build ms: cube L5 {t_build_c5:.1}, cube L6 {t_build_c6:.1})",
             (2.0 * radius / (world.w / 32.0) + 1.0) * (2.0 * radius / (world.h / 32.0) + 1.0) * (2.0 * radius / (world.d / 32.0) + 1.0),
             (2.0 * radius / (side / 32.0) + 1.0).powi(3),
             (2.0 * radius / (side / 64.0) + 1.0).powi(3));
    println!("  occupancy slab L5 {:?}", mor.occupancy());
    println!("  occupancy cube L5 {:?}", mor_c5.occupancy());
    println!("  occupancy cube L6 {:?}", mor_c6.occupancy());
    assert_eq!(mor.occupancy().items, mor_c6.occupancy().items,
               "a cubic world must not drop items — every point is inside both boxes");

    // ---- cull ----
    let mut sink = 0usize;
    let t_cull_lin = best(6, || { for q in &queries { sink += lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_cull_oct = best(6, || { for q in &queries { sink += oct.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_cull_mor = best(6, || { for q in &queries { sink += mor.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });

    // ---- knn (k=8) ----
    let t_knn_lin = best(6, || { for q in &queries { sink += lin.knn(*q, 8).len(); } });
    let t_knn_oct = best(6, || { for q in &queries { sink += oct.knn(*q, 8).len(); } });
    let t_knn_mor = best(6, || { for q in &queries { sink += mor.knn(*q, 8).len(); } });

    // ---- the ratios, measured properly ----
    // The table above times each structure to completion and then divides, which is the
    // thing docs/MEASURING.md exists to warn about: whoever runs second inherits a machine
    // the first one warmed, and the same ratio has moved between 1.57 and 3.28 that way.
    // These four numbers are the ones quoted in the docs, so they are measured interleaved
    // (A B B A per round, median of the per-round ratios) and reported with their spread.
    let rate = common::rate();
    let pair = |label: &str, key: &str,
                mut a: &mut dyn FnMut(), mut b: &mut dyn FnMut()| {
        let (a_cy, b_cy, ratio, spread) = common::compare2(7, &mut a, &mut b);
        println!("  {label:<28} {:>9.3} {:>9.3} {:>8.2}x {:>8.1}%", a_cy / rate * 1e3, b_cy / rate * 1e3, ratio, spread);
        println!("#M {key} {ratio:.3} x");
        println!("#M {key}_spread {spread:.1} pct");
    };
    println!("\npaired ratios (interleaved, median of per-round ratios — the quotable ones)");
    println!("  {:<28} {:>9} {:>9} {:>9} {:>9}", "comparison", "lin ms", "rival ms", "speed-up", "spread");
    pair("cull vs Octree3", "cull_vs_octree3",
        &mut || { for q in &queries { std::hint::black_box(lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(oct.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("cull vs MortonGrid3", "cull_vs_morton3",
        &mut || { for q in &queries { std::hint::black_box(lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("knn vs Octree3", "knn_vs_octree3",
        &mut || { for q in &queries { std::hint::black_box(lin.knn(*q, 8).len()); } },
        &mut || { for q in &queries { std::hint::black_box(oct.knn(*q, 8).len()); } });
    pair("knn vs MortonGrid3", "knn_vs_morton3",
        &mut || { for q in &queries { std::hint::black_box(lin.knn(*q, 8).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor.knn(*q, 8).len()); } });

    // The same two rivals again, with the grid's cells cubic. If these differ from the rows above,
    // the published grid ratios were measured against a handicapped grid. **L5 is the row that
    // matters** — same number of levels as the published slab, only the declared world box cubic,
    // so nothing but the cell ASPECT changes. L6 is here to show that finer is not automatically
    // better: it quarters the occupancy and quintuples the cells a query has to look up.
    pair("cull vs Morton3 CUBE L5", "cull_vs_morton3_cube5",
        &mut || { for q in &queries { std::hint::black_box(lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c5.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("knn vs Morton3 CUBE L5", "knn_vs_morton3_cube5",
        &mut || { for q in &queries { std::hint::black_box(lin.knn(*q, 8).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c5.knn(*q, 8).len()); } });
    pair("cull vs Morton3 CUBE L6", "cull_vs_morton3_cube",
        &mut || { for q in &queries { std::hint::black_box(lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c6.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("knn vs Morton3 CUBE L6", "knn_vs_morton3_cube",
        &mut || { for q in &queries { std::hint::black_box(lin.knn(*q, 8).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c6.knn(*q, 8).len()); } });
    // And the grid against itself — the number that says how much the slab cost it, with nothing
    // else varying: same items, same queries, same code, only the declared world box.
    pair("Morton3 slab vs CUBE L5", "morton3_slab_vs_cube5",
        &mut || { for q in &queries { std::hint::black_box(mor.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c5.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("Morton3 slab vs CUBE L6", "morton3_slab_vs_cube6",
        &mut || { for q in &queries { std::hint::black_box(mor.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c6.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len()); } });
    pair("Morton3 knn slab vs CUBE L6", "morton3_knn_slab_vs_cube6",
        &mut || { for q in &queries { std::hint::black_box(mor.knn(*q, 8).len()); } },
        &mut || { for q in &queries { std::hint::black_box(mor_c6.knn(*q, 8).len()); } });

    println!("structure       build(ms)   cull {nq}q(ms)   knn {nq}q(ms)   leaves/cells   depth");
    println!("LinearOctree3   {t_build_lin:8.2}   {t_cull_lin:11.2}   {t_knn_lin:10.2}   {:>12}   {}", lin.leaf_count(), lin.depth());
    println!("Octree3         {t_build_oct:8.2}   {t_cull_oct:11.2}   {t_knn_oct:10.2}   {:>12}   —", "—");
    println!("MortonGrid3     {t_build_mor:8.2}   {t_cull_mor:11.2}   {t_knn_mor:10.2}   {:>12}   1 (flat)", mor.cell_count());
    println!("\nunpaired, from the table above (kept for comparison with the paired numbers,");
    println!("NOT for quoting): cull vs Octree3 {:.2}x  vs Morton {:.2}x | knn vs Octree3 {:.2}x  vs Morton {:.2}x",
        t_cull_oct / t_cull_lin, t_cull_mor / t_cull_lin, t_knn_oct / t_knn_lin, t_knn_mor / t_knn_lin);
    if sink == usize::MAX { println!("{sink}"); } // keep the queries live
}
