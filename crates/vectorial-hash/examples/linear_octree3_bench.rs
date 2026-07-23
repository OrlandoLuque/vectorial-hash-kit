//! `LinearOctree3` vs `Octree3` (pointer octree) vs `MortonGrid3` (uniform grid):
//! build, sphere-cull and k-NN over the same clustered 3D point set. The linear
//! octree's niche is *adaptive depth without pointers* — this measures whether that
//! actually pays against the two structures it sits between.
//!
//! ```bash
//! cargo run -p vectorial-hash --example linear_octree3_bench --release
//! ```
//! Env: `LO_N` (points), `LO_Q` (queries), `LO_R` (cull radius).

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

    // ---- cull ----
    let mut sink = 0usize;
    let t_cull_lin = best(6, || { for q in &queries { sink += lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_cull_oct = best(6, || { for q in &queries { sink += oct.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_cull_mor = best(6, || { for q in &queries { sink += mor.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });

    // ---- knn (k=8) ----
    let t_knn_lin = best(6, || { for q in &queries { sink += lin.knn(*q, 8).len(); } });
    let t_knn_oct = best(6, || { for q in &queries { sink += oct.knn(*q, 8).len(); } });
    let t_knn_mor = best(6, || { for q in &queries { sink += mor.knn(*q, 8).len(); } });

    println!("structure       build(ms)   cull {nq}q(ms)   knn {nq}q(ms)   leaves/cells   depth");
    println!("LinearOctree3   {t_build_lin:8.2}   {t_cull_lin:11.2}   {t_knn_lin:10.2}   {:>12}   {}", lin.leaf_count(), lin.depth());
    println!("Octree3         {t_build_oct:8.2}   {t_cull_oct:11.2}   {t_knn_oct:10.2}   {:>12}   —", "—");
    println!("MortonGrid3     {t_build_mor:8.2}   {t_cull_mor:11.2}   {t_knn_mor:10.2}   {:>12}   1 (flat)", mor.cell_count());
    println!("\ncull speed   vs Octree3 {:.2}x   vs Morton {:.2}x", t_cull_oct / t_cull_lin, t_cull_mor / t_cull_lin);
    println!("knn  speed   vs Octree3 {:.2}x   vs Morton {:.2}x", t_knn_oct / t_knn_lin, t_knn_mor / t_knn_lin);
    if sink == usize::MAX { println!("{sink}"); } // keep the queries live
}
