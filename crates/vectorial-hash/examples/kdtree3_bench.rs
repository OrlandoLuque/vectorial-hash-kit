//! `KdTree3` (median split) vs `Tree3` / `Octree3` (midpoint split) vs `LinearOctree3`
//! — the measured question the k-d tree exists to answer: does balancing by point
//! COUNT (median) beat splitting the empty middle (midpoint) when the data is
//! CLUSTERED? Run over a uniform set (where midpoint is fine) and a heavily clustered
//! one (where the midpoint trees go deep over empty space).
//!
//! ```bash
//! cargo run -p vectorial-hash --example kdtree3_bench --release
//! ```
//! Env: `KD_N` (points), `KD_Q` (queries), `KD_R` (cull radius).
use std::time::Instant;
use vectorial_hash::{Aabb, KdTree3, LinearOctree3, Octree3, Point3, Positioned3, Sphere3, Tree3};

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
    lo * 1e3
}

fn run(name: &str, items: &[P], queries: &[Point3], radius: f64) {
    let world = Aabb::new(0.0, 0.0, 0.0, 1000.0, 300.0, 1000.0);
    let v: Vec<P> = items.to_vec();
    let t_b_kd = best(5, || { let _ = KdTree3::from_items(16, v.clone()); });
    let t_b_bin = best(5, || { let _ = Tree3::bulk_load(world, 16, v.clone()); });
    let t_b_oct = best(5, || { let _ = Octree3::bulk_load(world, 16, v.clone()); });
    let t_b_lin = best(5, || { let _ = LinearOctree3::from_items(world, 16, 18, v.clone()); });

    let kd = KdTree3::from_items(16, v.clone());
    let bin = Tree3::bulk_load(world, 16, v.clone());
    let oct = Octree3::bulk_load(world, 16, v.clone());
    let lin = LinearOctree3::from_items(world, 16, 18, v.clone());

    let mut s = 0usize;
    let t_c_kd = best(6, || { for q in queries { s += kd.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_c_bin = best(6, || { for q in queries { s += bin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_c_oct = best(6, || { for q in queries { s += oct.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_c_lin = best(6, || { for q in queries { s += lin.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } });
    let t_k_kd = best(6, || { for q in queries { s += kd.knn(*q, 8).len(); } });
    let t_k_bin = best(6, || { for q in queries { s += bin.knn(*q, 8).len(); } });

    println!("── {name} ──   (KdTree3 depth {}, {} nodes | Tree3 {} nodes | Octree3 {} nodes)", kd.depth(), kd.node_count(), bin.node_count(), oct.node_count());
    println!("  build ms   KdTree3 {t_b_kd:6.2} | Tree3 {t_b_bin:6.2} | Octree3 {t_b_oct:6.2} | LinearOct {t_b_lin:6.2}");
    println!("  cull  ms   KdTree3 {t_c_kd:6.2} | Tree3 {t_c_bin:6.2} | Octree3 {t_c_oct:6.2} | LinearOct {t_c_lin:6.2}   (kd vs Tree3 {:.2}×)", t_c_bin / t_c_kd);
    println!("  knn   ms   KdTree3 {t_k_kd:6.2} | Tree3 {t_k_bin:6.2}                                    (kd vs Tree3 {:.2}×)", t_k_bin / t_k_kd);
    if s == usize::MAX { println!("{s}"); }
}

fn main() {
    let n: usize = std::env::var("KD_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = std::env::var("KD_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let radius: f64 = std::env::var("KD_R").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let mut r = Lcg(0x051A_51A5);
    println!("KdTree3 (median split) vs midpoint trees — {n} points, {nq} queries r={radius}\n");

    // uniform
    let uni: Vec<P> = (0..n).map(|_| P { p: Point3::new(r.r(0.0, 1000.0), r.r(0.0, 300.0), r.r(0.0, 1000.0)) }).collect();
    // heavily clustered: 6 tight blobs in a big empty world (the midpoint trees' worst case)
    let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(100.0, 900.0), r.r(50.0, 250.0), r.r(100.0, 900.0))).collect();
    let clus: Vec<P> = (0..n).map(|i| { let (bx, by, bz) = blobs[i % blobs.len()]; P { p: Point3::new((bx + r.r(-12.0, 12.0)).clamp(0.0, 1000.0), (by + r.r(-12.0, 12.0)).clamp(0.0, 300.0), (bz + r.r(-12.0, 12.0)).clamp(0.0, 1000.0)) } }).collect();
    let queries: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, 1000.0), r.r(0.0, 300.0), r.r(0.0, 1000.0))).collect();

    run("UNIFORM", &uni, &queries, radius);
    println!();
    run("CLUSTERED (6 tight blobs)", &clus, &queries, radius);
    println!("\n→ median split balances by point COUNT, so its depth stays ~log₂(n/leaf) even when\n  the points clump; the midpoint trees halve the empty middle and go deeper on clusters.\n  The payoff shows in the CLUSTERED cull/knn — on uniform data there's nothing to balance.");
}
