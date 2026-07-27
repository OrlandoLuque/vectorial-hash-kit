//! `LinearQuadTree` vs `QuadTree` (pointer quadtree) vs `MortonGrid` (uniform grid):
//! build, circle-cull and k-NN over the same clustered 2D point set — the 2D twin of
//! `linear_octree3_bench`. Measures whether the adaptive-but-pointer-free layout pays
//! against the two structures it sits between.
//!
//! ```bash
//! cargo run -p vectorial-hash --example linear_quadtree_bench --release
//! ```
//! Env: `LQ_N` (points), `LQ_Q` (queries), `LQ_R` (cull radius).

use std::time::Instant;
use vectorial_hash::linear_quadtree::LinearQuadTree;
use vectorial_hash::{Circle, KdTree2, MortonGrid, Point, Positioned, QuadTree, Rect};

#[derive(Clone, Copy)]
struct P { p: Point }
impl Positioned for P { fn position(&self) -> Point { self.p } }

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
    let n: usize = std::env::var("LQ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = std::env::var("LQ_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let radius: f64 = std::env::var("LQ_R").ok().and_then(|s| s.parse().ok()).unwrap_or(40.0);
    let world = Rect::new(0.0, 0.0, 1000.0, 1000.0);

    // A few dense blobs in a sparse field — the case single-cell grids handle worst.
    let mut r = Lcg(0x1234_5678);
    let blobs: Vec<(f64, f64)> = (0..8).map(|_| (r.r(80.0, 920.0), r.r(80.0, 920.0))).collect();
    let items: Vec<P> = (0..n).map(|i| {
        let p = if i % 3 == 0 { Point::new(r.r(0.0, 1000.0), r.r(0.0, 1000.0)) }
                else { let (bx, by) = blobs[i % blobs.len()]; Point::new((bx + r.r(-30.0, 30.0)).clamp(0.0, 1000.0), (by + r.r(-30.0, 30.0)).clamp(0.0, 1000.0)) };
        P { p }
    }).collect();
    let queries: Vec<Point> = (0..nq).map(|_| Point::new(r.r(0.0, 1000.0), r.r(0.0, 1000.0))).collect();

    println!("LinearQuadTree bench — {n} points ({:.0}% clustered), {nq} queries, cull r={radius}\n", 200.0 / 3.0);

    let t_build_lin = best(5, || { let _ = LinearQuadTree::from_items(world, 32, 18, items.clone()); });
    let t_build_qt = best(5, || { let mut q = QuadTree::new(world, 32); for it in &items { q.insert(*it); } });
    let t_build_mor = best(5, || { let mut g = MortonGrid::new(world, 6); for it in &items { g.insert(*it); } });
    let t_build_kd = best(5, || { let _ = KdTree2::from_items(32, items.clone()); });
    #[cfg(feature = "parallel")]
    let t_build_kd_par = best(5, || { let _ = KdTree2::from_items_par(32, items.clone()); });

    let lin = LinearQuadTree::from_items(world, 32, 18, items.clone());
    let mut qt = QuadTree::new(world, 32);
    for it in &items { qt.insert(*it); }
    let mut mor = MortonGrid::new(world, 6);
    for it in &items { mor.insert(*it); }
    let kd = KdTree2::from_items(32, items.clone());

    let mut sink = 0usize;
    let t_cull_lin = best(6, || { for q in &queries { sink += lin.cull(&Circle::new(*q, radius)).len(); } });
    let t_cull_qt = best(6, || { for q in &queries { sink += qt.cull(&Circle::new(*q, radius)).len(); } });
    let t_cull_mor = best(6, || { for q in &queries { sink += mor.cull(&Circle::new(*q, radius)).len(); } });

    let t_knn_lin = best(6, || { for q in &queries { sink += lin.knn(*q, 8).len(); } });
    let t_knn_qt = best(6, || { for q in &queries { sink += qt.knn(*q, 8).len(); } });
    let t_knn_mor = best(6, || { for q in &queries { sink += mor.knn(*q, 8).len(); } });
    let t_cull_kd = best(6, || { for q in &queries { sink += kd.cull(&Circle::new(*q, radius)).len(); } });
    let t_knn_kd = best(6, || { for q in &queries { sink += kd.knn(*q, 8).len(); } });

    println!("structure       build(ms)   cull {nq}q(ms)   knn {nq}q(ms)   leaves/cells   depth");
    println!("LinearQuadTree  {t_build_lin:8.2}   {t_cull_lin:11.2}   {t_knn_lin:10.2}   {:>12}   {}", lin.leaf_count(), lin.depth());
    println!("QuadTree        {t_build_qt:8.2}   {t_cull_qt:11.2}   {t_knn_qt:10.2}   {:>12}   —", "—");
    println!("MortonGrid      {t_build_mor:8.2}   {t_cull_mor:11.2}   {t_knn_mor:10.2}   {:>12}   1 (flat)", mor.cell_count());
    println!("KdTree2 median  {t_build_kd:8.2}   {t_cull_kd:11.2}   {t_knn_kd:10.2}   {:>12}   {}", kd.node_count(), kd.depth());
    #[cfg(feature = "parallel")]
    println!("  KdTree2 parallel build {t_build_kd_par:.2} ms ({:.2}x, {} threads)", t_build_kd / t_build_kd_par, rayon::current_num_threads());
    println!("\ncull speed   vs QuadTree {:.2}x   vs Morton {:.2}x", t_cull_qt / t_cull_lin, t_cull_mor / t_cull_lin);
    println!("knn  speed   vs QuadTree {:.2}x   vs Morton {:.2}x", t_knn_qt / t_knn_lin, t_knn_mor / t_knn_lin);

    // ---- maintain (per-frame relocate ALL points): keep-index vs rebuild ----
    // QuadTree relocates in place via the stable ItemRef (O(1) while an item stays in
    // its leaf); Morton and LinearQuadTree have no in-place handle, so their "maintain"
    // IS a full rebuild — the reason a rebuild-per-frame structure wants a cheap build.
    let mut qk = QuadTree::new(world, 32);
    let refs: Vec<_> = items.iter().map(|it| qk.insert_ref(*it).unwrap()).collect();
    let mut jr = Lcg(0xBEEF);
    let t_maint_qt = best(6, || {
        for (i, &rf) in refs.iter().enumerate() {
            let np = Point::new((items[i].p.x + jr.r(-0.5, 0.5)).clamp(1.0, 999.0), (items[i].p.y + jr.r(-0.5, 0.5)).clamp(1.0, 999.0));
            qk.update_ref(rf, |p| p.p = np);
        }
    });
    let t_maint_mor = best(6, || { let mut g = MortonGrid::new(world, 6); for it in &items { g.insert(*it); } std::hint::black_box(&g); });
    let t_maint_lin = best(6, || { let g = LinearQuadTree::from_items(world, 32, 18, items.clone()); std::hint::black_box(&g); });
    println!("\nmaintain, relocate all {n}/frame:  QuadTree keep-index {t_maint_qt:7.2} ms | Morton rebuild {t_maint_mor:7.2} ms | LinearQuadTree rebuild {t_maint_lin:7.2} ms");
    println!("  → the 2D echo of the 3D decision map: the keep-index QuadTree beats the LinearQuadTree rebuild {:.2}× on relocate-all", t_maint_lin / t_maint_qt);
    println!("    (so on MOVING data prefer the kept tree; LinearQuadTree's edge is STATIC/rebuild-often skewed data — cheap build, adaptive query)");
    // Machine-readable lines for `bench-runner`.
    println!("#M build_kdtree2 {t_build_kd:.3} ms");
    println!("#M build_quadtree {t_build_qt:.3} ms");
    println!("#M build_linear_quadtree {t_build_lin:.3} ms");
    println!("#M build_morton {t_build_mor:.3} ms");
    println!("#M cull_kdtree2 {t_cull_kd:.3} ms");
    println!("#M cull_quadtree {t_cull_qt:.3} ms");
    println!("#M cull_linear_quadtree {t_cull_lin:.3} ms");
    println!("#M cull_morton {t_cull_mor:.3} ms");
    println!("#M knn_kdtree2 {t_knn_kd:.3} ms");
    println!("#M knn_quadtree {t_knn_qt:.3} ms");
    println!("#M cull_ratio_kd2_over_quadtree {:.3} x", t_cull_qt / t_cull_kd);
    println!("#M maintain_quadtree_keep {t_maint_qt:.3} ms");
    println!("#M maintain_linear_rebuild {t_maint_lin:.3} ms");
    #[cfg(feature = "parallel")]
    println!("#M build_kdtree2_speedup {:.3} x", t_build_kd / t_build_kd_par);
    if sink == usize::MAX { println!("{sink}"); }
}
