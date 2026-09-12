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

#[path = "common/mod.rs"]
mod common;
use vectorial_hash::linear_quadtree::LinearQuadTree;
use vectorial_hash::{Circle, KdTree2, MortonGrid, Point, Positioned, QuadTree, Rect};

#[derive(Clone, Copy)]
/// The `id` exists so `MortonGrid::update` / `LinearQuadTree::update` have a predicate that
/// identifies ONE item. Matching on position instead would be ambiguous the moment two points
/// coincide, and this workload piles 67 % of them into eight blobs.
struct P { id: u32, p: Point }
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
        P { id: i as u32, p }
    }).collect();
    let queries: Vec<Point> = (0..nq).map(|_| Point::new(r.r(0.0, 1000.0), r.r(0.0, 1000.0))).collect();

    println!("LinearQuadTree bench — {n} points ({:.0}% clustered), {nq} queries, cull r={radius}\n", 200.0 / 3.0);

    // Every arm gets its input prepared OUTSIDE the clock, whether or not it consumes it.
    //
    // This used to clone inside the timer for the two `from_items` arms and not for the two
    // insert arms, which charged the build-once structures for an allocation their rivals never
    // paid — on the one table that concludes the k-d tree has the fastest 2D build. (It still
    // does, so the conclusion was safe; the margin was not.) A clone inside the clock is also
    // not the neutral constant it looks like: see `docs/MEASURING.md` § 8g, where the same
    // mistake reported a 2.65x speed-up as 0.85x.
    let t_build_lin = common::wall_ms_consuming(5, &items, |v| { let _ = LinearQuadTree::from_items(world, 32, 18, v); });
    let t_build_qt = common::wall_ms_consuming(5, &items, |v| { let mut q = QuadTree::new(world, 32); for it in &v { q.insert(*it); } std::hint::black_box(&q); });
    let t_build_mor = common::wall_ms_consuming(5, &items, |v| { let mut g = MortonGrid::new(world, 6); for it in &v { g.insert(*it); } std::hint::black_box(&g); });
    let t_build_kd = common::wall_ms_consuming(5, &items, |v| { let _ = KdTree2::from_items(32, v); });
    #[cfg(feature = "parallel")]
    let t_build_kd_par = common::wall_ms_consuming(5, &items, |v| { let _ = KdTree2::from_items_par(32, v); });

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
    //
    // ★ CORRECTED 2026-09-13. This section used to have THREE arms — QuadTree keeping via
    // `update_ref`, and Morton and LinearQuadTree each doing a full rebuild — on the stated
    // grounds that "Morton and LinearQuadTree have no in-place handle". That stopped being true
    // when `MortonGrid::update` landed (#122) and `LinearQuadTree::update` followed (#127), and
    // the sweep that was supposed to catch every such site (#138) missed this bench. So the
    // comparison was keep-vs-rebuild, read as a verdict about the structures.
    //
    // Both paths are now measured for all three, and the result is NOT what the fix was expected
    // to show. I wrote the summary line asserting that keeping would win for the other two — it
    // loses, 3-4x. The curve was already known (`grid_keep_bench`: crossover near 70 % moving) and
    // this arm relocates 100 % of the population every frame, the one end where a rebuild wins.
    // What the 2D numbers add is WHY it loses so badly: `update` still pays a lookup and a
    // predicate scan per call even when the item has not changed cell, so at 100 % churn it is
    // 200 000 lookups against a single sequential refill. `QuadTree` is exempt only because
    // `update_ref` is O(1) — the handle layer, not the method, is what makes keeping cheap.
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

    // The same jitter, through the KEEP path the other two structures also have.
    let mut mk = MortonGrid::new(world, 6);
    for it in &items { mk.insert(*it); }
    let mut mpos: Vec<Point> = items.iter().map(|it| it.p).collect();
    let t_keep_mor = best(6, || {
        for (i, p) in mpos.iter_mut().enumerate() {
            let np = Point::new((p.x + jr.r(-0.5, 0.5)).clamp(1.0, 999.0), (p.y + jr.r(-0.5, 0.5)).clamp(1.0, 999.0));
            let id = i as u32;
            mk.update(*p, |it: &P| it.id == id, |it: &mut P| it.p = np);
            *p = np;
        }
    });
    let mut lk = LinearQuadTree::from_items(world, 32, 18, items.clone());
    let mut lpos: Vec<Point> = items.iter().map(|it| it.p).collect();
    let t_keep_lin = best(6, || {
        for (i, p) in lpos.iter_mut().enumerate() {
            let np = Point::new((p.x + jr.r(-0.5, 0.5)).clamp(1.0, 999.0), (p.y + jr.r(-0.5, 0.5)).clamp(1.0, 999.0));
            let id = i as u32;
            lk.update(*p, |it: &P| it.id == id, |it: &mut P| it.p = np);
            *p = np;
        }
    });

    println!("\nmaintain, relocate all {n}/frame — both paths, for all three:");
    println!("  {:<16} {:>10} {:>10}", "structure", "keep ms", "rebuild ms");
    println!("  {:<16} {t_maint_qt:>10.2} {:>10}", "QuadTree", "—");
    println!("  {:<16} {t_keep_mor:>10.2} {t_maint_mor:>10.2}", "MortonGrid");
    println!("  {:<16} {t_keep_lin:>10.2} {t_maint_lin:>10.2}", "LinearQuadTree");
    println!("  → and REBUILDING WINS for both of them: Morton {:.2}x, LinearQuadTree {:.2}x faster",
             t_keep_mor / t_maint_mor, t_keep_lin / t_maint_lin);
    println!("    to rebuild than to keep. That is not a contradiction of #122/#127, it is their");
    println!("    curve read at its far end: `grid_keep_bench` puts the crossover near 70 % moving");
    println!("    and this relocates 100 %, every frame. Rebuilding costs the same whatever moved.");
    println!();
    println!("    The mechanism is the one `grid_update_cost` named: UPDATE SAVES THE CALLS YOU DO");
    println!("    NOT MAKE, NOT THE CALLS YOU DO. The jitter here is +-0.5 wu against ~15.6 wu");
    println!("    cells, so almost nothing changes cell — and it does not matter, because each of");
    println!("    the 200 000 calls still pays a lookup plus a predicate scan over its bucket");
    println!("    (mean ~49 items at levels 6). `QuadTree` escapes that and wins outright because");
    println!("    `update_ref` is O(1): the ItemRef IS the index, so there is nothing to look up.");
    println!("    **The handle layer is what makes keeping cheap, not the existence of `update`.**");
    println!();
    println!("    So the old three-arm line got the RANKING right and the REASON wrong: it said the");
    println!("    other two could not keep (true until #122/#127, false since) when the real point");
    println!("    is that they can and, without handles and at this churn, should not.");
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
    // The quoted ratio is measured PAIRED (A/B/B/A, median of per-round ratios): taking
    // the two culls separately made the equivalent 3D figure swing 1.57-3.28 between runs.
    let (mut sa, mut sb) = (0usize, 0usize);
    let (_, _, ratio_kd2, ratio_spread) = common::compare2(7,
        || { for q in &queries { sa += kd.cull(&Circle::new(*q, radius)).len(); } },
        || { for q in &queries { sb += qt.cull(&Circle::new(*q, radius)).len(); } });
    sink += sa + sb;
    println!("#M cull_ratio_kd2_over_quadtree {:.3} x", t_cull_qt / t_cull_kd);
    println!("#M cull_ratio_kd2_paired {ratio_kd2:.3} x");
    println!("#M cull_ratio_kd2_paired_spread {ratio_spread:.1} pct");
    println!("#M maintain_quadtree_keep {t_maint_qt:.3} ms");
    println!("#M maintain_linear_rebuild {t_maint_lin:.3} ms");
    #[cfg(feature = "parallel")]
    println!("#M build_kdtree2_speedup {:.3} x", t_build_kd / t_build_kd_par);
    if sink == usize::MAX { println!("{sink}"); }
}
