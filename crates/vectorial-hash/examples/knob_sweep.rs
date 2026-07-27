//! `knob_sweep` — the **sweet spots**, measured instead of assumed.
//!
//! Every structure in the kit has one tuning knob that decides how deep it goes before a
//! leaf holds points outright: `item_limit` (the trees), `capacity` (the k-d trees),
//! `capacity`/`max_depth` (the linear trees), `levels` (the Morton grids). The docs have
//! been recommending "8–16, profile if it matters" for a long time without a curve to
//! point at. This sweeps each knob over a wide range on the same data and reports **where
//! the optimum actually is**, per operation — build, cull and k-NN want different things,
//! which is the whole reason a single default is a compromise.
//!
//! ```bash
//! cargo run -p vectorial-hash --example knob_sweep --release
//! ```
//! Env: `KS_N` (points, default 200k), `KS_Q` (queries, default 500), `KS_R` (cull radius),
//! `KS_DIST` = `clustered` (default) or `uniform`.
//!
//! Prints a `#M` line per (structure, knob, metric) for `bench-runner`, plus the argmin
//! of each curve as `#M <structure>.best_<op> <knob>`.

use std::time::Instant;
use vectorial_hash::{
    Aabb, Circle, ItemRef, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid, MortonGrid3,
    Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Sphere3, Tree, Tree3,
};

#[derive(Clone, Copy)]
struct P3 { p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { self.p } }
#[derive(Clone, Copy)]
struct P2 { p: Point }
impl Positioned for P2 { fn position(&self) -> Point { self.p } }

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

/// Min-of-`runs` milliseconds: noise only ever adds time, so the fastest sample is the
/// one least disturbed by it.
fn best<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let mut lo = f64::INFINITY;
    for _ in 0..runs { let t = Instant::now(); f(); lo = lo.min(t.elapsed().as_secs_f64()); }
    lo * 1e3
}

/// One curve: knob value -> milliseconds. Prints it, then the argmin.
struct Curve { rows: Vec<(usize, f64)> }
impl Curve {
    fn new() -> Self { Curve { rows: Vec::new() } }
    fn push(&mut self, knob: usize, ms: f64) { self.rows.push((knob, ms)); }
    fn best(&self) -> (usize, f64) { self.rows.iter().fold((0, f64::INFINITY), |acc, &(k, v)| if v < acc.1 { (k, v) } else { acc }) }
    /// How much the worst setting costs over the best — i.e. whether this knob is worth
    /// tuning at all. A flat curve is a finding too.
    fn spread(&self) -> f64 {
        let (lo, hi) = self.rows.iter().fold((f64::INFINITY, 0.0f64), |a, &(_, v)| (a.0.min(v), a.1.max(v)));
        hi / lo.max(1e-12)
    }
    fn report(&self, structure: &str, op: &str) {
        let (bk, bv) = self.best();
        let cells: Vec<String> = self.rows.iter().map(|(k, v)| format!("{k}:{v:.2}")).collect();
        println!("  {structure:<16} {op:<6} {}", cells.join("  "));
        println!("    -> best knob {bk} at {bv:.2} ms; worst setting costs {:.2}x the best", self.spread());
        for (k, v) in &self.rows { println!("#M {structure}.{op}_knob{k} {v:.4} ms"); }
        println!("#M {structure}.best_{op} {bk} knob");
        println!("#M {structure}.{op}_knob_spread {:.3} x", self.spread());
    }
}

fn main() {
    let n: usize = std::env::var("KS_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = std::env::var("KS_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let radius: f64 = std::env::var("KS_R").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let clustered = std::env::var("KS_DIST").map(|s| s != "uniform").unwrap_or(true);
    let w = 1000.0;

    let mut r = Lcg(0x5EED_1234);
    let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(100.0, 900.0), r.r(50.0, 250.0), r.r(100.0, 900.0))).collect();
    let mk3 = |r: &mut Lcg| -> Point3 {
        if clustered { let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
            Point3::new((b.0 + r.r(-14.0, 14.0)).clamp(0.0, w), (b.1 + r.r(-14.0, 14.0)).clamp(0.0, 300.0), (b.2 + r.r(-14.0, 14.0)).clamp(0.0, w)) }
        else { Point3::new(r.r(0.0, w), r.r(0.0, 300.0), r.r(0.0, w)) }
    };
    let items3: Vec<P3> = (0..n).map(|_| P3 { p: mk3(&mut r) }).collect();
    let items2: Vec<P2> = items3.iter().map(|it| P2 { p: Point::new(it.p.x, it.p.z) }).collect();
    let q3: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, w), r.r(0.0, 300.0), r.r(0.0, w))).collect();
    let q2: Vec<Point> = q3.iter().map(|p| Point::new(p.x, p.z)).collect();
    let world3 = Aabb::new(0.0, 0.0, 0.0, w, 300.0, w);
    let world2 = Rect::new(0.0, 0.0, w, w);

    println!("knob sweep | {n} points ({}) | {nq} queries r={radius} | min-of-3\n",
        if clustered { "clustered in 6 blobs" } else { "uniform" });
    println!("  {:<16} {:<6} knob:ms ...", "structure", "op");

    let mut sink = 0usize;
    const LEAF: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

    // ---- Tree3 (3D binary split) -------------------------------------------------
    let (mut b, mut c, mut k) = (Curve::new(), Curve::new(), Curve::new());
    for il in LEAF {
        b.push(il, best(3, || { let _ = Tree3::bulk_load(world3, il, items3.clone()); }));
        let t = Tree3::bulk_load(world3, il, items3.clone());
        c.push(il, best(3, || { for q in &q3 { sink += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } }));
        k.push(il, best(3, || { for q in q3.iter().take(200) { sink += t.knn(*q, 8).len(); } }));
    }
    b.report("tree3", "build"); c.report("tree3", "cull"); k.report("tree3", "knn");

    // ---- MAINTAIN: the knob the static curves above cannot see ---------------------
    // Relocate every item once (the keep-index path, `update_ref`) at each leaf size. A
    // small leaf makes queries cheap but makes items cross leaves more often, and that
    // cost only shows up here — which is the whole reason the static optimum and the
    // dynamic one are different numbers.
    let mut m = Curve::new();
    for il in LEAF {
        let mut t = Tree3::<P3>::new(world3, il);
        let refs: Vec<ItemRef> = items3.iter().map(|it| t.insert_ref(*it).expect("insert")).collect();
        let mut pos: Vec<Point3> = items3.iter().map(|it| it.p).collect();
        let mut step = 0.0f64;
        m.push(il, best(3, || {
            step += 1.7;
            for (i, r) in refs.iter().enumerate() {
                let np = Point3::new((pos[i].x + step % 3.0).min(w), pos[i].y, (pos[i].z + step % 2.0).min(w));
                pos[i] = np;
                t.update_ref(*r, |c| c.p = np);
            }
        }));
    }
    m.report("tree3", "maintain");

    // ---- Octree3 (8-way midpoint) ------------------------------------------------
    let (mut b, mut c) = (Curve::new(), Curve::new());
    for il in LEAF {
        b.push(il, best(3, || { let _ = Octree3::bulk_load(world3, il, items3.clone()); }));
        let t = Octree3::bulk_load(world3, il, items3.clone());
        c.push(il, best(3, || { for q in &q3 { sink += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } }));
    }
    b.report("octree3", "build"); c.report("octree3", "cull");

    // ---- KdTree3 (median split) --------------------------------------------------
    let (mut b, mut c, mut k) = (Curve::new(), Curve::new(), Curve::new());
    for cap in LEAF {
        b.push(cap, best(3, || { let _ = KdTree3::from_items(cap, items3.clone()); }));
        let t = KdTree3::from_items(cap, items3.clone());
        c.push(cap, best(3, || { for q in &q3 { sink += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } }));
        k.push(cap, best(3, || { for q in q3.iter().take(200) { sink += t.knn(*q, 8).len(); } }));
    }
    b.report("kdtree3", "build"); c.report("kdtree3", "cull"); k.report("kdtree3", "knn");

    // ---- LinearOctree3: capacity at a fixed depth, then depth at the best capacity --
    let mut c = Curve::new();
    for cap in LEAF { let t = LinearOctree3::from_items(world3, cap, 12, items3.clone());
        c.push(cap, best(3, || { for q in &q3 { sink += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } })); }
    c.report("linear_octree3", "cull");
    let bestcap = c.best().0;
    let mut d = Curve::new();
    for depth in [4usize, 6, 8, 10, 12, 14, 16] {
        let t = LinearOctree3::from_items(world3, bestcap, depth as u8, items3.clone());
        d.push(depth, best(3, || { for q in &q3 { sink += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } }));
    }
    d.report("linear_octree3", "depth");

    // ---- MortonGrid3: the level count IS the cell size ----------------------------
    let mut c = Curve::new();
    for levels in [3usize, 4, 5, 6, 7, 8] {
        let mut g = MortonGrid3::new(world3, levels as u32);
        for it in &items3 { g.insert(*it); }
        c.push(levels, best(3, || { for q in &q3 { sink += g.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); } }));
    }
    c.report("morton3", "cull");

    // ---- 2D: Tree vs QuadTree vs KdTree2 vs LinearQuadTree vs MortonGrid ----------
    let (mut bt, mut ct) = (Curve::new(), Curve::new());
    let (mut bq, mut cq) = (Curve::new(), Curve::new());
    for il in LEAF {
        bt.push(il, best(3, || { let mut t = Tree::new(world2, il); for it in &items2 { t.insert(*it); } }));
        let mut t = Tree::new(world2, il); for it in &items2 { t.insert(*it); }
        ct.push(il, best(3, || { for q in &q2 { sink += t.cull(&Circle::new(*q, radius)).len(); } }));
        bq.push(il, best(3, || { let mut t = QuadTree::new(world2, il); for it in &items2 { t.insert(*it); } }));
        let mut t = QuadTree::new(world2, il); for it in &items2 { t.insert(*it); }
        cq.push(il, best(3, || { for q in &q2 { sink += t.cull(&Circle::new(*q, radius)).len(); } }));
    }
    bt.report("tree2", "build"); ct.report("tree2", "cull");
    bq.report("quadtree", "build"); cq.report("quadtree", "cull");

    let (mut b, mut c) = (Curve::new(), Curve::new());
    for cap in LEAF {
        b.push(cap, best(3, || { let _ = KdTree2::from_items(cap, items2.clone()); }));
        let t = KdTree2::from_items(cap, items2.clone());
        c.push(cap, best(3, || { for q in &q2 { sink += t.cull(&Circle::new(*q, radius)).len(); } }));
    }
    b.report("kdtree2", "build"); c.report("kdtree2", "cull");

    let mut c = Curve::new();
    for cap in LEAF { let t = LinearQuadTree::from_items(world2, cap, 14, items2.clone());
        c.push(cap, best(3, || { for q in &q2 { sink += t.cull(&Circle::new(*q, radius)).len(); } })); }
    c.report("linear_quadtree", "cull");

    let mut c = Curve::new();
    for levels in [3usize, 4, 5, 6, 7, 8] {
        let mut g = MortonGrid::new(world2, levels as u32);
        for it in &items2 { g.insert(*it); }
        c.push(levels, best(3, || { for q in &q2 { sink += g.cull(&Circle::new(*q, radius)).len(); } }));
    }
    c.report("morton2", "cull");

    println!("\nreading: build always wants a BIGGER leaf (fewer nodes to allocate). Queries want a");
    println!("smaller one (fewer points per leaf to test) only until the descent itself dominates —");
    println!("on 200k points that crossover sits at 32-128, NOT at the 8-16 the docs used to");
    println!("recommend. MAINTAIN is the counterweight: a small leaf means items cross leaves more");
    println!("often, and only the maintain curve can see that. The 'worst costs Nx the best' figure");
    println!("says whether the knob is worth tuning at all for your workload.");
    if sink == usize::MAX { println!("{sink}"); }
}
