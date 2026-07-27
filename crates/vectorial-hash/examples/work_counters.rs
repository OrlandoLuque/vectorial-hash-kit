//! `work_counters` — comparing structures **without a clock at all**.
//!
//! Every timing problem in this repo (background load, cache state, turbo, run order)
//! comes from measuring *seconds*. But what actually differs between a binary tree, an
//! octree, a k-d tree and a grid is **how much work each one does**: how many node boxes
//! it has to classify, and how many points it has to test, to answer the same query.
//! Those are integers. They are identical on every run, on any machine, under any load —
//! run this twice while compiling something and the numbers do not move by one.
//!
//! It needs no library changes: `cull` takes any `Shape3`, so the query itself is wrapped
//! in a counter that tallies each `classify_aabb` / `contains_point` the traversal asks
//! for, and forwards to the real shape.
//!
//! What the columns mean:
//! - **boxes** — node boxes classified per query. The descent cost.
//! - **tested** — points the traversal had to test individually. The leaf cost.
//! - **found** — points actually returned (identical for every structure, or one of them
//!   is wrong — this doubles as a correctness check).
//! - **waste** — tested / found. Points looked at per point returned. Note it can be
//!   BELOW 1: when a node box falls entirely inside the query, the traversal takes all of
//!   its items without testing any of them, so a low ratio is that fully-inside shortcut
//!   working, not a rounding error.
//!
//! Time still matters for constant factors (a grid's flat array is friendlier to the
//! prefetcher than a pointer chase), so this does not replace the timed benches — it
//! tells you which part of a timing difference is *algorithmic* and which is the machine.
//!
//! ```bash
//! cargo run -p vectorial-hash --example work_counters --release
//! ```
//! Env: `WC_N`, `WC_Q`, `WC_R`, `WC_LEAF`, `WC_DIST=clustered|uniform`.

use std::cell::Cell;
use vectorial_hash::{
    Aabb, Circle, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid, MortonGrid3,
    Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Shape, Shape3, Sphere3,
    Tree, Tree3,
};
use vectorial_hash::template::CellState;

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

// ------------------------------------------------------ the counting query volumes

#[derive(Default)]
struct Tally { boxes: Cell<u64>, tested: Cell<u64> }
impl Tally {
    fn take(&self) -> (u64, u64) { (self.boxes.replace(0), self.tested.replace(0)) }
}

/// A `Shape3` that forwards to the real one and counts what the traversal asked it.
struct Counted3<'a, S: Shape3> { inner: S, t: &'a Tally }
impl<S: Shape3> Shape3 for Counted3<'_, S> {
    fn bounding_box(&self) -> Aabb { self.inner.bounding_box() }
    fn contains_point(&self, p: Point3) -> bool { self.t.tested.set(self.t.tested.get() + 1); self.inner.contains_point(p) }
    fn classify_aabb(&self, b: &Aabb) -> CellState { self.t.boxes.set(self.t.boxes.get() + 1); self.inner.classify_aabb(b) }
}

/// The 2D counterpart. `classify_box` is optional in `Shape`, so a structure that skips it
/// simply reports zero boxes — which is itself the finding for a flat grid.
struct Counted2<'a, S: Shape> { inner: S, t: &'a Tally }
impl<S: Shape> Shape for Counted2<'_, S> {
    fn bounding_box(&self) -> Rect { self.inner.bounding_box() }
    fn contains_point(&self, p: Point) -> bool { self.t.tested.set(self.t.tested.get() + 1); self.inner.contains_point(p) }
    fn classify_box(&self, b: &Rect) -> Option<CellState> { self.t.boxes.set(self.t.boxes.get() + 1); self.inner.classify_box(b) }
}

struct Row { name: &'static str, boxes: f64, tested: f64, found: f64 }
fn print_rows(title: &str, rows: &[Row]) {
    println!("\n{title}");
    println!("  {:<18} {:>10} {:>10} {:>9} {:>8}", "structure", "boxes/q", "tested/q", "found/q", "waste");
    let base = rows.iter().map(|r| r.found).fold(f64::NAN, f64::max);
    for r in rows {
        let waste = if r.found > 0.0 { r.tested / r.found } else { f64::NAN };
        let flag = if (r.found - base).abs() > 1e-9 { "  <-- DIFFERENT RESULT SET" } else { "" };
        println!("  {:<18} {:>10.1} {:>10.1} {:>9.1} {:>8.2}{flag}", r.name, r.boxes, r.tested, r.found, waste);
        println!("#M {}.boxes_per_query {:.2} n", r.name, r.boxes);
        println!("#M {}.tested_per_query {:.2} n", r.name, r.tested);
        println!("#M {}.waste_ratio {:.3} x", r.name, waste);
    }
}

fn main() {
    let n: usize = std::env::var("WC_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = std::env::var("WC_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let radius: f64 = std::env::var("WC_R").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let leaf: usize = std::env::var("WC_LEAF").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let clustered = std::env::var("WC_DIST").map(|s| s != "uniform").unwrap_or(true);
    let w = 1000.0;

    let mut r = Lcg(0x5EED_1234);
    let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(100.0, 900.0), r.r(50.0, 250.0), r.r(100.0, 900.0))).collect();
    let items3: Vec<P3> = (0..n).map(|_| {
        let p = if clustered {
            let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
            Point3::new((b.0 + r.r(-14.0, 14.0)).clamp(0.0, w), (b.1 + r.r(-14.0, 14.0)).clamp(0.0, 300.0), (b.2 + r.r(-14.0, 14.0)).clamp(0.0, w))
        } else { Point3::new(r.r(0.0, w), r.r(0.0, 300.0), r.r(0.0, w)) };
        P3 { p }
    }).collect();
    let items2: Vec<P2> = items3.iter().map(|it| P2 { p: Point::new(it.p.x, it.p.z) }).collect();
    let q3: Vec<Point3> = (0..nq).map(|_| Point3::new(r.r(0.0, w), r.r(0.0, 300.0), r.r(0.0, w))).collect();
    let q2: Vec<Point> = q3.iter().map(|p| Point::new(p.x, p.z)).collect();
    let world3 = Aabb::new(0.0, 0.0, 0.0, w, 300.0, w);
    let world2 = Rect::new(0.0, 0.0, w, w);

    println!("work counters | {n} points ({}) | {nq} queries r={radius} | leaf {leaf}",
        if clustered { "clustered in 6 blobs" } else { "uniform" });
    println!("no clock involved: these are counts, identical on every run and every machine.");

    let t = Tally::default();
    let per = |f: &mut dyn FnMut(&Counted3<Sphere3>) -> usize| {
        let (mut bx, mut te, mut fo) = (0u64, 0u64, 0u64);
        for q in &q3 {
            t.take();
            let hits = f(&Counted3 { inner: Sphere3::new(q.x, q.y, q.z, radius), t: &t });
            let (b, s) = t.take();
            bx += b; te += s; fo += hits as u64;
        }
        (bx as f64 / nq as f64, te as f64 / nq as f64, fo as f64 / nq as f64)
    };

    let tree3 = Tree3::bulk_load(world3, leaf, items3.clone());
    let oct3 = Octree3::bulk_load(world3, leaf, items3.clone());
    let kd3 = KdTree3::from_items(leaf, items3.clone());
    let lin3 = LinearOctree3::from_items(world3, leaf, 12, items3.clone());
    let mut mor3 = MortonGrid3::new(world3, MortonGrid3::<P3>::levels_for_cell_size(world3, radius));
    for it in &items3 { mor3.insert(*it); }

    let mut rows = Vec::new();
    let (b, s, f) = per(&mut |q| tree3.cull(q).len()); rows.push(Row { name: "tree3", boxes: b, tested: s, found: f });
    let (b, s, f) = per(&mut |q| oct3.cull(q).len()); rows.push(Row { name: "octree3", boxes: b, tested: s, found: f });
    let (b, s, f) = per(&mut |q| kd3.cull(q).len()); rows.push(Row { name: "kdtree3", boxes: b, tested: s, found: f });
    let (b, s, f) = per(&mut |q| lin3.cull(q).len()); rows.push(Row { name: "linear_octree3", boxes: b, tested: s, found: f });
    let (b, s, f) = per(&mut |q| mor3.cull(q).len()); rows.push(Row { name: "morton3", boxes: b, tested: s, found: f });
    print_rows("3D — work per sphere cull", &rows);

    let per2 = |f: &mut dyn FnMut(&Counted2<Circle>) -> usize| {
        let (mut bx, mut te, mut fo) = (0u64, 0u64, 0u64);
        for q in &q2 {
            t.take();
            let hits = f(&Counted2 { inner: Circle::new(*q, radius), t: &t });
            let (b, s) = t.take();
            bx += b; te += s; fo += hits as u64;
        }
        (bx as f64 / nq as f64, te as f64 / nq as f64, fo as f64 / nq as f64)
    };

    let tree2 = { let mut t = Tree::new(world2, leaf); for it in &items2 { t.insert(*it); } t };
    let quad = { let mut t = QuadTree::new(world2, leaf); for it in &items2 { t.insert(*it); } t };
    let kd2 = KdTree2::from_items(leaf, items2.clone());
    let lin2 = LinearQuadTree::from_items(world2, leaf, 14, items2.clone());
    let mor2 = { let mut g = MortonGrid::new(world2, MortonGrid::<P2>::levels_for_cell_size(world2, radius)); for it in &items2 { g.insert(*it); } g };

    let mut rows = Vec::new();
    let (b, s, f) = per2(&mut |q| tree2.cull(q).len()); rows.push(Row { name: "tree2", boxes: b, tested: s, found: f });
    let (b, s, f) = per2(&mut |q| quad.cull(q).len()); rows.push(Row { name: "quadtree", boxes: b, tested: s, found: f });
    let (b, s, f) = per2(&mut |q| kd2.cull(q).len()); rows.push(Row { name: "kdtree2", boxes: b, tested: s, found: f });
    let (b, s, f) = per2(&mut |q| lin2.cull(q).len()); rows.push(Row { name: "linear_quadtree", boxes: b, tested: s, found: f });
    let (b, s, f) = per2(&mut |q| mor2.cull(q).len()); rows.push(Row { name: "morton2", boxes: b, tested: s, found: f });
    print_rows("2D — work per circle cull", &rows);

    println!("\nreading: 'found' must match across a block or a structure is answering a different");
    println!("question. 'waste' is the algorithmic quality of the index — points looked at per point");
    println!("returned. 'boxes' is what the descent costs. A structure can lose on time while winning");
    println!("here, and then the difference is the machine (cache, prefetch), not the algorithm.");
}
