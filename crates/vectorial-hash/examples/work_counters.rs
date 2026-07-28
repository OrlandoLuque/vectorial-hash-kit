//! `work_counters` — comparing structures **without a clock at all**.
//!
//! Every timing problem in this repo (background load, cache state, turbo, run order)
//! comes from measuring *seconds*. But what actually differs between a binary tree, an
//! octree, a k-d tree and a grid is **how much work each one does**: how many node boxes
//! it has to classify, and how many points it has to test, to answer the same query.
//! Those are integers. They are identical on every run, on any machine, under any load —
//! run this twice while compiling something and the numbers do not move by one.
//!
//! It needs no library changes. `cull` takes any `Shape3`, so the query itself is wrapped
//! in a counter that tallies each `classify_aabb` / `contains_point` the traversal asks
//! for, and forwards to the real shape. `knn` and `raycast` take a point and a ray, so
//! there is nothing to wrap on that side — and their APIs do not even agree with each
//! other. For those the counter goes in the **item**: every traversal has to ask an item
//! where it is before it can test it, so counting `position()` counts the leaf work for
//! all three verbs, uniformly. (A structure that reads the same position twice counts
//! twice, which is correct: that is work it is doing.)
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
//! Three findings that came out of the counts alone, no clock anywhere:
//!
//! - **The median split earns its keep only in clusters.** `KdTree3`'s k-NN tests 219
//!   points per query against `Tree3`'s 404 on clustered data (1.8x) — and 86.6 against
//!   92.1 when the same points are spread uniformly (1.06x). The advantage is not the
//!   structure, it is the structure *meeting skewed data*.
//! - **A uniform grid's k-NN collapses on clustered data**: `MortonGrid3` tests 596 points
//!   per query uniform, **166 640** clustered — 280x, because the shell expansion has to
//!   cross empty cells until it reaches a blob and then swallows the whole blob at once.
//! - **The DDA walks are honest but partial.** They visit only leaves the centre ray
//!   crosses, so they return a strict subset of the exact capsule: 75% of the hits for 23%
//!   of the point tests (`Tree3`), 50% for 11% (`Octree3`). Verified subset here — zero
//!   invented hits — which is the documented guarantee, now checked.
//!
//! Env: `WC_N`, `WC_Q`, `WC_R`, `WC_LEAF`, `WC_K`, `WC_RAY_R`, `WC_DIST=clustered|uniform`.

use std::cell::Cell;
use vectorial_hash::{
    Aabb, Circle, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid, MortonGrid3,
    Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Shape, Shape3, Sphere3,
    Tree, Tree3,
};
use vectorial_hash::template::CellState;

// Calls to `position()`, i.e. **points the traversal actually looked at**.
//
// `cull` takes a `Shape`, so it can be counted by wrapping the query. `knn` and `raycast`
// take a point and a ray, so there is nothing to wrap on that side — and their APIs do not
// even agree with each other (three structures return a `RaycastOut` carrying counters, the
// rest return a bare `Vec`). The counter that works for all three verbs and all eleven
// structures goes in the **item** instead: every traversal must ask an item where it is
// before it can test it, so counting that call counts the leaf work, uniformly, with no
// library change at all.
thread_local! { static POS: Cell<u64> = const { Cell::new(0) }; }
fn pos_take() -> u64 { POS.with(|c| c.replace(0)) }
#[inline]
fn pos_hit() { POS.with(|c| c.set(c.get() + 1)); }

#[derive(Clone, Copy)]
struct P3 { p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { pos_hit(); self.p } }
#[derive(Clone, Copy)]
struct P2 { p: Point }
impl Positioned for P2 { fn position(&self) -> Point { pos_hit(); self.p } }

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

/// A row for `knn`/`raycast`: no `boxes` column, because neither verb classifies the query
/// against a node box — they compare distances to it. Leaf work is the whole story here.
struct QRow { name: String, tested: f64, found: f64 }
fn print_qrows(title: &str, found_label: &str, rows: &[QRow]) {
    println!("
{title}");
    println!("  {:<22} {:>12} {:>12} {:>10}", "structure", "tested/query", found_label, "waste");
    for r in rows {
        let waste = if r.found > 0.0 { r.tested / r.found } else { f64::NAN };
        println!("  {:<22} {:>12.1} {:>12.2} {:>9.1}x", r.name, r.tested, r.found, waste);
        println!("#M {}.tested_per_query {:.2} n", r.name, r.tested);
        println!("#M {}.waste_ratio {:.3} x", r.name, waste);
    }
}

/// Same distances, in order, allowing for ties. If two structures disagree here one of them
/// is wrong and the work counts above are comparing different answers.
fn same_dists(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| (p - q).abs() <= 1e-9 * (1.0 + q.abs()))
    })
}

/// A corridor walk must never invent a hit. Returns `(recall, hits_not_in_the_exact_set)`:
/// the second number is a correctness failure, the first is the price of being cheap.
fn subset_recall(sub: &[Vec<f64>], sup: &[Vec<f64>]) -> (f64, usize) {
    let (mut found, mut total, mut extra) = (0usize, 0usize, 0usize);
    for (a, b) in sub.iter().zip(sup) {
        total += b.len();
        for x in a {
            if b.iter().any(|y| (x - y).abs() <= 1e-9 * (1.0 + y.abs())) { found += 1; } else { extra += 1; }
        }
    }
    (if total > 0 { found as f64 / total as f64 } else { f64::NAN }, extra)
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
    // Its own generator: drawing the rays from `r` would shift every random number after
    // it and silently change the cull numbers above, which are published.
    let mut rr = Lcg(0x0A11_CE00);
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

    // -------------------------------------------------------------- k-NN
    // Same structures, a verb with no query volume at all: k-NN prunes on distance to the
    // node box, so there is no `classify` to count and leaf work is the whole cost. `k`
    // neighbours always come back, so `waste` reads directly as "points examined per
    // neighbour delivered" — the cleanest quality number in the file.
    let k: usize = std::env::var("WC_K").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let knn3 = |f: &mut dyn FnMut(Point3) -> Vec<f64>| -> (f64, f64, Vec<Vec<f64>>) {
        let (mut te, mut fo) = (0u64, 0u64);
        let mut all = Vec::with_capacity(nq);
        for q in &q3 { pos_take(); let d = f(*q); te += pos_take(); fo += d.len() as u64; all.push(d); }
        (te as f64 / nq as f64, fo as f64 / nq as f64, all)
    };
    let mut rows = Vec::new();
    let (mut ref3, mut bad3): (Option<Vec<Vec<f64>>>, Vec<&str>) = (None, Vec::new());
    for (name, f) in [
        ("knn_tree3", &mut (|q| tree3.knn(q, k).iter().map(|(d, _)| *d).collect()) as &mut dyn FnMut(Point3) -> Vec<f64>),
        ("knn_octree3", &mut |q| oct3.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_kdtree3", &mut |q| kd3.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_linear_octree3", &mut |q| lin3.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_morton3", &mut |q| mor3.knn(q, k).iter().map(|(d, _)| *d).collect()),
    ] {
        let (te, fo, all) = knn3(f);
        match &ref3 { None => ref3 = Some(all), Some(r) if !same_dists(r, &all) => bad3.push(name), _ => {} }
        rows.push(QRow { name: name.to_string(), tested: te, found: fo });
    }
    print_qrows(&format!("3D - work per k-NN query (k={k})"), "neighbours", &rows);
    println!("  agreement: {}", if bad3.is_empty() { "EXACT (same k distances everywhere)".to_string() } else { format!("MISMATCH in {bad3:?}") });

    let knn2 = |f: &mut dyn FnMut(Point) -> Vec<f64>| -> (f64, f64, Vec<Vec<f64>>) {
        let (mut te, mut fo) = (0u64, 0u64);
        let mut all = Vec::with_capacity(nq);
        for q in &q2 { pos_take(); let d = f(*q); te += pos_take(); fo += d.len() as u64; all.push(d); }
        (te as f64 / nq as f64, fo as f64 / nq as f64, all)
    };
    let mut rows = Vec::new();
    let (mut ref2, mut bad2): (Option<Vec<Vec<f64>>>, Vec<&str>) = (None, Vec::new());
    for (name, f) in [
        ("knn_tree2", &mut (|q| tree2.knn(q, k).iter().map(|(d, _)| *d).collect()) as &mut dyn FnMut(Point) -> Vec<f64>),
        ("knn_quadtree", &mut |q| quad.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_kdtree2", &mut |q| kd2.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_linear_quadtree", &mut |q| lin2.knn(q, k).iter().map(|(d, _)| *d).collect()),
        ("knn_morton2", &mut |q| mor2.knn(q, k).iter().map(|(d, _)| *d).collect()),
    ] {
        let (te, fo, all) = knn2(f);
        match &ref2 { None => ref2 = Some(all), Some(r) if !same_dists(r, &all) => bad2.push(name), _ => {} }
        rows.push(QRow { name: name.to_string(), tested: te, found: fo });
    }
    print_qrows(&format!("2D - work per k-NN query (k={k})"), "neighbours", &rows);
    println!("  agreement: {}", if bad2.is_empty() { "EXACT (same k distances everywhere)".to_string() } else { format!("MISMATCH in {bad2:?}") });

    // -------------------------------------------------------------- raycast
    // Rays start anywhere in the world and are aimed at a jittered blob centre, so every one
    // passes through dense data: a ray that hits nothing measures the empty-space skip and
    // nothing else, and a table of those would be vacuous (docs/MEASURING.md). The row pair
    // worth staring at is descent vs DDA on the SAME structure - same answer, two ways of
    // walking to it, and the counts say which one has to look at less.
    let ray_r: f64 = std::env::var("WC_RAY_R").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let rays: Vec<(Point3, Point3)> = (0..nq).map(|_| {
        let b = blobs[(rr.f() * blobs.len() as f64) as usize % blobs.len()];
        let aim = Point3::new(b.0 + rr.r(-14.0, 14.0), b.1 + rr.r(-14.0, 14.0), b.2 + rr.r(-14.0, 14.0));
        let o = Point3::new(rr.r(0.0, w), rr.r(0.0, 300.0), rr.r(0.0, w));
        let (dx, dy, dz) = (aim.x - o.x, aim.y - o.y, aim.z - o.z);
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-9);
        (o, Point3::new(dx / len, dy / len, dz / len))
    }).collect();
    let per_ray = |f: &mut dyn FnMut(Point3, Point3) -> Vec<f64>| -> (f64, f64, Vec<Vec<f64>>) {
        let (mut te, mut fo) = (0u64, 0u64);
        let mut all = Vec::with_capacity(nq);
        for (o, d) in &rays { pos_take(); let h = f(*o, *d); te += pos_take(); fo += h.len() as u64; all.push(h); }
        (te as f64 / nq as f64, fo as f64 / nq as f64, all)
    };
    fn srt(mut v: Vec<f64>) -> Vec<f64> { v.sort_by(f64::total_cmp); v }
    // Two families here, and conflating them would report a bug that is not one. `raycast`
    // culls a real capsule, so it is EXACT and every structure must return the same hits.
    // The DDA walks (and the Morton cell walk) visit only the leaves/cells the CENTRE ray
    // crosses, so an item within `radius` of the ray in a leaf the centreline misses is not
    // reported: documented, deliberate, and a strict SUBSET of the exact answer. The useful
    // number for those is not agreement but recall - what fraction of the thick band a much
    // cheaper walk actually finds.
    let mut rows = Vec::new();
    let mut got: Vec<(&str, Vec<Vec<f64>>)> = Vec::new();
    for (name, f) in [
        ("ray_tree3", &mut (|o, d| srt(tree3.raycast(o, d, 1500.0, ray_r).iter().map(|(t, _)| *t).collect())) as &mut dyn FnMut(Point3, Point3) -> Vec<f64>),
        ("ray_octree3", &mut |o, d| srt(oct3.raycast(o, d, 1500.0, ray_r).iter().map(|(t, _)| *t).collect())),
        ("ray_kdtree3", &mut |o, d| srt(kd3.raycast(o, d, 1500.0, ray_r).iter().map(|(t, _)| *t).collect())),
        ("ray_linear_octree3", &mut |o, d| srt(lin3.raycast(o, d, 1500.0, ray_r).iter().map(|(t, _)| *t).collect())),
        ("ray_tree3_dda", &mut |o, d| srt(tree3.raycast_dda(o, d, 1500.0, ray_r).hits.iter().map(|(t, _)| *t).collect())),
        ("ray_octree3_dda", &mut |o, d| srt(oct3.raycast_dda(o, d, 1500.0, ray_r).hits.iter().map(|(t, _)| *t).collect())),
        ("ray_morton3", &mut |o, d| srt(mor3.raycast(o, d, 1500.0, ray_r).hits.iter().map(|(t, _)| *t).collect())),
    ] {
        let (te, fo, all) = per_ray(f);
        rows.push(QRow { name: name.to_string(), tested: te, found: fo });
        got.push((name, all));
    }
    print_qrows(&format!("3D - work per ray (capsule radius {ray_r}, length 1500)"), "hits", &rows);
    let exact = &got[0].1;
    let bad: Vec<&str> = got[1..4].iter().filter(|(_, a)| !same_dists(exact, a)).map(|(n, _)| *n).collect();
    println!("  exact capsule (raycast): {}", if bad.is_empty() { "EXACT agreement".to_string() } else { format!("MISMATCH in {bad:?}") });
    for (name, all) in &got[4..] {
        let (sub, extra) = subset_recall(all, exact);
        println!("  {name:<20} subset of the exact answer: {} | recall {:.1}% of its hits for {:.0}% of its point tests",
            if extra == 0 { "yes".to_string() } else { format!("NO ({extra} hits it should not have)") },
            sub * 100.0,
            rows.iter().find(|r| r.name == *name).map(|r| r.tested).unwrap_or(0.0) / rows[0].tested * 100.0);
    }
    println!("  non-vacuous: {:.1} hits per ray on average", rows[0].found);

    println!("\nreading: 'found' must match across a block or a structure is answering a different");
    println!("question. 'waste' is the algorithmic quality of the index — points looked at per point");
    println!("returned. 'boxes' is what the descent costs. A structure can lose on time while winning");
    println!("here, and then the difference is the machine (cache, prefetch), not the algorithm.");
}
