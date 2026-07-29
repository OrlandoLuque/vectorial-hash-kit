//! **The one gate in this repo that can demand exact equality.**
//!
//! Every timed check has to carry a tolerance, because a number of milliseconds depends on
//! the machine, the cache, and whatever else had the CPU. So the timing gate
//! (`examples/regression_gate.rs`) passes anything within 25% — which is the right call for
//! a clock, and also means a traversal change that costs 15% more work sails straight
//! through it looking like noise.
//!
//! The *work* a traversal does is not a duration. It is a count of node boxes classified and
//! points tested, it is computed from deterministic arithmetic over a deterministic point
//! set, and it is the same integer on every machine. So this gate compares with `==`.
//!
//! If a number here changes, one of two things happened:
//!
//! 1. **A traversal changed** — a split rule, a pruning bound, a leaf capacity default, an
//!    iteration order. That is exactly what this test exists to make visible. Read the diff,
//!    decide whether it is an improvement, and bless the new numbers deliberately.
//! 2. **The platform's floating point differs** from x86-64 SSE2 (contraction, or a
//!    different rounding of the same expression). Nothing here relies on bit-exact floats by
//!    design; if this ever fires on a new architecture for that reason, the honest fix is a
//!    small tolerance on this test, not a chase for bit-equality.
//!
//! Bless new numbers with `VH_WORK_BLESS=1 cargo test -p vectorial-hash --test work_counts`,
//! which prints the table ready to paste.
//!
//! The narrative version of these counts — what they say about which structure to use — is
//! `examples/work_counters.rs` and `docs/MEASURING.md`. This file is only the ratchet.

use std::cell::Cell;
use vectorial_hash::template::CellState;
use vectorial_hash::{
    Aabb, Circle, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid, MortonGrid3,
    Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Shape, Shape3, Sphere3,
    Tree, Tree3,
};

// Deliberately smaller than the example's 200k: this runs in CI, in debug, on every push.
// The counts are just as exact at 20k, and the point is the ratchet, not the headline.
const N: usize = 20_000;
const NQ: usize = 60;
const R: f64 = 30.0;
const LEAF: usize = 16;
const K: usize = 8;
const W: f64 = 1000.0;

/// The blessed numbers. `(key, boxes, tested)` — see the module docs before editing.
const EXPECT: &[(&str, u64, u64)] = &[
    ("cull3/tree3", 11008, 25166),
    ("cull3/octree3", 18440, 21881),
    ("cull3/kdtree3", 7550, 18281),
    ("cull3/linear_octree3", 18440, 21881),
    ("cull3/morton3", 396, 93764),
    ("cull2/tree2", 3894, 4779),
    ("cull2/quadtree", 3656, 3893),
    ("cull2/kdtree2", 2848, 3884),
    ("cull2/linear_quadtree", 3656, 3893),
    ("cull2/morton2", 100, 87373),
    ("knn3/tree3", 0, 11179),
    ("knn3/octree3", 0, 11522),
    ("knn3/kdtree3", 0, 6824),
    ("knn3/linear_octree3", 0, 11522),
    ("knn3/morton3", 0, 179636),
    ("knn2/tree2", 0, 4281),
    ("knn2/quadtree", 0, 2818),
    ("knn2/kdtree2", 0, 3008),
    ("knn2/linear_quadtree", 0, 2818),
    ("knn2/morton2", 0, 173867),
];

thread_local! {
    static POS: Cell<u64> = const { Cell::new(0) };
    static BOX: Cell<u64> = const { Cell::new(0) };
}
fn take() -> (u64, u64) { (BOX.with(|c| c.replace(0)), POS.with(|c| c.replace(0))) }

#[derive(Clone, Copy)]
struct P3 { p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { POS.with(|c| c.set(c.get() + 1)); self.p } }
#[derive(Clone, Copy)]
struct P2 { p: Point }
impl Positioned for P2 { fn position(&self) -> Point { POS.with(|c| c.set(c.get() + 1)); self.p } }

/// Counts the descent as well as the leaf work. `cull` takes any shape, so this needs no
/// library change — the same trick `examples/work_counters.rs` uses.
struct C3<S: Shape3> { inner: S }
impl<S: Shape3> Shape3 for C3<S> {
    fn bounding_box(&self) -> Aabb { self.inner.bounding_box() }
    fn contains_point(&self, p: Point3) -> bool { self.inner.contains_point(p) }
    fn classify_aabb(&self, b: &Aabb) -> CellState { BOX.with(|c| c.set(c.get() + 1)); self.inner.classify_aabb(b) }
}
struct C2<S: Shape> { inner: S }
impl<S: Shape> Shape for C2<S> {
    fn bounding_box(&self) -> Rect { self.inner.bounding_box() }
    fn contains_point(&self, p: Point) -> bool { self.inner.contains_point(p) }
    fn classify_box(&self, b: &Rect) -> Option<CellState> { BOX.with(|c| c.set(c.get() + 1)); self.inner.classify_box(b) }
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

#[test]
fn traversal_work_is_exactly_what_it_was() {
    // Clustered, because that is where the structures actually differ — on uniform points
    // four of the five agree to within 20% and the gate would barely be watching anything.
    let mut r = Lcg(0x5EED_1234);
    let blobs: Vec<(f64, f64, f64)> = (0..6).map(|_| (r.r(100.0, 900.0), r.r(50.0, 250.0), r.r(100.0, 900.0))).collect();
    let items3: Vec<P3> = (0..N).map(|_| {
        let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
        P3 { p: Point3::new((b.0 + r.r(-14.0, 14.0)).clamp(0.0, W), (b.1 + r.r(-14.0, 14.0)).clamp(0.0, 300.0), (b.2 + r.r(-14.0, 14.0)).clamp(0.0, W)) }
    }).collect();
    let items2: Vec<P2> = items3.iter().map(|it| P2 { p: Point::new(it.p.x, it.p.z) }).collect();
    // Half the queries are aimed at a blob, half are uniform. All-uniform queries mostly
    // land in empty space: the first version of this test blessed `tested = 0` for four of
    // the five 3D structures, i.e. a ratchet holding nothing. Aimed queries straddle leaf
    // boundaries, which is where a traversal change would actually show.
    let q3: Vec<Point3> = (0..NQ).map(|i| {
        if i % 2 == 0 {
            let b = blobs[i / 2 % blobs.len()];
            Point3::new(b.0 + r.r(-20.0, 20.0), b.1 + r.r(-20.0, 20.0), b.2 + r.r(-20.0, 20.0))
        } else { Point3::new(r.r(0.0, W), r.r(0.0, 300.0), r.r(0.0, W)) }
    }).collect();
    let q2: Vec<Point> = q3.iter().map(|p| Point::new(p.x, p.z)).collect();
    let world3 = Aabb::new(0.0, 0.0, 0.0, W, 300.0, W);
    let world2 = Rect::new(0.0, 0.0, W, W);

    let tree3 = Tree3::bulk_load(world3, LEAF, items3.clone());
    let oct3 = Octree3::bulk_load(world3, LEAF, items3.clone());
    let kd3 = KdTree3::from_items(LEAF, items3.clone());
    let lin3 = LinearOctree3::from_items(world3, LEAF, 12, items3.clone());
    let mor3 = { let mut g = MortonGrid3::new(world3, MortonGrid3::<P3>::levels_for_cell_size(world3, R)); for it in &items3 { g.insert(*it); } g };
    let tree2 = { let mut t = Tree::new(world2, LEAF); for it in &items2 { t.insert(*it); } t };
    let quad = { let mut t = QuadTree::new(world2, LEAF); for it in &items2 { t.insert(*it); } t };
    let kd2 = KdTree2::from_items(LEAF, items2.clone());
    let lin2 = LinearQuadTree::from_items(world2, LEAF, 14, items2.clone());
    let mor2 = { let mut g = MortonGrid::new(world2, MortonGrid::<P2>::levels_for_cell_size(world2, R)); for it in &items2 { g.insert(*it); } g };

    let mut got: Vec<(String, u64, u64)> = Vec::new();
    // `found` is collected per structure and cross-checked: a gate on how much work a
    // traversal does is worthless if the traversals are not answering the same question.
    let mut found3: Vec<(&str, usize)> = Vec::new();
    let mut cull3 = |name: &str, f: &mut dyn FnMut(&C3<Sphere3>) -> usize| {
        let (mut bx, mut te, mut fo) = (0, 0, 0);
        for q in &q3 { take(); fo += f(&C3 { inner: Sphere3::new(q.x, q.y, q.z, R) }); let (b, t) = take(); bx += b; te += t; }
        got.push((format!("cull3/{name}"), bx, te));
        found3.push((Box::leak(name.to_string().into_boxed_str()), fo));
    };
    cull3("tree3", &mut |q| tree3.cull(q).len());
    cull3("octree3", &mut |q| oct3.cull(q).len());
    cull3("kdtree3", &mut |q| kd3.cull(q).len());
    cull3("linear_octree3", &mut |q| lin3.cull(q).len());
    cull3("morton3", &mut |q| mor3.cull(q).len());
    for (n, f) in &found3 { assert_eq!(*f, found3[0].1, "{n} culled a different number of items than tree3 — the gate would be comparing different questions"); }
    // Non-vacuous, asserted rather than assumed: a gate over queries that find nothing
    // passes for ever and guards nothing.
    assert!(found3[0].1 > NQ * 20, "3D culls found only {} items over {NQ} queries", found3[0].1);

    let mut found2: Vec<(&str, usize)> = Vec::new();
    let mut cull2 = |name: &str, f: &mut dyn FnMut(&C2<Circle>) -> usize| {
        let (mut bx, mut te, mut fo) = (0, 0, 0);
        for q in &q2 { take(); fo += f(&C2 { inner: Circle::new(*q, R) }); let (b, t) = take(); bx += b; te += t; }
        got.push((format!("cull2/{name}"), bx, te));
        found2.push((Box::leak(name.to_string().into_boxed_str()), fo));
    };
    cull2("tree2", &mut |q| tree2.cull(q).len());
    cull2("quadtree", &mut |q| quad.cull(q).len());
    cull2("kdtree2", &mut |q| kd2.cull(q).len());
    cull2("linear_quadtree", &mut |q| lin2.cull(q).len());
    cull2("morton2", &mut |q| mor2.cull(q).len());
    for (n, f) in &found2 { assert_eq!(*f, found2[0].1, "{n} culled a different number of items than tree2"); }
    assert!(found2[0].1 > NQ * 20, "2D culls found only {} items over {NQ} queries", found2[0].1);

    // k-NN has no query volume to classify, so only the leaf column moves. Distances are
    // cross-checked the same way the cull counts are.
    let mut ref3: Option<Vec<f64>> = None;
    let mut knn3 = |name: &str, f: &mut dyn FnMut(Point3) -> Vec<f64>| {
        let (mut te, mut all) = (0, Vec::new());
        for q in &q3 { take(); let d = f(*q); te += take().1; all.extend(d); }
        match &ref3 {
            None => ref3 = Some(all),
            Some(rf) => assert!(rf.len() == all.len() && rf.iter().zip(&all).all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b.abs())),
                "{name} returned different neighbours than tree3"),
        }
        got.push((format!("knn3/{name}"), 0, te));
    };
    knn3("tree3", &mut |q| tree3.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn3("octree3", &mut |q| oct3.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn3("kdtree3", &mut |q| kd3.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn3("linear_octree3", &mut |q| lin3.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn3("morton3", &mut |q| mor3.knn(q, K).iter().map(|(d, _)| *d).collect());

    let mut ref2: Option<Vec<f64>> = None;
    let mut knn2 = |name: &str, f: &mut dyn FnMut(Point) -> Vec<f64>| {
        let (mut te, mut all) = (0, Vec::new());
        for q in &q2 { take(); let d = f(*q); te += take().1; all.extend(d); }
        match &ref2 {
            None => ref2 = Some(all),
            Some(rf) => assert!(rf.len() == all.len() && rf.iter().zip(&all).all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b.abs())),
                "{name} returned different neighbours than tree2"),
        }
        got.push((format!("knn2/{name}"), 0, te));
    };
    knn2("tree2", &mut |q| tree2.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn2("quadtree", &mut |q| quad.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn2("kdtree2", &mut |q| kd2.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn2("linear_quadtree", &mut |q| lin2.knn(q, K).iter().map(|(d, _)| *d).collect());
    knn2("morton2", &mut |q| mor2.knn(q, K).iter().map(|(d, _)| *d).collect());

    if std::env::var("VH_WORK_BLESS").is_ok() {
        println!("\nconst EXPECT: &[(&str, u64, u64)] = &[");
        for (k, b, t) in &got { println!("    (\"{k}\", {b}, {t}),"); }
        println!("];");
        panic!("VH_WORK_BLESS set — numbers printed above, nothing was checked");
    }

    // A missing key is a failure, not a skip: silently not checking a structure is the
    // failure mode a ratchet has.
    let mut bad = Vec::new();
    assert_eq!(got.len(), EXPECT.len(), "structure count changed — bless the table");
    for (k, b, t) in &got {
        let Some(&(_, eb, et)) = EXPECT.iter().find(|(ek, _, _)| ek == k) else {
            bad.push(format!("{k}: not in the blessed table")); continue;
        };
        if *b != eb || *t != et { bad.push(format!("{k}: boxes {eb} -> {b}, tested {et} -> {t}")); }
    }
    assert!(bad.is_empty(), "traversal work changed:\n  {}\n\nsee this file's docs: a change here is a real algorithmic change, \
        not noise. Bless with VH_WORK_BLESS=1 once you have decided it is an improvement.", bad.join("\n  "));
}
