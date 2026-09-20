//! **#180 — what is each structure's sweet spot, and do the objectives agree on it?**
//!
//! `examples/index_quality` compares the twelve at *matched granularity*, which is the only way
//! its geometric metrics (dead volume, margin) can be read at all — and the cost is that matching
//! pushed structures to settings nobody would ship (`Octree3` at `cap 48`, utilisation 0.56).
//! The user's challenge was exactly right: that table's knobs are ladder-search results, not
//! sweet spots. This is the third table, the one that was missing.
//!
//! It asks a different question, and the difference is the point. Not *"at equal resolution, whose
//! boxes are tighter"* but *"for THIS objective, where does THIS structure peak"* — and the
//! objectives are plural on purpose, because the user asked for "menor memoria, menor tiempo,
//! etc." and those are not the same knob. A leaf capacity that minimises query time splits the
//! tree hard, and every split is a node you pay for in bytes.
//!
//! ## What is measured
//!
//! Per (structure, knob): **build** µs, **cull** µs/query, **k-NN** µs/query, **bytes**, and two
//! composites — `build + NQ*cull` at a light and a heavy query load, which is what a caller who
//! rebuilds every frame actually pays. Plus the two **counts** (`classify_aabb` calls and
//! `position()` calls per cull), which are deterministic integers and therefore the only columns
//! that mean the same thing on another machine.
//!
//! ## What is held fixed, and what is not (MEASURING.md § 8i)
//!
//! Radius is an **axis**, not a constant. It has to be: `grid_min_hits` exists because the right
//! structure depends on how much a query finds, so the knob that wins a radius-4 cull has no
//! reason to win a radius-60 one. Distribution is an axis for the same reason. What IS fixed is
//! N and the query count, and both are reported.
//!
//! Two guards the sweep gets for free:
//!
//! * **A knob must not change the answer.** Every cull's result is compared against the ladder's
//!   first setting. A structure whose `item_limit` alters what it returns is broken, and this is
//!   the cheapest place that would ever show.
//! * **Builds clone outside the clock** (§ 8g), and the ladder is **rotated per repetition** so a
//!   machine that drifts during a run does not systematically favour one end of it (§ 8d).
//!
//! Run: `cargo run -p vectorial-hash --example sweet_spot --release`
//! Env: `SS_N` (default 50 000), `SS_REPS` (3), `SS_DIM` (`3`, `2`, or `both`).

use std::cell::Cell;
use std::mem::size_of;
use vectorial_hash::template::CellState;
use vectorial_hash::{
    Aabb, Circle, IPoint, IRect, IntegerTree, KdTree2, KdTree3, LinearOctree3, LinearQuadTree,
    MortonGrid, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, RadixTrie3,
    Rect, Shape, Shape3, Sphere3, Tree, Tree3,
};
use vectorial_hash::itree::IPositioned;

#[path = "common/mod.rs"]
mod common;

const W: f64 = 1000.0;
/// The `max_depth` given to the linear trees. It is a *safety stop* against coincident points
/// recursing forever, not the knob under test — so it is set generously and `capacity` is swept.
const LINEAR_MAX_DEPTH: u8 = 12;

// ---------------------------------------------------------------- items, with an optional counter

thread_local! {
    static POS: Cell<u64> = const { Cell::new(0) };
    static BOXES: Cell<u64> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}
fn take_counts() -> (u64, u64) { (BOXES.with(|c| c.replace(0)), POS.with(|c| c.replace(0))) }

/// One item type for both passes, with the counter behind a flag that is **off while the clock
/// runs**. Two separate types would be the obvious design and it is the wrong one: the timing
/// pass and the counting pass would then build different structures, and a difference between
/// them could be the item rather than the knob.
#[derive(Clone, Copy, PartialEq, Debug)]
struct P3 { id: u32, p: Point3 }
impl Positioned3 for P3 {
    fn position(&self) -> Point3 {
        if COUNTING.with(|c| c.get()) { POS.with(|c| c.set(c.get() + 1)); }
        self.p
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
struct P2 { id: u32, p: Point }
impl Positioned for P2 {
    fn position(&self) -> Point {
        if COUNTING.with(|c| c.get()) { POS.with(|c| c.set(c.get() + 1)); }
        self.p
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
struct PI { id: u32, p: IPoint }
impl IPositioned for PI {
    fn position(&self) -> IPoint {
        if COUNTING.with(|c| c.get()) { POS.with(|c| c.set(c.get() + 1)); }
        self.p
    }
}

/// Counts the descent, not only the leaf work — `cull` takes any shape, so no library change is
/// needed. Same trick as `tests/work_counts.rs`.
struct C3<S: Shape3> { inner: S }
impl<S: Shape3> Shape3 for C3<S> {
    fn bounding_box(&self) -> Aabb { self.inner.bounding_box() }
    fn contains_point(&self, p: Point3) -> bool { self.inner.contains_point(p) }
    fn classify_aabb(&self, b: &Aabb) -> CellState { BOXES.with(|c| c.set(c.get() + 1)); self.inner.classify_aabb(b) }
}
struct C2<S: Shape> { inner: S }
impl<S: Shape> Shape for C2<S> {
    fn bounding_box(&self) -> Rect { self.inner.bounding_box() }
    fn contains_point(&self, p: Point) -> bool { self.inner.contains_point(p) }
    fn classify_box(&self, b: &Rect) -> Option<CellState> { BOXES.with(|c| c.set(c.get() + 1)); self.inner.classify_box(b) }
}

// ---------------------------------------------------------------- workload

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

#[derive(Clone, Copy, PartialEq)]
enum Dist { Uniform, Clustered }
impl Dist {
    fn name(self) -> &'static str { match self { Dist::Uniform => "uniform", Dist::Clustered => "clustered" } }
}

fn points3(n: usize, d: Dist, seed: u64) -> Vec<P3> {
    let mut r = Lcg(seed);
    match d {
        Dist::Uniform => (0..n)
            .map(|i| P3 { id: i as u32, p: Point3::new(r.r(0.0, W - 0.1), r.r(0.0, W - 0.1), r.r(0.0, W - 0.1)) })
            .collect(),
        Dist::Clustered => {
            let blobs: Vec<(f64, f64, f64)> = (0..12).map(|_| (r.r(80.0, 920.0), r.r(80.0, 920.0), r.r(80.0, 920.0))).collect();
            (0..n)
                .map(|i| {
                    let b = blobs[(r.f() * blobs.len() as f64) as usize % blobs.len()];
                    P3 {
                        id: i as u32,
                        p: Point3::new(
                            (b.0 + r.r(-28.0, 28.0)).clamp(0.0, W - 0.1),
                            (b.1 + r.r(-28.0, 28.0)).clamp(0.0, W - 0.1),
                            (b.2 + r.r(-28.0, 28.0)).clamp(0.0, W - 0.1),
                        ),
                    }
                })
                .collect()
        }
    }
}

/// Query centres drawn FROM the points, so a query finds something. Centres in empty space
/// measure the descent and nothing else, and on clustered data most of the world is empty.
fn centres3(items: &[P3], nq: usize, seed: u64) -> Vec<Point3> {
    let mut r = Lcg(seed);
    (0..nq).map(|_| items[(r.f() * items.len() as f64) as usize % items.len()].p).collect()
}

// ---------------------------------------------------------------- one measured cell

/// Everything measured for one (structure, knob) pair. `hits` is carried so the ladder can be
/// checked for answer-identity, and `boxes`/`points` are the machine-independent columns.
#[derive(Clone)]
struct Cell3 {
    knob: usize,
    build_us: f64,
    cull_us: f64,
    knn_us: f64,
    bytes: usize,
    hits: usize,
    boxes: u64,
    points: u64,
}

/// Every knob within `TIE` of the best is **tied for best**, and the report is the band rather
/// than the argmin.
///
/// This is not politeness. The first version of this summary printed the argmin per objective and
/// announced "5 distinct knobs across six objectives" off readings like `cull` 0.43 / 0.44 / 0.44
/// us — a 2% spread on a machine whose single-sample noise is documented at up to 40%
/// (MEASURING.md 8e). It was reporting noise as a sweet spot. A band says something a re-run will
/// still agree with, and it is also the more useful answer: "anything from 48 to 128 is within 5%"
/// tells a caller they may pick on another axis, where "48 is optimal" tells them to chase a digit.
const TIE: f64 = 1.05;

fn band(cells: &[Cell3], f: impl Fn(&Cell3) -> f64) -> (f64, Vec<usize>) {
    let best = cells.iter().map(&f).fold(f64::INFINITY, f64::min);
    (best, cells.iter().filter(|c| f(c) <= best * TIE).map(|c| c.knob).collect())
}

fn show(knobs: &[usize]) -> String {
    if knobs.len() == 1 { return knobs[0].to_string(); }
    // Contiguous in the ladder is the common case and reads better as a range.
    let joined: Vec<String> = knobs.iter().map(|k| k.to_string()).collect();
    joined.join(",")
}

fn summarise(label: &str, cells: &[Cell3], nq_light: usize, nq_heavy: usize) {
    println!();
    println!("  {label}");
    println!("    {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}",
             "knob", "build us", "cull us", "knn us", "KB", "+N/10q ms", "+Nq ms", "boxes/q", "pts/q");
    for c in cells {
        let light = c.build_us + nq_light as f64 * c.cull_us;
        let heavy = c.build_us + nq_heavy as f64 * c.cull_us;
        println!("    {:>6}  {:>10.0}  {:>10.2}  {:>10.2}  {:>10.0}  {:>10.1}  {:>10.1}  {:>9.1}  {:>9.1}",
                 c.knob, c.build_us, c.cull_us, c.knn_us, c.bytes as f64 / 1024.0, light / 1e3, heavy / 1e3,
                 c.boxes as f64, c.points as f64);
    }

    type Objective<'a> = (&'a str, Box<dyn Fn(&Cell3) -> f64>);
    let objectives: [Objective; 6] = [
        ("build", Box::new(|c: &Cell3| c.build_us)),
        ("cull", Box::new(|c: &Cell3| c.cull_us)),
        ("knn", Box::new(|c: &Cell3| c.knn_us)),
        ("bytes", Box::new(|c: &Cell3| c.bytes as f64)),
        ("+N/10q", Box::new(move |c: &Cell3| c.build_us + nq_light as f64 * c.cull_us)),
        ("+Nq", Box::new(move |c: &Cell3| c.build_us + nq_heavy as f64 * c.cull_us)),
    ];
    let mut common: Option<std::collections::BTreeSet<usize>> = None;
    let mut parts: Vec<String> = Vec::new();
    for (name, f) in &objectives {
        let (_, knobs) = band(cells, f);
        parts.push(format!("{name} {}", show(&knobs)));
        let set: std::collections::BTreeSet<usize> = knobs.into_iter().collect();
        common = Some(match common { None => set, Some(prev) => prev.intersection(&set).copied().collect() });
    }
    let common = common.unwrap_or_default();
    println!("    within {:.0}%: {}", (TIE - 1.0) * 100.0, parts.join(" | "));
    if common.is_empty() {
        println!("    -> NO knob is within 5% on all six. The objectives genuinely disagree.");
    } else {
        let cs: Vec<String> = common.iter().map(|k| k.to_string()).collect();
        println!("    -> one setting serves all six: {}", cs.join(","));
    }

    // How much room the knob actually has on each axis. This is the arithmetic behind any
    // recommendation: if build varies 3x across the ladder and cull varies 1.3x, then a caller who
    // rebuilds every frame should pick for the build even when the queries dominate the total,
    // because the build is where the CHOICE has leverage.
    let spread = |f: &dyn Fn(&Cell3) -> f64| {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for c in cells { let v = f(c); lo = lo.min(v); hi = hi.max(v); }
        hi / lo
    };
    println!("    leverage: build {:.2}x across the ladder, cull {:.2}x, knn {:.2}x, bytes {:.2}x",
             spread(&|c| c.build_us), spread(&|c| c.cull_us), spread(&|c| c.knn_us), spread(&|c| c.bytes as f64));

    // The counts explain WHY an interior optimum can exist at all, and they are exact, so this
    // needs no repetitions. Direction is REPORTED, not assumed: a tree's ladder runs toward
    // coarser leaves while a grid's runs toward finer cells, so the same mechanism shows with
    // opposite signs. What matters is that the two costs move in OPPOSITE directions.
    let dir = |f: &dyn Fn(&Cell3) -> u64| -> &'static str {
        if cells.windows(2).all(|w| f(&w[1]) <= f(&w[0])) { "falls" }
        else if cells.windows(2).all(|w| f(&w[1]) >= f(&w[0])) { "rises" }
        else { "NOT monotone" }
    };
    let (bd, pd) = (dir(&|c| c.boxes), dir(&|c| c.points));
    let opposed = (bd == "falls" && pd == "rises") || (bd == "rises" && pd == "falls");
    println!("    counts: boxes/q {bd} ({} -> {}), pts/q {pd} ({} -> {}){}",
             cells[0].boxes, cells[cells.len() - 1].boxes, cells[0].points, cells[cells.len() - 1].points,
             if opposed { " — two monotone costs in OPPOSITE directions, which is the only way a middle can win" }
             else { " — not opposed, so no interior optimum is forced" });
}

fn main() {
    let n: usize = std::env::var("SS_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let reps: usize = std::env::var("SS_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let dim = std::env::var("SS_DIM").unwrap_or_else(|_| "3".into());
    let nq = 200usize;

    println!("#180 — each structure's sweet spot, per objective");
    println!("N = {n}, {nq} queries per measurement, min of {reps} reps, ladder order rotated per rep");
    println!("machine: {}", vectorial_hash::machine::machine_id());
    println!();
    println!("Radius is an AXIS, not a constant (MEASURING.md 8i): the knob that wins a small");
    println!("query has no reason to win a large one. Distribution likewise.");
    println!();
    println!("`boxes/q` and `pts/q` are exact integers and mean the same on any machine.");
    println!("The microsecond columns are this box, right now.");

    if dim == "3" || dim == "both" { run3(n, nq, reps); }
    if dim == "2" || dim == "both" { run2(n, nq, reps); }
}

/// Rotate a ladder by `k` so repetition `k` does not measure the same knob first.
fn rotated<T: Copy>(ladder: &[T], k: usize) -> Vec<T> {
    let m = ladder.len();
    (0..m).map(|i| ladder[(i + k) % m]).collect()
}

fn run3(n: usize, nq: usize, reps: usize) {
    // Query loads for the composites are N-PROPORTIONAL. A fixed 20/400 made them 99.9% build
    // at N=50k (build ~10 000 us against a 0.6 us cull), so both columns simply re-reported the
    // build's argmin and the composite measured nothing. A frame in which every item queries its
    // own neighbourhood is N culls; N/10 is the light case.
    let caps: Vec<usize> = vec![4, 8, 16, 24, 32, 48, 64, 96, 128];
    let levels: Vec<usize> = vec![3, 4, 5, 6, 7];
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);

    for dist in [Dist::Uniform, Dist::Clustered] {
        let items = points3(n, dist, 0x5EED_0100);
        let centres = centres3(&items, nq, 0x5EED_0200);
        for radius in [4.0f64, 60.0f64] {
            println!();
            println!("================================================================");
            println!(" 3D · {} · radius {radius}", dist.name());
            println!("================================================================");

            macro_rules! sweep {
                ($label:expr, $ladder:expr, $build:expr) => {{
                    let ladder = $ladder;
                    let mut acc: Vec<Option<Cell3>> = vec![None; ladder.len()];
                    for rep in 0..reps {
                        for knob in rotated(&ladder, rep) {
                            let slot = ladder.iter().position(|&k| k == knob).unwrap();
                            // Build: input cloned OUTSIDE the clock (MEASURING 8g).
                            let mut build_us = f64::INFINITY;
                            let mut idx = $build(items.clone(), knob);
                            for _ in 0..2 {
                                let input = items.clone();
                                let t = std::time::Instant::now();
                                idx = $build(input, knob);
                                build_us = build_us.min(t.elapsed().as_secs_f64() * 1e6);
                            }
                            let mut hits = 0usize;
                            let cull_us = common::wall_ms(2, || {
                                hits = 0;
                                for c in &centres { hits += idx.cull(&Sphere3::new(c.x, c.y, c.z, radius)).len(); }
                            }) * 1e3 / nq as f64;
                            let knn_us = common::wall_ms(2, || {
                                for c in &centres { std::hint::black_box(idx.knn(*c, 8).len()); }
                            }) * 1e3 / nq as f64;
                            // Counts: deterministic, so one pass, and NOT inside any clock.
                            let (boxes, points) = {
                                COUNTING.with(|c| c.set(true));
                                take_counts();
                                for c in &centres { std::hint::black_box(idx.cull(&C3 { inner: Sphere3::new(c.x, c.y, c.z, radius) }).len()); }
                                let got = take_counts();
                                COUNTING.with(|c| c.set(false));
                                got
                            };
                            let cell = Cell3 { knob, build_us, cull_us, knn_us, bytes: idx.bytes(), hits, boxes: boxes / nq as u64, points: points / nq as u64 };
                            acc[slot] = Some(match acc[slot].take() {
                                None => cell,
                                Some(p) => Cell3 {
                                    build_us: p.build_us.min(cell.build_us),
                                    cull_us: p.cull_us.min(cell.cull_us),
                                    knn_us: p.knn_us.min(cell.knn_us),
                                    ..cell
                                },
                            });
                        }
                    }
                    let cells: Vec<Cell3> = acc.into_iter().map(|c| c.unwrap()).collect();
                    // A knob must not change the answer.
                    for c in &cells {
                        assert_eq!(c.hits, cells[0].hits,
                            "{}: knob {} returned {} items where knob {} returned {} — a capacity/resolution \
                             setting must not change WHAT a query finds", $label, c.knob, c.hits, cells[0].knob, cells[0].hits);
                    }
                    summarise($label, &cells, n / 10, n);
                    cells
                }};
            }

            sweep!("Tree3 (item_limit)", caps.clone(), |v: Vec<P3>, k: usize| {
                let mut t: Tree3<P3> = Tree3::new(world, k);
                for it in v { t.insert(it); }
                t
            });
            sweep!("Octree3 (item_limit)", caps.clone(), |v: Vec<P3>, k: usize| {
                let mut t: Octree3<P3> = Octree3::new(world, k);
                for it in v { t.insert(it); }
                t
            });
            sweep!("KdTree3 (capacity)", caps.clone(), |v: Vec<P3>, k: usize| KdTree3::from_items(k, v));
            sweep!("LinearOctree3 (capacity)", caps.clone(), |v: Vec<P3>, k: usize| {
                LinearOctree3::from_items(world, k, LINEAR_MAX_DEPTH, v)
            });
            sweep!("MortonGrid3 (levels)", levels.clone(), |v: Vec<P3>, k: usize| {
                let mut g: MortonGrid3<P3> = MortonGrid3::new(world, k as u32);
                for it in v { g.insert(it); }
                g
            });
            sweep!("RadixTrie3 (bits)", levels.clone(), |v: Vec<P3>, k: usize| {
                RadixTrie3::from_items(world, k as u32, v)
            });
        }
    }
}

fn run2(n: usize, nq: usize, reps: usize) {
    let caps: Vec<usize> = vec![4, 8, 16, 24, 32, 48, 64, 96, 128];
    let levels: Vec<usize> = vec![3, 4, 5, 6, 7, 8];
    let world = Rect::new(0.0, 0.0, W, W);
    let iworld = IRect::new(0, 0, 1024, 1024);

    for dist in [Dist::Uniform, Dist::Clustered] {
        let items3 = points3(n, dist, 0x5EED_0100);
        let items: Vec<P2> = items3.iter().map(|it| P2 { id: it.id, p: Point::new(it.p.x, it.p.z) }).collect();
        let itemsi: Vec<PI> = items.iter().map(|it| PI { id: it.id, p: IPoint::new(it.p.x as i32, it.p.y as i32) }).collect();
        let centres: Vec<Point> = {
            let mut r = Lcg(0x5EED_0300);
            (0..nq).map(|_| items[(r.f() * items.len() as f64) as usize % items.len()].p).collect()
        };
        // 2D needs its OWN radii: a disc of radius 60 over a 1000-square holds an order of
        // magnitude more points than a sphere of radius 60 over a 1000-cube, so reusing the 3D
        // numbers would measure result-vector growth (the same trap the regression gate hit).
        for radius in [2.0f64, 20.0f64] {
            println!();
            println!("================================================================");
            println!(" 2D · {} · radius {radius}", dist.name());
            println!("================================================================");

            macro_rules! sweep2 {
                ($label:expr, $ladder:expr, $build:expr, $items:expr, $mk:expr, $kq:expr) => {{
                    let ladder = $ladder;
                    let src = $items;
                    let mut acc: Vec<Option<Cell3>> = vec![None; ladder.len()];
                    for rep in 0..reps {
                        for knob in rotated(&ladder, rep) {
                            let slot = ladder.iter().position(|&k| k == knob).unwrap();
                            let mut build_us = f64::INFINITY;
                            let mut idx = $build(src.clone(), knob);
                            for _ in 0..2 {
                                let input = src.clone();
                                let t = std::time::Instant::now();
                                idx = $build(input, knob);
                                build_us = build_us.min(t.elapsed().as_secs_f64() * 1e6);
                            }
                            let mut hits = 0usize;
                            let cull_us = common::wall_ms(2, || {
                                hits = 0;
                                for c in &centres { hits += idx.cull(&$mk(*c, radius)).len(); }
                            }) * 1e3 / nq as f64;
                            let knn_us = common::wall_ms(2, || {
                                for c in &centres { std::hint::black_box(idx.knn($kq(*c), 8).len()); }
                            }) * 1e3 / nq as f64;
                            let (boxes, points) = {
                                COUNTING.with(|c| c.set(true));
                                take_counts();
                                for c in &centres { std::hint::black_box(idx.cull(&C2 { inner: $mk(*c, radius) }).len()); }
                                let got = take_counts();
                                COUNTING.with(|c| c.set(false));
                                got
                            };
                            let cell = Cell3 { knob, build_us, cull_us, knn_us, bytes: idx.bytes(), hits, boxes: boxes / nq as u64, points: points / nq as u64 };
                            acc[slot] = Some(match acc[slot].take() {
                                None => cell,
                                Some(p) => Cell3 {
                                    build_us: p.build_us.min(cell.build_us),
                                    cull_us: p.cull_us.min(cell.cull_us),
                                    knn_us: p.knn_us.min(cell.knn_us),
                                    ..cell
                                },
                            });
                        }
                    }
                    let cells: Vec<Cell3> = acc.into_iter().map(|c| c.unwrap()).collect();
                    for c in &cells {
                        assert_eq!(c.hits, cells[0].hits,
                            "{}: knob {} returned {} items where knob {} returned {}", $label, c.knob, c.hits, cells[0].knob, cells[0].hits);
                    }
                    summarise($label, &cells, n / 10, n);
                }};
            }

            sweep2!("Tree (item_limit)", caps.clone(), |v: Vec<P2>, k: usize| {
                let mut t: Tree<P2> = Tree::new(world, k);
                for it in v { t.insert(it); }
                t
            }, &items, |c: Point, r: f64| Circle::new(c, r), |c: Point| c);
            sweep2!("QuadTree (item_limit)", caps.clone(), |v: Vec<P2>, k: usize| {
                let mut t: QuadTree<P2> = QuadTree::new(world, k);
                for it in v { t.insert(it); }
                t
            }, &items, |c: Point, r: f64| Circle::new(c, r), |c: Point| c);
            sweep2!("KdTree2 (capacity)", caps.clone(), |v: Vec<P2>, k: usize| KdTree2::from_items(k, v),
                    &items, |c: Point, r: f64| Circle::new(c, r), |c: Point| c);
            sweep2!("LinearQuadTree (capacity)", caps.clone(), |v: Vec<P2>, k: usize| {
                LinearQuadTree::from_items(world, k, LINEAR_MAX_DEPTH, v)
            }, &items, |c: Point, r: f64| Circle::new(c, r), |c: Point| c);
            sweep2!("MortonGrid (levels)", levels.clone(), |v: Vec<P2>, k: usize| {
                let mut g: MortonGrid<P2> = MortonGrid::new(world, k as u32);
                for it in v { g.insert(it); }
                g
            }, &items, |c: Point, r: f64| Circle::new(c, r), |c: Point| c);
            sweep2!("IntegerTree (item_limit)", caps.clone(), |v: Vec<PI>, k: usize| {
                let mut t: IntegerTree<PI> = IntegerTree::new(iworld, k);
                for it in v { t.insert(it); }
                t
            }, &itemsi, |c: Point, r: f64| Circle::new(c, r), |c: Point| IPoint::new(c.x as i32, c.y as i32));
        }
    }
    let _ = size_of::<P2>();
}
