//! **Structural stability under motion** — the axis the query benches never measure.
//!
//! `radix_trie_bench` asked which structure answers a sphere query fastest over a *static* point
//! set in memory on one machine, and `Octree3` won by 3.6–5.2×. That is a real answer to a narrow
//! question, and it says nothing about the property that separates these two shapes in a world
//! where things move:
//!
//! - A **pointer tree with an item limit** changes its own shape as items move. A leaf that grows
//!   past the limit **splits**; children that between them fall back under it **merge**.
//! - A **fixed-resolution key** cannot. An item either stays in its cell or is re-keyed into
//!   another one, and the grid you end up with is bit-for-bit the grid you would have got had
//!   every item started where it now is.
//!
//! ## ★ The hypothesis this was written to test, and how it came out
//!
//! I expected the second bullet to be an *advantage*: that a maintained tree, unlike a grid, ends
//! up a function of its whole **history** and therefore drifts away from the tree those same
//! points would have built from scratch. **It does not, and the drift is exactly zero** — section
//! B reads `kept == fresh` as integer equality for all three arms at every churn level, and the
//! bench now asserts it.
//!
//! The reason is worth stating because it is not obvious: **these three** split **positionally**
//! (fixed octant midpoints, not a data-dependent median), and `merge_limit == item_limit`, so a
//! node is subdivided *iff* it holds more than the limit — the same predicate a fresh build
//! evaluates. Split and merge share one threshold, so there is no hysteresis band for history to
//! hide in. A structure whose split point depended on the data (a k-d tree's median) could not do
//! this, which is the same reason those two cannot maintain at all.
//!
//! **"These three" and not "the kit's trees", which is what this file first said.**
//! `tests/shape_is_history_free.rs` sweeps twelve seeds across all nine maintainable structures
//! and finds **seven** exactly history-free and **two** that are not: `Tree` and `IntegerTree` are
//! binary, and for a **square** node they choose the split *axis* by counting which way
//! distributes the items more evenly. That count is taken on whatever the node held at the moment
//! it split, so it is a data-dependent decision after all — a small one (worst 1.014x / 1.034x)
//! and a real one. `Tree3` is binary too and is exempt only because it splits the longest axis
//! with a `>=` tie-break, pure geometry. One arm of a policy being geometric is not the same as
//! the policy being geometric.
//!
//! So the difference between a keyed grid and an adaptive tree here is **not shape, it is work**:
//! the trees pay 0.10-0.15 splits-plus-merges per boundary crossing and the grid pays exactly 0.
//! That is what section A measures, and it is where the maintenance time goes.
//!
//! The headline is deliberately a **count**, not a clock: splits and merges per frame come from
//! the `struct-stats` feature and are exactly reproducible, whereas this machine's timings are
//! episodic (`docs/MEASURING.md` § 8e). The clock is here to price what the counts explain, not to
//! carry the conclusion.
//!
//! Three arms, all **keeping** (none rebuilds), all fed the identical deterministic motion:
//!
//! | arm | key | restructures? |
//! | --- | --- | --- |
//! | `Octree3` + `update_ref` | pointer + `ItemRef` | yes — splits and merges |
//! | `LinearOctree3` + `update` | Morton path **+ level** | yes — same, over a hash |
//! | `MortonGrid3` + `update` | Morton at a **fixed** level | **never** |
//!
//! ## Two things this file is careful about
//!
//! **The motion is generated outside the clock.** Choosing who moves and where costs an RNG draw
//! per item per frame, which at 50 000 × 300 is 15 M draws — comfortably more than the work being
//! measured at low churn. `IntegerTree::bulk_load_par` was measured *backwards* by leaving a clone
//! inside the timed closure (`MEASURING.md` § 8g); this builds the frame's move list first and
//! times only its application.
//!
//! **Nothing may leave the world.** New positions are clamped inside the box with a margin, so no
//! arm can silently drop an item. An index only knows what it holds, and an arm that quietly loses
//! points is answering a different question — which is exactly how `stealth_wgpu`'s index and scan
//! came to disagree on 77 % of frames.
//!
//! ```bash
//! cargo run -p vectorial-hash --example restructure_churn --release --features struct-stats
//! ```
//! Without the feature the split/merge columns are absent (and the arms carry no counter at all —
//! run it both ways to confirm the instrumentation is free).
//!
//! Env: `RC_N` (population), `RC_FRAMES`, `RC_STEP` (displacement per move, world units).

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;
use vectorial_hash::linear_octree3::LinearOctree3;
use vectorial_hash::morton3::Crossed;
use vectorial_hash::{Aabb, MortonGrid3, OCrossing, Octree3, Point3, Positioned3, Sphere3};

const W: f64 = 1000.0;
/// Leaf capacity, shared by the two adaptive arms so their splits are comparable.
const CAP: usize = 16;
/// `levels 4` = 4 096 cells, ~12 items each at the default population — inside the occupancy band
/// `MortonGrid3` documents (aim for roughly the `k` you ask k-NN for). Reported, not assumed.
const LEVELS: u32 = 4;
/// Max depth for `LinearOctree3`, matching `grid_keep_bench`'s configuration so this bench's
/// drift column can be compared against the one already published in `CHOOSING.md`.
const MAXD: u8 = 12;

#[derive(Clone, Copy)]
struct M {
    id: u32,
    p: Point3,
}
impl Positioned3 for M {
    fn position(&self) -> Point3 { self.p }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// The starting layout, identical for every arm.
fn initial(n: usize) -> Vec<Point3> {
    let mut r = Rng(0x9E3779B97F4A7C15);
    (0..n).map(|_| Point3 { x: r.f() * W, y: r.f() * W, z: r.f() * W }).collect()
}

fn clamp(v: f64) -> f64 { v.clamp(1.0, W - 1.0) }

/// One frame of motion, **generated outside the clock**. Each item moves with probability
/// `churn`, independently — deliberately not a strided or prefix subset. A fixed prefix would
/// make "30 % churn" mean "70 % of the population are statues" and would flatter the measurement
/// with locality a real 30 % does not have; a stride can alternate its membership every frame
/// (`MEASURING.md` § 11). Both were real defects in `adaptive_lab`.
fn frame_moves(r: &mut Rng, pos: &mut [Point3], churn: f64, step: f64, out: &mut Vec<(usize, Point3)>) {
    out.clear();
    for (i, p) in pos.iter_mut().enumerate() {
        if r.f() >= churn { continue; }
        let dx = (r.f() - 0.5) * 2.0 * step;
        let dy = (r.f() - 0.5) * 2.0 * step;
        let dz = (r.f() - 0.5) * 2.0 * step;
        *p = Point3 { x: clamp(p.x + dx), y: clamp(p.y + dy), z: clamp(p.z + dz) };
        out.push((i, *p));
    }
}

#[derive(Default)]
struct Row {
    maintain_ms: f64,
    moved: u64,
    stayed: u64,
    splits: u64,
    merges: u64,
    /// Leaves (trees) or non-empty cells (grid) after the run.
    shape: usize,
    /// The same, for a structure built fresh from the FINAL positions.
    fresh_shape: usize,
    cull_us: f64,
    fresh_cull_us: f64,
}

#[cfg(feature = "struct-stats")]
fn reset_counts() { let _ = vectorial_hash::restructure::reset(); }
#[cfg(feature = "struct-stats")]
fn read_counts() -> (u64, u64) { vectorial_hash::restructure::counts() }
#[cfg(not(feature = "struct-stats"))]
fn reset_counts() {}
#[cfg(not(feature = "struct-stats"))]
fn read_counts() -> (u64, u64) { (0, 0) }

fn world() -> Aabb { Aabb { x: 0.0, y: 0.0, z: 0.0, w: W, h: W, d: W } }

/// Query probes, the same set for every arm and every churn level.
fn probes(k: usize) -> Vec<Point3> {
    let mut r = Rng(0xDEADBEEFCAFEF00D);
    (0..k).map(|_| Point3 { x: r.f() * W, y: r.f() * W, z: r.f() * W }).collect()
}
const RADIUS: f64 = 30.0;

fn arm_octree(n: usize, frames: usize, churn: f64, step: f64) -> Row {
    let mut pos = initial(n);
    let mut r = Rng(0x2545F4914F6CDD1D);
    let mut t = Octree3::new(world(), CAP);
    let refs: Vec<_> = (0..n)
        .map(|i| t.insert_ref(M { id: i as u32, p: pos[i] }).expect("starts inside the world"))
        .collect();

    let mut row = Row::default();
    let mut moves = Vec::new();
    reset_counts();
    for _ in 0..frames {
        frame_moves(&mut r, &mut pos, churn, step, &mut moves);
        let t0 = Instant::now();
        for &(i, np) in &moves {
            match t.update_ref_tracked(refs[i], |m| m.p = np) {
                OCrossing::Stayed(_) => row.stayed += 1,
                OCrossing::Moved { .. } => row.moved += 1,
                OCrossing::Left => unreachable!("clamped positions never leave the world"),
            }
        }
        row.maintain_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    let (s, m) = read_counts();
    (row.splits, row.merges) = (s, m);

    let mut fresh = Octree3::new(world(), CAP);
    for (i, p) in pos.iter().enumerate() { fresh.insert_ref(M { id: i as u32, p: *p }); }
    row.shape = t.leaf_count();
    row.fresh_shape = fresh.leaf_count();
    assert_shape_is_history_free("Octree3", &row);

    let qs = probes(200);
    let (a, b, _, _) = common::compare2(
        5,
        || for q in &qs { std::hint::black_box(t.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
        || for q in &qs { std::hint::black_box(fresh.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
    );
    let hz = common::rate();
    row.cull_us = a / hz * 1e6 / qs.len() as f64;
    row.fresh_cull_us = b / hz * 1e6 / qs.len() as f64;
    row
}

/// The finding, as an assertion rather than a sentence — for **every** arm, not just the grid.
///
/// A maintained structure must end up the shape a rebuild from its own current contents would
/// have produced. For the grid that is a truism about fixed-resolution keys. For the two adaptive
/// arms it is the result this bench exists to report, and it held for every row measured, so it
/// is pinned here: if a future change introduces a hysteresis band between splitting and merging,
/// or a data-dependent split, this fires instead of the table quietly reading 1.03x and nobody
/// noticing.
fn assert_shape_is_history_free(arm: &str, row: &Row) {
    assert_eq!(
        row.shape, row.fresh_shape,
        "{arm}: a maintained structure must have the SAME shape as a rebuild from its own current \
         contents — kept {} vs fresh {}", row.shape, row.fresh_shape,
    );
}

fn arm_linear(n: usize, frames: usize, churn: f64, step: f64) -> Row {
    let mut pos = initial(n);
    let mut r = Rng(0x2545F4914F6CDD1D);
    let items: Vec<M> = pos.iter().enumerate().map(|(i, p)| M { id: i as u32, p: *p }).collect();
    let mut t = LinearOctree3::from_items(world(), CAP, MAXD,items);

    let mut row = Row::default();
    let mut moves = Vec::new();
    let mut old = pos.clone();
    reset_counts();
    for _ in 0..frames {
        frame_moves(&mut r, &mut pos, churn, step, &mut moves);
        let t0 = Instant::now();
        for &(i, np) in &moves {
            let id = i as u32;
            match t.update(old[i], |m: &M| m.id == id, |m: &mut M| m.p = np) {
                Crossed::Stayed => row.stayed += 1,
                Crossed::Moved => row.moved += 1,
                c => unreachable!("clamped positions never leave the world, and the id is there: {c:?}"),
            }
        }
        row.maintain_ms += t0.elapsed().as_secs_f64() * 1000.0;
        for &(i, np) in &moves { old[i] = np; }
    }
    let (s, m) = read_counts();
    (row.splits, row.merges) = (s, m);

    let fresh_items: Vec<M> = pos.iter().enumerate().map(|(i, p)| M { id: i as u32, p: *p }).collect();
    let fresh = LinearOctree3::from_items(world(), CAP, MAXD,fresh_items);
    row.shape = t.leaf_count();
    row.fresh_shape = fresh.leaf_count();
    assert_shape_is_history_free("LinearOctree3", &row);

    let qs = probes(200);
    let (a, b, _, _) = common::compare2(
        5,
        || for q in &qs { std::hint::black_box(t.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
        || for q in &qs { std::hint::black_box(fresh.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
    );
    let hz = common::rate();
    row.cull_us = a / hz * 1e6 / qs.len() as f64;
    row.fresh_cull_us = b / hz * 1e6 / qs.len() as f64;
    row
}

fn arm_grid(n: usize, frames: usize, churn: f64, step: f64) -> Row {
    let mut pos = initial(n);
    let mut r = Rng(0x2545F4914F6CDD1D);
    let mut g = MortonGrid3::new(world(), LEVELS);
    for (i, p) in pos.iter().enumerate() { g.insert(M { id: i as u32, p: *p }); }

    let mut row = Row::default();
    let mut moves = Vec::new();
    let mut old = pos.clone();
    reset_counts();
    for _ in 0..frames {
        frame_moves(&mut r, &mut pos, churn, step, &mut moves);
        let t0 = Instant::now();
        for &(i, np) in &moves {
            let id = i as u32;
            match g.update(old[i], |m: &M| m.id == id, |m: &mut M| m.p = np) {
                Crossed::Stayed => row.stayed += 1,
                Crossed::Moved => row.moved += 1,
                c => unreachable!("clamped positions never leave the world, and the id is there: {c:?}"),
            }
        }
        row.maintain_ms += t0.elapsed().as_secs_f64() * 1000.0;
        for &(i, np) in &moves { old[i] = np; }
    }
    let (s, m) = read_counts();
    (row.splits, row.merges) = (s, m);

    let mut fresh = MortonGrid3::new(world(), LEVELS);
    for (i, p) in pos.iter().enumerate() { fresh.insert(M { id: i as u32, p: *p }); }
    let (oc, of) = (g.occupancy(), fresh.occupancy());
    row.shape = oc.cells;
    row.fresh_shape = of.cells;

    // The grid gets the stronger form of the check: not just the same number of cells, but the
    // same population and the same fullest cell. A fixed-resolution key has nothing to drift, so
    // there is no reason to settle for a proxy here.
    assert_eq!(
        (oc.cells, oc.items, oc.max), (of.cells, of.items, of.max),
        "a maintained grid must equal a fresh one: kept {oc:?} vs fresh {of:?}",
    );
    assert_shape_is_history_free("MortonGrid3", &row);

    let qs = probes(200);
    let (a, b, _, _) = common::compare2(
        5,
        || for q in &qs { std::hint::black_box(g.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
        || for q in &qs { std::hint::black_box(fresh.cull(&Sphere3::new(q.x, q.y, q.z, RADIUS)).len()); },
    );
    let hz = common::rate();
    row.cull_us = a / hz * 1e6 / qs.len() as f64;
    row.fresh_cull_us = b / hz * 1e6 / qs.len() as f64;
    row
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

fn main() {
    let n: usize = env("RC_N", 50_000);
    // 300 frames and a +-40 wu step are `grid_keep_bench`'s configuration, and they are the
    // defaults here for a reason worth stating: the FIRST run of this bench used 60 frames and
    // a +-12 wu step, and read the drift column as 1.00x for every arm at every churn level.
    // That is not a finding, it is `MEASURING.md` § 8c happening again — a short window measures
    // a RATE perfectly well and an ACCUMULATION not at all, and a small step barely crosses a
    // leaf boundary so there is nothing to accumulate. Shorten these at your own risk.
    let frames: usize = env("RC_FRAMES", 300);
    let step: f64 = env("RC_STEP", 40.0);

    println!("restructure_churn — {} items, {frames} frames, step {step} wu (cell {:.1} wu, leaf cap {CAP})",
             n, W / (1u32 << LEVELS) as f64);
    println!("{}", vectorial_hash::machine_line());
    if cfg!(feature = "struct-stats") {
        println!("struct-stats: ON (split/merge columns live; timings carry the counter's cost)");
    } else {
        println!("struct-stats: off — rerun with `--features struct-stats` for the split/merge columns");
    }
    println!();

    let churns = [0.001_f64, 0.01, 0.1, 0.5, 1.0];

    // Every arm runs ONCE, and both tables read the same row. The first version of this file
    // called each arm twice, once per table — so the two halves of one printed line could come
    // from different runs of the same experiment, which is a reader's problem long before it is
    // a correctness one.
    let mut rows: Vec<(f64, &str, Row)> = Vec::new();
    for &c in &churns {
        rows.push((c, "Octree3", arm_octree(n, frames, c, step)));
        rows.push((c, "LinearOctree3", arm_linear(n, frames, c, step)));
        rows.push((c, "MortonGrid3", arm_grid(n, frames, c, step)));
    }
    let f = frames as f64;

    println!("A) Restructuring: what each shape does when the same items move");
    println!("{:<14} {:>7} {:>9} {:>9} {:>9} {:>9} {:>11} {:>10}",
             "arm", "churn", "moved/f", "xing%", "splits/f", "merges/f", "restr/xing", "maint us/f");
    let mut last = f64::NAN;
    for (c, name, row) in &rows {
        if *c != last && !last.is_nan() { println!(); }
        last = *c;
        let restr = (row.splits + row.merges) as f64;
        let touched = (row.moved + row.stayed) as f64;
        // With the feature off the counters are not compiled, so 0.00 would be indistinguishable
        // from the grid's genuine zero — the one number this table exists to show. Print `-`.
        let (sp, mg, px) = if cfg!(feature = "struct-stats") {
            (format!("{:.2}", row.splits as f64 / f), format!("{:.2}", row.merges as f64 / f),
             format!("{:.4}", if row.moved > 0 { restr / row.moved as f64 } else { 0.0 }))
        } else {
            ("-".to_string(), "-".to_string(), "-".to_string())
        };
        println!("{name:<14} {:>6.1}% {:>9.0} {:>8.0}% {sp:>9} {mg:>9} {px:>11} {:>10.1}",
                 c * 100.0, row.moved as f64 / f,
                 if touched > 0.0 { row.moved as f64 / touched * 100.0 } else { 0.0 },
                 row.maintain_ms / f * 1000.0);
    }

    println!();
    println!("B) Shape drift, and what it costs the QUERY (after {frames} maintained frames)");
    println!("{:<14} {:>7} {:>10} {:>10} {:>8} {:>10} {:>10} {:>7}",
             "arm", "churn", "kept", "fresh", "drift", "cull us", "fresh us", "cost");
    last = f64::NAN;
    for (c, name, row) in &rows {
        if *c != last && !last.is_nan() { println!(); }
        last = *c;
        println!("{name:<14} {:>6.1}% {:>10} {:>10} {:>7.2}x {:>10.2} {:>10.2} {:>6.2}x",
                 c * 100.0, row.shape, row.fresh_shape,
                 row.shape as f64 / row.fresh_shape as f64,
                 row.cull_us, row.fresh_cull_us, row.cull_us / row.fresh_cull_us);
    }
    println!();

    println!("`kept` vs `fresh` is the shape after maintenance against the shape the same points");
    println!("would have produced from scratch. ALL THREE are asserted equal, which is the result:");
    println!("this bench was written expecting the two adaptive arms to drift and they do not.");
    println!("These three split positionally and use ONE threshold for both splitting and merging,");
    println!("so \"subdivided iff it holds more than the limit\" is a property of the current points,");
    println!("not of the history. A data-dependent split (a k-d median) could not do this — which");
    println!("is the same reason those two structures cannot maintain at all.");
    println!();
    println!("It does NOT generalise to all nine maintainable structures, which is what this file");
    println!("first claimed. tests/shape_is_history_free.rs sweeps 12 seeds over all of them: seven");
    println!("are exactly history-free, and Tree and IntegerTree are not — being binary, they pick a");
    println!("SQUARE node's split axis by counting which way distributes the items better, which is");
    println!("data-dependent after all (worst 1.014x / 1.034x).");
    println!();
    println!("The difference is therefore not SHAPE but WORK: the `restr/xing` column in A. The");
    println!("grid reads exactly 0.0000 because it has no shape to change; the trees pay 0.10-0.15");
    println!("splits-plus-merges for every boundary a moving item crosses, and that is where their");
    println!("maintenance time goes. `cost` prices the shapes against each other and finds nothing,");
    println!("as it now must.");
}
