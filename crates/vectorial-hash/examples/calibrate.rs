//! `calibrate` — measure THIS machine and write the thresholds the adaptive index uses.
//!
//! The defaults in [`Thresholds`] are this repo's measurements on one box. A different
//! cache hierarchy moves them, and it moves them exactly where it matters: the binary
//! tree beats the quadtree by 4-10% in 2D, a margin another CPU could plausibly flip.
//! Rather than guess, run this on the target machine and ship the file it writes.
//!
//! ```bash
//! cargo run -p vectorial-hash --example calibrate --release            # prints + writes
//! cargo run -p vectorial-hash --example calibrate --release -- out.txt # to a path
//! VH_CALIBRATION=out.txt ./my_game                                     # and use it
//! ```
//!
//! What it actually measures (the rest of the thresholds are policy, not hardware, and
//! are carried through unchanged):
//!
//! - **`brute_max`** — the population at which a linear scan stops beating an indexed
//!   cull. Found by bisection on the real structures, not assumed.
//! - **`high_churn`** — the fraction of relocations at which rebuilding a grid becomes
//!   cheaper than fixing the kept tree in place. Swept against QUERIES PER FRAME too,
//!   because that is what really decides it: a rebuild pays a big fixed cost and then
//!   answers from a perfectly fitted structure, so the more you query the sooner it wins.
//!   With a handful of culls a frame the kept tree wins at every churn level — writing
//!   `high_churn = 1` is a real answer meaning "never switch on churn alone here".
//!
//!   That was measured when a grid could only be refilled. Since `MortonGrid3::update`, the
//!   sweep runs **three** arms and churn matters again: the kept grid takes the whole
//!   zero-churn row and most of the heavy-query column, while a pure rebuild is left with the
//!   single corner at full churn. The frontier used to be vertical in query load; it is
//!   diagonal now.
//!
//! Both are measured with min-of-N on an otherwise quiet machine; run it on an idle box
//! or the numbers it writes will be pessimistic in ways the policy then bakes in.

use std::time::Instant;
use vectorial_hash::KdTree3;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3, Thresholds, Tree3};

#[path = "common/mod.rs"]
mod common;

#[derive(Clone, Copy)]
struct P { id: u32, p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

const W: f64 = 512.0;
fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, W, W, W) }

fn cloud(n: usize, seed: u64) -> Vec<P> {
    let mut r = Lcg(seed);
    (0..n).map(|i| P { id: i as u32, p: Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W)) }).collect()
}

/// Is an index worth it at this population? — the probe behind `brute_max`.
///
/// **Rewritten 2026-08-03, because it was answering a different question than the threshold
/// asks.** It used to time culls alone against a pre-built tree: no build, no maintenance, and
/// 64 queries whatever the population. That is the regime most favourable to an index, and it
/// is not what `brute_max` governs — that is an *unconditional floor*, consulted before the
/// load-aware `scan_budget` rule, so it must be set by the case least favourable to a scan and
/// it must charge the index for existing.
///
/// So a frame now looks like a frame: a quarter of the points move (the index pays `update_ref`,
/// the scan pays nothing to maintain), then `n` culls run — the heaviest load in
/// `examples/brute_edge`, which is where an index would otherwise have been chosen and this
/// floor would have overridden it.
fn index_beats_scan(n: usize, radius: f64) -> bool {
    #![allow(clippy::needless_range_loop)]
    let items = cloud(n, 0xC0FFEE + n as u64);
    let mut r = Lcg(7);
    let qs: Vec<Point3> = (0..n.max(1)).map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();
    let moves: Vec<Point3> = (0..n / 4).map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();

    // Built ONCE, outside the timed region. A first draft built it inside, which quietly turned
    // the probe into "rebuild the index every frame" — the harshest possible reading — and moved
    // the answer from ~130 to ~940. The policy's index is *kept*, so the probe keeps it too.
    let mut tree = Tree3::new(world(), 8);
    let refs: Vec<_> = items.iter().map(|it| tree.insert_ref(*it).expect("inside the world")).collect();
    let indexed = common::measure(5, || {
        for (i, np) in moves.iter().enumerate() { tree.update_ref(refs[i], |it| it.p = *np); }
        let mut acc = 0usize;
        for q in &qs { acc += tree.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;
    let scanned = common::measure(5, || {
        // The scan's "maintain" is writing the new positions into its own array, which is what
        // a caller would do anyway — it has no index to tell.
        let mut items = items.clone();
        for (i, np) in moves.iter().enumerate() { items[i].p = *np; }
        let mut acc = 0usize;
        for q in &qs {
            let (r2, qx, qy, qz) = (radius * radius, q.x, q.y, q.z);
            acc += items.iter().filter(|it| {
                let (dx, dy, dz) = (it.p.x - qx, it.p.y - qy, it.p.z - qz);
                dx * dx + dy * dy + dz * dz <= r2
            }).count();
        }
        std::hint::black_box(acc);
    }).cycles;
    indexed < scanned
}

/// One frame at a given churn fraction: keep the tree with `update_ref`, or refill a grid.
/// Returns (keep_ns, rebuild_ns) as cycles.
/// One frame's cost under each strategy, in cycles: `(keep_tree, rebuild_grid, keep_grid)`.
///
/// The third arm is new, and it is the one the index actually uses now. Until `MortonGrid3`
/// grew `update`, a grid could only be refilled, so a two-armed model was the whole truth and
/// `rebuild_query_ratio` was derived from it. It is not any more: the grid backend keeps in
/// place, and pricing it as a refill overstates its cost by whatever the refill would have
/// been — which pushes the crossover the wrong way and makes the policy reach for the grid
/// later than it should.
fn frame_costs(n: usize, churn: f64, radius: f64, n_queries: usize, leaf: usize) -> (f64, f64, f64, f64) {
    let items = cloud(n, 0xBEEF);
    let mut r = Lcg(11);
    let qs: Vec<Point3> = (0..n_queries).map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();
    let movers: Vec<usize> = (0..n).filter(|i| (*i as f64 / n as f64) < churn).collect();
    let dests: Vec<Point3> = movers.iter().map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();

    let mut tree = Tree3::new(world(), leaf);
    let refs: Vec<_> = items.iter().filter_map(|it| tree.insert_ref(*it)).collect();
    let keep = common::measure(4, || {
        for (j, &i) in movers.iter().enumerate() { tree.update_ref(refs[i], |c| c.p = dests[j]); }
        let mut acc = 0usize;
        for q in &qs { acc += tree.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;

    let levels = MortonGrid3::<P>::levels_for_cell_size(world(), radius);

    // The grid, kept: only the movers are touched, and only the ones that leave their cell
    // re-bucket. Built once outside the timed closure, like the tree above.
    let mut kg = MortonGrid3::new(world(), levels);
    for it in &items { kg.insert(*it); }
    let mut kg_pos: Vec<Point3> = items.iter().map(|it| it.p).collect();
    let mut round = 0usize;
    let keep_grid = common::measure(4, || {
        for (j, &i) in movers.iter().enumerate() {
            // Each repetition must move the item from wherever the last one left it, or the
            // second repetition would be updating from a stale `old` and measuring Missing.
            let d = dests[(j + round) % dests.len().max(1)];
            let old = kg_pos[i];
            kg.update(old, |c| c.id == i as u32, |c| c.p = d);
            kg_pos[i] = d;
        }
        round += 1;
        let mut acc = 0usize;
        for q in &qs { acc += kg.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;

    // The fourth arm, and it should have been here from the start: rebuild a KdTree3. The
    // policy's `Static` backend does exactly this whenever anything moved, so it is a strategy
    // the index actually uses — and the threshold that sends query-heavy workloads to the GRID
    // was derived from a field that did not contain it. Measured in `adaptive_vs_pinned`, at
    // one cull per item a rebuilt k-d tree beat the kept grid by 3.5x, which the calibration
    // had no way of noticing.
    let kd = common::measure(4, || {
        let t = KdTree3::from_items(leaf, items.clone());
        let mut acc = 0usize;
        for q in &qs { acc += t.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;

    let rebuild = common::measure(4, || {
        let mut g = MortonGrid3::new(world(), levels);
        for it in &items { g.insert(*it); }
        let mut acc = 0usize;
        for q in &qs { acc += g.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;
    (keep, rebuild, keep_grid, kd)
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "vh-calibration.txt".into());
    // Both knobs are env-settable so this sweep can be run at the geometry another bench
    // uses — the two disagreed about the k-d arm and the first suspect was that they were
    // simply not measuring the same query.
    let radius: f64 = std::env::var("CAL_R").ok().and_then(|s| s.parse().ok()).unwrap_or(24.0);
    let leaf: usize = std::env::var("CAL_LEAF").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let t0 = Instant::now();
    println!("calibrating on this machine (keep it idle)…\n");

    // --- brute_max: a LADDER, not a bisection.
    //
    // This bisected until 2026-08-03, and a bisection assumes the predicate is monotone. Near
    // the crossover it is not: the two costs are within noise of each other, so "does the index
    // win?" flips at random and the search walks off. The trace it printed said so plainly and
    // was read as a result for months — 527 scan, 975 index, 927 scan, 942 index, converging on
    // whichever way the last coin landed.
    //
    // A ladder cannot do that. Every rung is measured, all of them are printed, and the answer
    // is the largest population where the scan wins on EVERY rung up to it — so a single noisy
    // flip costs one rung's worth of conservatism instead of an order of magnitude.
    let ladder = [16usize, 32, 48, 64, 96, 128, 182, 256, 384, 512, 768, 1024, 2048];
    println!("  {:<10} {:>12}", "population", "winner");
    let mut brute_max = 1usize;
    let mut still_scanning = true;
    for &n in &ladder {
        let indexed_wins = index_beats_scan(n, radius);
        println!("  {:<10} {:>12}", n, if indexed_wins { "index" } else { "scan" });
        if indexed_wins { still_scanning = false; } else if still_scanning { brute_max = n; }
    }

    // --- high_churn: the fraction at which a rebuild starts beating the kept tree.
    // Swept against QUERIES PER FRAME as well, because that is what actually decides it:
    // a rebuild pays a big fixed cost and then answers from a perfectly fitted structure,
    // so the more you query, the sooner rebuilding wins. With a handful of culls a frame
    // the kept tree wins at every churn level; the fluid demo (one neighbour query PER
    // PARTICLE) sits at the other end of that axis, and there the rebuild takes the frame.
    const N: usize = 20_000;
    let query_loads = [16usize, 256, 4096, N / 4];
    println!("
  churn x queries/frame - winner of the whole frame, and by how much");
    println!("  (each cell names the winner of THREE: keep = tree+ItemRef, REBUILD = refilled grid,");
    println!("   gridkeep = the same grid maintained in place, KDTREE = a rebuilt KdTree3,");
    println!("   which is what the policy calls Static)");
    print!("  {:<8}", "churn");
    for q in query_loads { print!("{:>20}", format!("{q} culls")); }
    println!();
    let mut high_churn = 1.0f64;
    for step in 0..=5 {
        let c = step as f64 / 5.0;
        print!("  {:<8.1}", c);
        for q in query_loads {
            let (keep, rebuild, keep_grid, kd) = frame_costs(N, c, radius, q, leaf);
            // Four strategies. The k-d arm is the one the policy calls `Static` and reaches
            // only when nothing has moved — which the sweep can now show is the wrong
            // condition, if it wins cells where things are moving.
            let arms = [("keep", keep), ("REBUILD", rebuild), ("gridkeep", keep_grid), ("KDTREE", kd)];
            let (who, best) = arms.iter().fold(("", f64::MAX), |acc, &(n, v)| if v < acc.1 { (n, v) } else { acc });
            let runner = arms.iter().map(|&(_, v)| v).filter(|&v| v > best).fold(f64::MAX, f64::min);
            print!("{:>20}", format!("{who} {:.2}x", runner / best.max(1.0)));
            if keep > rebuild.min(keep_grid) && high_churn > c { high_churn = c; }
        }
        println!();
    }
    if high_churn >= 1.0 {
        println!("  -> a rebuild never won here: the kept tree stays ahead at every churn and query");
        println!("     load tested, so this machine's policy will not switch to the grid on churn.");
    }

    // The crossover the policy actually uses: queries per item per tick at which the GRID
    // takes the frame from the kept tree. It is derived from whichever grid strategy is
    // cheaper, because the index will use that one — pricing the grid as a refill when it
    // keeps in place is what made the shipped default conservative.
    let mut rebuild_query_ratio = f64::INFINITY;
    let mut old_ratio = f64::INFINITY;
    for q in query_loads {
        let (keep, rebuild, keep_grid, kd) = frame_costs(N, 1.0, radius, q, leaf);
        if rebuild.min(keep_grid).min(kd) < keep { rebuild_query_ratio = rebuild_query_ratio.min(q as f64 / N as f64); }
        // What the two-armed model would have said, kept so the difference is visible rather
        // than asserted.
        if rebuild < keep { old_ratio = old_ratio.min(q as f64 / N as f64); }
    }
    if !rebuild_query_ratio.is_finite() { rebuild_query_ratio = f64::MAX; }
    println!("  -> the grid takes the frame from {rebuild_query_ratio:.3} queries per item per tick");
    if old_ratio.is_finite() && old_ratio > rebuild_query_ratio {
        println!("     (the old rebuild-only model said {old_ratio:.3} — it was pricing a refill that no longer happens)");
    } else if !old_ratio.is_finite() {
        println!("     (the old rebuild-only model never saw the grid win at all at this radius)");
    }

    let th = Thresholds { brute_max, high_churn, rebuild_query_ratio, ..Thresholds::default() };
    let text = th.to_text();
    std::fs::write(&out, &text).unwrap_or_else(|e| panic!("cannot write {out}: {e}"));

    println!("\nmeasured in {:.1}s\n{}", t0.elapsed().as_secs_f64(), text);
    println!("#M calibrate.brute_max {brute_max} n");
    println!("#M calibrate.high_churn {high_churn:.3} frac");
    println!("#M calibrate.rebuild_query_ratio {rebuild_query_ratio:.4} q_per_item");
    println!("wrote {out} — point VH_CALIBRATION at it and AdaptiveIndex::new picks it up.");
    let d = Thresholds::default();
    if brute_max.abs_diff(d.brute_max) * 100 / d.brute_max.max(1) > 25 {
        println!("\nNOTE: this machine's scan/index crossover is {brute_max}; the shipped `brute_max`");
        println!("default is {} and is deliberately LOWER, not wrong. That threshold is an", d.brute_max);
        println!("unconditional floor, consulted before the load-aware `scan_budget` rule, so it sits");
        println!("below the crossover on purpose: above it `scan_budget` decides and can see the");
        println!("query load. Raising it to the crossover would force a scan onto workloads an index");
        println!("wins. Ship the file if this machine's crossover is far from the shipped one.");
    }
}
