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
//! Both are measured with min-of-N on an otherwise quiet machine; run it on an idle box
//! or the numbers it writes will be pessimistic in ways the policy then bakes in.

use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3, Thresholds, Tree3};

#[path = "common/mod.rs"]
mod common;

#[derive(Clone, Copy)]
struct P { p: Point3 }
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
    (0..n).map(|_| P { p: Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W)) }).collect()
}

/// Is an index worth it at this population? Compares one indexed cull against one scan,
/// both min-of-N, INCLUDING neither build (the index is assumed already built — that is
/// the regime the threshold is about).
fn index_beats_scan(n: usize, radius: f64) -> bool {
    let items = cloud(n, 0xC0FFEE + n as u64);
    let tree = Tree3::bulk_load(world(), 8, items.clone());
    let mut r = Lcg(7);
    let qs: Vec<Point3> = (0..64).map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();

    let indexed = common::measure(5, || {
        let mut acc = 0usize;
        for q in &qs { acc += tree.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;
    let scanned = common::measure(5, || {
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
fn frame_costs(n: usize, churn: f64, radius: f64, n_queries: usize) -> (f64, f64) {
    let items = cloud(n, 0xBEEF);
    let mut r = Lcg(11);
    let qs: Vec<Point3> = (0..n_queries).map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();
    let movers: Vec<usize> = (0..n).filter(|i| (*i as f64 / n as f64) < churn).collect();
    let dests: Vec<Point3> = movers.iter().map(|_| Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W))).collect();

    let mut tree = Tree3::new(world(), 8);
    let refs: Vec<_> = items.iter().filter_map(|it| tree.insert_ref(*it)).collect();
    let keep = common::measure(4, || {
        for (j, &i) in movers.iter().enumerate() { tree.update_ref(refs[i], |c| c.p = dests[j]); }
        let mut acc = 0usize;
        for q in &qs { acc += tree.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;

    let levels = MortonGrid3::<P>::levels_for_cell_size(world(), radius);
    let rebuild = common::measure(4, || {
        let mut g = MortonGrid3::new(world(), levels);
        for it in &items { g.insert(*it); }
        let mut acc = 0usize;
        for q in &qs { acc += g.cull(&Sphere3::new(q.x, q.y, q.z, radius)).len(); }
        std::hint::black_box(acc);
    }).cycles;
    (keep, rebuild)
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "vh-calibration.txt".into());
    let radius = 24.0;
    let t0 = Instant::now();
    println!("calibrating on this machine (keep it idle)…\n");

    // --- brute_max: bisect on "does the index win yet?"
    let (mut lo, mut hi) = (16usize, 8192usize);
    println!("  {:<10} {:>12}", "population", "winner");
    while lo < hi {
        let mid = (lo + hi) / 2;
        let indexed_wins = index_beats_scan(mid, radius);
        println!("  {:<10} {:>12}", mid, if indexed_wins { "index" } else { "scan" });
        if indexed_wins { hi = mid; } else { lo = mid + 1; }
    }
    let brute_max = lo.saturating_sub(1).max(1);

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
    print!("  {:<8}", "churn");
    for q in query_loads { print!("{:>20}", format!("{q} culls")); }
    println!();
    let mut high_churn = 1.0f64;
    for step in 0..=5 {
        let c = step as f64 / 5.0;
        print!("  {:<8.1}", c);
        for q in query_loads {
            let (keep, rebuild) = frame_costs(N, c, radius, q);
            let (who, by) = if keep <= rebuild { ("keep", rebuild / keep.max(1.0)) } else { ("REBUILD", keep / rebuild.max(1.0)) };
            print!("{:>20}", format!("{who} {by:.2}x"));
            if keep > rebuild && high_churn > c { high_churn = c; }
        }
        println!();
    }
    if high_churn >= 1.0 {
        println!("  -> a rebuild never won here: the kept tree stays ahead at every churn and query");
        println!("     load tested, so this machine's policy will not switch to the grid on churn.");
    }

    // The crossover the policy actually uses: queries per item per tick at which the
    // rebuild takes the frame. Read off the sweep above at the highest churn tested.
    let mut rebuild_query_ratio = f64::INFINITY;
    for q in query_loads {
        let (keep, rebuild) = frame_costs(N, 1.0, radius, q);
        if rebuild < keep { rebuild_query_ratio = rebuild_query_ratio.min(q as f64 / N as f64); }
    }
    if !rebuild_query_ratio.is_finite() { rebuild_query_ratio = f64::MAX; }
    println!("  -> rebuild takes the frame from {rebuild_query_ratio:.3} queries per item per tick");

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
        println!("\nNOTE: brute_max is {brute_max} here vs the shipped default {} — a >25% difference,", d.brute_max);
        println!("which is exactly the case this tool exists for. Ship the file with the build.");
    }
}
