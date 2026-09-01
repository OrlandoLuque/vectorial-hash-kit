//! `pick_a_structure` — describe your workload, get a measured answer for YOUR machine.
//!
//! The kit's decision maps answer "which structure wins" for the workloads the kit happens to
//! measure. That is not the same as answering it for yours, and the margins are often small
//! enough that a different cache hierarchy flips them. This runs the comparison on the machine
//! you will deploy on, with the population, churn and query load you actually have.
//!
//! ```bash
//! cargo run -p vectorial-hash --example pick_a_structure --release
//! cargo run -p vectorial-hash --example pick_a_structure --release -- \
//!     n=50000 churn=0.3 queries=2 radius=30 dist=clustered
//! ```
//!
//! - `n` items, `churn` = fraction that move each tick, `queries` = culls per item per tick,
//!   `radius` = typical query size in world units, `dist` = `uniform` | `clustered`.
//!
//! **What it reports, and why in that order.** Per candidate: the per-frame cost of *keeping the
//! index current* and the cost of *answering the queries*, then the total. Those two are the
//! whole trade — an index costs per move and a scan costs per query — and separating them is
//! what lets you see WHY the winner won rather than just that it did.
//!
//! It closes with the honest recommendation, which is sometimes "do not use the adaptive index":
//! that layer is insurance for a workload you cannot predict, and if this run describes your
//! workload accurately, you can predict it, so pinning the winner is strictly cheaper.
mod common;
use common::wall_ms_consuming;
use std::time::Instant;
use vectorial_hash::{AdaptiveIndex, Aabb, Backend, Hints, KdTree3, MortonGrid3, Point3, Positioned3, Shape3, Sphere3, Tree3};

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const W: f64 = 512.0;
fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, W, W, W) }

fn arg(name: &str, default: f64) -> f64 {
    std::env::args().find_map(|a| a.strip_prefix(&format!("{name}="))?.parse().ok()).unwrap_or(default)
}

fn main() {
    let n = arg("n", 20_000.0) as usize;
    let churn = arg("churn", 0.2);
    let qpi = arg("queries", 1.0);
    let radius = arg("radius", 36.0);
    let clustered = std::env::args().any(|a| a == "dist=clustered");
    let frames = arg("frames", 60.0) as usize;

    println!("workload: {n} items · {:.0}% moving per tick · {qpi} culls/item/tick · radius {radius} \
              · {}\nworld {W}^3 · {frames} frames measured · THIS machine\n",
             churn * 100.0, if clustered { "clustered" } else { "uniform" });

    let mut rng = Rng(0xA11CE);
    let items: Vec<P> = (0..n).map(|_| {
        if clustered {
            // Eight blobs: the case where a chosen split has something to choose and a uniform
            // grid starts putting everything in a handful of cells.
            let c = [(90.0, 90.0, 90.0), (400.0, 120.0, 300.0), (250.0, 400.0, 100.0), (60.0, 300.0, 420.0),
                     (430.0, 430.0, 430.0), (200.0, 60.0, 200.0), (120.0, 240.0, 60.0), (330.0, 200.0, 380.0)]
                     [(rng.next() % 8) as usize];
            P { p: Point3::new(c.0 + (rng.f() - 0.5) * 70.0, c.1 + (rng.f() - 0.5) * 70.0, c.2 + (rng.f() - 0.5) * 70.0) }
        } else {
            P { p: Point3::new(rng.f() * W, rng.f() * W, rng.f() * W) }
        }
    }).collect();

    let n_moves = (n as f64 * churn) as usize;
    let n_queries = ((n as f64 * qpi) as usize).max(1);
    let probes: Vec<Sphere3> = (0..n_queries.min(4096)).map(|i| {
        let it = items[i * items.len() / n_queries.clamp(1, 4096)];
        Sphere3::new(it.p.x, it.p.y, it.p.z, radius)
    }).collect();

    println!("{:>22} {:>13} {:>13} {:>13}", "candidate", "maintain ms", "query ms", "total ms");
    let mut results: Vec<(String, f64)> = Vec::new();

    // --- brute scan: no maintenance at all, pays per query
    {
        let mut v = items.clone();
        let maint = bench(frames, || { for it in v.iter_mut().take(n_moves) { it.p.x = (it.p.x + 1.0) % W; } });
        let query = bench(frames, || {
            for s in &probes { std::hint::black_box(v.iter().filter(|it| s.contains_point(it.position())).count()); }
        });
        row("brute scan", maint, query, &mut results);
    }
    // --- the keep-index tree: O(1) relocation
    {
        let mut t = Tree3::new(world(), 8);
        let refs: Vec<_> = items.iter().filter_map(|it| t.insert_ref(*it)).collect();
        let mut pos: Vec<P> = items.clone();
        let maint = bench(frames, || {
            for i in 0..n_moves { pos[i].p.x = (pos[i].p.x + 1.0) % W; let p = pos[i]; t.update_ref(refs[i], |c| *c = p); }
        });
        let query = bench(frames, || { for s in &probes { std::hint::black_box(t.cull(s).len()); } });
        row("Tree3 + ItemRef", maint, query, &mut results);
    }
    // --- the grid: keeps in place too, since 2026-07-31
    {
        let levels = MortonGrid3::<P>::levels_for_cell_size(world(), radius);
        let mut g = MortonGrid3::new(world(), levels);
        for it in &items { g.insert(*it); }
        // The grid finds an item by where it WAS, so the predicate has to identify it. Exact
        // float equality is the right test here and only here: it is the same value read back,
        // not a computed one.
        let mut pos: Vec<P> = items.clone();
        let maint = bench(frames, || {
            for it in pos.iter_mut().take(n_moves) {
                let was = it.p;
                it.p.x = (it.p.x + 1.0) % W;
                let p = *it;
                #[allow(clippy::float_cmp)]
                g.update(was, |c| c.p.x == was.x && c.p.y == was.y && c.p.z == was.z, |c| *c = p);
            }
        });
        let query = bench(frames, || { for s in &probes { std::hint::black_box(g.cull(s).len()); } });
        row("MortonGrid3 (kept)", maint, query, &mut results);
    }
    // --- build-once: rebuilt every frame, which is the honest way to enter it here
    {
        let src = items.clone();
        let maint = wall_ms_consuming(frames.min(9), &src, |v| { std::hint::black_box(KdTree3::from_items(8, v)); });
        let k = KdTree3::from_items(8, items.clone());
        let query = bench(frames, || { for s in &probes { std::hint::black_box(k.cull(s).len()); } });
        row("KdTree3 (rebuilt)", maint, query, &mut results);
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (best, best_ms) = results[0].clone();
    let (_, worst_ms) = results.last().unwrap().clone();

    // What the adaptive policy would pick, told the same thing — without measuring anything.
    //
    // Read `backend()` after `prepare`, NOT `recommended()`: the latter answers for the index as
    // it stands, and right now it stands empty, so it would truthfully say "a scan" and be
    // useless here. `prepare` is the call that reasons about the promised population, and it has
    // already migrated to its answer. (This tool caught that on its first run.)
    let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
    ix.prepare(Hints { expected_count: Some(n), churn: Some(churn), queries_per_item: Some(qpi),
                       query_extent: Some(radius * 2.0), distribution: None });
    let policy = ix.backend();

    println!("\nfastest here: {best} ({best_ms:.2} ms/frame), {:.2}x ahead of the slowest", worst_ms / best_ms);
    println!("the adaptive policy, given the same description, would pick: {policy:?}");
    let agrees = matches!((policy, best.as_str()),
        (Backend::Brute, "brute scan") | (Backend::KeepTree, "Tree3 + ItemRef")
        | (Backend::Grid, "MortonGrid3 (kept)") | (Backend::Static, "KdTree3 (rebuilt)"));
    println!("{}", if agrees { "  -> it agrees with the measurement." }
                   else { "  -> it DISAGREES with the measurement on this machine. Trust the measurement,\n     \
                          and consider shipping a calibration: `cargo run --example calibrate`." });

    println!();
    println!("A DISAGREEMENT IS INFORMATION, not necessarily a bug in either side. One is known");
    println!("and reproducible: at low churn with light query load, this reports the grid and the");
    println!("policy reports the keep-tree. `rebuild_query_ratio` was derived at MAXIMUM churn,");
    println!("where keeping a grid is worthless -- but a grid that barely has to be maintained");
    println!("keeps its cheaper cull for free, so the real frontier is diagonal in");
    println!("(churn x query load) and the threshold is a vertical line through it.");
    println!("  try: pick_a_structure -- churn=0.001 queries=0.05");
    println!();
    println!("If this description matches your real workload, PIN the winner: construct it");
    println!("directly and skip the adaptive layer. That layer is insurance for a workload you");
    println!("cannot predict — it costs a migration and a detector lag to discover what you just");
    println!("read off a table. Reach for it when the numbers above would be different at");
    println!("different moments of your program's life, and pin otherwise.");
    println!();
    println!("If you do want it adaptive but with your own rule, the switch is yours to drive:");
    println!("  ix.observed()      -> (items, queries/item, moves/item), the policy's whole input");
    println!("  ix.recommended()   -> what the built-in policy would do, without doing it");
    println!("  ix.migrate_to(b)   -> switch now; slots keep addressing the same items");
    println!("  ix.freeze()/thaw() -> and stop it being second-guessed");
}

fn bench<F: FnMut()>(frames: usize, mut f: F) -> f64 {
    f(); // warm
    let mut best = f64::INFINITY;
    for _ in 0..frames.clamp(1, 9) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn row(name: &str, maint: f64, query: f64, out: &mut Vec<(String, f64)>) {
    println!("{name:>22} {maint:>13.3} {query:>13.3} {:>13.3}", maint + query);
    out.push((name.to_string(), maint + query));
}
