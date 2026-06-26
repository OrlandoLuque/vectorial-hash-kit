//! Ray-cast comparison — the **capsule cull** vs the **DDA leaf-walk**, the
//! neighbour strategies against each other, and the **before/after of the
//! distance-math optimisations**. An exhaustive sweep over density, leaf size
//! and ray thickness.
//!
//! ```bash
//! cargo run -p vectorial-hash --example raycast_compare --release
//! # include the Ropes strategy (stored neighbour lists):
//! cargo run -p vectorial-hash --example raycast_compare --release --features neighbors
//! ```
//!
//! Methods, all answering "items within `radius` of the ray":
//! - **cap-naive**: capsule `cull` with the *unoptimised* `classify_box` — exact
//!   segment↔box distance per node (Liang–Barsky + 6 distances), naïve
//!   `contains_point` (per-call division). The "before".
//! - **cap-opt**: same capsule, *optimised* — precomputed segment invariants,
//!   branch-on-projection perpendicular `contains_point` (no division, no
//!   projected point), and a conservative slab `classify_box` in segment-aligned
//!   (u,n) coords + centre-based `In` (no exact box distance). The "after".
//! - **DDA/{samet,probe,ropes}**: thin-corridor walk, neighbour step by strategy.
//!
//! `coverage` = how much of cap-opt's exact result the thin DDA recovers.

use vectorial_hash::{Capsule, CellState, Point, Positioned, Rect, Shape, Tree, WalkNeighbors};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct P(Point);
impl Positioned for P { fn position(&self) -> Point { self.0 } }

fn key(p: &P) -> (u64, u64) { (p.0.x.to_bits(), p.0.y.to_bits()) }

// ---------------------------------------------------------------- naïve capsule
/// Squared distance from `p` to segment `a`–`b` — naïve clamped projection (a
/// division per call, builds the projected point).
#[inline]
fn seg_dist2(p: Point, a: Point, b: Point) -> f64 {
    let (abx, aby) = (b.x - a.x, b.y - a.y);
    let (apx, apy) = (p.x - a.x, p.y - a.y);
    let denom = abx * abx + aby * aby;
    let t = if denom > 0.0 { ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (apx - abx * t, apy - aby * t);
    dx * dx + dy * dy
}
fn seg_hits_box(a: Point, b: Point, r: &Rect) -> bool {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for &(p, q) in &[(-dx, a.x - r.x), (dx, r.x_max() - a.x), (-dy, a.y - r.y), (dy, r.y_max() - a.y)] {
        if p == 0.0 {
            if q < 0.0 { return false; }
        } else {
            let s = q / p;
            if p < 0.0 {
                if s > t1 { return false; }
                if s > t0 { t0 = s; }
            } else {
                if s < t0 { return false; }
                if s < t1 { t1 = s; }
            }
        }
    }
    true
}
fn seg_box_min_dist2(a: Point, b: Point, r: &Rect) -> f64 {
    if seg_hits_box(a, b, r) { return 0.0; }
    let mut m = f64::INFINITY;
    for p in [a, b] {
        let cx = p.x.clamp(r.x, r.x_max());
        let cy = p.y.clamp(r.y, r.y_max());
        m = m.min((p.x - cx).powi(2) + (p.y - cy).powi(2));
    }
    for c in [Point::new(r.x, r.y), Point::new(r.x_max(), r.y), Point::new(r.x, r.y_max()), Point::new(r.x_max(), r.y_max())] {
        m = m.min(seg_dist2(c, a, b));
    }
    m
}
struct CapsuleNaive { a: Point, b: Point, r: f64 }
impl Shape for CapsuleNaive {
    fn bounding_box(&self) -> Rect {
        Rect::new(self.a.x.min(self.b.x) - self.r, self.a.y.min(self.b.y) - self.r, (self.a.x.max(self.b.x) - self.a.x.min(self.b.x)) + 2.0 * self.r, (self.a.y.max(self.b.y) - self.a.y.min(self.b.y)) + 2.0 * self.r)
    }
    fn contains_point(&self, p: Point) -> bool { seg_dist2(p, self.a, self.b) <= self.r * self.r }
    fn classify_box(&self, b: &Rect) -> Option<CellState> {
        let r2 = self.r * self.r;
        if seg_box_min_dist2(self.a, self.b, b) > r2 { return Some(CellState::Out); }
        let far2 = [Point::new(b.x, b.y), Point::new(b.x_max(), b.y), Point::new(b.x, b.y_max()), Point::new(b.x_max(), b.y_max())]
            .iter().map(|&c| seg_dist2(c, self.a, self.b)).fold(0.0_f64, f64::max);
        if far2 <= r2 { Some(CellState::In) } else { Some(CellState::Maybe) }
    }
}

/// The lib `Capsule` but with the SoA batch narrowphase **off** (per-point) —
/// same `classify_box`, so an A/B of just the narrowphase kernel.
struct CapsulePP(Capsule);
impl Shape for CapsulePP {
    fn bounding_box(&self) -> Rect { self.0.bounding_box() }
    fn contains_point(&self, p: Point) -> bool { self.0.contains_point(p) }
    fn classify_box(&self, b: &Rect) -> Option<CellState> { self.0.classify_box(b) }
    // wants_batch defaults false → the per-point path.
}

// ------------------------------------------------------------------- harness
struct Ray { o: Point, d: Point, len: f64 }

fn build(n: usize, il: usize, world: f64, seed: u64) -> Tree<P> {
    let mut rng = Rng::new(seed);
    let mut tree = Tree::<P>::new(Rect::new(0.0, 0.0, world, world), il);
    for _ in 0..n { tree.insert(P(Point::new(rng.range(0.0, world), rng.range(0.0, world)))); }
    tree
}
fn gen_rays(n_rays: usize, world: f64, seed: u64) -> Vec<Ray> {
    let mut rng = Rng::new(seed);
    (0..n_rays).map(|_| {
        let a = rng.range(0.0, std::f64::consts::TAU);
        Ray { o: Point::new(rng.range(0.0, world), rng.range(0.0, world)), d: Point::new(a.cos(), a.sin()), len: rng.range(world * 0.4, world) }
    }).collect()
}

#[inline]
fn endpoint(r: &Ray) -> Point { Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len) }

#[derive(Clone, Copy)]
enum Method { Naive, Opt, OptPp, Dda(WalkNeighbors), First(WalkNeighbors), Walk(WalkNeighbors) }

/// One full ray-batch for a method; returns total hits (the blackholed work).
fn run_batch(tree: &Tree<P>, rays: &[Ray], radius: f64, m: Method) -> usize {
    match m {
        Method::Naive => rays.iter().map(|r| tree.cull(&CapsuleNaive { a: r.o, b: endpoint(r), r: radius }).len()).sum(),
        Method::Opt => rays.iter().map(|r| tree.cull(&Capsule::new(r.o, endpoint(r), radius)).len()).sum(),
        Method::OptPp => rays.iter().map(|r| tree.cull(&CapsulePP(Capsule::new(r.o, endpoint(r), radius))).len()).sum(),
        Method::Dda(w) => rays.iter().map(|r| tree.raycast(r.o, r.d, r.len, radius, w).hits.len()).sum(),
        Method::First(w) => rays.iter().map(|r| tree.raycast_first(r.o, r.d, r.len, radius, w).is_some() as usize).sum(),
        // Exact thick band via the neighbour flood (cull_walk seeded at the ray
        // origin) — the "widened DDA": same result as the descent cull, found by
        // walking instead of descending, with the selectable neighbour method.
        Method::Walk(w) => rays.iter().map(|r| tree.cull_walk(&Capsule::new(r.o, endpoint(r), radius), r.o, w).len()).sum(),
    }
}

/// **Interleaved** min/median-of-`rounds` ms per method. Each round times every
/// method exactly once, **rotating the start order** so no method sits in a
/// fixed position; a transient background load in any round therefore hits all
/// methods, not one — keeping the *comparison* fair even on a busy machine. The
/// `min` is the least-disturbed estimate; the `median` is reported alongside so
/// contamination is visible (median ≫ min ⟹ the run was noisy).
fn time_interleaved(rounds: usize, tree: &Tree<P>, rays: &[Ray], radius: f64, methods: &[Method]) -> Vec<(f64, f64)> {
    let n = methods.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); n];
    for &m in methods { for _ in 0..2 { std::hint::black_box(run_batch(tree, rays, radius, m)); } }
    for round in 0..rounds {
        for k in 0..n {
            let idx = (k + round) % n; // rotate start each round
            let t = Instant::now();
            let acc = run_batch(tree, rays, radius, methods[idx]);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(acc);
            samples[idx].push(ms);
        }
    }
    samples.into_iter().map(|mut v| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[0], v[v.len() / 2]) }).collect()
}

const WORLD: f64 = 1024.0;
const N_RAYS: usize = 64;
const ROUNDS: usize = 40;

fn main() {
    // DDA neighbour strategies (used for the walk stats + coverage).
    let walks: &[WalkNeighbors] = &[
        WalkNeighbors::Samet,
        WalkNeighbors::Probe,
        #[cfg(feature = "neighbors")]
        WalkNeighbors::Ropes,
    ];
    // Every timed method, fed to ONE interleaved timer so a background-load
    // spike in any round is shared across all of them (fair comparison).
    let mut timed = vec![Method::Naive, Method::Opt];
    for &w in walks { timed.push(Method::Dda(w)); }

    println!("ray-cast — exhaustive comparison | world={WORLD}² | {N_RAYS} rays | interleaved min-of-{ROUNDS} (ms/batch)");
    #[cfg(not(feature = "neighbors"))]
    println!("(Ropes not compiled — re-run with --features neighbors to include it)");
    println!("cap-naive/opt = capsule before/after the distance-math tuning. `noise` = worst median/min");
    println!("across methods this row (≈1.0 clean; ≫1 means the machine was busy → re-run).\n");

    for &n in &[10_000usize, 50_000, 200_000] {
        for &il in &[8usize, 16] {
            let tree = build(n, il, WORLD, 1);
            let rays = gen_rays(N_RAYS, WORLD, 99);
            println!("── N={n}  item_limit={il} ──────────────────────────────────────────────");
            println!("{:>6} | {:>9} {:>8} {:>7} | {:>8} {:>8} | {:>7} {:>8} {:>8} {:>6}", "radius", "cap-naive", "cap-opt", "speedup", "dda best", "hits", "leaves", "tested", "cover", "noise");
            for &radius in &[2.0_f64, 8.0, 32.0, 128.0] {
                let stats = time_interleaved(ROUNDS, &tree, &rays, radius, &timed);
                let (naive_min, opt_min) = (stats[0].0, stats[1].0);
                let dda_min = stats[2..].iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
                // Contamination indicator: worst median/min ratio this row.
                let noise = stats.iter().map(|&(mn, md)| if mn > 0.0 { md / mn } else { 1.0 }).fold(1.0_f64, f64::max);

                // Exact reference (cap-opt) + DDA walk stats / coverage (deterministic).
                let refsets: Vec<std::collections::HashSet<(u64, u64)>> = rays.iter().map(|r| {
                    tree.cull(&Capsule::new(r.o, endpoint(r), radius)).iter().map(|p| key(p)).collect()
                }).collect();
                let cap_hits: usize = refsets.iter().map(|s| s.len()).sum();
                let (mut leaves, mut tested, mut found, mut dda_hits) = (0usize, 0usize, 0usize, 0usize);
                for (r, rs) in rays.iter().zip(&refsets) {
                    let out = tree.raycast(r.o, r.d, r.len, radius, walks[0]);
                    leaves += out.leaves_visited;
                    tested += out.items_tested;
                    dda_hits += out.hits.len();
                    let got: std::collections::HashSet<(u64, u64)> = out.hits.iter().map(|(_, p)| key(p)).collect();
                    found += rs.iter().filter(|k| got.contains(k)).count();
                }
                let cov = if cap_hits > 0 { 100.0 * found as f64 / cap_hits as f64 } else { 100.0 };
                println!("{:>6.0} | {:>9.3} {:>8.3} {:>6.1}× | {:>8.3} {:>8} | {:>7} {:>8} {:>7.1}% {:>6.2}",
                    radius, naive_min, opt_min, naive_min / opt_min, dda_min, dda_hits, leaves / N_RAYS, tested / N_RAYS, cov, noise);
            }
            println!();
        }
    }
    // ── First-hit (early-exit) vs walking the whole corridor ──────────────────
    println!("── first-hit (early-exit) vs full DDA corridor — N=200000 item_limit=8 ──");
    println!("{:>6} | {:>9} {:>9} {:>8} | {:>6}", "radius", "full ms", "first ms", "speedup", "noise");
    let tree = build(200_000, 8, WORLD, 1);
    let rays = gen_rays(N_RAYS, WORLD, 99);
    for &radius in &[2.0_f64, 8.0, 32.0, 128.0] {
        let s = time_interleaved(ROUNDS, &tree, &rays, radius, &[Method::Dda(WalkNeighbors::Samet), Method::First(WalkNeighbors::Samet)]);
        let noise = s.iter().map(|&(mn, md)| if mn > 0.0 { md / mn } else { 1.0 }).fold(1.0_f64, f64::max);
        println!("{:>6.0} | {:>9.3} {:>9.3} {:>7.1}× | {:>6.2}", radius, s[0].0, s[1].0, s[0].0 / s[1].0, noise);
    }
    println!();

    // ── Exact thick band: descent cull vs neighbour flood (cull_walk) ─────────
    println!("── exact thick band: descent (cull) vs flood-walk (cull_walk) — N=200000 il=8 ──");
    println!("{:>6} | {:>9} {:>9} | {:>9} | {:>8}", "radius", "descent", "walk best", "thin dda", "walk cov");
    let tree = build(200_000, 8, WORLD, 1);
    let rays = gen_rays(N_RAYS, WORLD, 99);
    for &radius in &[2.0_f64, 8.0, 32.0, 128.0] {
        let mut timed = vec![Method::Opt];
        for &w in walks { timed.push(Method::Walk(w)); }
        timed.push(Method::Dda(WalkNeighbors::Samet));
        let s = time_interleaved(ROUNDS, &tree, &rays, radius, &timed);
        let descent = s[0].0;
        let walk_best = s[1..1 + walks.len()].iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
        let thin = s[1 + walks.len()].0;
        // Coverage: flood-walk vs the exact descent — should be 100%.
        let (mut found, mut total) = (0usize, 0usize);
        for r in &rays {
            let refset: std::collections::HashSet<(u64, u64)> = tree.cull(&Capsule::new(r.o, endpoint(r), radius)).iter().map(|p| key(p)).collect();
            let got: std::collections::HashSet<(u64, u64)> = tree.cull_walk(&Capsule::new(r.o, endpoint(r), radius), r.o, walks[0]).iter().map(|p| key(p)).collect();
            total += refset.len();
            found += refset.iter().filter(|k| got.contains(k)).count();
        }
        let cov = if total > 0 { 100.0 * found as f64 / total as f64 } else { 100.0 };
        println!("{:>6.0} | {:>9.3} {:>9.3} | {:>9.3} | {:>7.1}%", radius, descent, walk_best, thin, cov);
    }
    println!();

    // ── SoA batch narrowphase: per-point vs vectorised, on big leaves ─────────
    println!("── SoA batch narrowphase (Capsule cull): per-point vs batch kernel — N=200000 ──");
    println!("{:>5} {:>7} | {:>10} {:>10} {:>8} | {:>6}", "il", "radius", "per-pt ms", "batch ms", "speedup", "noise");
    for &il in &[16usize, 64, 256] {
        let tree = build(200_000, il, WORLD, 1);
        let rays = gen_rays(N_RAYS, WORLD, 99);
        for &radius in &[8.0_f64, 32.0, 128.0] {
            let s = time_interleaved(ROUNDS, &tree, &rays, radius, &[Method::OptPp, Method::Opt]);
            let noise = s.iter().map(|&(mn, md)| if mn > 0.0 { md / mn } else { 1.0 }).fold(1.0_f64, f64::max);
            println!("{:>5} {:>7.0} | {:>10.3} {:>10.3} {:>7.2}× | {:>6.2}", il, radius, s[0].0, s[1].0, s[0].0 / s[1].0, noise);
        }
    }
    println!("(High item_limit = bigger leaves = more per-item work = where SoA pays. Build with");
    println!(" RUSTFLAGS=\"-C target-cpu=native\" to AVX-vectorise both → the gap shrinks.)\n");

    println!("DDA hits/leaves/tested are per-ray-batch / per-ray; the walk (leaves/tested) is identical");
    println!("across samet/probe/ropes — only neighbour-finding time differs (ropes O(1) is fastest).");
    println!("first-hit early-exits at the first cell with a hit → touches a handful of leaves, not the corridor.");
}
