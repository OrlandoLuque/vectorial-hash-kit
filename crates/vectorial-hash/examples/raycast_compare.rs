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

use vectorial_hash::{CellState, Point, Positioned, Rect, Shape, Tree, WalkNeighbors};
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

// ------------------------------------------------------------ optimised capsule
struct Capsule2 {
    a: Point,
    abx: f64, aby: f64,
    len2: f64, inv_len2: f64, len: f64,
    ux: f64, uy: f64,
    nx: f64, ny: f64,
    r: f64, r2: f64,
    bbox: Rect,
}
impl Capsule2 {
    fn new(a: Point, b: Point, r: f64) -> Self {
        let (abx, aby) = (b.x - a.x, b.y - a.y);
        let len2 = abx * abx + aby * aby;
        let len = len2.sqrt();
        let (ux, uy) = if len > 0.0 { (abx / len, aby / len) } else { (1.0, 0.0) };
        let (nx, ny) = (-uy, ux);
        let bbox = Rect::new(a.x.min(b.x) - r, a.y.min(b.y) - r, (a.x.max(b.x) - a.x.min(b.x)) + 2.0 * r, (a.y.max(b.y) - a.y.min(b.y)) + 2.0 * r);
        Self { a, abx, aby, len2, inv_len2: if len2 > 0.0 { 1.0 / len2 } else { 0.0 }, len, ux, uy, nx, ny, r, r2: r * r, bbox }
    }
    #[inline]
    fn spine_dist2(&self, p: Point) -> f64 {
        let (apx, apy) = (p.x - self.a.x, p.y - self.a.y);
        let dot = apx * self.abx + apy * self.aby;
        if dot <= 0.0 {
            apx * apx + apy * apy
        } else if dot >= self.len2 {
            let (bpx, bpy) = (p.x - (self.a.x + self.abx), p.y - (self.a.y + self.aby));
            bpx * bpx + bpy * bpy
        } else {
            (apx * apx + apy * apy) - dot * dot * self.inv_len2
        }
    }
}
impl Shape for Capsule2 {
    fn bounding_box(&self) -> Rect { self.bbox }
    fn contains_point(&self, p: Point) -> bool { self.spine_dist2(p) <= self.r2 }
    fn classify_box(&self, b: &Rect) -> Option<CellState> {
        let pick = |dx: f64, dy: f64| {
            let lo = dx * (if dx > 0.0 { b.x } else { b.x_max() }) + dy * (if dy > 0.0 { b.y } else { b.y_max() });
            let hi = dx * (if dx > 0.0 { b.x_max() } else { b.x }) + dy * (if dy > 0.0 { b.y_max() } else { b.y });
            (lo, hi)
        };
        let off_u = self.ux * self.a.x + self.uy * self.a.y;
        let off_n = self.nx * self.a.x + self.ny * self.a.y;
        let (u_lo, u_hi) = pick(self.ux, self.uy);
        let (n_lo, n_hi) = pick(self.nx, self.ny);
        let (u_lo, u_hi) = (u_lo - off_u, u_hi - off_u);
        let (n_lo, n_hi) = (n_lo - off_n, n_hi - off_n);
        if u_hi < -self.r || u_lo > self.len + self.r || n_hi < -self.r || n_lo > self.r {
            return Some(CellState::Out);
        }
        let (cx, cy) = (b.x + b.width * 0.5, b.y + b.height * 0.5);
        let half_diag = 0.5 * (b.width * b.width + b.height * b.height).sqrt();
        if self.r > half_diag && self.spine_dist2(Point::new(cx, cy)) <= (self.r - half_diag) * (self.r - half_diag) {
            return Some(CellState::In);
        }
        Some(CellState::Maybe)
    }
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

/// Median-of-reps milliseconds for one whole ray batch (low-variance timing).
fn time_ms<F: FnMut() -> usize>(reps: usize, mut f: F) -> f64 {
    for _ in 0..2 { std::hint::black_box(f()); }
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let acc = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(acc);
        if ms < best { best = ms; }
    }
    best
}

const WORLD: f64 = 1024.0;
const N_RAYS: usize = 64;
const REPS: usize = 25;

fn main() {
    let methods: &[(&str, WalkNeighbors)] = &[
        ("samet", WalkNeighbors::Samet),
        ("probe", WalkNeighbors::Probe),
        #[cfg(feature = "neighbors")]
        ("ropes", WalkNeighbors::Ropes),
    ];

    println!("ray-cast — exhaustive comparison | world={WORLD}² | {N_RAYS} rays × best-of-{REPS} | times = ms / whole batch");
    #[cfg(not(feature = "neighbors"))]
    println!("(Ropes not compiled — re-run with --features neighbors to include it)");
    println!("cap-naive = capsule, unoptimised classify/contains; cap-opt = optimised; speedup = naive/opt.\n");

    for &n in &[10_000usize, 50_000, 200_000] {
        for &il in &[8usize, 16] {
            let tree = build(n, il, WORLD, 1);
            let rays = gen_rays(N_RAYS, WORLD, 99);
            println!("── N={n}  item_limit={il} ──────────────────────────────────────────────");
            println!("{:>6} | {:>9} {:>8} {:>7} | {:>8} {:>8} | {:>7} {:>8} {:>9}", "radius", "cap-naive", "cap-opt", "speedup", "dda best", "hits", "leaves", "tested", "coverage");
            for &radius in &[2.0_f64, 8.0, 32.0, 128.0] {
                // capsule (before / after) — same exact result, different cost.
                let naive_ms = time_ms(REPS, || rays.iter().map(|r| { let b = Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len); tree.cull(&CapsuleNaive { a: r.o, b, r: radius }).len() }).sum());
                let opt_ms = time_ms(REPS, || rays.iter().map(|r| { let b = Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len); tree.cull(&Capsule2::new(r.o, b, radius)).len() }).sum());

                // reference hit-sets (cap-opt is exact) for coverage.
                let refsets: Vec<std::collections::HashSet<(u64, u64)>> = rays.iter().map(|r| {
                    let b = Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len);
                    tree.cull(&Capsule2::new(r.o, b, radius)).iter().map(|p| key(p)).collect()
                }).collect();
                let cap_hits: usize = refsets.iter().map(|s| s.len()).sum();

                // DDA: time each method, keep the best; stats + coverage (same walk).
                let mut best_ms = f64::INFINITY;
                let (mut leaves, mut tested) = (0usize, 0usize);
                for &(_, walk) in methods {
                    let ms = time_ms(REPS, || rays.iter().map(|r| tree.raycast(r.o, r.d, r.len, radius, walk).hits.len()).sum());
                    if ms < best_ms { best_ms = ms; }
                }
                // walk stats + coverage (method-independent → one pass).
                let (mut found, mut dda_hits) = (0usize, 0usize);
                for (r, rs) in rays.iter().zip(&refsets) {
                    let out = tree.raycast(r.o, r.d, r.len, radius, methods[0].1);
                    leaves += out.leaves_visited;
                    tested += out.items_tested;
                    dda_hits += out.hits.len();
                    let got: std::collections::HashSet<(u64, u64)> = out.hits.iter().map(|(_, p)| key(p)).collect();
                    found += rs.iter().filter(|k| got.contains(k)).count();
                }
                let cov = if cap_hits > 0 { 100.0 * found as f64 / cap_hits as f64 } else { 100.0 };
                println!("{:>6.0} | {:>9.3} {:>8.3} {:>6.1}× | {:>8.3} {:>8} | {:>7} {:>8} {:>8.1}%",
                    radius, naive_ms, opt_ms, naive_ms / opt_ms, best_ms, dda_hits, leaves / N_RAYS, tested / N_RAYS, cov);
            }
            println!();
        }
    }
    println!("DDA hits/leaves/tested are per-ray-batch / per-ray; the walk (leaves/tested) is identical");
    println!("across samet/probe/ropes — only neighbour-finding time differs (ropes O(1) is fastest).");
}
