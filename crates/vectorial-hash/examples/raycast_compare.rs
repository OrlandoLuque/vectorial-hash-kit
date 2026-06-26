//! Ray-cast comparison — the **capsule cull** vs the **DDA leaf-walk**, and the
//! three neighbour-finding strategies against each other. The prototype harness
//! for "which ray-cast, and does maintaining ropes pay off?".
//!
//! ```bash
//! cargo run -p vectorial-hash --example raycast_compare --release
//! # to include the Ropes strategy (stored neighbour lists):
//! cargo run -p vectorial-hash --example raycast_compare --release --features neighbors
//! ```
//!
//! Two ray-casts answer "items within `radius` of the ray":
//! - **capsule**: `Tree::cull(&Capsule)` — exact, descends the segment's fat
//!   AABB (visits off-corridor cells; gathers everything, unordered).
//! - **DDA**: `Tree::raycast(.., walk)` — walks only the cells the centre line
//!   crosses, front-to-back, stepping via Samet / Probe / Ropes.
//!
//! The DDA's thin corridor can MISS items within `radius` that sit in cells the
//! centre line doesn't enter — the coverage column quantifies that vs the
//! capsule (the precision price of the thin walk; grows with `radius`).

use std::time::Instant;

use vectorial_hash::{Point, Positioned, Rect, Shape, Tree, WalkNeighbors};

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

/// Squared distance from `p` to segment `a`–`b` (clamped projection).
#[inline]
fn seg_dist2(p: Point, a: Point, b: Point) -> f64 {
    let (abx, aby) = (b.x - a.x, b.y - a.y);
    let (apx, apy) = (p.x - a.x, p.y - a.y);
    let denom = abx * abx + aby * aby;
    let t = if denom > 0.0 { ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (apx - abx * t, apy - aby * t);
    dx * dx + dy * dy
}

/// A 2D capsule (segment + radius) as a `Shape`: the cull-based ray-cast.
struct Capsule2 { a: Point, b: Point, r: f64 }
impl Shape for Capsule2 {
    fn bounding_box(&self) -> Rect {
        let lx = self.a.x.min(self.b.x) - self.r;
        let ly = self.a.y.min(self.b.y) - self.r;
        let hx = self.a.x.max(self.b.x) + self.r;
        let hy = self.a.y.max(self.b.y) + self.r;
        Rect::new(lx, ly, hx - lx, hy - ly)
    }
    fn contains_point(&self, p: Point) -> bool { seg_dist2(p, self.a, self.b) <= self.r * self.r }
}

const WORLD: f64 = 1024.0;
const N: usize = 50_000;
const IL: usize = 8;
const N_RAYS: usize = 128;
const REPS: usize = 60;

/// A pre-generated ray: origin, unit direction, length.
struct Ray { o: Point, d: Point, len: f64 }

fn key(p: &P) -> (u64, u64) { (p.0.x.to_bits(), p.0.y.to_bits()) }

fn main() {
    let mut rng = Rng::new(42);
    let mut tree = Tree::<P>::new(Rect::new(0.0, 0.0, WORLD, WORLD), IL);
    for _ in 0..N { tree.insert(P(Point::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD)))); }

    let rays: Vec<Ray> = (0..N_RAYS).map(|_| {
        let a = rng.range(0.0, std::f64::consts::TAU);
        Ray { o: Point::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD)), d: Point::new(a.cos(), a.sin()), len: rng.range(WORLD * 0.4, WORLD) }
    }).collect();

    let methods: &[(&str, WalkNeighbors)] = &[
        ("DDA/samet", WalkNeighbors::Samet),
        ("DDA/probe", WalkNeighbors::Probe),
        #[cfg(feature = "neighbors")]
        ("DDA/ropes", WalkNeighbors::Ropes),
    ];

    println!("ray-cast comparison | world={WORLD}² | N={N} | item_limit={IL} | {N_RAYS} rays × {REPS} reps");
    #[cfg(not(feature = "neighbors"))]
    println!("(Ropes not compiled — re-run with --features neighbors to include it)");
    println!("\n{:<11} {:>6} | {:>10} {:>8} | {:>8} {:>9} {:>10}", "method", "radius", "total ms", "hits", "leaves", "tested", "coverage");

    for &radius in &[2.0_f64, 8.0, 24.0, 64.0] {
        // capsule (cull) — the reference result + a timing.
        let mut cap_hits = 0usize;
        let t = Instant::now();
        for _ in 0..REPS {
            for r in &rays {
                let b = Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len);
                cap_hits += tree.cull(&Capsule2 { a: r.o, b, r: radius }).len();
            }
        }
        let cap_ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
        println!("{:<11} {:>6.0} | {:>10.4} {:>8} | {:>8} {:>9} {:>10}", "capsule", radius, cap_ms, cap_hits / REPS, "-", "-", "100% (ref)");

        // reference hit-sets per ray (for the coverage comparison).
        let refsets: Vec<std::collections::HashSet<(u64, u64)>> = rays.iter().map(|r| {
            let b = Point::new(r.o.x + r.d.x * r.len, r.o.y + r.d.y * r.len);
            tree.cull(&Capsule2 { a: r.o, b, r: radius }).iter().map(|p| key(p)).collect()
        }).collect();

        for &(name, walk) in methods {
            let t = Instant::now();
            let (mut hits, mut leaves, mut tested) = (0usize, 0usize, 0usize);
            for _ in 0..REPS {
                for r in &rays {
                    let out = tree.raycast(r.o, r.d, r.len, radius, walk);
                    hits += out.hits.len();
                    leaves += out.leaves_visited;
                    tested += out.items_tested;
                }
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
            // coverage: of the capsule's hits, how many did the DDA also find?
            let (mut found, mut total) = (0usize, 0usize);
            for (r, rs) in rays.iter().zip(&refsets) {
                let got: std::collections::HashSet<(u64, u64)> = tree.raycast(r.o, r.d, r.len, radius, walk).hits.iter().map(|(_, p)| key(p)).collect();
                total += rs.len();
                found += rs.iter().filter(|k| got.contains(k)).count();
            }
            let cov = if total > 0 { 100.0 * found as f64 / total as f64 } else { 100.0 };
            println!("{:<11} {:>6.0} | {:>10.4} {:>8} | {:>8} {:>9} {:>9.1}%", name, radius, ms, hits / REPS, leaves / (REPS * N_RAYS), tested / (REPS * N_RAYS), cov);
        }
        println!();
    }
    println!("Reading it: DDA `leaves`/`tested` are the thin-corridor cost (same across methods —");
    println!("only neighbour-finding time differs: samet/probe O(depth), ropes O(1)). `coverage` is");
    println!("how much of the capsule's exact result the thin walk recovers — it drops as radius grows.");
}
