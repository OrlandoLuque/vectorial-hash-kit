//! Crossover thresholds — where does an index start beating a linear scan?
//! Pins the numbers the `advisor` uses. For a single AoI query over N points:
//! a brute linear scan is O(N) but contiguous + SIMD-friendly; `Tree3::cull`
//! adds a descent but only touches the neighbourhood. Below some N the scan
//! wins (build/descent overhead > scanning everything); above it the tree wins.
//!
//! ```bash
//! cargo run -p vectorial-hash --example threshold_bench --release
//! ```

use std::time::Instant;
use vectorial_hash::{Aabb, Point3, Positioned3, Shape3, Sphere3, Tree3};

const WORLD: f64 = 1_000.0;

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 { f(); let mut b = f64::MAX; for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b }

fn main() {
    // bubble sized to return ~a handful of hits regardless of N (a local query)
    println!("brute linear scan vs Tree3::cull — single AoI query — ns/query\n");
    println!("{:>6} | {:>14} {:>14} | {:>8}", "N", "brute ns", "Tree3 ns", "winner");
    for &n in &[16usize, 32, 64, 128, 256, 512, 1024, 4096] {
        let mut r = Rng(42);
        let pts: Vec<P> = (0..n).map(|_| P(Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD))).collect();
        let tree = Tree3::bulk_load(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), 8, pts.clone());
        // radius so the sphere holds ~5 points: r ≈ (5/N * V / (4/3 π))^(1/3)
        let rad = (5.0 / n as f64 * WORLD.powi(3) / 4.18).cbrt();
        let mut rq = Rng(7);
        let qs: Vec<Sphere3> = (0..2000).map(|_| Sphere3::new(rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD, rad)).collect();
        let brute = best(20, || { let mut h = 0usize; for s in &qs { for p in &pts { if s.contains_point(p.0) { h += 1; } } } std::hint::black_box(h); }) / qs.len() as f64 * 1e9;
        let idx = best(20, || { let mut h = 0usize; for s in &qs { h += tree.cull(s).len(); } std::hint::black_box(h); }) / qs.len() as f64 * 1e9;
        let win = if brute < idx { "brute" } else { "Tree3" };
        println!("{:>6} | {:>14.0} {:>14.0} | {:>8}", n, brute, idx, win);
    }
    println!("\nreading: brute wins at small N (contiguous scan, no descent overhead); the tree\ntakes over as N grows and scanning everything costs more than the descent. The\ncrossover is the advisor's BRUTE_FORCE_MAX. (Depends on element size + query\nradius; measure per workload.)");
}
