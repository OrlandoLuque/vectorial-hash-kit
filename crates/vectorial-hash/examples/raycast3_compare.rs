//! 3D ray-cast comparison — `MortonGrid3` DDA (Amanatides–Woo) vs the capsule
//! cull (`cull(&Segment3)`), plus first-hit early-exit. The 3D counterpart of
//! `raycast_compare`.
//!
//! ```bash
//! cargo run -p vectorial-hash --example raycast3_compare --release
//! ```
//!
//! - **capsule**: `MortonGrid3::cull(&Segment3)` — exact "all within `r`" (pruned by `Segment3::classify_aabb`).
//! - **dda**: `MortonGrid3::raycast` — thin-corridor voxel walk, all hits.
//! - **first**: `MortonGrid3::raycast_first` — nearest hit, front-to-back early-exit.
//!
//! `coverage` = fraction of the capsule's exact hits the thin DDA recovers.

use std::time::Instant;

use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Segment3, Tree3};

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

struct Ray { o: Point3, d: Point3, len: f64 }
fn endpoint(r: &Ray) -> Point3 {
    let m = (r.d.x * r.d.x + r.d.y * r.d.y + r.d.z * r.d.z).sqrt();
    Point3::new(r.o.x + r.d.x / m * r.len, r.o.y + r.d.y / m * r.len, r.o.z + r.d.z / m * r.len)
}

#[derive(Clone, Copy)]
enum Method { Capsule, Dda, First }
fn run_batch(grid: &MortonGrid3<P>, rays: &[Ray], radius: f64, m: Method) -> usize {
    match m {
        Method::Capsule => rays.iter().map(|r| grid.cull(&Segment3::new(r.o, endpoint(r), radius)).len()).sum(),
        Method::Dda => rays.iter().map(|r| grid.raycast(r.o, r.d, r.len, radius).hits.len()).sum(),
        Method::First => rays.iter().map(|r| grid.raycast_first(r.o, r.d, r.len, radius).is_some() as usize).sum(),
    }
}

/// Interleaved min-of-`rounds` ms per method (rotating order → fair under load).
fn time_interleaved(rounds: usize, grid: &MortonGrid3<P>, rays: &[Ray], radius: f64, methods: &[Method]) -> Vec<f64> {
    let n = methods.len();
    let mut best = vec![f64::INFINITY; n];
    for &m in methods { for _ in 0..2 { std::hint::black_box(run_batch(grid, rays, radius, m)); } }
    for round in 0..rounds {
        for k in 0..n {
            let idx = (k + round) % n;
            let t = Instant::now();
            let acc = run_batch(grid, rays, radius, methods[idx]);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(acc);
            if ms < best[idx] { best[idx] = ms; }
        }
    }
    best
}

#[derive(Clone, Copy)]
enum T3 { Cap, Dda, First }
fn run_t3(t: &Tree3<P>, rays: &[Ray], radius: f64, m: T3) -> usize {
    match m {
        T3::Cap => rays.iter().map(|r| t.raycast(r.o, r.d, r.len, radius).len()).sum(),
        T3::Dda => rays.iter().map(|r| t.raycast_dda(r.o, r.d, r.len, radius).hits.len()).sum(),
        T3::First => rays.iter().map(|r| t.raycast_dda_first(r.o, r.d, r.len, radius).is_some() as usize).sum(),
    }
}
fn time_il3(rounds: usize, t: &Tree3<P>, rays: &[Ray], radius: f64, methods: &[T3]) -> Vec<f64> {
    let n = methods.len();
    let mut best = vec![f64::INFINITY; n];
    for &m in methods { for _ in 0..2 { std::hint::black_box(run_t3(t, rays, radius, m)); } }
    for round in 0..rounds {
        for k in 0..n {
            let idx = (k + round) % n;
            let ti = Instant::now();
            let acc = run_t3(t, rays, radius, methods[idx]);
            let ms = ti.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(acc);
            if ms < best[idx] { best[idx] = ms; }
        }
    }
    best
}

const WORLD: f64 = 1024.0;
const N: usize = 200_000;
const LEVELS: u32 = 6; // 64 cells/axis → 16-unit cells
const N_RAYS: usize = 64;
const ROUNDS: usize = 40;

fn key(p: &P) -> (u64, u64, u64) { (p.0.x.to_bits(), p.0.y.to_bits(), p.0.z.to_bits()) }

fn main() {
    let mut rng = Rng::new(1);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let pts: Vec<P> = (0..N).map(|_| P(Point3::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD), rng.range(0.0, WORLD)))).collect();
    let mut grid = MortonGrid3::<P>::new(world, LEVELS);
    for p in &pts { grid.insert(*p); }
    let mut tree3 = Tree3::<P>::new(world, 8);
    for p in &pts { tree3.insert(*p); }

    let rays: Vec<Ray> = (0..N_RAYS).map(|_| {
        let (a, b) = (rng.range(0.0, std::f64::consts::TAU), rng.range(-1.0, 1.0));
        let s = (1.0_f64 - b * b).max(0.0).sqrt();
        Ray { o: Point3::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD), rng.range(0.0, WORLD)), d: Point3::new(s * a.cos(), s * a.sin(), b), len: rng.range(WORLD * 0.4, WORLD) }
    }).collect();

    println!("3D ray-cast — MortonGrid3 | world={WORLD}³ | N={N} | levels={LEVELS} (≈{:.0}-unit cells) | {N_RAYS} rays × min-of-{ROUNDS}", WORLD / (1u64 << LEVELS) as f64);
    println!("{:>6} | {:>10} {:>10} {:>10} | {:>8} {:>8} | {:>8}", "radius", "capsule ms", "dda ms", "first ms", "cells", "tested", "coverage");
    for &radius in &[4.0_f64, 16.0, 64.0, 256.0] {
        let t = time_interleaved(ROUNDS, &grid, &rays, radius, &[Method::Capsule, Method::Dda, Method::First]);
        // DDA stats + coverage (deterministic).
        let (mut cells, mut tested, mut found, mut total) = (0usize, 0usize, 0usize, 0usize);
        for r in &rays {
            let out = grid.raycast(r.o, r.d, r.len, radius);
            cells += out.leaves_visited;
            tested += out.items_tested;
            let refset: std::collections::HashSet<(u64, u64, u64)> = grid.cull(&Segment3::new(r.o, endpoint(r), radius)).iter().map(|p| key(p)).collect();
            let got: std::collections::HashSet<(u64, u64, u64)> = out.hits.iter().map(|(_, p)| key(p)).collect();
            total += refset.len();
            found += refset.iter().filter(|k| got.contains(k)).count();
        }
        let cov = if total > 0 { 100.0 * found as f64 / total as f64 } else { 100.0 };
        println!("{:>6.0} | {:>10.4} {:>10.4} {:>10.4} | {:>8} {:>8} | {:>7.1}%", radius, t[0], t[1], t[2], cells / N_RAYS, tested / N_RAYS, cov);
    }
    println!("\ncells/tested = DDA per-ray (the thin corridor). first = nearest-hit early-exit.");

    // ── Tree3 (variable cells) — capsule cull vs the Probe-style DDA walk ──────
    println!("\n── Tree3 (variable cells, item_limit 8): capsule vs DDA-walk vs first ──");
    println!("{:>6} | {:>10} {:>10} {:>10} | {:>8} {:>8} | {:>8}", "radius", "capsule ms", "dda ms", "first ms", "leaves", "tested", "coverage");
    for &radius in &[4.0_f64, 16.0, 64.0, 256.0] {
        let t = time_il3(ROUNDS, &tree3, &rays, radius, &[T3::Cap, T3::Dda, T3::First]);
        let (mut leaves, mut tested, mut found, mut total) = (0usize, 0usize, 0usize, 0usize);
        for r in &rays {
            let out = tree3.raycast_dda(r.o, r.d, r.len, radius);
            leaves += out.leaves_visited;
            tested += out.items_tested;
            let refset: std::collections::HashSet<(u64, u64, u64)> = tree3.raycast(r.o, r.d, r.len, radius).iter().map(|(_, p)| key(p)).collect();
            let got: std::collections::HashSet<(u64, u64, u64)> = out.hits.iter().map(|(_, p)| key(p)).collect();
            total += refset.len();
            found += refset.iter().filter(|k| got.contains(k)).count();
        }
        let cov = if total > 0 { 100.0 * found as f64 / total as f64 } else { 100.0 };
        println!("{:>6.0} | {:>10.4} {:>10.4} {:>10.4} | {:>8} {:>8} | {:>7.1}%", radius, t[0], t[1], t[2], leaves / N_RAYS, tested / N_RAYS, cov);
    }
    println!("\nTree3 capsule = hierarchical descent (Segment3 + classify_aabb); DDA = Probe leaf-walk (locate per step).");
}
