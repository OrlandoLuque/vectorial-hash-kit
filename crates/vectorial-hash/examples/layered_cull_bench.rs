//! layered_cull_bench — the in-memory complement of `cold_layered_bench`: does
//! `MortonGrid3::cull_layered` (coarse-block skip) actually beat the plain `cull`
//! on a big query, and where does it flip? Same adaptive story as the on-disk
//! layered study: a large query over SPARSE space skips the void and wins; over
//! DENSE space the coarse pass is overhead and plain cull wins. We sweep the
//! populated fraction (clumps in a big world) at a fixed big query and report both.
//!
//! ```bash
//! cargo run -p vectorial-hash --example layered_cull_bench --release
//! ```
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3};

const WORLD: f64 = 4096.0;

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn u(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn best<F: FnMut() -> usize>(reps: usize, mut f: F) -> (f64, usize) {
    let h = f();
    let mut b = f64::MAX;
    for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); }
    (b * 1e6, h) // µs
}

fn main() {
    let n = 300_000usize;
    let levels = 8u32; // 256 cells/axis
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    // a big query spanning most of the world (the case where coarse-skip matters)
    let query = Sphere3::new(WORLD * 0.5, WORLD * 0.5, WORLD * 0.5, WORLD * 0.5);
    println!("MortonGrid3 layered cull | {n} points | levels {levels} | big query r={:.0}\n", WORLD * 0.5);
    println!("{:>16} | {:>10} | {:>12} | {:>12} | {:>8}", "populated", "hits", "cull µs", "layered µs", "speedup");

    // `blobs` clumps concentrate the points into a shrinking fraction of the world.
    for (label, blobs, spread) in [("dense (uniform)", 0usize, 0.0), ("~30%", 40, 0.30), ("~10%", 12, 0.16), ("sparse ~3%", 4, 0.09)] {
        let mut r = Rng(1234 + blobs as u64);
        let centres: Vec<(f64, f64, f64)> = (0..blobs.max(1)).map(|_| (r.u() * WORLD, r.u() * WORLD, r.u() * WORLD)).collect();
        let pts: Vec<P> = (0..n).map(|_| {
            if blobs == 0 { P(Point3::new(r.u() * (WORLD - 1.0), r.u() * (WORLD - 1.0), r.u() * (WORLD - 1.0))) }
            else { let (cx, cy, cz) = centres[(r.next() as usize) % centres.len()]; let s = WORLD * spread;
                P(Point3::new((cx + (r.u() - 0.5) * s).clamp(0.0, WORLD - 1.0), (cy + (r.u() - 0.5) * s).clamp(0.0, WORLD - 1.0), (cz + (r.u() - 0.5) * s).clamp(0.0, WORLD - 1.0))) }
        }).collect();
        let mut grid = MortonGrid3::<P>::new(world, levels);
        for p in &pts { grid.insert(*p); }
        let (t_cull, h1) = best(20, || grid.cull(&query).len());
        let (t_lay, h2) = best(20, || grid.cull_layered(&query).len());
        assert_eq!(h1, h2, "layered must return the same hit count as cull");
        println!("{:>16} | {:>10} | {:>12.1} | {:>12.1} | {:>7.2}x", label, h1, t_cull, t_lay, t_cull / t_lay);
    }
    println!("\nLayered skips empty coarse blocks (Z-order prefix = hierarchy) in O(1); the win\ntracks how much of the query bbox is void — 1.2x (≈uniform) → 40x+ (sparse). Note a\nFINE linear grid is already per-cell sparse (300k in 256^3 = 16.7M cells), so layered\nwins even 'uniform' here; the coarse pass only becomes overhead when nearly every\nqueried cell is occupied — a coarse grid, or a small query inside a packed region.");
}
