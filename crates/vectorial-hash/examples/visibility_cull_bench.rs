//! visibility_cull_bench — **occlusion-aware visibility culling** measured.
//!
//! The composable pieces the kit already has, wired into one pipeline: for each
//! *viewer*, find the *targets* within a radius (`cull` / `cull_many_par`), then
//! keep only those with a clear **line of sight** — the segment viewer→target
//! must not pass through any occluder box first (`Polyhedron3::segment_hit`,
//! pruned by an occluder index so each ray tests only nearby blockers). This is
//! the "forests as line-of-sight cover" capability, benchmarked.
//!
//! It answers: cost of the LoS test **per viewer–target pair**, and under what
//! (viewers × targets × geometry) the serial / 16-thread CPU path stays within a
//! frame — i.e. where a GPU offload becomes necessary. Knobs via env:
//!   VIS_VIEWERS  VIS_TARGETS  VIS_R (interest radius)  VIS_OCCLUDERS  VIS_CLUSTER
//!
//! ```bash
//! cargo run -p vectorial-hash --example visibility_cull_bench --release --features parallel
//! ```
use std::time::Instant;
use vectorial_hash::{Aabb, Point3, Positioned3, Polyhedron3, Segment3, Sphere3, Tree3};

const WORLD: f64 = 4000.0;
const OCC_HALF: f64 = 30.0; // occluder box half-extent (a wall/prop chunk)

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

/// An occluder indexed by its centroid; `.1` indexes the parallel polytope array.
#[derive(Clone, Copy)]
struct OccPt(Point3, usize);
impl Positioned3 for OccPt { fn position(&self) -> Point3 { self.0 } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn u(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn env_f64(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// An axis-aligned box occluder centred at `c` with half-extent `h`, as a convex polytope.
fn box_poly(c: Point3, h: f64) -> Polyhedron3 {
    let (lo, hi) = (Point3::new(c.x - h, c.y - h, c.z - h), Point3::new(c.x + h, c.y + h, c.z + h));
    Polyhedron3::from_corners([
        Point3::new(lo.x, lo.y, lo.z), Point3::new(hi.x, lo.y, lo.z), Point3::new(hi.x, hi.y, lo.z), Point3::new(lo.x, hi.y, lo.z),
        Point3::new(lo.x, lo.y, hi.z), Point3::new(hi.x, lo.y, hi.z), Point3::new(hi.x, hi.y, hi.z), Point3::new(lo.x, hi.y, hi.z),
    ])
}

fn main() {
    let n_view = env_usize("VIS_VIEWERS", 2_000);
    let n_tgt = env_usize("VIS_TARGETS", 50_000);
    let r = env_f64("VIS_R", 400.0);
    let n_occ = env_usize("VIS_OCCLUDERS", 4_000);
    let cluster = std::env::var("VIS_CLUSTER").is_ok();
    println!("visibility cull | {n_view} viewers | {n_tgt} targets | r={r} | {n_occ} occluders{}\n", if cluster { " | clustered" } else { "" });

    // ---- data
    let mut rg = Rng(7);
    let blobs: Vec<(f64, f64, f64)> = (0..16).map(|_| (rg.u() * WORLD, rg.u() * WORLD, rg.u() * WORLD)).collect();
    let place = |rg: &mut Rng| -> Point3 {
        if cluster { let (cx, cy, cz) = blobs[(rg.next() as usize) % blobs.len()]; let s = WORLD * 0.05;
            Point3::new((cx + (rg.u() - 0.5) * s).clamp(0.0, WORLD - 1.0), (cy + (rg.u() - 0.5) * s).clamp(0.0, WORLD - 1.0), (cz + (rg.u() - 0.5) * s).clamp(0.0, WORLD - 1.0)) }
        else { Point3::new(rg.u() * (WORLD - 1.0), rg.u() * (WORLD - 1.0), rg.u() * (WORLD - 1.0)) }
    };
    let targets: Vec<P> = (0..n_tgt).map(|_| P(place(&mut rg))).collect();
    let viewers: Vec<Point3> = (0..n_view).map(|_| place(&mut rg)).collect();
    let occ_centers: Vec<Point3> = (0..n_occ).map(|_| Point3::new(rg.u() * WORLD, rg.u() * WORLD, rg.u() * WORLD)).collect();
    let occ_polys: Vec<Polyhedron3> = occ_centers.iter().map(|&c| box_poly(c, OCC_HALF)).collect();

    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let tgt_tree = Tree3::bulk_load(world, 8, targets.clone());
    let occ_tree = Tree3::bulk_load(world, 8, occ_centers.iter().enumerate().map(|(i, &c)| OccPt(c, i)).collect::<Vec<_>>());

    // ---- raw segment_hit micro-cost (one call vs one occluder box)
    let one = box_poly(Point3::new(WORLD * 0.5, WORLD * 0.5, WORLD * 0.5), OCC_HALF);
    let mut rc = Rng(99);
    let raw_ns = { let mut b = f64::MAX; for _ in 0..7 { let t = Instant::now();
        let mut acc = 0.0; for _ in 0..200_000 { let a = Point3::new(rc.u() * WORLD, rc.u() * WORLD, rc.u() * WORLD); let z = Point3::new(rc.u() * WORLD, rc.u() * WORLD, rc.u() * WORLD); if let Some(t) = one.segment_hit(a, z) { acc += t; } }
        std::hint::black_box(acc); b = b.min(t.elapsed().as_secs_f64() / 200_000.0 * 1e9); } b };

    // ---- phase 1: candidate sets (interest cull). Serial + (feature) parallel.
    let spheres: Vec<Sphere3> = viewers.iter().map(|v| Sphere3::new(v.x, v.y, v.z, r)).collect();
    let (cull_ms, cands) = { let t = Instant::now(); let c: Vec<Vec<Point3>> = spheres.iter().map(|s| tgt_tree.cull(s).iter().map(|p| p.0).collect()).collect(); (t.elapsed().as_secs_f64() * 1e3, c) };
    let pairs: u64 = cands.iter().map(|c| c.len() as u64).sum();

    // ---- phase 2: line-of-sight over every (viewer, candidate) pair.
    let los = |v: Point3, c: Point3| -> bool { // true = visible (no occluder blocks)
        let seg = Segment3::new(v, c, OCC_HALF); // capsule radius = occluder half-extent → conservative prune
        for op in occ_tree.cull(&seg) { if let Some(t) = occ_polys[op.1].segment_hit(v, c) { if t < 1.0 { return false; } } }
        true
    };
    let (los_ms, visible) = { let t = Instant::now(); let mut vis = 0u64;
        for (vi, cl) in cands.iter().enumerate() { let v = viewers[vi]; for &c in cl { if los(v, c) { vis += 1; } } }
        (t.elapsed().as_secs_f64() * 1e3, vis) };

    #[cfg(feature = "parallel")]
    let (cull_par_ms, los_par_ms) = {
        use rayon::prelude::*;
        let cull = { let t = Instant::now(); let c = tgt_tree.cull_many_par(&spheres); std::hint::black_box(&c); t.elapsed().as_secs_f64() * 1e3 };
        let losp = { let t = Instant::now(); let v: u64 = cands.par_iter().enumerate().map(|(vi, cl)| { let vv = viewers[vi]; cl.iter().filter(|&&c| los(vv, c)).count() as u64 }).sum(); std::hint::black_box(v); t.elapsed().as_secs_f64() * 1e3 };
        (cull, losp)
    };
    #[cfg(not(feature = "parallel"))]
    let (cull_par_ms, los_par_ms): (f64, f64) = (f64::NAN, f64::NAN);

    // ---- report
    let cbar = pairs as f64 / n_view as f64;
    println!("candidate pairs (viewer×target): {pairs}  (C̄ = {cbar:.1}/viewer)   visible {visible}  ({:.0}% occluded)", 100.0 * (1.0 - visible as f64 / pairs.max(1) as f64));
    println!("raw segment_hit: {raw_ns:.1} ns/call\n");
    println!("{:>26} | {:>12} | {:>16}", "phase", "serial ms", "per pair (µs)");
    println!("{:>26} | {:>12.2} | {:>16}", "interest cull", cull_ms, "—");
    println!("{:>26} | {:>12.2} | {:>16.4}", "line-of-sight", los_ms, los_ms * 1e3 / pairs.max(1) as f64);
    println!("{:>26} | {:>12.2} | {:>16.4}", "TOTAL (serial)", cull_ms + los_ms, (cull_ms + los_ms) * 1e3 / pairs.max(1) as f64);
    if cull_par_ms.is_finite() {
        println!("\n{:>26} | {:>12} ", "phase (16 threads)", "parallel ms");
        println!("{:>26} | {:>12.2}", "interest cull_many_par", cull_par_ms);
        println!("{:>26} | {:>12.2}", "line-of-sight (par)", los_par_ms);
        println!("{:>26} | {:>12.2}  ⇐ CPU frame (parallel)", "TOTAL", cull_par_ms + los_par_ms);
        let frame = cull_par_ms + los_par_ms;
        let budget = 16.0; // 60 Hz
        let max_pairs_k = pairs as f64 * budget / (cull_par_ms + los_par_ms).max(1e-9) / 1e3;
        println!("\nfits a {budget:.0} ms (60 Hz) frame: {} ({:.1}× headroom). Cost scales with viewers×C̄;", if frame <= budget { "YES" } else { "NO" }, budget / frame);
        println!("this per-pair cost tops out at ~{max_pairs_k:.0}k pairs/frame on CPU-parallel — beyond that → GPU");
    }
}
