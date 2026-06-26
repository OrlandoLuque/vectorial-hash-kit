//! Narrowphase ceiling — how much faster is the per-item "within `r` of the
//! segment" test in **SoA + branchless** form (which LLVM auto-vectorises on
//! stable) than the **AoS branch-on-projection** form the tree runs today?
//!
//! ```bash
//! cargo run -p vectorial-hash --example narrowphase_simd --release
//! # to see the real SIMD width of this CPU:
//! RUSTFLAGS="-C target-cpu=native" cargo run -p vectorial-hash --example narrowphase_simd --release
//! ```
//!
//! This is a *microbenchmark of the ceiling*, not wired into the tree: the leaf
//! stores items AoS, so realising this needs the SoA-leaf-storage refactor
//! (backlog). It answers "is it worth it?" — measure the gap, then decide.
//!
//! - **AoS**: `Vec<Point>`, the branch-on-projection perpendicular distance
//!   (caps via `dot ≤ 0` / `dot ≥ len²`) — exact, but the branches block
//!   vectorisation.
//! - **SoA**: `xs[]`, `ys[]`, the branchless clamped-projection form
//!   (`t = clamp(dot·inv_len², 0, 1)`; `|ap − t·ab|²`) in a tight loop that
//!   accumulates a mask — same result, auto-vectorises.

// Microbench kernels take the precomputed segment invariants as loose args.
#![allow(clippy::too_many_arguments)]

use std::time::Instant;

#[derive(Clone, Copy)]
struct Point { x: f64, y: f64 }

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// AoS, branch-on-projection (what the tree / capsule run per item today).
fn count_aos(pts: &[Point], a: Point, abx: f64, aby: f64, len2: f64, inv_len2: f64, r2: f64) -> usize {
    let mut c = 0usize;
    for p in pts {
        let (apx, apy) = (p.x - a.x, p.y - a.y);
        let dot = apx * abx + apy * aby;
        let d2 = if dot <= 0.0 {
            apx * apx + apy * apy
        } else if dot >= len2 {
            let (bx, by) = (apx - abx, apy - aby);
            bx * bx + by * by
        } else {
            (apx * apx + apy * apy) - dot * dot * inv_len2
        };
        c += (d2 <= r2) as usize;
    }
    c
}

/// SoA, branchless clamped projection (auto-vectorises: no data-dependent
/// branches, a mask accumulated into the count). Same exact result.
fn count_soa(xs: &[f64], ys: &[f64], ax: f64, ay: f64, abx: f64, aby: f64, inv_len2: f64, r2: f64) -> usize {
    let mut c = 0usize;
    for i in 0..xs.len() {
        let apx = xs[i] - ax;
        let apy = ys[i] - ay;
        let dot = apx * abx + apy * aby;
        let t = (dot * inv_len2).clamp(0.0, 1.0); // branchless clamp = min/max
        let dx = apx - abx * t;
        let dy = apy - aby * t;
        let d2 = dx * dx + dy * dy;
        c += (d2 <= r2) as usize;
    }
    c
}

fn best_ms<F: FnMut() -> usize>(reps: usize, mut f: F) -> (f64, usize) {
    let mut last = 0;
    for _ in 0..3 { last = std::hint::black_box(f()); }
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        last = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(last);
        if ms < best { best = ms; }
    }
    (best, last)
}

fn main() {
    let n = 4_000_000usize;
    let mut rng = Rng::new(1);
    let pts: Vec<Point> = (0..n).map(|_| Point { x: rng.unit() * 1024.0, y: rng.unit() * 1024.0 }).collect();
    let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p.y).collect();

    // A representative segment + radius.
    let a = Point { x: 100.0, y: 500.0 };
    let b = Point { x: 900.0, y: 540.0 };
    let (abx, aby) = (b.x - a.x, b.y - a.y);
    let len2 = abx * abx + aby * aby;
    let inv_len2 = 1.0 / len2;

    println!("narrowphase ceiling | N={n} points | one segment + radius | best-of-30");
    println!("{:>8} | {:>10} {:>10} | {:>8}", "radius", "AoS ms", "SoA ms", "speedup");
    for &r in &[8.0_f64, 40.0, 160.0] {
        let r2 = r * r;
        let (aos, ca) = best_ms(30, || count_aos(&pts, a, abx, aby, len2, inv_len2, r2));
        let (soa, cs) = best_ms(30, || count_soa(&xs, &ys, a.x, a.y, abx, aby, inv_len2, r2));
        assert_eq!(ca, cs, "AoS and SoA disagree on the count (radius {r})");
        println!("{:>8.0} | {:>10.3} {:>10.3} | {:>7.2}×", r, aos, soa, aos / soa);
    }
    println!("\nSoA = branchless clamped-projection over xs[]/ys[] (auto-vectorised); AoS = the");
    println!("branch-on-projection form the leaf runs today. The gap is the ceiling a SoA leaf");
    println!("+ this kernel would buy the capsule narrowphase. Try RUSTFLAGS=\"-C target-cpu=native\"");
    println!("for this CPU's real SIMD width (default x86_64 = SSE2, 2×f64/lane).");
}
