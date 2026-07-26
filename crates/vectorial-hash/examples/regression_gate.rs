//! Deterministic performance **regression gate** — the committed counterpart to
//! the Criterion benches. Criterion gives rich local reports but its baselines
//! live under `target/` and CI runners are too noisy to gate on. This gate is
//! the opposite: a small, fixed, low-variance set of timings (median of many
//! reps) checked against a **committed** baseline, so a real regression can fail
//! a build.
//!
//! ```bash
//! # capture the baseline on this machine (run on a quiet system):
//! cargo run -p vectorial-hash --example regression_gate --release -- --save
//! # later / in CI — compare, exit 1 if any op regressed past the threshold:
//! cargo run -p vectorial-hash --example regression_gate --release
//! cargo run -p vectorial-hash --example regression_gate --release -- --threshold 0.30
//! ```
//!
//! The baseline file is `benches/baseline.tsv` (op<TAB>nanoseconds). It is
//! hardware-specific: regenerate with `--save` when you change machines, and
//! treat cross-machine numbers as orientation only. The gate compares *ratios*,
//! so it is robust to absolute speed as long as the baseline was taken here.

// The `x < lo || x > hi` bounce test reads clearer than `!(lo..=hi).contains()`.
#![allow(clippy::manual_range_contains)]

use std::hint::black_box;
use std::time::Instant;
use vectorial_hash::{Aabb, KdTree3, LinearOctree3, MortonGrid3, Octree3, Point3, Positioned3, Sphere3, Tree3};

const WORLD: f64 = 512.0;
const N: usize = 20_000;
const IL: usize = 8;
const VISION: f64 = 36.0;
const N_QUERY: usize = 64;
const MARGIN: f64 = 4.0;
const REPS: usize = 40;
const BASELINE: &str = "crates/vectorial-hash/benches/baseline.tsv";

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct C3 { id: u32, p: Point3 }
impl Positioned3 for C3 { fn position(&self) -> Point3 { self.p } }

fn items() -> Vec<C3> {
    let mut r = Rng::new(1);
    (0..N).map(|id| C3 { id: id as u32, p: Point3::new(r.range(MARGIN, WORLD - MARGIN), r.range(MARGIN, WORLD - MARGIN), r.range(MARGIN, WORLD - MARGIN)) }).collect()
}
fn vels() -> Vec<(f64, f64, f64)> {
    let mut r = Rng::new(5);
    (0..N).map(|_| { let s = r.range(0.35 * 120.0, 120.0); let (a, b) = (r.range(0.0, std::f64::consts::TAU), r.range(-1.0, 1.0)); let h = (1.0_f64 - b * b).max(0.0).sqrt(); (s * h * a.cos(), s * h * a.sin(), s * b) }).collect()
}
fn queries() -> Vec<Sphere3> {
    let mut r = Rng::new(99);
    (0..N_QUERY).map(|_| Sphere3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD), VISION)).collect()
}
#[inline]
fn step(p: &mut Point3, v: &mut (f64, f64, f64)) -> Point3 {
    let dt = 1.0 / 60.0;
    let mut nx = p.x + v.0 * dt; let mut ny = p.y + v.1 * dt; let mut nz = p.z + v.2 * dt;
    if nx < MARGIN || nx > WORLD - MARGIN { v.0 = -v.0; nx = nx.clamp(MARGIN, WORLD - MARGIN); }
    if ny < MARGIN || ny > WORLD - MARGIN { v.1 = -v.1; ny = ny.clamp(MARGIN, WORLD - MARGIN); }
    if nz < MARGIN || nz > WORLD - MARGIN { v.2 = -v.2; nz = nz.clamp(MARGIN, WORLD - MARGIN); }
    *p = Point3::new(nx, ny, nz); *p
}

/// **Min-of-`REPS`** nanoseconds for one op, after a short warmup. The fastest
/// sample is the truest measure of the code's cost: noise (interrupts,
/// scheduling, turbo dips) only ever *adds* time, so the minimum is the run
/// least disturbed by it — far more stable than the median for microbenchmarks.
/// The closure returns a blackholed accumulator so the work cannot be elided.
fn bench<F: FnMut() -> u64>(mut f: F) -> f64 {
    for _ in 0..5 { black_box(f()); }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        let acc = f();
        let ns = t.elapsed().as_nanos() as f64;
        black_box(acc);
        if ns < best { best = ns; }
    }
    best
}

/// A fixed, allocation-free CPU workload used to **normalise away global clock
/// scaling** (turbo/thermal/background load) between the baseline run and a
/// later run. Every op is compared as a ratio to this number, so a machine
/// running, say, 1.3× slower overall does not read as a regression — only a
/// *relative* slowdown of the op against the CPU does. Deterministic, so it is
/// itself a stable yardstick.
fn calibrate() -> f64 {
    bench(|| {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..4_000_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
            x ^= x >> 29;
        }
        x
    })
}

fn measure() -> Vec<(&'static str, f64)> {
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let its = items();
    let qs = queries();
    let levels = MortonGrid3::<C3>::levels_for_cell_size(aabb, VISION);
    let mut out = Vec::new();

    out.push(("build_tree3", bench(|| { let mut t = Tree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_octree3", bench(|| { let mut t = Octree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_morton3", bench(|| { let mut t = MortonGrid3::<C3>::new(aabb, levels); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    // The two build-once structures: they have no incremental-insert story to defend, so
    // they are gated on the build the caller actually uses (bulk) plus their queries.
    out.push(("build_kdtree3", bench(|| KdTree3::from_items(IL, its.clone()).item_count() as u64)));
    out.push(("build_linear_octree3", bench(|| LinearOctree3::from_items(aabb, IL, 12, its.clone()).item_count() as u64)));

    let mut tree3 = Tree3::<C3>::new(aabb, IL); for it in &its { tree3.insert(*it); }
    let mut octree3 = Octree3::<C3>::new(aabb, IL); for it in &its { octree3.insert(*it); }
    let mut morton3 = MortonGrid3::<C3>::new(aabb, levels); for it in &its { morton3.insert(*it); }

    out.push(("cull_tree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += tree3.cull(s).len() as u64; } n })));
    out.push(("cull_octree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += octree3.cull(s).len() as u64; } n })));
    out.push(("cull_morton3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += morton3.cull(s).len() as u64; } n })));
    let kd3 = KdTree3::from_items(IL, its.clone());
    let lo3 = LinearOctree3::from_items(aabb, IL, 12, its.clone());
    out.push(("cull_kdtree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += kd3.cull(s).len() as u64; } n })));
    out.push(("cull_linear_octree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += lo3.cull(s).len() as u64; } n })));

    let qp: Vec<Point3> = { let mut r = Rng::new(7); (0..256).map(|_| Point3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD))).collect() };
    out.push(("knn_tree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += tree3.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_kdtree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += kd3.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_linear_octree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += lo3.knn(*q, 16).len() as u64; } n })));

    // update: one frame of N relocations, predicate vs ItemRef.
    {
        let mut t = Tree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); }
        let mut pos: Vec<Point3> = its.iter().map(|i| i.p).collect();
        let mut vel = vels();
        out.push(("update_predicate_frame", bench(|| { for id in 0..N { let old = pos[id]; let np = step(&mut pos[id], &mut vel[id]); let cid = id as u32; t.update(old, |c| c.id == cid, |c| c.p = np); } 0 })));
    }
    {
        let mut t = Tree3::<C3>::new(aabb, IL);
        let mut refs = Vec::with_capacity(N); for it in &its { refs.push(t.insert_ref(*it).unwrap()); }
        let mut pos: Vec<Point3> = its.iter().map(|i| i.p).collect();
        let mut vel = vels();
        out.push(("update_ref_frame", bench(|| { for id in 0..N { let np = step(&mut pos[id], &mut vel[id]); t.update_ref(refs[id], |c| c.p = np); } 0 })));
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let save = argv.iter().any(|a| a == "--save");
    let threshold = argv.iter().position(|a| a == "--threshold").and_then(|i| argv.get(i + 1)).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.25);

    println!("regression gate | N={N} | item_limit={IL} | vision r={VISION} | {N_QUERY} culls | min of {REPS} reps");
    let calib = calibrate();
    let results = measure();

    if save {
        let mut s = String::from("# vectorial-hash regression baseline — op<TAB>nanoseconds (min-of-N).\n");
        s.push_str("# _calib is a fixed CPU loop; the gate compares op/_calib ratios to cancel clock scaling. Regenerate with --save on a quiet machine.\n");
        s.push_str(&format!("_calib\t{calib:.0}\n"));
        for (name, ns) in &results { s.push_str(&format!("{name}\t{ns:.0}\n")); }
        std::fs::write(BASELINE, &s).unwrap_or_else(|e| panic!("cannot write {BASELINE}: {e}"));
        println!("\nsaved baseline -> {BASELINE}");
        println!("  {:<24} {:>12.0} ns (calibration yardstick)", "_calib", calib);
        for (name, ns) in &results { println!("  {name:<24} {:>12.0} ns", ns); }
        return;
    }

    let base = match std::fs::read_to_string(BASELINE) {
        Ok(s) => s,
        Err(_) => { eprintln!("no baseline at {BASELINE} — run with --save first."); std::process::exit(2); }
    };
    let mut baseline = std::collections::HashMap::new();
    for line in base.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let mut it = line.split('\t');
        if let (Some(k), Some(v)) = (it.next(), it.next()) { if let Ok(ns) = v.parse::<f64>() { baseline.insert(k.to_string(), ns); } }
    }

    // Normalise current op times by how much the CPU itself shifted since the
    // baseline run: scale > 1 ⟹ machine is slower now ⟹ shrink current numbers.
    let base_calib = baseline.get("_calib").copied().unwrap_or(calib);
    let scale = base_calib / calib;
    println!("calibration: baseline {base_calib:.0} ns vs current {calib:.0} ns → scale ×{scale:.3} (clock-normalised)\n");
    println!("{:<24} {:>12} {:>12} {:>9}  verdict", "op", "baseline ns", "norm. ns", "delta");
    let mut regressed = Vec::new();
    let mut missing = false;
    for (name, ns) in &results {
        let cur = ns * scale;
        match baseline.get(*name) {
            Some(&b) => {
                let delta = (cur - b) / b;
                let tag = if delta > threshold { regressed.push((*name, delta)); "REGRESSED" }
                    else if delta < -0.10 { "improved" } else { "ok" };
                println!("{name:<24} {b:>12.0} {cur:>12.0} {:>+8.1}%  {tag}", delta * 100.0);
            }
            None => { missing = true; println!("{name:<24} {:>12} {cur:>12.0}      new  (not in baseline)", "—"); }
        }
    }

    let pct = (threshold * 100.0) as i64;
    if !regressed.is_empty() {
        println!("\nFAIL: {} op(s) regressed beyond +{pct}%:", regressed.len());
        for (n, d) in &regressed { println!("  {n}: {:+.1}%", d * 100.0); }
        std::process::exit(1);
    }
    if missing { println!("\nNOTE: new ops not yet in the baseline — re-run with --save to record them."); }
    println!("\nPASS: no op regressed beyond +{pct}%.");
}
