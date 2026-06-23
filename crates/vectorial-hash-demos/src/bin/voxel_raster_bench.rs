//! Where the 1×1×1 voxel raster finally pays: a many-faced convex
//! polyhedron whose `contains_point` is expensive (one dot product per
//! face). The raster turns each leaf per-point test into a single memory
//! lookup; analytic does N plane evaluations. As the face count grows the
//! raster crosses over from "loses to analytic" (cheap shape, like the
//! sphere) to "wins" (expensive shape).
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin voxel_raster_bench --release -- \
//!     --pop 100000 --item-limit 64 --queries 300 --seed 42
//! ```
//!
//! Both paths go through `Tree3::cull`; the ONLY difference is whether the
//! shape carries a `VoxelRaster` (lookup) or not (analytic `contains_point`
//! at leaves). Correctness of both is gated against brute force.

use std::time::Instant;

use vectorial_hash::{Aabb, Point3, Polyhedron3, Positioned3, Shape3, Tree3};

const WORLD: f64 = 512.0;

struct Args { pop: usize, item_limit: usize, queries: usize, seed: u64 }
fn parse_args() -> Args {
    let mut a = Args { pop: 100_000, item_limit: 64, queries: 300, seed: 42 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].as_str();
        let v = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {k}"));
        match k {
            "--pop" => a.pop = v().parse().unwrap(),
            "--item-limit" => a.item_limit = v().parse().unwrap(),
            "--queries" => a.queries = v().parse().unwrap(),
            "--seed" => a.seed = v().parse().unwrap(),
            other => panic!("unknown argument: {other}"),
        }
        i += 2;
    }
    a
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.max(1)) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct I3 { id: u32, p: Point3 }
impl Positioned3 for I3 { fn position(&self) -> Point3 { self.p } }

struct Stats(Vec<f64>);
impl Stats {
    fn new() -> Self { Stats(Vec::new()) }
    fn push(&mut self, v: f64) { self.0.push(v); }
    fn mean(&self) -> f64 { if self.0.is_empty() { 0.0 } else { self.0.iter().sum::<f64>() / self.0.len() as f64 } }
}

fn main() {
    let args = parse_args();
    println!("voxel raster bench | pop={} | item_limit={} | queries={} | world={}^3 | seed={}",
        args.pop, args.item_limit, args.queries, WORLD, args.seed);

    let mut rng = Rng::new(args.seed);
    let mut tree = Tree3::<I3>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), args.item_limit);
    let mut pos = Vec::with_capacity(args.pop);
    for id in 0..args.pop {
        let p = Point3::new(rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0));
        tree.insert(I3 { id: id as u32, p });
        pos.push(p);
    }
    println!("Tree3: {} leaves, {} arena nodes\n", tree.leaf_count(), tree.node_count());

    // The raster is precomputed once per query shape and reused (the realistic
    // model — like the 2D attack templates reused for every attack of that
    // shape). We amortise the build over REPS culls per query location.
    const REPS: usize = 20;
    println!("{:<8} {:>14} {:>14} {:>9} {:>13} {:>9}",
        "faces", "analytic ns/q", "raster ns/q", "speedup", "raster build ns", "correct");
    for &faces in &[8usize, 24, 48, 96, 192] {
        let mut s_analytic = Stats::new();
        let mut s_raster = Stats::new();
        let mut s_build = Stats::new();
        let mut ok = true;
        let mut qrng = Rng::new(args.seed ^ 0x9E3779B9);
        for _ in 0..args.queries {
            let r = qrng.range(18.0, 40.0);
            let (cx, cy, cz) = (qrng.range(r, WORLD - r), qrng.range(r, WORLD - r), qrng.range(r, WORLD - r));

            let poly_a = Polyhedron3::faceted_ball(cx, cy, cz, r, faces);
            let mut brute: Vec<u32> = pos.iter().enumerate()
                .filter(|(_, p)| poly_a.contains_point(**p)).map(|(i, _)| i as u32).collect();
            brute.sort();

            // analytic narrowphase (no raster) — averaged over REPS.
            let t = Instant::now();
            let mut a = Vec::new();
            for _ in 0..REPS { a = tree.cull(&poly_a).iter().map(|i| i.id).collect(); }
            s_analytic.push(t.elapsed().as_secs_f64() * 1e9 / REPS as f64);
            a.sort();

            // raster: build once (timed separately), then REPS lookups-cull.
            let t = Instant::now();
            let poly_r = Polyhedron3::faceted_ball(cx, cy, cz, r, faces).with_raster();
            s_build.push(t.elapsed().as_secs_f64() * 1e9);
            let t = Instant::now();
            let mut b = Vec::new();
            for _ in 0..REPS { b = tree.cull(&poly_r).iter().map(|i| i.id).collect(); }
            s_raster.push(t.elapsed().as_secs_f64() * 1e9 / REPS as f64);
            b.sort();

            if a != brute || b != brute { ok = false; }
        }
        let speedup = s_analytic.mean() / s_raster.mean().max(1e-9);
        println!("{:<8} {:>14.0} {:>14.0} {:>8.2}x {:>13.0} {:>9}",
            faces, s_analytic.mean(), s_raster.mean(), speedup, s_build.mean(),
            if ok { "EXACT" } else { "MISMATCH!" });
    }
    println!("\nRaster cull = a memory lookup per leaf item (In/Out) + exact only on\nMaybe voxels; analytic = N plane evals per item. The raster wins once the\nface count makes contains_point expensive — and its build is amortised\nacross the REPS culls of one precomputed shape (here {REPS}).");
}
