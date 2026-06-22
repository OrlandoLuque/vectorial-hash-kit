//! 3D culling: true-3D-tree vs projection-indexing (3 × 2D trees), on
//! time AND precision, against a brute-force ground truth.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin tree3d_bench --release -- \
//!     --pop 50000 --item-limit 8 --queries 200 --seed 42
//! ```
//!
//! Two ways to answer "which 3D points lie inside this sphere?":
//!
//! 1. **True 3D tree** (`Tree3`): binary split in 3D, sphere classified
//!    against each node box (green/white/yellow), 1×1×1 voxel raster at
//!    leaves. Exact, but a real 3D template bank would be N³ in memory
//!    (here the sphere is analytic, so no bank is needed — best case for
//!    the 3D tree).
//! 2. **Projection indexing** (author's idea): three 2D trees on the (x,y),
//!    (x,z), (y,z) projections. Cull each with the sphere's circular
//!    shadow, intersect the candidate id sets, then run the exact 3D test
//!    on survivors. Reuses the 2D machinery, no N³ memory — but the
//!    intersection is a *broadphase* (a superset; the corners of the
//!    three-cylinder intersection that stick out of the sphere are false
//!    positives that the exact test drops).
//!
//! Reports: ns/query for each, and the projection's false-positive ratio
//! (candidates after intersection ÷ true hits) which drives its exact-test
//! cost. Correctness of both is gated against brute force.

use std::collections::HashSet;
use std::time::Instant;

use vectorial_hash::{
    Aabb, Point, Point3, Positioned, Positioned3, Rect, Sphere3, Tree, Tree3,
};

const WORLD: f64 = 512.0;

struct Args {
    pop: usize,
    item_limit: usize,
    queries: usize,
    seed: u64,
    rmin: f64,
    rmax: f64,
}

fn parse_args() -> Args {
    let mut a = Args { pop: 50000, item_limit: 8, queries: 200, seed: 42, rmin: 10.0, rmax: 80.0 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--pop" => a.pop = val().parse().unwrap(),
            "--item-limit" => a.item_limit = val().parse().unwrap(),
            "--queries" => a.queries = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            "--rmin" => a.rmin = val().parse().unwrap(),
            "--rmax" => a.rmax = val().parse().unwrap(),
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

// 3D item.
#[derive(Clone, Copy)]
struct I3 { id: u32, p: Point3 }
impl Positioned3 for I3 { fn position(&self) -> Point3 { self.p } }

// 2D projection item (carries the id so we can intersect across planes).
#[derive(Clone, Copy)]
struct I2 { id: u32, p: Point }
impl Positioned for I2 { fn position(&self) -> Point { self.p } }

// 2D circle (the sphere's shadow). No template — bbox reject + exact test,
// i.e. the projection approach reusing the 2D index out of the box.
struct Circle2 { cx: f64, cy: f64, r: f64 }
impl vectorial_hash::Shape for Circle2 {
    fn bounding_box(&self) -> Rect {
        Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r)
    }
    fn contains_point(&self, p: Point) -> bool {
        let dx = p.x - self.cx; let dy = p.y - self.cy;
        dx * dx + dy * dy <= self.r * self.r
    }
}

struct Stats { ns: Vec<f64> }
impl Stats {
    fn new() -> Self { Stats { ns: Vec::new() } }
    fn push(&mut self, v: f64) { self.ns.push(v); }
    fn mean(&self) -> f64 { if self.ns.is_empty() { 0.0 } else { self.ns.iter().sum::<f64>() / self.ns.len() as f64 } }
    fn p95(&self) -> f64 {
        if self.ns.is_empty() { return 0.0; }
        let mut v = self.ns.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * 0.95).round() as usize]
    }
}

fn main() {
    let args = parse_args();
    println!("tree3d bench | pop={} | item_limit={} | queries={} | world={}^3 | seed={}",
        args.pop, args.item_limit, args.queries, WORLD, args.seed);

    let mut rng = Rng::new(args.seed);
    let mut items: Vec<I3> = Vec::with_capacity(args.pop);
    let mut pos: Vec<Point3> = Vec::with_capacity(args.pop);
    for id in 0..args.pop {
        let p = Point3::new(rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0));
        items.push(I3 { id: id as u32, p });
        pos.push(p);
    }

    // --- build the structures ---
    let t_build3 = Instant::now();
    let mut tree3 = Tree3::<I3>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), args.item_limit);
    for it in &items { tree3.insert(*it); }
    let build3_ms = t_build3.elapsed().as_secs_f64() * 1e3;

    let t_buildp = Instant::now();
    let mut tree_xy = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    let mut tree_xz = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    let mut tree_yz = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    for it in &items {
        tree_xy.insert(I2 { id: it.id, p: Point::new(it.p.x, it.p.y) });
        tree_xz.insert(I2 { id: it.id, p: Point::new(it.p.x, it.p.z) });
        tree_yz.insert(I2 { id: it.id, p: Point::new(it.p.y, it.p.z) });
    }
    let buildp_ms = t_buildp.elapsed().as_secs_f64() * 1e3;

    println!("build: 3D tree {:.1} ms ({} nodes) | 3×2D trees {:.1} ms ({}+{}+{} nodes)",
        build3_ms, tree3.node_count(), buildp_ms,
        tree_xy.node_count(), tree_xz.node_count(), tree_yz.node_count());

    // --- queries ---
    let mut s_brute = Stats::new();
    let mut s_tree3 = Stats::new();
    let mut s_proj = Stats::new();
    let mut s_proj_broad = Stats::new(); // projection broadphase only (before exact filter)
    let mut s_proj1 = Stats::new();      // single-projection + exact filter
    let mut total_true = 0u64;
    let mut total_cand = 0u64;
    let mut total_cand1 = 0u64;
    let mut mismatches3 = 0u64;
    let mut mismatchesp = 0u64;
    let mut mismatchesp1 = 0u64;

    for q in 0..args.queries {
        // Radius spread so some queries are small, some large.
        let r = rng.range(args.rmin, args.rmax);
        let cx = rng.range(r, WORLD - r);
        let cy = rng.range(r, WORLD - r);
        let cz = rng.range(r, WORLD - r);

        // Ground truth (brute force).
        let t = Instant::now();
        let mut brute: Vec<u32> = items.iter()
            .filter(|it| { let dx = it.p.x - cx; let dy = it.p.y - cy; let dz = it.p.z - cz; dx*dx+dy*dy+dz*dz <= r*r })
            .map(|it| it.id).collect();
        s_brute.push(t.elapsed().as_secs_f64() * 1e9);
        brute.sort();
        let brute_set: HashSet<u32> = brute.iter().copied().collect();
        total_true += brute.len() as u64;

        // True 3D tree.
        let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
        let t = Instant::now();
        let hits3: Vec<u32> = tree3.cull(&sphere).iter().map(|it| it.id).collect();
        s_tree3.push(t.elapsed().as_secs_f64() * 1e9);
        let set3: HashSet<u32> = hits3.iter().copied().collect();
        if set3 != brute_set { mismatches3 += 1; }

        // Projection: cull 3 circles, intersect, exact 3D filter.
        let t = Instant::now();
        let cull_xy = tree_xy.cull(&Circle2 { cx, cy, r });
        let cull_xz = tree_xz.cull(&Circle2 { cx: cx, cy: cz, r });
        let cull_yz = tree_yz.cull(&Circle2 { cx: cy, cy: cz, r });
        // Intersect the smallest against the others via id sets.
        let set_xz: HashSet<u32> = cull_xz.iter().map(|i| i.id).collect();
        let set_yz: HashSet<u32> = cull_yz.iter().map(|i| i.id).collect();
        let cand: Vec<u32> = cull_xy.iter().map(|i| i.id)
            .filter(|id| set_xz.contains(id) && set_yz.contains(id))
            .collect();
        let broad_ns = t.elapsed().as_secs_f64() * 1e9;
        s_proj_broad.push(broad_ns);
        // Exact 3D narrowphase on the candidates.
        let projhits: Vec<u32> = cand.iter().copied()
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj.push(t.elapsed().as_secs_f64() * 1e9);
        let setp: HashSet<u32> = projhits.iter().copied().collect();
        if setp != brute_set { mismatchesp += 1; }
        total_cand += cand.len() as u64;

        // Single-projection broadphase: cull ONE plane (xy), exact-filter
        // its shadow in 3D. Larger candidate set than the 3-way intersect,
        // but no extra culls/hashing — wins when the exact test is cheap.
        let t = Instant::now();
        let cand1 = tree_xy.cull(&Circle2 { cx, cy, r });
        let proj1hits: Vec<u32> = cand1.iter().map(|i| i.id)
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj1.push(t.elapsed().as_secs_f64() * 1e9);
        let setp1: HashSet<u32> = proj1hits.iter().copied().collect();
        if setp1 != brute_set { mismatchesp1 += 1; }
        total_cand1 += cand1.len() as u64;

        let _ = q;
    }

    let fp_ratio = total_cand as f64 / total_true.max(1) as f64;
    let fp_ratio1 = total_cand1 as f64 / total_true.max(1) as f64;
    println!("\nqueries={} | true hits total={} | mean hits/query={:.0}",
        args.queries, total_true, total_true as f64 / args.queries as f64);
    println!("broadphase candidate/true ratio: 3-projection {:.2}x | 1-projection {:.2}x",
        fp_ratio, fp_ratio1);
    println!("correctness vs brute: 3D tree {} | 3-projection {} | 1-projection {}",
        if mismatches3 == 0 { "EXACT" } else { "MISMATCH!" },
        if mismatchesp == 0 { "EXACT" } else { "MISMATCH!" },
        if mismatchesp1 == 0 { "EXACT" } else { "MISMATCH!" });

    println!("\n{:<28} {:>12} {:>12}", "method", "mean ns/q", "p95 ns/q");
    println!("{:<28} {:>12.0} {:>12.0}", "brute force", s_brute.mean(), s_brute.p95());
    println!("{:<28} {:>12.0} {:>12.0}", "true 3D tree", s_tree3.mean(), s_tree3.p95());
    println!("{:<28} {:>12.0} {:>12.0}", "3-projection (intersect+exact)", s_proj.mean(), s_proj.p95());
    println!("{:<28} {:>12.0} {:>12.0}", "  ...broadphase only", s_proj_broad.mean(), s_proj_broad.p95());
    println!("{:<28} {:>12.0} {:>12.0}", "1-projection (+exact)", s_proj1.mean(), s_proj1.p95());

    println!("\nspeedup vs brute: 3D tree {:.1}x | 3-proj {:.2}x | 1-proj {:.2}x",
        s_brute.mean() / s_tree3.mean().max(1e-9),
        s_brute.mean() / s_proj.mean().max(1e-9),
        s_brute.mean() / s_proj1.mean().max(1e-9));
}
