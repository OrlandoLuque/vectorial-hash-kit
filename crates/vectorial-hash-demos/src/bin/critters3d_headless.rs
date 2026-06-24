//! 3D critters, headless — the dynamic 3D workload, independent of the 2D
//! critters. N points move in a cube; every frame each one is relocated in
//! the index (ascend-to-LCA `update`) and a fraction run a sphere "vision"
//! cull. Reports per-frame update and per-cull timings.
//!
//! Runs the **same deterministic simulation** against the binary `Tree3` and
//! the 8-way `Octree3` so the *dynamic* octree can be measured head-to-head
//! against the binary tree (the 3D analogue of the 2D update-strategy study).
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin critters3d_headless --release -- \
//!     --structure both --pop 20000 --item-limit 8 --frames 240 --warmup 60 --vision 60 --seed 42
//! ```
//!
//! `--structure binary|octree|both` (default `both`). In `both` the positions
//! are integrated once and applied to both indexes, both are timed separately,
//! and their vision culls are cross-checked for agreement. Running `binary`
//! and `octree` in separate processes gives cleaner cache-isolated numbers;
//! `both` gives a same-run apples-to-apples plus the agreement gate.

use std::time::Instant;

use vectorial_hash::{Aabb, Octree3, Point3, Positioned3, Sphere3, Tree3};

const WORLD: f64 = 512.0;
const MARGIN: f64 = 4.0;
const VISION_R: f64 = 36.0;

#[derive(Clone, Copy, PartialEq)]
enum Mode { Binary, Octree, Both }
impl Mode {
    fn has_binary(self) -> bool { self != Mode::Octree }
    fn has_octree(self) -> bool { self != Mode::Binary }
    fn label(self) -> &'static str {
        match self { Mode::Binary => "binary", Mode::Octree => "octree", Mode::Both => "both" }
    }
}

struct Args {
    structure: Mode,
    pop: usize,
    item_limit: usize,
    frames: usize,
    warmup: usize,
    vision: usize, // how many critters run a vision cull per frame
    dt: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args { structure: Mode::Both, pop: 20000, item_limit: 8, frames: 240, warmup: 60, vision: 60, dt: 1.0 / 60.0, seed: 42 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--structure" => a.structure = match val().as_str() {
                "binary" => Mode::Binary,
                "octree" => Mode::Octree,
                "both" => Mode::Both,
                other => panic!("unknown structure: {other} (binary|octree|both)"),
            },
            "--pop" => a.pop = val().parse().unwrap(),
            "--item-limit" => a.item_limit = val().parse().unwrap(),
            "--frames" => a.frames = val().parse().unwrap(),
            "--warmup" => a.warmup = val().parse().unwrap(),
            "--vision" => a.vision = val().parse().unwrap(),
            "--dt" => a.dt = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
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
struct C3 { id: u32, p: Point3 }
impl Positioned3 for C3 { fn position(&self) -> Point3 { self.p } }

struct Series(Vec<f64>);
impl Series {
    fn new() -> Self { Series(Vec::new()) }
    fn push(&mut self, v: f64) { self.0.push(v); }
    fn mean(&self) -> f64 { if self.0.is_empty() { 0.0 } else { self.0.iter().sum::<f64>() / self.0.len() as f64 } }
    fn p95(&self) -> f64 {
        if self.0.is_empty() { return 0.0; }
        let mut v = self.0.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * 0.95).round() as usize]
    }
}

fn main() {
    let args = parse_args();
    println!("3D critters | structure={} | pop={} | item_limit={} | frames={} (+{} warmup) | vision/frame={} | world={}^3 | seed={}",
        args.structure.label(), args.pop, args.item_limit, args.frames, args.warmup, args.vision, WORLD, args.seed);

    let mut rng = Rng::new(args.seed);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);

    // State: position + velocity per critter (id == index). The simulation is
    // structure-independent — positions are integrated once per frame and then
    // applied to whichever index(es) are active.
    let mut pos: Vec<Point3> = Vec::with_capacity(args.pop);
    let mut vel: Vec<(f64, f64, f64)> = Vec::with_capacity(args.pop);
    let mut old_pos: Vec<Point3> = vec![Point3::new(0.0, 0.0, 0.0); args.pop];
    let mut tree = Tree3::<C3>::new(world, args.item_limit);
    let mut octree = Octree3::<C3>::new(world, args.item_limit);
    for id in 0..args.pop {
        let p = Point3::new(rng.range(MARGIN, WORLD - MARGIN), rng.range(MARGIN, WORLD - MARGIN), rng.range(MARGIN, WORLD - MARGIN));
        let speed = rng.range(40.0, 120.0);
        let (a, b) = (rng.range(0.0, std::f64::consts::TAU), rng.range(-1.0, 1.0));
        let s = (1.0_f64 - b * b).max(0.0).sqrt();
        let v = (speed * s * a.cos(), speed * s * a.sin(), speed * b);
        pos.push(p);
        vel.push(v);
        if args.structure.has_binary() { tree.insert(C3 { id: id as u32, p }); }
        if args.structure.has_octree() { octree.insert(C3 { id: id as u32, p }); }
    }

    let mut mv_bin = Series::new();
    let mut mv_oct = Series::new();
    let mut vis_bin = Series::new();
    let mut vis_oct = Series::new();
    let mut blackhole = 0usize; // keep cull results from being optimized away
    let mut culls_checked = 0u64;
    let mut disagreements = 0u64;

    let total_frames = args.warmup + args.frames;
    for frame in 0..total_frames {
        // Integrate + bounce off the cube walls (once, structure-independent).
        for id in 0..args.pop {
            old_pos[id] = pos[id];
            let (mut vx, mut vy, mut vz) = vel[id];
            let mut nx = pos[id].x + vx * args.dt;
            let mut ny = pos[id].y + vy * args.dt;
            let mut nz = pos[id].z + vz * args.dt;
            if nx < MARGIN || nx > WORLD - MARGIN { vx = -vx; nx = nx.clamp(MARGIN, WORLD - MARGIN); }
            if ny < MARGIN || ny > WORLD - MARGIN { vy = -vy; ny = ny.clamp(MARGIN, WORLD - MARGIN); }
            if nz < MARGIN || nz > WORLD - MARGIN { vz = -vz; nz = nz.clamp(MARGIN, WORLD - MARGIN); }
            vel[id] = (vx, vy, vz);
            pos[id] = Point3::new(nx, ny, nz);
        }

        // Relocate in each active index (timed separately).
        if args.structure.has_binary() {
            let t = Instant::now();
            for id in 0..args.pop {
                let cid = id as u32;
                tree.update(old_pos[id], |c| c.id == cid, |c| c.p = pos[id]);
            }
            if frame >= args.warmup { mv_bin.push(t.elapsed().as_secs_f64() * 1e6); }
        }
        if args.structure.has_octree() {
            let t = Instant::now();
            for id in 0..args.pop {
                let cid = id as u32;
                octree.update(old_pos[id], |c| c.id == cid, |c| c.p = pos[id]);
            }
            if frame >= args.warmup { mv_oct.push(t.elapsed().as_secs_f64() * 1e6); }
        }

        // Vision: a sample of critters run a sphere cull around themselves.
        // Sample the same ids for both structures so the timing is comparable
        // and the results can be cross-checked.
        let ids: Vec<usize> = (0..args.vision).map(|_| (rng.next() as usize) % args.pop).collect();
        if args.structure.has_binary() {
            let t = Instant::now();
            for &id in &ids {
                let c = pos[id];
                let hits = tree.cull(&Sphere3::new(c.x, c.y, c.z, VISION_R));
                blackhole = blackhole.wrapping_add(hits.len());
            }
            if frame >= args.warmup && !ids.is_empty() { vis_bin.push(t.elapsed().as_secs_f64() * 1e6 / ids.len() as f64); }
        }
        if args.structure.has_octree() {
            let t = Instant::now();
            for &id in &ids {
                let c = pos[id];
                let hits = octree.cull(&Sphere3::new(c.x, c.y, c.z, VISION_R));
                blackhole = blackhole.wrapping_add(hits.len());
            }
            if frame >= args.warmup && !ids.is_empty() { vis_oct.push(t.elapsed().as_secs_f64() * 1e6 / ids.len() as f64); }
        }

        // Agreement gate (both mode only, untimed): the two indexes must return
        // identical id sets for every sampled sphere.
        if args.structure == Mode::Both && frame >= args.warmup {
            for &id in &ids {
                let c = pos[id];
                let s = Sphere3::new(c.x, c.y, c.z, VISION_R);
                let mut gb: Vec<u32> = tree.cull(&s).iter().map(|c| c.id).collect();
                let mut go: Vec<u32> = octree.cull(&s).iter().map(|c| c.id).collect();
                gb.sort_unstable(); go.sort_unstable();
                culls_checked += 1;
                if gb != go { disagreements += 1; }
            }
        }
    }

    if blackhole == usize::MAX { println!("unreachable"); } // touch the accumulator

    println!("\nstructure stats:");
    if args.structure.has_binary() {
        println!("  binary Tree3 : {:>7} leaves, {:>7} arena nodes, {:>7} items", tree.leaf_count(), tree.node_count(), tree.item_count());
    }
    if args.structure.has_octree() {
        println!("  octree3 (8w) : {:>7} leaves, {:>7} arena nodes, {:>7} items", octree.leaf_count(), octree.node_count(), octree.item_count());
    }

    println!("\n{:<28} {:>22} {:>22}", "op", "binary (mean / p95)", "octree (mean / p95)");
    let row = |label: &str, b: &Series, o: &Series, has_b: bool, has_o: bool, scale: &str| {
        let cell = |s: &Series, on: bool| if on { format!("{:>8.1} / {:>8.1}{}", s.mean(), s.p95(), scale) } else { format!("{:>20}", "-") };
        println!("{:<28} {:>22} {:>22}", label, cell(b, has_b), cell(o, has_o));
    };
    row("move+update (per frame)", &mv_bin, &mv_oct, args.structure.has_binary(), args.structure.has_octree(), "us");
    row("vision cull (per cull)", &vis_bin, &vis_oct, args.structure.has_binary(), args.structure.has_octree(), "us");

    if args.structure == Mode::Both {
        println!("\ncull agreement: {} mismatches over {} sampled culls{}",
            disagreements, culls_checked,
            if disagreements == 0 { " (binary == octree)" } else { "  <-- DISAGREEMENT" });
    }
}
