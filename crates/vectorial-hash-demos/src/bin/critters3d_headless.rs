//! 3D critters, headless — the dynamic 3D workload, independent of the 2D
//! critters. N points move in a cube; every frame each one is relocated in
//! the index (`Tree3::update`, ascend-to-LCA) and a fraction run a
//! sphere "vision" cull. Reports per-frame update and per-cull timings.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin critters3d_headless --release -- \
//!     --pop 20000 --item-limit 8 --frames 240 --warmup 60 --vision 60 --seed 42
//! ```
//!
//! This is the 3D analogue of `critters_headless`'s movement+cull loop
//! (no combat/kills — just the index workload that the structure choice
//! affects). It validates `Tree3`'s dynamic path under churn and measures
//! update/cull cost at 3D scale.

use std::time::Instant;

use vectorial_hash::{Aabb, Point3, Positioned3, Sphere3, Tree3};

const WORLD: f64 = 512.0;
const MARGIN: f64 = 4.0;
const VISION_R: f64 = 36.0;

struct Args {
    pop: usize,
    item_limit: usize,
    frames: usize,
    warmup: usize,
    vision: usize, // how many critters run a vision cull per frame
    dt: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args { pop: 20000, item_limit: 8, frames: 240, warmup: 60, vision: 60, dt: 1.0 / 60.0, seed: 42 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
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
    println!("3D critters | pop={} | item_limit={} | frames={} (+{} warmup) | vision/frame={} | world={}^3 | seed={}",
        args.pop, args.item_limit, args.frames, args.warmup, args.vision, WORLD, args.seed);

    let mut rng = Rng::new(args.seed);

    // State: position + velocity per critter (id == index).
    let mut pos: Vec<Point3> = Vec::with_capacity(args.pop);
    let mut vel: Vec<(f64, f64, f64)> = Vec::with_capacity(args.pop);
    let mut tree = Tree3::<C3>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), args.item_limit);
    for id in 0..args.pop {
        let p = Point3::new(rng.range(MARGIN, WORLD - MARGIN), rng.range(MARGIN, WORLD - MARGIN), rng.range(MARGIN, WORLD - MARGIN));
        let speed = rng.range(40.0, 120.0);
        let (a, b) = (rng.range(0.0, std::f64::consts::TAU), rng.range(-1.0, 1.0));
        let s = (1.0_f64 - b * b).max(0.0).sqrt();
        let v = (speed * s * a.cos(), speed * s * a.sin(), speed * b);
        pos.push(p);
        vel.push(v);
        tree.insert(C3 { id: id as u32, p });
    }

    let mut mv = Series::new();
    let mut vis = Series::new();

    let total_frames = args.warmup + args.frames;
    for frame in 0..total_frames {
        // Movement: integrate, bounce off the cube walls, relocate.
        let t = Instant::now();
        for id in 0..args.pop {
            let old = pos[id];
            let (mut vx, mut vy, mut vz) = vel[id];
            let mut nx = old.x + vx * args.dt;
            let mut ny = old.y + vy * args.dt;
            let mut nz = old.z + vz * args.dt;
            if nx < MARGIN || nx > WORLD - MARGIN { vx = -vx; nx = nx.clamp(MARGIN, WORLD - MARGIN); }
            if ny < MARGIN || ny > WORLD - MARGIN { vy = -vy; ny = ny.clamp(MARGIN, WORLD - MARGIN); }
            if nz < MARGIN || nz > WORLD - MARGIN { vz = -vz; nz = nz.clamp(MARGIN, WORLD - MARGIN); }
            vel[id] = (vx, vy, vz);
            let np = Point3::new(nx, ny, nz);
            let cid = id as u32;
            tree.update(old, |c| c.id == cid, |c| c.p = np);
            pos[id] = np;
        }
        let mv_us = t.elapsed().as_secs_f64() * 1e6;

        // Vision: a sample of critters run a sphere cull around themselves.
        let t = Instant::now();
        let mut vn = 0u32;
        for k in 0..args.vision {
            let id = (rng.next() as usize) % args.pop;
            let c = pos[id];
            let sphere = Sphere3::new(c.x, c.y, c.z, VISION_R);
            let hits = tree.cull(&sphere);
            // touch the result so it isn't optimized away
            if hits.len() == usize::MAX { println!("unreachable"); }
            vn += 1;
            let _ = k;
        }
        let vis_us = t.elapsed().as_secs_f64() * 1e6;

        if frame >= args.warmup {
            mv.push(mv_us);
            if vn > 0 { vis.push(vis_us / vn as f64); }
        }
    }

    println!("\nstructure: {} leaves, {} arena nodes, {} items",
        tree.leaf_count(), tree.node_count(), tree.item_count());
    println!("\n{:<24} {:>12} {:>12}", "op", "mean", "p95");
    println!("{:<24} {:>10.1}us {:>10.1}us", "move+update (per frame)", mv.mean(), mv.p95());
    println!("{:<24} {:>10.2}us {:>10.2}us", "vision cull (per cull)", vis.mean(), vis.p95());
}
