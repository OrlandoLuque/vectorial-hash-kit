//! Head-to-head: `Tree<T>` (f64) vs `IntegerTree<T>` (i32, pow2 extent).
//!
//! Pure movement workload. Each `step`, every item drifts a few units;
//! that's it. No cull, no spawns, no removals — we just measure how fast
//! the relocation path is on each tree.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin itree_bench --release -- \
//!     --pop 10000 --item-limit 3 --steps 240 --warmup 60 --seed 42
//! ```
//!
//! Both trees are populated with the SAME items at the same positions, and
//! each step's per-item motion vector is identical for both. The only
//! difference is the underlying tree representation.

use std::time::Instant;

use vectorial_hash::{
    IntegerTree, IPoint, IPositioned, IRect, Point, Positioned, Rect, Tree, UpdateStrategy,
    IUpdateStrategy,
};

// World is 1024 × 1024 (pow2) so the integer tree's invariant holds.
const WORLD: i32 = 1024;

struct Args {
    pop: usize,
    item_limit: usize,
    steps: usize,
    warmup: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args { pop: 10000, item_limit: 3, steps: 240, warmup: 60, seed: 42 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--pop" => a.pop = val().parse().unwrap(),
            "--item-limit" => a.item_limit = val().parse().unwrap(),
            "--steps" => a.steps = val().parse().unwrap(),
            "--warmup" => a.warmup = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            other => panic!("unknown argument: {other}"),
        }
        i += 2;
    }
    a
}

// ---------------------------------------------------------------- xorshift
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

// ---------------------------------------------------------------- items
#[derive(Clone, Copy)]
struct FItem { id: u32, pos: Point }
impl Positioned for FItem { fn position(&self) -> Point { self.pos } }

#[derive(Clone, Copy)]
struct IItem { id: u32, pos: IPoint }
impl IPositioned for IItem { fn position(&self) -> IPoint { self.pos } }

// ---------------------------------------------------------------- stats
#[derive(Default)]
struct Series(Vec<f64>);
impl Series {
    fn push(&mut self, v: f64) { self.0.push(v); }
    fn mean(&self) -> f64 {
        if self.0.is_empty() { 0.0 } else { self.0.iter().sum::<f64>() / self.0.len() as f64 }
    }
    fn pct(&self, p: f64) -> f64 {
        if self.0.is_empty() { return 0.0; }
        let mut v = self.0.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * p).round() as usize]
    }
}

fn main() {
    let args = parse_args();
    println!("itree bench | pop={} | item_limit={} | steps={} (+{} warmup) | seed={}",
        args.pop, args.item_limit, args.steps, args.warmup, args.seed);

    // 1. Seed initial positions and motion deltas; share them between trees.
    let mut rng = Rng::new(args.seed);
    let mut initial_x = Vec::with_capacity(args.pop);
    let mut initial_y = Vec::with_capacity(args.pop);
    for _ in 0..args.pop {
        // Stay 4 px away from the boundary so float and int both stay in
        // bounds; world extent is 1024.
        initial_x.push(rng.range(4.0, (WORLD - 4) as f64));
        initial_y.push(rng.range(4.0, (WORLD - 4) as f64));
    }
    // Per-item heading + speed (px/step).
    let mut heading: Vec<f64> = (0..args.pop).map(|_| rng.range(0.0, std::f64::consts::TAU)).collect();
    let speeds: Vec<f64> = (0..args.pop).map(|_| rng.range(0.5, 1.5)).collect();
    let turn: Vec<f64> = (0..args.pop).map(|_| rng.range(-0.05, 0.05)).collect();

    // 2. Build both trees with the same initial population.
    let mut ftree = Tree::<FItem>::with_limits(
        Rect::new(0.0, 0.0, WORLD as f64, WORLD as f64),
        args.item_limit, args.item_limit);
    let mut itree = IntegerTree::<IItem>::with_limits(
        IRect::new(0, 0, WORLD, WORLD), args.item_limit, args.item_limit);
    for i in 0..args.pop {
        let pos_f = Point::new(initial_x[i], initial_y[i]);
        let pos_i = IPoint::new(initial_x[i].round() as i32, initial_y[i].round() as i32);
        ftree.insert(FItem { id: i as u32, pos: pos_f });
        itree.insert(IItem { id: i as u32, pos: pos_i });
    }

    // 3. Step both for `warmup + steps`. During warmup, no measurement.
    //    During measured steps, time each tree's relocation pass separately.
    let mut series_f = Series::default();
    let mut series_i = Series::default();
    let mut last_x = initial_x.clone();
    let mut last_y = initial_y.clone();

    // Pre-allocate scratch vectors so per-step allocation isn't in the
    // timed sections.
    let mut next_x = vec![0.0; args.pop];
    let mut next_y = vec![0.0; args.pop];
    let mut old_fpts: Vec<Point> = vec![Point::new(0.0, 0.0); args.pop];
    let mut new_fpts: Vec<Point> = vec![Point::new(0.0, 0.0); args.pop];
    let mut old_ipts: Vec<IPoint> = vec![IPoint::new(0, 0); args.pop];
    let mut new_ipts: Vec<IPoint> = vec![IPoint::new(0, 0); args.pop];

    for step in 0..(args.warmup + args.steps) {
        // Compute next positions and prepare both representations OUTSIDE
        // the timed sections.
        for i in 0..args.pop {
            heading[i] += turn[i];
            let mut nx = last_x[i] + heading[i].cos() * speeds[i];
            let mut ny = last_y[i] + heading[i].sin() * speeds[i];
            if nx < 4.0 { nx = 4.0; heading[i] = std::f64::consts::PI - heading[i]; }
            if nx > (WORLD - 4) as f64 { nx = (WORLD - 4) as f64; heading[i] = std::f64::consts::PI - heading[i]; }
            if ny < 4.0 { ny = 4.0; heading[i] = -heading[i]; }
            if ny > (WORLD - 4) as f64 { ny = (WORLD - 4) as f64; heading[i] = -heading[i]; }
            next_x[i] = nx;
            next_y[i] = ny;
            old_fpts[i] = Point::new(last_x[i], last_y[i]);
            new_fpts[i] = Point::new(nx, ny);
            old_ipts[i] = IPoint::new(last_x[i].round() as i32, last_y[i].round() as i32);
            new_ipts[i] = IPoint::new(nx.round() as i32, ny.round() as i32);
        }

        // Float tree pass.
        let t0 = Instant::now();
        for i in 0..args.pop {
            let id = i as u32;
            let np = new_fpts[i];
            ftree.update_with(UpdateStrategy::Lca, old_fpts[i], |c| c.id == id, |c| c.pos = np);
        }
        let f_us = t0.elapsed().as_secs_f64() * 1e6;

        // Int tree pass.
        let t0 = Instant::now();
        for i in 0..args.pop {
            let id = i as u32;
            let np = new_ipts[i];
            itree.update_with(IUpdateStrategy::Lca, old_ipts[i], |c| c.id == id, |c| c.pos = np);
        }
        let i_us = t0.elapsed().as_secs_f64() * 1e6;

        if step >= args.warmup {
            series_f.push(f_us);
            series_i.push(i_us);
        }
        for i in 0..args.pop {
            last_x[i] = next_x[i];
            last_y[i] = next_y[i];
        }
    }

    println!("\nfloat tree  : {} leaves, {} arena nodes", ftree.leaf_count(), ftree.node_count());
    println!("int   tree  : {} leaves, {} arena nodes", itree.leaf_count(), itree.node_count());

    println!("\n{:<14} {:>10} {:>10} {:>10}", "tree", "mean", "p50", "p95");
    println!("{:<14} {:>10.1} {:>10.1} {:>10.1}",
        "Tree<T>",        series_f.mean(), series_f.pct(0.5), series_f.pct(0.95));
    println!("{:<14} {:>10.1} {:>10.1} {:>10.1}",
        "IntegerTree<T>", series_i.mean(), series_i.pct(0.5), series_i.pct(0.95));

    let delta = (series_i.mean() - series_f.mean()) / series_f.mean() * 100.0;
    println!("\nIntegerTree vs Tree: {:+.1}% on mean", delta);
}
