//! Where does a linear scan stop winning? — resolving `brute_max`.
//!
//! `Thresholds::brute_max` ships at 512 (from `advisor::BRUTE_FORCE_MAX`) while the calibration
//! tool measures 182, and the two have disagreed for as long as both existed. The gap matters
//! because this threshold is an **unconditional floor**: below it the policy returns
//! `Backend::Brute` whatever else is true, before the query-load rule (`scan_budget`) is even
//! consulted.
//!
//! That is what decides the right value. A floor that fires regardless of load must be set by
//! the case *least* favourable to a scan — the heaviest query load — because that is where an
//! index would otherwise have been chosen. Set it by an average load and it will keep a scan on
//! workloads an index would have won; set it too low and it merely defers to `scan_budget`,
//! which is load-aware and can decide properly.
//!
//! So this sweeps population against query load and reports the crossover for each, all through
//! `AdaptiveIndex` with pinned thresholds so the call path, item storage and allocation are
//! identical and only the backend differs.
//!
//! ```bash
//! cargo run -p vectorial-hash --example brute_edge --release
//! ```
use std::time::Instant;
use vectorial_hash::{AdaptiveIndex, Aabb, Backend, Point3, Positioned3, Slot, Sphere3, Thresholds};

const W: f64 = 500.0;
const FRAMES: usize = 300;
const REPS: usize = 5;

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

fn pin(b: Backend) -> Thresholds {
    match b {
        Backend::Brute => Thresholds { brute_max: usize::MAX, ..Default::default() },
        Backend::KeepTree => Thresholds { brute_max: 0, scan_budget: 0.0, rebuild_query_ratio: f64::MAX, static_ticks: u32::MAX, ..Default::default() },
        Backend::Grid => Thresholds { brute_max: 0, scan_budget: 0.0, rebuild_query_ratio: 0.0, static_ticks: u32::MAX, ..Default::default() },
        Backend::Static => Thresholds { brute_max: 0, scan_budget: 0.0, static_ticks: 0, ..Default::default() },
    }
}

/// One run: `n` items, a quarter of them moving each frame, `culls` queries per frame.
fn run(n: usize, culls: usize, backend: Backend) -> f64 {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    let mut best = f64::MAX;
    for rep in 0..REPS {
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::with_thresholds(world, 16, pin(backend));
        let mut rng = Rng(0x5EED_0001 ^ (rep as u64) << 32 ^ n as u64);
        let mut slots: Vec<Slot> = Vec::with_capacity(n);
        let mut pos: Vec<Point3> = Vec::with_capacity(n);
        for _ in 0..n {
            let p = Point3::new(rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1));
            slots.push(ix.insert(P { p }));
            pos.push(p);
        }
        let mut sink = 0usize;
        let t = Instant::now();
        for _ in 0..FRAMES {
            for i in 0..n / 4 {
                let o = pos[i];
                let np = Point3::new(
                    (o.x + rng.r(-20.0, 20.0)).clamp(0.0, W - 0.1),
                    (o.y + rng.r(-20.0, 20.0)).clamp(0.0, W - 0.1),
                    (o.z + rng.r(-20.0, 20.0)).clamp(0.0, W - 0.1));
                ix.update(slots[i], |c| c.p = np);
                pos[i] = np;
            }
            for c in 0..culls {
                let q = pos[(c * 7919) % n];
                sink += ix.cull(&Sphere3::new(q.x, q.y, q.z, 30.0)).len();
            }
            ix.tick();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(sink);
    }
    best
}

fn main() {
    println!("brute_edge | world {W}^3, {FRAMES} frames, a quarter moving each frame, best of {REPS}");
    println!("Every arm runs through AdaptiveIndex with pinned thresholds, so the call path and");
    println!("item storage are identical and only the backend differs.\n");
    println!("Each cell: the fastest backend at that (population, culls-per-frame), and by how much.\n");

    let pops = [64usize, 128, 182, 256, 384, 512, 768, 1024, 2048];
    /// A query load: how many culls per frame, as a function of the population.
    type Load = (&'static str, fn(usize) -> usize);
    let loads: [Load; 4] = [
        ("q=1", |_| 1),
        ("q=n/16", |n| (n / 16).max(1)),
        ("q=n/4", |n| (n / 4).max(1)),
        ("q=n", |n| n),
    ];

    print!("{:>7}", "pop");
    for (name, _) in &loads { print!("{name:>22}"); }
    println!();

    let mut edge_at_heaviest = None;
    for &n in &pops {
        print!("{n:>7}");
        for (li, (_, load)) in loads.iter().enumerate() {
            let culls = load(n);
            let brute = run(n, culls, Backend::Brute);
            let keep = run(n, culls, Backend::KeepTree);
            let grid = run(n, culls, Backend::Grid);
            let best_idx = keep.min(grid);
            let (who, ratio) = if brute <= best_idx { ("scan", best_idx / brute) } else if keep <= grid { ("keep", brute / keep) } else { ("grid", brute / grid) };
            print!("{:>22}", format!("{who} {ratio:.2}x"));
            // the heaviest column is the one that sets an unconditional floor
            if li == loads.len() - 1 && who != "scan" && edge_at_heaviest.is_none() { edge_at_heaviest = Some(n); }
        }
        println!();
    }

    println!();
    match edge_at_heaviest {
        Some(n) => {
            println!("At the heaviest load an index first wins at n = {n}. `brute_max` fires BEFORE the");
            println!("load-aware rule, so it must be set below that: above it, a workload an index");
            println!("would have won is forced onto a scan with no way to appeal.");
        }
        None => println!("No index won at any population measured — `brute_max` is not the binding constraint here."),
    }
    println!("Read the low-load columns as the scan's real reach: that is `scan_budget`'s job, and");
    println!("it can see the query load that this floor cannot.");
}
