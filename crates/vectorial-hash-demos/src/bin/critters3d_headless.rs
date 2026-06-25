//! 3D critters, headless — the dynamic 3D workload, and the **structure
//! decision map**. N points move in a cube; every frame each is relocated in
//! the index and a sample run a sphere "vision" cull. The same deterministic
//! simulation drives **all four** structures so they can be ranked head-to-head:
//!
//! - **binary** `Tree3` — persistent, predicate `update` (ascend-to-LCA),
//! - **octree** `Octree3` — persistent, predicate `update`,
//! - **morton** `MortonGrid3` — pointer-free, **rebuilt every frame**,
//! - **projection** — a 2D `Tree` on xy **rebuilt every frame** + disc cull,
//!   z-slab reject, exact 3D narrowphase,
//! - **binary-ref** — the binary `Tree3` maintained via the stable `ItemRef`
//!   handle (`update_ref`, O(1) — no predicate scan). This is what flips the
//!   maintain winner; see `THREE_D.md` § "The fix: Stable ItemRef".
//!
//! Two numbers per structure: **maintain** (per-frame update or rebuild cost)
//! and **cull** (per-cull cost). The persistent-vs-rebuilt asymmetry is the
//! dominant effect (see `THREE_D.md` § "Synthesis").
//!
//! ```bash
//! # one config (detailed table for all four structures):
//! cargo run -p vectorial-hash-demos --bin critters3d_headless --release -- \
//!     --pop 20000 --item-limit 8 --vision 36 --frames 120 --warmup 30 --seed 42
//! # the decision map (sweep world × pop × vision × item_limit):
//! cargo run -p vectorial-hash-demos --bin critters3d_headless --release -- --sweep
//! ```

use std::time::Instant;

use vectorial_hash::{
    Aabb, ItemRef, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, Rect, Shape,
    Sphere3, Tree, Tree3,
};

const MARGIN: f64 = 4.0;
// "binary-ref" is the binary Tree3 maintained through the stable ItemRef handle
// (O(1) update_ref) instead of the predicate update — the Stable-ItemRef test.
const NAMES: [&str; 5] = ["binary", "octree", "morton", "projection", "binary-ref"];
const SHORT: [&str; 5] = ["bin", "oct", "mor", "prj", "binR"];

struct Args {
    sweep: bool,
    parallel: bool,
    world: f64,
    pop: usize,
    item_limit: usize,
    vision: f64, // sphere radius
    speed: f64, // max speed (range 0.35×..1× of it) — sets the churn rate
    n_cull: usize, // culls per frame (cull timing + total weighting)
    frames: usize,
    warmup: usize,
    dt: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        sweep: false, parallel: false, world: 512.0, pop: 20000, item_limit: 8, vision: 36.0, speed: 120.0,
        n_cull: 16, frames: 120, warmup: 30, dt: 1.0 / 60.0, seed: 42,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--sweep" => { a.sweep = true; i -= 1; }
            "--parallel" => { a.parallel = true; i -= 1; }
            "--world" => a.world = val().parse().unwrap(),
            "--pop" => a.pop = val().parse().unwrap(),
            "--item-limit" => a.item_limit = val().parse().unwrap(),
            "--vision" => a.vision = val().parse().unwrap(),
            "--speed" => a.speed = val().parse().unwrap(),
            "--n-cull" | "--vision-count" => a.n_cull = val().parse().unwrap(),
            "--frames" => a.frames = val().parse().unwrap(),
            "--warmup" => a.warmup = val().parse().unwrap(),
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

// 2D projection item (carries id + z for the narrowphase).
#[derive(Clone, Copy)]
struct P2 { id: u32, p: Point, z: f64 }
impl Positioned for P2 { fn position(&self) -> Point { self.p } }

// The sphere's xy shadow (a disc) — bbox reject + exact, the 2D index out of
// the box (no template).
struct Disc { cx: f64, cy: f64, r: f64 }
impl Shape for Disc {
    fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
    fn contains_point(&self, p: Point) -> bool { let dx = p.x - self.cx; let dy = p.y - self.cy; dx * dx + dy * dy <= self.r * self.r }
}

struct Series(Vec<f64>);
impl Series {
    fn new() -> Self { Series(Vec::new()) }
    fn push(&mut self, v: f64) { self.0.push(v); }
    fn mean(&self) -> f64 { if self.0.is_empty() { 0.0 } else { self.0.iter().sum::<f64>() / self.0.len() as f64 } }
}

struct Cfg { world: f64, pop: usize, item_limit: usize, vision: f64, speed: f64, n_cull: usize, frames: usize, warmup: usize, dt: f64, seed: u64 }

/// Run the deterministic sim with all four structures maintained on the same
/// positions; return per-structure (maintain µs/frame, cull µs/cull) and
/// whether all four culls agreed.
fn measure(cfg: &Cfg) -> ([f64; 5], [f64; 5], bool) {
    let mut rng = Rng::new(cfg.seed);
    let world = Aabb::new(0.0, 0.0, 0.0, cfg.world, cfg.world, cfg.world);
    let rect = Rect::new(0.0, 0.0, cfg.world, cfg.world);
    let us = |t: Instant| t.elapsed().as_secs_f64() * 1e6;

    let mut pos: Vec<Point3> = Vec::with_capacity(cfg.pop);
    let mut vel: Vec<(f64, f64, f64)> = Vec::with_capacity(cfg.pop);
    let mut old_pos: Vec<Point3> = vec![Point3::new(0.0, 0.0, 0.0); cfg.pop];
    let mut tree = Tree3::<C3>::new(world, cfg.item_limit);
    let mut octree = Octree3::<C3>::new(world, cfg.item_limit);
    let mut tree_h = Tree3::<C3>::new(world, cfg.item_limit); // maintained via ItemRef
    let mut refs: Vec<ItemRef> = Vec::with_capacity(cfg.pop);
    for id in 0..cfg.pop {
        let p = Point3::new(rng.range(MARGIN, cfg.world - MARGIN), rng.range(MARGIN, cfg.world - MARGIN), rng.range(MARGIN, cfg.world - MARGIN));
        let speed = rng.range(0.35 * cfg.speed, cfg.speed);
        let (a, b) = (rng.range(0.0, std::f64::consts::TAU), rng.range(-1.0, 1.0));
        let s = (1.0_f64 - b * b).max(0.0).sqrt();
        pos.push(p);
        vel.push((speed * s * a.cos(), speed * s * a.sin(), speed * b));
        tree.insert(C3 { id: id as u32, p });
        octree.insert(C3 { id: id as u32, p });
        refs.push(tree_h.insert_ref(C3 { id: id as u32, p }).unwrap());
    }
    let levels = MortonGrid3::<C3>::levels_for_cell_size(world, cfg.vision.max(4.0));

    let mut mt: [Series; 5] = std::array::from_fn(|_| Series::new());
    let mut cl: [Series; 5] = std::array::from_fn(|_| Series::new());
    let mut blackhole = 0usize;
    let mut agree = true;

    for frame in 0..(cfg.warmup + cfg.frames) {
        for id in 0..cfg.pop {
            old_pos[id] = pos[id];
            let (mut vx, mut vy, mut vz) = vel[id];
            let mut nx = pos[id].x + vx * cfg.dt;
            let mut ny = pos[id].y + vy * cfg.dt;
            let mut nz = pos[id].z + vz * cfg.dt;
            if nx < MARGIN || nx > cfg.world - MARGIN { vx = -vx; nx = nx.clamp(MARGIN, cfg.world - MARGIN); }
            if ny < MARGIN || ny > cfg.world - MARGIN { vy = -vy; ny = ny.clamp(MARGIN, cfg.world - MARGIN); }
            if nz < MARGIN || nz > cfg.world - MARGIN { vz = -vz; nz = nz.clamp(MARGIN, cfg.world - MARGIN); }
            vel[id] = (vx, vy, vz);
            pos[id] = Point3::new(nx, ny, nz);
        }
        let measuring = frame >= cfg.warmup;

        // --- maintain (persistent update vs full rebuild), timed each ---
        // binary, predicate update (the O(item_limit) leaf scan):
        let t = Instant::now();
        for id in 0..cfg.pop { let cid = id as u32; tree.update(old_pos[id], |c| c.id == cid, |c| c.p = pos[id]); }
        if measuring { mt[0].push(us(t)); }

        // binary, stable-handle update (O(1), no scan) — timed right after the
        // predicate one so both read `pos` under the same cache conditions:
        let t = Instant::now();
        for id in 0..cfg.pop { tree_h.update_ref(refs[id], |c| c.p = pos[id]); }
        if measuring { mt[4].push(us(t)); }

        let t = Instant::now();
        for id in 0..cfg.pop { let cid = id as u32; octree.update(old_pos[id], |c| c.id == cid, |c| c.p = pos[id]); }
        if measuring { mt[1].push(us(t)); }

        let t = Instant::now();
        let mut morton = MortonGrid3::<C3>::new(world, levels);
        for id in 0..cfg.pop { morton.insert(C3 { id: id as u32, p: pos[id] }); }
        if measuring { mt[2].push(us(t)); }

        let t = Instant::now();
        let mut proj = Tree::<P2>::new(rect, cfg.item_limit);
        for id in 0..cfg.pop { let p = pos[id]; proj.insert(P2 { id: id as u32, p: Point::new(p.x, p.y), z: p.z }); }
        if measuring { mt[3].push(us(t)); }

        // --- cull (same sampled ids for all four), timed each ---
        let ids: Vec<usize> = (0..cfg.n_cull).map(|_| (rng.next() as usize) % cfg.pop).collect();
        let n = ids.len().max(1) as f64;

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(tree.cull(&Sphere3::new(c.x, c.y, c.z, cfg.vision)).len()); }
        if measuring { cl[0].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(tree_h.cull(&Sphere3::new(c.x, c.y, c.z, cfg.vision)).len()); }
        if measuring { cl[4].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(octree.cull(&Sphere3::new(c.x, c.y, c.z, cfg.vision)).len()); }
        if measuring { cl[1].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(morton.cull(&Sphere3::new(c.x, c.y, c.z, cfg.vision)).len()); }
        if measuring { cl[2].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids {
            let c = pos[id];
            let cand = proj.cull(&Disc { cx: c.x, cy: c.y, r: cfg.vision });
            let mut hits = 0usize;
            for p2 in &cand {
                let dz = p2.z - c.z;
                if dz.abs() <= cfg.vision {
                    let dx = p2.p.x - c.x; let dy = p2.p.y - c.y;
                    if dx * dx + dy * dy + dz * dz <= cfg.vision * cfg.vision { hits += 1; }
                }
            }
            blackhole = blackhole.wrapping_add(hits);
        }
        if measuring { cl[3].push(us(t) / n); }

        // Light agreement gate (untimed): all structures return the same id set
        // for the first sampled sphere — including the handle-maintained tree.
        if measuring && !ids.is_empty() {
            let c = pos[ids[0]];
            let s = Sphere3::new(c.x, c.y, c.z, cfg.vision);
            let mut a0: Vec<u32> = tree.cull(&s).iter().map(|x| x.id).collect();
            let mut a1: Vec<u32> = octree.cull(&s).iter().map(|x| x.id).collect();
            let mut a2: Vec<u32> = morton.cull(&s).iter().map(|x| x.id).collect();
            let mut a3: Vec<u32> = proj.cull(&Disc { cx: c.x, cy: c.y, r: cfg.vision }).iter()
                .filter(|p2| { let dz = p2.z - c.z; let dx = p2.p.x - c.x; let dy = p2.p.y - c.y; dx * dx + dy * dy + dz * dz <= cfg.vision * cfg.vision })
                .map(|p2| p2.id).collect();
            let mut a4: Vec<u32> = tree_h.cull(&s).iter().map(|x| x.id).collect();
            a0.sort_unstable(); a1.sort_unstable(); a2.sort_unstable(); a3.sort_unstable(); a4.sort_unstable();
            if a1 != a0 || a2 != a0 || a3 != a0 || a4 != a0 { agree = false; }
        }
    }
    if blackhole == usize::MAX { println!("unreachable"); }
    let maintain = std::array::from_fn(|k| mt[k].mean());
    let cull = std::array::from_fn(|k| cl[k].mean());
    (maintain, cull, agree)
}

/// (winner index, margin = 2nd-best / best) for a 4-vector to minimise.
fn winner(v: &[f64]) -> (usize, f64) {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let (w, second) = (order[0], order[1]);
    let margin = if v[w] > 0.0 { v[second] / v[w] } else { 1.0 };
    (w, margin)
}

fn main() {
    let args = parse_args();
    if args.parallel {
        #[cfg(feature = "parallel")]
        { run_parallel(&args); return; }
        #[cfg(not(feature = "parallel"))]
        {
            eprintln!("--parallel needs the `parallel` feature. Re-run with:\n  cargo run -p vectorial-hash-demos --bin critters3d_headless --features parallel --release -- --parallel");
            return;
        }
    }
    if args.sweep { run_sweep(&args); return; }

    let cfg = Cfg { world: args.world, pop: args.pop, item_limit: args.item_limit, vision: args.vision, speed: args.speed, n_cull: args.n_cull, frames: args.frames, warmup: args.warmup, dt: args.dt, seed: args.seed };
    println!("3D critters | world={}^3 | pop={} | item_limit={} | vision r={} | speed={} | {} culls/frame | frames={} (+{} warmup) | seed={}",
        cfg.world, cfg.pop, cfg.item_limit, cfg.vision, cfg.speed, cfg.n_cull, cfg.frames, cfg.warmup, cfg.seed);
    let (maintain, cull, agree) = measure(&cfg);
    println!("\n{:<12} {:>16} {:>14} {:>20}", "structure", "maintain us/frame", "cull us/cull", "per-frame total us");
    for k in 0..5 {
        let total = maintain[k] + cfg.n_cull as f64 * cull[k];
        println!("{:<12} {:>16.1} {:>14.3} {:>20.1}", NAMES[k], maintain[k], cull[k], total);
    }
    let total: [f64; 5] = std::array::from_fn(|k| maintain[k] + cfg.n_cull as f64 * cull[k]);
    let (wm, mm) = winner(&maintain);
    let (wc, mc) = winner(&cull);
    let (wt, mt) = winner(&total);
    println!("\nwinner — maintain: {} ({:.2}× over 2nd) | cull: {} ({:.2}×) | total@{}culls: {} ({:.2}×)",
        NAMES[wm], mm, NAMES[wc], mc, cfg.n_cull, NAMES[wt], mt);
    // binary predicate vs handle, the Stable-ItemRef headline:
    println!("maintain: binary(predicate) {:.1}us  ->  binary(ItemRef) {:.1}us  =  {:.2}× faster",
        maintain[0], maintain[4], maintain[0] / maintain[4].max(1e-9));
    println!("cull agreement: all structures {}", if agree { "EXACT (identical id sets)" } else { "DISAGREE <-- BUG" });
}

fn run_sweep(args: &Args) {
    // Churn (movement speed) is the pivotal axis for the maintain cost — the
    // trees' incremental `update` is near-free when points stay in their leaf
    // (low churn) but pays ascend-to-LCA + split/merge when they cross (high
    // churn), whereas Morton/projection re-bucket flat regardless. So we sweep
    // speed alongside world / pop / item_limit; vision (cull radius) is fixed.
    let worlds = [256.0, 1024.0];
    let pops = [10_000usize, 50_000];
    let ils = [16usize, 64];
    let speeds = [(20.0, "slow"), (180.0, "fast")];
    let vision = args.vision;
    let frames = args.frames.min(60);
    let warmup = args.warmup.min(20);
    println!("structure decision map | sweep world × pop × item_limit × churn | vision r={} | {} culls/frame | {} frames (+{} warmup) | seed={}",
        vision, args.n_cull, frames, warmup, args.seed);
    println!("(maintain = per-frame update[bin/oct] or rebuild[mor/prj]; cull = per-cull. winner = lowest, margin = 2nd/1st.)\n");
    println!("{:>6} {:>7} {:>4} {:>5} | {:<24} | {:<24} | {:<10}", "world", "pop", "il", "churn", "maintain winner", "cull winner", "agree");

    let mut wins_m = [0u32; 5];
    let mut wins_c = [0u32; 5];
    let mut all_agree = true;
    let mut n = 0;
    for &world in &worlds {
        for &pop in &pops {
            for &il in &ils {
                for &(speed, sname) in &speeds {
                    let cfg = Cfg { world, pop, item_limit: il, vision, speed, n_cull: args.n_cull, frames, warmup, dt: args.dt, seed: args.seed };
                    let (maintain, cull, agree) = measure(&cfg);
                    let (wm, mm) = winner(&maintain);
                    let (wc, mc) = winner(&cull);
                    wins_m[wm] += 1; wins_c[wc] += 1; n += 1;
                    if !agree { all_agree = false; }
                    println!("{:>5.0}³ {:>7} {:>4} {:>5} | {:<4} {:>8.0}us ({:.2}×) | {:<4} {:>7.3}us ({:.2}×) | {:<10}",
                        world, pop, il, sname,
                        SHORT[wm], maintain[wm], mm,
                        SHORT[wc], cull[wc], mc,
                        if agree { "exact" } else { "DISAGREE!" });
                }
            }
        }
    }
    println!("\nwins over {n} configs (binR = binary via stable ItemRef):");
    println!("  maintain: bin {} | oct {} | mor {} | prj {} | binR {}", wins_m[0], wins_m[1], wins_m[2], wins_m[3], wins_m[4]);
    println!("  cull:     bin {} | oct {} | mor {} | prj {} | binR {}", wins_c[0], wins_c[1], wins_c[2], wins_c[3], wins_c[4]);
    println!("\nagreement across all structures, every config: {}", if all_agree { "EXACT" } else { "DISAGREEMENT <-- BUG" });
}

/// Parallel batch-cull crossover (`--parallel`, feature `parallel`). For a grid
/// of index size × query count, time serial `cull_many` against rayon-backed
/// `cull_many_par` on the *same* immutable `Tree3`, and print the speedup so the
/// crossover (where forking onto threads starts to pay) is visible. This is the
/// combat phase in isolation: many independent culls over one shared index —
/// the part of the per-frame work that genuinely parallelises (the relocation
/// pass mutates the tree and stays serial; the lever there is `ItemRef`).
#[cfg(feature = "parallel")]
fn run_parallel(args: &Args) {
    let world = Aabb::new(0.0, 0.0, 0.0, args.world, args.world, args.world);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("parallel batch-cull crossover | rayon over {threads} hardware threads | world={}³ | vision r={} | item_limit={} | seed={}",
        args.world, args.vision, args.item_limit, args.seed);
    println!("(serial = cull_many, par = cull_many_par; speedup = serial/par, averaged over reps. verdict at ±10%.)\n");
    println!("{:>9} {:>8} | {:>12} {:>12} {:>8} | {}", "pop", "queries", "serial ms", "par ms", "speedup", "verdict");

    let pops = [2_000usize, 20_000, 100_000];
    let qs = [4usize, 16, 64, 256, 1024];
    let reps = 40usize;
    for &pop in &pops {
        let mut rng = Rng::new(args.seed);
        let mut tree = Tree3::<C3>::new(world, args.item_limit);
        for id in 0..pop {
            let p = Point3::new(
                rng.range(MARGIN, args.world - MARGIN),
                rng.range(MARGIN, args.world - MARGIN),
                rng.range(MARGIN, args.world - MARGIN),
            );
            tree.insert(C3 { id: id as u32, p });
        }
        for &q in &qs {
            let shapes: Vec<Sphere3> = (0..q)
                .map(|_| Sphere3::new(rng.range(0.0, args.world), rng.range(0.0, args.world), rng.range(0.0, args.world), args.vision))
                .collect();
            // Warm both paths (page-in, thread-pool spin-up) before timing.
            let mut bh = tree.cull_many(&shapes).iter().map(|v| v.len()).sum::<usize>();
            bh = bh.wrapping_add(tree.cull_many_par(&shapes).iter().map(|v| v.len()).sum::<usize>());

            let t = Instant::now();
            for _ in 0..reps { bh = bh.wrapping_add(tree.cull_many(&shapes).iter().map(|v| v.len()).sum::<usize>()); }
            let serial_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

            let t = Instant::now();
            for _ in 0..reps { bh = bh.wrapping_add(tree.cull_many_par(&shapes).iter().map(|v| v.len()).sum::<usize>()); }
            let par_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

            if bh == usize::MAX { println!("unreachable"); }
            let speedup = serial_ms / par_ms.max(1e-9);
            let verdict = if speedup >= 1.1 { "parallel wins" } else if speedup <= 0.9 { "serial wins (fork cost)" } else { "tie" };
            println!("{:>9} {:>8} | {:>12.4} {:>12.4} {:>8.2} | {}", pop, q, serial_ms, par_ms, speedup, verdict);
        }
        println!();
    }
}
