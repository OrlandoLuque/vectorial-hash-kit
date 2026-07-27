//! 2D structure decision map — the 2D counterpart of `critters3d_headless
//! --sweep`. N points move in a square; every frame each is relocated in the
//! index and a sample of disc culls is run. The same deterministic sim drives
//! three structures so they rank head-to-head:
//!
//! - **binary** `Tree` — persistent, maintained via the O(1) `update_ref` handle,
//! - **quad** `QuadTree` — persistent, `update_ref`,
//! - **morton** `MortonGrid` — pointer-free, **rebuilt every frame** (`clear` + refill).
//!
//! Two numbers per structure: **maintain** (per-frame relocate or rebuild) and
//! **cull** (per-cull). The persistent-vs-rebuilt asymmetry is the headline.
//!
//! ```bash
//! cargo run -p vectorial-hash --example decision2d --release
//! cargo run -p vectorial-hash --example decision2d --release -- --sweep
//! ```

#![allow(clippy::manual_range_contains, clippy::needless_range_loop)]

use std::time::Instant;

use vectorial_hash::{CellState, ItemRef, MortonGrid, Point, Positioned, Rect, Shape, Tree, QuadTree};

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct C2 { id: u32, p: Point }
impl Positioned for C2 { fn position(&self) -> Point { self.p } }

/// A disc query with an analytic `classify_box` (so the trees prune hierarchically).
struct Disc { cx: f64, cy: f64, r: f64 }
impl Shape for Disc {
    fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
    fn contains_point(&self, p: Point) -> bool { let (dx, dy) = (p.x - self.cx, p.y - self.cy); dx * dx + dy * dy <= self.r * self.r }
    fn classify_box(&self, b: &Rect) -> Option<CellState> {
        let (nx, ny) = (self.cx.clamp(b.x, b.x_max()), self.cy.clamp(b.y, b.y_max()));
        if (nx - self.cx).powi(2) + (ny - self.cy).powi(2) > self.r * self.r {
            return Some(CellState::Out);
        }
        let fx = if (self.cx - b.x).abs() > (self.cx - b.x_max()).abs() { b.x } else { b.x_max() };
        let fy = if (self.cy - b.y).abs() > (self.cy - b.y_max()).abs() { b.y } else { b.y_max() };
        if (fx - self.cx).powi(2) + (fy - self.cy).powi(2) <= self.r * self.r {
            Some(CellState::In)
        } else {
            Some(CellState::Maybe)
        }
    }
}

const MARGIN: f64 = 4.0;
const NAMES: [&str; 3] = ["binary", "quad", "morton"];

struct Cfg { world: f64, pop: usize, item_limit: usize, vision: f64, speed: f64, n_cull: usize, frames: usize, warmup: usize, dt: f64, seed: u64 }

/// Per-structure (maintain µs/frame, cull µs/cull) + whether all culls agreed.
fn measure(cfg: &Cfg) -> ([f64; 3], [f64; 3], bool) {
    let mut rng = Rng::new(cfg.seed);
    let rect = Rect::new(0.0, 0.0, cfg.world, cfg.world);
    let levels = MortonGrid::<C2>::levels_for_cell_size(rect, cfg.vision.max(2.0));
    let us = |t: Instant| t.elapsed().as_secs_f64() * 1e6;

    let mut pos: Vec<Point> = Vec::with_capacity(cfg.pop);
    let mut vel: Vec<(f64, f64)> = Vec::with_capacity(cfg.pop);
    let mut tree = Tree::<C2>::new(rect, cfg.item_limit);
    let mut quad = QuadTree::<C2>::new(rect, cfg.item_limit);
    let mut tr: Vec<ItemRef> = Vec::with_capacity(cfg.pop);
    let mut qr: Vec<ItemRef> = Vec::with_capacity(cfg.pop);
    for id in 0..cfg.pop {
        let p = Point::new(rng.range(MARGIN, cfg.world - MARGIN), rng.range(MARGIN, cfg.world - MARGIN));
        let speed = rng.range(0.35 * cfg.speed, cfg.speed);
        let a = rng.range(0.0, std::f64::consts::TAU);
        pos.push(p);
        vel.push((speed * a.cos(), speed * a.sin()));
        tr.push(tree.insert_ref(C2 { id: id as u32, p }).unwrap());
        qr.push(quad.insert_ref(C2 { id: id as u32, p }).unwrap());
    }

    let mut mt: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut cl: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut blackhole = 0usize;
    let mut agree = true;

    for frame in 0..(cfg.warmup + cfg.frames) {
        for id in 0..cfg.pop {
            let (mut vx, mut vy) = vel[id];
            let mut nx = pos[id].x + vx * cfg.dt;
            let mut ny = pos[id].y + vy * cfg.dt;
            if nx < MARGIN || nx > cfg.world - MARGIN { vx = -vx; nx = nx.clamp(MARGIN, cfg.world - MARGIN); }
            if ny < MARGIN || ny > cfg.world - MARGIN { vy = -vy; ny = ny.clamp(MARGIN, cfg.world - MARGIN); }
            vel[id] = (vx, vy);
            pos[id] = Point::new(nx, ny);
        }
        let measuring = frame >= cfg.warmup;

        // maintain: binary (update_ref), quad (update_ref), morton (rebuild).
        let t = Instant::now();
        for id in 0..cfg.pop { tree.update_ref(tr[id], |c| c.p = pos[id]); }
        if measuring { mt[0].push(us(t)); }

        let t = Instant::now();
        for id in 0..cfg.pop { quad.update_ref(qr[id], |c| c.p = pos[id]); }
        if measuring { mt[1].push(us(t)); }

        let t = Instant::now();
        let mut morton = MortonGrid::<C2>::new(rect, levels);
        for id in 0..cfg.pop { morton.insert(C2 { id: id as u32, p: pos[id] }); }
        if measuring { mt[2].push(us(t)); }

        // cull: the same sampled disc centres for all three.
        let ids: Vec<usize> = (0..cfg.n_cull).map(|_| (rng.next() as usize) % cfg.pop).collect();
        let n = ids.len().max(1) as f64;

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(tree.cull(&Disc { cx: c.x, cy: c.y, r: cfg.vision }).len()); }
        if measuring { cl[0].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(quad.cull(&Disc { cx: c.x, cy: c.y, r: cfg.vision }).len()); }
        if measuring { cl[1].push(us(t) / n); }

        let t = Instant::now();
        for &id in &ids { let c = pos[id]; blackhole = blackhole.wrapping_add(morton.cull(&Disc { cx: c.x, cy: c.y, r: cfg.vision }).len()); }
        if measuring { cl[2].push(us(t) / n); }

        if measuring && !ids.is_empty() {
            let c = pos[ids[0]];
            let s = Disc { cx: c.x, cy: c.y, r: cfg.vision };
            let mut a0: Vec<u32> = tree.cull(&s).iter().map(|x| x.id).collect();
            let mut a1: Vec<u32> = quad.cull(&s).iter().map(|x| x.id).collect();
            let mut a2: Vec<u32> = morton.cull(&s).iter().map(|x| x.id).collect();
            a0.sort_unstable(); a1.sort_unstable(); a2.sort_unstable();
            if a1 != a0 || a2 != a0 { agree = false; }
        }
    }
    if blackhole == usize::MAX { println!("unreachable"); }
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    (std::array::from_fn(|k| mean(&mt[k])), std::array::from_fn(|k| mean(&cl[k])), agree)
}

fn winner(v: &[f64]) -> (usize, f64) {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let (w, second) = (order[0], order[1]);
    (w, if v[w] > 0.0 { v[second] / v[w] } else { 1.0 })
}

fn main() {
    let sweep = std::env::args().any(|a| a == "--sweep");
    if sweep {
        run_sweep();
        return;
    }
    // Knobs, because the answer moves with population: the 2D demos run hundreds to a
    // few thousand critters, not the 50k this used to hardcode, and the winner is not the
    // same at both ends. Sweep with `bench-runner --group sweeps`.
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let cfg = Cfg {
        world: env("D2_WORLD", 1024.0), pop: env("D2_POP", 50_000.0) as usize,
        item_limit: env("D2_IL", 8.0) as usize, vision: env("D2_R", 24.0),
        speed: env("D2_SPEED", 240.0), n_cull: env("D2_CULLS", 16.0) as usize,
        frames: 90, warmup: 20, dt: 1.0 / 60.0, seed: 42,
    };
    println!("2D decision map | world={}² | pop={} | item_limit={} | vision r={} | {} culls/frame | {} frames (+{} warmup)",
        cfg.world, cfg.pop, cfg.item_limit, cfg.vision, cfg.n_cull, cfg.frames, cfg.warmup);
    let (m, c, agree) = measure(&cfg);
    println!("\n{:<8} {:>16} {:>14} {:>18}", "structure", "maintain us/frame", "cull us/cull", "per-frame total us");
    for k in 0..3 {
        println!("{:<8} {:>16.1} {:>14.3} {:>18.1}", NAMES[k], m[k], c[k], m[k] + cfg.n_cull as f64 * c[k]);
    }
    let total: [f64; 3] = std::array::from_fn(|k| m[k] + cfg.n_cull as f64 * c[k]);
    let (wm, mm) = winner(&m);
    let (wc, mc) = winner(&c);
    let (wt, mtt) = winner(&total);
    println!("\nwinner — maintain: {} ({:.2}×) | cull: {} ({:.2}×) | total@{}culls: {} ({:.2}×)",
        NAMES[wm], mm, NAMES[wc], mc, cfg.n_cull, NAMES[wt], mtt);
    println!("cull agreement: {}", if agree { "EXACT (identical id sets)" } else { "DISAGREE <-- BUG" });
}

fn run_sweep() {
    let worlds = [256.0, 1024.0];
    let pops = [10_000usize, 50_000];
    let ils = [8usize, 32];
    let speeds = [(20.0, "slow"), (240.0, "fast")];
    println!("2D structure decision map | world × pop × item_limit × churn | vision r=24 | 16 culls/frame");
    println!("(maintain = per-frame update[bin/quad] or rebuild[morton]; cull = per-cull. winner = lowest.)\n");
    println!("{:>6} {:>7} {:>4} {:>5} | {:<22} | {:<22} | {:<8}", "world", "pop", "il", "churn", "maintain winner", "cull winner", "agree");
    let mut wins_m = [0u32; 3];
    let mut wins_c = [0u32; 3];
    let mut all_agree = true;
    for &world in &worlds {
        for &pop in &pops {
            for &il in &ils {
                for &(speed, sname) in &speeds {
                    let cfg = Cfg { world, pop, item_limit: il, vision: 24.0, speed, n_cull: 16, frames: 50, warmup: 15, dt: 1.0 / 60.0, seed: 42 };
                    let (m, c, agree) = measure(&cfg);
                    let (wm, mm) = winner(&m);
                    let (wc, mc) = winner(&c);
                    wins_m[wm] += 1; wins_c[wc] += 1;
                    if !agree { all_agree = false; }
                    println!("{:>5.0}² {:>7} {:>4} {:>5} | {:<4} {:>8.0}us ({:.2}×) | {:<4} {:>7.3}us ({:.2}×) | {:<8}",
                        world, pop, il, sname, NAMES[wm], m[wm], mm, NAMES[wc], c[wc], mc, if agree { "exact" } else { "DISAGREE!" });
                }
            }
        }
    }
    println!("\nmaintain wins: binary {} | quad {} | morton {}", wins_m[0], wins_m[1], wins_m[2]);
    println!("cull wins:     binary {} | quad {} | morton {}", wins_c[0], wins_c[1], wins_c[2]);
    println!("agreement across all configs: {}", if all_agree { "EXACT" } else { "DISAGREEMENT <-- BUG" });
}
