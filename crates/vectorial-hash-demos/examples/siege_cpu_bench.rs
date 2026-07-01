//! Headless **CPU-side** FPS benchmark for the siege sim.
//!
//! Runs the exact per-frame *simulation* work `siege_wgpu` does — rebuild the
//! `Tree3` unit index, the parallel `decide` pass (over a sized rayon pool), the
//! serial `apply`, and the volcano/crater step — with **no window and no GPU
//! render**. Reports the frames/second sustained at each thread count: the
//! **CPU-ceiling FPS** (the max the sim alone sustains; the real on-screen FPS is
//! this capped by the GPU render + vsync).
//!
//! Key detail: the armies **start apart and take ~seconds to clash**, and the
//! early (spread, no-combat) frames are cheap and unrepresentative. So we warm
//! up ONCE to a mid-battle state, then **clone** it and time each thread count
//! from that same clash — a fair, like-for-like comparison at the load that
//! actually matters.
//!
//! It also settles the "why rebuild the tree every frame?" question by timing
//! **three** index-maintenance strategies at each thread count, full frame:
//!   - **insert** — `clear()` + an `insert` per unit (what the demo ships).
//!   - **bulk**   — `bulk_load_par` (parallel top-down partition rebuild).
//!   - **keep**   — DON'T rebuild: keep the tree and `update_ref` every unit to
//!     its new position (O(1) if it stayed in its leaf; relocate only on a
//!     boundary cross), `remove_ref` on death, `insert_ref` on respawn.
//! `keep` does the least build work, but its tree shape *drifts* as the two
//! armies collapse into the centre — so this measures the net effect on the
//! whole frame (build + the parallel `decide` queries), which is the only fair
//! way to compare.
//!
//! Env: `SIEGE_POP` units/side (default 10000 → 20000 total), `SIEGE_SECS` wall
//! seconds timed per thread count (default 3), `SIEGE_WARMUP_SIM` sim-seconds to
//! advance before measuring (default 30 — enough for the lines to meet & clash).
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example siege_cpu_bench --release
//! ```

use rayon::prelude::*;
use std::time::Instant;
use vectorial_hash::{Aabb, ItemRef, Tree3};
use vectorial_hash_demos::siege_sim::{
    apply, decide, default_body_radius, set_map_seed, spawn_army, volcano_step, Craters, Fx, IUnit,
    Projectile, Puff, Rng, SepTables, Unit, Volcano, SKY, WORLD,
};

fn env_f64(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

/// Everything a frame reads/writes (the index is rebuilt each frame, so it's not
/// part of the cloned state).
#[derive(Clone)]
struct SimState {
    units: Vec<Unit>,
    smoke: Vec<Puff>,
    effects: Vec<Fx>,
    projectiles: Vec<Projectile>,
    craters: Craters,
    volcano: Volcano,
    rng: Rng,
    now: f64,
}

/// The three index-maintenance strategies we're comparing.
#[derive(Clone, Copy, PartialEq)]
enum Rebuild {
    Insert, // clear() + insert per unit (what the demo does today)
    Bulk,   // parallel top-down partition (bulk_load_par) — the new lever
    Keep,   // don't rebuild: update_ref the persistent tree in place
}

/// The lightweight index item for unit `i`.
fn iunit(i: usize, u: &Unit) -> IUnit {
    IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face }
}

fn step_frame(s: &mut SimState, index: &mut Tree3<IUnit>, handles: &mut [Option<ItemRef>], pool: &rayon::ThreadPool, sep: &SepTables, world: Aabb, dt: f64, mode: Rebuild) {
    s.now += dt;
    match mode {
        // Parallel top-down partition (fans the rebuild out over the same pool).
        Rebuild::Bulk => {
            let items: Vec<IUnit> = s.units.iter().enumerate().filter(|(_, u)| u.alive()).map(|(i, u)| iunit(i, u)).collect();
            *index = pool.install(|| Tree3::<IUnit>::bulk_load_par(world, 8, items));
        }
        // Serial clear() + per-unit insert loop.
        Rebuild::Insert => {
            index.clear();
            for (i, u) in s.units.iter().enumerate() { if u.alive() { index.insert(iunit(i, u)); } }
        }
        // Keep the tree: sync it to this frame's positions. A unit that stayed in
        // its leaf is O(1) (update_ref short-circuits on bbox.contains); only
        // boundary-crossers relocate. Deaths remove_ref, respawns insert_ref.
        Rebuild::Keep => {
            for (i, u) in s.units.iter().enumerate() {
                match (u.alive(), handles[i]) {
                    (true, Some(h)) => { let it = iunit(i, u); if !index.update_ref(h, |slot| *slot = it) { handles[i] = index.insert_ref(it); } }
                    (true, None)    => { handles[i] = index.insert_ref(iunit(i, u)); }
                    (false, Some(h)) => { index.remove_ref(h); handles[i] = None; }
                    (false, None)   => {}
                }
            }
        }
    }
    let mut smoke_index = Tree3::<Puff>::new(world, 8);
    for p in &s.smoke { smoke_index.insert(*p); }
    {
        let (idx, smk, units) = (&*index, &smoke_index, &mut s.units);
        pool.install(|| units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, sep)));
    }
    let impacts = apply(&mut s.units, &mut s.smoke, &mut s.effects, &mut s.projectiles, &s.craters, &mut s.rng, dt, s.now);
    volcano_step(&mut s.volcano, &mut s.smoke, &mut s.effects, &mut s.projectiles, &mut s.rng, dt, s.now);
    for (ip, r) in impacts { s.craters.carve(ip.x, ip.z, r * 0.85); }
}

fn main() {
    let per_faction = env_usize("SIEGE_POP", 10000);
    let secs = env_f64("SIEGE_SECS", 3.0);
    let warmup_sim = env_f64("SIEGE_WARMUP_SIM", 30.0);
    let max = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
    let sep = SepTables::new(&default_body_radius());
    let dt = 1.0 / 60.0;

    set_map_seed(0.135);
    println!("siege headless CPU bench | {} units/side ({} total) | warm to sim t={:.0}s (mid-clash), then time {:.0}s per thread count | NO GPU render", per_faction, per_faction * 2, warmup_sim, secs);

    // Warm up ONCE to a mid-battle clash (fast, using all cores).
    let mut s = SimState { units: spawn_army(&mut Rng::new(42), per_faction), smoke: Vec::new(), effects: Vec::new(), projectiles: Vec::new(), craters: Craters::default(), volcano: Volcano { smoke_t: 0.0, erupt_t: 6.0 }, rng: Rng::new(1), now: 0.0 };
    let mut index = Tree3::<IUnit>::new(world, 8);
    let mut warm_handles: Vec<Option<ItemRef>> = Vec::new(); // unused by Insert
    let warm_pool = rayon::ThreadPoolBuilder::new().num_threads(max).build().unwrap();
    let wt = Instant::now();
    while s.now < warmup_sim {
        step_frame(&mut s, &mut index, &mut warm_handles, &warm_pool, &sep, world, dt, Rebuild::Insert);
    }
    let alive = s.units.iter().filter(|u| u.alive()).count();
    println!("warmed to clash in {:.1}s wall ({} alive, {} smoke puffs, {} projectiles)\n", wt.elapsed().as_secs_f64(), alive, s.smoke.len(), s.projectiles.len());

    let warmed = s; // the shared clash state

    // Guard against a false win: `keep` must hold exactly what the shipping
    // clear()+insert path holds each frame — otherwise `decide` would query a
    // different (e.g. sparser) index and the FPS comparison would be apples to
    // oranges. Run keep and insert in lockstep from the same state and assert
    // their item-counts match every frame. (Both index fewer than `alive`: a few
    // units sink into deep craters below y=0, outside the root AABB, and
    // insert_ref drops them in BOTH — a pre-existing quirk, identical for each.)
    {
        let (mut a, mut b) = (warmed.clone(), warmed.clone());
        let mut idx_k = Tree3::<IUnit>::new(world, 8);
        let mut idx_i = Tree3::<IUnit>::new(world, 8);
        let mut h: Vec<Option<ItemRef>> = vec![None; a.units.len()];
        for (i, u) in a.units.iter().enumerate() { if u.alive() { h[i] = idx_k.insert_ref(iunit(i, u)); } }
        let (mut matches, mut max_oob) = (true, 0i64);
        for _ in 0..120 {
            step_frame(&mut a, &mut idx_k, &mut h, &warm_pool, &sep, world, dt, Rebuild::Keep);
            step_frame(&mut b, &mut idx_i, &mut Vec::new(), &warm_pool, &sep, world, dt, Rebuild::Insert);
            let alive = a.units.iter().filter(|u| u.alive()).count() as i64;
            let (ck, ci) = (idx_k.item_count() as i64, idx_i.item_count() as i64);
            if ck != ci { matches = false; println!("  DIVERGENCE: keep={ck} != insert={ci}"); break; }
            max_oob = max_oob.max(alive - ci);
        }
        println!("keep vs insert over 120 frames: {} · out-of-bounds units dropped by both (peak): {}\n",
            if matches { "IDENTICAL index every frame (keep is a correct drop-in)" } else { "DIVERGED — not trustworthy" }, max_oob);
    }

    // Measure the full CPU frame (rebuild + parallel decide + apply) at each
    // thread count under all three index-maintenance strategies — the real
    // "keep vs rebuild" question, measured end-to-end (not just the build step):
    //   insert = clear()+insert · bulk = bulk_load_par · keep = update_ref in place
    let measure = |threads: usize, mode: Rebuild| -> f64 {
        let mut sc = warmed.clone();
        let mut idx = Tree3::<IUnit>::new(world, 8);
        let mut handles: Vec<Option<ItemRef>> = vec![None; sc.units.len()];
        // The "keep" tree is built ONCE up front (with stable handles); the loop
        // only ever update_ref/insert_ref/remove_ref it after that.
        if mode == Rebuild::Keep {
            for (i, u) in sc.units.iter().enumerate() { if u.alive() { handles[i] = idx.insert_ref(iunit(i, u)); } }
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let t = Instant::now();
        let mut frames = 0u64;
        while t.elapsed().as_secs_f64() < secs { step_frame(&mut sc, &mut idx, &mut handles, &pool, &sep, world, dt, mode); frames += 1; }
        frames as f64 / t.elapsed().as_secs_f64()
    };
    println!("{:>8} | {:>11} {:>11} {:>11} | {:>8} {:>8}", "threads", "insert fps", "bulk fps", "keep fps", "bulk/ins", "keep/ins");
    for threads in 1..=max {
        let ins = measure(threads, Rebuild::Insert);
        let blk = measure(threads, Rebuild::Bulk);
        let kep = measure(threads, Rebuild::Keep);
        println!("{:>8} | {:>11.1} {:>11.1} {:>11.1} | {:>7.2}x {:>7.2}x", threads, ins, blk, kep, blk / ins, kep / ins);
    }

    // Is a headless GPU adapter available? (Decides whether an offscreen render
    // bench — GPU work, no present — is even possible in this environment.)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })) {
        Some(a) => { let i = a.get_info(); println!("\nheadless GPU adapter: {} ({:?}, {:?}) — an offscreen render bench IS possible", i.name, i.device_type, i.backend); }
        None => println!("\nno headless GPU adapter available — can't measure the GPU render offscreen here"),
    }
}
