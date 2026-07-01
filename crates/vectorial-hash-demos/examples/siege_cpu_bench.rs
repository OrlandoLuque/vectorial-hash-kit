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
//! Env: `SIEGE_POP` units/side (default 10000 → 20000 total), `SIEGE_SECS` wall
//! seconds timed per thread count (default 3), `SIEGE_WARMUP_SIM` sim-seconds to
//! advance before measuring (default 30 — enough for the lines to meet & clash).
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example siege_cpu_bench --release
//! ```

use rayon::prelude::*;
use std::time::Instant;
use vectorial_hash::{Aabb, Tree3};
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

fn step_frame(s: &mut SimState, index: &mut Tree3<IUnit>, pool: &rayon::ThreadPool, sep: &SepTables, world: Aabb, dt: f64) {
    s.now += dt;
    index.clear();
    for (i, u) in s.units.iter().enumerate() {
        if u.alive() { index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face }); }
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
    let warm_pool = rayon::ThreadPoolBuilder::new().num_threads(max).build().unwrap();
    let wt = Instant::now();
    while s.now < warmup_sim {
        step_frame(&mut s, &mut index, &warm_pool, &sep, world, dt);
    }
    let alive = s.units.iter().filter(|u| u.alive()).count();
    println!("warmed to clash in {:.1}s wall ({} alive, {} smoke puffs, {} projectiles)\n", wt.elapsed().as_secs_f64(), alive, s.smoke.len(), s.projectiles.len());
    println!("{:>8} | {:>9} {:>10} {:>8} {:>7}", "threads", "frames", "CPU fps", "vs 1", "eff");

    let warmed = s; // the shared clash state
    let mut base = 0.0f64;
    for threads in 1..=max {
        let mut sc = warmed.clone();
        let mut idx = Tree3::<IUnit>::new(world, 8);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let t = Instant::now();
        let mut frames = 0u64;
        while t.elapsed().as_secs_f64() < secs {
            step_frame(&mut sc, &mut idx, &pool, &sep, world, dt);
            frames += 1;
        }
        let fps = frames as f64 / t.elapsed().as_secs_f64();
        if threads == 1 { base = fps; }
        println!("{:>8} | {:>9} {:>10.1} {:>7.2}x {:>6.0}%", threads, frames, fps, fps / base, fps / base / threads as f64 * 100.0);
    }

    // Is a headless GPU adapter available? (Decides whether an offscreen render
    // bench — GPU work, no present — is even possible in this environment.)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })) {
        Some(a) => { let i = a.get_info(); println!("\nheadless GPU adapter: {} ({:?}, {:?}) — an offscreen render bench IS possible", i.name, i.device_type, i.backend); }
        None => println!("\nno headless GPU adapter available — can't measure the GPU render offscreen here"),
    }
}
