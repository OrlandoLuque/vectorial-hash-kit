//! Micro-benchmark for the **per-frame index rebuild** — the one serial step
//! `siege`/`siege_wgpu` do every frame and the thing [`Tree3::bulk_load`] exists
//! to speed up.
//!
//! Both binaries rebuild a fresh `Tree3<IUnit>` from live positions each frame
//! (`index.clear()` then an `insert` per unit) because the units all moved. That
//! rebuild is a pure **serial tail** in the decide→apply pipeline (Amdahl): the
//! decide pass fans out over rayon, but the rebuild can't — so as thread count
//! rises it's an ever-larger fraction of the frame. `bulk_load` replaces the N
//! root-descending `insert`s with **one top-down partition**; `bulk_load_par`
//! fans *that* out over rayon.
//!
//! This isolates just that step: warm two armies to a real mid-battle clash
//! (the representative, clustered distribution — matches `siege_cpu_bench`), then
//! time the three build strategies on that same snapshot.
//!
//! Env: `SIEGE_POP` units/side (default 10000 → 20000 total), `SIEGE_REPS`
//! rebuilds timed per strategy (default 200), `SIEGE_WARMUP_SIM` sim-seconds to
//! advance first (default 30).
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example bulk_load_bench --release --features parallel
//! ```

use std::time::Instant;
use vectorial_hash::{Aabb, Tree3};
use vectorial_hash_demos::siege_sim::{
    apply, decide, default_body_radius, set_map_seed, spawn_army, volcano_step, Craters, Fx, IUnit,
    Projectile, Puff, Rng, SepTables, Unit, Volcano, SKY, WORLD,
};

fn env_f64(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

/// Snapshot the live units into the item type the index actually holds.
fn snapshot(units: &[Unit]) -> Vec<IUnit> {
    units.iter().enumerate().filter(|(_, u)| u.alive())
        .map(|(i, u)| IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face })
        .collect()
}

fn main() {
    let per_faction = env_usize("SIEGE_POP", 10000);
    let reps = env_usize("SIEGE_REPS", 200);
    let warmup_sim = env_f64("SIEGE_WARMUP_SIM", 30.0);
    let max = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
    let sep = SepTables::new(&default_body_radius());
    let dt = 1.0 / 60.0;
    let item_limit = 8;

    set_map_seed(0.135);
    println!("bulk_load micro-bench | {} units/side ({} total) | warm to sim t={:.0}s (mid-clash), then time {} rebuilds/strategy", per_faction, per_faction * 2, warmup_sim, reps);

    // Warm to a mid-battle clash (all cores), exactly like siege_cpu_bench.
    let mut units = spawn_army(&mut Rng::new(42), per_faction);
    let (mut smoke, mut effects, mut projectiles): (Vec<Puff>, Vec<Fx>, Vec<Projectile>) = (Vec::new(), Vec::new(), Vec::new());
    let mut craters = Craters::default();
    let mut volcano = Volcano { smoke_t: 0.0, erupt_t: 6.0 };
    let mut rng = Rng::new(1);
    let mut now = 0.0f64;
    let warm_pool = rayon::ThreadPoolBuilder::new().num_threads(max).build().unwrap();
    let mut index = Tree3::<IUnit>::new(world, item_limit);
    while now < warmup_sim {
        now += dt;
        index.clear();
        for u in snapshot(&units) { index.insert(u); }
        let mut smoke_index = Tree3::<Puff>::new(world, 8);
        for p in &smoke { smoke_index.insert(*p); }
        { let (idx, smk, us) = (&index, &smoke_index, &mut units);
          warm_pool.install(|| { use rayon::prelude::*; us.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, &sep)); }); }
        let impacts = apply(&mut units, &mut smoke, &mut effects, &mut projectiles, &craters, &mut rng, dt, now);
        volcano_step(&mut volcano, &mut smoke, &mut effects, &mut projectiles, &mut rng, dt, now);
        for (ip, r) in impacts { craters.carve(ip.x, ip.z, r * 0.85); }
    }

    let items = snapshot(&units);
    println!("clash snapshot: {} live units\n", items.len());
    println!("{:>26} | {:>10} {:>9} {:>8}", "strategy", "us/rebuild", "Mitems/s", "vs insert");

    // Every arm is timed per-rep with its INPUT PREPARED OUTSIDE THE CLOCK, and reported as the
    // minimum over reps.
    //
    // The previous version wrapped the whole rep loop in one timer and wrote `bulk_load(world,
    // item_limit, items.clone())` inside it. Two things were wrong with that. The clone was
    // charged to the two bulk arms and NOT to the `clear + insert` baseline, which iterates
    // `&items` — so the headline "vs insert" column was biased against the thing being proposed.
    // And a clone inside the clock is not the harmless constant it appears to be: on
    // `IntegerTree` the same mistake reported a 2.65x parallel win as 0.85x, i.e. inverted it.
    // See `docs/MEASURING.md` § 8g. The old footnote argued the clone away on the grounds that
    // the demo builds its snapshot Vec fresh each frame anyway — true of the demo, and no
    // defence at all of a bench whose arms did not all pay it.
    //
    // Minimum rather than mean, for the reason § 8e gives: this machine's noise is episodic, so
    // the mean of a run that caught one interruption is not a measurement of the code.
    let mut best = |mut f: Box<dyn FnMut(Vec<IUnit>)>| -> f64 {
        let mut b = f64::INFINITY;
        f(items.clone());                       // warm: first touch, and rayon's lazy pool
        for _ in 0..reps {
            let input = items.clone();          // outside the clock
            let t = Instant::now();
            f(input);
            b = b.min(t.elapsed().as_secs_f64());
        }
        b
    };

    let mut idx = Tree3::<IUnit>::new(world, item_limit);
    let base = best(Box::new(move |v| { idx.clear(); for it in &v { idx.insert(*it); } std::hint::black_box(&idx); }));
    let report = |name: &str, per: f64| println!("{:>26} | {:>10.1} {:>9.1} {:>7.2}x", name, per * 1e6, items.len() as f64 / per / 1e6, base / per);
    report("clear + insert (current)", base);

    let ser = best(Box::new(|v| { std::hint::black_box(Tree3::<IUnit>::bulk_load(world, item_limit, v)); }));
    report("bulk_load (serial)", ser);

    for threads in [2usize, 4, 8, max].into_iter().collect::<std::collections::BTreeSet<_>>() {
        if threads > max { continue; }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let par = best(Box::new(move |v| { pool.install(|| { std::hint::black_box(Tree3::<IUnit>::bulk_load_par(world, item_limit, v)); }); }));
        report(&format!("bulk_load_par ({} thr)", threads), par);
    }
}
