//! Headless CPU bench for the horde sim — the demo's headline measurement:
//! **an indexed population that is mostly DORMANT costs almost nothing to
//! keep**, because the keep-index sync skips unmoved sleepers entirely and the
//! decide pass early-outs on them; you pay only for the active front.
//!
//! Sweeps population × activity level and reports ms/step (and the fps ceiling)
//! for: everyone asleep · ~10% woken (a nest cascade) · a full wave assault.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_bench --release --features parallel
//! ```

use std::time::Instant;
use vectorial_hash_demos::horde_sim::{Horde, Scenario, SKind, ZState, WORLD};

fn measure(h: &mut Horde, secs: f64) -> (f64, usize) {
    let dt = 1.0 / 60.0;
    let t = Instant::now();
    let mut frames = 0u64;
    while t.elapsed().as_secs_f64() < secs { h.step(dt); frames += 1; }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    let (_, active) = h.counts();
    (ms, active)
}

fn main() {
    let secs: f64 = std::env::var("HORDE_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("horde headless CPU bench | {threads} threads | {secs:.0}s per cell | ms/step (fps ceiling) + active count");
    // NB: the wake cascades chain (that's the sim working), so the middle and
    // right columns wake far more than the seed — the a= column is the truth.
    println!("{:>9} | {:>22} {:>22} {:>22}", "pop", "all dormant", "cascades woken", "mass assault");
    for pop in [20_000usize, 50_000, 100_000] {
        // (a) everyone asleep — the keep-index cost of a huge indexed carpet.
        let mut h = Horde::new(7, pop);
        h.step(1.0 / 60.0); // build the index
        let (ms_a, _) = measure(&mut h, secs);
        // (b) wake ~10%: detonate big noises over a few nests.
        let mut woken = 0usize;
        for k in 0..(pop / 1500).max(1) {
            let p = h.units[(k * 1499) % h.units.len()].p;
            h.emit_noise(p, 2000.0);
            h.step(1.0 / 60.0);
            let (_, a) = h.counts();
            woken = a;
            if a * 10 >= pop { break; }
        }
        let (ms_b, act_b) = measure(&mut h, secs);
        let _ = woken;
        // (c) a wave assault mid-flight (spawned marching column + the fights).
        let mut hw = Horde::new(7, pop);
        hw.step(1.0 / 60.0);
        for k in 0..(pop / 10) {
            let a = k as f64 * 0.001;
            let r = WORLD / 2.0 - 40.0 - (k % 37) as f64;
            let (x, z) = (WORLD / 2.0 + a.cos() * r, WORLD / 2.0 + a.sin() * r);
            hw.spawn_zombie(vectorial_hash_demos::horde_sim::ZClass::Walker, x.clamp(2.0, WORLD - 2.0), z.clamp(2.0, WORLD - 2.0), ZState::Marching);
        }
        for _ in 0..600 { hw.step(1.0 / 60.0); } // let the wave close + fights start
        let (ms_c, act_c) = measure(&mut hw, secs);
        println!("{:>9} | {:>10.2}ms ({:>5.0}fps) {:>7.2}ms ({:>4.0}fps) a={:<6} {:>4.2}ms ({:>4.0}fps) a={}",
            pop, ms_a, 1000.0 / ms_a, ms_b, 1000.0 / ms_b, act_b, ms_c, 1000.0 / ms_c, act_c);
    }
    println!("\nreading: the dormant column is the keep-index headline — the sleepers'\nmaintenance is ~free (skipped by the moved-only sync), so cost scales with the\nACTIVE front, not the indexed population.");

    // ---- the flow-field experiment: single-CC goal vs the user's multi-source
    // idea (0-seed at EVERY live building, one flood). Time a rebuild in each
    // mode (min-of-N, warm) — the point is that N goals cost ~the same as 1,
    // because it's still a single Dijkstra; more seeds just start it wider.
    println!("\nflow-field rebuild — single-CC goal vs multi-building (the multi-source idea):");
    println!("{:>18} | {:>18} {:>18}", "map (150-cell)", "single-CC", "multi-building");
    for sc in [Scenario::Classic, Scenario::Patches] {
        let mut h = Horde::with_scenario(7, 5000, sc);
        let goals = h.structures.iter().filter(|s| s.hp > 0.0 && matches!(s.kind, SKind::House | SKind::Storehouse | SKind::CommandCenter)).count();
        let bench = |h: &mut Horde, multi: bool| -> f64 {
            h.set_flow_multi(multi);
            h.force_flow_rebuild(); // warm
            let mut best = f64::MAX;
            for _ in 0..200 { let t = Instant::now(); h.force_flow_rebuild(); best = best.min(t.elapsed().as_secs_f64()); }
            best * 1e6 // microseconds
        };
        let one = bench(&mut h, false);
        let many = bench(&mut h, true);
        println!("{:>18} | {:>13.1} us {:>13.1} us   ({goals} building goals)", sc.label(), one, many);
    }
    println!("reading: seeding every building barely moves the rebuild cost — it's one\nDijkstra either way — but the field now points each zombie at its NEAREST\nbuilding and re-routes to the next as buildings fall (the `O` toggle).");
}
