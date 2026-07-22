//! Headless BALANCE harness — run the horde sim with no graphics and log the
//! defence outcome per population, so tuning is measured, not guessed. For each
//! (pop, scenario) it steps the sim to game-over (or a time cap), snapshotting the
//! wave number, Command-Center HP, live fighters and awake zombies every ~60 s.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_balance --release
//! ```
//! Env: `BAL_POPS=2000,20000,100000` · `BAL_T=500` (time cap) · `BAL_SEED=7`.
use vectorial_hash_demos::horde_sim::{Horde, SKind, Scenario};

fn run(pop: usize, sc: Scenario, seed: u64, max_t: f64) {
    let mut h = Horde::with_scenario(seed, pop, sc);
    let dt = 1.0 / 20.0; // sim step (coarse but the balance is dt-robust); faster than 60
    let cc_max = SKind::CommandCenter.max_hp();
    let f0 = h.defenders.iter().filter(|d| d.kind.fighter()).count();
    println!("── pop {pop:>6}  {sc:?}  (seed {seed}) — {f0} fighters ──");
    let (mut t, mut next_snap) = (0.0f64, 0.0f64);
    loop {
        if t >= next_snap {
            let cc = h.structures.iter().find(|s| s.kind == SKind::CommandCenter).map(|s| s.hp).unwrap_or(0.0);
            let fa = h.defenders.iter().filter(|d| d.kind.fighter() && d.alive()).count();
            let walls = h.structures.iter().filter(|s| matches!(s.kind, SKind::Wall | SKind::Gate | SKind::Tower) && s.hp > 0.0).count();
            let (_, act) = h.counts();
            println!("   t={t:>4.0}s  wave {:>2}  CC {:>3.0}%  fighters {fa:>3}/{f0}  walls {walls:>2}  awake {act}", h.wave_k, 100.0 * cc / cc_max);
            next_snap += 60.0;
        }
        if let Some((got, vic)) = h.game_over {
            println!("   → {} at t={got:.0}s, wave {}\n", if vic { "VICTORY" } else { "DEFEAT " }, h.wave_k);
            break;
        }
        if t >= max_t {
            let fa = h.defenders.iter().filter(|d| d.kind.fighter() && d.alive()).count();
            println!("   → HELD to t={max_t:.0}s, wave {} ({fa}/{f0} fighters up)\n", h.wave_k);
            break;
        }
        h.step(dt);
        t += dt;
    }
}

fn main() {
    let pops: Vec<usize> = std::env::var("BAL_POPS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2000, 20000, 100000]);
    let max_t: f64 = std::env::var("BAL_T").ok().and_then(|s| s.parse().ok()).unwrap_or(500.0);
    let seed: u64 = std::env::var("BAL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    println!("horde balance harness — Patches, to game-over or t={max_t:.0}s\n");
    for &pop in &pops {
        run(pop, Scenario::Patches, seed, max_t);
    }
}
