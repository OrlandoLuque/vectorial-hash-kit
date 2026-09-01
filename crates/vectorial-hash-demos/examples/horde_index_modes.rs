//! The same battle, three indexes: does letting it choose actually pay here?
//!
//! `adaptive_vs_pinned` asks this on a synthetic script and `fluid_wgpu` on a workload that never
//! changes. The horde is the case the layer was built for — a dormant carpet that never moves and
//! is barely queried, becoming an assault where everything relocates and every awake unit culls
//! its neighbourhood — so this runs one seeded battle three times and reports what each cost.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_index_modes --release
//! ```
//!
//! Read it as a comparison BETWEEN THE THREE ARMS on whatever machine ran it. The absolute
//! milliseconds are that machine's; the ranking is the answer.
use std::time::Instant;
use vectorial_hash_demos::horde_sim::{Horde, ZMode};

const SEED: u64 = 23;
const POP: usize = 30_000;
const FRAMES: usize = 900;
const DT: f64 = 1.0 / 30.0;

fn main() {
    println!("one seeded battle ({POP} zombies, {FRAMES} frames), three index modes\n");
    println!("{:>10} {:>12} {:>12} {:>10} {:>26}", "mode", "total ms", "ms/step", "kills", "notes");

    let mut rows: Vec<(String, f64)> = Vec::new();
    for mode in [ZMode::Tree, ZMode::Morton, ZMode::Adaptive] {
        let mut h = Horde::new(SEED, POP);
        h.set_zmode(mode);
        // Waves at fixed frames so all three arms live the same battle: the point is to cross
        // the quiet -> assault boundary, which is where a fixed choice has to be wrong at one
        // end or the other.
        let t = Instant::now();
        for f in 0..FRAMES {
            if f == 60 || f == 300 || f == 600 { h.trigger_wave(); }
            h.step(DT);
        }
        let total = t.elapsed().as_secs_f64() * 1e3;

        let note = if mode == ZMode::Adaptive {
            let st = h.zadapt.stats();
            let hot = st.hottest_pair().map_or("-".to_string(), |(a, b, n)| format!("{a:?}->{b:?} x{n}"));
            format!("{} sw, {} near, {hot}", st.switches(), st.near_misses)
        } else { String::new() };
        println!("{:>10} {:>12.0} {:>12.3} {:>10} {:>26}", mode.label(), total, total / FRAMES as f64, h.kills, note);
        rows.push((mode.label().to_string(), total));
    }

    rows.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (best, best_ms) = rows[0].clone();
    let adaptive = rows.iter().find(|r| r.0 == "ADAPTIVE").map(|r| r.1).unwrap_or(f64::NAN);
    println!("\nfastest: {best} ({best_ms:.0} ms). Adaptive is {:.2}x the best fixed choice.",
             adaptive / rows.iter().filter(|r| r.0 != "ADAPTIVE").map(|r| r.1).fold(f64::INFINITY, f64::min));
    println!();
    println!("What to look for, in order of what it would actually mean:");
    println!("  · 0 switches   -> the battle never crossed a threshold. The demo is not exercising");
    println!("                    the policy and the number above says nothing about it.");
    println!("  · many near    -> it wanted to move and hysteresis held it. If that comes with a");
    println!("                    poor time, the band is sitting on top of this workload.");
    println!("  · < 1.0x       -> it beat every fixed choice, which is the case that justifies the");
    println!("                    layer: no single structure was right for the whole battle.");
    println!("  · > 1.0x       -> insurance, not optimisation — the honest headline everywhere");
    println!("                    else in this kit. Pin the winner if you can predict the fight.");
}
