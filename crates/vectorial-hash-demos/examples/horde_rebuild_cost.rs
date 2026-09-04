//! #166 — how long does the horde block for when you press `G` or move the population slider?
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_rebuild_cost --release
//! ```
//!
//! `adaptive_lab` hung for seconds because `resize` ran unbounded inside one frame, and the fix
//! there was to spread it: the knob became a target and the population walks toward it. The horde
//! has the same *shape* — `set_population` and `set_scenario` both call `Horde::with_scenario`,
//! rebuilding the whole world — but **not the same fix available**. A new scenario is atomic: you
//! cannot render half the old map and half the new one, so there is nothing to spread.
//!
//! When work cannot be spread, the honest options are to make it faster or to say it is happening.
//! Which one is worth doing depends on how long it actually is, and nobody had measured it. A
//! 40 ms hitch needs nothing; a two-second freeze is indistinguishable from a hang, which is
//! exactly how the lab's was reported.
//!
//! Wall time, minimum of a few, because the question is the floor of the stall rather than its
//! distribution — and an order of magnitude survives this laptop's noise even where a percentage
//! would not.
use std::time::Instant;
use vectorial_hash_demos::horde_sim::{Horde, Scenario};

const POPS: [usize; 5] = [2_000, 10_000, 30_000, 60_000, 100_000];
const REPS: usize = 3;

fn best<F: FnMut()>(mut f: F) -> f64 {
    let mut lo = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        lo = lo.min(t.elapsed().as_secs_f64() * 1e3);
    }
    lo
}

fn main() {
    println!("horde rebuild cost (#166) — one frame's work when the world changes, min of {REPS}\n");
    println!("{:>10} {:>12} {:>12} {:>12} {:>12}", "pop", "OPEN ms", "FOREST ms", "PATCHES ms", "frames@60");
    let mut worst: f64 = 0.0;
    for pop in POPS {
        let mut ms = [0.0f64; 3];
        for (i, sc) in [Scenario::Classic, Scenario::Forest, Scenario::Patches].into_iter().enumerate() {
            ms[i] = best(|| { std::hint::black_box(Horde::with_scenario(7, pop, sc)); });
        }
        let hi = ms.iter().copied().fold(0.0f64, f64::max);
        worst = worst.max(hi);
        println!("{pop:>10} {:>12.0} {:>12.0} {:>12.0} {:>12.0}", ms[0], ms[1], ms[2], hi / 16.7);
    }
    println!();
    println!("That last column is how many 60 Hz frames the window is not answering for. Under");
    println!("about 3 it is a hitch nobody reports; past ~30 (half a second) it reads as a hang,");
    println!("and past ~120 people close the window.");
    println!();
    if worst > 500.0 {
        println!("Worst case {worst:.0} ms. This cannot be spread — a new scenario is atomic, and the");
        println!("population slider rebuilds the dormant field the whole sim is arranged around. So");
        println!("the fix is not the lab's: it is to SAY it is happening. A frame that draws");
        println!("REBUILDING and then blocks is honest; one that simply stops is not, and the");
        println!("viewer cannot tell the second from a crash.");
    } else {
        println!("Worst case {worst:.0} ms — a hitch, not a hang. Nothing to do but record that it");
        println!("was checked, so the next person does not re-open the question from the same");
        println!("suspicion.");
    }
}
