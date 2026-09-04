//! #165 — does the horde's decision stagger *alias*? Answered on outcomes, not by argument.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_bucket_phase --release
//! ```
//!
//! The horde re-decides a zombie away from the walls every `decide_n` frames, staggered by id:
//! `(frame + id) % decide_n == 0`. That is a **stride**. Consecutive frames select disjoint sets,
//! and each unit's phase is a fixed linear function of its id.
//!
//! Striding is correct here in the way it was **not** correct in `adaptive_lab`, and the
//! difference is worth stating because it is the whole content of this audit. In the lab the
//! sampled set *was the picture*: which agents got queried decided which dots lit, so a
//! partitioning schedule made two colourings alternate at 30 Hz. In the horde the sampled set is
//! only *who re-plans*; position advances every frame on cached velocity, so nothing a viewer sees
//! is sampled. **A partitioning schedule is a display bug only where the display reads the
//! sample.**
//!
//! What remains is not visual: a fixed phase can put a whole class of units in a fixed
//! relationship to some other periodic system — tower reload, the commander's 1 Hz sweep, the
//! wave clock. Half the units would then always decide just after a volley and half just before.
//! That is a real possibility and reasoning about it proves nothing, so this measures it:
//! `$HORDE_DECIDE_HASH=1` scatters the phase through a hash of the id, keeping the same per-unit
//! rate and destroying any linear relationship. If the stride aliases, outcomes shift.
//!
//! **Outcomes, not milliseconds**: kills, structures standing, units alive. They are integers
//! produced by deterministic arithmetic, so this says the same thing on any machine — which is
//! the only kind of measurement worth trusting on an unfamiliar laptop.
use vectorial_hash_demos::horde_sim::Horde;

const POP: usize = 20_000;
const FRAMES: usize = 900;
const DT: f64 = 1.0 / 30.0;
const SEEDS: [u64; 6] = [1, 7, 23, 42, 101, 512];

/// One battle. Returns the outcome triple.
fn run(seed: u64) -> (u64, usize, usize) {
    let mut h = Horde::new(seed, POP);
    for f in 0..FRAMES {
        if f == 60 || f == 300 || f == 600 { h.trigger_wave(); }
        h.step(DT);
    }
    let standing = h.structures.iter().filter(|s| s.hp > 0.0).count();
    let alive = h.units.iter().filter(|z| z.hp > 0.0).count();
    (h.kills, standing, alive)
}

fn main() {
    // The knob is read when a `Horde` is built, so each arm has to be a separate process — set
    // the variable here and the child inherits it. Running both in one process would have the
    // first arm's setting leak into the second, which is the kind of thing that produces two
    // identical columns and a confident wrong conclusion.
    let hashed = std::env::var("HORDE_DECIDE_HASH").is_ok();
    println!("horde decision-phase audit (#165) — {} phase, {POP} zombies, {FRAMES} frames",
             if hashed { "HASHED" } else { "STRIDED (default)" });
    println!("{:>8} {:>12} {:>12} {:>12}", "seed", "kills", "standing", "alive");
    let mut rows = Vec::new();
    for seed in SEEDS {
        let (k, s, a) = run(seed);
        println!("{seed:>8} {k:>12} {s:>12} {a:>12}");
        rows.push((k, s, a));
    }
    let n = rows.len() as f64;
    let mk = rows.iter().map(|r| r.0 as f64).sum::<f64>() / n;
    let ms = rows.iter().map(|r| r.1 as f64).sum::<f64>() / n;
    let ma = rows.iter().map(|r| r.2 as f64).sum::<f64>() / n;
    println!("{:>8} {mk:>12.0} {ms:>12.1} {ma:>12.0}", "mean");
    println!("#M horde_phase.kills {mk:.0} n");
    println!("#M horde_phase.standing {ms:.2} n");
    println!("#M horde_phase.alive {ma:.0} n");
    println!();
    println!("Run it again with HORDE_DECIDE_HASH=1 and compare the means. The seeds are the");
    println!("control: a difference smaller than the spread ACROSS seeds is not a difference,");
    println!("and six battles is enough to see that without pretending it is a distribution.");
}
