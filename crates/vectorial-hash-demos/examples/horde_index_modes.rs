//! The same battle, three indexes: does letting it choose actually pay here?
//!
//! `adaptive_vs_pinned` asks this on a synthetic script and `fluid_wgpu` on a workload that never
//! changes. The horde is the case the layer was built for — a dormant carpet that never moves and
//! is barely queried, becoming an assault where everything relocates and every awake unit culls
//! its neighbourhood — so this runs one seeded battle under each index and reports what it cost.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_index_modes --release
//! ```
//!
//! Read it as a comparison BETWEEN THE THREE ARMS on whatever machine ran it. The absolute
//! milliseconds are that machine's; the ranking is the answer.
//!
//! **Repeated, with the arm order rotated, and quoted as a median with its range.** The first
//! version ran each arm once, to completion, in a fixed order, and reported the adaptive index at
//! **1.03x** the best fixed choice. Five runs of that version gave **1.03, 1.36, 1.47, 1.67,
//! 1.63** — the published figure was the luckiest draw. A 900-frame battle takes long enough that
//! the machine drifts underneath the three arms, so whoever runs first meets a different machine
//! than whoever runs last. That is `docs/MEASURING.md` § 7, committed in a brand-new example by
//! the person who wrote § 7. Rotating the order each round and taking the median of the
//! **per-round** ratios is the same fix `common::compare2` applies at micro scale.
use std::time::Instant;
use vectorial_hash_demos::horde_sim::{Horde, ZMode};

const SEED: u64 = 23;
const POP: usize = 30_000;
const FRAMES: usize = 900;
const DT: f64 = 1.0 / 30.0;
const ROUNDS: usize = 3;

/// One battle under one index. Returns the wall time and, for the adaptive arm, what it saw.
fn run(mode: ZMode) -> (f64, String) {
    let mut h = Horde::new(SEED, POP);
    h.set_zmode(mode);
    let t = Instant::now();
    for f in 0..FRAMES {
        // Waves at fixed frames so every arm lives the same battle: the point is crossing the
        // quiet -> assault boundary, which is where a fixed choice has to be wrong at one end.
        if f == 60 || f == 300 || f == 600 { h.trigger_wave(); }
        h.step(DT);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let note = if mode == ZMode::Adaptive {
        let st = h.zadapt.stats();
        let hot = st.hottest_pair().map_or("-".into(), |(a, b, n)| format!("{a:?}->{b:?} x{n}"));
        let (n, q, mv) = h.zadapt.observed();
        // WHERE this workload sits in the (churn x query-load) plane — the two numbers the policy
        // decides from. Printed because `grid_tree_frontier` maps that plane on uniform data and
        // reaches a DIFFERENT conclusion at this very point, and a contradiction between two of
        // our own measurements is only resolvable if each one says where it stood.
        format!("{} sw, {} near, {hot} | saw {n} items, {q:.4} q/item, {mv:.4} mv/item",
                st.switches(), st.near_misses)
    } else { String::new() };
    (ms, note)
}

fn main() {
    println!("one seeded battle ({POP} zombies, {FRAMES} frames), three index modes");
    println!("{ROUNDS} rounds, arm order rotated each round, median of the per-round ratios\n");

    let modes = [ZMode::Tree, ZMode::Morton, ZMode::Adaptive];
    let mut ms: Vec<Vec<f64>> = vec![Vec::new(); 3];
    let mut adaptive_note = String::new();
    let mut ratios: Vec<f64> = Vec::new();

    for r in 0..ROUNDS {
        let mut this = [0.0f64; 3];
        for k in 0..3 {
            let i = (k + r) % 3; // rotate: no arm always meets the machine in the same state
            let (t, note) = run(modes[i]);
            this[i] = t;
            ms[i].push(t);
            if i == 2 { adaptive_note = note; }
        }
        // Formed WITHIN a round, from arms that met the same machine.
        ratios.push(this[2] / this[0].min(this[1]));
    }

    let med = |v: &mut Vec<f64>| { v.sort_by(f64::total_cmp); v[v.len() / 2] };
    println!("{:>10} {:>14} {:>12}   {}", "mode", "median ms", "ms/step", "notes");
    for (i, m) in modes.iter().enumerate() {
        let t = med(&mut ms[i]);
        println!("{:>10} {:>14.0} {:>12.3}   {}", m.label(), t, t / FRAMES as f64,
                 if i == 2 { adaptive_note.as_str() } else { "" });
    }

    ratios.sort_by(f64::total_cmp);
    let (lo, hi, mid) = (ratios[0], ratios[ratios.len() - 1], ratios[ratios.len() / 2]);
    println!("\nadaptive vs the best fixed choice: median {mid:.2}x   (range {lo:.2}-{hi:.2})");
    if hi / lo > 1.15 {
        println!("  that spread is wide enough that any SINGLE round would have been a story rather");
        println!("  than a measurement — quote the median and the range, never one round");
    }

    println!();
    println!("What to look for, in order of what it would actually mean:");
    println!("  · 0 switches -> the battle never crossed a threshold, and the times above say");
    println!("                  nothing about the policy at all");
    println!("  · many near  -> it wanted to move and hysteresis held it; with a poor time, the");
    println!("                  band is sitting on top of this workload");
    println!("  · < 1.0x     -> it beat every fixed choice: no single structure was right for the");
    println!("                  whole battle, which is the case that justifies the layer");
    println!("  · > 1.0x     -> insurance, not optimisation. Weigh it against what the WRONG fixed");
    println!("                  choice costs here, which is the MORTON row.");
}
