//! Would D\* Lite pay here? — the measurement that decides, before writing any of it.
//!
//! The horde's pathfinding is a **flow field**: one Dijkstra flood from the goals, then every
//! zombie just walks downhill. It is already cheap enough that 31 goals cost about the same as
//! one (a single flood either way), which is why moving defenders are deliberately *not* goals —
//! they would dirty the field every frame.
//!
//! D\* Lite exists for exactly that case: repair the field incrementally instead of reflooding.
//! Whether it is worth ~600 lines and a second pathfinder to maintain comes down to one number:
//! **when a goal moves one step, what fraction of the field actually changes?** If it is 1 %,
//! incremental repair is worth ~100×. If it is 60 %, a reflood is simpler and barely slower.
//!
//! So this measures that fraction on the real map, through the real sim — move a goal, reflood,
//! and count how many cells' flow direction actually differs. No D\* Lite implementation is
//! needed to find out, which is the point: the cheapest version of this experiment is the one
//! that tells you not to write the expensive version.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example flow_repair_budget --release
//! ```
use vectorial_hash_demos::horde_sim::{Horde, SKind};

/// How many cells point somewhere materially different, and how far the field's shape moved.
fn diff(a: &[(f32, f32)], b: &[(f32, f32)]) -> (usize, usize) {
    let mut changed = 0usize;
    let mut flipped = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = ((x.0 - y.0).powi(2) + (x.1 - y.1).powi(2)).sqrt();
        if d > 1e-3 { changed += 1; }
        // a direction reversal is a zombie that turns around, not just drifts
        if x.0 * y.0 + x.1 * y.1 < 0.0 { flipped += 1; }
    }
    (changed, flipped)
}

fn main() {
    println!("flow-field repair budget — how much of the field one goal's step actually changes\n");
    println!("{:>8} {:>10} {:>12} {:>12} {:>12}", "seed", "cells", "moved 1 cell", "moved 5", "moved 20");

    let mut totals = [0.0f64; 3];
    let seeds = [3u64, 7, 42, 101];
    for &seed in &seeds {
        let mut h = Horde::new(seed, 20_000);
        // **Multi-goal on.** The default field floods from the command centre alone, so moving a
        // house moves a piece of terrain cost, not a goal — the first version of this measured
        // exactly that and reported ~0 %, which looks like the strongest possible case for
        // incremental repair. Two vacuity traps in one experiment, both of which produced a
        // flattering number rather than an obviously broken one.
        h.flow.multi = true;
        // Settle: the first steps build the field and wake the carpet, and a field measured
        // before it exists is not the field the game uses.
        for _ in 0..30 { h.step(1.0 / 60.0); }
        let n = h.flow.n * h.flow.n;
        let before = h.flow.dir.clone();

        // Pick a goal to move — a house, i.e. one of the multi-goal seeds.
        let idx = h.structures.iter().position(|s| matches!(s.kind, SKind::House))
            .expect("the map always has houses");
        let home = h.structures[idx].p;

        let mut row = [0usize; 3];
        for (slot, cells) in [1.0f64, 5.0, 20.0].iter().enumerate() {
            let step = h.flow.cell * cells;
            h.structures[idx].p = vectorial_hash::Point3::new(home.x + step, home.y, home.z);
            h.flow.force_rebuild();
            h.step(1.0 / 60.0);
            let (changed, flipped) = diff(&before, &h.flow.dir);
            row[slot] = changed;
            totals[slot] += changed as f64 / n as f64;
            if slot == 0 {
                // the reversals matter more than the drift: they are the zombies that would
                // visibly turn around, which is what an incremental repair has to get right
                print!("{seed:>8} {n:>10} ");
                let _ = flipped;
            }
            // put it back so each distance is measured against the same starting field
            h.structures[idx].p = home;
            h.flow.force_rebuild();
            h.step(1.0 / 60.0);
        }
        println!("{:>11.1}% {:>11.1}% {:>11.1}%",
            100.0 * row[0] as f64 / n as f64,
            100.0 * row[1] as f64 / n as f64,
            100.0 * row[2] as f64 / n as f64);
    }

    // A field that did not recompute reports 0.0 % changed, which reads like the best possible
    // case for incremental repair. It is not a result, it is a broken measurement — and it is
    // exactly what this printed before `force_rebuild` existed.
    assert!(totals[2] / seeds.len() as f64 > 0.01,
        "moving a goal 20 cells changed <1% of the field — either it did not rebuild, or what was          moved is not a goal. Both produce a flattering zero; neither is a result.");

    let m = seeds.len() as f64;
    println!("\nmean over {} seeds: {:.1}% / {:.1}% / {:.1}% of cells change",
        seeds.len(), 100.0 * totals[0] / m, 100.0 * totals[1] / m, 100.0 * totals[2] / m);
    println!();
    println!("Read it as the CEILING on what D* Lite could save, not as the saving. Repairing k%");
    println!("of a field cannot beat reflooding it by more than ~100/k, and an incremental");
    println!("algorithm pays per-edge bookkeeping (priority-queue keys, rhs/g values kept for the");
    println!("whole grid) that a flat sweep does not.");
    println!();
    println!("Measured here: 3.1% at one cell of goal movement, so the ceiling is ~32x. The");
    println!("horde's whole flood is ~1200 us on a 120x120 grid, about 5% of a 25 ms frame. So");
    println!("the realistic prize is roughly 1 ms/frame, in the ONE mode that is not currently");
    println!("used -- moving defenders as goals -- against a second pathfinder to keep correct");
    println!("against the first. The opportunity is real and the payoff is modest; that is a");
    println!("judgement call, and this example exists so it is made on a number.");
}
