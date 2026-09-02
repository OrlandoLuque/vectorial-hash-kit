//! Why does the horde's Morton arm cost several times its tree arm? Counted, not timed.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_query_counts --release \
//!     --features sim-counters,grid-stats
//! ```
//!
//! `horde_index_modes` measures the two arms in milliseconds and reports the grid ~3.8x worse.
//! A duration cannot say *why*, and on a loaded or unfamiliar machine it cannot even say *how
//! much* very precisely — the same battle read 6.087 and 4.643 ms/step on two nights. This
//! counts instead: every `position()` the index asks for (`--features sim-counters`) and every
//! cell the grid looks up (`--features grid-stats`). Both are exact integers computed from
//! deterministic arithmetic, so they are the same on this laptop, on the desktop, and in CI.
//!
//! **Result: the hypothesis below was REFUTED, and so was its successor.** k-NN costs the grid
//! only 1.6-1.9x the tree's points and 18-30 cell lookups — nothing. Maintenance was the next
//! suspect and is also innocent: with waves switched off and a **peak awake count of zero** the
//! grid's disadvantage does not shrink, it grows (6.51x to 8.18x). What is left is the standing
//! defence sweeping the map with large rings that find nothing, where a tree prunes empty space
//! for free and a grid pays a hash lookup per cell regardless. A radius-300 ring: 325 point tests
//! for the tree, 1001 points and **6072 cells** for the grid.
//!
//! **The hypothesis under test**, left over from the extent work: the horde's culls were fully
//! explained by query extent (its commonest is radius 3, where a tree wins), but the horde also
//! runs **k-NN** through the index — tower targeting at k=8 and the commander at k=48 — and
//! `tests/work_counts.rs` already blesses `knn3/morton3` at **179 636** points against
//! `knn3/tree3`'s **11 179** on 20k points. If k-NN is a large share of the horde's index work,
//! that is where the grid's loss lives, and the fix is to stop asking a grid for k-NN rather than
//! to tune the grid.
//!
//! What this deliberately does NOT do is convert counts into a predicted speed-up. A cell the
//! grid visits and finds empty costs a hash lookup and calls nobody's `position()`, which is why
//! the cells column exists at all — see `docs/MEASURING.md` § 8b. Counts prove an algorithmic
//! difference; they do not bound a timing one.
use vectorial_hash::{MortonGrid3, Point3, Sphere3};
use vectorial_hash_demos::horde_sim::{counters, zgrid_world, Horde, IZombie, ZMode, ZGRID_LEVELS, WORLD};

const SEED: u64 = 23;
const POP: usize = 30_000;
const FRAMES: usize = 900;
const DT: f64 = 1.0 / 30.0;

/// One battle under one index, returning the work it charged for.
fn run(mode: ZMode, waves: bool) -> (u64, u64, usize) {
    let mut h = Horde::new(SEED, POP);
    h.set_zmode(mode);
    // Build cost is not the question and would swamp the early frames, so the counters are
    // zeroed after construction rather than before it.
    let _ = counters::take();
    let _ = reset_cells();
    let mut awake_peak = 0usize;
    for f in 0..FRAMES {
        if waves && (f == 60 || f == 300 || f == 600) { h.trigger_wave(); }
        h.step(DT);
        awake_peak = awake_peak.max(h.units.iter().filter(|z| !z.dormant()).count());
    }
    (counters::take(), reset_cells(), awake_peak)
}

#[cfg(feature = "grid-stats")]
fn reset_cells() -> u64 { vectorial_hash::morton3::reset_cell_visits() }
#[cfg(not(feature = "grid-stats"))]
fn reset_cells() -> u64 { 0 }

fn main() {
    println!("one seeded battle ({POP} zombies, {FRAMES} frames), work COUNTED not timed\n");
    #[cfg(not(feature = "grid-stats"))]
    println!("(built without `grid-stats` — the cells column will read 0, and for a grid that is\n\
              exactly the blind spot the column exists to close. Re-run with the feature.)\n");

    // Two scenarios, because the ratio between them is the experiment. An ASSAULT has a large
    // moving front (maintenance + queries); a QUIET battle has almost nothing awake, so the
    // dormant carpet is skipped by the keep-index and queries are nearly all that is left. If
    // the grid's disadvantage is in its QUERIES the two ratios agree; if it is in its
    // MAINTENANCE the quiet ratio collapses.
    let mut ratios = Vec::new();
    for (scenario, waves) in [("assault (3 waves)", true), ("quiet (no waves)", false)] {
        println!("--- {scenario} ---");
        println!("{:>10} {:>16} {:>16} {:>13} {:>12}", "index", "points tested", "cells visited", "pts/frame", "peak awake");
        let mut rows = Vec::new();
        for (label, mode) in [("TREE3", ZMode::Tree), ("MORTON", ZMode::Morton), ("ADAPTIVE", ZMode::Adaptive)] {
            let (pos, cells, awake) = run(mode, waves);
            println!("{label:>10} {pos:>16} {cells:>16} {:>13.0} {awake:>12}", pos as f64 / FRAMES as f64);
            rows.push((pos, cells));
        }
        let r = rows[1].0 as f64 / rows[0].0 as f64;
        println!("  -> Morton tests {r:.2}x the tree points");
        println!();
        ratios.push(r);
    }

    let (assault, quiet) = (ratios[0], ratios[1]);
    println!("assault {assault:.2}x vs quiet {quiet:.2}x. For scale, `horde_index_modes` measures");
    println!("the same two arms at ~3.8x in TIME.");
    println!();
    if quiet < assault / 1.5 {
        println!("The disadvantage largely DISAPPEARS when nothing is moving, so it is not the");
        println!("queries -- it is MAINTENANCE. A grid update finds its item by scanning a");
        println!("predicate over everything in the cell, and the horde cells are 28 wu wide over a");
        println!("dense carpet, so every relocation tests a crowd. update_ref goes straight there.");
    } else if quiet > assault / 1.2 {
        println!("The disadvantage survives -- indeed GROWS -- with a peak awake count of zero,");
        println!("so it is neither relocation nor the front: it is the standing defence sweeping");
        println!("the map for zombies it never finds. Those are the big rings, and the per-shape");
        println!("table below prices them: a radius-300 ring costs the grid 6072 cell lookups");
        println!("against the tree's 325 point tests, because a tree prunes empty space for free");
        println!("and a grid pays a hash lookup per cell whether or not anything is in it.");
        println!();
        println!("That is exactly the mechanism Thresholds::grid_min_hits encodes -- so the horde");
        println!("CONFIRMS that rule while also being the shape its density estimator gets wrong");
        println!("(a carpet is a slab in a cube). Right mechanism, wrong input: see #154.");
    } else {
        println!("The two ratios differ, but not decisively. Maintenance and queries both");
        println!("contribute; do not attribute this to either alone on the strength of one run.");
    }

    per_shape();

    println!();
    println!("Both columns are exact integers on any machine, which is why this run is worth more");
    println!("here than a millisecond figure would be — see docs/MEASURING.md § 10.");
}

/// What ONE query of each shape costs, on the horde's real positions and its real indexes.
///
/// The totals above say the grid does 6.5x the work; they cannot say which call site spends it.
/// These are the actual radii from `horde_sim`'s call sites, put through the actual `zindex` and
/// `zmorton` after a real battle has arranged the units — not a synthetic point cloud, which is
/// the substitution that has produced two wrong conclusions in this repo already.
///
/// **Two battles, not one.** The first version of this ran a single `ZMode::Morton` battle on the
/// assumption that both indexes stay maintained, probed both, and printed a tree column of
/// zeroes: only the selected index is kept current, so the tree was empty. The sim is
/// deterministic, so the same seed gives the same positions in both runs, and the assertions
/// below check that rather than trusting it — an empty index answers every query for free and
/// would have looked like the tree winning by an infinite margin.
fn per_shape() {
    let battle = |mode| {
        let mut h = Horde::new(SEED, POP);
        h.set_zmode(mode);
        for f in 0..400 { if f == 60 || f == 300 { h.trigger_wave(); } h.step(DT); }
        h
    };
    let a = battle(ZMode::Tree);
    assert!(a.zindex.item_count() > 0, "the tree's index is empty — nothing would be measured");

    // Build the grid FROM the tree, so the two hold byte-identical contents. Two separate
    // battles will not do it: the first attempt ran one under each mode and they disagreed by
    // three units at radius 55, because a tree and a grid return their hits in different orders
    // and the sim picks targets from that order. The battles diverge — which is a real finding
    // about the demo (see HORDE.md), and fatal to a per-query comparison.
    let whole = Sphere3::new(WORLD / 2.0, 0.0, WORLD / 2.0, WORLD * 2.0);
    let all: Vec<IZombie> = a.zindex.cull(&whole).into_iter().copied().collect();
    assert_eq!(all.len(), a.zindex.item_count(), "the extraction cull missed part of the world");
    let mut grid: MortonGrid3<IZombie> = MortonGrid3::new(zgrid_world(), ZGRID_LEVELS);
    for z in &all { grid.insert(*z); }
    assert_eq!(grid.item_count(), a.zindex.item_count());

    // Aim at the thick of it: the centre of mass of the awake front, where every one of these
    // queries is actually issued. A probe into empty map would flatter the grid enormously.
    let awake: Vec<Point3> = a.units.iter().filter(|z| !z.dormant()).map(|z| z.p).collect();
    let n = awake.len().max(1) as f64;
    let c = Point3::new(awake.iter().map(|p| p.x).sum::<f64>() / n,
                        awake.iter().map(|p| p.y).sum::<f64>() / n,
                        awake.iter().map(|p| p.z).sum::<f64>() / n);

    println!("
per-query cost at the front ({} awake, {} indexed), one query of each real shape:",
             awake.len(), a.zindex.item_count());
    println!("{:>26} {:>11} {:>11} {:>11} {:>9}", "call site (radius)", "tree pts", "grid pts", "grid cells", "hits");
    let shapes: [(&str, f64); 6] = [
        ("separation (3)", 3.0), ("bite (4.5)", 4.5), ("guard ring (55)", 55.0),
        ("tower ring (84)", 84.0), ("pack ring (110)", 110.0), ("wave ring (300)", 300.0),
    ];
    let mut worst = ("", 0u64);
    for (name, r) in shapes {
        let s = Sphere3::new(c.x, c.y, c.z, r);
        let _ = counters::take(); let _ = reset_cells();
        let th = a.zindex.cull(&s).len();
        let tree_pts = counters::take();
        let _ = reset_cells();
        let gh = grid.cull(&s).len();
        let (grid_pts, grid_cells) = (counters::take(), reset_cells());
        assert_eq!(th, gh, "{name}: the two indexes disagree ({th} vs {gh}) — one of them is wrong");
        if grid_cells > worst.1 { worst = (name, grid_cells); }
        println!("{name:>26} {tree_pts:>11} {grid_pts:>11} {grid_cells:>11} {th:>9}");
    }
    // k-NN, the shape this example set out to convict.
    for k in [8usize, 48] {
        let _ = counters::take(); let _ = reset_cells();
        let _ = a.zindex.knn(c, k);
        let tree_pts = counters::take();
        let _ = reset_cells();
        let _ = grid.knn(c, k);
        let (grid_pts, grid_cells) = (counters::take(), reset_cells());
        println!("{:>26} {tree_pts:>11} {grid_pts:>11} {grid_cells:>11} {k:>9}", format!("k-NN k={k}"));
    }
    println!("
the grid's most expensive single query here is `{}`, at {} cell lookups.", worst.0, worst.1);
}
