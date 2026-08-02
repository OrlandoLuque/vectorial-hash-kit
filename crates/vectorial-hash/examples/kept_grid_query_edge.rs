//! Why does a **kept** grid answer queries faster than a rebuilt one holding the same points?
//!
//! Two independent benches now say it does. `adaptive_vs_pinned` saw the adaptive index's grid
//! query ~15 % faster than an identically-configured fixed one (#136), and the 3D decision map
//! saw `morton-keep` cull 1.18× faster than `morton` — same world, same `levels`, same items,
//! same query set. Two sightings in unrelated settings is no longer a quirk.
//!
//! The question splits cleanly, and the first half is decidable without a clock:
//!
//! 1. **Is either grid doing less work?** With `grid-stats` on, every cell lookup and every
//!    `position()` is counted. If the counts are *equal*, no traversal difference exists and
//!    the gap is entirely in the memory system — which no amount of algorithm reading would
//!    have found.
//! 2. **If it is memory, what about it?** The candidate is allocation history. A rebuilt grid
//!    allocates ~20 000 bucket `Vec`s from scratch every frame, in the order items arrive,
//!    which is spatially arbitrary. A kept grid allocated its buckets once and has been
//!    swapping items between them since. So a *third* grid is built here: rebuilt from
//!    scratch, but in **Z-order** — the warm start. If allocation order is the cause, the warm
//!    rebuild should close most of the gap despite being just as freshly built as the cold one.
//!
//! ```bash
//! cargo run -p vectorial-hash --example kept_grid_query_edge --release --features grid-stats
//! ```
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3};

#[derive(Clone, Copy)]
struct P { id: u32, p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const N: usize = 20_000;
const W: f64 = 512.0;
const FRAMES: usize = 150;   // as long as the decision map's measured window
const VISION: f64 = 36.0;
const CULLS: usize = 4_000;

fn main() {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    let mut rng = Rng(0xC0FFEE);
    let mut pos: Vec<Point3> = (0..N).map(|_| Point3::new(rng.f() * W, rng.f() * W, rng.f() * W)).collect();
    let vel: Vec<(f64, f64, f64)> = (0..N).map(|_| (rng.f() * 240.0 - 120.0, rng.f() * 240.0 - 120.0, rng.f() * 240.0 - 120.0)).collect();
    let levels = MortonGrid3::<P>::levels_for_cell_size(world, VISION);

    // The KEPT grid: built once, maintained through every frame — exactly what `morton-keep`
    // is when the decision map finally measures its cull.
    let mut kept = MortonGrid3::<P>::new(world, levels);
    for (i, p) in pos.iter().enumerate() { kept.insert(P { id: i as u32, p: *p }); }

    // A grid that is REBUILT every frame, kept alive across frames the same way the decision
    // map does it, so its allocator has seen the same number of build/drop cycles.
    let mut churned = MortonGrid3::<P>::new(world, levels);

    let dt = 1.0 / 60.0;
    for _ in 0..FRAMES {
        for i in 0..N {
            let old = pos[i];
            let (vx, vy, vz) = vel[i];
            let np = Point3::new(
                (old.x + vx * dt).rem_euclid(W),
                (old.y + vy * dt).rem_euclid(W),
                (old.z + vz * dt).rem_euclid(W));
            pos[i] = np;
            let id = i as u32;
            kept.update(old, |c| c.id == id, |c| c.p = np);
        }
        churned = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { churned.insert(P { id: i as u32, p: *p }); }
    }

    // The third grid: as freshly built as `churned`, but fed in the order the kept grid holds
    // its points — a warm start. Same contents, same levels, different arrival order.
    let zorder: Vec<P> = kept.iter_z_order().copied().collect();
    let mut warm = MortonGrid3::<P>::new(world, levels);
    for it in &zorder { warm.insert(*it); }

    assert_eq!(kept.item_count(), churned.item_count(), "the grids must hold the same points");
    assert_eq!(kept.item_count(), warm.item_count(), "the grids must hold the same points");

    // The same queries for all three, so any difference is the structure and not the workload.
    let probes: Vec<Sphere3> = (0..CULLS).map(|k| { let c = pos[(k * 7919) % N]; Sphere3::new(c.x, c.y, c.z, VISION) }).collect();

    // Answers must be identical before any timing is worth reading.
    let mut a: Vec<u32> = kept.cull(&probes[0]).iter().map(|x| x.id).collect();
    let mut b: Vec<u32> = churned.cull(&probes[0]).iter().map(|x| x.id).collect();
    let mut c: Vec<u32> = warm.cull(&probes[0]).iter().map(|x| x.id).collect();
    a.sort_unstable(); b.sort_unstable(); c.sort_unstable();
    assert_eq!(a, b, "kept and rebuilt must answer identically");
    assert_eq!(a, c, "warm-rebuilt must answer identically");
    assert!(!a.is_empty(), "the probe must hit something or the check proves nothing");

    println!("kept vs rebuilt grid, same {N} points, world {W}^3, levels {levels}, {CULLS} culls r={VISION}");
    println!("(all three hold identical contents; the first probe returns {} items)\n", a.len());

    let run = |g: &MortonGrid3<P>| -> (f64, u64, usize) {
        // warm the caches equally, then take the best of a few passes
        let mut hits = 0usize;
        for p in probes.iter().take(200) { hits += g.cull(p).len(); }
        #[cfg(feature = "grid-stats")]
        let cells = { vectorial_hash::morton3::reset_cell_visits(); 0u64 };
        #[cfg(not(feature = "grid-stats"))]
        let cells = 0u64;
        let _ = cells;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for p in &probes { hits += g.cull(p).len(); }
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / probes.len() as f64);
        }
        #[cfg(feature = "grid-stats")]
        let visited = vectorial_hash::morton3::reset_cell_visits() / 6; // 5 timed passes + 1 count pass
        #[cfg(not(feature = "grid-stats"))]
        let visited = 0u64;
        (best, visited, hits)
    };

    let (t_kept, v_kept, h_kept) = run(&kept);
    let (t_churn, v_churn, h_churn) = run(&churned);
    let (t_warm, v_warm, h_warm) = run(&warm);
    assert_eq!(h_kept, h_churn, "the same culls must return the same number of items");
    assert_eq!(h_kept, h_warm, "the same culls must return the same number of items");

    println!("{:>26} {:>12} {:>16} {:>10}", "grid", "us/cull", "cells visited", "vs rebuilt");
    println!("{:>26} {:>12.3} {:>16} {:>10.2}", "rebuilt every frame", t_churn, v_churn, 1.0);
    println!("{:>26} {:>12.3} {:>16} {:>10.2}", "kept (update in place)", t_kept, v_kept, t_churn / t_kept);
    println!("{:>26} {:>12.3} {:>16} {:>10.2}", "rebuilt in Z-order (warm)", t_warm, v_warm, t_churn / t_warm);

    // ------------------------------------------------------------------ the COLD probe
    //
    // Everything above measures 4 000 culls, five times over, on a grid nobody has touched
    // since. That is a warm cache by construction — and it is NOT what a frame loop does. The
    // 3D decision map runs **16 culls immediately after the maintain**, which is the first
    // touch of whatever the maintain just moved through memory, and there the kept grid reads
    // 1.09-1.17x faster than the rebuilt one however the arms are ordered.
    //
    // So: the same comparison again, cold. Per frame, move the points, maintain each grid the
    // way it is meant to be maintained, then time a handful of culls on each. The arms
    // alternate by frame parity so ordering cannot be the explanation this time either.
    {
        let mut kept2 = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { kept2.insert(P { id: i as u32, p: *p }); }
        let (mut t_kept_c, mut t_churn_c) = (0.0f64, 0.0f64);
        let mut sink = 0usize;
        const COLD_FRAMES: usize = 200;
        const COLD_CULLS: usize = 16;
        for f in 0..COLD_FRAMES {
            for i in 0..N {
                let old = pos[i];
                let (vx, vy, vz) = vel[i];
                let np = Point3::new((old.x + vx * dt).rem_euclid(W), (old.y + vy * dt).rem_euclid(W), (old.z + vz * dt).rem_euclid(W));
                pos[i] = np;
                let id = i as u32;
                kept2.update(old, |c| c.id == id, |c| c.p = np);
            }
            let mut churn2 = MortonGrid3::<P>::new(world, levels);
            for (i, p) in pos.iter().enumerate() { churn2.insert(P { id: i as u32, p: *p }); }

            let qs: Vec<Sphere3> = (0..COLD_CULLS)
                .map(|k| { let c = pos[(f * 131 + k * 7919) % N]; Sphere3::new(c.x, c.y, c.z, VISION) }).collect();
            let kept_first = f % 2 == 0;
            for pass in 0..2 {
                let t = Instant::now();
                if (pass == 0) == kept_first {
                    for q in &qs { sink += kept2.cull(q).len(); }
                    t_kept_c += t.elapsed().as_secs_f64() * 1e6 / COLD_CULLS as f64;
                } else {
                    for q in &qs { sink += churn2.cull(q).len(); }
                    t_churn_c += t.elapsed().as_secs_f64() * 1e6 / COLD_CULLS as f64;
                }
            }
        }
        std::hint::black_box(sink);
        let (ck, cc) = (t_kept_c / COLD_FRAMES as f64, t_churn_c / COLD_FRAMES as f64);
        println!("\nCOLD probe — {COLD_CULLS} culls right after the maintain, {COLD_FRAMES} frames, arms alternating:");
        println!("{:>26} {:>12}", "grid", "us/cull");
        println!("{:>26} {:>12.3}", "rebuilt every frame", cc);
        println!("{:>26} {:>12.3}   ({:.2}x)", "kept (update in place)", ck, cc / ck);
        println!("#M cold.kept_us {ck:.3} us");
        println!("#M cold.rebuilt_us {cc:.3} us");
        println!("#M cold.rebuilt_over_kept {:.3} x", cc / ck);
    }

    // ------------------------------------------------- the COLD probe, with company
    //
    // One hypothesis the six earlier ones did not cover. The decision map does not time these
    // two grids alone: seven other structures are maintained and culled in the same frame, and
    // their working sets pass through the cache in between. Rotating the arms equalises
    // *position* but not that — every arm is still preceded by everyone else's traffic.
    //
    // An isolated bench has no company at all, which is exactly why it might be blind to this.
    // So: the same cold probe, with a large scratch buffer walked between the maintain and the
    // culls to stand in for the other arms. If the gap appears here it is an interaction with
    // the rest of the frame rather than a property of either grid — and that would explain why
    // every isolated reproduction so far has read 1.00x.
    {
        let mut kept3 = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { kept3.insert(P { id: i as u32, p: *p }); }
        let mut scratch: Vec<u64> = (0..2_000_000u64).collect();   // ~16 MB, past any L3 here
        let (mut t_kept_c, mut t_churn_c) = (0.0f64, 0.0f64);
        let mut sink = 0usize;
        const CF: usize = 120;
        const CC: usize = 16;
        for f in 0..CF {
            for i in 0..N {
                let old = pos[i];
                let (vx, vy, vz) = vel[i];
                let np = Point3::new((old.x + vx * dt).rem_euclid(W), (old.y + vy * dt).rem_euclid(W), (old.z + vz * dt).rem_euclid(W));
                pos[i] = np;
                let id = i as u32;
                kept3.update(old, |c| c.id == id, |c| c.p = np);
            }
            let mut churn3 = MortonGrid3::<P>::new(world, levels);
            for (i, p) in pos.iter().enumerate() { churn3.insert(P { id: i as u32, p: *p }); }

            let qs: Vec<Sphere3> = (0..CC).map(|k| { let c = pos[(f * 131 + k * 7919) % N]; Sphere3::new(c.x, c.y, c.z, VISION) }).collect();
            let kept_first = f % 2 == 0;
            for pass in 0..2 {
                // the company: a stride walk that evicts, before EACH arm, symmetrically
                for j in (0..scratch.len()).step_by(8) { scratch[j] = scratch[j].wrapping_add(1); }
                sink += scratch[f % scratch.len()] as usize;
                let t = Instant::now();
                if (pass == 0) == kept_first {
                    for q in &qs { sink += kept3.cull(q).len(); }
                    t_kept_c += t.elapsed().as_secs_f64() * 1e6 / CC as f64;
                } else {
                    for q in &qs { sink += churn3.cull(q).len(); }
                    t_churn_c += t.elapsed().as_secs_f64() * 1e6 / CC as f64;
                }
            }
        }
        std::hint::black_box(sink);
        let (ck, cc) = (t_kept_c / CF as f64, t_churn_c / CF as f64);
        println!("
COLD probe WITH COMPANY — 16 MB walked before each arm, {CF} frames:");
        println!("{:>26} {:>12.3}", "rebuilt every frame", cc);
        println!("{:>26} {:>12.3}   ({:.2}x)", "kept (update in place)", ck, cc / ck);
        println!("#M polluted.rebuilt_over_kept {:.3} x", cc / ck);
    }

    #[cfg(not(feature = "grid-stats"))]
    println!("\n(cells visited reads 0 — re-run with `--features grid-stats` for the counts.)");
    #[cfg(feature = "grid-stats")]
    {
        println!("\nEqual cell counts mean neither grid traverses less: the difference is entirely");
        println!("in the memory system, which is invisible to every clock-free method the kit has.");
        println!("If the warm rebuild lands near the kept grid, arrival order — not the keeping —");
        println!("is what buys the query time, and a rebuild can have it for free.");
    }
}
