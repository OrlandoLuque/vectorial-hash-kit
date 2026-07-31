//! `grid_keep_bench` — the grid that keeps, against the grid that rebuilds.
//!
//! "Trees keep, grids rebuild" is one of this repo's load-bearing sentences, and it was a
//! property of the API rather than of grids. A uniform grid has no handles to maintain and
//! nothing to rebalance: told where an item *was*, it can find it in that one cell, and if the
//! item has not left the cell there is **nothing to do at all**. `MortonGrid3::update` does
//! that; this measures whether it was worth having.
//!
//! The variable is the **fraction of the population that moves in a frame**, because that is
//! what separates the two strategies. A rebuild costs the same whether one item moved or all of
//! them did. Keeping costs only what moved — which is why the horde, whose 50 000 units are
//! nearly all asleep, was paying 50 000 insertions a frame to relocate a few dozen.
//!
//! ```bash
//! cargo run -p vectorial-hash --example grid_keep_bench --release
//! ```
//! Env: `GK_N` (population), `GK_LEVELS`, `GK_FRAMES`.

#[path = "common/mod.rs"]
mod common;

use vectorial_hash::linear_octree3::LinearOctree3;
use vectorial_hash::morton3::Crossed;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3};

const W: f64 = 1000.0;

#[derive(Clone, Copy)]
struct M {
    id: u32,
    p: Point3,
}
impl Positioned3 for M {
    fn position(&self) -> Point3 {
        self.p
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f64) / (1u64 << 24) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}

fn main() {
    let n: usize = std::env::var("GK_N").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let levels: u32 = std::env::var("GK_LEVELS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let frames: usize = std::env::var("GK_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    let mut rng = Rng(0x5EED_1234);
    let base: Vec<M> = (0..n)
        .map(|i| M { id: i as u32, p: Point3::new(rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1)) })
        .collect();
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);

    let g0: MortonGrid3<M> = {
        let mut g = MortonGrid3::new(world, levels);
        for it in &base {
            g.insert(*it);
        }
        g
    };
    let occ = g0.occupancy();
    println!("grid keep vs rebuild | {n} points | levels {levels} | {frames} frames per sample");
    println!("cells {} | mean {:.1} items/cell | max {}\n", occ.cells, occ.mean, occ.max);
    println!("{:<10} {:>13} {:>13} {:>11} {:>22}", "moving", "keep ms/frame", "rebuild ms", "speed-up", "crossings");

    for pct in [100.0f64, 50.0, 10.0, 1.0, 0.1] {
        let movers = ((n as f64 * pct / 100.0) as usize).max(1);

        // One shared movement script, so both strategies see exactly the same frame.
        let mut script = Rng(0xA11CE);
        let steps: Vec<Point3> = (0..movers * frames)
            .map(|_| Point3::new(script.r(-40.0, 40.0), script.r(-40.0, 40.0), script.r(-40.0, 40.0)))
            .collect();

        // ---- keep: only the movers are touched, and only the ones that leave re-bucket.
        let mut kept: MortonGrid3<M> = MortonGrid3::new(world, levels);
        for it in &base {
            kept.insert(*it);
        }
        let mut pos: Vec<Point3> = base.iter().map(|m| m.p).collect();
        let (mut stayed, mut moved) = (0u64, 0u64);
        let keep_ms = common::wall_ms(3, || {
            for f in 0..frames {
                for i in 0..movers {
                    let d = steps[f * movers + i];
                    let old = pos[i];
                    let np = Point3::new((old.x + d.x).clamp(0.0, W - 0.1), (old.y + d.y).clamp(0.0, W - 0.1), (old.z + d.z).clamp(0.0, W - 0.1));
                    let id = i as u32;
                    match kept.update(old, |it| it.id == id, |it| it.p = np) {
                        Crossed::Stayed => stayed += 1,
                        Crossed::Moved => moved += 1,
                        other => panic!("update failed: {other:?}"),
                    }
                    pos[i] = np;
                }
            }
        }) / frames as f64;

        // ---- rebuild: clear and refill the whole population, every frame.
        let mut items = base.clone();
        let mut rebuilt: MortonGrid3<M> = MortonGrid3::new(world, levels);
        let rebuild_ms = common::wall_ms(3, || {
            for f in 0..frames {
                for i in 0..movers {
                    let d = steps[f * movers + i];
                    let old = items[i].p;
                    items[i].p = Point3::new((old.x + d.x).clamp(0.0, W - 0.1), (old.y + d.y).clamp(0.0, W - 0.1), (old.z + d.z).clamp(0.0, W - 0.1));
                }
                rebuilt.clear();
                for it in &items {
                    rebuilt.insert(*it);
                }
            }
        }) / frames as f64;

        // Same answers, or the comparison is meaningless.
        assert_eq!(kept.item_count(), rebuilt.item_count());
        for (cx, cy, cz, r) in [(200.0, 200.0, 200.0, 60.0), (700.0, 400.0, 300.0, 90.0)] {
            let s = vectorial_hash::Sphere3::new(cx, cy, cz, r);
            let mut a: Vec<u32> = kept.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = rebuilt.cull(&s).iter().map(|m| m.id).collect();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "kept and rebuilt disagree at {pct}% moving");
        }

        let total = (stayed + moved).max(1);
        println!(
            "{:<10} {keep_ms:>13.4} {rebuild_ms:>13.4} {:>10.2}x {:>21}",
            format!("{pct}%"),
            rebuild_ms / keep_ms,
            format!("{:.0}% stayed put", 100.0 * stayed as f64 / total as f64)
        );
        println!("#M moving{}.keep_ms {keep_ms:.5} ms", (pct * 10.0) as u32);
        println!("#M moving{}.speedup {:.3} x", (pct * 10.0) as u32, rebuild_ms / keep_ms);
    }

    // ---- the adaptive twin, and the price of keeping an adaptive structure ----------------
    // LinearOctree3 has the same bucket-hash storage and had the same omission. It also has
    // something the flat grid does not: adaptive depth. A kept copy holds splits made for a
    // distribution the points have left and never merges an emptied leaf, so its SHAPE drifts
    // from a rebuild's. The answers stay identical (tested); the question this measures is
    // whether the queries stay as fast.
    println!("\nLinearOctree3 — same sweep, plus what the shape drift costs");
    println!("{:<10} {:>13} {:>13} {:>11} {:>10} {:>12}", "moving", "keep ms/frame", "rebuild ms", "speed-up", "leaves", "cull vs fresh");
    for pct in [100.0f64, 10.0, 1.0] {
        let movers = ((n as f64 * pct / 100.0) as usize).max(1);
        let mut script = Rng(0xA11CE);
        let steps: Vec<Point3> = (0..movers * frames)
            .map(|_| Point3::new(script.r(-40.0, 40.0), script.r(-40.0, 40.0), script.r(-40.0, 40.0)))
            .collect();

        let mut kept = LinearOctree3::from_items(world, 16, 12, base.clone());
        let mut pos: Vec<Point3> = base.iter().map(|m| m.p).collect();
        let keep_ms = common::wall_ms(3, || {
            for f in 0..frames {
                for i in 0..movers {
                    let d = steps[f * movers + i];
                    let old = pos[i];
                    let np = Point3::new((old.x + d.x).clamp(0.0, W - 0.1), (old.y + d.y).clamp(0.0, W - 0.1), (old.z + d.z).clamp(0.0, W - 0.1));
                    let id = i as u32;
                    kept.update(old, |it| it.id == id, |it| it.p = np);
                    pos[i] = np;
                }
            }
        }) / frames as f64;

        let mut items = base.clone();
        let mut rebuild_ms = 0.0;
        let mut rebuilt = LinearOctree3::from_items(world, 16, 12, items.clone());
        rebuild_ms += common::wall_ms(3, || {
            for f in 0..frames {
                for i in 0..movers {
                    let d = steps[f * movers + i];
                    let old = items[i].p;
                    items[i].p = Point3::new((old.x + d.x).clamp(0.0, W - 0.1), (old.y + d.y).clamp(0.0, W - 0.1), (old.z + d.z).clamp(0.0, W - 0.1));
                }
                rebuilt = LinearOctree3::from_items(world, 16, 12, items.clone());
            }
        }) / frames as f64;

        // Same answers, still?
        for (cx, cy, cz, r) in [(200.0, 200.0, 200.0, 60.0), (700.0, 400.0, 300.0, 90.0)] {
            let s = vectorial_hash::Sphere3::new(cx, cy, cz, r);
            let mut a: Vec<u32> = kept.cull(&s).iter().map(|m| m.id).collect();
            let mut b: Vec<u32> = rebuilt.cull(&s).iter().map(|m| m.id).collect();
            a.sort_unstable(); b.sort_unstable();
            assert_eq!(a, b, "kept and rebuilt LinearOctree3 disagree at {pct}% moving");
        }

        // The drift, priced: the same culls against the kept tree and against a fresh one
        // built from its own current contents.
        let probes: Vec<Point3> = (0..200).map(|i| pos[(i * 7919) % pos.len()]).collect();
        let fresh = LinearOctree3::from_items(world, 16, 12, items.clone());
        let cull_kept = common::wall_ms(5, || { for q in &probes { std::hint::black_box(kept.cull(&vectorial_hash::Sphere3::new(q.x, q.y, q.z, 30.0)).len()); } });
        let cull_fresh = common::wall_ms(5, || { for q in &probes { std::hint::black_box(fresh.cull(&vectorial_hash::Sphere3::new(q.x, q.y, q.z, 30.0)).len()); } });

        println!("{:<10} {keep_ms:>13.4} {rebuild_ms:>13.4} {:>10.2}x {:>10} {:>11.2}x",
            format!("{pct}%"), rebuild_ms / keep_ms, kept.leaf_count(), cull_kept / cull_fresh);
        println!("#M lin_moving{}.speedup {:.3} x", (pct * 10.0) as u32, rebuild_ms / keep_ms);
        println!("#M lin_moving{}.cull_drift {:.3} x", (pct * 10.0) as u32, cull_kept / cull_fresh);
    }
    println!("  (leaves: a rebuild at 100% moving ends with {} — the kept tree's count is the drift.)",
        LinearOctree3::from_items(world, 16, 12, base.clone()).leaf_count());

    println!("\nreading: a rebuild costs the same however few items moved, so the speed-up is roughly");
    println!("the reciprocal of the moving fraction — until the moving fraction is high enough that");
    println!("finding each item in its cell costs more than just re-inserting everything. Where that");
    println!("crossover sits depends on cell occupancy (see the header), which is the same tuning");
    println!("knob that decides query cost.");
}
