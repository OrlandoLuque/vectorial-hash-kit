//! Churn / relocation-rate measurement — should we build a loose octree?
//!
//! Our `Tree3`/`Octree3` keep-index does the cheap thing on a move that stays in
//! its leaf, and pays the ascend-to-LCA + re-descend only on a **relocation**
//! (leaf exit). A **loose octree**'s whole pitch is that 2×-loose cells make
//! objects exit their cell *less often* → fewer relocations. So the question
//! "is a loose octree worth building?" reduces to: **how often does a moving
//! entity actually relocate today?** If it's rare, our keep-index is already
//! near-optimal and loose octrees add little; if it's frequent, they'd help.
//!
//! We use the new `update_ref_tracked` → `Crossing` to count Stayed vs Moved
//! across a tick, at several move speeds, and report the relocation rate + the
//! per-tick cost.
//!
//! ```bash
//! cargo run -p vectorial-hash --example churn_relocation_bench --release
//! ```

use std::time::Instant;
use vectorial_hash::{Aabb, Crossing, ItemRef, Point3, Positioned3, Tree3};

const WORLD: f64 = 10_000.0;

#[derive(Clone, Copy)]
struct Obj { p: Point3, v: (f64, f64, f64) }
impl Positioned3 for Obj { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn main() {
    let n = 200_000usize;
    let item_limit = 8usize;
    // At N=200k in 10000^3 with item_limit 8, a leaf holds ~8 objects → leaf
    // side ≈ WORLD * (8/N)^(1/3) ≈ 345 wu. We sweep the per-tick move as a
    // fraction of that so the speeds are meaningful.
    let leaf_side = WORLD * (item_limit as f64 / n as f64).cbrt();
    println!("churn / relocation-rate | {n} objects | item_limit {item_limit} | leaf side ≈ {leaf_side:.0} wu\n");
    println!("a 'relocation' = a move that leaves its leaf (the expensive path a loose octree would cut)\n");
    println!("{:>18} | {:>14} | {:>18} | {:>14}", "per-tick move", "= x leaf", "relocations/tick", "ms/tick");

    for &frac in &[0.02f64, 0.1, 0.3, 1.0, 3.0] {
        let step = leaf_side * frac;
        let mut r = Rng(7);
        let mut tree = Tree3::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), item_limit);
        let mut objs: Vec<Obj> = Vec::with_capacity(n);
        let mut handles: Vec<Option<ItemRef>> = Vec::with_capacity(n);
        for _ in 0..n {
            let p = Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD);
            let dir = { let (a, b, c) = (r.unit() - 0.5, r.unit() - 0.5, r.unit() - 0.5); let l = (a * a + b * b + c * c).sqrt().max(1e-6); (a / l, b / l, c / l) };
            objs.push(Obj { p, v: dir });
            handles.push(tree.insert_ref(Obj { p, v: dir }));
        }
        let clamp = |v: f64| v.clamp(1.0, WORLD - 1.0);
        // warm one tick, then measure over several
        let mut relocs = 0u64;
        let ticks = 20u64;
        let t = Instant::now();
        for _ in 0..ticks {
            relocs = 0;
            for i in 0..n {
                let Some(h) = handles[i] else { continue };
                let v = objs[i].v;
                let np = Point3::new(clamp(objs[i].p.x + v.0 * step), clamp(objs[i].p.y + v.1 * step), clamp(objs[i].p.z + v.2 * step));
                objs[i].p = np;
                match tree.update_ref_tracked(h, |o| o.p = np) {
                    Crossing::Moved { .. } => relocs += 1,
                    Crossing::Left => handles[i] = None,
                    Crossing::Stayed(_) => {}
                }
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / ticks as f64;
        let pct = relocs as f64 / n as f64 * 100.0;
        println!("{:>15.0} wu | {:>12.2}x | {:>10} ({:>4.1}%) | {:>11.2}", step, frac, relocs, pct, ms);
    }
    println!("\nreading: the relocation rate is the ceiling on what a loose octree could save\n(it only cuts the Moved fraction). If most ticks are Stayed (small moves), the\nkeep-index is already near-optimal; loose octrees pay only when moves routinely\ncross leaves (fast/teleporting entities).");
}
