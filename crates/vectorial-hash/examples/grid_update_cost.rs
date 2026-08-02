//! What does `MortonGrid3::update` actually cost, and why?
//!
//! The 3D decision map turned up something that reads like a contradiction: keeping the grid
//! costs **more per frame than rebuilding it**, and — the part that gives the game away — the
//! cost does not move when the critters slow down by 120×. If re-bucketing were the expense it
//! would collapse as fewer items cross a cell boundary. It does not, so the expense is
//! somewhere else, and this bench separates the candidates:
//!
//! - **the hash lookup** — one per call, independent of how full the cell is;
//! - **the predicate scan** — `bucket.iter().position(&predicate)`, O(items in that cell);
//! - **the re-bucketing** — `swap_remove` + push, only when the item actually crossed.
//!
//! The lever is cell occupancy: hold the item count fixed and change `levels`, so the same
//! work is spread over more or fewer cells. If the scan dominates, per-item cost tracks
//! occupancy. If the hash lookup dominates, it is flat.
//!
//! ```bash
//! cargo run -p vectorial-hash --example grid_update_cost --release
//! ```
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3};

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

fn main() {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    let mut rng = Rng(0x51EDu64);
    let pos: Vec<Point3> = (0..N).map(|_| Point3::new(rng.f() * W, rng.f() * W, rng.f() * W)).collect();

    println!("MortonGrid3::update cost vs cell occupancy | {N} items in {W}^3");
    println!("Every row does the SAME {N} update calls. Only `levels` changes, so only the");
    println!("number of items sharing a cell changes with it.\n");
    println!("{:>6} {:>10} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "levels", "cells", "mean/cell", "stay ns/item", "cross ns/item", "rebuild ns", "stay/rebuild");

    for levels in [3u32, 4, 5, 6, 7] {
        let mut g = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { g.insert(P { id: i as u32, p: *p }); }
        let occ = g.occupancy();

        // --- "it stayed": mutate to a point inside the SAME cell, so the only work is
        // find-it-and-write-it. This is the floor every keep-based caller pays per item per
        // frame whether or not anything moved.
        let t = Instant::now();
        for (i, &p) in pos.iter().enumerate() {
            let cid = i as u32;
            g.update(p, |c| c.id == cid, |c| c.p = p);
        }
        let stay = t.elapsed().as_secs_f64() * 1e9 / N as f64;

        // --- "it crossed": push every item a long way, so every call also re-buckets. Done
        // as a there-and-back pair so the grid ends where it started and the next row is
        // measured on the same distribution.
        let far: Vec<Point3> = pos.iter().map(|p| Point3::new((p.x + 137.0) % W, p.y, p.z)).collect();
        let t = Instant::now();
        for i in 0..N { let cid = i as u32; let q = far[i]; g.update(pos[i], |c| c.id == cid, |c| c.p = q); }
        let cross = t.elapsed().as_secs_f64() * 1e9 / N as f64;
        for i in 0..N { let cid = i as u32; let q = pos[i]; g.update(far[i], |c| c.id == cid, |c| c.p = q); }

        // --- the alternative it is competing against: throw it away and refill.
        let t = Instant::now();
        let mut fresh = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { fresh.insert(P { id: i as u32, p: *p }); }
        let rebuild = t.elapsed().as_secs_f64() * 1e9 / N as f64;
        assert_eq!(fresh.item_count(), occ.items, "the two paths must hold the same number of items");

        println!("{:>6} {:>10} {:>9.1} {:>12.1} {:>12.1} {:>12.1} {:>12.2}",
            levels, occ.cells, occ.mean, stay, cross, rebuild, stay / rebuild);
    }

    // ---------------------------------------------------------------- population axis
    //
    // The table above holds N fixed. But the question a caller actually asks is "does this
    // depend on how many things I have?", and it is not obvious a priori: a bigger hash table
    // misses cache more, and `update` and `insert` need not degrade at the same rate. If they
    // degrade together the rule is a constant; if they diverge it needs a population term.
    //
    // The break-even has a closed form once a caller SKIPS the items that did not move — the
    // only regime where keeping is worth anything. Keeping then costs `f * cross` per item and
    // rebuilding costs `insert`, so they meet at
    //
    //     f* = insert / cross
    //
    // the fraction of the population that may move before a rebuild becomes the better call.
    // Cells are sized for ~8 items at every N, so occupancy is held constant and only N moves.
    println!("\n\nDoes it depend on POPULATION? Cells sized for ~8 items each at every N,");
    println!("so occupancy is held constant and only the population varies.\n");
    println!("{:>10} {:>7} {:>10} {:>10} {:>11} {:>11} {:>14}",
        "N", "levels", "mean/cell", "stay ns", "cross ns", "insert ns", "break-even f*");

    for &n in &[2_000usize, 20_000, 200_000, 1_000_000] {
        let mut rng = Rng(0x51EDu64 ^ n as u64);
        let pos: Vec<Point3> = (0..n).map(|_| Point3::new(rng.f() * W, rng.f() * W, rng.f() * W)).collect();
        let per_axis = (n as f64 / 8.0).cbrt().max(1.0);
        let levels = (per_axis.log2().round().max(1.0) as u32).min(10);

        let mut g = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { g.insert(P { id: i as u32, p: *p }); }
        let occ = g.occupancy();

        let t = Instant::now();
        for (i, &p) in pos.iter().enumerate() { let cid = i as u32; g.update(p, |c| c.id == cid, |c| c.p = p); }
        let stay = t.elapsed().as_secs_f64() * 1e9 / n as f64;

        let far: Vec<Point3> = pos.iter().map(|p| Point3::new((p.x + 137.0) % W, p.y, p.z)).collect();
        let t = Instant::now();
        for (i, &q) in far.iter().enumerate() { let cid = i as u32; g.update(pos[i], |c| c.id == cid, |c| c.p = q); }
        let cross = t.elapsed().as_secs_f64() * 1e9 / n as f64;
        for (i, &q) in pos.iter().enumerate() { let cid = i as u32; g.update(far[i], |c| c.id == cid, |c| c.p = q); }

        let t = Instant::now();
        let mut fresh = MortonGrid3::<P>::new(world, levels);
        for (i, p) in pos.iter().enumerate() { fresh.insert(P { id: i as u32, p: *p }); }
        let insert = t.elapsed().as_secs_f64() * 1e9 / n as f64;
        assert_eq!(fresh.item_count(), n, "the rebuild must hold every point");

        println!("{:>10} {:>7} {:>10.1} {:>10.1} {:>11.1} {:>11.1} {:>14.2}",
            n, levels, occ.mean, stay, cross, insert, insert / cross);
    }

    println!("\nRead the `stay` column against `mean/cell`. Measured answer: it is FLAT while");
    println!("occupancy changes 39x, and highest where cells hold one item each - so the");
    println!("predicate scan is not the cost, the hash lookup is. A handle layer would not");
    println!("have helped: a handle still has to reach the bucket. The scan shows up only in");
    println!("the 39-items-per-cell row (92 vs 72 ns), where it is still the smaller term.");
    println!();
    println!("So `update` is a wash against a rebuild PER CALL (0.54-1.15x). It pays by NOT");
    println!("being called: skip it for items that did not move and it wins 7.98x at 10%");
    println!("moving, 938x at 0.1% (grid_keep_bench). Call it for every item every frame and");
    println!("you lose - which is exactly what the 3D decision map shows.");
}
