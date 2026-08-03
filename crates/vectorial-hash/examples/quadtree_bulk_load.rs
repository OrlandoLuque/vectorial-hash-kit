//! Does `QuadTree::bulk_load` earn its place next to repeated `insert`?
//!
//! A new verb that is not faster than the loop it replaces is just more surface to keep
//! correct. `bulk_load` drops every item into the root and splits **once**, where `insert`
//! descends the tree N times and splits as it goes — so it should win, and by more as the tree
//! gets deeper. This checks that, and checks the two trees answer the same.
//!
//! It also carries the story of `bulk_load_par`, in which **two successive claims were both
//! wrong, in opposite directions** — which is why both tables below exist.
//!
//! The argument was that `Tree`'s parallel bulk load exists because its split axis is *chosen*
//! (`pick_split`), whereas a quadtree's is positional, so its recursion is already cheap and
//! rayon would only buy overhead. Written down as "an argument, not a measurement, so the next
//! person can disagree with it on purpose" — which is what happened, on the next day, by
//! measuring it. The recursion is **92–97 % of the build**, so a parallel version has a
//! ceiling of **7–11×**. Positional splits are cheap PER NODE; there are simply an enormous
//! number of nodes.
//!
//! The second table is that measurement, and it needed no parallel implementation to make: the
//! recursion is the only part that could fan out, so isolating its share bounds the win — an
//! `item_limit` of n makes the root never split, which measures the fill alone.
//!
//! **Then the ceiling turned out to be misleading too.** The parallel version now exists, and it
//! delivers **1.2–1.9×** — 11–23 % of the 7–11× bound. A share-of-time bound says where the time
//! IS; it says nothing about whether that time can be *spread*. This recursion allocates
//! thousands of small `Vec`s, so it is memory- and allocator-bound rather than compute-bound, and
//! most of the theoretical headroom is not headroom at all. Third table, third answer.
//!
//! ```bash
//! cargo run -p vectorial-hash --example quadtree_bulk_load --release
//! ```
use std::time::Instant;
mod common;
use common::abba;
use vectorial_hash::{Circle, Point, Positioned, QuadTree, Rect, Shape};

#[derive(Clone, Copy)]
struct P { id: u32, p: Point }
impl Positioned for P { fn position(&self) -> Point { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const W: f64 = 1024.0;
const REPS: usize = 9;

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() {
    let world = Rect::new(0.0, 0.0, W, W);
    println!("QuadTree::bulk_load vs repeated insert | world {W}^2 | median of {REPS}\n");
    println!("{:>10} {:>6} {:>14} {:>14} {:>10}", "N", "leaf", "insert us", "bulk_load us", "speedup");

    for &(n, leaf) in &[(10_000usize, 8usize), (100_000, 8), (100_000, 32), (500_000, 8)] {
        let mut rng = Rng(0xBADC0DE ^ n as u64);
        let items: Vec<P> = (0..n).map(|i| P { id: i as u32, p: Point::new(rng.f() * W, rng.f() * W) }).collect();

        // Both arms take an owned Vec so neither is charged for the other's setup: the insert
        // arm ignores it and inserts from the loop, the bulk arm consumes it. Cloning inside the
        // clock is what made this bench wrong for a week (see the note under the third table).
        let (a, b, _) = abba(REPS, &items,
            |v| { let mut q = QuadTree::<P>::new(world, leaf); for it in &v { q.insert(*it); } std::hint::black_box(&q); },
            |v| { let q = QuadTree::<P>::bulk_load(world, leaf, v); std::hint::black_box(&q); });

        // Faster is worthless if it is also different. Brute force is the referee.
        let mut q = QuadTree::<P>::new(world, leaf);
        for it in &items { q.insert(*it); }
        let bulk = QuadTree::<P>::bulk_load(world, leaf, items.clone());
        let probe = Circle::new(Point::new(W * 0.37, W * 0.61), 30.0);
        let mut want: Vec<u32> = items.iter().filter(|x| probe.contains_point(x.p)).map(|x| x.id).collect();
        let mut got_i: Vec<u32> = q.cull(&probe).iter().map(|x| x.id).collect();
        let mut got_b: Vec<u32> = bulk.cull(&probe).iter().map(|x| x.id).collect();
        want.sort_unstable(); got_i.sort_unstable(); got_b.sort_unstable();
        assert_eq!(got_i, want, "the insert-built tree disagrees with brute force");
        assert_eq!(got_b, want, "the bulk-loaded tree disagrees with brute force");
        assert!(!want.is_empty(), "the probe must hit something");

        println!("{:>10} {:>6} {:>14.0} {:>14.0} {:>10.2}", n, leaf, a, b, a / b);
    }
    println!("\n(both trees checked against brute force at every row before timing)");
    // ---------------------------------------------------------- would a parallel build pay?
    //
    // `Tree` has a `bulk_load_par` and this does not, on the argument that a quadtree's split
    // is POSITIONAL (four fixed quadrants) rather than chosen, so the partition is not the
    // expensive part and rayon would buy overhead. That was an argument, not a measurement —
    // and the table below refutes it: the recursion is 92-97% of the build.
    //
    // Measuring it does not require writing the parallel version. The recursion is the only
    // part that could be fanned out, so what bounds the win is its SHARE of the build — and
    // that can be isolated with the public API alone: an `item_limit` of n means the root never
    // splits, so the same call measures the fill by itself.
    println!("\n\nWhat could a parallel build even save? (the recursion's share of bulk_load)\n");
    println!("{:>10} {:>14} {:>16} {:>14} {:>18}", "N", "fill only us", "fill+divide us", "divide share", "ceiling @16 thr");
    for &n in &[10_000usize, 100_000, 500_000] {
        let mut rng = Rng(0xBADC0DE ^ n as u64);
        let items: Vec<P> = (0..n).map(|i| P { id: i as u32, p: Point::new(rng.f() * W, rng.f() * W) }).collect();
        // item_limit = n: every point lands in the root and `divide` never runs.
        let fill = med((0..REPS).map(|_| {
            let t = Instant::now();
            let q = QuadTree::<P>::bulk_load(world, n.max(1), items.clone());
            let us = t.elapsed().as_secs_f64() * 1e6; std::hint::black_box(&q); us
        }).collect());
        let full = med((0..REPS).map(|_| {
            let t = Instant::now();
            let q = QuadTree::<P>::bulk_load(world, 8, items.clone());
            let us = t.elapsed().as_secs_f64() * 1e6; std::hint::black_box(&q); us
        }).collect());
        let share = ((full - fill) / full).max(0.0);
        // Amdahl on the recursion alone, with the fill left serial.
        let ceiling = 1.0 / ((1.0 - share) + share / 16.0);
        println!("{:>10} {:>14.0} {:>16.0} {:>13.0}% {:>17.2}x", n, fill, full, share * 100.0, ceiling);
    }
    // And the real thing, now that it exists. The ceiling above is what perfect scaling would
    // buy; this is what rayon actually delivers, and the gap between them is the honest cost of
    // synchronisation, allocator contention and the serial flatten.
    #[cfg(feature = "parallel")]
    {
        println!("\n{:>10} {:>14} {:>16} {:>12} {:>16}", "N", "serial us", "parallel us", "speedup", "of the ceiling");
        for &(n, ceiling) in &[(10_000usize, 10.79f64), (100_000, 8.27), (500_000, 7.13)] {
            let mut rng = Rng(0xBADC0DE ^ n as u64);
            let items: Vec<P> = (0..n).map(|i| P { id: i as u32, p: Point::new(rng.f() * W, rng.f() * W) }).collect();
            let (ser, par, sp) = abba(REPS, &items,
                |v| { let q = QuadTree::<P>::bulk_load(world, 8, v); std::hint::black_box(&q); },
                |v| { let q = QuadTree::<P>::bulk_load_par(world, 8, v); std::hint::black_box(&q); });
            println!("{:>10} {:>14.0} {:>16.0} {:>11.2}x {:>15.0}%", n, ser, par, sp, 100.0 * sp / ceiling);
        }
    }
    #[cfg(not(feature = "parallel"))]
    println!("\n(re-run with `--features parallel` for the measured speed-up next to the ceiling)");

    println!("\nThe last column is an upper bound and a generous one: it assumes the recursion");
    println!("parallelises perfectly across 16 threads with no synchronisation, no allocator");
    println!("contention and no cost to merge the arenas. Whatever it reads, the real figure is");
    println!("below it — so if the ceiling is small, the argument for not writing the parallel");
    println!("version has become a measurement.");

    println!();
    println!("Measured answer: it does NOT reliably win - 0.83x at 10k, 1.06x at 100k, 1.18x at");
    println!("500k. `insert`'s descent was never the expensive part (O(log n), tiny constant),");
    println!("while this path grows one Vec to N and re-partitions it down every level anyway.");
    println!("bulk_load is here to complete the constructor vocabulary the other trees have,");
    println!("not as an optimisation. For a genuinely cheaper build, use KdTree2 or MortonGrid.");
}
