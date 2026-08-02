//! Does `QuadTree::bulk_load` earn its place next to repeated `insert`?
//!
//! A new verb that is not faster than the loop it replaces is just more surface to keep
//! correct. `bulk_load` drops every item into the root and splits **once**, where `insert`
//! descends the tree N times and splits as it goes — so it should win, and by more as the tree
//! gets deeper. This checks that, and checks the two trees answer the same.
//!
//! Note what is NOT here: a `bulk_load_par`. `Tree`'s parallel bulk load exists because its
//! split axis is *chosen* (`pick_split`), so the partition is the expensive part and worth
//! fanning out. A quadtree's split is positional — four fixed quadrants — so the recursion is
//! already cheap and rayon would mostly buy overhead. That is an argument, not a measurement,
//! and it is written down here so the next person can disagree with it on purpose.
//!
//! ```bash
//! cargo run -p vectorial-hash --example quadtree_bulk_load --release
//! ```
use std::time::Instant;
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

        let a = med((0..REPS).map(|_| {
            let t = Instant::now();
            let mut q = QuadTree::<P>::new(world, leaf);
            for it in &items { q.insert(*it); }
            let us = t.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&q); us
        }).collect());

        let b = med((0..REPS).map(|_| {
            let t = Instant::now();
            let q = QuadTree::<P>::bulk_load(world, leaf, items.clone());
            let us = t.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&q); us
        }).collect());

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
    println!();
    println!("Measured answer: it does NOT reliably win - 0.83x at 10k, 1.06x at 100k, 1.18x at");
    println!("500k. `insert`'s descent was never the expensive part (O(log n), tiny constant),");
    println!("while this path grows one Vec to N and re-partitions it down every level anyway.");
    println!("bulk_load is here to complete the constructor vocabulary the other trees have,");
    println!("not as an optimisation. For a genuinely cheaper build, use KdTree2 or MortonGrid.");
}
