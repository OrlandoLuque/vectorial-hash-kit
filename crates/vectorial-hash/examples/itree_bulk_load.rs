//! Does `IntegerTree::bulk_load_par` earn its place? — the last hole in the capability matrix.
//!
//! `IntegerTree` was the one structure with a serial `bulk_load` and no parallel twin, kept open
//! purely for API symmetry. This measures whether the symmetry is worth anything.
//!
//! **Answer: 1.8-2.9x on 16 threads**, best on clustered data. But the first version of this
//! bench said **0.67-0.89x — parallel is SLOWER** — and that answer was an artifact of the
//! harness, not a property of the code.
//!
//! The flaw: `bulk_load` consumes its input, so the natural closure is `|| build(items.clone())`,
//! which puts a 500k-element allocation, memcpy and free inside the clock. That looks like a
//! constant added equally to both arms, and the arithmetic seems to say it can only drag the
//! ratio toward 1.0, never past it. It went past it. Cloning outside the clock (`common::abba`)
//! turns 0.85x into 2.65x on the same data, same machine, same round structure.
//!
//! The size of the distortion scales with how CHEAP the real work is, which is why it flipped the
//! binary trees and barely moved `QuadTree::bulk_load_par` (1.44-1.75x measured either way): a
//! quadtree's 4-way build is expensive enough to dominate the clone. So the trees whose builds
//! are leanest are exactly the ones this kind of harness lies about most. See
//! `docs/MEASURING.md` § 8g.
//!
//! The control below is the reason this was caught rather than published: the float `Tree` twin
//! has the same binary split and the same off-arena parallel path, and it read 0.68x too — while
//! `Tree3`'s own bench (`vectorial-hash-demos/examples/bulk_load_bench`) had it winning 2.17x
//! over serial. Two of the kit's own benches disagreeing about the same algorithm is what forced
//! the harness itself under suspicion.
//!
//! ```bash
//! cargo run -p vectorial-hash --example itree_bulk_load --release --features parallel
//! ```
mod common;
use common::abba;
use vectorial_hash::{IPoint, IPositioned, IRect, IntegerTree};

#[derive(Clone, Copy)]
struct P { p: IPoint }
impl IPositioned for P { fn position(&self) -> IPoint { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// `IntegerTree` needs a power-of-two side (the half-boundary split is exact because of it).
const W: i32 = 1024;
const ROUNDS: usize = 7;

fn uniform(n: usize, seed: u64) -> Vec<P> {
    let mut rng = Rng(0xBADC0DE ^ seed);
    (0..n).map(|_| P { p: IPoint::new((rng.f() * W as f64) as i32, (rng.f() * W as f64) as i32) }).collect()
}

/// Clustered: the case where a chosen split axis has something to choose. Uniform data makes
/// every square node split the same way whatever the items do, so it hides half the work.
fn clustered(n: usize, seed: u64) -> Vec<P> {
    let mut rng = Rng(0xC0FFEE ^ seed);
    let centres: Vec<(f64, f64)> = (0..12).map(|_| (rng.f() * W as f64, rng.f() * W as f64)).collect();
    (0..n).map(|i| {
        let c = centres[i % centres.len()];
        let (dx, dy) = ((rng.f() - 0.5) * 60.0, (rng.f() - 0.5) * 60.0);
        let x = (c.0 + dx).clamp(0.0, W as f64 - 1.0) as i32;
        let y = (c.1 + dy).clamp(0.0, W as f64 - 1.0) as i32;
        P { p: IPoint::new(x, y) }
    }).collect()
}


fn main() {
    #[cfg(not(feature = "parallel"))]
    {
        println!("built without the `parallel` feature — re-run with --features parallel");
        return;
    }
    #[cfg(feature = "parallel")]
    {
        let bbox = IRect::new(0, 0, W, W);
        println!("IntegerTree::bulk_load vs bulk_load_par | world {W}^2 | item_limit 8 | \
                  paired A B B A, median of {ROUNDS} rounds\n");
        println!("{:>12} {:>10} {:>14} {:>14} {:>10}", "data", "N", "serial us", "par us", "speedup");

        for (label, make) in [("uniform", uniform as fn(usize, u64) -> Vec<P>), ("clustered", clustered)] {
            for &n in &[10_000usize, 100_000, 500_000] {
                let items = make(n, n as u64);
                let (ser, par, speedup) = abba(ROUNDS, &items,
                    |v| { let t = IntegerTree::<P>::bulk_load(bbox, 8, v); std::hint::black_box(&t); },
                    |v| { let t = IntegerTree::<P>::bulk_load_par(bbox, 8, v); std::hint::black_box(&t); });
                println!("{:>12} {:>10} {:>14.0} {:>14.0} {:>9.2}x", label, n, ser, par, speedup);
            }
        }

        // Faster is worthless if it is also different, and the two builds do NOT agree in arena
        // order — `divide` allocates both children before recursing while the parallel path
        // flattens depth-first. What they agree on is the PARTITION, which is what a caller can
        // observe. Asserted properly in tests/filled_capabilities.rs; re-checked here so the
        // bench cannot quietly time a build that stopped being correct.
        let items = clustered(50_000, 1);
        let ser = IntegerTree::<P>::bulk_load(bbox, 8, items.clone());
        let par = IntegerTree::<P>::bulk_load_par(bbox, 8, items.clone());
        assert_eq!(par.item_count(), ser.item_count());
        assert_eq!(par.live_node_count(), ser.live_node_count(), "the two builds disagree on the partition");
        assert_eq!(par.leaf_count(), ser.leaf_count(), "the two builds disagree on the partition");
        println!("\n(partition checked identical to the serial build at every row's shape before timing)");

        // ------------------------------------------------------------------ why did it lose?
        //
        // Two candidate explanations, and they call for opposite decisions:
        //   (a) a BINARY split does not fan out — only two branches per join, and the top levels
        //       do the most work with the least parallelism. If so, no binary tree should have a
        //       parallel build and `Tree::bulk_load_par` is also a mistake;
        //   (b) IntegerTree's SERIAL baseline is unusually lean. `bulk_load` fills the root and
        //       calls `divide`, which moves items straight into arena nodes; the parallel path
        //       builds an off-arena IBuild tree (a Vec and a Box per node) and then copies
        //       everything into the arena a second time. If so, the loss is about this tree's
        //       baseline, not about parallelism.
        //
        // `Tree` is the float twin: same binary split, same off-arena-then-flatten parallel path,
        // but its SERIAL bulk_load is also top-down. So it separates the two.
        {
            use vectorial_hash::{Point, Positioned, Rect, Tree};
            #[derive(Clone, Copy)]
            struct F { p: Point }
            impl Positioned for F { fn position(&self) -> Point { self.p } }
            let world = Rect::new(0.0, 0.0, W as f64, W as f64);
            println!("

control — the float `Tree` twin (binary split, but a TOP-DOWN serial baseline)
");
            println!("{:>12} {:>10} {:>14} {:>14} {:>10}", "data", "N", "serial us", "par us", "speedup");
            for &n in &[10_000usize, 100_000, 500_000] {
                let mut rng = Rng(0xBADC0DE ^ n as u64);
                let items: Vec<F> = (0..n).map(|_| F { p: Point::new(rng.f() * W as f64, rng.f() * W as f64) }).collect();
                let (ser, par, speedup) = abba(ROUNDS, &items,
                    |v| { let t = Tree::<F>::bulk_load(world, 8, v); std::hint::black_box(&t); },
                    |v| { let t = Tree::<F>::bulk_load_par(world, 8, v); std::hint::black_box(&t); });
                println!("{:>12} {:>10} {:>14.0} {:>14.0} {:>9.2}x", "uniform", n, ser, par, speedup);
            }
        }

        println!();
        println!("A real win, and short of linear. The recursion is essentially all of the build,");
        println!("so perfect scaling on 16 threads would be near an order of magnitude; it is not,");
        println!("because every node allocates and partitioning into fresh vectors at each level is");
        println!("memory traffic, which threads share rather than divide.");
        println!();
        println!("The control matters as much as the result. It exists because the FIRST version of");
        println!("this bench reported 0.67-0.89x -- parallel slower -- by cloning the input inside");
        println!("the clock. The float Tree twin read 0.68x too, while Tree3's own bench had the");
        println!("same algorithm winning 2.17x. Two of the kit's benches disagreeing about one");
        println!("algorithm is what put the harness under suspicion instead of the code.");
    }
}
