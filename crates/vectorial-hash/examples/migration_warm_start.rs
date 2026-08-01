//! Is a **warm-start migration** worth it?
//!
//! When `AdaptiveIndex` decides its workload has changed, it migrates: it throws the old
//! backend away and builds the new one from `items` in **slot order**, which is insertion
//! order, which is spatially arbitrary. But the backend it is abandoning is not arbitrary —
//! it spent the whole time it was alive sorting these very points in space. A grid can hand
//! them back in Z-order for free; a tree can hand them back in DFS order, which is also
//! spatially coherent.
//!
//! So: does feeding the new structure a spatially-ordered sequence build it faster than
//! feeding it an arbitrary one? Two reasons it might.
//!
//! - **The grid**: consecutive items in Z-order land in the *same* cell, so `entry(cell)`
//!   hits the same bucket repeatedly instead of jumping around a hash table with tens of
//!   thousands of entries.
//! - **The k-d tree**: its build is a recursive median partition. Input that is already
//!   spatially sorted is already partially partitioned.
//!
//! This is the same class of trick as bulk-loading a B-tree from sorted input rather than
//! inserting one key at a time. It is only worth doing if it measures, which is what this is
//! for — the last thing I assumed about grid maintenance turned out to be wrong.
//!
//! ```bash
//! cargo run -p vectorial-hash --example migration_warm_start --release
//! ```
use std::time::Instant;
use vectorial_hash::{Aabb, KdTree3, MortonGrid3, Point3, Positioned3, Sphere3, Tree3};

#[derive(Clone, Copy)]
struct P { id: u32, p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const N: usize = 50_000;
const W: f64 = 1024.0;
const REPS: usize = 12;

/// Median of `REPS` runs. Min-of-N is the kit's usual estimator for a clock, but a build
/// allocates, and the allocator's state is part of what is being measured — a median keeps
/// that honest rather than reporting the one lucky run where nothing had to be mapped.
fn timed<F: FnMut()>(mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..REPS).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e6 }).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[REPS / 2]
}

fn main() {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    let mut rng = Rng(0x1A6D3F);
    let cold: Vec<P> = (0..N).map(|i| P { id: i as u32, p: Point3::new(rng.f() * W, rng.f() * W, rng.f() * W) }).collect();

    // The order a grid would hand its contents back in: cell by cell, i.e. Z-order. Produced
    // here by actually building one and draining it, so this is the real sequence a warm-start
    // migration would see and not an idealised sort.
    let levels = MortonGrid3::<P>::levels_for_cell_size(world, 32.0);
    let mut g = MortonGrid3::<P>::new(world, levels);
    for it in &cold { g.insert(*it); }
    let warm: Vec<P> = g.iter_z_order().copied().collect();
    assert_eq!(warm.len(), cold.len(), "draining must not lose items");

    println!("warm-start migration | {N} items in {W}^3 | median of {REPS}\n");
    println!("Target built from an ARBITRARY order (what migration does today) vs from the");
    println!("order the abandoned backend already had them in (Z-order).\n");
    println!("{:>22} {:>12} {:>12} {:>10}", "target backend", "cold us", "warm us", "speedup");

    let a = timed(|| { let mut t = MortonGrid3::<P>::new(world, levels); for it in &cold { t.insert(*it); } std::hint::black_box(&t); });
    let b = timed(|| { let mut t = MortonGrid3::<P>::new(world, levels); for it in &warm { t.insert(*it); } std::hint::black_box(&t); });
    println!("{:>22} {:>12.0} {:>12.0} {:>10.2}", "MortonGrid3", a, b, a / b);

    let a = timed(|| { let t = KdTree3::<P>::from_items(8, cold.clone()); std::hint::black_box(&t); });
    let b = timed(|| { let t = KdTree3::<P>::from_items(8, warm.clone()); std::hint::black_box(&t); });
    println!("{:>22} {:>12.0} {:>12.0} {:>10.2}", "KdTree3", a, b, a / b);

    let a = timed(|| { let mut t = Tree3::<P>::new(world, 8); for it in &cold { t.insert(*it); } std::hint::black_box(&t); });
    let b = timed(|| { let mut t = Tree3::<P>::new(world, 8); for it in &warm { t.insert(*it); } std::hint::black_box(&t); });
    println!("{:>22} {:>12.0} {:>12.0} {:>10.2}", "Tree3 (insert)", a, b, a / b);

    let a = timed(|| { let t = Tree3::<P>::bulk_load(world, 8, cold.clone()); std::hint::black_box(&t); });
    let b = timed(|| { let t = Tree3::<P>::bulk_load(world, 8, warm.clone()); std::hint::black_box(&t); });
    println!("{:>22} {:>12.0} {:>12.0} {:>10.2}", "Tree3 (bulk_load)", a, b, a / b);

    // A structure that answers differently after a reorder would make the whole idea unusable,
    // so check the built results agree rather than assuming order is only a performance knob.
    let q = Sphere3::new(W * 0.5, W * 0.5, W * 0.5, 60.0);
    let mut c: Vec<u32> = KdTree3::<P>::from_items(8, cold.clone()).cull(&q).iter().map(|x| x.id).collect();
    let mut w: Vec<u32> = KdTree3::<P>::from_items(8, warm.clone()).cull(&q).iter().map(|x| x.id).collect();
    c.sort_unstable(); w.sort_unstable();
    assert_eq!(c, w, "a warm-started build must answer identically to a cold one");
    assert!(!c.is_empty(), "the agreement check must not pass vacuously");
    println!("\nsame answers from both orders: {} items in the probe cull", c.len());
}
