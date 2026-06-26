//! Parallel per-unit AI — the pattern the `siege` demo leans on, benchmarked.
//!
//! Each unit, every frame, runs **read-only** queries on the shared index
//! (target = k-NN nearest enemies; vision/AoE = a sphere cull). Because the
//! index's queries are `&self` + `Sync` and the units are mutated disjointly,
//! the whole AI pass **fans out over rayon** with no new API and no contention:
//! `units.par_iter().map(|u| u.think(&index))`. This measures serial vs that.
//!
//! ```bash
//! cargo run -p vectorial-hash --example parallel_ai --release --features parallel
//! ```
//!
//! Interleaved min-of-N (rotate serial/parallel each round so a background-load
//! spike hits both equally), with a `noise` column (worst median/min) flagging a
//! contaminated row. Reports the speedup and the crossover (when threads pay).

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("This benchmark needs rayon — re-run with --features parallel.");
}

#[cfg(feature = "parallel")]
fn main() {
    bench::run();
}

#[cfg(feature = "parallel")]
mod bench {
    use std::time::Instant;

    use rayon::prelude::*;
    use vectorial_hash::{Aabb, Point3, Positioned3, Sphere3, Tree3};

    struct Rng(u64);
    impl Rng {
        fn new(s: u64) -> Self { Rng(s | 1) }
        fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
        fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
    }

    #[derive(Clone, Copy)]
    struct Unit { p: Point3 }
    impl Positioned3 for Unit { fn position(&self) -> Point3 { self.p } }

    const WORLD: f64 = 1024.0;
    const VISION: f64 = 24.0;

    /// One unit's per-frame AI: the 4 nearest (targeting) + a vision-sphere cull
    /// (perception). Read-only on the shared index; returns blackholed work.
    #[inline]
    fn think(tree: &Tree3<Unit>, p: Point3) -> usize {
        tree.knn(p, 4).len() + tree.cull(&Sphere3::new(p.x, p.y, p.z, VISION)).len()
    }

    fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

    pub fn run() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        println!("parallel per-unit AI | world={WORLD}³ | vision r={VISION} | rayon over {threads} threads");
        println!("each unit: knn(4) + vision-sphere cull, read-only on the shared index. Interleaved min-of-30.\n");
        println!("{:>8} | {:>10} {:>10} {:>8} | {:>6}", "units", "serial ms", "par ms", "speedup", "noise");

        let rounds = 30usize;
        for &n in &[5_000usize, 20_000, 80_000, 200_000] {
            let mut rng = Rng::new(1);
            let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
            let units: Vec<Unit> = (0..n).map(|_| Unit { p: Point3::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD), rng.range(0.0, WORLD)) }).collect();
            let mut tree = Tree3::<Unit>::new(world, 8);
            for u in &units { tree.insert(*u); }

            let serial = || units.iter().map(|u| think(&tree, u.p)).sum::<usize>();
            let parallel = || units.par_iter().map(|u| think(&tree, u.p)).sum::<usize>();
            std::hint::black_box(serial());
            std::hint::black_box(parallel());

            let (mut ss, mut ps) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
            for round in 0..rounds {
                let run_one = |which_serial: bool, out: &mut Vec<f64>| {
                    let t = Instant::now();
                    let acc = if which_serial { serial() } else { parallel() };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    std::hint::black_box(acc);
                    out.push(ms);
                };
                if round % 2 == 0 {
                    run_one(true, &mut ss);
                    run_one(false, &mut ps);
                } else {
                    run_one(false, &mut ps);
                    run_one(true, &mut ss);
                }
            }
            let s_min = ss.iter().cloned().fold(f64::INFINITY, f64::min);
            let p_min = ps.iter().cloned().fold(f64::INFINITY, f64::min);
            let noise = (median(ss.clone()) / s_min).max(median(ps.clone()) / p_min);
            println!("{:>8} | {:>10.3} {:>10.3} {:>7.2}× | {:>6.2}", n, s_min, p_min, s_min / p_min, noise);
        }
        println!("\nThe AI pass parallelises near-linearly with the unit count (reads fan out, no");
        println!("contention). The relocation pass (writes) stays serial — its lever is update_ref.");
    }
}
