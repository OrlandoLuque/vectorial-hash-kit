//! The third axis: at a fixed churn and query load, what makes the grid stop winning?
//!
//! `grid_tree_frontier` mapped (churn × query load) at one radius and found the kept grid ahead in
//! all 42 cells. At `radius=8` the same table flips to the tree. So query extent is an axis, and
//! this asks the next question: **is extent itself the thing that decides, or is extent a proxy?**
//!
//! ```bash
//! cargo run -p vectorial-hash --example extent_axis --release
//! ```
//!
//! **The hypothesis under test.** A grid query touches a roughly fixed number of cells no matter
//! what it finds, and pays a hash lookup for each. A tree descends and prunes, so an empty region
//! costs it one rejected node. If that is the mechanism, the grid should lose whenever a query
//! **returns almost nothing** — and the predictor is not the radius but the expected number of
//! points inside it, `density × volume`. Radius would then be a proxy that only works at one
//! density.
//!
//! **The control that makes it a test rather than a demonstration:** the sweep runs at two
//! densities 4× apart. Under the hypothesis the crossover lands at the same *points per query*
//! and therefore at *different radii* — those two predictions disagree, so the run can refute it.
//! A single-density sweep could not: any curve crosses somewhere.
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3, Tree3};

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const W: f64 = 512.0;
const REPS: usize = 5;
/// The horde's own operating point, so this sweep passes through the case that raised the
/// question rather than through a convenient one.
const CHURN: f64 = 0.056;
const QPI: f64 = 0.06;

fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, W, W, W) }

fn main() {
    let radii = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 36.0, 64.0];
    println!("the extent axis at a FIXED operating point: churn {CHURN}, {QPI} culls/item, world {W}^3");
    println!("both arms kept (update_ref / update), total ms/frame, min of {REPS}\n");
    println!("expected points/query = density x sphere volume — the quantity the hypothesis says decides\n");

    // Two densities, 4x apart. Same radii, so if the crossover is a property of RADIUS the two
    // rows flip in the same column, and if it is a property of POINTS PER QUERY they do not.
    let mut crossovers: Vec<(usize, f64, f64)> = Vec::new();
    for &n in &[20_000usize, 80_000usize] {
        let mut rng = Rng(0xF00D);
        let items: Vec<P> = (0..n).map(|_| P { p: Point3::new(rng.f() * W, rng.f() * W, rng.f() * W) }).collect();
        let density = n as f64 / (W * W * W);

        println!("n = {n}  (density {:.3e} points/wu^3)", density);
        println!("{:>8} {:>14} {:>11} {:>11} {:>9}", "radius", "pts/query", "tree ms", "grid ms", "winner");

        let mut first_grid_win: Option<(f64, f64)> = None;
        for &r in &radii {
            let pts = density * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
            let moves = (n as f64 * CHURN) as usize;
            let queries = ((n as f64 * QPI) as usize).clamp(1, 4096);
            let probes: Vec<Sphere3> = (0..queries)
                .map(|i| { let it = items[i * n / queries]; Sphere3::new(it.p.x, it.p.y, it.p.z, r) })
                .collect();

            let tree_ms = {
                let mut t = Tree3::new(world(), 8);
                let refs: Vec<_> = items.iter().filter_map(|it| t.insert_ref(*it)).collect();
                let mut pos = items.clone();
                best(|| {
                    for i in 0..moves { pos[i].p.x = (pos[i].p.x + 1.0) % W; let p = pos[i]; t.update_ref(refs[i], |c| *c = p); }
                    for s in &probes { std::hint::black_box(t.cull(s).len()); }
                })
            };
            let grid_ms = {
                // Sized to the radius, so the grid is never handicapped by a cell that does not
                // suit the query: this measures the mechanism, not a mis-sized index.
                let levels = MortonGrid3::<P>::levels_for_cell_size(world(), r);
                let mut g = MortonGrid3::new(world(), levels);
                for it in &items { g.insert(*it); }
                let mut pos = items.clone();
                best(|| {
                    for it in pos.iter_mut().take(moves) {
                        let was = it.p;
                        it.p.x = (it.p.x + 1.0) % W;
                        let p = *it;
                        #[allow(clippy::float_cmp)]
                        g.update(was, |c| c.p.x == was.x && c.p.y == was.y && c.p.z == was.z, |c| *c = p);
                    }
                    for s in &probes { std::hint::black_box(g.cull(s).len()); }
                })
            };
            let grid_wins = grid_ms < tree_ms;
            if grid_wins && first_grid_win.is_none() { first_grid_win = Some((r, pts)); }
            println!("{r:>8} {pts:>14.3} {tree_ms:>11.3} {grid_ms:>11.3} {:>9}",
                     if grid_wins { "grid" } else { "tree" });
        }
        println!();
        if let Some((r, pts)) = first_grid_win { crossovers.push((n, r, pts)); }
    }

    println!("{:>10} {:>16} {:>18}", "n", "grid wins from r", "...= pts/query");
    for (n, r, pts) in &crossovers { println!("{n:>10} {r:>16} {pts:>18.3}"); }

    if crossovers.len() == 2 {
        let (r0, p0) = (crossovers[0].1, crossovers[0].2);
        let (r1, p1) = (crossovers[1].1, crossovers[1].2);
        println!();
        // Radii are swept in 2x steps, so "the same rung" is the finest agreement available and
        // a ratio inside 2x cannot separate the two explanations. Say so instead of picking one.
        let same_radius = r0 == r1;
        let pts_ratio = if p0 > p1 { p0 / p1 } else { p1 / p0 };
        if same_radius && pts_ratio > 2.0 {
            println!("Both densities flip at the SAME RADIUS ({r0}) and at points/query {pts_ratio:.1}x apart.");
            println!("That refutes the hypothesis: extent decides directly, and expected points per");
            println!("query does not. A rule reading `density x extent^3` would be reading a proxy");
            println!("for the thing that matters, and would be wrong at other densities.");
        } else if !same_radius && pts_ratio < 2.0 {
            println!("The two densities flip at DIFFERENT radii ({r0} and {r1}) but at points/query");
            println!("within {pts_ratio:.2}x of each other. That supports the hypothesis: what decides is");
            println!("how much a query FINDS, not how wide it is. A grid pays its lookups whether or");
            println!("not there is anything in the cells; a tree prunes empty space for free.");
            println!("Any extent-aware rule should therefore read `density x extent^3`, not extent.");
        } else {
            println!("Neither prediction is cleanly met (radii {r0} vs {r1}, points/query {pts_ratio:.2}x apart).");
            println!("The 2x radius rungs cannot separate the two explanations at this resolution —");
            println!("that is a statement about this sweep, not evidence for either side.");
        }
    }
    println!();
    println!("Whatever this says, it is measured at ONE churn and ONE query load, which is the");
    println!("mistake `grid_tree_frontier` made along a different axis. See MEASURING.md 8i.");
}

fn best<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut lo = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        lo = lo.min(t.elapsed().as_secs_f64() * 1e3);
    }
    lo
}
