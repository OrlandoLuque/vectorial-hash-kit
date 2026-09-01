//! Where does the kept grid actually beat the keep-tree? — the shape of the boundary, not a point.
//!
//! `AdaptiveIndex` picks between them with one scalar: take the grid when `queries per item`
//! exceeds [`Thresholds::rebuild_query_ratio`]. That threshold was derived at **maximum churn**,
//! which is exactly where keeping a grid is worth least — and since the grid learned to keep in
//! place (2026-07-31) a grid that hardly needs maintaining keeps its cheaper cull almost for
//! free. So the suspicion is that the real boundary depends on *both* axes and a vertical line
//! cannot express it.
//!
//! `examples/pick_a_structure` made that concrete in one command (at `churn=0.001 queries=0.05`
//! the grid measures ~2x better while the policy picks the tree). This maps the whole plane.
//!
//! ```bash
//! cargo run -p vectorial-hash --example grid_tree_frontier --release
//! ```
//!
//! Both arms are *kept*, never rebuilt: `Tree3` through `update_ref`, `MortonGrid3` through
//! `update`. Each cell reports the total per-frame cost of maintain + query for both, so the
//! winner is decided on the number a caller actually pays.
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3, Thresholds, Tree3};

#[derive(Clone, Copy)]
struct P { p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

const W: f64 = 512.0;
const N: usize = 20_000;
const RADIUS: f64 = 36.0;
const REPS: usize = 5;

fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, W, W, W) }

fn main() {
    let churns = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0];
    let qpis = [0.01, 0.05, 0.2, 0.5, 1.0, 2.0];

    let mut rng = Rng(0xF00D);
    let items: Vec<P> = (0..N).map(|_| P { p: Point3::new(rng.f() * W, rng.f() * W, rng.f() * W) }).collect();

    println!("kept grid vs keep-tree — total ms/frame, {N} items, radius {RADIUS}, min of {REPS}\n");
    println!("the policy's current rule: take the grid when queries/item > {:.3}\n", Thresholds::default().rebuild_query_ratio);
    print!("{:>8} |", "churn\\q");
    for q in qpis { print!("{q:>10}"); }
    println!("\n{}", "-".repeat(8 + 2 + 10 * qpis.len()));

    // Where the boundary actually sits, per churn row: the smallest query load at which the
    // grid wins. That column IS the threshold, and if it moves down the rows the rule needs
    // both axes.
    let mut boundary: Vec<(f64, Option<f64>)> = Vec::new();

    for &churn in &churns {
        print!("{churn:>8} |");
        let mut first_grid_win: Option<f64> = None;
        for &qpi in &qpis {
            let moves = (N as f64 * churn) as usize;
            let queries = ((N as f64 * qpi) as usize).clamp(1, 4096);
            let probes: Vec<Sphere3> = (0..queries).map(|i| {
                let it = items[i * N / queries];
                Sphere3::new(it.p.x, it.p.y, it.p.z, RADIUS)
            }).collect();

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
                let levels = MortonGrid3::<P>::levels_for_cell_size(world(), RADIUS);
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
            if grid_wins && first_grid_win.is_none() { first_grid_win = Some(qpi); }
            // ratio > 1 means the grid is ahead
            print!("{:>10}", format!("{:.2}{}", tree_ms / grid_ms, if grid_wins { "G" } else { "T" }));
        }
        println!();
        boundary.push((churn, first_grid_win));
    }

    println!("\n  (x.xxG = the grid wins by that factor · x.xxT = the tree does)\n");
    println!("{:>10} {:>28}", "churn", "grid wins from queries/item");
    for (c, b) in &boundary {
        println!("{c:>10} {:>28}", b.map_or("never in range".to_string(), |q| format!("{q}")));
    }

    let moving: Vec<f64> = boundary.iter().filter_map(|(_, b)| *b).collect();
    println!();
    if moving.windows(2).any(|w| w[0] != w[1]) {
        println!("The boundary MOVES with churn, so a single `rebuild_query_ratio` cannot express it:");
        println!("whatever value it holds is right for one row of this table and wrong for the rest.");
        println!("The rule wants to read both rates — roughly `q_per_item > f(m_per_item)` — and the");
        println!("column above is the data to fit f to.");
    } else {
        println!("The boundary does NOT move with churn across this range: one scalar is enough after");
        println!("all, and `rebuild_query_ratio` should simply be set to the column above.");
    }
    println!();
    println!("Both arms KEEP (update_ref / update); neither is rebuilt. That is the comparison the");
    println!("policy makes, and it only became the right one when the grid learned to keep in place.");
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
