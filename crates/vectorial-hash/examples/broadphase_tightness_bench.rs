//! broadphase_tightness_bench — the `THREE_D.md` open item, answered directly:
//! **when does a tighter broad-phase pay for itself as the narrow-phase gets
//! expensive?** The query is a convex `Polyhedron3` (`faceted_ball`, N faces) whose
//! point test costs **one dot product per face** — so the narrow-phase cost is the
//! knob (sweep N). Two ways to get the exact set of points inside it:
//!
//!   - **TIGHT**  — `tree.cull(&poly)`: the tree prunes each node by the poly's N
//!     half-spaces (`classify_box`) and tests surviving leaf points against all N —
//!     tight, but N-plane work *per node* and *per point*.
//!   - **BOX+NP** — `tree.cull(&box6)` by the poly's 6-face bounding box (cheap
//!     prune, over-collects the corners), then an exact N-plane narrow-phase on the
//!     candidates.
//!
//! Both are verified == brute force. The crossover in N answers the question: a
//! cheap narrow-phase (few faces) favours the loose box broad-phase; an expensive
//! one (many faces) should favour paying for the tight prune. Reports ns/query.
//!
//! `cargo run -p vectorial-hash --example broadphase_tightness_bench --release`  (`BPT_N`)
use std::time::Instant;
use vectorial_hash::{Aabb, Point3, Polyhedron3, Positioned3, Tree3};

#[derive(Clone, Copy)]
struct P { p: Point3, id: u32 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } fn unit(&mut self) -> f64 { (self.next() & 0xffffff) as f64 / (1u32 << 24) as f64 } }

// exact point-in-poly: satisfies every half-space n·p <= d (the expensive test).
fn in_poly(planes: &[(f64, f64, f64, f64)], p: Point3) -> bool {
    for &(nx, ny, nz, d) in planes { if nx * p.x + ny * p.y + nz * p.z > d + 1e-9 { return false; } }
    true
}

// the 6 axis half-spaces of an AABB (the loose bounding-box broad-phase volume).
fn box6(b: Aabb) -> Polyhedron3 {
    let pl = vec![
        (1.0, 0.0, 0.0, b.x_max()), (-1.0, 0.0, 0.0, -b.x),
        (0.0, 1.0, 0.0, b.y_max()), (0.0, -1.0, 0.0, -b.y),
        (0.0, 0.0, 1.0, b.z_max()), (0.0, 0.0, -1.0, -b.z),
    ];
    Polyhedron3::new(pl, b)
}

fn main() {
    let n: usize = std::env::var("BPT_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let w = 512.0f64;
    let mut rng = Rng(0xB0A7);
    let pts: Vec<P> = (0..n).map(|i| P { p: Point3::new(rng.unit() * w, rng.unit() * w, rng.unit() * w), id: i as u32 }).collect();
    let mut tree = Tree3::<P>::new(Aabb::new(0.0, 0.0, 0.0, w, w, w), 12);
    for p in &pts { tree.insert(*p); }

    // a set of query balls (fixed across face counts so the comparison is apples-to-apples)
    let mut qr = Rng(0x0FACE);
    let queries: Vec<(f64, f64, f64, f64)> = (0..120).map(|_| (qr.unit() * w, qr.unit() * w, qr.unit() * w, 30.0 + qr.unit() * 70.0)).collect();

    println!("broadphase tightness — cull(&poly, N faces) vs cull(&box6)+N-plane narrowphase\n{n} points, {} queries, radius 30..100\n", queries.len());
    println!("  faces | TIGHT poly-cull | BOX6 + narrowphase | winner");

    for &faces in &[8usize, 16, 32, 64, 128, 256] {
        // verify all three agree (a subset of the queries)
        for &(cx, cy, cz, r) in &queries[..12] {
            let poly = Polyhedron3::faceted_ball(cx, cy, cz, r, faces);
            let mut tight: Vec<u32> = tree.cull(&poly).iter().map(|p| p.id).collect();
            let cand = tree.cull(&box6(poly.bbox));
            let mut boxn: Vec<u32> = cand.iter().filter(|p| in_poly(&poly.planes, p.p)).map(|p| p.id).collect();
            let mut brute: Vec<u32> = pts.iter().filter(|p| in_poly(&poly.planes, p.p)).map(|p| p.id).collect();
            tight.sort_unstable(); boxn.sort_unstable(); brute.sort_unstable();
            assert_eq!(tight, brute, "TIGHT poly-cull != brute ({faces} faces)");
            assert_eq!(boxn, brute, "BOX6+narrowphase != brute ({faces} faces)");
        }

        // time each (min of 6 over all queries, warm)
        let polys: Vec<Polyhedron3> = queries.iter().map(|&(cx, cy, cz, r)| Polyhedron3::faceted_ball(cx, cy, cz, r, faces)).collect();
        let boxes: Vec<Polyhedron3> = polys.iter().map(|p| box6(p.bbox)).collect();
        let bench = |f: &dyn Fn() -> usize| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..6 { let t = Instant::now(); let acc = f(); std::hint::black_box(acc); best = best.min(t.elapsed().as_secs_f64()); }
            best * 1e9 / queries.len() as f64
        };
        let tight_ns = bench(&|| { let mut acc = 0usize; for poly in &polys { acc += tree.cull(poly).len(); } acc });
        let boxn_ns = bench(&|| {
            let mut acc = 0usize;
            for (poly, bx) in polys.iter().zip(&boxes) {
                let cand = tree.cull(bx);
                acc += cand.iter().filter(|p| in_poly(&poly.planes, p.p)).count();
            }
            acc
        });
        let winner = if tight_ns < boxn_ns { format!("TIGHT {:.2}×", boxn_ns / tight_ns) } else { format!("BOX+NP {:.2}×", tight_ns / boxn_ns) };
        println!("  {faces:>5} | {tight_ns:>13.0} ns | {boxn_ns:>16.0} ns | {winner}");
    }

    // over-collection of the box broad-phase (how much the narrow-phase throws away)
    let poly = Polyhedron3::faceted_ball(w * 0.5, w * 0.5, w * 0.5, 80.0, 64);
    let cand = tree.cull(&box6(poly.bbox)).len();
    let kept = tree.cull(&poly).len();
    println!("\nbox6 broad-phase over-collects ~{:.0}% (candidates {} vs kept {}) — that surplus is what\nthe expensive narrow-phase pays for, and what the tight poly-cull prunes away up front.", (cand as f64 / kept.max(1) as f64 - 1.0) * 100.0, cand, kept);
    println!("reading: few faces → BOX+NP wins (cheap prune, cheap test); as faces (narrow-phase cost)\ngrow, the tight poly-cull's per-node prune pays off — the crossover above is the honest verdict.");
}
