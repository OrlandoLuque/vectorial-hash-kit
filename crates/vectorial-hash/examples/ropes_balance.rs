//! Ropes maintenance cost — the *other* half of the rope ledger. The neighbour
//! flood / DDA walk is ~30–45 % faster per query with stored ropes
//! (`docs/RAYCAST.md`), but ropes are rewired on every split/merge. This
//! measures that upkeep: build + churn **with vs without** the `neighbors`
//! feature. Run both and compare:
//!
//! ```bash
//! cargo run -p vectorial-hash --example ropes_balance --release
//! cargo run -p vectorial-hash --example ropes_balance --release --features neighbors
//! ```
//!
//! `build` = insert N points (splits → rope wiring). `update/frame` = relocate
//! every point one frame via `update_ref` (boundary crossings → split/merge →
//! rope rewiring). Min-of-N for low variance; the **delta between the two runs**
//! is the rope maintenance cost. Cross it with the per-query win to settle
//! whether ropes pay off for a given query : mutation ratio.

// The `x < lo || x > hi` bounce test reads clearer than `!(lo..=hi).contains()`.
#![allow(clippy::manual_range_contains)]

use std::time::Instant;

use vectorial_hash::{ItemRef, Point, Positioned, Rect, Tree};

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct P(Point);
impl Positioned for P { fn position(&self) -> Point { self.0 } }

const WORLD: f64 = 1024.0;
const N: usize = 50_000;
const IL: usize = 8;
const MARGIN: f64 = 2.0;

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() {
    let on = cfg!(feature = "neighbors");

    // Initial positions + velocities (a high speed so points cross leaf borders
    // often → plenty of splits/merges → rope rewiring on the feature build).
    let mut rng = Rng::new(7);
    let pts: Vec<P> = (0..N).map(|_| P(Point::new(rng.range(MARGIN, WORLD - MARGIN), rng.range(MARGIN, WORLD - MARGIN)))).collect();
    let mut vel: Vec<(f64, f64)> = (0..N).map(|_| (rng.range(-200.0, 200.0), rng.range(-200.0, 200.0))).collect();

    // BUILD — best of a few rebuilds.
    let mut build = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let mut tree = Tree::<P>::new(Rect::new(0.0, 0.0, WORLD, WORLD), IL);
        for p in &pts { std::hint::black_box(tree.insert_ref(*p)); }
        build.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&tree);
    }
    let build_ms = build.iter().cloned().fold(f64::INFINITY, f64::min);

    // CHURN — relocate every point per frame via update_ref (O(1) handle path).
    let mut tree = Tree::<P>::new(Rect::new(0.0, 0.0, WORLD, WORLD), IL);
    let mut refs: Vec<ItemRef> = pts.iter().map(|p| tree.insert_ref(*p).unwrap()).collect();
    let mut pos: Vec<Point> = pts.iter().map(|p| p.0).collect();
    let dt = 1.0 / 60.0;
    let mut samples = Vec::new();
    for _ in 0..120 {
        let t = Instant::now();
        for i in 0..N {
            let (mut vx, mut vy) = vel[i];
            let mut nx = pos[i].x + vx * dt;
            let mut ny = pos[i].y + vy * dt;
            if nx < MARGIN || nx > WORLD - MARGIN { vx = -vx; nx = nx.clamp(MARGIN, WORLD - MARGIN); }
            if ny < MARGIN || ny > WORLD - MARGIN { vy = -vy; ny = ny.clamp(MARGIN, WORLD - MARGIN); }
            vel[i] = (vx, vy);
            pos[i] = Point::new(nx, ny);
            tree.update_ref(refs[i], |p| p.0 = pos[i]);
        }
        samples.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&tree);
    }
    let _ = &mut refs;
    let upd_min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let upd_med = median(samples);

    println!("ropes_balance | neighbors feature: {}", if on { "ON (ropes maintained)" } else { "off" });
    println!("  world={WORLD}² | N={N} | item_limit={IL}");
    println!("  build (insert {N})      : {:>8.3} ms  (min of 7)", build_ms);
    println!("  update/frame (relocate {N}): {:>8.3} ms  (min)  {:>8.3} ms (median)", upd_min, upd_med);
    println!("\nRun this with AND without --features neighbors; the delta is the rope");
    println!("maintenance cost. Cross with the ~30-45% per-query DDA/flood win in");
    println!("docs/RAYCAST.md to find the break-even query:mutation ratio.");
}
