//! Cache-locality bench for [`Tree3::compact`] **and [`Octree3::compact`]**: does
//! DFS-reordering the node arena speed up culls on a tree whose node order has been
//! *scrambled* by a long keep-index run (per-frame `update_ref` relocations →
//! splits/merges land in scattered free slots)? Measured on both the binary tree
//! and the 8-way octree (same churn, same query set).
//!
//! ```bash
//! cargo run -p vectorial-hash --example compact_bench --release
//! N=200000 cargo run -p vectorial-hash --example compact_bench --release
//! ```
//! Reports µs/cull scrambled vs after `compact()` (min-of-N, warm). Pure layout
//! change — the cull results are identical (asserted in the lib test), only the
//! memory order of the nodes differs.

use std::time::Instant;
use vectorial_hash::{Aabb, ItemRef, Octree3, Point3, Positioned3, Sphere3, Tree3};

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let frames: usize = std::env::var("FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let w = 512.0;
    let world = Aabb::new(0.0, 0.0, 0.0, w, w, w);
    let mut x = 0x1234_5678u64;
    let mut rng = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };

    // Build + churn: the keep-index workload. Random walks push items across leaf
    // boundaries, so splits/merges recycle scattered free slots and the arena
    // drifts out of any locality (insertion order ≠ traversal order).
    let mut tree = Tree3::<P>::new(world, 8);
    let mut refs: Vec<ItemRef> = Vec::with_capacity(n);
    for _ in 0..n {
        let p = Point3::new(rng() * w, rng() * w, rng() * w);
        refs.push(tree.insert_ref(P(p)).unwrap());
    }
    for _ in 0..frames {
        for &r in &refs {
            let (dx, dy, dz) = ((rng() - 0.5) * 24.0, (rng() - 0.5) * 24.0, (rng() - 0.5) * 24.0);
            // Keep strictly INSIDE the root (the demos clamp to MARGIN..W-MARGIN):
            // landing on the half-open upper edge would leave the root, and
            // update_ref frees the handle — reusing it next frame would panic.
            tree.update_ref(r, |q| q.0 = Point3::new(
                (q.0.x + dx).clamp(1.0, w - 1.0), (q.0.y + dy).clamp(1.0, w - 1.0), (q.0.z + dz).clamp(1.0, w - 1.0)));
        }
    }

    let queries: Vec<Sphere3> = (0..3000).map(|_| Sphere3::new(rng() * w, rng() * w, rng() * w, 24.0)).collect();
    let bench = |t: &Tree3<P>| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..10 {
            let s = Instant::now();
            let mut acc = 0usize;
            for q in &queries { acc += t.cull(q).len(); }
            std::hint::black_box(acc);
            best = best.min(s.elapsed().as_secs_f64());
        }
        best * 1e6 / queries.len() as f64
    };

    let scrambled = bench(&tree);
    let (nc0, lc0) = (tree.node_count(), tree.live_node_count());
    let t0 = Instant::now();
    tree.compact();
    let compact_us = t0.elapsed().as_secs_f64() * 1e6;
    let compacted = bench(&tree);

    // --- Octree3: the same churn + compact question on the 8-way tree ---
    let mut otree = Octree3::<P>::new(world, 8);
    let mut orefs: Vec<ItemRef> = Vec::with_capacity(n);
    for _ in 0..n { let p = Point3::new(rng() * w, rng() * w, rng() * w); orefs.push(otree.insert_ref(P(p)).unwrap()); }
    for _ in 0..frames {
        for &r in &orefs {
            let (dx, dy, dz) = ((rng() - 0.5) * 24.0, (rng() - 0.5) * 24.0, (rng() - 0.5) * 24.0);
            otree.update_ref(r, |q| q.0 = Point3::new((q.0.x + dx).clamp(1.0, w - 1.0), (q.0.y + dy).clamp(1.0, w - 1.0), (q.0.z + dz).clamp(1.0, w - 1.0)));
        }
    }
    let obench = |t: &Octree3<P>| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..10 { let s = Instant::now(); let mut acc = 0usize; for q in &queries { acc += t.cull(q).len(); } std::hint::black_box(acc); best = best.min(s.elapsed().as_secs_f64()); }
        best * 1e6 / queries.len() as f64
    };
    let oscrambled = obench(&otree);
    let (onc0, olc0) = (otree.node_count(), otree.live_node_count());
    let ot0 = Instant::now();
    otree.compact();
    let ocompact_us = ot0.elapsed().as_secs_f64() * 1e6;
    let ocompacted = obench(&otree);

    println!("compact() locality bench | N={n} items | {frames} churn frames | 3000 random culls (min of 10)\n");
    println!("  Tree3   arena {nc0} ({lc0} live) → {} | compact {:.0} µs | cull {scrambled:.3} → {compacted:.3} µs/query | {:.2}×", tree.node_count(), compact_us, scrambled / compacted);
    println!("  Octree3 arena {onc0} ({olc0} live) → {} | compact {ocompact_us:.0} µs | cull {oscrambled:.3} → {ocompacted:.3} µs/query | {:.2}×", otree.node_count(), oscrambled / ocompacted);
}
