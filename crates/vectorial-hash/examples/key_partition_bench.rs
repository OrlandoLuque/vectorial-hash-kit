//! **How should you cut a point set into K shards — by key order, or by tree structure?**
//!
//! The first version of this bench asked a narrower question and answered it with one number. It
//! compared four *orderings* (Morton, Hilbert, one coordinate, random), sliced each into K equal
//! runs, and reported how many shards a query touches. The conclusion it drew — that a key is what
//! makes a shard boundary mean anything — was true of what it measured and **left out the strongest
//! alternative**: you can walk a tree, see which branch is heavy, and cut along node boundaries.
//!
//! That alternative is not a hypothetical. It is the standard approach and it has a decade of
//! literature: SpatialHadoop ships **Quadtree, KD-tree, STR and STR+** partitioners beside its
//! Z-curve and Hilbert ones (Eldawy, Alarabi & Mokbel, VLDB 2015), and R\*-Grove describes the
//! family as *"reuse existing index search trees as-is … by building a temporary tree for a sample
//! of the input and use its leaf nodes as partition boundaries"* (Vu & Eldawy, 2020). So the arm
//! was missing, not absent for a reason.
//!
//! ## What the literature says the trade-off is, which is not what I said
//!
//! R\*-Grove is explicit, and it splits the axis differently than I did:
//!
//! - of the SFC family: *"While this method can ensure a **near-perfect load balance**, it produces
//!   an even **bigger spatial overlap** between partitions."*
//! - of the tree family: *"Some partitioning techniques (STR, Kd-tree) prioritize load balance over
//!   **spatial quality** which results in suboptimal partitions."*
//!
//! So the contest is **balance against spatial quality**, and "spatial quality" is measured — not
//! by shards-touched, which is what this bench used to print, but by the volume and the *overlap*
//! of the partitions' bounding boxes. Those are R\*-Grove's quality metrics Q1 and Q2; Q5 is the
//! standard deviation of partition sizes. This bench now reports all of them beside the fan-out,
//! because reporting only the metric that happens to favour your conclusion is how § 8i happens.
//!
//! ## ★ What it measured, and how it went against my prediction twice
//!
//! I expected the tree arm to win on spatial quality, since its shards are unions of compact cubes
//! rather than fragmented key ranges. **Grouped the obvious way — largest leaf first into the
//! emptiest shard, which is what "the tree knows which branch is heavy" actually licenses — it is
//! the worst arm but one**: Q2 overlap of 288 and 13 381 against Morton's 4.5 and 6.8. Each shard
//! *is* a union of compact cubes, but cubes scattered across the whole world, because greedy packing
//! groups by **size**. A leaf's compactness does not survive the grouping.
//!
//! Group the leaves in **curve order** instead and the leaf size becomes the dial. Coarse leaves
//! buy spatial quality and wreck balance (30× max/mean at K=512, because a shard can only be cut at
//! a leaf boundary and one dense leaf overflows it). Fine leaves land on Morton's numbers in every
//! column, balance included — which is the second thing I did not predict, and the real answer:
//!
//! > **A tree whose leaves are much finer than a shard, traversed in curve order, IS a key sort at
//! > leaf granularity.** The key is not a *better* partitioner than the tree; it is the *same* one.
//! > What it adds is that it can cut **anywhere**, where a tree cuts only at node boundaries, and
//! > that quantisation is what forces the choice between balance and locality.
//!
//! The difference that survives is not in the table: which shard owns a point is two comparisons on
//! a number its holder computes, where a tree partition is a node→shard directory somebody has to
//! ship, agree on and keep in step. MD-HBase does both — *"applying the Z-curve on the input data
//! and customizing the region split method in HBase to respect the structure of both indexes"*.
//!
//! And one methodological catch: **`X-stripe` has the best Q1/Q2 of every arm and by far the worst
//! fan-out.** Total area is a *join* metric — VLDB 2015 says so in as many words — so reading it
//! alone would pick the worst partitioner here. Neither column ranks partitioners by itself.
//!
//! ## One idealisation, stated rather than hidden
//!
//! The four ordering arms slice the sorted array into K runs of **exactly** `n/K`, so their load
//! balance is perfect *by construction*. Real SFC partitioners cut on a **sample** and at cell
//! boundaries, so they do not achieve this — which is precisely why R\*-Grove lists imbalance as a
//! limitation of Z-curve and Hilbert partitioning too. Read the ordering arms' Q5 as a floor that
//! a deployed system would not reach, and the octree arm's as what a real one looks like.
//!
//! ```bash
//! cargo run -p vectorial-hash --example key_partition_bench --release
//! ```
//! Env: `KP_N` (points), `KP_Q` (queries).

#[path = "common/mod.rs"]
mod common;

use common::{hilbert3, morton3};

const WORLD: f64 = 10_000.0;
const BITS: u32 = 10;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

fn cell(v: f64) -> u32 {
    (((v / WORLD) * (1u32 << BITS) as f64) as i64).clamp(0, ((1u32 << BITS) - 1) as i64) as u32
}

#[derive(Clone, Copy)]
struct P { x: f64, y: f64, z: f64 }

/// An axis-aligned box, accumulated from points. `None` means empty.
#[derive(Clone, Copy)]
struct Box3 { lo: [f64; 3], hi: [f64; 3] }
impl Box3 {
    fn of(p: &P) -> Self { Box3 { lo: [p.x, p.y, p.z], hi: [p.x, p.y, p.z] } }
    fn add(&mut self, p: &P) {
        for (k, v) in [p.x, p.y, p.z].iter().enumerate() {
            if *v < self.lo[k] { self.lo[k] = *v; }
            if *v > self.hi[k] { self.hi[k] = *v; }
        }
    }
    fn volume(&self) -> f64 { (0..3).map(|k| (self.hi[k] - self.lo[k]).max(0.0)).product() }
    fn overlap(&self, o: &Box3) -> f64 {
        (0..3).map(|k| (self.hi[k].min(o.hi[k]) - self.lo[k].max(o.lo[k])).max(0.0)).product()
    }
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// **The arm the first version of this bench was missing.** Partition by recursively splitting
/// octants until a node holds no more than `target`, then bin-packing the resulting leaves into `k`
/// shards — largest leaf first into the emptiest shard (LPT, the standard greedy).
///
/// This is deliberately the *good* version of the user's objection: it uses the tree's own knowledge
/// of which branch is heavy, and each shard is a union of compact cubes rather than a fragmented key
/// range. What it cannot do is hit exactly `n/k` per shard, because leaves come in quantised sizes —
/// and that quantisation is the whole trade-off.
/// How leaves are grouped into shards once the tree has been cut into them.
#[derive(Clone, Copy, PartialEq)]
enum Group {
    /// Largest leaf first into the emptiest shard — LPT, the standard greedy for balance.
    BySize,
    /// Consecutive leaves in **curve order**, cutting when a shard is full. Groups by locality
    /// instead of by size, which is what MD-HBase does: the Z-curve supplies the ordering and the
    /// tree supplies the split points.
    ByCurve,
}

fn octree_leaf_partition(pts: &[P], k: usize, target: usize) -> Vec<u32> {
    octree_leaf_partition_grouped(pts, k, target, Group::BySize)
}

fn octree_leaf_partition_grouped(pts: &[P], k: usize, target: usize, how: Group) -> Vec<u32> {
    // Leaves as index lists. Iterative to keep the stack shallow at high resolution.
    let mut leaves: Vec<Vec<u32>> = Vec::new();
    let all: Vec<u32> = (0..pts.len() as u32).collect();
    let mut stack: Vec<(Vec<u32>, [f64; 3], f64)> = vec![(all, [0.0, 0.0, 0.0], WORLD)];
    while let Some((idx, lo, side)) = stack.pop() {
        // `side <= tiny` stops the recursion on coincident or near-coincident points, which would
        // otherwise subdivide forever and is exactly the `inseparable` guard the kit's trees carry.
        if idx.len() <= target || side <= WORLD / (1u32 << BITS) as f64 {
            if !idx.is_empty() { leaves.push(idx); }
            continue;
        }
        let h = side * 0.5;
        let mut buckets: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        for &i in &idx {
            let p = &pts[i as usize];
            let o = ((p.x >= lo[0] + h) as usize) | (((p.y >= lo[1] + h) as usize) << 1)
                | (((p.z >= lo[2] + h) as usize) << 2);
            buckets[o].push(i);
        }
        for (o, b) in buckets.into_iter().enumerate() {
            if b.is_empty() { continue; }
            let clo = [
                lo[0] + if o & 1 != 0 { h } else { 0.0 },
                lo[1] + if o & 2 != 0 { h } else { 0.0 },
                lo[2] + if o & 4 != 0 { h } else { 0.0 },
            ];
            stack.push((b, clo, h));
        }
    }
    let mut owner = vec![0u32; pts.len()];
    match how {
        Group::BySize => {
            // LPT bin-packing: biggest leaf first, into whichever shard is currently emptiest.
            leaves.sort_by_key(|l| std::cmp::Reverse(l.len()));
            let mut load = vec![0usize; k];
            for leaf in &leaves {
                let s = (0..k).min_by_key(|&s| load[s]).expect("k > 0");
                load[s] += leaf.len();
                for &i in leaf { owner[i as usize] = s as u32; }
            }
        }
        Group::ByCurve => {
            // Leaves in CURVE order, filled sequentially. Each leaf is keyed by the Morton code of
            // its own first point, which is enough: all of a leaf's points share the prefix that
            // defines the leaf, so any of them orders it identically.
            leaves.sort_by_key(|l| {
                let p = &pts[l[0] as usize];
                morton3(cell(p.x), cell(p.y), cell(p.z))
            });
            let mut s = 0usize;
            let mut load = 0usize;
            let quota = pts.len().div_ceil(k);
            for leaf in &leaves {
                // Move on when the current shard is full, but never past the last one.
                if load + leaf.len() > quota && s + 1 < k { s += 1; load = 0; }
                load += leaf.len();
                for &i in leaf { owner[i as usize] = s as u32; }
            }
        }
    }
    owner
}

/// R*-Grove's quality metrics, for one partitioning. `(Q5 max/mean, Q5 stddev/mean, Q1 volume,
/// Q2 pairwise overlap)`, the volumes normalised by the world's so they read as fractions.
fn quality(pts: &[P], owner: &[u32], k: usize) -> (f64, f64, f64, f64) {
    let mut sizes = vec![0usize; k];
    let mut boxes: Vec<Option<Box3>> = vec![None; k];
    for (i, p) in pts.iter().enumerate() {
        let s = owner[i] as usize;
        sizes[s] += 1;
        match &mut boxes[s] {
            Some(b) => b.add(p),
            none => *none = Some(Box3::of(p)),
        }
    }
    let mean = pts.len() as f64 / k as f64;
    let max = *sizes.iter().max().unwrap_or(&0) as f64;
    let var = sizes.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / k as f64;
    let world_v = WORLD * WORLD * WORLD;
    let q1: f64 = boxes.iter().flatten().map(|b| b.volume()).sum::<f64>() / world_v;
    let mut q2 = 0.0;
    for a in 0..k {
        for b in (a + 1)..k {
            if let (Some(x), Some(y)) = (&boxes[a], &boxes[b]) { q2 += x.overlap(y); }
        }
    }
    (max / mean, var.sqrt() / mean, q1, q2 / world_v)
}

fn main() {
    let n: usize = env("KP_N", 120_000);
    let nq: usize = env("KP_Q", 200);
    println!("key_partition_bench — {n} points, {nq} queries, world {WORLD}, {BITS} bits/axis");
    println!("{}", vectorial_hash::machine_line());
    println!("\nKey order vs TREE STRUCTURE, on the metrics the literature uses (R*-Grove Q1/Q2/Q5)");
    println!("as well as query fan-out. The four ordering arms get exactly n/K per shard BY");
    println!("CONSTRUCTION, which a real sampled SFC partitioner does not — read their balance as a");
    println!("floor. The octree arm bin-packs leaves, so its imbalance is the honest kind.\n");

    for (label, clustered) in [("uniform", false), ("clustered", true)] {
        let mut r = Rng(0xC0FFEE);
        let pts: Vec<P> = if clustered {
            let blobs: Vec<(f64, f64, f64)> = (0..12)
                .map(|_| (r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD)).collect();
            (0..n).map(|i| {
                let b = blobs[i % blobs.len()];
                let g = |a: f64, r: &mut Rng| (a + (r.unit() - 0.5) * 600.0).clamp(0.0, WORLD - 1.0);
                P { x: g(b.0, &mut r), y: g(b.1, &mut r), z: g(b.2, &mut r) }
            }).collect()
        } else {
            (0..n).map(|_| P { x: r.unit() * WORLD, y: r.unit() * WORLD, z: r.unit() * WORLD }).collect()
        };

        let key_of = |o: usize, p: &P| -> u64 {
            match o {
                0 => morton3(cell(p.x), cell(p.y), cell(p.z)),
                1 => hilbert3(cell(p.x), cell(p.y), cell(p.z), BITS),
                2 => cell(p.x) as u64, // x-stripe: one coordinate, nothing else
                _ => 0,
            }
        };
        const ORDERS: [&str; 7] = ["Morton", "Hilbert", "X-stripe", "random",
                                   "Oct/size", "Oct/curve x4", "Oct/curve x32"];
        const OCT_SIZE: usize = 4;
        const OCT_CURVE: usize = 5;
        const OCT_CURVE_FINE: usize = 6;
        let mut perms: Vec<Vec<u32>> = Vec::new();
        let mut rr = Rng(0x5EED);
        for o in 0..4 {
            let mut idx: Vec<u32> = (0..n as u32).collect();
            if o == 3 {
                for i in (1..idx.len()).rev() { let j = ((rr.unit() * (i + 1) as f64) as usize).min(i); idx.swap(i, j); }
            } else {
                idx.sort_by_key(|&i| key_of(o, &pts[i as usize]));
            }
            perms.push(idx);
        }

        // Query centres are drawn FROM THE POINTS, in both distributions. Uniform centres in a
        // 10 000-wide world miss twelve blobs of radius ~300 almost every time, so the clustered
        // table came out with every query empty and the guard below caught it on the first run.
        let queries: Vec<(f64, f64, f64)> = (0..nq)
            .map(|_| { let p = pts[(r.next() % n as u64) as usize]; (p.x, p.y, p.z) })
            .collect();

        const KS: [usize; 3] = [8, 64, 512];
        const RADII: [f64; 3] = [100.0, 300.0, 900.0];

        // owners[order][k index][point]. Precomputed: recomputing inside the query loop makes this
        // O(orders x K x radii x queries x n) and it does not finish. The brute-force membership
        // scan runs ONCE per (query, radius) and is shared by every (order, K) cell, which is also
        // what makes them exactly comparable.
        let mut owners: Vec<Vec<Vec<u32>>> = Vec::with_capacity(ORDERS.len());
        // Indexed by arm, not iterated over `perms`: only the first four arms HAVE a permutation,
        // the octree ones build their owner map directly.
        for (o, _) in ORDERS.iter().enumerate() {
            let mut per_k = Vec::with_capacity(KS.len());
            for &k in &KS {
                if o == OCT_SIZE {
                    per_k.push(octree_leaf_partition(&pts, k, (n / k).max(1)));
                } else if o == OCT_CURVE || o == OCT_CURVE_FINE {
                    // Leaf size is the DIAL, so both ends of it are shown. A shard can only be cut
                    // at a leaf boundary, so the finer the leaves the better the balance and the
                    // more leaves each shard has to stitch together. x4 means ~4 leaves per shard,
                    // x32 means ~32 — and the balance column is where that shows.
                    let per_shard = if o == OCT_CURVE { 4 } else { 32 };
                    per_k.push(octree_leaf_partition_grouped(&pts, k, (n / (per_shard * k)).max(1), Group::ByCurve));
                } else {
                    let mut owner = vec![0u32; n];
                    for (rank, &i) in perms[o].iter().enumerate() { owner[i as usize] = (rank * k / n) as u32; }
                    per_k.push(owner);
                }
            }
            owners.push(per_k);
        }

        let mut acc = vec![[[0usize; RADII.len()]; KS.len()]; ORDERS.len()];
        let mut nonempty = [0usize; RADII.len()];
        let mut hits: Vec<u32> = Vec::new();
        for (ri, &radius) in RADII.iter().enumerate() {
            for &(cx, cy, cz) in &queries {
                hits.clear();
                for (i, p) in pts.iter().enumerate() {
                    let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz);
                    if dx * dx + dy * dy + dz * dz <= radius * radius { hits.push(i as u32); }
                }
                // Empty queries are skipped rather than counted as zero: they touch no shard under
                // every ordering, so including them would pull all rows toward 0 by the same amount
                // and compress the differences this table exists to show.
                if hits.is_empty() { continue; }
                nonempty[ri] += 1;
                for o in 0..ORDERS.len() {
                    for ki in 0..KS.len() {
                        let mut seen = vec![false; KS[ki]];
                        for &i in &hits { seen[owners[o][ki][i as usize] as usize] = true; }
                        acc[o][ki][ri] += seen.iter().filter(|&&b| b).count();
                    }
                }
            }
        }
        assert!(nonempty.iter().all(|&c| c > 0), "every query was empty — this proves nothing");

        println!("== {label} ==");
        for (ki, &k) in KS.iter().enumerate() {
            println!("  K = {k}");
            println!("    {:<12} {:>7} {:>7} {:>7} | {:>8} {:>8} | {:>8} {:>9}",
                     "partitioner", "r=100", "r=300", "r=900", "max/mean", "sd/mean", "Q1 vol", "Q2 ovlp");
            for (o, name) in ORDERS.iter().enumerate() {
                let c: Vec<f64> = (0..RADII.len())
                    .map(|ri| acc[o][ki][ri] as f64 / nonempty[ri] as f64).collect();
                let (mx, sd, q1, q2) = quality(&pts, &owners[o][ki], k);
                println!("    {:<12} {:>7.2} {:>7.2} {:>7.2} | {:>8.2} {:>8.3} | {:>8.3} {:>9.3}",
                         name, c[0], c[1], c[2], mx, sd, q1, q2);
            }
        }
        println!();
    }

    println!("Reading it. The headline is that a tree partition done PROPERLY converges on the key");
    println!("partition, and the two ways of getting it wrong fail in opposite directions.");
    println!();
    println!("  `random` is the control: balanced shards with no spatial meaning, so a query reaches");
    println!("  nearly every one. Everything above it is what an ORDER is worth. `X-stripe` is the");
    println!("  warning against reading one column: it has the BEST Q1/Q2 of all (a slab has a tight");
    println!("  bounding box by construction) and by far the worst fan-out. Total area is a JOIN");
    println!("  metric -- VLDB 2015 says exactly that -- and neither metric alone ranks partitioners.");
    println!();
    println!("  ★ `Oct/size` is the natural reading of \"the tree knows which branch is heavy\": cut");
    println!("  into leaves, then bin-pack them largest-first for balance. Balance is fine (1.03-1.06)");
    println!("  and SPATIAL QUALITY COLLAPSES -- Q2 overlap of 288 and 13 381 against Morton's 4.5 and");
    println!("  6.8. The reason is worth more than the number: each shard IS a union of compact cubes,");
    println!("  but cubes from all over the world, because greedy packing groups by SIZE. A leaf's");
    println!("  compactness does not survive the grouping.");
    println!();
    println!("  ★ `Oct/curve` fixes that by grouping consecutive leaves in CURVE order instead, and");
    println!("  then the leaf size is the dial. Coarse leaves (x4 = ~4 per shard) buy spatial quality");
    println!("  and wreck balance -- 30x max/mean at K=512 -- because a shard can only be cut at a");
    println!("  leaf boundary and one dense leaf overflows it. Fine leaves (x32) land on Morton's");
    println!("  numbers in every column, balance included.");
    println!();
    println!("  Which is the answer, and it is not the one this bench originally implied. A tree whose");
    println!("  leaves are much finer than a shard, traversed in curve order, IS a key sort at leaf");
    println!("  granularity. So the key is not a BETTER partitioner than the tree -- it is the SAME");
    println!("  partitioner. What the key adds is that it can cut ANYWHERE, where a tree can only cut");
    println!("  at a node boundary, and that quantisation is what forces the choice between balance");
    println!("  and locality that `Oct/curve x4` shows.");
    println!();
    println!("  The remaining, real difference is not in this table at all: which shard owns a point");
    println!("  is two comparisons on a number the holder computes itself, where a tree partition is a");
    println!("  node->shard directory somebody has to ship, agree on and keep in step. MD-HBase does");
    println!("  both at once -- Z-curve for the address space, tree-shaped region splits for quality.");
}
