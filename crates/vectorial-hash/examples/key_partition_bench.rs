//! **What a prefix key buys that a tree does not: the SPLIT.**
//!
//! `radix_trie_bench` asks which structure answers a sphere fastest in one process, and the trie
//! loses. That is a real answer to a narrow question, and it is not the question anyone choosing a
//! key-ordered index is asking. The property they are after is that **the key orders the data, so
//! any contiguous run of keys is a valid, spatially-coherent partition** — you cut the sorted key
//! space at K−1 points and you have K shards, balanced by construction, each of which is a range
//! any participant can name without consulting the index.
//!
//! A pointer tree cannot do that. Its subtree populations are whatever the data made them, so
//! "give me K equal parts" has no answer in the tree — you would first have to produce an ordering,
//! which is the key. That is the whole argument, and this bench turns it into two numbers.
//!
//! ## The number that decides it: how many shards does one query touch?
//!
//! A query served by 1 shard is one machine answering. A query touching 8 is a scatter-gather with
//! 8 round trips and a merge. So **partitions-touched is the cost of partitioning**, and it is a
//! **count** — exactly reproducible, which is what this laptop can be trusted to produce
//! (`docs/MEASURING.md` § 8, § 8e).
//!
//! Four orderings, same points, same queries:
//!
//! | order | what it models |
//! | --- | --- |
//! | **Morton** | the kit's curve, and a geohash |
//! | **Hilbert** | the better-clustering curve (`SPACE_FILLING_CURVES.md`) |
//! | **X-stripe** | the obvious non-curve answer: sort by one coordinate |
//! | **random** | the strawman — balanced shards, no spatial meaning |
//!
//! The random row is not padding. It is the control that says how much of the result is "sorting
//! by *anything* helps" versus "sorting by a **space-filling curve** helps", and without it the
//! other three numbers have nothing to be better *than*.
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

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n: usize = env("KP_N", 200_000);
    let nq: usize = env("KP_Q", 300);
    println!("key_partition_bench — {n} points, {nq} queries, world {WORLD}, {BITS} bits/axis");
    println!("{}", vectorial_hash::machine_line());
    println!("\nThe question is not which index is fastest in one process. It is what the KEY buys:");
    println!("a contiguous run of keys is a shard, balanced by construction. Cost of that = how");
    println!("many shards a query has to visit. Counts, so they reproduce exactly.\n");

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

        // Four orderings of the SAME points. Each produces a permutation; shard s owns the slice
        // [s*n/K, (s+1)*n/K), so every ordering gets shards of identical size — the comparison is
        // purely about whether the order means anything spatially.
        let key_of = |o: usize, p: &P| -> u64 {
            match o {
                0 => morton3(cell(p.x), cell(p.y), cell(p.z)),
                1 => hilbert3(cell(p.x), cell(p.y), cell(p.z), BITS),
                2 => cell(p.x) as u64, // x-stripe: one coordinate, nothing else
                _ => 0,
            }
        };
        const ORDERS: [&str; 4] = ["Morton", "Hilbert", "X-stripe", "random"];
        // shard_of[order][point index]
        let mut shard_of: Vec<Vec<u32>> = Vec::new();
        let mut rr = Rng(0x5EED);
        for o in 0..4 {
            let mut idx: Vec<u32> = (0..n as u32).collect();
            if o == 3 {
                for i in (1..idx.len()).rev() { let j = ((rr.unit() * (i + 1) as f64) as usize).min(i); idx.swap(i, j); }
            } else {
                idx.sort_by_key(|&i| key_of(o, &pts[i as usize]));
            }
            shard_of.push(idx);
        }

        // Query centres are drawn FROM THE POINTS, in both distributions, and that is not a
        // convenience. Uniform centres in a 10 000-wide world miss twelve blobs of radius ~300
        // almost every time, so the clustered table came out with every query empty — the
        // non-vacuity guard below caught it on the first run. Querying where the data is not
        // measures nothing, and doing it the same way for both distributions is what keeps the
        // two tables comparable to each other.
        let queries: Vec<(f64, f64, f64)> = (0..nq)
            .map(|_| { let p = pts[(r.next() % n as u64) as usize]; (p.x, p.y, p.z) })
            .collect();

        const KS: [usize; 3] = [8, 64, 512];
        const RADII: [f64; 3] = [100.0, 300.0, 900.0];

        // owners[order][k index][point] = shard. Precomputed, because the alternative — recomputing
        // inside the query loop — makes this O(orders x K x radii x queries x n) and it simply does
        // not finish. The brute-force membership scan is done ONCE per (query, radius) and shared
        // by all twelve (order, K) cells, which is also what makes them exactly comparable.
        let mut owners: Vec<Vec<Vec<u32>>> = Vec::with_capacity(4);
        for perm in shard_of.iter() {
            let mut per_k = Vec::with_capacity(KS.len());
            for &k in &KS {
                let mut owner = vec![0u32; n];
                for (rank, &i) in perm.iter().enumerate() { owner[i as usize] = (rank * k / n) as u32; }
                per_k.push(owner);
            }
            owners.push(per_k);
        }

        let mut acc = [[[0usize; RADII.len()]; KS.len()]; 4];
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
                // every ordering, so including them would pull all four rows toward 0 by the same
                // amount and compress the very differences this table exists to show.
                if hits.is_empty() { continue; }
                nonempty[ri] += 1;
                for o in 0..4 {
                    for ki in 0..KS.len() {
                        let mut seen = vec![false; KS[ki]];
                        for &i in &hits { seen[owners[o][ki][i as usize] as usize] = true; }
                        acc[o][ki][ri] += seen.iter().filter(|&&b| b).count();
                    }
                }
            }
        }
        assert!(nonempty.iter().all(|&c| c > 0), "every query was empty — this proves nothing");

        println!("== {label} ==   (mean shards touched per non-empty query)");
        for (ki, &k) in KS.iter().enumerate() {
            println!("  K = {k} shards of {} points each", n / k);
            println!("    {:<10} {:>8} {:>8} {:>8}   {:>13}", "order", "r=100", "r=300", "r=900", "of K at r=900");
            for (o, name) in ORDERS.iter().enumerate() {
                let c: Vec<f64> = (0..RADII.len())
                    .map(|ri| acc[o][ki][ri] as f64 / nonempty[ri] as f64).collect();
                println!("    {:<10} {:>8.2} {:>8.2} {:>8.2}   {:>12.1}%",
                         name, c[0], c[1], c[2], c[2] / k as f64 * 100.0);
            }
        }
        println!();
    }

    println!("Reading it: `random` is the control — balanced shards with no spatial meaning, so a");
    println!("query fans out to essentially every shard it could. Everything above it is what the");
    println!("ORDER is worth. `X-stripe` is the reminder that a cheap order already buys most of");
    println!("it in one axis and nothing in the other two, which is why it degrades as the query");
    println!("grows. The curves buy the rest.");
    println!();
    println!("What this does NOT measure: any of it in one process, where `radix_trie_bench` shows");
    println!("the trie losing to every structure in this kit. These are different questions, and");
    println!("conflating them is how a structure gets chosen for a property nobody measured.");
}
