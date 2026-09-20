//! **#179 — what a point index costs you when your objects have SIZE.**
//!
//! Every structure in this crate indexes points. `Positioned3`'s rustdoc and the top of
//! `docs/CHOOSING.md` now say so as a precondition, because it produces a wrong **answer** and not
//! a slow one: file a 3 km ship by its centre and a query for "everything within 500 m" misses it
//! while its hull is 100 m away. That much is arithmetic. What was argued rather than measured is
//! **what each way out costs**, and the argument named a specific hope — that the size distribution
//! is so skewed you can carry the few big objects in a list beside the index. This measures it.
//!
//! The world is deliberately the shape the question came from: 10 km across, with a handful of
//! km-scale objects among tens of thousands of metre-scale ones.
//!
//! Four arms, all answering the SAME question — *which objects' hulls intersect this sphere* — so
//! that recall is comparable and a faster arm cannot be faster by answering less:
//!
//! 1. **`centre`** — index the centres, query at `r`, take what comes back. Fast and **wrong**; the
//!    point of the run is its recall.
//! 2. **`enlarged`** — query at `r + R_max`, then test each candidate exactly. Correct. Pays for a
//!    sphere grown by the largest object in the world, whether or not any are nearby.
//! 3. **`tiered`** — one index per size class, each queried at `r + R_max(tier)`. Correct, and the
//!    enlargement is local to the tier.
//! 4. **`hybrid`** — small objects in an index queried at `r + R_max(small)`, every large object
//!    scanned linearly. Correct. The arm the argument predicted would win — it ties instead, see
//!    below.
//!
//! ## What it measured, including the two things that corrected me
//!
//! **Recall 0.9566** for the centre index: 234 missed out of 5 392 true intersections. And the
//! misses are **not mostly the big objects** — 94 are *small* ones, against 118 huge. Per object the
//! huge ones are ~1 200× more likely to be missed (118 from 48, against 94 from 49 499), but in
//! absolute terms the boundary cases of a numerous small population matter just as much. So this
//! cannot be fixed by "handle the ships specially": a radius-5 object whose centre sits at 502 from
//! a radius-500 query is missed too, and there are tens of thousands of them.
//!
//! **`tiered` and `hybrid` tie** (3.38 vs 3.35 µs/query) and both beat `enlarged` by **8.5×**. The
//! win is entirely in *not enlarging by the global maximum*; whether the large objects live in an
//! index or a list is worth nothing at these counts. `tiered` examines **27.9** candidates per query
//! against the hybrid's 527.3 and the global enlargement's 1 282.5, so if anything it is the arm to
//! prefer — it is the tightest, and an index over 501 objects costs nothing to build.
//!
//! **A linear scan carries far more than expected**: still 0.78× the enlarged arm at **16 032**
//! large objects, losing only between 16 k and 32 k. "Count the large objects first" turns out to be
//! very forgiving advice.
//!
//! Recall is a **count** and says the same thing on any machine (MEASURING.md § 8). The
//! microseconds are this box. Exact-hit counts are asserted equal across arms 2–4, so a bug that
//! dropped objects would fail rather than print a good number.
//!
//! Run: `cargo run -p vectorial-hash --example extent_tax --release`
//! Env: `ET_N` (default 50 000), `ET_R` (query radius, default 500), `ET_REPS` (5).

use vectorial_hash::{Aabb, Point3, Positioned3, Sphere3, Tree3};

#[path = "common/mod.rs"]
mod common;

const W: f64 = 10_000.0; // 10 km on a side
const LEAF: usize = 32;

/// An object with extent. `position()` returns its CENTRE, which is exactly the lossy step under
/// examination — the index never learns `radius` exists.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Obj { id: u32, p: Point3, radius: f64 }
impl Positioned3 for Obj { fn position(&self) -> Point3 { self.p } }

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

/// The size classes. Skewed the way real worlds are: almost everything is small, a handful are
/// kilometres. The shares are the load-bearing part of the whole question — if `huge` held 30% of
/// the population the hybrid arm could not exist.
const TIERS: [(&str, f64, f64, f64); 3] = [
    // (name, share, min radius, max radius)
    ("small",  0.990,   1.0,    5.0),
    ("medium", 0.009,  20.0,  100.0),
    ("huge",   0.001, 500.0, 1500.0),
];

fn tier_of(radius: f64) -> usize { TIERS.iter().position(|t| radius <= t.3).unwrap_or(TIERS.len() - 1) }

fn world_objects(n: usize, seed: u64) -> Vec<Obj> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|i| {
            let u = r.f();
            let mut acc = 0.0;
            let mut t = TIERS.len() - 1;
            for (k, tier) in TIERS.iter().enumerate() {
                acc += tier.1;
                if u < acc { t = k; break; }
            }
            let radius = r.r(TIERS[t].2, TIERS[t].3);
            // Centres are kept inside the world; hulls may stick out, which is realistic and
            // harmless — every arm sees the same objects.
            Obj { id: i as u32, p: Point3::new(r.r(0.0, W), r.r(0.0, W), r.r(0.0, W)), radius }
        })
        .collect()
}

#[inline]
fn hull_intersects(o: &Obj, c: Point3, r: f64) -> bool {
    let (dx, dy, dz) = (o.p.x - c.x, o.p.y - c.y, o.p.z - c.z);
    let reach = r + o.radius;
    dx * dx + dy * dy + dz * dz <= reach * reach
}

fn main() {
    let n: usize = std::env::var("ET_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let r: f64 = std::env::var("ET_R").ok().and_then(|v| v.parse().ok()).unwrap_or(500.0);
    let reps: usize = std::env::var("ET_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

    let objs = world_objects(n, 0x5EED_0E01);
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    let r_max = objs.iter().map(|o| o.radius).fold(0.0f64, f64::max);

    // Per-tier populations and per-tier maximum radius, both measured from the data rather than
    // taken from TIERS, because the generator's shares are probabilistic.
    let mut pop = [0usize; TIERS.len()];
    let mut tier_max = [0.0f64; TIERS.len()];
    for o in &objs {
        let t = tier_of(o.radius);
        pop[t] += 1;
        tier_max[t] = tier_max[t].max(o.radius);
    }

    println!("#179 — the extended-object tax");
    println!("machine: {}", vectorial_hash::machine::machine_id());
    println!();
    println!("world {W:.0} wu on a side ({:.0} km if 1 wu = 1 m), {n} objects, query radius {r:.0}", W / 1000.0);
    println!("largest object radius {r_max:.0} wu — so the global enlargement is r + {r_max:.0} = {:.0}", r + r_max);
    println!();
    println!("  {:<8} {:>8} {:>7}  {:>12}  {:>14}", "tier", "count", "share", "max radius", "enlargement");
    for (k, t) in TIERS.iter().enumerate() {
        println!("  {:<8} {:>8} {:>6.2}%  {:>12.0}  {:>13.0}", t.0, pop[k], pop[k] as f64 / n as f64 * 100.0, tier_max[k], r + tier_max[k]);
    }

    // The swept volume each enlargement implies, which is the arithmetic the docs quote.
    println!();
    println!("swept volume against an un-enlarged query of radius {r:.0}:");
    println!("  global enlargement  (r + {:>6.0}) : {:>8.1}x", r_max, ((r + r_max) / r).powi(3));
    for (k, t) in TIERS.iter().enumerate() {
        println!("  {:<6} tier         (r + {:>6.0}) : {:>8.1}x", t.0, tier_max[k], ((r + tier_max[k]) / r).powi(3));
    }

    // --------------------------------------------------------------- indexes
    let mut all: Tree3<Obj> = Tree3::new(world, LEAF);
    for o in &objs { all.insert(*o); }
    let mut per_tier: Vec<Tree3<Obj>> = (0..TIERS.len()).map(|_| Tree3::new(world, LEAF)).collect();
    for o in &objs { per_tier[tier_of(o.radius)].insert(*o); }
    let mut smalls: Tree3<Obj> = Tree3::new(world, LEAF);
    let mut bigs: Vec<Obj> = Vec::new();
    for o in &objs { if tier_of(o.radius) == 0 { smalls.insert(*o); } else { bigs.push(*o); } }

    // Query centres drawn from the objects, so every query has something to find.
    let mut rng = Lcg(0x5EED_0E02);
    let nq = 200usize;
    let centres: Vec<Point3> = (0..nq).map(|_| objs[(rng.f() * n as f64) as usize % n].p).collect();

    // --------------------------------------------------------------- truth, and recall
    let truth: Vec<Vec<u32>> = centres.iter().map(|c| {
        let mut v: Vec<u32> = objs.iter().filter(|o| hull_intersects(o, *c, r)).map(|o| o.id).collect();
        v.sort_unstable(); v
    }).collect();
    let total_true: usize = truth.iter().map(|t| t.len()).sum();

    let mut naive_found = 0usize;
    let mut naive_missed_by_tier = [0usize; TIERS.len()];
    for (q, c) in centres.iter().enumerate() {
        let got: std::collections::HashSet<u32> =
            all.cull(&Sphere3::new(c.x, c.y, c.z, r)).iter().map(|o| o.id).collect();
        for id in &truth[q] {
            if got.contains(id) { naive_found += 1; } else {
                naive_missed_by_tier[tier_of(objs[*id as usize].radius)] += 1;
            }
        }
    }

    println!();
    println!("--- correctness: what a CENTRE index actually returns (a count, machine-independent) ---");
    println!("  true hull intersections over {nq} queries : {total_true}");
    println!("  found by querying the centres at r={r:.0}  : {naive_found}  (recall {:.4})",
             naive_found as f64 / total_true as f64);
    println!("  missed                                    : {}", total_true - naive_found);
    for (k, t) in TIERS.iter().enumerate() {
        println!("    of which {:<6} : {:>6}", t.0, naive_missed_by_tier[k]);
    }

    // --------------------------------------------------------------- the three correct arms
    let exact = |cand: Vec<&Obj>, c: Point3| -> usize { cand.iter().filter(|o| hull_intersects(o, c, r)).count() };

    let mut hits = [0usize; 3];
    let mut cand = [0usize; 3];
    for c in &centres {
        let e = all.cull(&Sphere3::new(c.x, c.y, c.z, r + r_max));
        cand[0] += e.len(); hits[0] += exact(e, *c);
        for (k, idx) in per_tier.iter().enumerate() {
            let e = idx.cull(&Sphere3::new(c.x, c.y, c.z, r + tier_max[k]));
            cand[1] += e.len(); hits[1] += exact(e, *c);
        }
        let e = smalls.cull(&Sphere3::new(c.x, c.y, c.z, r + tier_max[0]));
        cand[2] += e.len(); hits[2] += exact(e, *c);
        cand[2] += bigs.len();
        hits[2] += bigs.iter().filter(|o| hull_intersects(o, *c, r)).count();
    }
    for (k, name) in ["enlarged", "tiered", "hybrid"].iter().enumerate() {
        assert_eq!(hits[k], total_true,
                   "{name} returned {} exact hits where the truth is {total_true} — an arm that \
                    answers a different question cannot be compared against the others", hits[k]);
    }

    let t_enlarged = common::wall_ms(reps, || {
        for c in &centres { std::hint::black_box(exact(all.cull(&Sphere3::new(c.x, c.y, c.z, r + r_max)), *c)); }
    }) * 1e3 / nq as f64;
    let t_tiered = common::wall_ms(reps, || {
        for c in &centres {
            for (k, idx) in per_tier.iter().enumerate() {
                std::hint::black_box(exact(idx.cull(&Sphere3::new(c.x, c.y, c.z, r + tier_max[k])), *c));
            }
        }
    }) * 1e3 / nq as f64;
    let t_hybrid = common::wall_ms(reps, || {
        for c in &centres {
            std::hint::black_box(exact(smalls.cull(&Sphere3::new(c.x, c.y, c.z, r + tier_max[0])), *c));
            std::hint::black_box(bigs.iter().filter(|o| hull_intersects(o, *c, r)).count());
        }
    }) * 1e3 / nq as f64;

    println!();
    println!("--- the three CORRECT arms, all returning exactly {total_true} hits ---");
    println!("  {:<10} {:>12}  {:>14}  {:>10}", "arm", "us/query", "candidates/q", "vs best");
    let best = t_enlarged.min(t_tiered).min(t_hybrid);
    for (k, name) in ["enlarged", "tiered", "hybrid"].iter().enumerate() {
        let t = [t_enlarged, t_tiered, t_hybrid][k];
        println!("  {:<10} {:>12.2}  {:>14.1}  {:>9.2}x", name, t, cand[k] as f64 / nq as f64, t / best);
    }

    // --------------------------------------------------------------- the hybrid's crossover
    //
    // The hybrid's whole case is that the large objects are few enough to scan. That is a claim
    // about a COUNT, so it gets swept rather than asserted: duplicate the large population and
    // find where the linear scan stops being free.
    println!();
    println!("--- how many large objects can a linear scan carry? ---");
    println!("  the hybrid's case is that they are few. Sweeping the count (small index unchanged):");
    println!("  {:>10}  {:>12}  {:>12}", "bigs", "us/query", "vs enlarged");
    let mut grown = bigs.clone();
    let mut step = 0;
    while grown.len() <= 200_000 {
        let t = common::wall_ms(reps.max(3), || {
            for c in &centres {
                std::hint::black_box(exact(smalls.cull(&Sphere3::new(c.x, c.y, c.z, r + tier_max[0])), *c));
                std::hint::black_box(grown.iter().filter(|o| hull_intersects(o, *c, r)).count());
            }
        }) * 1e3 / nq as f64;
        println!("  {:>10}  {:>12.2}  {:>11.2}x", grown.len(), t, t / t_enlarged);
        if t > t_enlarged { println!("  ^ past here the linear scan costs more than the global enlargement it replaces"); break; }
        let cur = grown.clone();
        grown.extend(cur);
        step += 1;
        if step > 12 { println!("  (the scan never lost, up to {} large objects)", grown.len()); break; }
    }

    println!();
    println!("The recall figure is the headline and it is a count, so it holds anywhere. The");
    println!("microsecond columns are one machine on one night; the ORDERING of the three correct");
    println!("arms is the part worth carrying (MEASURING.md 8e).");
}
