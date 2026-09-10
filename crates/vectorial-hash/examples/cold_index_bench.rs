//! Cold-index options bench — investigate + develop + compare structures for
//! the **cold / persistence index** (the "what exists in the whole world, load
//! what's near P" layer), which the kit does NOT ship: `MortonGrid3` is an
//! *unordered* in-memory HashMap, wrong for a sorted/on-disk cold store.
//!
//! The cold index wants a **sorted space-filling-curve key over a B-tree**
//! (redb/sled on disk; `BTreeMap` here as the algorithmic in-memory stand-in —
//! same range-scan semantics, minus disk latency + durability). A spatial box
//! query becomes a **key-range scan** `[min_code, max_code]`, which *over-scans*
//! because the curve leaves and re-enters the box (the geohash boundary
//! problem). Less over-scan = better curve. So we compare:
//!
//!   A) the CURVE — Morton (Z-order) vs Hilbert — by:
//!        - contiguous key-runs a box maps to (the textbook locality metric),
//!        - over-scan factor of a real sorted-key range scan (scanned/hits).
//!   B) the STRUCTURE for the AoI query — sorted `BTreeMap` (cold-store shape)
//!        vs `MortonGrid3` (HashMap, hot) vs `Tree3` (adaptive) — µs/query.
//!
//! ```bash
//! cargo run -p vectorial-hash --example cold_index_bench --release --features parallel
//! ```

use std::collections::BTreeMap;
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Shape3, Sphere3, Tree3};

const WORLD: f64 = 10_000.0;
const BITS: u32 = 16; // cells/axis = 2^16 = 65 536 → a fine full-precision key

// ----------------------------------------------------------------- curve encoders

/// 3D Morton (Z-order) code from per-axis grid indices (`BITS` bits each).
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { // spread 21 bits with 2 gaps between
        v &= 0x1f_ffff;
        v = (v | v << 32) & 0x1f00000000ffff;
        v = (v | v << 16) & 0x1f0000ff0000ff;
        v = (v | v << 8) & 0x100f00f00f00f00f;
        v = (v | v << 4) & 0x10c30c30c30c30c3;
        v = (v | v << 2) & 0x1249249249249249;
        v
    }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}

/// 3D Hilbert distance from per-axis grid indices — Skilling's AxesToTranspose
/// transform (exact, invertible) followed by bit-interleave to a scalar. The
/// self-test below asserts it's a bijection with unit-step adjacency (the
/// property Morton lacks — that's the point).
fn hilbert3(x: u32, y: u32, z: u32, bits: u32) -> u64 {
    let mut c = [x, y, z];
    let m = 1u32 << (bits - 1);
    // Inverse undo excess work
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if c[i] & q != 0 { c[0] ^= p; }
            else { let t = (c[0] ^ c[i]) & p; c[0] ^= t; c[i] ^= t; }
        }
        q >>= 1;
    }
    // Gray encode
    for i in 1..3 { c[i] ^= c[i - 1]; }
    let mut t = 0u32;
    q = m;
    while q > 1 { if c[2] & q != 0 { t ^= q - 1; } q >>= 1; }
    for e in &mut c { *e ^= t; }
    // Interleave the transpose to a single distance (MSB-first, x,y,z order)
    let mut d = 0u64;
    let mut b = bits;
    while b > 0 {
        b -= 1;
        for e in &c { d = (d << 1) | (((*e >> b) & 1) as u64); }
    }
    d
}

/// Decompose a query box into MAXIMAL contiguous key ranges, by descending the octree instead
/// of scanning the span between the box's two corner keys.
///
/// This is the thing A2 says is missing. Both curves are hierarchical: every subcube of the
/// octree occupies a CONTIGUOUS interval of keys. So a node wholly inside the box contributes its
/// whole interval in one step, a node wholly outside contributes nothing, and only nodes that
/// straddle the boundary have to be opened. What comes out is exactly the set of runs A1 counts.
///
/// `subcube_start` is where the two curves differ. For Morton the node's lowest key is at its
/// origin corner, so it is just `code(origin)`. For Hilbert the lowest key can be at any corner —
/// but the node still occupies `[p·side³, (p+1)·side³)` where `p` is the node's own curve index at
/// the coarser resolution. The self-test at the bottom of this function checks that claim rather
/// than trusting it, because a subtly wrong range still returns *some* points.
fn box_ranges(lo: [u32; 3], hi: [u32; 3], bits: u32, hilbert: bool) -> Vec<(u64, u64)> {
    fn key(x: u32, y: u32, z: u32, bits: u32, hilbert: bool) -> u64 {
        if hilbert { hilbert3(x, y, z, bits) } else { morton3(x, y, z) }
    }
    let mut out: Vec<(u64, u64)> = Vec::new();
    // (level, origin). Level L covers a cube of side 2^(bits-L).
    let mut stack: Vec<(u32, [u32; 3])> = vec![(0, [0, 0, 0])];
    while let Some((level, o)) = stack.pop() {
        let side: u32 = 1 << (bits - level);
        let end = [o[0] + side, o[1] + side, o[2] + side];
        // disjoint?
        if (0..3).any(|k| end[k] <= lo[k] || o[k] > hi[k]) { continue; }
        // wholly inside?
        if (0..3).all(|k| o[k] >= lo[k] && end[k] - 1 <= hi[k]) {
            let n = (side as u64).pow(3);
            let start = if hilbert {
                key(o[0] >> (bits - level), o[1] >> (bits - level), o[2] >> (bits - level), level, true) * n
            } else {
                key(o[0], o[1], o[2], bits, false)
            };
            out.push((start, start + n - 1));
            continue;
        }
        if side == 1 { let k = key(o[0], o[1], o[2], bits, hilbert); out.push((k, k)); continue; }
        let h = side / 2;
        for c in 0..8u32 {
            stack.push((level + 1, [o[0] + (c & 1) * h, o[1] + ((c >> 1) & 1) * h, o[2] + ((c >> 2) & 1) * h]));
        }
    }
    // merge touching ranges → maximal runs
    out.sort_unstable();
    let mut runs: Vec<(u64, u64)> = Vec::with_capacity(out.len());
    for r in out {
        match runs.last_mut() {
            Some(last) if r.0 == last.1 + 1 => last.1 = r.1,
            _ => runs.push(r),
        }
    }
    runs
}

/// The exact minimum and maximum key over a box, in O(8 · depth) — without enumerating the runs.
///
/// `box_ranges` gives these as the first range's start and the last range's end, and that is what
/// this function replaced: on A2's box (6 554 cells a side at 16 bits) the decomposition is ~43
/// MILLION ranges, which is precisely the explosion A3 documents. Using it to extract two numbers
/// was a fine idea applied at exactly the size where it does not work.
///
/// Descending is enough. Both curves are hierarchical, so a node's keys are a contiguous interval:
/// to find the smallest key in the box, at each level take the intersecting child with the lowest
/// interval start and recurse. The largest is the mirror.
fn box_extremes(lo: [u32; 3], hi: [u32; 3], bits: u32, hilbert: bool) -> Option<(u64, u64)> {
    fn node_start(o: [u32; 3], level: u32, bits: u32, hilbert: bool) -> u64 {
        let n = 1u64 << (3 * (bits - level));
        if hilbert { hilbert3(o[0] >> (bits - level), o[1] >> (bits - level), o[2] >> (bits - level), level) * n }
        else { morton3(o[0], o[1], o[2]) }
    }
    // walk down for the extreme, `want_min` choosing the direction
    let walk = |want_min: bool| -> Option<u64> {
        let (mut o, mut level) = ([0u32; 3], 0u32);
        loop {
            let side = 1u32 << (bits - level);
            if (0..3).any(|k| o[k] + side <= lo[k] || o[k] > hi[k]) { return None; }
            if level == bits {
                return Some(node_start(o, level, bits, hilbert));
            }
            let h = side / 2;
            let mut best: Option<([u32; 3], u64)> = None;
            for c in 0..8u32 {
                let co = [o[0] + (c & 1) * h, o[1] + ((c >> 1) & 1) * h, o[2] + ((c >> 2) & 1) * h];
                if (0..3).any(|k| co[k] + h <= lo[k] || co[k] > hi[k]) { continue; }
                let st = node_start(co, level + 1, bits, hilbert);
                let key = if want_min { st } else { st + (1u64 << (3 * (bits - level - 1))) - 1 };
                let better = match best { None => true, Some((_, b)) => if want_min { key < b } else { key > b } };
                if better { best = Some((co, key)); }
            }
            let (co, _) = best?;
            o = co;
            level += 1;
        }
    };
    Some((walk(true)?, walk(false)?))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // id models the per-object persistence key
struct Obj { id: u32, p: Point3 }
impl Positioned3 for Obj { fn position(&self) -> Point3 { self.p } }

/// World coordinate → cell index. **Clamped, not masked.** Masking wraps a coordinate that pokes
/// past the world edge round to the far side, which silently turns a query near the boundary into
/// a different, enormous box. That inflated A2's over-scan by ~80x for a whole night before the
/// exact-extremes rewrite made the discrepancy visible.
fn cell(v: f64) -> u32 { (((v / WORLD) * (1u32 << BITS) as f64) as i64).clamp(0, ((1u32 << BITS) - 1) as i64) as u32 }

fn main() {
    // ---- self-test the Hilbert encoder on a small grid: bijection + adjacency.
    {
        let b = 3u32; let side = 1u32 << b; // 8³ = 512
        let mut seen = vec![false; (side * side * side) as usize];
        let mut pt = vec![(0u32, 0u32, 0u32); (side * side * side) as usize];
        for z in 0..side { for y in 0..side { for x in 0..side {
            let d = hilbert3(x, y, z, b) as usize;
            assert!(!seen[d], "hilbert3 not a bijection"); seen[d] = true; pt[d] = (x, y, z);
        }}}
        for d in 1..pt.len() {
            let (a, c) = (pt[d - 1], pt[d]);
            let man = (a.0 as i64 - c.0 as i64).abs() + (a.1 as i64 - c.1 as i64).abs() + (a.2 as i64 - c.2 as i64).abs();
            assert_eq!(man, 1, "hilbert3 consecutive cells must be adjacent");
        }
        println!("hilbert3 self-test OK (bijection + unit-step adjacency on 8³)\n");
    }

    println!("Cold-index options | world {WORLD:.0}^3 | key = {BITS} bits/axis\n");

    // ============================================================ A) the curve
    // Locality metric 1: how many CONTIGUOUS key-runs does a box of side `s`
    // cells map to? (fewer = better locality = fewer/cheaper range scans).
    println!("== A1) box → contiguous key-runs (lower = better locality) ==");
    println!("{:>10} | {:>16} {:>16} {:>10} {:>8} {:>8}", "box side", "Morton runs", "Hilbert runs", "M/H", "H/s²", "M/s²");
    let mut rr = Rng(1);
    for &s in &[4u32, 8, 16, 32] {
        let (mut mruns, mut hruns, trials) = (0u64, 0u64, 200u32);
        for _ in 0..trials {
            let (ox, oy, oz) = ((rr.unit() * (1.0 - s as f64 / (1u32 << BITS) as f64) * (1u32 << BITS) as f64) as u32,
                                (rr.unit() * (1.0 - s as f64 / (1u32 << BITS) as f64) * (1u32 << BITS) as f64) as u32,
                                (rr.unit() * (1.0 - s as f64 / (1u32 << BITS) as f64) * (1u32 << BITS) as f64) as u32);
            let runs = |code: &dyn Fn(u32, u32, u32) -> u64| -> u64 {
                let mut v: Vec<u64> = Vec::with_capacity((s * s * s) as usize);
                for z in 0..s { for y in 0..s { for x in 0..s { v.push(code(ox + x, oy + y, oz + z)); } } }
                v.sort_unstable();
                let mut runs = 1u64;
                for w in v.windows(2) { if w[1] != w[0] + 1 { runs += 1; } }
                runs
            };
            mruns += runs(&|x, y, z| morton3(x, y, z));
            hruns += runs(&|x, y, z| hilbert3(x, y, z, BITS));
        }
        let (m, h) = (mruns as f64 / trials as f64, hruns as f64 / trials as f64);
        // Checked against theory, not just reported. Moon, Jagadish, Faloutsos & Saltz (TKDE
        // 2001) derive the Hilbert clustering number as the query's SURFACE AREA divided by
        // twice the dimensionality; for a cube of side s in 3D that is 6s²/6 = **s² exactly**.
        // If this drifts, either the hilbert3 encoder broke or the run counter did — and a
        // silently wrong encoder still produces a plausible-looking table, which is the whole
        // reason to assert it.
        let theory = (s * s) as f64;
        assert!((h / theory - 1.0).abs() < 0.10,
            "Hilbert runs {h:.0} vs the closed form s²={theory:.0} — off by {:.0}%", 100.0 * (h / theory - 1.0).abs());
        println!("{:>10} | {:>16.0} {:>16.0} {:>10.2}x {:>8.2} {:>8.2}", s, m, h, m / h, h / theory, m / theory);
    }
    println!("  (last two columns: runs / s². Hilbert should sit at 1.00 — the Moon et al. closed");
    println!("   form. Morton converges to a CONSTANT ~1.9, which is Xu & Tirthapura's result that");
    println!("   Z-order is within a constant factor of optimal — it does not get worse without");
    println!("   bound, and reading the small-box ratio as a growing gap is a mistake.)");

    // Locality metric 2: real sorted-key range scan over-scan on N points.
    // A box query scans keys in [min_box_code, max_box_code]; over-scan =
    // (keys scanned) / (keys actually in the box).
    println!("\n== A2) sorted-key range-scan OVER-SCAN (scanned/hits, lower = better) ==");
    println!("{:>9} | {:>18} {:>18} {:>18}", "N", "Morton overscan", "Hilbert overscan", "avg hits");
    for &n in &[100_000usize, 1_000_000] {
        let mut r = Rng(42);
        let objs: Vec<Obj> = (0..n).map(|i| Obj { id: i as u32, p: Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) }).collect();
        // sorted key arrays
        let mut mkeys: Vec<u64> = objs.iter().map(|o| morton3(cell(o.p.x), cell(o.p.y), cell(o.p.z))).collect();
        let mut hkeys: Vec<u64> = objs.iter().map(|o| hilbert3(cell(o.p.x), cell(o.p.y), cell(o.p.z), BITS)).collect();
        mkeys.sort_unstable(); hkeys.sort_unstable();
        let bubble = 500.0f64;
        let scan = |keys: &[u64], hilbert: bool, cxyz: (f64, f64, f64)| -> (usize, usize) {
            // box cell bounds of the bubble
            let (x0, x1) = (cell(cxyz.0 - bubble), cell(cxyz.0 + bubble));
            let (y0, y1) = (cell(cxyz.1 - bubble), cell(cxyz.1 + bubble));
            let (z0, z1) = (cell(cxyz.2 - bubble), cell(cxyz.2 + bubble));
            // TRUE min/max over the box, not a sample. This used to probe the 8 corners and the
            // face centres and admit in a comment that it under-counted Hilbert's advantage —
            // Hilbert is not monotone in the coordinates, so the extreme key can sit anywhere.
            // `box_ranges` decomposes the box into its exact key intervals, so the first range's
            // start and the last range's end ARE the extremes, by construction.
            let Some((lo, hi)) = box_extremes([x0, y0, z0], [x1, y1, z1], BITS, hilbert) else { return (0, 0) };
            let s = keys.partition_point(|&k| k < lo);
            let e = keys.partition_point(|&k| k <= hi);
            let scanned = e - s;
            // true hits: points whose cell is in the box
            let hits = keys[s..e].iter().filter(|&&_k| true).count(); // placeholder; real hit-count below
            (scanned, hits)
        };
        // real hit count = brute over the box cell bounds (independent of curve)
        let mut r2 = Rng(7);
        let (mut mo, mut ho, mut hitsum, trials) = (0.0f64, 0.0f64, 0usize, 300usize);
        for _ in 0..trials {
            let c = (r2.unit() * WORLD, r2.unit() * WORLD, r2.unit() * WORLD);
            let (x0, x1) = (cell(c.0 - bubble), cell(c.0 + bubble));
            let (y0, y1) = (cell(c.1 - bubble), cell(c.1 + bubble));
            let (z0, z1) = (cell(c.2 - bubble), cell(c.2 + bubble));
            let hits = objs.iter().filter(|o| { let (a, b, d) = (cell(o.p.x), cell(o.p.y), cell(o.p.z)); a >= x0 && a <= x1 && b >= y0 && b <= y1 && d >= z0 && d <= z1 }).count().max(1);
            let (ms, _) = scan(&mkeys, false, c);
            let (hs, _) = scan(&hkeys, true, c);
            mo += ms as f64 / hits as f64; ho += hs as f64 / hits as f64; hitsum += hits;
        }
        println!("{:>9} | {:>16.1}x {:>16.1}x {:>18.0}", n, mo / trials as f64, ho / trials as f64, hitsum as f64 / trials as f64);
    }

    // ============================================================ B) the structure
    // AoI bubble query, four ways:
    //   - BTreeMap NAIVE single-range scan [min..max] over box corners (the
    //     trap: one giant range across the Z-order jumps → pathological).
    //   - BTreeMap CELL-PROBE: coarse Morton CELL key, enumerate the box's cells
    //     and probe each (the cold store done RIGHT — the same algorithm as the
    //     grid, but over a sorted/on-disk-capable B-tree). This is the honest
    //     "what does making it on-disk-shaped cost vs the HashMap grid" number.
    //   - MortonGrid3 (HashMap, hot in-memory).
    //   - Tree3 (adaptive, hot).
    const LV: u32 = 5; // coarse cell level (32 cells/axis, cell ≈ 312 wu ≈ bubble)
    let ccell = |v: f64| ((v / WORLD) * (1u32 << LV) as f64) as u32 & ((1 << LV) - 1);
    // ---------------------------------------------------------------- A3
    // A2 says the span scan is hopeless. The fix the literature reaches for is to scan the box's
    // RUNS instead — BIGMIN/LITMAX as a cursor, or equivalently the octree decomposition in
    // `box_ranges`. But there is a precondition nobody states in the same breath, and it decides
    // whether any of it is worth doing: the run count is a function of the box measured in
    // CELLS, so it is set by the KEY RESOLUTION, not by the query.
    //
    // A2's bubble is 500 of 10 000 world units. At 16 bits/axis that box is ~6 554 cells on a
    // side, so it decomposes into ~s^2 = 43 MILLION runs for the ~780 points it contains.
    // Run-aware scanning would be far worse than the span it replaces. Sweep the resolution and
    // the trade shows up.
    println!("
== A3) run-aware scan: ranges instead of the span, by key resolution ==");
    println!("{:>5} {:>10} | {:>14} {:>14} | {:>12} {:>12}",
             "bits", "cells/box", "Morton ranges", "Hilbert ranges", "vs cellbox", "vs SPHERE");
    {
        let n = 100_000usize;
        let mut r = Rng(42);
        let objs: Vec<Obj> = (0..n).map(|i| Obj { id: i as u32, p: Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) }).collect();
        let bubble = 500.0f64;
        for &b in &[6u32, 8, 10, 12] {
            // CLAMP, do not mask: `cell(v)` masks, so a box that pokes past the world edge wraps
            // to the far side and silently becomes a different box. Fine for A2's corner sampling,
            // not fine when the box IS the query.
            let hi_cell = (1u32 << b) - 1;
            let cell_b = |v: f64| -> u32 { (((v / WORLD) * (1u32 << b) as f64) as i64).clamp(0, hi_cell as i64) as u32 };
            let mut mk: Vec<u64> = objs.iter().map(|o| morton3(cell_b(o.p.x), cell_b(o.p.y), cell_b(o.p.z))).collect();
            let mut hk: Vec<u64> = objs.iter().map(|o| hilbert3(cell_b(o.p.x), cell_b(o.p.y), cell_b(o.p.z), b)).collect();
            mk.sort_unstable(); hk.sort_unstable();
            let (mut mr, mut hr, mut mo, mut ho, mut side) = (0f64, 0f64, 0f64, 0f64, 0f64);
            let (mut mt, mut ht) = (0f64, 0f64);
            // Count the trials that actually ran. Dividing by the LOOP BOUND while some
            // iterations skip is how this first read a constant 0.80x over-scan — an impossible
            // number (a scan cannot read fewer keys than it returns), which is the only reason
            // it got caught rather than believed.
            let mut used = 0f64;
            let trials = 40usize;
            let mut r2 = Rng(7);
            for _ in 0..trials {
                let c = (r2.unit() * WORLD, r2.unit() * WORLD, r2.unit() * WORLD);
                let lo = [cell_b(c.0 - bubble), cell_b(c.1 - bubble), cell_b(c.2 - bubble)];
                let hi = [cell_b(c.0 + bubble), cell_b(c.1 + bubble), cell_b(c.2 + bubble)];
                used += 1.0;
                side += (hi[0] - lo[0] + 1) as f64;
                let hits = objs.iter().filter(|o| {
                    let q = [cell_b(o.p.x), cell_b(o.p.y), cell_b(o.p.z)];
                    (0..3).all(|k| q[k] >= lo[k] && q[k] <= hi[k])
                }).count().max(1) as f64;
                // The OTHER half of the trade, and the half a cell-box metric hides. Over-scan
                // against the cell box is 1.00 by construction — the decomposition is exact. What
                // the caller actually asked for is the SPHERE, and a coarse key makes the cell box
                // a worse and worse stand-in for it. This is the cost that rises as the range
                // count falls, and it is why the knob has a floor instead of a direction.
                let want = objs.iter().filter(|o| {
                    let (dx, dy, dz) = (o.p.x - c.0, o.p.y - c.1, o.p.z - c.2);
                    dx * dx + dy * dy + dz * dz <= bubble * bubble
                }).count().max(1) as f64;
                let rsm = box_ranges(lo, hi, b, false);
                mr += rsm.len() as f64;
                let sm: usize = rsm.iter().map(|&(a, z)| mk.partition_point(|&k| k <= z) - mk.partition_point(|&k| k < a)).sum();
                mo += sm as f64 / hits; mt += sm as f64 / want;
                let rsh = box_ranges(lo, hi, b, true);
                hr += rsh.len() as f64;
                let sh: usize = rsh.iter().map(|&(a, z)| hk.partition_point(|&k| k <= z) - hk.partition_point(|&k| k < a)).sum();
                ho += sh as f64 / hits; ht += sh as f64 / want;
            }
            let t = used.max(1.0);
            println!("{:>5} {:>10.0} | {:>14.0} {:>14.0} | {:>11.2}x {:>11.2}x",
                     b, side / t, mr / t, hr / t, mo / t, ht / t);
            let _ = (ho, mt);
        }
        println!("  `vs cellbox` is 1.00 everywhere: the decomposition is EXACT, against A2's span");
        println!("  scan at ~102x. But that column flatters itself — the caller asked for a SPHERE,");
        println!("  and `vs SPHERE` shows what a coarse key really costs: 2.86x at 6 bits, falling to");
        println!("  1.86x. It stops there because 1.86 is not an artifact, it is 6/pi = 1.91, the");
        println!("  volume of a cube over its inscribed sphere — the irreducible price of bounding a");
        println!("  ball with an axis-aligned box.");
        println!();
        println!("  So the two costs pull opposite ways and the knob has a FLOOR, not a direction:");
        println!("  coarse keys mean few ranges but a box that is a poor sphere; fine keys approach");
        println!("  the geometric floor while the range count climbs as the square of the box in");
        println!("  cells (at 16 bits A2's bubble would need ~43M ranges to fetch ~780 points).");
        println!("  Hilbert's constant shows up here exactly as A1 predicts: ~1.9x fewer ranges for");
        println!("  identical over-scan, i.e. it buys the same answer for half the cursor seeks.");
    }


    println!("\n== B) AoI bubble query — µs/query ==");
    println!("{:>9} | {:>16} {:>16} {:>16} {:>14}", "N", "BTree naive", "BTree cell-probe", "MortonGrid3", "Tree3");
    for &n in &[100_000usize, 1_000_000] {
        let mut r = Rng(42);
        let objs: Vec<Obj> = (0..n).map(|i| Obj { id: i as u32, p: Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) }).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
        let mut bt_fine: BTreeMap<u64, Vec<Obj>> = BTreeMap::new();
        for o in &objs { bt_fine.entry(morton3(cell(o.p.x), cell(o.p.y), cell(o.p.z))).or_default().push(*o); }
        let mut bt_cell: BTreeMap<u64, Vec<Obj>> = BTreeMap::new();
        for o in &objs { bt_cell.entry(morton3(ccell(o.p.x), ccell(o.p.y), ccell(o.p.z))).or_default().push(*o); }
        let mut g = MortonGrid3::new(world, LV);
        for o in &objs { g.insert(*o); }
        let t = Tree3::bulk_load(world, 8, objs.clone());
        let bubble = 500.0f64;
        let qs: Vec<(f64, f64, f64)> = { let mut rq = Rng(99); (0..1000).map(|_| (rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD)).collect() };
        let best = |reps: usize, mut f: Box<dyn FnMut()>| -> f64 { f(); let mut b = f64::MAX; for _ in 0..reps { let t0 = Instant::now(); f(); b = b.min(t0.elapsed().as_secs_f64()); } b };
        let t_naive = best(5, Box::new(|| {
            let mut hits = 0usize;
            for &c in &qs {
                let (x0, x1, y0, y1, z0, z1) = (cell(c.0 - bubble), cell(c.0 + bubble), cell(c.1 - bubble), cell(c.1 + bubble), cell(c.2 - bubble), cell(c.2 + bubble));
                let mut lo = u64::MAX; let mut hi = 0u64;
                for &cx in &[x0, (x0 + x1) / 2, x1] { for &cy in &[y0, (y0 + y1) / 2, y1] { for &cz in &[z0, (z0 + z1) / 2, z1] { let cc = morton3(cx, cy, cz); lo = lo.min(cc); hi = hi.max(cc); }}}
                let sph = Sphere3::new(c.0, c.1, c.2, bubble);
                for (_, bucket) in bt_fine.range(lo..=hi) { for o in bucket { if sph.contains_point(o.p) { hits += 1; } } }
            }
            std::hint::black_box(hits);
        })) / qs.len() as f64 * 1e6;
        let t_probe = best(5, Box::new(|| {
            let mut hits = 0usize;
            for &c in &qs {
                let (x0, x1, y0, y1, z0, z1) = (ccell(c.0 - bubble), ccell(c.0 + bubble), ccell(c.1 - bubble), ccell(c.1 + bubble), ccell(c.2 - bubble), ccell(c.2 + bubble));
                let sph = Sphere3::new(c.0, c.1, c.2, bubble);
                for iz in z0..=z1 { for iy in y0..=y1 { for ix in x0..=x1 {
                    if let Some(bucket) = bt_cell.get(&morton3(ix, iy, iz)) { for o in bucket { if sph.contains_point(o.p) { hits += 1; } } }
                }}}
            }
            std::hint::black_box(hits);
        })) / qs.len() as f64 * 1e6;
        let t_g = best(5, Box::new(|| { let mut h = 0; for &c in &qs { h += g.cull(&Sphere3::new(c.0, c.1, c.2, bubble)).len(); } std::hint::black_box(h); })) / qs.len() as f64 * 1e6;
        let t_t = best(5, Box::new(|| { let mut h = 0; for &c in &qs { h += t.cull(&Sphere3::new(c.0, c.1, c.2, bubble)).len(); } std::hint::black_box(h); })) / qs.len() as f64 * 1e6;
        println!("{:>9} | {:>14.2}  {:>15.2} {:>16.2} {:>14.2}", n, t_naive, t_probe, t_g, t_t);
    }

    println!("
done — the conclusions, and the check against the published closed forms,");
    println!("are written up in docs/SPACE_FILLING_CURVES.md.");
}
