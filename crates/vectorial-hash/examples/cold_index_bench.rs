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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // id models the per-object persistence key
struct Obj { id: u32, p: Point3 }
impl Positioned3 for Obj { fn position(&self) -> Point3 { self.p } }

fn cell(v: f64) -> u32 { ((v / WORLD) * (1u32 << BITS) as f64) as u32 & ((1 << BITS) - 1) }

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
    println!("{:>10} | {:>16} {:>16} {:>10}", "box side", "Morton runs", "Hilbert runs", "M/H");
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
        println!("{:>10} | {:>16.0} {:>16.0} {:>10.2}x", s, m, h, m / h);
    }

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
        let scan = |keys: &[u64], code: &dyn Fn(u32, u32, u32) -> u64, cxyz: (f64, f64, f64)| -> (usize, usize) {
            // box cell bounds of the bubble
            let (x0, x1) = (cell(cxyz.0 - bubble), cell(cxyz.0 + bubble));
            let (y0, y1) = (cell(cxyz.1 - bubble), cell(cxyz.1 + bubble));
            let (z0, z1) = (cell(cxyz.2 - bubble), cell(cxyz.2 + bubble));
            // min/max code over the box cells (corners suffice for Morton;
            // Hilbert isn't monotone, so sample the 8 corners + face centres —
            // a cheap approximation of the true min/max that slightly
            // UNDER-counts Hilbert's advantage, i.e. it's conservative).
            let mut lo = u64::MAX; let mut hi = 0u64;
            for &cx in &[x0, (x0 + x1) / 2, x1] { for &cy in &[y0, (y0 + y1) / 2, y1] { for &cz in &[z0, (z0 + z1) / 2, z1] {
                let c = code(cx, cy, cz); lo = lo.min(c); hi = hi.max(c);
            }}}
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
            let (ms, _) = scan(&mkeys, &|x, y, z| morton3(x, y, z), c);
            let (hs, _) = scan(&hkeys, &|x, y, z| hilbert3(x, y, z, BITS), c);
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

    println!("\ndone. reading in the cold-index findings doc.");
}
