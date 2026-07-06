//! Layered cold store on a SPARSE sandbox — does the cold index need the same
//! adaptivity as the hot side? A huge world with big empty regions.
//!
//! The fixed-level cell-probe cold store (what the first prototype did) probes
//! every fine cell a query box covers — even over the void, where they're all
//! empty. A LAYERED store adds a coarse **occupancy tier** (which coarse cells
//! hold anything): the query checks coarse cells first and only descends into
//! fine probes inside occupied ones, skipping voids in O(1). The Morton key is
//! already hierarchical (prefix = level), so the tier is nearly free.
//!
//! We sweep the populated fraction of the world (dense → very sparse) and a
//! big "load this region" query box, and compare cells-probed + µs/query.
//! Both approaches return the SAME hit set (asserted).
//!
//! ```bash
//! cargo run -p vectorial-hash --example cold_layered_bench --release
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use vectorial_hash::{Point3, Shape3, Sphere3};

const WORLD: f64 = 100_000.0; // a BIG sandbox
const LF: u32 = 9;  // fine level: 512 cells/axis, cell ≈ 195 wu
const LC: u32 = 6;  // coarse tier: 64 cells/axis, cell ≈ 1562 wu (ratio 8/axis)

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}
fn cellf(v: f64, lv: u32) -> u32 { let n = (1u32 << lv) as f64; ((v / WORLD * n) as i64).clamp(0, (1i64 << lv) - 1) as u32 }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn main() {
    let n = 1_000_000usize;
    let half = 3_000.0f64; // "load this region" query half-size (box 6000 wide)
    let ratio = 1u32 << (LF - LC); // fine cells per coarse cell, per axis
    println!("layered vs fixed-level cold store | world {WORLD:.0}^3 | {n} objects | region query ±{half:.0}");
    println!("fine level {LF} (cell {:.0}), coarse tier {LC} (cell {:.0}, {ratio}^3 fine/coarse)\n", WORLD / (1u32 << LF) as f64, WORLD / (1u32 << LC) as f64);
    println!("{:>12} | {:>26} | {:>26} | {:>8}", "populated", "fixed-fine (probes, µs)", "layered (probes, µs)", "speedup");

    for &frac in &[1.0f64, 0.3, 0.1, 0.03] {
        // objects uniformly inside a corner cube of side frac*WORLD → the rest
        // of the sandbox is empty (sparser as frac shrinks).
        let span = WORLD * frac.cbrt(); // cbrt so the OCCUPIED VOLUME fraction ≈ frac
        let mut r = Rng(42);
        // fine store + coarse occupancy tier
        let mut fine: HashMap<u64, Vec<Point3>> = HashMap::new();
        let mut occ: HashSet<u64> = HashSet::new();
        for _ in 0..n {
            let p = Point3::new(r.unit() * span, r.unit() * span, r.unit() * span);
            fine.entry(morton3(cellf(p.x, LF), cellf(p.y, LF), cellf(p.z, LF))).or_default().push(p);
            occ.insert(morton3(cellf(p.x, LC), cellf(p.y, LC), cellf(p.z, LC)));
        }
        // queries at random positions across the FULL world (most over the void
        // as frac shrinks). Report per-query cells probed + time.
        let mut rq = Rng(99);
        let queries: Vec<(f64, f64, f64)> = (0..300).map(|_| (rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD)).collect();

        // --- fixed-fine: enumerate every fine cell in the box, probe each.
        let fixed = |qs: &[(f64, f64, f64)]| -> (u64, usize) {
            let (mut probes, mut hits) = (0u64, 0usize);
            for &c in qs {
                let (x0, x1) = (cellf(c.0 - half, LF), cellf(c.0 + half, LF));
                let (y0, y1) = (cellf(c.1 - half, LF), cellf(c.1 + half, LF));
                let (z0, z1) = (cellf(c.2 - half, LF), cellf(c.2 + half, LF));
                let sph = Sphere3::new(c.0, c.1, c.2, half); // box≈sphere here; a real query filters exactly
                for iz in z0..=z1 { for iy in y0..=y1 { for ix in x0..=x1 {
                    probes += 1;
                    if let Some(v) = fine.get(&morton3(ix, iy, iz)) { for &p in v { if sph.contains_point(p) { hits += 1; } } }
                }}}
            }
            (probes, hits)
        };
        // --- layered: coarse cells first, descend only into occupied ones.
        let layered = |qs: &[(f64, f64, f64)]| -> (u64, usize) {
            let (mut probes, mut hits) = (0u64, 0usize);
            for &c in qs {
                let (cx0, cx1) = (cellf(c.0 - half, LC), cellf(c.0 + half, LC));
                let (cy0, cy1) = (cellf(c.1 - half, LC), cellf(c.1 + half, LC));
                let (cz0, cz1) = (cellf(c.2 - half, LC), cellf(c.2 + half, LC));
                let (fx0, fx1) = (cellf(c.0 - half, LF), cellf(c.0 + half, LF));
                let (fy0, fy1) = (cellf(c.1 - half, LF), cellf(c.1 + half, LF));
                let (fz0, fz1) = (cellf(c.2 - half, LF), cellf(c.2 + half, LF));
                let sph = Sphere3::new(c.0, c.1, c.2, half);
                for ccz in cz0..=cz1 { for ccy in cy0..=cy1 { for ccx in cx0..=cx1 {
                    probes += 1; // one O(1) coarse-tier check
                    if !occ.contains(&morton3(ccx, ccy, ccz)) { continue; } // skip the void
                    // fine sub-cells of this coarse cell, intersected with the query box
                    let (sx0, sx1) = ((ccx * ratio).max(fx0), (ccx * ratio + ratio - 1).min(fx1));
                    let (sy0, sy1) = ((ccy * ratio).max(fy0), (ccy * ratio + ratio - 1).min(fy1));
                    let (sz0, sz1) = ((ccz * ratio).max(fz0), (ccz * ratio + ratio - 1).min(fz1));
                    for iz in sz0..=sz1 { for iy in sy0..=sy1 { for ix in sx0..=sx1 {
                        probes += 1;
                        if let Some(v) = fine.get(&morton3(ix, iy, iz)) { for &p in v { if sph.contains_point(p) { hits += 1; } } }
                    }}}
                }}}
            }
            (probes, hits)
        };

        // correctness: identical hit sets
        let (fp, fh) = fixed(&queries);
        let (lp, lh) = layered(&queries);
        assert_eq!(fh, lh, "layered must return the same hits as fixed-fine");

        let best = |mut f: Box<dyn FnMut()>| -> f64 { f(); let mut b = f64::MAX; for _ in 0..5 { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b };
        let qs2 = queries.clone();
        let t_fixed = best(Box::new(|| { std::hint::black_box(fixed(&qs2)); })) / queries.len() as f64 * 1e6;
        let qs3 = queries.clone();
        let t_lay = best(Box::new(|| { std::hint::black_box(layered(&qs3)); })) / queries.len() as f64 * 1e6;
        let (pf, pl) = (fp as f64 / queries.len() as f64, lp as f64 / queries.len() as f64);
        let tag = if frac >= 1.0 { "dense" } else if frac <= 0.03 { "sparse" } else { "" };
        println!("{:>5.0}% {:<7} | {:>13.0} probes {:>7.1} | {:>13.0} probes {:>7.1} | {:>6.1}x",
            frac * 100.0, tag, pf, t_fixed, pl, t_lay, t_fixed / t_lay.max(1e-9));
    }
    println!("\nreading: as the sandbox gets sparser, the coarse occupancy tier skips the void\nin O(1), so the layered store probes far fewer cells — the fixed level wastes a\nprobe on every empty fine cell. Symmetric with the hot adaptive tree.");
}
