//! `voxel_select_bench` — how do you select every block inside a sphere, fastest?
//!
//! The kit answers sphere queries analytically (`classify_box` on node boxes, then an
//! exact per-point test) with an optional 1x1x1 [`VoxelRaster`](vectorial_hash::VoxelRaster)
//! for the leaf test. That is the right shape for *sparse points in space*. A Minecraft-like
//! world is the opposite problem — the data is a **dense uniform array**, positions are
//! integers, and the index is the array itself — so this measures what actually wins there.
//!
//! Four ways to answer "every solid block within r of this centre":
//!
//! 1. **naive** — triple loop over the bounding box, `dx²+dy²+dz² <= r²` per block.
//! 2. **bitmap** — precompute the sphere once as a `(2r+1)³` bit per radius, then look the
//!    membership up instead of computing it. This is the direct analogue of the kit's
//!    voxel raster.
//! 3. **spans** — precompute, per `(dy,dz)` row, the single **x-range** the sphere covers.
//!    A sphere's intersection with a row is always contiguous, so this is O(r²) entries
//!    instead of O(r³), and the inner loop becomes a straight run over adjacent memory
//!    with no test at all.
//! 4. **spans + chunk skip** — the same, but the world is 16³ chunks carrying a "uniform"
//!    flag, so a chunk that is entirely air is skipped whole and never touched.
//!
//! ```bash
//! cargo run -p vectorial-hash --example voxel_select_bench --release
//! ```
//! Env: `VS_SIDE` (world side in blocks), `VS_R` (radii, comma-separated), `VS_Q` (queries).

use std::collections::HashMap;

#[path = "common/mod.rs"]
mod common;

const CHUNK: usize = 16;

/// One selection strategy: world, sphere centre, radius -> how many solid blocks.
type Method<'a> = dyn Fn(&World, (i64, i64, i64), i64) -> usize + 'a;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 11 }
    fn range(&mut self, lo: usize, hi: usize) -> usize { lo + (self.next() as usize) % (hi - lo).max(1) }
}

/// A dense block world plus, per 16³ chunk, whether it is entirely air. Real voxel engines
/// carry exactly this (a palette of one entry), and it is what makes skipping legal.
struct World { side: usize, blocks: Vec<u8>, chunk_side: usize, chunk_empty: Vec<bool> }

impl World {
    /// Solid where a coarse value-noise-ish function says so: big connected regions with
    /// large empty volumes, which is what makes the chunk skip worth having.
    fn new(side: usize) -> World {
        let mut blocks = vec![0u8; side * side * side];
        let h = |x: usize, z: usize| {
            let (fx, fz) = (x as f64 * 0.05, z as f64 * 0.05);
            (side as f64 * 0.45) + 12.0 * (fx.sin() + fz.cos()) + 6.0 * (fx * 1.7).cos()
        };
        for z in 0..side { for x in 0..side {
            let top = h(x, z).clamp(1.0, side as f64 - 1.0) as usize;
            for y in 0..top { blocks[(z * side + y) * side + x] = 1; }
        }}
        let cs = side / CHUNK;
        let mut chunk_empty = vec![true; cs * cs * cs];
        for z in 0..side { for y in 0..side { for x in 0..side {
            if blocks[(z * side + y) * side + x] != 0 {
                chunk_empty[((z / CHUNK) * cs + (y / CHUNK)) * cs + (x / CHUNK)] = false;
            }
        }}}
        World { side, blocks, chunk_side: cs, chunk_empty }
    }
    #[inline] fn at(&self, x: usize, y: usize, z: usize) -> u8 { self.blocks[(z * self.side + y) * self.side + x] }
    #[inline] fn chunk_is_empty(&self, cx: usize, cy: usize, cz: usize) -> bool {
        self.chunk_empty[(cz * self.chunk_side + cy) * self.chunk_side + cx]
    }
    #[inline] fn clamp_box(&self, c: (i64, i64, i64), r: i64) -> (usize, usize, usize, usize, usize, usize) {
        let s = self.side as i64 - 1;
        ((c.0 - r).clamp(0, s) as usize, (c.0 + r).clamp(0, s) as usize,
         (c.1 - r).clamp(0, s) as usize, (c.1 + r).clamp(0, s) as usize,
         (c.2 - r).clamp(0, s) as usize, (c.2 + r).clamp(0, s) as usize)
    }
}

// ---------------------------------------------------------------- the four methods

fn naive(w: &World, c: (i64, i64, i64), r: i64) -> usize {
    let (x0, x1, y0, y1, z0, z1) = w.clamp_box(c, r);
    let r2 = r * r;
    let mut n = 0;
    for z in z0..=z1 { let dz = z as i64 - c.2;
        for y in y0..=y1 { let dy = y as i64 - c.1;
            for x in x0..=x1 { let dx = x as i64 - c.0;
                if dx * dx + dy * dy + dz * dz <= r2 && w.at(x, y, z) != 0 { n += 1; }
            }}}
    n
}

/// `(2r+1)³` membership bits — the kit's voxel-raster idea applied to a block world.
struct Bitmap { r: i64, bits: Vec<u64> }
impl Bitmap {
    fn new(r: i64) -> Bitmap {
        let d = (2 * r + 1) as usize;
        let mut bits = vec![0u64; d * d * d / 64 + 1];
        let r2 = r * r;
        for dz in -r..=r { for dy in -r..=r { for dx in -r..=r {
            if dx * dx + dy * dy + dz * dz <= r2 {
                let i = (((dz + r) as usize * d) + (dy + r) as usize) * d + (dx + r) as usize;
                bits[i / 64] |= 1 << (i % 64);
            }
        }}}
        Bitmap { r, bits }
    }
    #[inline] fn has(&self, dx: i64, dy: i64, dz: i64) -> bool {
        let d = (2 * self.r + 1) as usize;
        let i = (((dz + self.r) as usize * d) + (dy + self.r) as usize) * d + (dx + self.r) as usize;
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }
}

fn bitmap(w: &World, c: (i64, i64, i64), r: i64, bm: &Bitmap) -> usize {
    let (x0, x1, y0, y1, z0, z1) = w.clamp_box(c, r);
    let mut n = 0;
    for z in z0..=z1 { let dz = z as i64 - c.2;
        for y in y0..=y1 { let dy = y as i64 - c.1;
            for x in x0..=x1 { let dx = x as i64 - c.0;
                if bm.has(dx, dy, dz) && w.at(x, y, z) != 0 { n += 1; }
            }}}
    n
}

/// Per `(dy,dz)` row, the half-width the sphere covers: the row is `dx ∈ [-hw, hw]`, one
/// contiguous run. O(r²) entries, and the inner loop has no membership test at all.
struct Spans { r: i64, half: Vec<i32> }
impl Spans {
    fn new(r: i64) -> Spans {
        let d = (2 * r + 1) as usize;
        let mut half = vec![-1i32; d * d];
        let r2 = r * r;
        for dz in -r..=r { for dy in -r..=r {
            let rem = r2 - dy * dy - dz * dz;
            if rem >= 0 { half[((dz + r) as usize) * d + (dy + r) as usize] = (rem as f64).sqrt() as i32; }
        }}
        Spans { r, half }
    }
    #[inline] fn half_width(&self, dy: i64, dz: i64) -> i32 {
        let d = (2 * self.r + 1) as usize;
        self.half[((dz + self.r) as usize) * d + (dy + self.r) as usize]
    }
}

fn spans(w: &World, c: (i64, i64, i64), r: i64, sp: &Spans) -> usize {
    let s = w.side as i64 - 1;
    let mut n = 0;
    for dz in -r..=r { let z = c.2 + dz; if z < 0 || z > s { continue; }
        for dy in -r..=r { let y = c.1 + dy; if y < 0 || y > s { continue; }
            let hw = sp.half_width(dy, dz); if hw < 0 { continue; }
            let x0 = (c.0 - hw as i64).clamp(0, s) as usize;
            let x1 = (c.0 + hw as i64).clamp(0, s) as usize;
            let row = (z as usize * w.side + y as usize) * w.side;
            // A straight run over adjacent memory. No distance test, no bitmap lookup.
            for x in x0..=x1 { if w.blocks[row + x] != 0 { n += 1; } }
        }}
    n
}

/// Same spans, but a row that lies entirely inside an all-air chunk is skipped whole.
fn spans_skip(w: &World, c: (i64, i64, i64), r: i64, sp: &Spans) -> usize {
    let s = w.side as i64 - 1;
    let mut n = 0;
    for dz in -r..=r { let z = c.2 + dz; if z < 0 || z > s { continue; }
        for dy in -r..=r { let y = c.1 + dy; if y < 0 || y > s { continue; }
            let hw = sp.half_width(dy, dz); if hw < 0 { continue; }
            let x0 = (c.0 - hw as i64).clamp(0, s) as usize;
            let x1 = (c.0 + hw as i64).clamp(0, s) as usize;
            let (cy, cz) = (y as usize / CHUNK, z as usize / CHUNK);
            let row = (z as usize * w.side + y as usize) * w.side;
            let mut x = x0;
            while x <= x1 {
                let cx = x / CHUNK;
                let chunk_end = ((cx + 1) * CHUNK - 1).min(x1);
                if w.chunk_is_empty(cx, cy, cz) { x = chunk_end + 1; continue; } // whole chunk is air
                for xx in x..=chunk_end { if w.blocks[row + xx] != 0 { n += 1; } }
                x = chunk_end + 1;
            }
        }}
    n
}

fn main() {
    let side: usize = std::env::var("VS_SIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let nq: usize = std::env::var("VS_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let radii: Vec<i64> = std::env::var("VS_R").unwrap_or_else(|_| "4,8,16,32".into())
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();

    let w = World::new(side);
    let solid = w.blocks.iter().filter(|b| **b != 0).count();
    let empty_chunks = w.chunk_empty.iter().filter(|e| **e).count();
    println!("voxel block selection | world {side}³ = {} blocks ({:.0}% solid) | {} of {} chunks all-air | {nq} queries per radius\n",
        w.blocks.len(), 100.0 * solid as f64 / w.blocks.len() as f64, empty_chunks, w.chunk_empty.len());

    let mut r = Lcg(0x5EED);
    let centres: Vec<(i64, i64, i64)> = (0..nq).map(|_| (r.range(0, side) as i64, r.range(0, side) as i64, r.range(0, side) as i64)).collect();

    let mut bitmaps: HashMap<i64, Bitmap> = HashMap::new();
    let mut spanmaps: HashMap<i64, Spans> = HashMap::new();
    for &rad in &radii { bitmaps.insert(rad, Bitmap::new(rad)); spanmaps.insert(rad, Spans::new(rad)); }

    println!("  {:>6} {:>12} {:>12} {:>12} {:>12}   {:>10}", "radius", "naive us", "bitmap us", "spans us", "spans+skip", "vs naive");
    for &rad in &radii {
        let bm = &bitmaps[&rad];
        let sp = &spanmaps[&rad];
        // Every method must return the same count, or the comparison is meaningless.
        let (a, b, c, d): (usize, usize, usize, usize) = centres.iter().fold((0, 0, 0, 0), |acc, &ct| {
            (acc.0 + naive(&w, ct, rad), acc.1 + bitmap(&w, ct, rad, bm), acc.2 + spans(&w, ct, rad, sp), acc.3 + spans_skip(&w, ct, rad, sp))
        });
        assert!(a == b && b == c && c == d, "methods disagree at r={rad}: {a} {b} {c} {d}");

        let per = |f: &Method| {
            common::measure(5, || { let mut acc = 0; for &ct in &centres { acc += f(&w, ct, rad); } std::hint::black_box(acc); }).ms * 1e3 / nq as f64
        };
        let t_naive = per(&|w, c, r| naive(w, c, r));
        let t_bm = per(&|w, c, r| bitmap(w, c, r, bm));
        let t_sp = per(&|w, c, r| spans(w, c, r, sp));
        let t_sk = per(&|w, c, r| spans_skip(w, c, r, sp));
        println!("  {:>6} {:>12.2} {:>12.2} {:>12.2} {:>12.2}   spans {:.2}x, +skip {:.2}x",
            rad, t_naive, t_bm, t_sp, t_sk, t_naive / t_sp, t_naive / t_sk);
        println!("#M r{rad}.naive {t_naive:.4} us");
        println!("#M r{rad}.bitmap {t_bm:.4} us");
        println!("#M r{rad}.spans {t_sp:.4} us");
        println!("#M r{rad}.spans_skip {t_sk:.4} us");
        println!("#M r{rad}.spans_speedup {:.3} x", t_naive / t_sp);
    }

    println!("\nreading: the bitmap is the kit's voxel-raster idea in a block world — it replaces");
    println!("arithmetic with a lookup, which is a bad trade when the arithmetic is three multiplies");
    println!("and the lookup is a cache miss. Spans delete the per-block question entirely: a sphere");
    println!("meets each row in ONE contiguous run, so the table is O(r²) and the inner loop is a");
    println!("straight walk over adjacent memory. Chunk skipping then removes the blocks you were");
    println!("never going to want.");
}
