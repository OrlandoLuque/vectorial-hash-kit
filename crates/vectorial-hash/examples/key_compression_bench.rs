//! Key-compression measurement — how much do sorted Morton keys shrink with
//! delta + Frame-of-Reference bit-packing (the Lucene-BKD trick), and how cheap
//! is the decode? Sorted space-filling-curve keys share long prefixes, so
//! adjacent keys differ by small deltas → few bits each.
//!
//! Encodes N sorted 64-bit Morton keys as: one anchor key + per-block
//! (bit-width byte + bit-packed deltas). Reports the compressed size, ratio,
//! and decode ns/key (verified lossless).
//!
//! ```bash
//! cargo run -p vectorial-hash --example key_compression_bench --release
//! ```

use std::time::Instant;

const WORLD: f64 = 10_000.0;
const BLOCK: usize = 128; // ≤ 255 so the per-block count fits one byte

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}
fn cell(v: f64, bits: u32) -> u32 { ((v / WORLD) * (1u32 << bits) as f64) as u32 & ((1 << bits) - 1) }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

/// Delta + FOR bit-pack the sorted keys → bytes (u128 accumulator so a
/// full-64-bit delta plus the <8-bit residue never overflows).
fn encode(keys: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(keys.len());
    out.extend_from_slice(&keys[0].to_le_bytes()); // anchor
    let mut prev = keys[0];
    for block in keys[1..].chunks(BLOCK) {
        let mut deltas = [0u64; BLOCK];
        let mut maxd = 0u64;
        let mut p = prev;
        for (i, &k) in block.iter().enumerate() { let d = k - p; deltas[i] = d; maxd = maxd.max(d); p = k; }
        prev = p;
        let w = if maxd == 0 { 0 } else { 64 - maxd.leading_zeros() } as u8; // bits per delta
        out.push(w);
        out.push(block.len() as u8);
        let (mut acc, mut nbits): (u128, u32) = (0, 0);
        for &d in &deltas[..block.len()] {
            acc |= (d as u128) << nbits; nbits += w as u32;
            while nbits >= 8 { out.push((acc & 0xff) as u8); acc >>= 8; nbits -= 8; }
        }
        if nbits > 0 { out.push((acc & 0xff) as u8); }
    }
    out
}

/// Decode back to the key list (for verification + timing).
fn decode(bytes: &[u8], n: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    let mut prev = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    out.push(prev);
    let mut pos = 8;
    while out.len() < n {
        let w = bytes[pos] as u32; pos += 1;
        let cnt = bytes[pos] as usize; pos += 1;
        let (mut acc, mut nbits): (u128, u32) = (0, 0);
        let mask: u128 = if w == 0 { 0 } else { (1u128 << w) - 1 };
        for _ in 0..cnt {
            while nbits < w { acc |= (bytes[pos] as u128) << nbits; pos += 1; nbits += 8; }
            let d = (acc & mask) as u64; acc >>= w; nbits -= w;
            prev += d; out.push(prev);
        }
    }
    out
}

fn gen_keys(n: usize, bits: u32, clustered: bool) -> Vec<u64> {
    let mut r = Rng(42);
    let mut keys: Vec<u64> = (0..n).map(|_| {
        let p = if clustered {
            // 64 tight clusters → keys dense in key-space → small deltas (realistic)
            let (cx, cy, cz) = ((r.next() % 8) as f64, (r.next() % 8) as f64, (r.next() % 8) as f64);
            let s = WORLD / 8.0;
            ((cx + r.unit() * 0.08) * s, (cy + r.unit() * 0.08) * s, (cz + r.unit() * 0.08) * s)
        } else {
            (r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) // uniform (worst case)
        };
        morton3(cell(p.0, bits), cell(p.1, bits), cell(p.2, bits))
    }).collect();
    keys.sort_unstable();
    keys
}

fn main() {
    println!("sorted Morton key compression (delta + FOR bit-pack, block {BLOCK})\n");
    println!("compression tracks density in KEY-space: dense/clustered → small deltas → few bits.\n");
    println!("{:>11} | {:>9} | {:>4} | {:>15} {:>13} {:>8} | {:>10}", "distribution", "N", "bits", "raw KB (8B/key)", "packed KB", "ratio", "dec ns/key");
    for &(label, clustered) in &[("uniform (worst)", false), ("clustered (real)", true)] {
        for &n in &[100_000usize, 1_000_000] {
            for &bits in &[12u32, 16, 21] {
                let keys = gen_keys(n, bits, clustered);
                let packed = encode(&keys);
                assert_eq!(decode(&packed, n), keys, "compression must be lossless");
                let raw = n * 8;
                let t = Instant::now();
                for _ in 0..5 { std::hint::black_box(decode(&packed, n)); }
                let dec_ns = t.elapsed().as_secs_f64() / 5.0 / n as f64 * 1e9;
                println!("{:>11} | {:>9} | {:>4} | {:>12.1}    {:>10.1}    {:>6.2}x | {:>10.1}",
                    label, n, bits, raw as f64 / 1024.0, packed.len() as f64 / 1024.0, raw as f64 / packed.len() as f64, dec_ns);
            }
        }
    }
    println!("\nreading: sorted Morton keys compress by the entropy of their deltas — coarser\nkeys (fewer bits) and denser/clustered data (a real world, not uniform noise)\nshrink the deltas → more compression, at ~a few int ops per key to decode.\nUniform-random-in-a-huge-space is the WORST case; real clustered worlds do much\nbetter. This shrinks the KEY bytes of an on-disk cold store (payload separate).");
}
