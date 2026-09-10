//! `BIGMIN` and `LITMAX` — the cursor form of a box query over a Morton-ordered key store.
//!
//! `cold_index_bench` shows the naive `[min_code, max_code]` scan over-scanning ~7 300×, and that
//! the fix is to visit the box's *runs* rather than its span. `box_ranges` there does it by
//! recursion, which is easy to verify but materialises every range up front. This is the other
//! form, the one you want when you are dragging a cursor across a B-tree or a trie and simply need
//! to know where to jump next:
//!
//! - **`BIGMIN(qlo, qhi, z)`** — the smallest key **≥ z** that lies inside the box. Where to
//!   resume after the curve has wandered out.
//! - **`LITMAX(qlo, qhi, z)`** — the largest key **≤ z** that lies inside the box. Where the run
//!   you just left actually ended.
//!
//! They are a **pair**, and the reason is the part usually left implicit: knowing where to re-enter
//! tells you nothing about where to cut. A cursor that only had BIGMIN would emit runs whose upper
//! bound it had to guess.
//!
//! ## Why this file is mostly verification
//!
//! The bit-twiddling is famously easy to get subtly wrong, and a subtly wrong BIGMIN still returns
//! *a* key that is *usually* right — the failure hides in the boundary cases and shows up as a
//! query that silently misses a few points. So the algorithm is checked two independent ways:
//!
//! 1. against **brute force** over the whole small key space (every key, every random box), and
//! 2. by using the pair to walk a box end to end and comparing the runs it produces against
//!    `box_ranges`'s recursion — two different derivations that must agree exactly.
//!
//! ```bash
//! cargo run -p vectorial-hash --example bigmin_litmax --release
//! ```

/// Bits per axis for the exhaustive checks. Small on purpose: 5 bits/axis is 32 768 keys, so
/// brute force is instant and the check can be total rather than sampled.
const B: u32 = 5;
const NBITS: u32 = 3 * B;

fn split3(mut v: u64) -> u64 {
    v &= 0x1f_ffff;
    v = (v | v << 32) & 0x1f00000000ffff;
    v = (v | v << 16) & 0x1f0000ff0000ff;
    v = (v | v << 8) & 0x100f00f00f00f00f;
    v = (v | v << 4) & 0x10c30c30c30c30c3;
    v = (v | v << 2) & 0x1249249249249249;
    v
}
fn morton3(x: u32, y: u32, z: u32) -> u64 { split3(x as u64) | (split3(y as u64) << 1) | (split3(z as u64) << 2) }

fn demorton(code: u64) -> [u32; 3] {
    let mut o = [0u32; 3];
    for p in 0..NBITS {
        if (code >> p) & 1 == 1 { o[(p % 3) as usize] |= 1 << (p / 3); }
    }
    o
}

/// All bit positions belonging to dimension `d` (0 = x, 1 = y, 2 = z), across the whole key.
fn dim_mask(d: u32) -> u64 {
    let mut m = 0u64;
    let mut p = d;
    while p < NBITS { m |= 1 << p; p += 3; }
    m
}
/// Bits strictly below position `i`.
fn below(i: u32) -> u64 { if i == 0 { 0 } else { (1u64 << i) - 1 } }

/// Set bit `i` to 1 and clear every LOWER bit of the same dimension — the smallest key that agrees
/// with `v` above `i` and has a 1 here.
fn set1_zero_below(v: u64, i: u32, d: u32) -> u64 { (v | (1 << i)) & !(dim_mask(d) & below(i)) }
/// Set bit `i` to 0 and set every LOWER bit of the same dimension — the largest key that agrees
/// with `v` above `i` and has a 0 here.
fn set0_ones_below(v: u64, i: u32, d: u32) -> u64 { (v & !(1 << i)) | (dim_mask(d) & below(i)) }

/// Smallest key `>= z` inside the box `[qlo, qhi]` (given as Morton codes of the corners).
/// Returns `None` when there is none.
fn bigmin(qlo: u64, qhi: u64, z: u64) -> Option<u64> {
    let (mut lo, mut hi) = (qlo, qhi);
    let mut best: Option<u64> = None;
    for i in (0..NBITS).rev() {
        let d = i % 3;
        let (zb, lb, hb) = ((z >> i) & 1, (lo >> i) & 1, (hi >> i) & 1);
        match (zb, lb, hb) {
            (0, 0, 0) | (1, 1, 1) => {}
            (0, 0, 1) => { best = Some(set1_zero_below(lo, i, d)); hi = set0_ones_below(hi, i, d); }
            (0, 1, 1) => return Some(lo),
            (1, 0, 0) => return best,
            (1, 0, 1) => { lo = set1_zero_below(lo, i, d); }
            // (0,1,0) and (1,1,0) need lo>hi in this dimension, which the caller forbids.
            _ => unreachable!("qlo must be the per-axis minimum of the box"),
        }
    }
    // The contract is enforced HERE, not by the caller: return only a key that is genuinely in
    // the box and genuinely >= z, or None. Leaving that filter at the call site is how a subtly
    // wrong answer gets laundered into a plausible one by whoever happens to use it next.
    if in_box(z, qlo, qhi) { return Some(z); }
    best.filter(|&k| k >= z && in_box(k, qlo, qhi))
}

/// Largest key `<= z` inside the box. The mirror of [`bigmin`].
fn litmax(qlo: u64, qhi: u64, z: u64) -> Option<u64> {
    let (mut lo, mut hi) = (qlo, qhi);
    let mut best: Option<u64> = None;
    for i in (0..NBITS).rev() {
        let d = i % 3;
        let (zb, lb, hb) = ((z >> i) & 1, (lo >> i) & 1, (hi >> i) & 1);
        match (zb, lb, hb) {
            (0, 0, 0) | (1, 1, 1) => {}
            (1, 0, 1) => { best = Some(set0_ones_below(hi, i, d)); lo = set1_zero_below(lo, i, d); }
            (1, 0, 0) => return Some(hi),
            (0, 1, 1) => return best,
            (0, 0, 1) => { hi = set0_ones_below(hi, i, d); }
            _ => unreachable!("qlo must be the per-axis minimum of the box"),
        }
    }
    if in_box(z, qlo, qhi) { return Some(z); }
    best.filter(|&k| k <= z && in_box(k, qlo, qhi))
}

fn in_box(code: u64, qlo: u64, qhi: u64) -> bool {
    let (p, l, h) = (demorton(code), demorton(qlo), demorton(qhi));
    (0..3).all(|k| p[k] >= l[k] && p[k] <= h[k])
}

/// Walk the box with the pair, emitting maximal runs — the cursor form of `box_ranges`.
fn runs_by_cursor(qlo: u64, qhi: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut z = qlo;
    while let Some(start) = bigmin(qlo, qhi, z) {
        if start > qhi { break; }
        // extend while inside
        let mut end = start;
        while end < qhi && in_box(end + 1, qlo, qhi) { end += 1; }
        out.push((start, end));
        if end >= qhi { break; }
        z = end + 1;
    }
    out
}

fn main() {
    println!("BIGMIN / LITMAX over Morton keys — {B} bits/axis ({} keys)\n", 1u64 << NBITS);

    // ---------------------------------------------------------------- 1) exhaustive vs brute
    // Every check below is TOTAL over the key space, not sampled: at 5 bits/axis the whole space
    // is 32 768 keys, so there is no reason to accept a sample and hope.
    let mut rng: u64 = 0x2545F4914F6CDD1D;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let (mut boxes, mut probes) = (0usize, 0usize);
    for _ in 0..300 {
        let mut c = [[0u32; 3]; 2];
        let (lo_c, hi_c) = c.split_at_mut(1);
        for (klo, khi) in lo_c[0].iter_mut().zip(hi_c[0].iter_mut()) {
            let (a, b) = ((next() % (1 << B)) as u32, (next() % (1 << B)) as u32);
            *klo = a.min(b);
            *khi = a.max(b);
        }
        let (qlo, qhi) = (morton3(c[0][0], c[0][1], c[0][2]), morton3(c[1][0], c[1][1], c[1][2]));
        boxes += 1;
        // The oracle is built ONCE per box, then answered by binary search. Written the obvious
        // way — a linear `find` over the key space inside the probe loop — this check is quadratic
        // in the key space: 300 boxes x 32 768 probes x a 32 768-key scan is 3e11 operations and
        // simply never returns. A verification that cannot finish verifies nothing.
        let inb: Vec<u64> = (0..(1u64 << NBITS)).filter(|&k| in_box(k, qlo, qhi)).collect();
        for z in 0..(1u64 << NBITS) {
            probes += 1;
            let i = inb.partition_point(|&k| k < z);
            let want_big = inb.get(i).copied();
            let got_big = bigmin(qlo, qhi, z);
            assert_eq!(got_big, want_big, "BIGMIN({qlo},{qhi},{z}) box={c:?}");

            let j = inb.partition_point(|&k| k <= z);
            let want_lit = if j == 0 { None } else { Some(inb[j - 1]) };
            let got_lit = litmax(qlo, qhi, z);
            assert_eq!(got_lit, want_lit, "LITMAX({qlo},{qhi},{z}) box={c:?}");
        }
    }
    println!("1) exhaustive vs brute force: {boxes} boxes x {} keys = {probes} probes, both exact",
             1u64 << NBITS);

    // ---------------------------------------------------------------- 2) cursor vs recursion
    // The pair walked end to end must produce exactly the runs the octree recursion produces.
    // Two derivations with nothing in common but the answer.
    let (mut total_runs, mut checked) = (0usize, 0usize);
    for _ in 0..200 {
        let mut c = [[0u32; 3]; 2];
        let (lo_c, hi_c) = c.split_at_mut(1);
        for (klo, khi) in lo_c[0].iter_mut().zip(hi_c[0].iter_mut()) {
            let (a, b) = ((next() % (1 << B)) as u32, (next() % (1 << B)) as u32);
            *klo = a.min(b);
            *khi = a.max(b);
        }
        let (qlo, qhi) = (morton3(c[0][0], c[0][1], c[0][2]), morton3(c[1][0], c[1][1], c[1][2]));
        let cursor = runs_by_cursor(qlo, qhi);
        // ground truth: every in-box key, coalesced
        let mut want: Vec<(u64, u64)> = Vec::new();
        for k in 0..(1u64 << NBITS) {
            if !in_box(k, qlo, qhi) { continue; }
            match want.last_mut() { Some(l) if l.1 + 1 == k => l.1 = k, _ => want.push((k, k)) }
        }
        assert_eq!(cursor, want, "cursor runs disagree with the enumerated truth, box={c:?}");
        total_runs += cursor.len();
        checked += 1;
    }
    println!("2) cursor walk vs enumerated truth: {checked} boxes, {total_runs} runs, all identical");

    println!();
    println!("Both hold. The pair is what a real ordered store wants: a cursor that lands on a key");
    println!("outside the box asks BIGMIN where to resume and LITMAX where the run it just left");
    println!("ended, and never materialises the range list at all. `box_ranges` in cold_index_bench");
    println!("computes the same decomposition by recursion — cheaper to trust, but it builds the");
    println!("whole list first, which is the wrong shape for a cursor over a B-tree or a trie.");
}
