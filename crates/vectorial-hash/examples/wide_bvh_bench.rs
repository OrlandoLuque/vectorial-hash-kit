//! wide_bvh_bench — the follow-through on `compressed_bvh_bench`: the literature
//! says the BVH *latency* win comes not from quantisation but from a wide (8-ary)
//! node — one node holds 8 children whose boxes are tested together (SIMD), so the
//! tree is shallow and the pointer-chase is amortised 8:1. This bench builds three
//! BVHs over the SAME clumpy cloud and measures cull latency. bin-f32 is the classic
//! binary BVH (2 children/node, f32 boxes). wide8-f32 is an 8-ary BVH with each
//! node's 8 child boxes in SoA so the sphere-vs-8-boxes test auto-vectorises (build
//! with -C target-cpu=native for the AVX path). wide8-u16 is the same 8-ary tree
//! with child boxes quantised to u16 relative to the root (footprint + the
//! compressed-node result, now on a wide node). All three test the exact point at
//! the leaf, so all three are verified bit-for-bit == brute force. The question:
//! does going wide+SIMD finally turn the footprint win into a latency win (where the
//! binary u16 node was only a wash)?
//!
//! `RUSTFLAGS=-Ctarget-cpu=native cargo run -p vectorial-hash --example wide_bvh_bench --release`  (`WBVH_N`)

// The explicit `for a in 0..3 { for i in 0..8 { .. lo[a][i] .. d[i] .. } }` indexing
// is DELIBERATE — it is what lets LLVM auto-vectorise the 8-wide box test into AVX;
// an iterator rewrite defeats the whole point of the bench.
#![allow(clippy::needless_range_loop)]
use std::time::Instant;

const LEAF: usize = 8; // a leaf holds up to this many points (tested exactly)

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } fn unit(&mut self) -> f32 { (self.next() & 0xffffff) as f32 / (1u32 << 24) as f32 } }

// ---------- binary BVH (baseline) ----------
#[derive(Clone, Copy)]
struct Bin { mn: [f32; 3], mx: [f32; 3], l: u32, r: u32 } // r==u32::MAX → leaf, l = point index
fn build_bin(pts: &[[f32; 3]], idx: &mut [u32], nodes: &mut Vec<Bin>) -> u32 {
    if idx.len() == 1 {
        let p = pts[idx[0] as usize];
        let id = nodes.len() as u32; nodes.push(Bin { mn: p, mx: p, l: idx[0], r: u32::MAX }); return id;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for &i in idx.iter() { let p = pts[i as usize]; for a in 0..3 { lo[a] = lo[a].min(p[a]); hi[a] = hi[a].max(p[a]); } }
    let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap()).unwrap();
    idx.sort_unstable_by(|&a, &b| pts[a as usize][axis].partial_cmp(&pts[b as usize][axis]).unwrap());
    let mid = idx.len() / 2;
    let id = nodes.len() as u32; nodes.push(Bin { mn: [0.0; 3], mx: [0.0; 3], l: 0, r: 0 });
    let (li, ri) = idx.split_at_mut(mid);
    let l = build_bin(pts, li, nodes); let r = build_bin(pts, ri, nodes);
    let (ln, rn) = (nodes[l as usize], nodes[r as usize]);
    let mut mn = [0.0f32; 3]; let mut mx = [0.0f32; 3];
    for a in 0..3 { mn[a] = ln.mn[a].min(rn.mn[a]); mx[a] = ln.mx[a].max(rn.mx[a]); }
    nodes[id as usize] = Bin { mn, mx, l, r };
    id
}

// ---------- wide (8-ary) BVH ----------
// SoA per node so the 8-box test vectorises. kind: 0 empty · 1 internal · 2 leaf.
#[derive(Clone)]
struct Wide { lo: [[f32; 8]; 3], hi: [[f32; 8]; 3], kind: [u8; 8], idx: [u32; 8], len: [u8; 8] }
impl Wide { fn empty() -> Self { Wide { lo: [[1e30; 8]; 3], hi: [[1e30; 8]; 3], kind: [0; 8], idx: [0; 8], len: [0; 8] } } }

// 3 rounds of longest-axis median split → up to 8 segments of idx.
fn octsplit(pts: &[[f32; 3]], idx: &mut [u32]) -> Vec<(usize, usize)> {
    let mut segs = vec![(0usize, idx.len())];
    for _ in 0..3 {
        let mut next = Vec::with_capacity(segs.len() * 2);
        for (a, b) in segs {
            if b - a <= 1 { next.push((a, b)); continue; }
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for &i in &idx[a..b] { let p = pts[i as usize]; for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
            let ax = (0..3).max_by(|&x, &y| (hi[x] - lo[x]).partial_cmp(&(hi[y] - lo[y])).unwrap()).unwrap();
            idx[a..b].sort_unstable_by(|&x, &y| pts[x as usize][ax].partial_cmp(&pts[y as usize][ax]).unwrap());
            let mid = a + (b - a) / 2; next.push((a, mid)); next.push((mid, b));
        }
        segs = next;
    }
    segs.retain(|&(a, b)| b > a); segs
}

// (kind, idx, len, mn, mx) that a child reports to its parent
type Child = (u8, u32, u8, [f32; 3], [f32; 3]);
fn build_wide(pts: &[[f32; 3]], idx: &mut [u32], nodes: &mut Vec<Wide>, leaf_pts: &mut Vec<u32>) -> Child {
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for &i in idx.iter() { let p = pts[i as usize]; for a in 0..3 { mn[a] = mn[a].min(p[a]); mx[a] = mx[a].max(p[a]); } }
    if idx.len() <= LEAF {
        let start = leaf_pts.len() as u32; leaf_pts.extend_from_slice(idx);
        return (2, start, idx.len() as u8, mn, mx);
    }
    let segs = octsplit(pts, idx);
    let node_id = nodes.len(); nodes.push(Wide::empty());
    let mut kids: Vec<Child> = Vec::with_capacity(8);
    for (a, b) in segs { let child = build_wide(pts, &mut idx[a..b], nodes, leaf_pts); kids.push(child); }
    let mut node = Wide::empty();
    for (ci, &(k, ix, ln, cmn, cmx)) in kids.iter().enumerate() {
        node.kind[ci] = k; node.idx[ci] = ix; node.len[ci] = ln;
        for a in 0..3 { node.lo[a][ci] = cmn[a]; node.hi[a][ci] = cmx[a]; }
    }
    nodes[node_id] = node;
    (1, node_id as u32, 0, mn, mx)
}

// quantised wide node (u16 boxes relative to the root)
#[derive(Clone)]
struct WideQ { qlo: [[u16; 8]; 3], qhi: [[u16; 8]; 3], kind: [u8; 8], idx: [u32; 8], len: [u8; 8] }

type CullFn<'a> = dyn Fn([f32; 3], f32, &mut u64) -> Vec<u32> + 'a;

fn main() {
    let n: usize = std::env::var("WBVH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let w = 1024.0f32;
    let mut rng = Rng(0xC0FFEE);
    let pts: Vec<[f32; 3]> = (0..n).map(|i| {
        if i % 3 == 0 { [rng.unit() * w, rng.unit() * w, rng.unit() * w] }
        else { let b = (i / 40000) as f32; let cx = (b * 137.0) % w; [(cx + (rng.unit() - 0.5) * 120.0).clamp(0.0, w), rng.unit() * w, rng.unit() * w] }
    }).collect();

    // build all three
    let mut idxb: Vec<u32> = (0..n as u32).collect();
    let mut bin: Vec<Bin> = Vec::with_capacity(2 * n);
    build_bin(&pts, &mut idxb, &mut bin);

    let mut idxw: Vec<u32> = (0..n as u32).collect();
    let mut wide: Vec<Wide> = Vec::new();
    let mut leaf_pts: Vec<u32> = Vec::with_capacity(n);
    let (_, _, _, rmn, rmx) = build_wide(&pts, &mut idxw, &mut wide, &mut leaf_pts);

    // quantise the wide nodes to u16 relative to root
    let inv = [65535.0 / (rmx[0] - rmn[0]).max(1e-6), 65535.0 / (rmx[1] - rmn[1]).max(1e-6), 65535.0 / (rmx[2] - rmn[2]).max(1e-6)];
    let step = [(rmx[0] - rmn[0]) / 65535.0, (rmx[1] - rmn[1]) / 65535.0, (rmx[2] - rmn[2]) / 65535.0];
    let wideq: Vec<WideQ> = wide.iter().map(|nd| {
        let mut qlo = [[0u16; 8]; 3]; let mut qhi = [[0u16; 8]; 3];
        for a in 0..3 { for ci in 0..8 {
            if nd.kind[ci] == 0 { qlo[a][ci] = 65535; qhi[a][ci] = 0; continue; } // empty: never hit (masked anyway)
            qlo[a][ci] = ((nd.lo[a][ci] - rmn[a]) * inv[a]).floor().clamp(0.0, 65535.0) as u16;
            qhi[a][ci] = ((nd.hi[a][ci] - rmn[a]) * inv[a]).ceil().clamp(0.0, 65535.0) as u16;
        } }
        WideQ { qlo, qhi, kind: nd.kind, idx: nd.idx, len: nd.len }
    }).collect();

    let hit_box = |mn: [f32; 3], mx: [f32; 3], c: [f32; 3], r: f32| -> bool { let mut d = 0.0f32; for a in 0..3 { let q = c[a].clamp(mn[a], mx[a]); d += (q - c[a]) * (q - c[a]); } d <= r * r };
    let in_sphere = |p: [f32; 3], c: [f32; 3], r: f32| (p[0]-c[0]).powi(2)+(p[1]-c[1]).powi(2)+(p[2]-c[2]).powi(2) <= r*r;

    // --- binary cull ---
    let cull_bin = |c: [f32; 3], r: f32, visits: &mut u64| -> Vec<u32> {
        let mut out = Vec::new(); let mut st = vec![0u32];
        while let Some(nd) = st.pop() {
            *visits += 1; let f = bin[nd as usize];
            if !hit_box(f.mn, f.mx, c, r) { continue; }
            if f.r == u32::MAX { if in_sphere(pts[f.l as usize], c, r) { out.push(f.l); } }
            else { st.push(f.l); st.push(f.r); }
        }
        out.sort_unstable(); out
    };
    // --- wide f32 cull (the 8-box test is the vectorisable kernel) ---
    let cull_wf = |c: [f32; 3], r: f32, visits: &mut u64| -> Vec<u32> {
        let mut out = Vec::new(); let mut st = vec![0u32]; let r2 = r * r;
        while let Some(nd) = st.pop() {
            *visits += 1; let nd = &wide[nd as usize];
            let mut d = [0.0f32; 8];
            for a in 0..3 { for i in 0..8 { let q = c[a].clamp(nd.lo[a][i], nd.hi[a][i]); let e = q - c[a]; d[i] += e * e; } }
            for i in 0..8 {
                if nd.kind[i] == 0 || d[i] > r2 { continue; }
                if nd.kind[i] == 1 { st.push(nd.idx[i]); }
                else { let s = nd.idx[i] as usize; for &pi in &leaf_pts[s..s + nd.len[i] as usize] { if in_sphere(pts[pi as usize], c, r) { out.push(pi); } } }
            }
        }
        out.sort_unstable(); out
    };
    // --- wide u16 cull (dequantise conservatively, then the same 8-box kernel) ---
    let cull_wq = |c: [f32; 3], r: f32, visits: &mut u64| -> Vec<u32> {
        let mut out = Vec::new(); let mut st = vec![0u32]; let r2 = r * r;
        while let Some(nd) = st.pop() {
            *visits += 1; let nd = &wideq[nd as usize];
            let mut d = [0.0f32; 8];
            for a in 0..3 { for i in 0..8 {
                let lo = rmn[a] + nd.qlo[a][i] as f32 * step[a]; let hi = rmn[a] + nd.qhi[a][i] as f32 * step[a];
                let q = c[a].clamp(lo, hi.max(lo)); let e = q - c[a]; d[i] += e * e;
            } }
            for i in 0..8 {
                if nd.kind[i] == 0 || d[i] > r2 { continue; }
                if nd.kind[i] == 1 { st.push(nd.idx[i]); }
                else { let s = nd.idx[i] as usize; for &pi in &leaf_pts[s..s + nd.len[i] as usize] { if in_sphere(pts[pi as usize], c, r) { out.push(pi); } } }
            }
        }
        out.sort_unstable(); out
    };

    // ---- verify all three == brute ----
    let mut qr = Rng(0x515);
    let (mut vb, mut vwf, mut vwq) = (0u64, 0u64, 0u64);
    for _ in 0..24 {
        let c = [qr.unit() * w, qr.unit() * w, qr.unit() * w]; let r = 20.0 + qr.unit() * 120.0;
        let mut brute: Vec<u32> = (0..n as u32).filter(|&j| in_sphere(pts[j as usize], c, r)).collect();
        brute.sort_unstable();
        assert_eq!(cull_bin(c, r, &mut vb), brute, "binary != brute");
        assert_eq!(cull_wf(c, r, &mut vwf), brute, "wide-f32 != brute");
        assert_eq!(cull_wq(c, r, &mut vwq), brute, "wide-u16 != brute");
    }

    // ---- measure ----
    let spheres: Vec<([f32; 3], f32)> = (0..200).map(|_| ([qr.unit() * w, qr.unit() * w, qr.unit() * w], 20.0 + qr.unit() * 120.0)).collect();
    let bench = |f: &CullFn| -> (f64, f64) {
        let mut best = f64::MAX; let mut vis = 0u64;
        for _ in 0..8 { let t = Instant::now(); let mut v = 0u64; let mut acc = 0usize; for &(c, r) in &spheres { acc += f(c, r, &mut v).len(); } std::hint::black_box(acc); let e = t.elapsed().as_secs_f64(); if e < best { best = e; vis = v; } }
        (best * 1e6 / spheres.len() as f64, vis as f64 / spheres.len() as f64)
    };
    let (tb, nb) = bench(&cull_bin);
    let (twf, nwf) = bench(&cull_wf);
    let (twq, nwq) = bench(&cull_wq);

    let bin_bytes = bin.len() * std::mem::size_of::<Bin>();
    let wf_bytes = wide.len() * std::mem::size_of::<Wide>() + leaf_pts.len() * 4;
    let wq_bytes = wideq.len() * std::mem::size_of::<WideQ>() + leaf_pts.len() * 4;
    println!("wide-BVH (8-ary, SoA/SIMD), {n} points  (LEAF={LEAF})\n");
    println!("verified: binary == wide-f32 == wide-u16 == brute force (exact leaf test) ✓\n");
    println!("            | node B | arena MB | nodes/query | cull µs/query | vs binary");
    println!("  bin-f32   | {:>6} | {:>7.1} | {:>11.0} | {:>13.3} | 1.00×", std::mem::size_of::<Bin>(), bin_bytes as f64 / 1e6, nb, tb);
    println!("  wide8-f32 | {:>6} | {:>7.1} | {:>11.0} | {:>13.3} | {}", std::mem::size_of::<Wide>(), wf_bytes as f64 / 1e6, nwf, twf, speed(tb, twf));
    println!("  wide8-u16 | {:>6} | {:>7.1} | {:>11.0} | {:>13.3} | {}", std::mem::size_of::<WideQ>(), wq_bytes as f64 / 1e6, nwq, twq, speed(tb, twq));
    println!("\nreading: the wide node visits ~{:.0}× fewer nodes than the binary tree (8:1 fan-out,\nshallow) and tests its 8 child boxes in one vectorisable sweep. Whether that beats the\nbinary pointer-chase — and whether u16 helps or the dequantise offsets it — is above.\n(For the AVX path build with RUSTFLAGS=-Ctarget-cpu=native.)", nb / nwf.max(1.0));
}

fn speed(base: f64, x: f64) -> String { if x < base { format!("{:.2}× FASTER", base / x) } else { format!("{:.2}× slower", x / base) } }
