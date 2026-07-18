//! compressed_bvh_bench — the research's "compressed/quantized wide-BVH nodes"
//! (the memory wall), done exactly. A binary BVH over N points, stored two ways
//! with the SAME topology. FULL keeps f32 min/max per node (32 B/node). QUANT
//! quantises each node's box to u16 relative to the root, min rounded DOWN and max
//! rounded UP so the dequantised box is a superset of the true box (20 B/node).
//! The quantised boxes are conservative, so a descent may visit a few extra nodes
//! — but the leaves test the exact point, so both cull answers equal brute force
//! (compression trades memory for a little traversal, NOT accuracy; the earlier
//! "breaks exactness" worry was wrong — exact leaf tests fix it). The bench reports
//! node bytes, over-visit %, and whether the smaller node's cache win beats the
//! extra traversal.
//!
//! `cargo run -p vectorial-hash --example compressed_bvh_bench --release`  (`CBVH_N`)
use std::time::Instant;

type CullFn<'a> = dyn Fn([f32; 3], f32, &mut u64) -> Vec<u32> + 'a;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } fn unit(&mut self) -> f32 { (self.next() & 0xffffff) as f32 / (1u32 << 24) as f32 } }

#[derive(Clone, Copy)]
struct Full { mn: [f32; 3], mx: [f32; 3], l: u32, r: u32 } // r==u32::MAX → leaf, l = point index
#[derive(Clone, Copy)]
struct Quant { qmn: [u16; 3], qmx: [u16; 3], l: u32, r: u32 }

// Top-down median-split BVH build (longest axis). Returns the node index; leaves
// hold their point index. Boxes are the exact point bounds, unioned up.
fn build(pts: &[[f32; 3]], idx: &mut [u32], nodes: &mut Vec<Full>) -> u32 {
    if idx.len() == 1 {
        let p = pts[idx[0] as usize];
        let id = nodes.len() as u32;
        nodes.push(Full { mn: p, mx: p, l: idx[0], r: u32::MAX });
        return id;
    }
    // longest-axis of the centroid spread
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for &i in idx.iter() { let p = pts[i as usize]; for a in 0..3 { lo[a] = lo[a].min(p[a]); hi[a] = hi[a].max(p[a]); } }
    let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap()).unwrap();
    idx.sort_unstable_by(|&a, &b| pts[a as usize][axis].partial_cmp(&pts[b as usize][axis]).unwrap());
    let mid = idx.len() / 2;
    let id = nodes.len() as u32;
    nodes.push(Full { mn: [0.0; 3], mx: [0.0; 3], l: 0, r: 0 }); // placeholder
    let (left_idx, right_idx) = idx.split_at_mut(mid);
    let l = build(pts, left_idx, nodes);
    let r = build(pts, right_idx, nodes);
    let (ln, rn) = (nodes[l as usize], nodes[r as usize]);
    let mut mn = [0.0f32; 3]; let mut mx = [0.0f32; 3];
    for a in 0..3 { mn[a] = ln.mn[a].min(rn.mn[a]); mx[a] = ln.mx[a].max(rn.mx[a]); }
    nodes[id as usize] = Full { mn, mx, l, r };
    id
}

fn main() {
    let n: usize = std::env::var("CBVH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let w = 1024.0f32;
    let mut rng = Rng(0xC0FFEE);
    // a clumpy cloud (a few gaussian-ish blobs + scatter) so the BVH is non-trivial
    let pts: Vec<[f32; 3]> = (0..n).map(|i| {
        if i % 3 == 0 { let c = [rng.unit() * w, rng.unit() * w, rng.unit() * w]; [c[0], c[1], c[2]] }
        else { let b = (i / 40000) as f32; let cx = (b * 137.0) % w; [(cx + (rng.unit() - 0.5) * 120.0).clamp(0.0, w), rng.unit() * w, rng.unit() * w] }
    }).collect();

    let mut idx: Vec<u32> = (0..n as u32).collect();
    let mut full: Vec<Full> = Vec::with_capacity(2 * n);
    build(&pts, &mut idx, &mut full);
    let root = 0u32; // build pushes the root first

    // root box → quantise every node's box to u16 relative to it (min↓, max↑).
    let (rmn, rmx) = (full[root as usize].mn, full[root as usize].mx);
    let inv = [65535.0 / (rmx[0] - rmn[0]).max(1e-6), 65535.0 / (rmx[1] - rmn[1]).max(1e-6), 65535.0 / (rmx[2] - rmn[2]).max(1e-6)];
    let quant: Vec<Quant> = full.iter().map(|f| {
        let mut qmn = [0u16; 3]; let mut qmx = [0u16; 3];
        for a in 0..3 {
            qmn[a] = ((f.mn[a] - rmn[a]) * inv[a]).floor().clamp(0.0, 65535.0) as u16;
            qmx[a] = ((f.mx[a] - rmn[a]) * inv[a]).ceil().clamp(0.0, 65535.0) as u16;
        }
        Quant { qmn, qmx, l: f.l, r: f.r }
    }).collect();
    let step = [(rmx[0] - rmn[0]) / 65535.0, (rmx[1] - rmn[1]) / 65535.0, (rmx[2] - rmn[2]) / 65535.0];

    // sphere-vs-box overlap
    let hit_box = |mn: [f32; 3], mx: [f32; 3], c: [f32; 3], r: f32| -> bool {
        let mut d = 0.0f32; for a in 0..3 { let q = c[a].clamp(mn[a], mx[a]); d += (q - c[a]) * (q - c[a]); } d <= r * r
    };
    // cull on the full BVH (exact leaf test)
    let cull_full = |c: [f32; 3], r: f32, visits: &mut u64| -> Vec<u32> {
        let mut out = Vec::new(); let mut st = vec![root];
        while let Some(nd) = st.pop() {
            *visits += 1; let f = full[nd as usize];
            if !hit_box(f.mn, f.mx, c, r) { continue; }
            if f.r == u32::MAX { let p = pts[f.l as usize]; if (p[0]-c[0]).powi(2)+(p[1]-c[1]).powi(2)+(p[2]-c[2]).powi(2) <= r*r { out.push(f.l); } }
            else { st.push(f.l); st.push(f.r); }
        }
        out.sort_unstable(); out
    };
    // cull on the quantised BVH (dequantise conservatively; exact leaf test)
    let cull_quant = |c: [f32; 3], r: f32, visits: &mut u64| -> Vec<u32> {
        let mut out = Vec::new(); let mut st = vec![root];
        while let Some(nd) = st.pop() {
            *visits += 1; let q = quant[nd as usize];
            let mn = [rmn[0] + q.qmn[0] as f32 * step[0], rmn[1] + q.qmn[1] as f32 * step[1], rmn[2] + q.qmn[2] as f32 * step[2]];
            let mx = [rmn[0] + q.qmx[0] as f32 * step[0], rmn[1] + q.qmx[1] as f32 * step[1], rmn[2] + q.qmx[2] as f32 * step[2]];
            if !hit_box(mn, mx, c, r) { continue; }
            if q.r == u32::MAX { let p = pts[q.l as usize]; if (p[0]-c[0]).powi(2)+(p[1]-c[1]).powi(2)+(p[2]-c[2]).powi(2) <= r*r { out.push(q.l); } }
            else { st.push(q.l); st.push(q.r); }
        }
        out.sort_unstable(); out
    };

    // ---- verify BOTH == brute force, over 24 spheres ----
    let mut qr = Rng(0x515);
    let (mut vf, mut vq) = (0u64, 0u64);
    for _ in 0..24 {
        let c = [qr.unit() * w, qr.unit() * w, qr.unit() * w]; let r = 20.0 + qr.unit() * 120.0;
        let mut brute: Vec<u32> = (0..n as u32).filter(|&j| { let p = pts[j as usize]; (p[0]-c[0]).powi(2)+(p[1]-c[1]).powi(2)+(p[2]-c[2]).powi(2) <= r*r }).collect();
        brute.sort_unstable();
        assert_eq!(cull_full(c, r, &mut vf), brute, "full BVH != brute");
        assert_eq!(cull_quant(c, r, &mut vq), brute, "quant BVH != brute (should be EXACT via leaf test)");
    }

    // ---- measure cull time (min of N over the same 200 spheres) ----
    let spheres: Vec<([f32; 3], f32)> = (0..200).map(|_| ([qr.unit() * w, qr.unit() * w, qr.unit() * w], 20.0 + qr.unit() * 120.0)).collect();
    let bench = |f: &CullFn| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..8 { let t = Instant::now(); let mut v = 0; let mut acc = 0usize; for &(c, r) in &spheres { acc += f(c, r, &mut v).len(); } std::hint::black_box(acc); best = best.min(t.elapsed().as_secs_f64()); }
        best * 1e6 / spheres.len() as f64
    };
    let tf = bench(&cull_full);
    let tq = bench(&cull_quant);

    let (bf, bq) = (std::mem::size_of::<Full>(), std::mem::size_of::<Quant>());
    println!("compressed-BVH (quantised nodes), {n} points, {} nodes\n", full.len());
    println!("verified: full == brute AND quant == brute (exact — conservative boxes, exact leaf test) ✓\n");
    println!("  node size : full {bf} B   quant {bq} B   ({:.2}× smaller)", bf as f64 / bq as f64);
    println!("  arena     : full {:.1} MB  quant {:.1} MB", (full.len() * bf) as f64 / 1e6, (full.len() * bq) as f64 / 1e6);
    println!("  over-visit: quant visits {:.1}% more nodes (conservative boxes)", (vq as f64 / vf as f64 - 1.0) * 100.0);
    println!("  cull      : full {tf:.3} µs/query   quant {tq:.3} µs/query   {}", if tq < tf { format!("quant {:.2}× FASTER", tf / tq) } else { format!("quant {:.2}× slower", tq / tf) });
    println!("\nreading: quantised nodes are {:.1}× smaller (the memory-wall lever) at ZERO accuracy cost\n(exact leaf tests). Whether that *net* helps cull is the smaller-node cache win vs the\nfew extra nodes the conservative boxes make you visit — the numbers above are the honest verdict.", bf as f64 / bq as f64);
}
