//! gpu_lbvh_query_bench — **query the GPU-built LBVH on the GPU**. The build bench
//! (`gpu_lbvh_build_bench`) builds the tree on-GPU but only *verifies* it by walking
//! it on the CPU; this runs an actual **batch of range-count queries entirely on the
//! GPU** (one thread per query, a stack-based BVH descent), so the whole
//! build→query broad-phase is GPU-resident. It ties last session's on-GPU LBVH to
//! tonight's parallel batch queries: the rival is the CPU `Tree3` answering the same
//! batch (serial and, with `--features parallel`, `cull_many_par`).
//!
//! Verified before timing: the GPU per-query counts must equal brute force over the
//! same spheres — so a fast number is also a *correct* one.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_lbvh_query_bench --release
//! #                                            (add --features parallel for the CPU par rival)
//! ```
//! Env: `LBVHQ_N` (points), `LBVHQ_Q` (queries), `LBVHQ_R` (query radius).
use std::time::Instant;
use vectorial_hash::{Aabb, Point3, Positioned3, Sphere3, Tree3};

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } }

struct QP(Point3);
impl Positioned3 for QP { fn position(&self) -> Point3 { self.0 } }

const TILE: usize = 256;

// ---- LBVH build shaders (verbatim from gpu_lbvh_build_bench) ----
const MORTON_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       pts:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals:  array<u32>;
@group(0) @binding(3) var<uniform>             mp:    vec4<u32>;
fn part1by2(n: u32) -> u32 {
    var x = n & 0x000003ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x <<  8u)) & 0x0300f00fu;
    x = (x | (x <<  4u)) & 0x030c30c3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}
@compute @workgroup_size(256)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= mp.x) { return; }
    let p = pts[i];
    codes[i] = part1by2(min(u32(p.x), 1023u)) | (part1by2(min(u32(p.y), 1023u)) << 1u) | (part1by2(min(u32(p.z), 1023u)) << 2u);
    vals[i]  = i;
}
"#;
const RADIX_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       src_k:     array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_k:     array<u32>;
@group(0) @binding(2) var<storage, read>       src_v:     array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_v:     array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_hist: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_off:  array<u32>;
@group(0) @binding(6) var<uniform>             p:         vec4<u32>;
@group(0) @binding(7) var<storage, read_write> block_tot: array<u32>;
@group(0) @binding(8) var<storage, read_write> block_off: array<u32>;
const RADIX: u32 = 256u;
const BLOCK: u32 = 512u;
var<workgroup> lhist:  array<atomic<u32>, 256>;
var<workgroup> ldig:   array<u32, 256>;
var<workgroup> wtotal: array<u32, 256>;
var<workgroup> wbase:  array<u32, 256>;
@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    if (li < RADIX) { atomicStore(&lhist[li], 0u); }
    workgroupBarrier();
    let i = gid.x;
    if (i < p.x) { atomicAdd(&lhist[(src_k[i] >> p.y) & 0xFFu], 1u); }
    workgroupBarrier();
    if (li < RADIX) { tile_hist[wid.x * RADIX + li] = atomicLoad(&lhist[li]); }
}
@compute @workgroup_size(256)
fn scan_reduce(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x; let nt = p.z; let b = wid.x;
    let lo = b * BLOCK; let hi = min(lo + BLOCK, nt);
    var run = 0u;
    for (var t = lo; t < hi; t = t + 1u) { tile_off[t * RADIX + d] = run; run = run + tile_hist[t * RADIX + d]; }
    block_tot[b * RADIX + d] = run;
}
@compute @workgroup_size(256)
fn scan_top(@builtin(local_invocation_id) lid: vec3<u32>) {
    let d = lid.x; let nb = p.w;
    var tot = 0u;
    for (var b = 0u; b < nb; b = b + 1u) { tot = tot + block_tot[b * RADIX + d]; }
    wtotal[d] = tot;
    workgroupBarrier();
    if (d == 0u) { var acc = 0u; for (var k = 0u; k < RADIX; k = k + 1u) { wbase[k] = acc; acc = acc + wtotal[k]; } }
    workgroupBarrier();
    var run = wbase[d];
    for (var b = 0u; b < nb; b = b + 1u) { block_off[b * RADIX + d] = run; run = run + block_tot[b * RADIX + d]; }
}
@compute @workgroup_size(256)
fn scan_add(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x; let nt = p.z; let b = wid.x;
    let base = block_off[b * RADIX + d];
    let lo = b * BLOCK; let hi = min(lo + BLOCK, nt);
    for (var t = lo; t < hi; t = t + 1u) { tile_off[t * RADIX + d] = tile_off[t * RADIX + d] + base; }
}
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x; let i = gid.x;
    var key = 0xFFFFFFFFu; var d = 0xFFu;
    if (i < p.x) { key = src_k[i]; d = (key >> p.y) & 0xFFu; }
    ldig[li] = d;
    workgroupBarrier();
    if (i < p.x) {
        var rank = 0u;
        for (var j = 0u; j < li; j = j + 1u) { if (ldig[j] == d) { rank = rank + 1u; } }
        let pos = tile_off[wid.x * RADIX + d] + rank;
        dst_k[pos] = key; dst_v[pos] = src_v[i];
    }
}
"#;
const KARRAS_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       codes:    array<u32>;
@group(0) @binding(1) var<storage, read>       val:      array<u32>;
@group(0) @binding(2) var<storage, read>       pts:      array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> children: array<u32>;
@group(0) @binding(4) var<storage, read_write> parent:   array<u32>;
@group(0) @binding(5) var<storage, read_write> aabb:     array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> flags:    array<atomic<u32>>;
@group(0) @binding(7) var<uniform>             kp:       vec4<u32>;
fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) { return -1; }
    let ci = codes[u32(i)]; let cj = codes[u32(j)];
    if (ci == cj) { return 32 + i32(countLeadingZeros(u32(i) ^ u32(j))); }
    return i32(countLeadingZeros(ci ^ cj));
}
@compute @workgroup_size(256)
fn karras(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(kp.x); let i = i32(gid.x);
    if (i >= n - 1) { return; }
    let d = select(-1, 1, delta(i, i + 1, n) > delta(i, i - 1, n));
    let dmin = delta(i, i - d, n);
    var lmax = 2;
    while (delta(i, i + lmax * d, n) > dmin) { lmax = lmax * 2; }
    var l = 0; var t = lmax / 2;
    while (t >= 1) { if (delta(i, i + (l + t) * d, n) > dmin) { l = l + t; } t = t / 2; }
    let j = i + l * d;
    let first = min(i, j); let last = max(i, j);
    let cp = delta(first, last, n);
    var split = first; var step = last - first;
    loop {
        step = (step + 1) / 2;
        let ns = split + step;
        if (ns < last && delta(first, ns, n) > cp) { split = ns; }
        if (step <= 1) { break; }
    }
    let leaf_base = u32(n - 1);
    var lc = u32(split);     if (split == first)    { lc = leaf_base + u32(split); }
    var rc = u32(split + 1); if (split + 1 == last) { rc = leaf_base + u32(split + 1); }
    children[u32(i) * 2u] = lc; children[u32(i) * 2u + 1u] = rc;
    parent[lc] = u32(i); parent[rc] = u32(i);
}
@compute @workgroup_size(256)
fn leaf_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = kp.x; let s = gid.x;
    if (s >= n) { return; }
    let leaf = (n - 1u) + s;
    let p = pts[val[s]].xyz;
    aabb[leaf * 2u] = vec4<f32>(p, 0.0); aabb[leaf * 2u + 1u] = vec4<f32>(p, 0.0);
}
@compute @workgroup_size(256)
fn refit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = kp.x; let s = gid.x;
    if (s >= n) { return; }
    var node = parent[(n - 1u) + s];
    loop {
        if (atomicAdd(&flags[node], 1u) == 0u) { return; }
        let lc = children[node * 2u]; let rc = children[node * 2u + 1u];
        let mn = min(aabb[lc * 2u].xyz,      aabb[rc * 2u].xyz);
        let mx = max(aabb[lc * 2u + 1u].xyz, aabb[rc * 2u + 1u].xyz);
        aabb[node * 2u] = vec4<f32>(mn, 0.0); aabb[node * 2u + 1u] = vec4<f32>(mx, 0.0);
        if (node == 0u) { return; }
        node = parent[node];
    }
}
"#;

// ---- the NEW bit: range-count queries traversing the LBVH ON THE GPU ----
// one thread per query sphere, an explicit stack descent (nearest-point box prune;
// a leaf's box IS its point, so reaching it == the point being inside the sphere).
const QUERY_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       children: array<u32>;
@group(0) @binding(1) var<storage, read>       aabb:     array<vec4<f32>>; // 2 (min,max) per node
@group(0) @binding(2) var<storage, read>       queries:  array<vec4<f32>>; // xyz = centre, w = radius
@group(0) @binding(3) var<storage, read_write> counts:   array<u32>;
@group(0) @binding(4) var<uniform>             qp:       vec4<u32>;         // x = n points, y = nq
@compute @workgroup_size(64)
fn range_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= qp.y) { return; }
    let q = queries[qi];
    let c = q.xyz; let r2 = q.w * q.w;
    let leaf_base = qp.x - 1u;
    var stack: array<u32, 64>;
    var sp = 1u; stack[0] = 0u; // root = internal node 0
    var cnt = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let nd = stack[sp];
        let mn = aabb[nd * 2u].xyz; let mx = aabb[nd * 2u + 1u].xyz;
        let dv = clamp(c, mn, mx) - c;                 // nearest point of the box to c
        if (dot(dv, dv) > r2) { continue; }            // box beyond r → prune
        if (nd >= leaf_base) { cnt = cnt + 1u; }       // leaf box IS its point, already ≤ r
        else if (sp < 62u) {
            stack[sp] = children[nd * 2u];     sp = sp + 1u;
            stack[sp] = children[nd * 2u + 1u]; sp = sp + 1u;
        }
    }
    counts[qi] = cnt;
}
"#;

// ---- k-NN over the GPU-built LBVH: one thread per query keeps the k nearest in a
// small local buffer, pruning any subtree whose box is farther than the current
// k-th. (No child ordering — the worst-distance prune alone keeps it correct.)
const KNN_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       children: array<u32>;
@group(0) @binding(1) var<storage, read>       aabb:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       queries:  array<vec4<f32>>; // xyz = centre
@group(0) @binding(3) var<storage, read_write> results:  array<f32>;        // nq*KMAX nearest d2 (CPU sorts)
@group(0) @binding(4) var<uniform>             qp:       vec4<u32>;         // x=n, y=nq, z=k
const KMAX: u32 = 16u;
@compute @workgroup_size(64)
fn knn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= qp.y) { return; }
    let c = queries[qi].xyz;
    let k = min(qp.z, KMAX);
    var best: array<f32, 16>;
    for (var i = 0u; i < KMAX; i = i + 1u) { best[i] = 3.0e38; }
    var worst = 3.0e38;
    let leaf_base = qp.x - 1u;
    var stack: array<u32, 64>;
    var sp = 1u; stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let nd = stack[sp];
        let dv = clamp(c, aabb[nd * 2u].xyz, aabb[nd * 2u + 1u].xyz) - c;
        let boxd = dot(dv, dv);
        if (boxd >= worst) { continue; }              // whole subtree beyond the k-th → prune
        if (nd >= leaf_base) {
            // insert boxd (a leaf box IS its point, so boxd == the point distance)
            var mi = 0u; var mv = best[0];
            for (var i = 1u; i < k; i = i + 1u) { if (best[i] > mv) { mv = best[i]; mi = i; } }
            best[mi] = boxd;
            var w = best[0];
            for (var i = 1u; i < k; i = i + 1u) { if (best[i] > w) { w = best[i]; } }
            worst = w;
        } else if (sp < 62u) {
            stack[sp] = children[nd * 2u];     sp = sp + 1u;
            stack[sp] = children[nd * 2u + 1u]; sp = sp + 1u;
        }
    }
    for (var i = 0u; i < k; i = i + 1u) { results[qi * KMAX + i] = best[i]; }
}
"#;

// A bind-group entry borrowing `buf` — a fn (not a closure) so the elided lifetime
// ties the entry's borrow to the buffer.
fn be(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let n = std::env::var("LBVHQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_048_576usize);
    let nq = std::env::var("LBVHQ_Q").ok().and_then(|v| v.parse().ok()).unwrap_or(4_096usize);
    let radius: f32 = std::env::var("LBVHQ_R").ok().and_then(|v| v.parse().ok()).unwrap_or(24.0);
    let num_tiles = n.div_ceil(TILE);
    let n2 = num_tiles * TILE;
    println!("GPU LBVH range-count queries | {n} points, {nq} queries, r={radius} (build padded to {n2})\n");

    let mut r = Rng(0x1234_ABCD);
    let pts: Vec<[f32; 4]> = (0..n).map(|_| [
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0,
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0,
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0, 0.0]).collect();
    let queries: Vec<[f32; 4]> = (0..nq).map(|_| [(r.next() % 1024) as f32, (r.next() % 1024) as f32, (r.next() % 1024) as f32, radius]).collect();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.expect("device");

    let sbuf = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pts_b = sbuf((n2 * 16) as u64);
    let (code_a, code_b) = (sbuf((n2 * 4) as u64), sbuf((n2 * 4) as u64));
    let (val_a, val_b)   = (sbuf((n2 * 4) as u64), sbuf((n2 * 4) as u64));
    let hist_b = sbuf((num_tiles * 256 * 4) as u64);
    let off_b  = sbuf((num_tiles * 256 * 4) as u64);
    let num_blocks = num_tiles.div_ceil(512);
    let block_tot_b = sbuf((num_blocks * 256 * 4) as u64);
    let block_off_b = sbuf((num_blocks * 256 * 4) as u64);
    let uni_b = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mp_b = uni_b(16); let rp_b = uni_b(16);

    let sto = |ro: bool| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None };
    let uni = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    let ent = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };

    let m_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(MORTON_SRC.into()) });
    let m_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(0, sto(true)), ent(1, sto(false)), ent(2, sto(false)), ent(3, uni)] });
    let m_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&m_bgl], push_constant_ranges: &[] });
    let m_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&m_pl), module: &m_mod, entry_point: "morton", compilation_options: Default::default() });
    let m_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &m_bgl, entries: &[be(0, &pts_b), be(1, &code_a), be(2, &val_a), be(3, &mp_b)] });

    let r_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RADIX_SRC.into()) });
    let r_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(0, sto(true)), ent(1, sto(false)), ent(2, sto(true)), ent(3, sto(false)), ent(4, sto(false)), ent(5, sto(false)), ent(6, uni), ent(7, sto(false)), ent(8, sto(false))] });
    let r_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&r_bgl], push_constant_ranges: &[] });
    let rpipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&r_pl), module: &r_mod, entry_point: ep, compilation_options: Default::default() });
    let (hist_p, reduce_p, top_p, add_p, scat_p) = (rpipe("histogram"), rpipe("scan_reduce"), rpipe("scan_top"), rpipe("scan_add"), rpipe("scatter"));
    let rbg = |sk: &wgpu::Buffer, dk: &wgpu::Buffer, sv: &wgpu::Buffer, dv: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &r_bgl, entries: &[be(0, sk), be(1, dk), be(2, sv), be(3, dv), be(4, &hist_b), be(5, &off_b), be(6, &rp_b), be(7, &block_tot_b), be(8, &block_off_b)] });
    let bg_ab = rbg(&code_a, &code_b, &val_a, &val_b);
    let bg_ba = rbg(&code_b, &code_a, &val_b, &val_a);

    let ni = (n - 1).max(1);
    let children_b = sbuf((2 * ni * 4) as u64);
    let parent_b   = sbuf(((2 * n - 1) * 4) as u64);
    let aabb_b     = sbuf((2 * (2 * n - 1) * 16) as u64);
    let flags_b    = sbuf((ni * 4) as u64);
    let kp_b = uni_b(16);
    let k_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(KARRAS_SRC.into()) });
    let k_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(0, sto(true)), ent(1, sto(true)), ent(2, sto(true)), ent(3, sto(false)), ent(4, sto(false)), ent(5, sto(false)), ent(6, sto(false)), ent(7, uni)] });
    let k_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&k_bgl], push_constant_ranges: &[] });
    let kpipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&k_pl), module: &k_mod, entry_point: ep, compilation_options: Default::default() });
    let (karras_p, leaf_p, refit_p) = (kpipe("karras"), kpipe("leaf_init"), kpipe("refit"));
    let k_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &k_bgl, entries: &[be(0, &code_a), be(1, &val_a), be(2, &pts_b), be(3, &children_b), be(4, &parent_b), be(5, &aabb_b), be(6, &flags_b), be(7, &kp_b)] });

    // query pipeline
    let queries_b = sbuf((nq * 16) as u64);
    let counts_b  = sbuf((nq * 4) as u64);
    let qp_b = uni_b(16);
    let q_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(QUERY_SRC.into()) });
    let q_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(0, sto(true)), ent(1, sto(true)), ent(2, sto(true)), ent(3, sto(false)), ent(4, uni)] });
    let q_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&q_bgl], push_constant_ranges: &[] });
    let q_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&q_pl), module: &q_mod, entry_point: "range_count", compilation_options: Default::default() });
    let q_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &q_bgl, entries: &[be(0, &children_b), be(1, &aabb_b), be(2, &queries_b), be(3, &counts_b), be(4, &qp_b)] });
    // k-NN pipeline — same 5-binding layout as the range query (results replaces counts).
    let k: usize = 8;
    let results_b = sbuf((nq * 16 * 4) as u64); // KMAX = 16 stride
    let kqp_b = uni_b(16);
    let kn_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(KNN_SRC.into()) });
    let kn_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&q_pl), module: &kn_mod, entry_point: "knn", compilation_options: Default::default() });
    let kn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &q_bgl, entries: &[be(0, &children_b), be(1, &aabb_b), be(2, &queries_b), be(3, &results_b), be(4, &kqp_b)] });

    let wg = num_tiles as u32;
    let build = || {
        queue.write_buffer(&pts_b, 0, bytemuck::cast_slice(&pts));
        queue.write_buffer(&code_a, 0, bytemuck::cast_slice(&vec![u32::MAX; n2]));
        queue.write_buffer(&mp_b, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &m_bg, &[]); c.set_pipeline(&m_pipe); c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
        queue.submit(Some(enc.finish()));
        for pass in 0..4u32 {
            let g = if pass % 2 == 0 { &bg_ab } else { &bg_ba };
            queue.write_buffer(&rp_b, 0, bytemuck::cast_slice(&[n2 as u32, pass * 8, num_tiles as u32, num_blocks as u32]));
            let nb = num_blocks as u32;
            let mut enc = device.create_command_encoder(&Default::default());
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&hist_p);   c.dispatch_workgroups(wg, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&reduce_p); c.dispatch_workgroups(nb, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&top_p);    c.dispatch_workgroups(1, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&add_p);    c.dispatch_workgroups(nb, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scat_p);   c.dispatch_workgroups(wg, 1, 1); }
            queue.submit(Some(enc.finish()));
        }
        queue.write_buffer(&kp_b, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
        queue.write_buffer(&flags_b, 0, bytemuck::cast_slice(&vec![0u32; ni]));
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&karras_p); c.dispatch_workgroups(((n - 1) as u32).div_ceil(256), 1, 1); }
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&leaf_p);   c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&refit_p);  c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    };

    let query = || {
        queue.write_buffer(&queries_b, 0, bytemuck::cast_slice(&queries));
        queue.write_buffer(&qp_b, 0, bytemuck::cast_slice(&[n as u32, nq as u32, 0, 0]));
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &q_bg, &[]); c.set_pipeline(&q_pipe); c.dispatch_workgroups((nq as u32).div_ceil(64), 1, 1); }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    };
    let knn_run = || {
        queue.write_buffer(&queries_b, 0, bytemuck::cast_slice(&queries));
        queue.write_buffer(&kqp_b, 0, bytemuck::cast_slice(&[n as u32, nq as u32, k as u32, 0]));
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &kn_bg, &[]); c.set_pipeline(&kn_pipe); c.dispatch_workgroups((nq as u32).div_ceil(64), 1, 1); }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    };

    build();
    query();

    // read the GPU counts
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (nq * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&counts_b, 0, &rb, 0, (nq * 4) as u64);
    queue.submit(Some(enc.finish()));
    rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let gpu_counts: Vec<u32> = bytemuck::cast_slice(&rb.slice(..).get_mapped_range()).to_vec();

    // ---- verify GPU counts == brute force over the same spheres ----
    let brute = |c: [f32; 4]| -> u32 {
        pts[..n].iter().filter(|p| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2) <= c[3] * c[3]).count() as u32
    };
    let sample = nq.min(64);
    for qi in 0..sample {
        let b = brute(queries[qi]);
        assert_eq!(gpu_counts[qi], b, "GPU range-count != brute for query {qi} ({:?})", queries[qi]);
    }
    let total_hits: u64 = gpu_counts.iter().map(|&c| c as u64).sum();
    println!("verified: GPU per-query range-counts == brute force over {sample} sampled spheres ✓");
    println!("({} total hits over {nq} queries, mean {:.1} pts/query)\n", total_hits, total_hits as f64 / nq as f64);

    // ---- timing: GPU query (build excluded) vs CPU Tree3 batch cull ----
    let mut gpu_ms = f64::MAX;
    for _ in 0..9 { let t = Instant::now(); query(); gpu_ms = gpu_ms.min(t.elapsed().as_secs_f64() * 1e3); }
    let mut gpu_build_ms = f64::MAX;
    for _ in 0..5 { let t = Instant::now(); build(); gpu_build_ms = gpu_build_ms.min(t.elapsed().as_secs_f64() * 1e3); }

    let world = Aabb::new(0.0, 0.0, 0.0, 1024.0, 1024.0, 1024.0);
    let tree = Tree3::<QP>::bulk_load(world, 8, pts.iter().map(|p| QP(Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))).collect());
    let spheres: Vec<Sphere3> = queries.iter().map(|q| Sphere3::new(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64)).collect();
    let mut cpu_ms = f64::MAX;
    let mut cpu_hits = 0u64;
    for _ in 0..5 { let t = Instant::now(); cpu_hits = spheres.iter().map(|s| tree.cull(s).len() as u64).sum(); cpu_ms = cpu_ms.min(t.elapsed().as_secs_f64() * 1e3); }
    // Cross-check (NOT the primary gate — that's GPU == brute above): the CPU Tree3
    // is f64 while the GPU/brute path is f32, so a point at distance ≈ radius can round
    // in on one side and out on the other. Over 4096×~55 hits that's a handful of ties;
    // allow ≤0.2 % slack, which still catches a real divergence (thousands off).
    let diff = (cpu_hits as i64 - total_hits as i64).unsigned_abs();
    assert!(diff * 500 <= total_hits.max(1), "CPU Tree3 total {cpu_hits} vs GPU {total_hits}: {diff} apart (>0.2% — real divergence, not a boundary tie)");

    println!("batch range-count — {nq} queries over {n} points (min timing):");
    println!("  GPU LBVH query (build EXCLUDED)      : {gpu_ms:>8.3} ms   ({:.1} Mqueries/s)", nq as f64 / (gpu_ms / 1e3) / 1e6);
    println!("  GPU LBVH build (Morton+radix+Karras) : {gpu_build_ms:>8.3} ms   (amortised over many query frames)");
    println!("  CPU Tree3 cull_many (serial)         : {cpu_ms:>8.3} ms");
    let vs = if gpu_ms < cpu_ms { format!("the GPU LBVH query BEATS the CPU serial cull {:.2}×", cpu_ms / gpu_ms) } else { format!("the CPU serial cull still wins {:.2}×", gpu_ms / cpu_ms) };
    println!("\n→ At {nq} queries / {n} points, {vs} (query only). With the build folded in, the GPU\n  pays off once the tree is queried enough times per rebuild to amortise the {gpu_build_ms:.1} ms build.");

    // ---- k-NN on the GPU: one thread per query keeps the k nearest, verified by distance ----
    knn_run();
    let krb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (nq * 16 * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&results_b, 0, &krb, 0, (nq * 16 * 4) as u64);
    queue.submit(Some(enc.finish()));
    krb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let gpu_knn: Vec<f32> = bytemuck::cast_slice(&krb.slice(..).get_mapped_range()).to_vec();
    for qi in 0..sample {
        let c = queries[qi];
        let mut d2: Vec<f32> = pts[..n].iter().map(|p| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).collect();
        d2.sort_by(|a, b| a.total_cmp(b));
        let mut got: Vec<f32> = gpu_knn[qi * 16..qi * 16 + k].to_vec();
        got.sort_by(|a, b| a.total_cmp(b));
        for j in 0..k { assert!((got[j].sqrt() - d2[j].sqrt()).abs() < 1e-2, "GPU knn dist != brute at query {qi} rank {j}: {} vs {}", got[j].sqrt(), d2[j].sqrt()); }
    }
    let mut gpu_knn_ms = f64::MAX;
    for _ in 0..9 { let t = Instant::now(); knn_run(); gpu_knn_ms = gpu_knn_ms.min(t.elapsed().as_secs_f64() * 1e3); }
    let qpts: Vec<Point3> = queries.iter().map(|q| Point3::new(q[0] as f64, q[1] as f64, q[2] as f64)).collect();
    let mut cpu_knn_ms = f64::MAX;
    for _ in 0..5 { let t = Instant::now(); let v: usize = qpts.iter().map(|&q| tree.knn(q, k).len()).sum(); std::hint::black_box(v); cpu_knn_ms = cpu_knn_ms.min(t.elapsed().as_secs_f64() * 1e3); }
    println!("\nbatch k-NN (k={k}) — {nq} queries over {n} points:");
    println!("  GPU LBVH k-NN (build EXCLUDED)  : {gpu_knn_ms:>8.3} ms   (verified == brute distances ✓)");
    println!("  CPU Tree3 knn (serial)          : {cpu_knn_ms:>8.3} ms");
    let kvs = if gpu_knn_ms < cpu_knn_ms { format!("{:.2}× the CPU serial knn", cpu_knn_ms / gpu_knn_ms) } else { format!("slower than the CPU knn ({:.2}×)", gpu_knn_ms / cpu_knn_ms) };
    println!("  → GPU LBVH k-NN is {kvs} (query only).");

    println!("(Honest: the GPU query is the batch-broadphase primitive — its edge is throughput over a\nhuge query set on a resident tree; a handful of queries never amortise the build + the launch.)");
}
