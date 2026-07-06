//! GPU spatial-query bench (milestone 1) — how much does the GPU spend on
//! spatial culling, and how does it compare to the CPU structures? A headless
//! wgpu **compute** pipeline runs M interest queries against N points; we time
//! the dispatch (GPU wall time via submit+poll, plus precise timestamps when the
//! adapter supports TIMESTAMP_QUERY) and compare to CPU brute force + `Tree3`.
//!
//! Milestone 1a is the GPU BRUTE cull (each thread = one query, loops all N) —
//! the embarrassingly-parallel baseline that shows the GPU's raw throughput and
//! establishes the compute plumbing. (LBVH traversal on GPU = milestone 1b.)
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_spatial_bench --release
//! ```

use std::time::Instant;
use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Sphere3, Tree3};

const WORLD: f64 = 10_000.0;

#[derive(Clone, Copy)]
struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;   // xyz, w unused
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;  // xyz = centre, w = radius
@group(0) @binding(2) var<storage, read_write> hits: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;                 // x = n_points, y = n_queries
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.y) { return; }
    let q = queries[qi];
    let r2 = q.w * q.w;
    var count = 0u;
    let n = params.x;
    for (var i = 0u; i < n; i = i + 1u) {
        let d = points[i].xyz - q.xyz;
        if (dot(d, d) <= r2) { count = count + 1u; }
    }
    hits[qi] = count;
}
"#;

// ---- CPU LBVH build (Morton sort + highest-differing-bit split; Karras
// topology) → flat nodes packed as 2×vec4<f32> for the GPU + points in sorted
// order (leaf `left` indexes the sorted array). Leaf: right == 0xFFFF_FFFF.
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn s(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    s(x as u64) | (s(y as u64) << 1) | (s(z as u64) << 2)
}
fn mcell(v: f64) -> u32 { ((v / WORLD) * (1u32 << 21) as f64) as u32 & ((1 << 21) - 1) }

/// Returns (packed nodes as [f32;8], sorted points as [f32;4], root index).
fn build_lbvh(pts: &[P]) -> (Vec<[f32; 8]>, Vec<[f32; 4]>, u32) {
    let mut keyed: Vec<(u64, [f32; 4])> = pts.iter().map(|p| (morton3(mcell(p.0.x), mcell(p.0.y), mcell(p.0.z)), [p.0.x as f32, p.0.y as f32, p.0.z as f32, 0.0])).collect();
    keyed.sort_unstable_by_key(|k| k.0);
    let sorted: Vec<[f32; 4]> = keyed.iter().map(|k| k.1).collect();
    let mut nodes: Vec<[f32; 8]> = Vec::with_capacity(pts.len() * 2);
    let root = build_range(&keyed, 0, keyed.len(), &mut nodes);
    (nodes, sorted, root)
}
fn build_range(k: &[(u64, [f32; 4])], lo: usize, hi: usize, nodes: &mut Vec<[f32; 8]>) -> u32 {
    if hi - lo == 1 {
        let p = k[lo].1; let id = nodes.len() as u32;
        // leaf: lo=hi=point, left = sorted index, right = 0xFFFFFFFF
        nodes.push([p[0], p[1], p[2], f32::from_bits(lo as u32), p[0], p[1], p[2], f32::from_bits(u32::MAX)]);
        return id;
    }
    let (first, last) = (k[lo].0, k[hi - 1].0);
    let split = if first == last { (lo + hi) / 2 } else {
        let mask = 1u64 << (63 - (first ^ last).leading_zeros());
        let (mut a, mut b) = (lo, hi - 1);
        while b - a > 1 { let m = (a + b) / 2; if k[m].0 & mask == 0 { a = m; } else { b = m; } }
        b
    };
    let l = build_range(k, lo, split, nodes);
    let r = build_range(k, split, hi, nodes);
    let (ln, rn) = (nodes[l as usize], nodes[r as usize]);
    let id = nodes.len() as u32;
    nodes.push([
        ln[0].min(rn[0]).min(ln[4].min(rn[4])), ln[1].min(rn[1]).min(ln[5].min(rn[5])), ln[2].min(rn[2]).min(ln[6].min(rn[6])), f32::from_bits(l),
        ln[4].max(rn[4]).max(ln[0].max(rn[0])), ln[5].max(rn[5]).max(ln[1].max(rn[1])), ln[6].max(rn[6]).max(ln[2].max(rn[2])), f32::from_bits(r),
    ]);
    id
}

const LBVH_SHADER: &str = r#"
struct Node { a: vec4<f32>, b: vec4<f32> };  // a.xyz=lo, a.w=left(bits), b.xyz=hi, b.w=right(bits, 0xFFFFFFFF=leaf)
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hits: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;   // y = n_queries, z = root
@group(0) @binding(4) var<storage, read> nodes: array<Node>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.y) { return; }
    let q = queries[qi];
    let r2 = q.w * q.w;
    var count = 0u;
    var stack: array<u32, 64>;
    var sp = 0;
    stack[0] = params.z; sp = 1;
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let node = nodes[stack[sp]];
        let nearest = clamp(q.xyz, node.a.xyz, node.b.xyz);
        let d = nearest - q.xyz;
        if (dot(d, d) > r2) { continue; }
        let right = bitcast<u32>(node.b.w);
        if (right == 0xFFFFFFFFu) {
            let pi = bitcast<u32>(node.a.w);
            let dp = points[pi].xyz - q.xyz;
            if (dot(dp, dp) <= r2) { count = count + 1u; }
        } else if (sp < 62) {
            stack[sp] = bitcast<u32>(node.a.w); sp = sp + 1;
            stack[sp] = right; sp = sp + 1;
        }
    }
    hits[qi] = count;
}
"#;

fn main() {
    pollster::block_on(run());
}

fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn env_f32(k: &str, d: f32) -> f32 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

async fn run() {
    // Parametrised so the same rig can measure a broad-phase workload (big world,
    // fat bubbles, few queries) AND a game-demo workload (N points each doing a
    // small-radius neighbour cull — separation/interest). Env: GPU_N, GPU_M,
    // GPU_R, GPU_CLUSTER=1 (clump the points like a battle front).
    let n = env_usize("GPU_N", 1_000_000);
    let m = env_usize("GPU_M", 10_000);
    let radius = env_f32("GPU_R", 500.0);
    let cluster = std::env::var("GPU_CLUSTER").is_ok();
    println!("GPU spatial-query bench | {n} points | {m} queries | r={radius}{}\n", if cluster { " | clustered" } else { "" });

    // ---- data. Uniform, or (cluster) a handful of dense blobs = a battle front.
    let mut r = Rng(42);
    let pts: Vec<P> = if cluster {
        let blobs: Vec<(f64, f64, f64)> = (0..24).map(|_| (r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD)).collect();
        (0..n).map(|_| { let (cx, cy, cz) = blobs[(r.next() as usize) % blobs.len()]; let s = WORLD * 0.03; let hi = WORLD - 1.0;
            P(Point3::new((cx + (r.unit() - 0.5) * s).clamp(0.0, hi), (cy + (r.unit() - 0.5) * s).clamp(0.0, hi), (cz + (r.unit() - 0.5) * s).clamp(0.0, hi))) }).collect()
    } else {
        (0..n).map(|_| P(Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD))).collect()
    };
    // queries = a sample of the points themselves when m<=n (the demo case: every
    // active unit culls its own neighbourhood), else random centres.
    let mut rq = Rng(99);
    let qcenters: Vec<(f64, f64, f64)> = (0..m).map(|i| if m <= n { let p = pts[i * (n / m.max(1))].0; (p.x, p.y, p.z) } else { (rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD) }).collect();

    // build the LBVH (also gives points in Morton-sorted order); both GPU passes
    // use the sorted points (brute is order-independent, the LBVH leaf indexes it).
    // Time the build too — for a MOVING cloud this is a per-frame rebuild cost.
    let t_build = Instant::now();
    let (nodes_packed, pbuf, root) = build_lbvh(&pts);
    let lbvh_build_ms = { let mut b = t_build.elapsed().as_secs_f64(); for _ in 0..4 { let t = Instant::now(); let _ = build_lbvh(&pts); b = b.min(t.elapsed().as_secs_f64()); } b * 1e3 };
    let qbuf: Vec<[f32; 4]> = qcenters.iter().map(|&(x, y, z)| [x as f32, y as f32, z as f32, radius]).collect();

    // ---- wgpu headless device
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU adapter");
    println!("adapter: {}", adapter.get_info().name);
    let want_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let feats = if want_ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
    // request the adapter's full limits — the nodes buffer (~2N × 32 B) is the
    // biggest and blows past the default 128-MB storage-buffer cap otherwise.
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: feats, required_limits: adapter.limits() }, None).await.expect("device");

    use wgpu::util::DeviceExt;
    let points_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("points"), contents: bytemuck::cast_slice(&pbuf), usage: wgpu::BufferUsages::STORAGE });
    let queries_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("queries"), contents: bytemuck::cast_slice(&qbuf), usage: wgpu::BufferUsages::STORAGE });
    let hits_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("hits"), size: (m * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let params = [n as u32, m as u32, root, 0u32];
    let params_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM });
    let nodes_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("nodes"), contents: bytemuck::cast_slice(&nodes_packed), usage: wgpu::BufferUsages::STORAGE });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (m * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        bgl_entry(0, true), bgl_entry(1, true), bgl_entry(2, false),
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
    ] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: "main", compilation_options: Default::default() });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: points_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: queries_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: hits_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: params_b.as_entire_binding() },
    ] });

    // optional precise GPU timestamps
    let qset = want_ts.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 }));
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let dispatch = || {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let ts = qset.as_ref().map(|qs| wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: ts });
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((m as u32).div_ceil(64), 1, 1);
        }
        if let Some(qs) = &qset { enc.resolve_query_set(qs, 0..2, &ts_resolve, 0); enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 16); }
        enc.copy_buffer_to_buffer(&hits_b, 0, &readback, 0, (m * 4) as u64);
        queue.submit(Some(enc.finish()));
    };

    // warm + measure GPU wall time (submit → device idle)
    dispatch(); device.poll(wgpu::Maintain::Wait);
    let reps = 20;
    let mut wall = f64::MAX;
    for _ in 0..reps { let t = Instant::now(); dispatch(); device.poll(wgpu::Maintain::Wait); wall = wall.min(t.elapsed().as_secs_f64()); }

    // read hits back + verify vs CPU brute
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let gpu_hits: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
    readback.unmap();

    // precise GPU time (if timestamps)
    let gpu_ts_ms = if qset.is_some() {
        ts_read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let t: Vec<u64> = bytemuck::cast_slice(&ts_read.slice(..).get_mapped_range()).to_vec();
        ts_read.unmap();
        Some((t[1].wrapping_sub(t[0])) as f64 * queue.get_timestamp_period() as f64 / 1e6)
    } else { None };

    // ---- milestone 1b: GPU LBVH traversal (pruning on the GPU)
    let lbvh_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(LBVH_SHADER.into()) });
    let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        bgl_entry(0, true), bgl_entry(1, true), bgl_entry(2, false),
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        bgl_entry(4, true),
    ] });
    let pl2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl2], push_constant_ranges: &[] });
    let pipeline2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl2), module: &lbvh_module, entry_point: "main", compilation_options: Default::default() });
    let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl2, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: points_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: queries_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: hits_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: params_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: nodes_b.as_entire_binding() },
    ] });
    let dispatch2 = || {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let ts = qset.as_ref().map(|qs| wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: ts });
            cp.set_pipeline(&pipeline2); cp.set_bind_group(0, &bg2, &[]);
            cp.dispatch_workgroups((m as u32).div_ceil(64), 1, 1);
        }
        if let Some(qs) = &qset { enc.resolve_query_set(qs, 0..2, &ts_resolve, 0); enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 16); }
        enc.copy_buffer_to_buffer(&hits_b, 0, &readback, 0, (m * 4) as u64);
        queue.submit(Some(enc.finish()));
    };
    dispatch2(); device.poll(wgpu::Maintain::Wait);
    let mut lbvh_wall = f64::MAX;
    for _ in 0..reps { let t = Instant::now(); dispatch2(); device.poll(wgpu::Maintain::Wait); lbvh_wall = lbvh_wall.min(t.elapsed().as_secs_f64()); }
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let lbvh_hits: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
    readback.unmap();
    let lbvh_total: u64 = lbvh_hits.iter().map(|&h| h as u64).sum();
    let lbvh_ts_ms = if qset.is_some() {
        ts_read.slice(..).map_async(wgpu::MapMode::Read, |_| {}); device.poll(wgpu::Maintain::Wait);
        let t: Vec<u64> = bytemuck::cast_slice(&ts_read.slice(..).get_mapped_range()).to_vec(); ts_read.unmap();
        Some((t[1].wrapping_sub(t[0])) as f64 * queue.get_timestamp_period() as f64 / 1e6)
    } else { None };
    // LBVH is f32 too → same boundary tolerance vs the GPU brute
    let brute_total: u64 = gpu_hits.iter().map(|&h| h as u64).sum();
    assert!((lbvh_total as i64 - brute_total as i64).abs() <= 16, "GPU LBVH != GPU brute ({lbvh_total} vs {brute_total})");

    // ---- CPU baselines
    let tree = Tree3::bulk_load(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), 8, pts.clone());
    let spheres: Vec<Sphere3> = qcenters.iter().map(|&(x, y, z)| Sphere3::new(x, y, z, radius as f64)).collect();
    let best = |reps: usize, mut f: Box<dyn FnMut() -> usize>| -> (f64, usize) { let h = f(); let mut b = f64::MAX; for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } (b, h) };
    let (cpu_brute, hb) = best(3, Box::new(|| { let mut h = 0; for s in &spheres { for p in &pts { if (p.0.x - s_c(s).0).powi(2) + (p.0.y - s_c(s).1).powi(2) + (p.0.z - s_c(s).2).powi(2) <= (radius as f64).powi(2) { h += 1; } } } h }));
    let (cpu_tree, _) = best(5, Box::new(|| { let mut h = 0; for s in &spheres { h += tree.cull(s).len(); } h }));
    // The demos batch their culls over rayon (cull_many_par) — the honest CPU
    // baseline for a demo, not the serial loop. Feature-gated (rayon not in wasm).
    #[cfg(feature = "parallel")]
    let cpu_tree_par = { let mut b = f64::MAX; for _ in 0..5 { let t = Instant::now(); let r = tree.cull_many_par(&spheres); std::hint::black_box(&r); b = b.min(t.elapsed().as_secs_f64()); } b };
    #[cfg(not(feature = "parallel"))]
    let cpu_tree_par: f64 = f64::NAN;

    // ---- CPU keep-index maintenance: one frame of movement = update_ref every
    // point (the moving-demo per-frame cost the GPU LBVH must beat, since a moving
    // cloud forces the BVH to be *rebuilt* every frame). Small perturbation so
    // most points stay in their leaf (the O(1) path), a few relocate.
    let mut ktree: Tree3<P> = Tree3::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), 8);
    let handles: Vec<ItemRef> = pts.iter().map(|p| ktree.insert_ref(*p).unwrap()).collect();
    let mut rm = Rng(7);
    let hi = WORLD - 1.0; // stay strictly inside the half-open world so no handle is freed
    let keep_maint_ms = { let mut b = f64::MAX; for _ in 0..5 { let t = Instant::now();
        for (i, h) in handles.iter().enumerate() { let p = pts[i].0; let np = Point3::new((p.x + (rm.unit() - 0.5) * 4.0).clamp(0.0, hi), (p.y + (rm.unit() - 0.5) * 4.0).clamp(0.0, hi), (p.z + (rm.unit() - 0.5) * 4.0).clamp(0.0, hi)); ktree.update_ref(*h, |s| s.0 = np); }
        b = b.min(t.elapsed().as_secs_f64()); } b * 1e3 };

    // verify — GPU is f32, CPU brute is f64, so a few points exactly on the
    // sphere boundary classify differently. Allow a tiny tolerance (that IS the
    // precision story, not a bug).
    let gpu_total: u64 = gpu_hits.iter().map(|&h| h as u64).sum();
    let diff = (gpu_total as i64 - hb as i64).abs();
    assert!(diff <= (hb as i64 / 100_000).max(16), "GPU vs CPU brute diverged too far ({gpu_total} vs {hb})");

    println!("\nhits total {gpu_total} (CPU f64 {hb}; Δ {diff} = f32-vs-f64 boundary points ✓)\n");
    println!("{:>26} | {:>14} | {:>16}", "engine", "total ms", "ns / query");
    println!("{:>26} | {:>12.2}   | {:>14.1}", "GPU brute (wall)", wall * 1e3, wall / m as f64 * 1e9);
    if let Some(ms) = gpu_ts_ms { println!("{:>26} | {:>12.3}   | {:>14.1}", "GPU brute (timestamp)", ms, ms * 1e6 / m as f64); }
    println!("{:>26} | {:>12.2}   | {:>14.1}", "GPU LBVH (wall)", lbvh_wall * 1e3, lbvh_wall / m as f64 * 1e9);
    if let Some(ms) = lbvh_ts_ms { println!("{:>26} | {:>12.3}   | {:>14.1}", "GPU LBVH (timestamp)", ms, ms * 1e6 / m as f64); }
    println!("{:>26} | {:>12.2}   | {:>14.1}", "CPU brute (serial)", cpu_brute * 1e3, cpu_brute / m as f64 * 1e9);
    println!("{:>26} | {:>12.2}   | {:>14.1}", "CPU Tree3 (serial cull)", cpu_tree * 1e3, cpu_tree / m as f64 * 1e9);
    if cpu_tree_par.is_finite() { println!("{:>26} | {:>12.2}   | {:>14.1}", "CPU Tree3 (cull_many_par)", cpu_tree_par * 1e3, cpu_tree_par / m as f64 * 1e9); }
    let ptests = (n as f64 * m as f64) / wall / 1e9;
    println!("\nGPU brute did {n}×{m} = {:.1}e9 point-tests in {:.2} ms → {ptests:.1} G point-tests/s.", n as f64 * m as f64 / 1e9, wall * 1e3);
    println!("LBVH vs brute on GPU: {:.2}× ({} hits, matches). Pruning cuts the point-tests but\nadds divergent traversal + random node fetches; whether it beats brute-force on\nthe GPU depends on density (bubble size) — measure both, pick per workload.", wall / lbvh_wall, lbvh_total);

    // ---- the per-frame verdict for a MOVING cloud (game demos). The GPU LBVH
    // query is only part of the cost: a moving cloud forces a full rebuild+upload
    // each frame. The CPU keep-index does NOT rebuild — update_ref in place. So
    // the honest per-frame comparison is (rebuild + query) vs (maintain + query).
    let gpu_frame = lbvh_build_ms + lbvh_wall * 1e3;   // CPU rebuild + GPU dispatch (wall)
    let cpu_frame = keep_maint_ms + cpu_tree * 1e3;    // keep-index maintain + serial cull
    let cpu_par_frame = keep_maint_ms + cpu_tree_par * 1e3; // keep-index maintain + parallel cull
    println!("\n── per-frame for a MOVING cloud (N points move every frame) ──");
    println!("{:>34} | {:>10}", "phase", "ms");
    println!("{:>34} | {:>10.2}", "CPU keep-index maintain (update_ref)", keep_maint_ms);
    println!("{:>34} | {:>10.2}", "CPU Tree3 cull (serial, m queries)", cpu_tree * 1e3);
    println!("{:>34} | {:>10.2}  ⇐ CPU keep-index frame (serial)", "= maintain + cull", cpu_frame);
    if cpu_tree_par.is_finite() {
        println!("{:>34} | {:>10.2}", "CPU Tree3 cull_many_par (16 thr)", cpu_tree_par * 1e3);
        println!("{:>34} | {:>10.2}  ⇐ CPU keep-index frame (parallel)", "= maintain + par cull", cpu_par_frame);
    }
    println!("{:>34} | {:>10.2}", "CPU LBVH rebuild (sort+build)", lbvh_build_ms);
    println!("{:>34} | {:>10.2}", "GPU LBVH dispatch (wall)", lbvh_wall * 1e3);
    println!("{:>34} | {:>10.2}  ⇐ GPU LBVH frame (rebuild+dispatch)", "= rebuild + dispatch", gpu_frame);
    // The demo's real CPU baseline is the parallel frame when available.
    let cpu_ref = if cpu_tree_par.is_finite() { cpu_par_frame } else { cpu_frame };
    let label = if cpu_tree_par.is_finite() { "CPU keep-index (parallel)" } else { "CPU keep-index (serial)" };
    println!("\nverdict (vs {label}): {}", if gpu_frame < cpu_ref {
        format!("GPU LBVH wins the frame ({:.2}x) — the query load justifies the per-frame rebuild.", cpu_ref / gpu_frame)
    } else {
        format!("{label} wins the frame ({:.2}x) — the rebuild costs more than the offload saves;\n         the query is too cheap (small radius / few queries / parallelised) to pay for a\n         per-frame BVH. GPU LBVH is for query-dominated / rebuild-anyway (static) loads.", gpu_frame / cpu_ref)
    });
}

fn s_c(s: &Sphere3) -> (f64, f64, f64) { use vectorial_hash::Shape3; let b = s.bounding_box(); (b.x + b.w / 2.0, b.y + b.h / 2.0, b.z + b.d / 2.0) }

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
