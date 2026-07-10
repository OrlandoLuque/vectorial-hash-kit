//! gpu_visibility_bench — the GPU **line-of-sight** offload, measured. The clean
//! GPU case: occluders are STATIC, so their BVH is built once (no per-frame
//! rebuild that sinks the moving broad-phase). For M viewer→target segments,
//! traverse an LBVH of occluder boxes in a compute shader and flag the ones
//! blocked before the target. Verified against the CPU (`Polyhedron3::segment_hit`
//! over the same occluders) by blocked-count, then timed.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_visibility_bench --release
//! ```
use std::time::Instant;
use vectorial_hash::{Point3, Polyhedron3};

const WORLD: f64 = 4000.0;
const H: f32 = 30.0; // occluder half-extent (a wall/prop chunk)

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn u(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

// ---- CPU LBVH over occluder centroids (Morton sort + highest-bit split), packed
// as 2×vec4<f32>: a.xyz=lo, a.w=left(bits), b.xyz=hi, b.w=right(bits, leaf=0xFFFF_FFFF).
fn morton3(x: u32, y: u32, z: u32) -> u64 { fn s(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v } s(x as u64) | (s(y as u64) << 1) | (s(z as u64) << 2) }
fn mcell(v: f64) -> u32 { ((v / WORLD) * (1u32 << 21) as f64) as u32 & ((1 << 21) - 1) }
fn build_lbvh(pts: &[[f32; 3]]) -> (Vec<[f32; 8]>, Vec<[f32; 4]>, u32) {
    let mut keyed: Vec<(u64, [f32; 4])> = pts.iter().map(|p| (morton3(mcell(p[0] as f64), mcell(p[1] as f64), mcell(p[2] as f64)), [p[0], p[1], p[2], 0.0])).collect();
    keyed.sort_unstable_by_key(|k| k.0);
    let sorted: Vec<[f32; 4]> = keyed.iter().map(|k| k.1).collect();
    let mut nodes = Vec::with_capacity(pts.len() * 2);
    let root = build_range(&keyed, 0, keyed.len(), &mut nodes);
    (nodes, sorted, root)
}
fn build_range(k: &[(u64, [f32; 4])], lo: usize, hi: usize, nodes: &mut Vec<[f32; 8]>) -> u32 {
    if hi - lo == 1 { let p = k[lo].1; let id = nodes.len() as u32;
        // leaf AABB = centroid ± H so the node bounds cover the actual occluder box
        nodes.push([p[0] - H, p[1] - H, p[2] - H, f32::from_bits(lo as u32), p[0] + H, p[1] + H, p[2] + H, f32::from_bits(u32::MAX)]); return id; }
    let (first, last) = (k[lo].0, k[hi - 1].0);
    let split = if first == last { (lo + hi) / 2 } else { let mask = 1u64 << (63 - (first ^ last).leading_zeros()); let (mut a, mut b) = (lo, hi - 1); while b - a > 1 { let m = (a + b) / 2; if k[m].0 & mask == 0 { a = m; } else { b = m; } } b };
    let l = build_range(k, lo, split, nodes); let r = build_range(k, split, hi, nodes);
    let (ln, rn) = (nodes[l as usize], nodes[r as usize]); let id = nodes.len() as u32;
    nodes.push([ln[0].min(rn[0]), ln[1].min(rn[1]), ln[2].min(rn[2]), f32::from_bits(l), ln[4].max(rn[4]), ln[5].max(rn[5]), ln[6].max(rn[6]), f32::from_bits(r)]);
    id
}

const LOS_SHADER: &str = r#"
struct Node { a: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var<storage, read> occ: array<vec4<f32>>;      // occluder centroids
@group(0) @binding(1) var<storage, read> segs: array<vec4<f32>>;     // pairs: [2*i]=viewer.xyz, [2*i+1]=target.xyz
@group(0) @binding(2) var<storage, read_write> blocked: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;                 // x=n_occ, y=n_seg, z=root
@group(0) @binding(4) var<storage, read> nodes: array<Node>;
const HH: f32 = 30.0;
// segment a + t*d, t∈[0,1], vs AABB [lo,hi]: entry t, or -1 on miss.
fn seg_aabb(a: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    var t0 = 0.0; var t1 = 1.0;
    for (var i = 0; i < 3; i = i + 1) {
        if (abs(d[i]) < 1e-9) { if (a[i] < lo[i] || a[i] > hi[i]) { return -1.0; } }
        else { var tn = (lo[i] - a[i]) / d[i]; var tf = (hi[i] - a[i]) / d[i];
            if (tn > tf) { let tmp = tn; tn = tf; tf = tmp; }
            t0 = max(t0, tn); t1 = min(t1, tf); if (t0 > t1) { return -1.0; } }
    }
    return t0;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x; if (qi >= params.y) { return; }
    let a = segs[2u * qi].xyz; let b = segs[2u * qi + 1u].xyz; let d = b - a;
    var hit = 0u; var stack: array<u32, 64>; var sp = 0; stack[0] = params.z; sp = 1;
    loop {
        if (sp == 0) { break; } sp = sp - 1; let node = nodes[stack[sp]];
        if (seg_aabb(a, d, node.a.xyz, node.b.xyz) < 0.0) { continue; } // segment misses this subtree
        let right = bitcast<u32>(node.b.w);
        if (right == 0xFFFFFFFFu) {
            let c = occ[bitcast<u32>(node.a.w)].xyz;
            let t = seg_aabb(a, d, c - vec3<f32>(HH), c + vec3<f32>(HH));
            if (t >= 0.0 && t < 1.0) { hit = 1u; break; } // an occluder blocks before the target
        } else if (sp < 62) { stack[sp] = bitcast<u32>(node.a.w); sp = sp + 1; stack[sp] = right; sp = sp + 1; }
    }
    blocked[qi] = hit;
}
"#;

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None } }

fn main() { pollster::block_on(run()); }

async fn run() {
    let n_occ = std::env::var("VIS_OCC").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000usize);
    let n_seg = std::env::var("VIS_SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000usize);
    println!("GPU line-of-sight bench | {n_occ} static occluders (LBVH built once) | {n_seg} viewer→target segments\n");

    let mut r = Rng(7);
    let occ: Vec<[f32; 3]> = (0..n_occ).map(|_| [(r.u() * WORLD) as f32, (r.u() * WORLD) as f32, (r.u() * WORLD) as f32]).collect();
    // segments span the world so many cross occluders
    let segs: Vec<[f32; 4]> = (0..n_seg * 2).map(|_| [(r.u() * WORLD) as f32, (r.u() * WORLD) as f32, (r.u() * WORLD) as f32, 0.0]).collect();

    let t_build = Instant::now();
    let (nodes, occ_sorted, root) = build_lbvh(&occ);
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

    // ---- wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let want_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let feats = if want_ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: feats, required_limits: adapter.limits() }, None).await.expect("device");
    use wgpu::util::DeviceExt;
    let occ_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&occ_sorted), usage: wgpu::BufferUsages::STORAGE });
    let seg_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&segs), usage: wgpu::BufferUsages::STORAGE });
    let blk_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n_seg * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let params = [n_occ as u32, n_seg as u32, root, 0u32];
    let par_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM });
    let nodes_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&nodes), usage: wgpu::BufferUsages::STORAGE });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n_seg * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(LOS_SHADER.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[bgl_entry(0, true), bgl_entry(1, true), bgl_entry(2, false),
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }, bgl_entry(4, true)] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: "main", compilation_options: Default::default() });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: occ_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: seg_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: blk_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: par_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: nodes_b.as_entire_binding() }] });

    let dispatch = || { let mut enc = device.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups((n_seg as u32).div_ceil(64), 1, 1); }
        enc.copy_buffer_to_buffer(&blk_b, 0, &readback, 0, (n_seg * 4) as u64); queue.submit(Some(enc.finish())); };
    dispatch(); device.poll(wgpu::Maintain::Wait);
    let mut wall = f64::MAX; for _ in 0..20 { let t = Instant::now(); dispatch(); device.poll(wgpu::Maintain::Wait); wall = wall.min(t.elapsed().as_secs_f64()); }
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {}); device.poll(wgpu::Maintain::Wait);
    let gpu: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec(); readback.unmap();
    let gpu_blocked: u64 = gpu.iter().map(|&b| b as u64).sum();

    // ---- CPU reference: segment_hit over the occluders (a Tree3 of centroids prunes)
    use vectorial_hash::{Aabb, Positioned3, Tree3};
    #[derive(Clone, Copy)] struct O(Point3, usize);
    impl Positioned3 for O { fn position(&self) -> Point3 { self.0 } }
    let polys: Vec<Polyhedron3> = occ.iter().map(|c| { let (cx, cy, cz) = (c[0] as f64, c[1] as f64, c[2] as f64); let h = H as f64;
        Polyhedron3::from_corners([Point3::new(cx - h, cy - h, cz - h), Point3::new(cx + h, cy - h, cz - h), Point3::new(cx + h, cy + h, cz - h), Point3::new(cx - h, cy + h, cz - h),
            Point3::new(cx - h, cy - h, cz + h), Point3::new(cx + h, cy - h, cz + h), Point3::new(cx + h, cy + h, cz + h), Point3::new(cx - h, cy + h, cz + h)]) }).collect();
    let otree = Tree3::bulk_load(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), 8, occ.iter().enumerate().map(|(i, c)| O(Point3::new(c[0] as f64, c[1] as f64, c[2] as f64), i)).collect::<Vec<_>>());
    let t_cpu = Instant::now();
    let mut cpu_blocked = 0u64;
    for q in 0..n_seg { let a = Point3::new(segs[2 * q][0] as f64, segs[2 * q][1] as f64, segs[2 * q][2] as f64); let b = Point3::new(segs[2 * q + 1][0] as f64, segs[2 * q + 1][1] as f64, segs[2 * q + 1][2] as f64);
        // capsule radius = the box circumradius (H·√3) so the cull can't miss a box
        // the segment clips at a corner (its centroid may be >H from the spine).
        let seg = vectorial_hash::Segment3::new(a, b, H as f64 * 3f64.sqrt());
        let mut blk = false;
        for o in otree.cull(&seg) { if let Some(t) = polys[o.1].segment_hit(a, b) { if t < 1.0 { blk = true; break; } } }
        if blk { cpu_blocked += 1; }
    }
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;

    // ---- verify (f32-vs-f64 boundary tolerance) + report
    let diff = (gpu_blocked as i64 - cpu_blocked as i64).abs();
    println!("\nblocked segments: GPU {gpu_blocked}  CPU {cpu_blocked}  (Δ {diff} = f32-vs-f64 boundary grazes)");
    assert!(diff <= (cpu_blocked as i64 / 5_000).max(64), "GPU LoS diverged from CPU ({gpu_blocked} vs {cpu_blocked})");
    println!("\n{:>28} | {:>12}", "stage", "ms");
    println!("{:>28} | {:>12.2}", "LBVH build (once, static)", build_ms);
    println!("{:>28} | {:>12.3}  ({:.0} ns/segment)", "GPU LoS dispatch (wall)", wall * 1e3, wall / n_seg as f64 * 1e9);
    println!("{:>28} | {:>12.2}  ({:.0} ns/segment)", "CPU LoS (serial)", cpu_ms, cpu_ms * 1e6 / n_seg as f64);
    println!("\nStatic occluders → the BVH is built ONCE ({build_ms:.1} ms), so unlike the moving\nbroad-phase the GPU pays no per-frame rebuild — {:.0}x the serial CPU on the LoS pass.", cpu_ms / (wall * 1e3));
}
