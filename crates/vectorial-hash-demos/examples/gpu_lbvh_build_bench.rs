//! gpu_lbvh_build_bench — building an **LBVH entirely on the GPU**, the thing that
//! would push the *moving*-data broad-phase crossover past 1 M (a per-frame CPU
//! rebuild's N·log N sort is what loses to the linear keep-index today).
//!
//! Stage 1 (this commit): the **front half** — 30-bit Morton codes computed on the
//! GPU, then a **stable key-value radix sort** (codes + primitive index) so the
//! payload follows its key. All GPU-resident, no CPU in the loop. Verified purely
//! against GPU-produced data (the un-sorted codes read back), so it doesn't lean
//! on CPU↔GPU float determinism:
//!   - sorted codes are non-decreasing,
//!   - `sorted_code[i] == unsorted_code[val[i]]` (payload carries the right key),
//!   - `val[..n]` is a permutation of `0..n`,
//!   - and `sorted == a CPU sort of the (GPU-produced) un-sorted codes`.
//! Stage 2 (next): Karras hierarchy + AABB refit + a traversal-vs-brute verify.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_lbvh_build_bench --release   # LBVH_N
//! ```
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } }

const TILE: usize = 256;

// Morton: quantise each axis to 10 bits (points are generated in [0,1024)³, so a
// truncation `u32(p) & 1023` — deterministic, no lo/hi normalisation).
const MORTON_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       pts:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals:  array<u32>;
@group(0) @binding(3) var<uniform>             mp:    vec4<u32>; // x = n

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
    let xi = min(u32(p.x), 1023u);
    let yi = min(u32(p.y), 1023u);
    let zi = min(u32(p.z), 1023u);
    codes[i] = part1by2(xi) | (part1by2(yi) << 1u) | (part1by2(zi) << 2u);
    vals[i]  = i;
}
"#;

// Key-value radix (the tonight sort primitive, extended to carry a u32 payload).
const RADIX_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       src_k:     array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_k:     array<u32>;
@group(0) @binding(2) var<storage, read>       src_v:     array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_v:     array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_hist: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_off:  array<u32>;
@group(0) @binding(6) var<uniform>             p:         vec4<u32>; // x=n2, y=shift, z=num_tiles

const RADIX: u32 = 16u;
var<workgroup> lhist:  array<atomic<u32>, 16>;
var<workgroup> ldig:   array<u32, 256>;
var<workgroup> wtotal: array<u32, 16>;
var<workgroup> wbase:  array<u32, 16>;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    if (li < RADIX) { atomicStore(&lhist[li], 0u); }
    workgroupBarrier();
    let i = gid.x;
    if (i < p.x) { atomicAdd(&lhist[(src_k[i] >> p.y) & 0xFu], 1u); }
    workgroupBarrier();
    if (li < RADIX) { tile_hist[wid.x * RADIX + li] = atomicLoad(&lhist[li]); }
}

@compute @workgroup_size(16)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let d = lid.x;
    let nt = p.z;
    var tot = 0u;
    for (var t = 0u; t < nt; t = t + 1u) { tot = tot + tile_hist[t * RADIX + d]; }
    wtotal[d] = tot;
    workgroupBarrier();
    if (d == 0u) { var acc = 0u; for (var k = 0u; k < RADIX; k = k + 1u) { wbase[k] = acc; acc = acc + wtotal[k]; } }
    workgroupBarrier();
    var run = wbase[d];
    for (var t = 0u; t < nt; t = t + 1u) { tile_off[t * RADIX + d] = run; run = run + tile_hist[t * RADIX + d]; }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    let i = gid.x;
    var key = 0xFFFFFFFFu;
    var d = 0xFu;
    if (i < p.x) { key = src_k[i]; d = (key >> p.y) & 0xFu; }
    ldig[li] = d;
    workgroupBarrier();
    if (i < p.x) {
        var rank = 0u;
        for (var j = 0u; j < li; j = j + 1u) { if (ldig[j] == d) { rank = rank + 1u; } }
        let pos = tile_off[wid.x * RADIX + d] + rank;
        dst_k[pos] = key;
        dst_v[pos] = src_v[i];
    }
}
"#;

fn main() { pollster::block_on(run()); }

async fn run() {
    let n = std::env::var("LBVH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_048_576usize);
    let num_tiles = n.div_ceil(TILE);
    let n2 = num_tiles * TILE;
    println!("GPU LBVH build — stage 1 (Morton + key-value radix) | {n} points (padded to {n2}, {num_tiles} tiles)\n");

    let mut r = Rng(0x1234_ABCD);
    // points in [0,1024)³ (Morton quantises by truncation).
    let pts: Vec<[f32; 4]> = (0..n).map(|_| [
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0,
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0,
        (r.next() % 1024) as f32 + (r.next() & 0xffff) as f32 / 65536.0, 0.0]).collect();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.expect("device");

    let sbuf = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pts_b = sbuf((n2 * 16) as u64);
    let (code_a, code_b) = (sbuf((n2 * 4) as u64), sbuf((n2 * 4) as u64));
    let (val_a, val_b)   = (sbuf((n2 * 4) as u64), sbuf((n2 * 4) as u64));
    let code_orig = sbuf((n2 * 4) as u64); // un-sorted codes, kept for the verify
    let hist_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (num_tiles * 16 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let off_b  = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (num_tiles * 16 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let mp_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let rp_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let rd = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let (rb_orig, rb_code, rb_val) = (rd((n2 * 4) as u64), rd((n2 * 4) as u64), rd((n2 * 4) as u64));

    let sto = |ro: bool| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None };
    let uni = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    let ent = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };

    // Morton pipeline
    let m_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(MORTON_SRC.into()) });
    let m_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(0, sto(true)), ent(1, sto(false)), ent(2, sto(false)), ent(3, uni)] });
    let m_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&m_bgl], push_constant_ranges: &[] });
    let m_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&m_pl), module: &m_mod, entry_point: "morton", compilation_options: Default::default() });
    let m_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &m_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: pts_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: code_a.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: val_a.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: mp_b.as_entire_binding() }] });

    // Radix pipelines
    let r_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RADIX_SRC.into()) });
    let r_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        ent(0, sto(true)), ent(1, sto(false)), ent(2, sto(true)), ent(3, sto(false)), ent(4, sto(false)), ent(5, sto(false)), ent(6, uni)] });
    let r_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&r_bgl], push_constant_ranges: &[] });
    let rpipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&r_pl), module: &r_mod, entry_point: ep, compilation_options: Default::default() });
    let (hist_p, scan_p, scat_p) = (rpipe("histogram"), rpipe("scan"), rpipe("scatter"));
    let rbg = |sk: &wgpu::Buffer, dk: &wgpu::Buffer, sv: &wgpu::Buffer, dv: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &r_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: sk.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dk.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: sv.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: dv.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: hist_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: off_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: rp_b.as_entire_binding() }] });
    let bg_ab = rbg(&code_a, &code_b, &val_a, &val_b);
    let bg_ba = rbg(&code_b, &code_a, &val_b, &val_a);

    let wg = num_tiles as u32;
    let build = || {
        queue.write_buffer(&pts_b, 0, bytemuck::cast_slice(&pts));
        // pad codes to MAX (sort to tail); Morton fills [0,n).
        queue.write_buffer(&code_a, 0, bytemuck::cast_slice(&vec![u32::MAX; n2]));
        queue.write_buffer(&mp_b, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &m_bg, &[]); c.set_pipeline(&m_pipe); c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
        enc.copy_buffer_to_buffer(&code_a, 0, &code_orig, 0, (n2 * 4) as u64); // snapshot before the sort
        queue.submit(Some(enc.finish()));
        // key-value radix, 8 passes ping-pong (final in code_a/val_a).
        for pass in 0..8u32 {
            let g = if pass % 2 == 0 { &bg_ab } else { &bg_ba };
            queue.write_buffer(&rp_b, 0, bytemuck::cast_slice(&[n2 as u32, pass * 4, num_tiles as u32, 0u32]));
            let mut enc = device.create_command_encoder(&Default::default());
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&hist_p); c.dispatch_workgroups(wg, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scan_p); c.dispatch_workgroups(1, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scat_p); c.dispatch_workgroups(wg, 1, 1); }
            queue.submit(Some(enc.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
    };

    build();
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&code_orig, 0, &rb_orig, 0, (n2 * 4) as u64);
    enc.copy_buffer_to_buffer(&code_a, 0, &rb_code, 0, (n2 * 4) as u64);
    enc.copy_buffer_to_buffer(&val_a, 0, &rb_val, 0, (n2 * 4) as u64);
    queue.submit(Some(enc.finish()));
    for b in [&rb_orig, &rb_code, &rb_val] { b.slice(..).map_async(wgpu::MapMode::Read, |_| {}); }
    device.poll(wgpu::Maintain::Wait);
    let orig: Vec<u32> = bytemuck::cast_slice(&rb_orig.slice(..).get_mapped_range()).to_vec();
    let code: Vec<u32> = bytemuck::cast_slice(&rb_code.slice(..).get_mapped_range()).to_vec();
    let val:  Vec<u32> = bytemuck::cast_slice(&rb_val.slice(..).get_mapped_range()).to_vec();

    // ---- verify (all against GPU-produced data — no CPU Morton recompute) ----
    for i in 1..n { assert!(code[i - 1] <= code[i], "codes not sorted at {i}"); }
    for i in 0..n { assert_eq!(code[i], orig[val[i] as usize], "payload {i} carries the wrong key"); }
    let mut perm = val[..n].to_vec(); perm.sort_unstable();
    assert!(perm.iter().copied().eq(0..n as u32), "val is not a permutation of 0..n");
    let mut cpu = orig[..n].to_vec(); cpu.sort_unstable();
    assert!(cpu == code[..n], "GPU sort != CPU sort of the same codes");
    assert!(code[n..].iter().all(|&x| x == u32::MAX), "padding not at the tail");

    let mut ms = f64::MAX;
    for _ in 0..7 { let t = Instant::now(); build(); ms = ms.min(t.elapsed().as_secs_f64() * 1e3); }

    println!("\nverified: Morton codes computed + key-value sorted on the GPU, payload intact ({n} points) ✓");
    println!("  full GPU-resident Morton+sort: {ms:.2} ms (min of 7) — {:.0} Mpts/s", n as f64 / (ms / 1e3) / 1e6);
    println!("\n(Front half of the on-GPU LBVH build: points → 30-bit Morton (GPU) → stable key-value\nradix (GPU), no CPU round-trip. Stage 2 adds the Karras hierarchy + AABB refit on top of\nthese sorted (code, index) pairs, then a traversal-vs-brute verify + a build-per-frame number.)");
}
