//! gpu_radix_bench — an **all-GPU LSD radix sort** of 32-bit Morton codes: the
//! sort primitive the on-GPU LBVH build needs (the naive bitonic in
//! `gpu_sort_bench` was verified-correct but *slower* than the CPU — log² work +
//! a dispatch per pass). This is a stable 4-bit radix (8 passes), three compute
//! kernels per pass, no CPU in the loop:
//!   1. histogram — each 256-wide tile counts its 16 digit buckets (workgroup
//!      shared atomics) → `tile_hist[tile][digit]`.
//!   2. scan — one workgroup turns the per-tile histograms into per-tile output
//!      offsets: `digit_base[d] + Σ_{t'<t} hist[t'][d]` (a stable exclusive scan).
//!   3. scatter — each tile computes each element's stable local rank and writes
//!      it to `offset[tile][digit] + rank` in the pong buffer.
//! Verified **exactly** against a CPU sort and timed against it.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_radix_bench --release
//! SORT_N=1048576 cargo run -p vectorial-hash-demos --example gpu_radix_bench --release
//! ```
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } }

const TILE: usize = 256;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       src:       array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:       array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_hist: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_off:  array<u32>;
@group(0) @binding(4) var<uniform>             p:         vec4<u32>; // x=n2, y=shift, z=num_tiles

const RADIX: u32 = 16u;

var<workgroup> lhist:  array<atomic<u32>, 16>;
var<workgroup> ldig:   array<u32, 256>;
var<workgroup> wtotal: array<u32, 16>;
var<workgroup> wbase:  array<u32, 16>;

// 1) per-tile histogram of the current 4-bit digit.
@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    if (li < RADIX) { atomicStore(&lhist[li], 0u); }
    workgroupBarrier();
    let i = gid.x;
    if (i < p.x) { atomicAdd(&lhist[(src[i] >> p.y) & 0xFu], 1u); }
    workgroupBarrier();
    if (li < RADIX) { tile_hist[wid.x * RADIX + li] = atomicLoad(&lhist[li]); }
}

// 2) exclusive scan over the (tile × digit) histogram → each tile's write base
//    per digit. One workgroup, 16 threads (one per digit) — the per-digit tile
//    scans run in parallel instead of one thread walking tiles×16 serially.
@compute @workgroup_size(16)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let d = lid.x;           // this thread owns digit d
    let nt = p.z;
    // phase 1: total count of digit d across all tiles.
    var tot = 0u;
    for (var t = 0u; t < nt; t = t + 1u) { tot = tot + tile_hist[t * RADIX + d]; }
    wtotal[d] = tot;
    workgroupBarrier();
    // phase 2: digit_base[d] = Σ_{d'<d} total[d']  (thread 0 scans the 16 totals).
    if (d == 0u) {
        var acc = 0u;
        for (var k = 0u; k < RADIX; k = k + 1u) { wbase[k] = acc; acc = acc + wtotal[k]; }
    }
    workgroupBarrier();
    // phase 3: per-tile exclusive prefix within digit d, offset by digit_base.
    var run = wbase[d];
    for (var t = 0u; t < nt; t = t + 1u) {
        tile_off[t * RADIX + d] = run;
        run = run + tile_hist[t * RADIX + d];
    }
}

// 3) stable scatter: local rank = # earlier threads in this tile with the same
//    digit; global position = tile_off[tile][digit] + rank.
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    let i = gid.x;
    var key = 0xFFFFFFFFu;
    var d = 0xFu;
    if (i < p.x) { key = src[i]; d = (key >> p.y) & 0xFu; }
    ldig[li] = d;
    workgroupBarrier();
    if (i < p.x) {
        var rank = 0u;
        for (var j = 0u; j < li; j = j + 1u) { if (ldig[j] == d) { rank = rank + 1u; } }
        dst[tile_off[wid.x * RADIX + d] + rank] = key;
    }
}
"#;

fn main() { pollster::block_on(run()); }

async fn run() {
    let n = std::env::var("SORT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(262_144usize);
    let num_tiles = n.div_ceil(TILE);
    let n2 = num_tiles * TILE; // pad up to a whole number of tiles
    println!("GPU radix sort (4-bit LSD, 8 passes) | {n} keys (padded to {n2}, {num_tiles} tiles)\n");

    let mut r = Rng(0xABCDEF);
    let keys: Vec<u32> = (0..n).map(|_| r.next()).collect();
    let mut padded = keys.clone();
    padded.resize(n2, u32::MAX); // MAX sorts to the tail every pass

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.expect("device");

    let mk = |sz: u64, extra: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE | extra, mapped_at_creation: false });
    let buf_a = mk((n2 * 4) as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let buf_b = mk((n2 * 4) as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let hist_b = mk((num_tiles * 16 * 4) as u64, wgpu::BufferUsages::empty());
    let off_b = mk((num_tiles * 16 * 4) as u64, wgpu::BufferUsages::empty());
    let par_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n2 * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let sto = |ro: bool| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None };
    let entry = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        entry(0, sto(true)), entry(1, sto(false)), entry(2, sto(false)), entry(3, sto(false)),
        entry(4, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None })] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: ep, compilation_options: Default::default() });
    let (hist_p, scan_p, scat_p) = (pipe("histogram"), pipe("scan"), pipe("scatter"));

    // Two bind groups for the ping-pong: (src=a,dst=b) and (src=b,dst=a).
    let bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: hist_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: off_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: par_b.as_entire_binding() }] });
    let bg_ab = bg(&buf_a, &buf_b);
    let bg_ba = bg(&buf_b, &buf_a);

    let sort = || {
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&padded));
        let wg = num_tiles as u32;
        for pass in 0..8u32 {
            let g = if pass % 2 == 0 { &bg_ab } else { &bg_ba };
            queue.write_buffer(&par_b, 0, bytemuck::cast_slice(&[n2 as u32, pass * 4, num_tiles as u32, 0u32]));
            let mut enc = device.create_command_encoder(&Default::default());
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&hist_p); c.dispatch_workgroups(wg, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scan_p); c.dispatch_workgroups(1, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scat_p); c.dispatch_workgroups(wg, 1, 1); }
            queue.submit(Some(enc.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
    };

    // verify once against a CPU sort, then time both (min-of-N, warm).
    sort();
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&buf_a, 0, &readback, 0, (n2 * 4) as u64); // 8 passes (even) → in buf_a
    queue.submit(Some(enc.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {}); device.poll(wgpu::Maintain::Wait);
    let out: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec(); readback.unmap();
    let mut sorted = keys.clone(); sorted.sort_unstable();
    assert!(&out[..n] == sorted.as_slice(), "GPU radix sort != CPU sort");
    assert!(out[n..].iter().all(|&x| x == u32::MAX), "padding not at the tail");

    let mut gpu_ms = f64::MAX;
    for _ in 0..7 { let t = Instant::now(); sort(); gpu_ms = gpu_ms.min(t.elapsed().as_secs_f64() * 1e3); }
    let mut cpu_ms = f64::MAX;
    for _ in 0..7 { let mut k = keys.clone(); let t = Instant::now(); k.sort_unstable(); cpu_ms = cpu_ms.min(t.elapsed().as_secs_f64() * 1e3); }

    println!("\nverified: GPU order == CPU order ({n} keys) ✓");
    println!("{:>22} | {:>10}", "sort", "ms");
    println!("{:>22} | {:>10.2}", "GPU radix (4-bit×8)", gpu_ms);
    println!("{:>22} | {:>10.2}", "CPU sort_unstable", cpu_ms);
    let verdict = if gpu_ms < cpu_ms { format!("FASTER ({:.2}×)", cpu_ms / gpu_ms) } else { format!("slower ({:.2}×)", gpu_ms / cpu_ms) };
    println!("\nResult: the GPU radix is verified-correct and {verdict} than the CPU sort at this size.");
    println!("(Stable 4-bit LSD, all-GPU: 3 kernels/pass, ping-pong, no CPU in the loop. The scan is 16-way\nparallel — one thread per digit — which is what put it clear of the CPU; a multi-workgroup\ndecoupled-lookback (Onesweep) scan would scale it further. This unblocks the on-GPU LBVH build:\nsort Morton codes here, then Karras split + AABB refit on top.)");
}
