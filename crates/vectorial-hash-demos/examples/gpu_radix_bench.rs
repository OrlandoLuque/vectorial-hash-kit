//! gpu_radix_bench — an **all-GPU LSD radix sort** of 32-bit Morton codes: the
//! sort primitive the on-GPU LBVH build needs (the naive bitonic in
//! `gpu_sort_bench` was verified-correct but *slower* than the CPU — log² work +
//! a dispatch per pass). Stable radix, three compute kernels per pass, no CPU in
//! the loop. (1) histogram: each 256-wide tile counts its digit buckets (workgroup
//! shared atomics) into `tile_hist[tile][digit]`. (2) scan: a HIERARCHICAL exclusive
//! scan (reduce per block of tiles, scan the blocks, add) turns the per-tile
//! histograms into each tile's write base per digit, so the per-digit tile prefix is
//! parallel over blocks, not one workgroup walking every tile. (3) scatter: each tile
//! computes each element's stable local rank and writes it to `offset[tile][digit] +
//! rank` in the pong buffer. Verified **exactly** against a CPU sort and timed too.
//!
//! This runs TWO radix widths and compares them — the "fewer global passes" lever
//! (the direction Onesweep pushes): **4-bit → 8 passes** vs **8-bit → 4 passes**.
//! (A *true* single-pass Onesweep — one decoupled-lookback pass — needs guaranteed
//! inter-workgroup forward progress, which WebGPU/WGSL does NOT promise, so it can
//! deadlock; the 8-bit/4-pass width is the portable step in that direction.)
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_radix_bench --release
//! SORT_N=1048576 cargo run -p vectorial-hash-demos --example gpu_radix_bench --release
//! ```
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } }

const TILE: usize = 256;

// The kernels, parametrised by radix width. `__RADIX__`/`__MASK__` are substituted
// per variant so one template serves both 4-bit (16 buckets) and 8-bit (256).
const TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read>       src:       array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:       array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_hist: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_off:  array<u32>;
@group(0) @binding(4) var<uniform>             p:         vec4<u32>; // x=n2, y=shift, z=num_tiles, w=num_blocks
@group(0) @binding(5) var<storage, read_write> block_tot: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_off: array<u32>;

const RADIX: u32 = __RADIX__u;
const BLOCK: u32 = 512u; // tiles per scan block (hierarchical scan)

var<workgroup> lhist:  array<atomic<u32>, __RADIX__>;
var<workgroup> ldig:   array<u32, 256>;
var<workgroup> wtotal: array<u32, __RADIX__>;
var<workgroup> wbase:  array<u32, __RADIX__>;

// 1) per-tile histogram of the current digit.
@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x;
    if (li < RADIX) { atomicStore(&lhist[li], 0u); }
    workgroupBarrier();
    let i = gid.x;
    if (i < p.x) { atomicAdd(&lhist[(src[i] >> p.y) & __MASK__u], 1u); }
    workgroupBarrier();
    if (li < RADIX) { tile_hist[wid.x * RADIX + li] = atomicLoad(&lhist[li]); }
}

// 2) hierarchical exclusive scan over the (tile × digit) histogram.
// 2a) each block: intra-block exclusive prefix into tile_off + the block total.
@compute @workgroup_size(__RADIX__)
fn scan_reduce(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x; let nt = p.z; let b = wid.x;
    let lo = b * BLOCK; let hi = min(lo + BLOCK, nt);
    var run = 0u;
    for (var t = lo; t < hi; t = t + 1u) { tile_off[t * RADIX + d] = run; run = run + tile_hist[t * RADIX + d]; }
    block_tot[b * RADIX + d] = run;
}
// 2b) one workgroup: digit_base + exclusive prefix over the (few) block totals.
@compute @workgroup_size(__RADIX__)
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
// 2c) each block: offset its tiles' prefixes by the block base.
@compute @workgroup_size(__RADIX__)
fn scan_add(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x; let nt = p.z; let b = wid.x;
    let base = block_off[b * RADIX + d];
    let lo = b * BLOCK; let hi = min(lo + BLOCK, nt);
    for (var t = lo; t < hi; t = t + 1u) { tile_off[t * RADIX + d] = tile_off[t * RADIX + d] + base; }
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
    var d = __MASK__u;
    if (i < p.x) { key = src[i]; d = (key >> p.y) & __MASK__u; }
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

// Build + run the radix sort at a given digit width, verify == CPU, return min-of-7 ms.
fn variant(device: &wgpu::Device, queue: &wgpu::Queue, keys: &[u32], bits: u32, num_tiles: usize, n2: usize) -> f64 {
    let n = keys.len();
    let radix = 1usize << bits;
    let passes = 32 / bits;
    let src = TEMPLATE.replace("__RADIX__", &radix.to_string()).replace("__MASK__", &format!("0x{:X}", radix - 1));

    let mut padded = keys.to_vec();
    padded.resize(n2, u32::MAX);

    let mk = |sz: u64, extra: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz.max(4), usage: wgpu::BufferUsages::STORAGE | extra, mapped_at_creation: false });
    let buf_a = mk((n2 * 4) as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let buf_b = mk((n2 * 4) as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let hist_b = mk((num_tiles * radix * 4) as u64, wgpu::BufferUsages::empty());
    let off_b = mk((num_tiles * radix * 4) as u64, wgpu::BufferUsages::empty());
    let num_blocks = num_tiles.div_ceil(512);
    let block_tot_b = mk((num_blocks * radix * 4) as u64, wgpu::BufferUsages::empty());
    let block_off_b = mk((num_blocks * radix * 4) as u64, wgpu::BufferUsages::empty());
    let par_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n2 * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let sto = |ro: bool| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None };
    let entry = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        entry(0, sto(true)), entry(1, sto(false)), entry(2, sto(false)), entry(3, sto(false)),
        entry(4, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
        entry(5, sto(false)), entry(6, sto(false))] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: ep, compilation_options: Default::default() });
    let (hist_p, reduce_p, top_p, add_p, scat_p) = (pipe("histogram"), pipe("scan_reduce"), pipe("scan_top"), pipe("scan_add"), pipe("scatter"));

    let bg = |s: &wgpu::Buffer, d: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: hist_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: off_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: par_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: block_tot_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: block_off_b.as_entire_binding() }] });
    let bg_ab = bg(&buf_a, &buf_b);
    let bg_ba = bg(&buf_b, &buf_a);

    let sort = || {
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&padded));
        let wg = num_tiles as u32; let nb = num_blocks as u32;
        for pass in 0..passes {
            let g = if pass % 2 == 0 { &bg_ab } else { &bg_ba };
            queue.write_buffer(&par_b, 0, bytemuck::cast_slice(&[n2 as u32, pass * bits, num_tiles as u32, num_blocks as u32]));
            let mut enc = device.create_command_encoder(&Default::default());
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&hist_p); c.dispatch_workgroups(wg, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&reduce_p); c.dispatch_workgroups(nb, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&top_p); c.dispatch_workgroups(1, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&add_p); c.dispatch_workgroups(nb, 1, 1); }
            { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scat_p); c.dispatch_workgroups(wg, 1, 1); }
            queue.submit(Some(enc.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
    };

    // verify (passes is even for both 4- and 8-bit → result lands in buf_a)
    sort();
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&buf_a, 0, &readback, 0, (n2 * 4) as u64);
    queue.submit(Some(enc.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {}); device.poll(wgpu::Maintain::Wait);
    let out: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec(); readback.unmap();
    let mut sorted = keys.to_vec(); sorted.sort_unstable();
    assert!(&out[..n] == sorted.as_slice(), "GPU radix ({bits}-bit) != CPU sort");

    let mut ms = f64::MAX;
    for _ in 0..7 { let t = Instant::now(); sort(); ms = ms.min(t.elapsed().as_secs_f64() * 1e3); }
    ms
}

async fn run() {
    let n = std::env::var("SORT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(262_144usize);
    let num_tiles = n.div_ceil(TILE);
    let n2 = num_tiles * TILE;
    println!("GPU radix sort — 4-bit (8 passes) vs 8-bit (4 passes) | {n} keys (padded to {n2}, {num_tiles} tiles)\n");

    let mut r = Rng(0xABCDEF);
    let keys: Vec<u32> = (0..n).map(|_| r.next()).collect();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.expect("device");

    let ms4 = variant(&device, &queue, &keys, 4, num_tiles, n2);
    let ms8 = variant(&device, &queue, &keys, 8, num_tiles, n2);
    let mut cpu_ms = f64::MAX;
    for _ in 0..7 { let mut k = keys.clone(); let t = Instant::now(); k.sort_unstable(); cpu_ms = cpu_ms.min(t.elapsed().as_secs_f64() * 1e3); }

    println!("\nverified: both GPU widths == CPU sort ({n} keys) ✓");
    println!("{:>24} | {:>9} | {:>12}", "sort", "ms", "vs CPU");
    println!("{:>24} | {:>9.2} | {:>10.2}×", "GPU radix 4-bit ×8", ms4, cpu_ms / ms4);
    println!("{:>24} | {:>9.2} | {:>10.2}×", "GPU radix 8-bit ×4", ms8, cpu_ms / ms8);
    println!("{:>24} | {:>9.2} |", "CPU sort_unstable", cpu_ms);
    let w = if ms8 < ms4 { format!("8-bit/4-pass is {:.2}× FASTER than 4-bit/8-pass", ms4 / ms8) } else { format!("8-bit/4-pass is {:.2}× slower than 4-bit/8-pass", ms8 / ms4) };
    println!("\nResult: {w} — halving the passes (fewer global read/scatter round-trips) vs\nthe bigger 256-bucket histogram/scan per pass. Both are verified-correct + stable.\n(A true single-pass Onesweep — one decoupled-lookback scan — would cut it further, but\nneeds guaranteed inter-workgroup forward progress, which WebGPU/WGSL does not promise;\nthe 8-bit width is the portable step in that direction.)");
}
