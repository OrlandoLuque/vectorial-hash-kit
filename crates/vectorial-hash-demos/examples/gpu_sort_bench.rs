//! gpu_sort_bench — a **GPU bitonic sort** of Morton codes, the first half of an
//! on-GPU LBVH build (sort keyed by Z-order, then Karras + refit). A parallel
//! compare-exchange network in a compute shader: log²(N) dispatches, each thread
//! swaps element `i` with `i^j` to keep the running bitonic order. Verified
//! **exactly** against a CPU sort (identical order) and timed against it.
//!
//! This is the enabling primitive for the GPU-side build noted in the LBVH work:
//! move the *N log N* sort — the cost that sinks a per-frame CPU rebuild — onto
//! the GPU. (32-bit keys here; a full build would emulate u64 as vec2<u32>.)
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example gpu_sort_bench --release
//! ```
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u32 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; (x >> 16) as u32 } }

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> p: vec4<u32>; // x = n (padded), y = k (block), z = j (stride)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= p.x) { return; }
    let ixj = i ^ p.z;
    if (ixj > i) {                     // the lower thread of each pair does the swap
        let ascending = (i & p.y) == 0u;
        let a = data[i]; let b = data[ixj];
        if ((ascending && a > b) || (!ascending && a < b)) { data[i] = b; data[ixj] = a; }
    }
}
"#;

fn main() { pollster::block_on(run()); }

async fn run() {
    let n = std::env::var("SORT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_048_576usize); // 2^20
    let n2 = n.next_power_of_two();
    println!("GPU bitonic sort | {n} keys (padded to {n2})\n");

    let mut r = Rng(0xABCDEF);
    let mut keys: Vec<u32> = (0..n).map(|_| r.next()).collect();
    let mut padded = keys.clone();
    padded.resize(n2, u32::MAX); // padding sorts to the end

    // ---- wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.expect("no GPU");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.expect("device");
    let data_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n2 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let par_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n2 * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: "main", compilation_options: Default::default() });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: data_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: par_b.as_entire_binding() }] });

    // one full bitonic sort: for each block size k, for each stride j, one dispatch
    let sort = || {
        queue.write_buffer(&data_b, 0, bytemuck::cast_slice(&padded));
        let mut k = 2usize;
        while k <= n2 {
            let mut j = k / 2;
            while j > 0 {
                queue.write_buffer(&par_b, 0, bytemuck::cast_slice(&[n2 as u32, k as u32, j as u32, 0u32]));
                let mut enc = device.create_command_encoder(&Default::default());
                { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups((n2 as u32).div_ceil(64), 1, 1); }
                queue.submit(Some(enc.finish()));
                j /= 2;
            }
            k *= 2;
        }
        device.poll(wgpu::Maintain::Wait);
    };

    let t = Instant::now(); sort(); let gpu_ms = t.elapsed().as_secs_f64() * 1e3;

    // read back the first n (padding is at the tail)
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&data_b, 0, &readback, 0, (n2 * 4) as u64);
    queue.submit(Some(enc.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {}); device.poll(wgpu::Maintain::Wait);
    let out: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec(); readback.unmap();

    // ---- verify against a CPU sort
    let t = Instant::now(); keys.sort_unstable(); let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let gpu_head = &out[..n];
    assert!(gpu_head == keys.as_slice(), "GPU bitonic sort != CPU sort");
    assert!(out[n..].iter().all(|&x| x == u32::MAX), "padding not at the tail");

    println!("\nverified: GPU order == CPU order ({n} keys) ✓");
    println!("{:>22} | {:>10}", "sort", "ms");
    println!("{:>22} | {:>10.2}", "GPU bitonic", gpu_ms);
    println!("{:>22} | {:>10.2}", "CPU sort_unstable", cpu_ms);
    let lg = n2.trailing_zeros();
    let verdict = if gpu_ms < cpu_ms { "faster" } else { "SLOWER" };
    println!("\nHonest result: the naive bitonic is verified-correct but {verdict} than the CPU sort.");
    println!("It runs log²(n) ≈ {} compare-exchange passes (vs the CPU's n·log n), and each pass is a\nseparate dispatch+submit — the log² work factor plus per-pass sync overhead sink it at this size.\nSo the on-GPU LBVH build wants a RADIX sort (or a workgroup-shared-memory bitonic that sorts\nlocal blocks per dispatch), not this. Useful to know before building the full GPU-side path.", lg * (lg + 1) / 2);
}
