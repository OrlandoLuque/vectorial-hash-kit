//! adaptive_broadphase — watch the moving-data **index-maintenance crossover**
//! live, and an adaptive controller ride it. A cloud of N points; a slider sets
//! what **fraction moves** each frame (movers are bright, the rest dim). Every
//! frame we measure BOTH per-frame index strategies on the same cloud:
//!   • **CPU keep-index** — `update_ref` just the movers (the rest cost nothing);
//!   • **GPU rebuild** — a whole LBVH from scratch on the GPU (Morton → stable
//!     key-value radix → Karras → atomic AABB refit), all GPU-resident.
//! The two on-screen bars are those costs. Keep is ~linear in the moving fraction
//! (it skips the unmoved); the GPU rebuild is flat. So as you raise the fraction
//! the bars **cross** at f*, and the **ADAPTIVE** controller switches keep→GPU —
//! with a **hysteresis** dead-band so it doesn't flip-flop at the boundary.
//!
//! This is the maintenance-cost story from `examples/gpu_lbvh_build_bench` made
//! visible (the query is a separate axis — see docs/GPU.md).
//!
//! Keys: `,` `.` moving fraction · `A` toggle auto/adaptive vs a forced mode
//! (`1` keep · `2` GPU) · drag orbit · scroll zoom. `ADAPT_N` sets the count.
//!
//! `cargo run -p vectorial-hash-demos --bin adaptive_broadphase --release`
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::{event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};
use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Tree3};

const WORLD: f32 = 1024.0;
const TILE: usize = 256;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u32 << 24) as f32 } }

struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] struct Cam { vp: [[f32; 4]; 4] }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] struct UiVertex { pos: [f32; 2], color: [f32; 4] }

fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let (x0, x1) = (px / sw * 2.0 - 1.0, (px + w) / sw * 2.0 - 1.0);
    let (y0, y1) = (1.0 - py / sh * 2.0, 1.0 - (py + h) / sh * 2.0);
    let q = |x, y| UiVertex { pos: [x, y], color };
    v.extend_from_slice(&[q(x0, y0), q(x1, y0), q(x1, y1), q(x0, y0), q(x1, y1), q(x0, y1)]);
}

#[derive(Clone, Copy, PartialEq)]
enum Mode { Keep, Gpu }

// ---- GPU build shaders (from examples/gpu_lbvh_build_bench, verbatim) ----
const MORTON_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       pts:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals:  array<u32>;
@group(0) @binding(3) var<uniform>             mp:    vec4<u32>;
fn part1by2(n: u32) -> u32 { var x = n & 0x000003ffu; x = (x | (x << 16u)) & 0x030000ffu; x = (x | (x << 8u)) & 0x0300f00fu; x = (x | (x << 4u)) & 0x030c30c3u; x = (x | (x << 2u)) & 0x09249249u; return x; }
@compute @workgroup_size(256)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= mp.x) { return; }
    let p = pts[i];
    codes[i] = part1by2(min(u32(p.x), 1023u)) | (part1by2(min(u32(p.y), 1023u)) << 1u) | (part1by2(min(u32(p.z), 1023u)) << 2u);
    vals[i] = i;
}
"#;
const RADIX_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> src_k: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_k: array<u32>;
@group(0) @binding(2) var<storage, read> src_v: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_v: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_hist: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_off: array<u32>;
@group(0) @binding(6) var<uniform> p: vec4<u32>;
const RADIX: u32 = 16u;
var<workgroup> lhist: array<atomic<u32>, 16>;
var<workgroup> ldig: array<u32, 256>;
var<workgroup> wtotal: array<u32, 16>;
var<workgroup> wbase: array<u32, 16>;
@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x; if (li < RADIX) { atomicStore(&lhist[li], 0u); } workgroupBarrier();
    let i = gid.x; if (i < p.x) { atomicAdd(&lhist[(src_k[i] >> p.y) & 0xFu], 1u); } workgroupBarrier();
    if (li < RADIX) { tile_hist[wid.x * RADIX + li] = atomicLoad(&lhist[li]); }
}
@compute @workgroup_size(16)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let d = lid.x; let nt = p.z; var tot = 0u;
    for (var t = 0u; t < nt; t = t + 1u) { tot = tot + tile_hist[t * RADIX + d]; }
    wtotal[d] = tot; workgroupBarrier();
    if (d == 0u) { var acc = 0u; for (var k = 0u; k < RADIX; k = k + 1u) { wbase[k] = acc; acc = acc + wtotal[k]; } } workgroupBarrier();
    var run = wbase[d];
    for (var t = 0u; t < nt; t = t + 1u) { tile_off[t * RADIX + d] = run; run = run + tile_hist[t * RADIX + d]; }
}
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.x; let i = gid.x; var key = 0xFFFFFFFFu; var d = 0xFu;
    if (i < p.x) { key = src_k[i]; d = (key >> p.y) & 0xFu; } ldig[li] = d; workgroupBarrier();
    if (i < p.x) { var rank = 0u; for (var j = 0u; j < li; j = j + 1u) { if (ldig[j] == d) { rank = rank + 1u; } } let pos = tile_off[wid.x * RADIX + d] + rank; dst_k[pos] = key; dst_v[pos] = src_v[i]; }
}
"#;
const KARRAS_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> val: array<u32>;
@group(0) @binding(2) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> children: array<u32>;
@group(0) @binding(4) var<storage, read_write> parent: array<u32>;
@group(0) @binding(5) var<storage, read_write> aabb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> kp: vec4<u32>;
fn delta(i: i32, j: i32, n: i32) -> i32 { if (j < 0 || j >= n) { return -1; } let ci = codes[u32(i)]; let cj = codes[u32(j)]; if (ci == cj) { return 32 + i32(countLeadingZeros(u32(i) ^ u32(j))); } return i32(countLeadingZeros(ci ^ cj)); }
@compute @workgroup_size(256)
fn karras(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(kp.x); let i = i32(gid.x); if (i >= n - 1) { return; }
    let d = select(-1, 1, delta(i, i + 1, n) > delta(i, i - 1, n));
    let dmin = delta(i, i - d, n); var lmax = 2;
    while (delta(i, i + lmax * d, n) > dmin) { lmax = lmax * 2; }
    var l = 0; var t = lmax / 2;
    while (t >= 1) { if (delta(i, i + (l + t) * d, n) > dmin) { l = l + t; } t = t / 2; }
    let j = i + l * d; let first = min(i, j); let last = max(i, j);
    let cp = delta(first, last, n); var split = first; var step = last - first;
    loop { step = (step + 1) / 2; let ns = split + step; if (ns < last && delta(first, ns, n) > cp) { split = ns; } if (step <= 1) { break; } }
    let leaf_base = u32(n - 1);
    var lc = u32(split); if (split == first) { lc = leaf_base + u32(split); }
    var rc = u32(split + 1); if (split + 1 == last) { rc = leaf_base + u32(split + 1); }
    children[u32(i) * 2u] = lc; children[u32(i) * 2u + 1u] = rc; parent[lc] = u32(i); parent[rc] = u32(i);
}
@compute @workgroup_size(256)
fn leaf_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = kp.x; let s = gid.x; if (s >= n) { return; }
    let leaf = (n - 1u) + s; let p = pts[val[s]].xyz;
    aabb[leaf * 2u] = vec4<f32>(p, 0.0); aabb[leaf * 2u + 1u] = vec4<f32>(p, 0.0);
}
@compute @workgroup_size(256)
fn refit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = kp.x; let s = gid.x; if (s >= n) { return; }
    var node = parent[(n - 1u) + s];
    loop {
        if (atomicAdd(&flags[node], 1u) == 0u) { return; }
        let lc = children[node * 2u]; let rc = children[node * 2u + 1u];
        let mn = min(aabb[lc * 2u].xyz, aabb[rc * 2u].xyz); let mx = max(aabb[lc * 2u + 1u].xyz, aabb[rc * 2u + 1u].xyz);
        aabb[node * 2u] = vec4<f32>(mn, 0.0); aabb[node * 2u + 1u] = vec4<f32>(mx, 0.0);
        if (node == 0u) { return; } node = parent[node];
    }
}
"#;
const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) col: vec3<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec4<f32>, @location(1) mover: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,1.0));
    var o: VOut; let clip = cam.vp * vec4<f32>(pos.xyz, 1.0);
    let sz = select(2.2, 4.0, mover == 1u);
    o.clip = vec4<f32>(clip.xy + corners[vi] * sz * clip.w / 700.0, clip.z, clip.w);
    o.col = select(vec3<f32>(0.22, 0.30, 0.55), vec3<f32>(0.35, 0.95, 1.0), mover == 1u);
    return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.col, 0.9); }
"#;
const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;

fn main() { pollster::block_on(run()); }

fn st(b: u32, ro: bool) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None }, count: None } }
fn unif(b: u32) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None } }

async fn run() {
    let n: usize = std::env::var("ADAPT_N").ok().and_then(|s| s.parse().ok()).unwrap_or(150_000);
    let num_tiles = n.div_ceil(TILE);
    let n2 = num_tiles * TILE;
    let ni = (n - 1).max(1);

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("adaptive_broadphase").with_inner_size(winit::dpi::LogicalSize::new(1400, 900)).build(&event_loop).unwrap());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.unwrap();
    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: wgpu::PresentMode::AutoNoVsync, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
    surface.configure(&device, &config);

    // ---- points + velocities + CPU keep tree ----
    let mut r = Rng(0xC0FFEE);
    let mut pos: Vec<[f32; 4]> = (0..n).map(|_| [r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD, 0.0]).collect();
    let mut vel: Vec<[f32; 3]> = (0..n).map(|_| [(r.unit() - 0.5) * 240.0, (r.unit() - 0.5) * 240.0, (r.unit() - 0.5) * 240.0]).collect();
    let mut tree = Tree3::new(Aabb::new(0.0, 0.0, 0.0, WORLD as f64, WORLD as f64, WORLD as f64), 16);
    let handles: Vec<ItemRef> = pos.iter().map(|p| tree.insert_ref(P(Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))).unwrap()).collect();
    let mut flags_cpu: Vec<u32> = vec![0; n]; // mover flag for render

    // ---- GPU build buffers ----
    let sb = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pts_b = sb((n2 * 16) as u64);
    let (code_a, code_b, val_a, val_b) = (sb((n2 * 4) as u64), sb((n2 * 4) as u64), sb((n2 * 4) as u64), sb((n2 * 4) as u64));
    let hist_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (num_tiles * 16 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let off_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (num_tiles * 16 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let children_b = sb((2 * ni * 4) as u64);
    let parent_b = sb(((2 * n - 1) * 4) as u64);
    let aabb_b = sb((2 * (2 * n - 1) * 16) as u64);
    let flags_b = sb((ni * 4) as u64);
    let ub = |sz: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let (mp_b, rp_b, kp_b) = (ub(16), ub(16), ub(16));

    // render buffers
    let render_pos_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 16) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let render_flag_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = ub(std::mem::size_of::<Cam>() as u64);
    let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (256 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // ---- pipelines ----
    let ent = |b, ty| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    let uni_ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    let m_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(MORTON_SRC.into()) });
    let m_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[st(0, true), st(1, false), st(2, false), unif(3)] });
    let m_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&m_bgl], push_constant_ranges: &[] });
    let m_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&m_pl), module: &m_mod, entry_point: "morton", compilation_options: Default::default() });
    let m_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &m_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: pts_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: code_a.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: val_a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: mp_b.as_entire_binding() }] });

    let r_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RADIX_SRC.into()) });
    let r_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[st(0, true), st(1, false), st(2, true), st(3, false), st(4, false), st(5, false), unif(6)] });
    let r_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&r_bgl], push_constant_ranges: &[] });
    let rpipe = |ep| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&r_pl), module: &r_mod, entry_point: ep, compilation_options: Default::default() });
    let (hist_p, scan_p, scat_p) = (rpipe("histogram"), rpipe("scan"), rpipe("scatter"));
    let rbg = |sk: &wgpu::Buffer, dk: &wgpu::Buffer, sv: &wgpu::Buffer, dv: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &r_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: sk.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: dk.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: sv.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: dv.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: hist_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: off_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 6, resource: rp_b.as_entire_binding() }] });
    let bg_ab = rbg(&code_a, &code_b, &val_a, &val_b);
    let bg_ba = rbg(&code_b, &code_a, &val_b, &val_a);

    let k_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(KARRAS_SRC.into()) });
    let k_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[st(0, true), st(1, true), st(2, true), st(3, false), st(4, false), st(5, false), st(6, false), ent(7, uni_ty)] });
    let k_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&k_bgl], push_constant_ranges: &[] });
    let kpipe = |ep| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&k_pl), module: &k_mod, entry_point: ep, compilation_options: Default::default() });
    let (karras_p, leaf_p, refit_p) = (kpipe("karras"), kpipe("leaf_init"), kpipe("refit"));
    let k_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &k_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: code_a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: val_a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: pts_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: children_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: parent_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: aabb_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: flags_b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 7, resource: kp_b.as_entire_binding() }] });

    // render + ui pipelines
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: uni_ty, count: None }] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { label: None, layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] },
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![1 => Uint32] }] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() }, depth_stencil: None, multisample: Default::default(), multiview: None });
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() }, depth_stencil: None, multisample: Default::default(), multiview: None });

    let wg = num_tiles as u32;
    let zeros_ni = vec![0u32; ni];

    // ---- state ----
    let mut moving_frac = 0.30f32;
    let mut forced: Option<Mode> = None; // None = adaptive
    let mut mode = Mode::Keep;
    let mut switches = 0u32;
    let (mut keep_ms, mut gpu_ms) = (0.5f32, 0.5f32);
    let (mut yaw, mut pitch, mut dist) = (0.8f32, 0.5f32, 2200.0f32);
    let (mut drag, mut last_mouse) = (false, (0.0f64, 0.0f64));
    let mut last = Instant::now();
    let mut fps = 0.0f32;
    let hyst = 1.20f32; // switch only if the alternative is >20% cheaper (dead-band)

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => drag = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => { let (dx, dy) = (position.x - last_mouse.0, position.y - last_mouse.1); last_mouse = (position.x, position.y); if drag { yaw += dx as f32 * 0.005; pitch = (pitch + dy as f32 * 0.004).clamp(-1.5, 1.5); } }
            WindowEvent::MouseWheel { delta, .. } => { let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 150.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 }; dist = (dist - d).clamp(300.0, 6000.0); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                KeyCode::Comma => moving_frac = (moving_frac - 0.02).max(0.0),
                KeyCode::Period => moving_frac = (moving_frac + 0.02).min(1.0),
                KeyCode::KeyA => forced = None,
                KeyCode::Digit1 => forced = Some(Mode::Keep),
                KeyCode::Digit2 => forced = Some(Mode::Gpu),
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let dt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };

                // move only the moving fraction (stable scattered subset) + flag it.
                let movers = ((moving_frac * n as f32) as usize).min(n);
                let t_keep = Instant::now();
                for i in 0..n {
                    let is_mover = i.wrapping_mul(2654435761) % n < movers;
                    flags_cpu[i] = is_mover as u32;
                    if is_mover {
                        for a in 0..3 { pos[i][a] += vel[i][a] * dt; if pos[i][a] < 1.0 || pos[i][a] > WORLD - 1.0 { vel[i][a] = -vel[i][a]; pos[i][a] = pos[i][a].clamp(1.0, WORLD - 1.0); } }
                        let p = pos[i]; tree.update_ref(handles[i], |it| it.0 = Point3::new(p[0] as f64, p[1] as f64, p[2] as f64));
                    }
                }
                keep_ms = keep_ms * 0.8 + 0.2 * t_keep.elapsed().as_secs_f32() * 1000.0;

                // GPU rebuild of the whole LBVH, timed (wall clock incl. poll(Wait)).
                queue.write_buffer(&pts_b, 0, bytemuck::cast_slice(&pos));
                let t_gpu = Instant::now();
                queue.write_buffer(&code_a, 0, bytemuck::cast_slice(&vec![u32::MAX; n2]));
                queue.write_buffer(&mp_b, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
                let mut enc = device.create_command_encoder(&Default::default());
                { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &m_bg, &[]); c.set_pipeline(&m_pipe); c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
                queue.submit(Some(enc.finish()));
                for pass in 0..8u32 {
                    let g = if pass % 2 == 0 { &bg_ab } else { &bg_ba };
                    queue.write_buffer(&rp_b, 0, bytemuck::cast_slice(&[n2 as u32, pass * 4, num_tiles as u32, 0u32]));
                    let mut enc = device.create_command_encoder(&Default::default());
                    { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&hist_p); c.dispatch_workgroups(wg, 1, 1); }
                    { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scan_p); c.dispatch_workgroups(1, 1, 1); }
                    { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, g, &[]); c.set_pipeline(&scat_p); c.dispatch_workgroups(wg, 1, 1); }
                    queue.submit(Some(enc.finish()));
                }
                queue.write_buffer(&kp_b, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
                queue.write_buffer(&flags_b, 0, bytemuck::cast_slice(&zeros_ni));
                let mut enc = device.create_command_encoder(&Default::default());
                { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&karras_p); c.dispatch_workgroups(((n - 1) as u32).div_ceil(256), 1, 1); }
                { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&leaf_p); c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
                { let mut c = enc.begin_compute_pass(&Default::default()); c.set_bind_group(0, &k_bg, &[]); c.set_pipeline(&refit_p); c.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
                queue.submit(Some(enc.finish()));
                device.poll(wgpu::Maintain::Wait);
                gpu_ms = gpu_ms * 0.8 + 0.2 * t_gpu.elapsed().as_secs_f32() * 1000.0;

                // adaptive pick with hysteresis (or a forced mode).
                let want = match forced {
                    Some(m) => m,
                    None => {
                        if mode == Mode::Keep && keep_ms > gpu_ms * hyst { Mode::Gpu }
                        else if mode == Mode::Gpu && gpu_ms > keep_ms * hyst { Mode::Keep }
                        else { mode }
                    }
                };
                if want as u8 != mode as u8 { switches += 1; mode = want; }

                // camera + render uploads
                let target = Vec3::splat(WORLD * 0.5);
                let eye = target + Vec3::new(dist * pitch.cos() * yaw.cos(), dist * pitch.sin(), dist * pitch.cos() * yaw.sin());
                let vp = Mat4::perspective_rh(45f32.to_radians(), config.width as f32 / config.height as f32, 1.0, 20000.0) * Mat4::look_at_rh(eye, target, Vec3::Y);
                queue.write_buffer(&cam_b, 0, bytemuck::cast_slice(&[Cam { vp: vp.to_cols_array_2d() }]));
                queue.write_buffer(&render_pos_b, 0, bytemuck::cast_slice(&pos));
                queue.write_buffer(&render_flag_b, 0, bytemuck::cast_slice(&flags_cpu));

                // HUD bars: keep vs GPU-rebuild cost, the picked one bright.
                let (sw, sh) = (config.width as f32, config.height as f32);
                let scale = 320.0 / 20.0; // px per ms (0..20 ms full scale)
                let mut ui: Vec<UiVertex> = Vec::new();
                push_quad(&mut ui, 12.0, 12.0, 344.0, 58.0, [0.0, 0.0, 0.0, 0.5], sw, sh);
                let bar = |ui: &mut Vec<UiVertex>, y: f32, ms: f32, col: [f32; 4], picked: bool| {
                    push_quad(ui, 18.0, y, 320.0, 18.0, [0.16, 0.16, 0.22, 0.7], sw, sh);
                    push_quad(ui, 18.0, y, (ms * scale).clamp(0.0, 320.0), 18.0, col, sw, sh);
                    if picked { push_quad(ui, 14.0, y - 2.0, 3.0, 22.0, [1.0, 1.0, 1.0, 0.9], sw, sh); }
                };
                bar(&mut ui, 18.0, keep_ms, [0.9, 0.35, 0.35, 0.95], mode == Mode::Keep);
                bar(&mut ui, 44.0, gpu_ms, [0.35, 0.9, 0.5, 0.95], mode == Mode::Gpu);
                queue.write_buffer(&ui_buf, 0, bytemuck::cast_slice(&ui));
                let ui_count = ui.len() as u32;

                let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view_tex = frame_tex.texture.create_view(&Default::default());
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.04, b: 0.06, a: 1.0 }), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None });
                    rp.set_pipeline(&render_pipe);
                    rp.set_bind_group(0, &cam_bg, &[]);
                    rp.set_vertex_buffer(0, render_pos_b.slice(..));
                    rp.set_vertex_buffer(1, render_flag_b.slice(..));
                    rp.draw(0..6, 0..n as u32);
                    rp.set_pipeline(&ui_pipeline); rp.set_vertex_buffer(0, ui_buf.slice(..)); rp.draw(0..ui_count, 0..1);
                }
                queue.submit(Some(enc.finish()));
                frame_tex.present();

                let policy = match forced { None => format!("ADAPTIVE → {}", if mode == Mode::Keep { "keep" } else { "GPU" }), Some(Mode::Keep) => "forced keep".into(), Some(Mode::Gpu) => "forced GPU".into() };
                window.set_title(&format!("adaptive_broadphase · {policy} [A/1/2] · moving {:.0}% [,.] · keep {keep_ms:.2}ms (red) vs GPU-rebuild {gpu_ms:.2}ms (green) · sw {switches} · {n} pts · {fps:.0} fps", moving_frac * 100.0));
                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => { elwt.set_control_flow(winit::event_loop::ControlFlow::Poll); window.request_redraw(); }
        _ => {}
    });
}
