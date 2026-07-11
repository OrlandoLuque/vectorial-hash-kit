//! gpu_lbvh_demo — the same spatial query, three backends, live. A cloud of
//! moving points, each coloured by its **neighbour count within a radius** (a
//! density heat-map = N interest queries per frame). Switch the backend that
//! computes it — CPU `Tree3` cull ↔ GPU brute ↔ **GPU LBVH** (a BVH built from
//! Morton codes, traversed in a compute shader) — and watch the on-screen meter:
//! the picture is identical, the cost is not.
//!
//! Keys: `1` CPU Tree3 · `2` GPU brute · `3` GPU LBVH · `[` `]` point count ·
//! drag orbit · scroll zoom. Meter (title): backend · query ms · FPS.
//!
//! `cargo run -p vectorial-hash-demos --bin gpu_lbvh_demo --release`
#![cfg(any(not(target_arch = "wasm32"), feature = "web-wgpu"))]
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::{event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};
use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Sphere3, Tree3};

const WORLD: f32 = 2000.0;
const RADIUS: f32 = 90.0; // neighbour radius

// ---------- CPU LBVH build (Morton sort + Karras split) → GPU nodes ----------
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn s(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    s(x as u64) | (s(y as u64) << 1) | (s(z as u64) << 2)
}
fn mcell(v: f32) -> u32 { ((v / WORLD) * (1u32 << 21) as f32) as u32 & ((1 << 21) - 1) }
/// (nodes [f32;8], sorted point xyzw, root)
fn build_lbvh(pos: &[[f32; 4]]) -> (Vec<[f32; 8]>, Vec<[f32; 4]>, u32) {
    let mut keyed: Vec<(u64, [f32; 4])> = pos.iter().map(|p| (morton3(mcell(p[0]), mcell(p[1]), mcell(p[2])), *p)).collect();
    keyed.sort_unstable_by_key(|k| k.0);
    let sorted: Vec<[f32; 4]> = keyed.iter().map(|k| k.1).collect();
    let mut nodes = Vec::with_capacity(pos.len() * 2);
    let root = build_range(&keyed, 0, keyed.len(), &mut nodes);
    (nodes, sorted, root)
}
fn build_range(k: &[(u64, [f32; 4])], lo: usize, hi: usize, nodes: &mut Vec<[f32; 8]>) -> u32 {
    if hi - lo == 1 {
        let p = k[lo].1; let id = nodes.len() as u32;
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }

#[derive(Clone, Copy, PartialEq)]
enum Backend { CpuTree, GpuBrute, GpuLbvh }
impl Backend { fn label(self) -> &'static str { match self { Backend::CpuTree => "CPU Tree3", Backend::GpuBrute => "GPU brute", Backend::GpuLbvh => "GPU LBVH" } } }

// --- 2D screen-space overlay (query/render GPU-load bars) ---
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }
fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let (x0, x1) = (px / sw * 2.0 - 1.0, (px + w) / sw * 2.0 - 1.0);
    let (y0, y1) = (1.0 - py / sh * 2.0, 1.0 - (py + h) / sh * 2.0);
    let q = |x, y| UiVertex { pos: [x, y], color };
    v.extend_from_slice(&[q(x0, y0), q(x1, y0), q(x1, y1), q(x0, y0), q(x1, y1), q(x0, y1)]);
}
const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;

struct P(Point3);
impl Positioned3 for P { fn position(&self) -> Point3 { self.0 } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u32 << 24) as f32 } }

#[cfg(not(target_arch = "wasm32"))]
fn main() { pollster::block_on(run()); }

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); wasm_bindgen_futures::spawn_local(run()); }

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("gpu_lbvh_demo").with_inner_size(winit::dpi::LogicalSize::new(1400, 900)).build(&event_loop).unwrap());
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        let _ = canvas.set_attribute("style", "width:100vw;height:100vh;display:block");
        web_sys::window().and_then(|w| w.document()).and_then(|d| d.body()).expect("body").append_child(&canvas.into()).expect("append canvas");
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.unwrap();
    let has_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: if has_ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() }, required_limits: adapter.limits() }, None).await.unwrap();
    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: wgpu::PresentMode::AutoNoVsync, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
    surface.configure(&device, &config);

    // ---- state
    let mut n = 50_000usize;
    let (mut pos, mut vel) = gen_points(n);
    let max_n = 400_000usize;

    // ---- GPU buffers (sized for max_n so [ ] never resizes)
    let points_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("points"), size: (max_n * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let counts_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("counts"), size: (max_n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let nodes_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("nodes"), size: (max_n * 2 * 32) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let params_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // ---- compute pipelines (brute + lbvh), sharing points/counts/params/nodes
    let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        st(0, true), st(1, false), unif(2), st(3, true),
    ] });
    let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&comp_bgl], push_constant_ranges: &[] });
    let brute_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(BRUTE.into()) });
    let lbvh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(LBVH.into()) });
    let brute_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&comp_pl), module: &brute_mod, entry_point: "main", compilation_options: Default::default() });
    let lbvh_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&comp_pl), module: &lbvh_mod, entry_point: "main", compilation_options: Default::default() });
    let comp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &comp_bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: points_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: counts_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: params_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: nodes_b.as_entire_binding() },
    ] });

    // ---- render pipeline: instanced billboard per point, coloured by count
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[unif_vs(0)] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] }, // pos.xyz
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![1 => Uint32] },   // count
        ] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });

    // 2D overlay pipeline (NDC quads, no camera, no depth) for the GPU-load bars.
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });
    let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui"), size: (128 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // 4 timestamps: query span (0,1) + render span (2,3) → separate GPU loads
    let qset = has_ts.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 4 }));
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // CPU Tree3 keep-index
    let mut tree = Tree3::new(Aabb::new(0.0, 0.0, 0.0, WORLD as f64, WORLD as f64, WORLD as f64), 16);
    let mut handles: Vec<Option<ItemRef>> = pos.iter().map(|p| tree.insert_ref(P(Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)))).collect();

    // GPU_DEMO_FRAMES=N → cycle the 3 backends and exit after N frames (smoke test)
    let max_frames: Option<u64> = std::env::var("GPU_DEMO_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut backend = Backend::GpuLbvh;
    let (mut yaw, mut pitch, mut dist) = (0.8f32, 0.5f32, 3200.0f32);
    let (mut drag, mut last_mouse) = (false, (0.0f64, 0.0f64));
    let mut last = Instant::now();
    let mut fps = 0.0f32;
    let mut query_ms = 0.0f32;
    let mut render_ms = 0.0f32;
    let mut frame = 0u64;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => drag = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => { let (dx, dy) = (position.x - last_mouse.0, position.y - last_mouse.1); last_mouse = (position.x, position.y); if drag { yaw += dx as f32 * 0.005; pitch = (pitch + dy as f32 * 0.004).clamp(-1.5, 1.5); } }
            WindowEvent::MouseWheel { delta, .. } => { let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 200.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 }; dist = (dist - d).clamp(400.0, 8000.0); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                KeyCode::Digit1 => backend = Backend::CpuTree,
                KeyCode::Digit2 => backend = Backend::GpuBrute,
                KeyCode::Digit3 => backend = Backend::GpuLbvh,
                KeyCode::BracketRight | KeyCode::BracketLeft => {
                    n = if c == KeyCode::BracketRight { (n + 25_000).min(max_n) } else { n.saturating_sub(25_000).max(5_000) };
                    let (p, v) = gen_points(n); pos = p; vel = v;
                    tree = Tree3::new(Aabb::new(0.0, 0.0, 0.0, WORLD as f64, WORLD as f64, WORLD as f64), 16);
                    handles = pos.iter().map(|q| tree.insert_ref(P(Point3::new(q[0] as f64, q[1] as f64, q[2] as f64)))).collect();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let dt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };
                frame += 1;

                // move points (CPU) + keep the tree in sync
                for i in 0..n {
                    for a in 0..3 { pos[i][a] += vel[i][a] * dt; if pos[i][a] < 1.0 || pos[i][a] > WORLD - 1.0 { vel[i][a] = -vel[i][a]; pos[i][a] = pos[i][a].clamp(1.0, WORLD - 1.0); } }
                    if backend == Backend::CpuTree { if let Some(h) = handles[i] { let p = pos[i]; tree.update_ref(h, |it| it.0 = Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)); } }
                }
                queue.write_buffer(&points_b, 0, bytemuck::cast_slice(&pos[..n]));
                queue.write_buffer(&params_b, 0, bytemuck::cast_slice(&[n as u32, 0u32, RADIUS.to_bits(), 0u32]));

                // camera
                let target = Vec3::splat(WORLD * 0.5);
                let eye = target + Vec3::new(dist * pitch.cos() * yaw.cos(), dist * pitch.sin(), dist * pitch.cos() * yaw.sin());
                let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                let proj = Mat4::perspective_rh(45f32.to_radians(), config.width as f32 / config.height as f32, 1.0, 20000.0);
                queue.write_buffer(&cam_b, 0, bytemuck::cast_slice(&[Cam { vp: (proj * view).to_cols_array_2d() }]));

                // ---- compute the neighbour counts via the active backend
                let t_cpu = Instant::now();
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                let use_gpu = backend != Backend::CpuTree;
                if backend == Backend::GpuLbvh {
                    let (nodes, sorted, root) = build_lbvh(&pos[..n]);
                    queue.write_buffer(&points_b, 0, bytemuck::cast_slice(&sorted)); // sorted for LBVH + render
                    queue.write_buffer(&nodes_b, 0, bytemuck::cast_slice(&nodes));
                    queue.write_buffer(&params_b, 0, bytemuck::cast_slice(&[n as u32, root, RADIUS.to_bits(), 0u32]));
                }
                if use_gpu {
                    let ts = qset.as_ref().map(|qs| wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: ts });
                    cp.set_pipeline(if backend == Backend::GpuLbvh { &lbvh_pipe } else { &brute_pipe });
                    cp.set_bind_group(0, &comp_bg, &[]);
                    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
                    drop(cp);
                } else {
                    // CPU Tree3: count neighbours per point, upload
                    let counts: Vec<u32> = (0..n).map(|i| { let p = pos[i]; (tree.cull(&Sphere3::new(p[0] as f64, p[1] as f64, p[2] as f64, RADIUS as f64)).len() as u32).saturating_sub(1) }).collect();
                    queue.write_buffer(&counts_b, 0, bytemuck::cast_slice(&counts));
                    query_ms = t_cpu.elapsed().as_secs_f32() * 1000.0;
                }

                // ---- render
                // ---- on-screen GPU-load bars (query + render, scaled to a 60 fps budget)
                let (sw, sh) = (config.width as f32, config.height as f32);
                let scale = 300.0 / (1000.0 / 60.0); // px per ms
                let qcol = match backend { Backend::CpuTree => [0.9, 0.3, 0.3, 0.95], Backend::GpuBrute => [1.0, 0.55, 0.2, 0.95], Backend::GpuLbvh => [0.35, 0.9, 0.45, 0.95] };
                let mut ui: Vec<UiVertex> = Vec::new();
                push_quad(&mut ui, 12.0, 12.0, 324.0, 44.0, [0.0, 0.0, 0.0, 0.45], sw, sh);
                for (i, (color, ms)) in [(qcol, query_ms), ([0.7, 0.45, 0.95, 0.95], render_ms)].iter().enumerate() {
                    let y = 18.0 + i as f32 * 17.0;
                    push_quad(&mut ui, 18.0, y, 300.0, 13.0, [0.16, 0.16, 0.22, 0.7], sw, sh);
                    push_quad(&mut ui, 18.0, y, (ms * scale).clamp(0.0, 316.0), 13.0, *color, sw, sh);
                }
                push_quad(&mut ui, 18.0 + 300.0 - 1.0, 16.0, 2.0, 38.0, [1.0, 1.0, 1.0, 0.55], sw, sh); // 60 fps tick
                queue.write_buffer(&ui_buf, 0, bytemuck::cast_slice(&ui));
                let ui_count = ui.len() as u32;

                let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view_tex = frame_tex.texture.create_view(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.05, b: 0.07, a: 1.0 }), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: qset.as_ref().map(|qs| wgpu::RenderPassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: Some(3) }), occlusion_query_set: None });
                    rp.set_pipeline(&render_pipe);
                    rp.set_bind_group(0, &cam_bg, &[]);
                    rp.set_vertex_buffer(0, points_b.slice(..));
                    rp.set_vertex_buffer(1, counts_b.slice(..));
                    rp.draw(0..6, 0..n as u32);
                    rp.set_pipeline(&ui_pipeline); rp.set_vertex_buffer(0, ui_buf.slice(..)); rp.draw(0..ui_count, 0..1);
                }
                if let Some(qs) = &qset { enc.resolve_query_set(qs, 0..4, &ts_resolve, 0); enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 32); }
                queue.submit(Some(enc.finish()));
                frame_tex.present();

                // native only: on WebGPU device.poll(Wait) doesn't block, so a
                // synchronous map_async→get_mapped_range readback throws (skip on web).
                if qset.is_some() && cfg!(not(target_arch = "wasm32")) {
                    device.poll(wgpu::Maintain::Wait);
                    ts_read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                    device.poll(wgpu::Maintain::Wait);
                    let t: Vec<u64> = bytemuck::cast_slice(&ts_read.slice(..).get_mapped_range()).to_vec();
                    ts_read.unmap();
                    let p = queue.get_timestamp_period();
                    render_ms = (t[3].wrapping_sub(t[2])) as f32 * p / 1e6;
                    if use_gpu { query_ms = (t[1].wrapping_sub(t[0])) as f32 * p / 1e6; }
                }

                if frame % 8 == 0 && std::env::var_os("SHOT").is_none() {
                    let load = (query_ms + render_ms) / (1000.0 / 60.0) * 100.0; // % of a 60 fps budget
                    window.set_title(&format!("gpu_lbvh_demo · {} [1/2/3] · {n} pts [ ] · query {query_ms:.2} + render {render_ms:.2} = {:.2}ms ({load:.0}% of 60fps) · {fps:.0} fps", backend.label(), query_ms + render_ms));
                }
                if let Some(mf) = max_frames {
                    if frame % (mf / 3).max(1) == 0 { backend = match backend { Backend::GpuLbvh => Backend::GpuBrute, Backend::GpuBrute => Backend::CpuTree, Backend::CpuTree => Backend::GpuLbvh }; }
                    println!("frame {frame} · {} · query {:.2} ms · {:.0} fps · {n} pts", backend.label(), query_ms, fps);
                    if frame >= mf { elwt.exit(); }
                }
                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => { elwt.set_control_flow(winit::event_loop::ControlFlow::Poll); window.request_redraw(); }
        _ => {}
    });
}

fn gen_points(n: usize) -> (Vec<[f32; 4]>, Vec<[f32; 3]>) {
    let mut r = Rng(0xC0FFEE);
    // a few gaussian-ish clumps + scatter → an interesting density field
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    for i in 0..n {
        let p = if i % 3 == 0 {
            [r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD]
        } else {
            let c = ((i * 2654435761) % 12) as f32;
            let cx = 200.0 + (c % 4.0) * 500.0; let cy = 200.0 + ((c / 4.0).floor() % 3.0) * 700.0; let cz = WORLD * 0.5;
            [(cx + (r.unit() - 0.5) * 500.0).clamp(1.0, WORLD - 1.0), (cy + (r.unit() - 0.5) * 500.0).clamp(1.0, WORLD - 1.0), (cz + (r.unit() - 0.5) * 800.0).clamp(1.0, WORLD - 1.0)]
        };
        pos.push([p[0], p[1], p[2], 0.0]);
        vel.push([(r.unit() - 0.5) * 120.0, (r.unit() - 0.5) * 120.0, (r.unit() - 0.5) * 120.0]);
    }
    (pos, vel)
}

fn st(b: u32, ro: bool) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None }, count: None } }
fn unif(b: u32) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None } }
fn unif_vs(b: u32) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None } }

const BRUTE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;  // x=n, z=radius(bits)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let n = params.x; if (i >= n) { return; }
    let r2 = bitcast<f32>(params.z) * bitcast<f32>(params.z);
    let p = points[i].xyz; var c = 0u;
    for (var j = 0u; j < n; j = j + 1u) { let d = points[j].xyz - p; if (dot(d, d) <= r2) { c = c + 1u; } }
    counts[i] = c - 1u;
}
"#;

const LBVH: &str = r#"
struct Node { a: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;  // x=n, y=root, z=radius(bits)
@group(0) @binding(3) var<storage, read> nodes: array<Node>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= params.x) { return; }
    let p = points[i].xyz; let r2 = bitcast<f32>(params.z) * bitcast<f32>(params.z);
    var c = 0u; var stack: array<u32, 64>; var sp = 0; stack[0] = params.y; sp = 1;
    loop {
        if (sp == 0) { break; } sp = sp - 1; let node = nodes[stack[sp]];
        let nearest = clamp(p, node.a.xyz, node.b.xyz); let d = nearest - p;
        if (dot(d, d) > r2) { continue; }
        let right = bitcast<u32>(node.b.w);
        if (right == 0xFFFFFFFFu) { let dp = points[bitcast<u32>(node.a.w)].xyz - p; if (dot(dp, dp) <= r2) { c = c + 1u; } }
        else if (sp < 62) { stack[sp] = bitcast<u32>(node.a.w); sp = sp + 1; stack[sp] = right; sp = sp + 1; }
    }
    counts[i] = c - 1u;
}
"#;

const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) col: vec3<f32> };
fn heat(t: f32) -> vec3<f32> {
    // blue → cyan → green → yellow → red
    let x = clamp(t, 0.0, 1.0);
    return clamp(vec3<f32>(1.5 - abs(4.0 * x - 3.0), 1.5 - abs(4.0 * x - 2.0), 1.5 - abs(4.0 * x - 1.0)), vec3<f32>(0.0), vec3<f32>(1.0));
}
@vertex
fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec4<f32>, @location(1) count: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,1.0));
    var o: VOut;
    let clip = cam.vp * vec4<f32>(pos.xyz, 1.0);
    let sz = 3.5;
    o.clip = vec4<f32>(clip.xy + corners[vi] * sz * clip.w / 700.0, clip.z, clip.w);
    o.col = heat(f32(count) / 40.0);
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.col, 0.9); }
"#;
