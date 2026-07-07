//! gpu_storm — a **GPU-resident** collision storm. Tens/hundreds of thousands of
//! particles bounce and collide in a box, and the *whole hot loop lives on the
//! GPU*: build a uniform grid → find colliding pairs → resolve (a spring-dashpot
//! DEM response) → integrate, all in compute shaders, positions/velocities never
//! leaving GPU memory (only the framebuffer is presented). That's the point the
//! game sims can't reach — no per-frame CPU↔GPU round-trip — so switching the
//! backend compares *the whole simulation on GPU vs on CPU*, not just a query.
//!
//! Keys: `1` CPU grid · `2` GPU-resident · `F` collision ↔ **influence field**
//! (each particle a moving emitter, coloured by neighbours within a glow radius —
//! the same grid, a bigger query, no forces) · `[` `]` particle count · drag
//! orbit · scroll zoom. Title meter: backend · mode · sim ms · FPS · count.
//!
//! `cargo run -p vectorial-hash-demos --bin gpu_storm --release`
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::{event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};

const WORLD: f32 = 1600.0;
const GD: u32 = 128;                 // grid dimension per axis
const CELL: f32 = WORLD / GD as f32; // 12.5
const R: f32 = 6.0;                  // particle radius (< CELL/2 so 3×3×3 suffices)
const CAP: u32 = 16;                 // per-cell bucket capacity
const DT: f32 = 0.006;               // fixed sim step (stability, not frame-rate coupled)
const SPRING: f32 = 600.0;
const DAMP: f32 = 6.0;
const GLOW_R: f32 = 30.0; // influence-field mode: proximity-glow radius (multi-ring)

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params { n: u32, gd: u32, cap: u32, _p0: u32, world: f32, cell: f32, radius: f32, dt: f32, spring: f32, damp: f32, _p1: f32, _p2: f32 }

#[derive(Clone, Copy, PartialEq)]
enum Backend { Cpu, Gpu }
impl Backend { fn label(self) -> &'static str { match self { Backend::Cpu => "CPU grid", Backend::Gpu => "GPU-resident" } } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn u(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u32 << 24) as f32 } }

fn spawn(n: usize) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut r = Rng(0xBADC0DE);
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push([R + r.u() * (WORLD - 2.0 * R), R + r.u() * (WORLD - 2.0 * R), R + r.u() * (WORLD - 2.0 * R), 0.0]);
        vel.push([(r.u() - 0.5) * 300.0, (r.u() - 0.5) * 300.0, (r.u() - 0.5) * 300.0, 0.0]);
    }
    (pos, vel)
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("gpu_storm").with_inner_size(winit::dpi::LogicalSize::new(1400, 900)).build(&event_loop).unwrap());

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
    let mut n = 150_000usize;
    let max_n = 500_000usize;
    let (mut pos, mut vel) = spawn(n);
    let ncells = (GD * GD * GD) as usize;

    // ---- GPU buffers (all sized for max_n / fixed grid). Sim state is RESIDENT.
    let pos_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pos"), size: (max_n * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let vel_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("vel"), size: (max_n * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let acc_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("acc"), size: (max_n * 16) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let coll_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("coll"), size: (max_n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let gcount_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gcount"), size: (ncells * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let gitems_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("gitems"), size: (ncells * CAP as usize * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let params_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: std::mem::size_of::<Params>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: Some("rb"), size: (max_n * 16) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // ---- one compute bind group layout shared by all sim passes (each shader uses a subset)
    let cbgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        st(0, false), st(1, false), st(2, false), st(3, false), st(4, false), st(5, false), unif(6),
    ] });
    let cpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cbgl], push_constant_ranges: &[] });
    let mk = |src: &str, ep: &str| { let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) }); device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&cpl), module: &m, entry_point: ep, compilation_options: Default::default() }) };
    let clear_pipe = mk(SIM, "clear_grid");
    let build_pipe = mk(SIM, "build_grid");
    let collide_pipe = mk(SIM, "collide");
    let integ_pipe = mk(SIM, "integrate");
    let cbg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cbgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: pos_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: vel_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: acc_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: coll_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: gcount_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: gitems_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: params_b.as_entire_binding() },
    ] });

    // ---- render: instanced billboard per particle, coloured by contact count
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[unif_vs(0)] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] },
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![1 => Uint32] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        // opaque discs + a depth buffer so nearer particles correctly occlude farther ones
        depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
        multisample: Default::default(), multiview: None,
    });
    let make_depth = |device: &wgpu::Device, w: u32, h: u32| device.create_texture(&wgpu::TextureDescriptor { label: Some("depth"), size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[] }).create_view(&Default::default());
    let mut depth_view = make_depth(&device, config.width, config.height);

    // 8 timestamps = 4 timed spans: grid(0,1) collide(2,3) integrate(4,5) render(6,7)
    let qset = has_ts.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 8 }));
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let params = |n: usize, mode: f32| Params { n: n as u32, gd: GD, cap: CAP, _p0: 0, world: WORLD, cell: CELL, radius: R, dt: DT, spring: SPRING, damp: DAMP, _p1: mode, _p2: GLOW_R };
    queue.write_buffer(&params_b, 0, bytemuck::bytes_of(&params(n, 0.0)));
    queue.write_buffer(&pos_b, 0, bytemuck::cast_slice(&pos));
    queue.write_buffer(&vel_b, 0, bytemuck::cast_slice(&vel));

    let smoke: Option<u64> = std::env::var("GPU_STORM_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut backend = Backend::Gpu;
    let mut mode = if std::env::var("GPU_STORM_INFLUENCE").is_ok() { 1.0f32 } else { 0.0f32 }; // 0 = collision, 1 = influence field (F toggles)
    let (mut yaw, mut pitch, mut dist) = (0.7f32, 0.4f32, 2600.0f32);
    let (mut drag, mut last_mouse) = (false, (0.0f64, 0.0f64));
    let mut last = Instant::now();
    let mut fps = 0.0f32;
    let mut sim_ms = 0.0f32;
    let (mut grid_ms, mut coll_ms, mut integ_ms, mut render_ms) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut frame = 0u64;
    // scratch for the CPU backend
    let mut heads: Vec<i32> = vec![-1; ncells];
    let mut nextp: Vec<i32> = vec![-1; max_n];

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); depth_view = make_depth(&device, config.width, config.height); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => drag = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => { let (dx, dy) = (position.x - last_mouse.0, position.y - last_mouse.1); last_mouse = (position.x, position.y); if drag { yaw += dx as f32 * 0.005; pitch = (pitch + dy as f32 * 0.004).clamp(-1.5, 1.5); } }
            WindowEvent::MouseWheel { delta, .. } => { let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 200.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 }; dist = (dist - d).clamp(300.0, 7000.0); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                KeyCode::Digit1 => { backend = Backend::Cpu; sync_from_gpu(&device, &queue, &pos_b, &vel_b, &readback, n, &mut pos, &mut vel); }
                KeyCode::Digit2 => { backend = Backend::Gpu; queue.write_buffer(&pos_b, 0, bytemuck::cast_slice(&pos[..n])); queue.write_buffer(&vel_b, 0, bytemuck::cast_slice(&vel[..n])); }
                KeyCode::BracketRight | KeyCode::BracketLeft => {
                    n = if c == KeyCode::BracketRight { (n + 50_000).min(max_n) } else { n.saturating_sub(50_000).max(10_000) };
                    let (p, v) = spawn(n); pos = p; vel = v;
                    queue.write_buffer(&params_b, 0, bytemuck::bytes_of(&params(n, mode)));
                    queue.write_buffer(&pos_b, 0, bytemuck::cast_slice(&pos)); queue.write_buffer(&vel_b, 0, bytemuck::cast_slice(&vel));
                }
                KeyCode::KeyF => mode = if mode > 0.5 { 0.0 } else { 1.0 }, // collision ↔ influence field
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let fdt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                fps = if fps == 0.0 { 1.0 / fdt } else { fps * 0.9 + 0.1 / fdt };
                frame += 1;
                queue.write_buffer(&params_b, 0, bytemuck::bytes_of(&params(n, mode)));

                let target = Vec3::splat(WORLD * 0.5);
                let eye = target + Vec3::new(dist * pitch.cos() * yaw.cos(), dist * pitch.sin(), dist * pitch.cos() * yaw.sin());
                let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                let proj = Mat4::perspective_rh(45f32.to_radians(), config.width as f32 / config.height as f32, 1.0, 20000.0);
                queue.write_buffer(&cam_b, 0, bytemuck::bytes_of(&Cam { vp: (proj * view).to_cols_array_2d() }));

                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                if backend == Backend::Gpu {
                    // three timed compute passes so each GPU load is visible separately
                    let span = |a: u32, b: u32| qset.as_ref().map(|qs| wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(a), end_of_pass_write_index: Some(b) });
                    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("grid"), timestamp_writes: span(0, 1) });
                      cp.set_bind_group(0, &cbg, &[]);
                      cp.set_pipeline(&clear_pipe); cp.dispatch_workgroups((ncells as u32).div_ceil(64), 1, 1);
                      cp.set_pipeline(&build_pipe); cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1); }
                    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("collide"), timestamp_writes: span(2, 3) });
                      cp.set_bind_group(0, &cbg, &[]); cp.set_pipeline(&collide_pipe); cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1); }
                    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("integrate"), timestamp_writes: span(4, 5) });
                      cp.set_bind_group(0, &cbg, &[]); cp.set_pipeline(&integ_pipe); cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1); }
                } else {
                    // CPU grid DEM (serial) — the same algorithm, on the CPU, for the switch
                    let t = Instant::now();
                    cpu_step(&mut pos, &mut vel, n, &mut heads, &mut nextp, ncells, mode);
                    sim_ms = t.elapsed().as_secs_f32() * 1000.0;
                    let counts: Vec<u32> = pos[..n].iter().map(|p| p[3] as u32).collect();
                    queue.write_buffer(&pos_b, 0, bytemuck::cast_slice(&pos[..n]));
                    queue.write_buffer(&coll_b, 0, bytemuck::cast_slice(&counts));
                }

                let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view_tex = frame_tex.texture.create_view(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.03, b: 0.05, a: 1.0 }), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }), timestamp_writes: qset.as_ref().map(|qs| wgpu::RenderPassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(6), end_of_pass_write_index: Some(7) }), occlusion_query_set: None });
                    rp.set_pipeline(&render_pipe);
                    rp.set_bind_group(0, &cam_bg, &[]);
                    rp.set_vertex_buffer(0, pos_b.slice(..));
                    rp.set_vertex_buffer(1, coll_b.slice(..));
                    rp.draw(0..6, 0..n as u32);
                }
                if let Some(qs) = &qset { enc.resolve_query_set(qs, 0..8, &ts_resolve, 0); enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 64); }
                queue.submit(Some(enc.finish()));
                frame_tex.present();

                if qset.is_some() {
                    device.poll(wgpu::Maintain::Wait);
                    ts_read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                    device.poll(wgpu::Maintain::Wait);
                    let t: Vec<u64> = bytemuck::cast_slice(&ts_read.slice(..).get_mapped_range()).to_vec();
                    ts_read.unmap();
                    let p = queue.get_timestamp_period();
                    let ms = |a: usize, b: usize| (t[b].wrapping_sub(t[a])) as f32 * p / 1e6;
                    render_ms = ms(6, 7);
                    if backend == Backend::Gpu { grid_ms = ms(0, 1); coll_ms = ms(2, 3); integ_ms = ms(4, 5); sim_ms = grid_ms + coll_ms + integ_ms; }
                    else { grid_ms = 0.0; coll_ms = 0.0; integ_ms = 0.0; }
                }

                if frame % 8 == 0 {
                    let mode_s = if mode > 0.5 { "influence" } else { "collision" };
                    let gpu_ms = sim_ms + render_ms;                 // total GPU work / frame
                    let load = gpu_ms / (1000.0 / 60.0) * 100.0;     // % of a 60 fps budget
                    let title = if backend == Backend::Gpu {
                        format!("gpu_storm · GPU-resident [1/2] · {mode_s} [F] · {n} pts [ ] · GPU: grid {grid_ms:.2}+coll {coll_ms:.2}+integ {integ_ms:.2}+render {render_ms:.2} = {gpu_ms:.2}ms ({load:.0}% of 60fps) · {fps:.0} fps")
                    } else {
                        format!("gpu_storm · CPU grid [1/2] · {mode_s} [F] · {n} pts [ ] · CPU sim {sim_ms:.1}ms + GPU render {render_ms:.2}ms ({load:.0}% of 60fps) · {fps:.0} fps")
                    };
                    window.set_title(&title);
                }
                if let Some(mf) = smoke {
                    if frame % (mf / 2).max(1) == 0 { backend = match backend { Backend::Gpu => Backend::Cpu, Backend::Cpu => Backend::Gpu }; }
                    // invariant check: read positions back, assert in-bounds + finite
                    if frame % 15 == 0 || frame >= mf {
                        sync_from_gpu(&device, &queue, &pos_b, &vel_b, &readback, n, &mut pos, &mut vel);
                        let bad = pos[..n].iter().filter(|p| !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) || p[0] < -1.0 || p[0] > WORLD + 1.0 || p[1] < -1.0 || p[1] > WORLD + 1.0 || p[2] < -1.0 || p[2] > WORLD + 1.0).count();
                        let contacts: u64 = pos[..n].iter().map(|p| p[3] as u64).sum();
                        println!("frame {frame} · {} · GPU grid {grid_ms:.2}+coll {coll_ms:.2}+integ {integ_ms:.2}+render {render_ms:.2} = {:.2}ms · {:.0} fps · {n} pts · oob/NaN {bad} · contacts {contacts}", backend.label(), sim_ms + render_ms, fps);
                        assert!(bad == 0, "particles left the box or went NaN ({bad})");
                    }
                    if frame >= mf { elwt.exit(); }
                }
                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    });
}

/// Read pos/vel back from the GPU into the CPU mirrors (for the switch + smoke test).
fn sync_from_gpu(device: &wgpu::Device, queue: &wgpu::Queue, pos_b: &wgpu::Buffer, _vel_b: &wgpu::Buffer, readback: &wgpu::Buffer, n: usize, pos: &mut [[f32; 4]], _vel: &mut [[f32; 4]]) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(pos_b, 0, readback, 0, (n * 16) as u64);
    queue.submit(Some(enc.finish()));
    readback.slice(..(n * 16) as u64).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    { let m = readback.slice(..(n * 16) as u64).get_mapped_range(); pos[..n].copy_from_slice(bytemuck::cast_slice(&m)); }
    readback.unmap();
}

/// CPU mirror of the GPU sim: a rebuilt uniform grid + spring-dashpot DEM. `pos[i][3]`
/// carries the per-particle contact count (for the render colour), matching the GPU.
fn cpu_step(pos: &mut [[f32; 4]], vel: &mut [[f32; 4]], n: usize, heads: &mut [i32], nextp: &mut [i32], ncells: usize, mode: f32) {
    let cell_of = |p: &[f32; 4]| -> usize {
        let gx = ((p[0] / CELL) as i32).clamp(0, GD as i32 - 1);
        let gy = ((p[1] / CELL) as i32).clamp(0, GD as i32 - 1);
        let gz = ((p[2] / CELL) as i32).clamp(0, GD as i32 - 1);
        (gx + gy * GD as i32 + gz * (GD * GD) as i32) as usize
    };
    for h in heads[..ncells].iter_mut() { *h = -1; }
    for i in 0..n { let c = cell_of(&pos[i]); nextp[i] = heads[c]; heads[c] = i as i32; }
    let influence = mode > 0.5;
    let rad = if influence { GLOW_R } else { 2.0 * R };
    let ring = if influence { (GLOW_R / CELL).ceil() as i32 } else { 1 };
    let (r, rad2) = (R, rad * rad);
    for i in 0..n {
        let pi = pos[i]; let vi = vel[i];
        let (gx, gy, gz) = (((pi[0] / CELL) as i32).clamp(0, GD as i32 - 1), ((pi[1] / CELL) as i32).clamp(0, GD as i32 - 1), ((pi[2] / CELL) as i32).clamp(0, GD as i32 - 1));
        let (mut fx, mut fy, mut fz) = (0.0f32, 0.0f32, 0.0f32);
        let mut contacts = 0u32;
        for dz in -ring..=ring { for dy in -ring..=ring { for dx in -ring..=ring {
            let (cx, cy, cz) = (gx + dx, gy + dy, gz + dz);
            if cx < 0 || cy < 0 || cz < 0 || cx >= GD as i32 || cy >= GD as i32 || cz >= GD as i32 { continue; }
            let c = (cx + cy * GD as i32 + cz * (GD * GD) as i32) as usize;
            let mut j = heads[c];
            while j >= 0 { let ju = j as usize; if ju != i {
                let pj = pos[ju]; let (ox, oy, oz) = (pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]);
                let d2 = ox * ox + oy * oy + oz * oz;
                if d2 < rad2 && d2 > 1e-8 {
                    if !influence { let d = d2.sqrt(); let inv = 1.0 / d; let (nx, ny, nz) = (ox * inv, oy * inv, oz * inv);
                        let overlap = 2.0 * r - d;
                        if overlap > 0.0 { let (rvx, rvy, rvz) = (vel[ju][0] - vi[0], vel[ju][1] - vi[1], vel[ju][2] - vi[2]);
                            fx += SPRING * overlap * nx + DAMP * rvx; fy += SPRING * overlap * ny + DAMP * rvy; fz += SPRING * overlap * nz + DAMP * rvz; } }
                    contacts += 1;
                }
            } j = nextp[ju]; }
        }}}
        let mut nv = [vi[0] + fx * DT, vi[1] + fy * DT, vi[2] + fz * DT, 0.0];
        let mut np = [pi[0] + nv[0] * DT, pi[1] + nv[1] * DT, pi[2] + nv[2] * DT, contacts as f32];
        for a in 0..3 { if np[a] < R { np[a] = R; nv[a] = nv[a].abs(); } if np[a] > WORLD - R { np[a] = WORLD - R; nv[a] = -nv[a].abs(); } }
        pos[i] = np; vel[i] = nv;
    }
}

fn st(b: u32, _ro: bool) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None } }
fn unif(b: u32) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None } }
fn unif_vs(b: u32) -> wgpu::BindGroupLayoutEntry { wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None } }

// ---- the whole simulation in one WGSL module (4 entry points sharing the layout)
const SIM: &str = r#"
struct Params { n: u32, gd: u32, cap: u32, _p0: u32, world: f32, cell: f32, radius: f32, dt: f32, spring: f32, damp: f32, mode: f32, glow_r: f32 };
@group(0) @binding(0) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> coll: array<u32>;
@group(0) @binding(4) var<storage, read_write> gcount: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> gitems: array<u32>;
@group(0) @binding(6) var<uniform> P: Params;

fn cell_of(p: vec3<f32>) -> u32 {
    let g = vec3<i32>(clamp(vec3<i32>(p / P.cell), vec3<i32>(0), vec3<i32>(i32(P.gd) - 1)));
    return u32(g.x + g.y * i32(P.gd) + g.z * i32(P.gd) * i32(P.gd));
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x; if (c >= P.gd * P.gd * P.gd) { return; }
    atomicStore(&gcount[c], 0u);
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= P.n) { return; }
    let c = cell_of(pos[i].xyz);
    let slot = atomicAdd(&gcount[c], 1u);
    if (slot < P.cap) { gitems[c * P.cap + slot] = i; }
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= P.n) { return; }
    let pi = pos[i].xyz; let vi = vel[i].xyz;
    let influence = P.mode > 0.5;
    // collide mode: contacts within 2r, 3×3×3. influence mode: neighbours within
    // glow_r (a proximity-glow field), scanning ⌈glow_r/cell⌉ rings, no force.
    let rad = select(2.0 * P.radius, P.glow_r, influence);
    let ring = select(1, i32(ceil(P.glow_r / P.cell)), influence);
    let g = clamp(vec3<i32>(pi / P.cell), vec3<i32>(0), vec3<i32>(i32(P.gd) - 1));
    var f = vec3<f32>(0.0); var cnt = 0u;
    for (var dz = -ring; dz <= ring; dz = dz + 1) {
    for (var dy = -ring; dy <= ring; dy = dy + 1) {
    for (var dx = -ring; dx <= ring; dx = dx + 1) {
        let cc = g + vec3<i32>(dx, dy, dz);
        if (cc.x < 0 || cc.y < 0 || cc.z < 0 || cc.x >= i32(P.gd) || cc.y >= i32(P.gd) || cc.z >= i32(P.gd)) { continue; }
        let c = u32(cc.x + cc.y * i32(P.gd) + cc.z * i32(P.gd) * i32(P.gd));
        let m = min(atomicLoad(&gcount[c]), P.cap);
        for (var s = 0u; s < m; s = s + 1u) {
            let j = gitems[c * P.cap + s];
            if (j == i) { continue; }
            let o = pi - pos[j].xyz; let d2 = dot(o, o);
            if (d2 < rad * rad && d2 > 1e-8) {
                if (!influence) {
                    let d = sqrt(d2); let nrm = o / d; let overlap = 2.0 * P.radius - d;
                    if (overlap > 0.0) { f = f + P.spring * overlap * nrm + P.damp * (vel[j].xyz - vi); }
                }
                cnt = cnt + 1u;
            }
        }
    }}}
    acc[i] = select(vec4<f32>(f, 0.0), vec4<f32>(0.0), influence); // influence = drift, no force
    coll[i] = cnt;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= P.n) { return; }
    var v = vel[i].xyz + acc[i].xyz * P.dt;
    var p = pos[i].xyz + v * P.dt;
    let lo = P.radius; let hi = P.world - P.radius;
    if (p.x < lo) { p.x = lo; v.x = abs(v.x); } if (p.x > hi) { p.x = hi; v.x = -abs(v.x); }
    if (p.y < lo) { p.y = lo; v.y = abs(v.y); } if (p.y > hi) { p.y = hi; v.y = -abs(v.y); }
    if (p.z < lo) { p.z = lo; v.z = abs(v.z); } if (p.z > hi) { p.z = hi; v.z = -abs(v.z); }
    pos[i] = vec4<f32>(p, f32(coll[i]));
    vel[i] = vec4<f32>(v, 0.0);
}
"#;

const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) heat: f32 };
@vertex
fn vs(@location(0) inst: vec4<f32>, @location(1) count: u32, @builtin(vertex_index) vi: u32) -> VO {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let c = corners[vi];
    var clip = cam.vp * vec4<f32>(inst.xyz, 1.0);
    let sz = 5.0;
    clip = vec4<f32>(clip.xy + c * sz * clip.w / 800.0, clip.z, clip.w);
    var o: VO; o.clip = clip; o.uv = c; o.heat = f32(count); return o;
}
fn heat(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return vec3<f32>(clamp(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0), clamp(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0), clamp(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0));
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    if (dot(v.uv, v.uv) > 1.0) { discard; }
    let col = heat(v.heat / 8.0);
    return vec4<f32>(col, 1.0);
}
"#;
