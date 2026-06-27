//! `siege_wgpu` — the siege battle rendered with **wgpu** (low-level, modern GPU
//! stack) instead of macroquad, so the two renderers can be compared. Native
//! only (wgpu/winit aren't in the wasm demo build).
//!
//! The foundation: a self-contained army battle (units target via the `Tree3`
//! index, advance and clash) drawn as GPU-instanced cubes with an orbit camera,
//! depth + simple lighting. It deliberately does NOT touch the macroquad `siege`
//! binary. The payoff of the wgpu road — real GPU skeletal skinning of thousands
//! of animated units (which macroquad's WebGL1 stack can't do) — is the next
//! layer. Drag the mouse to orbit; scroll to zoom.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin siege_wgpu --release`

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use vectorial_hash::{Aabb, Point3, Positioned3, Tree3};

// ============================================================ simulation

const WORLD: f64 = 640.0;
const PER_FACTION: usize = 1500;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + ((self.next() >> 11) as f64 / (1u64 << 53) as f64) * (hi - lo) }
}

#[derive(Clone, Copy)]
struct IUnit { id: u32, fac: u8, p: Point3 }
impl Positioned3 for IUnit { fn position(&self) -> Point3 { self.p } }

struct Unit { p: Point3, vel: (f64, f64), fac: u8, hp: f64, cd: f64, respawn: f64, face: f32 }

fn castle(fac: u8) -> (f64, f64) { if fac == 0 { (80.0, 80.0) } else { (WORLD - 80.0, WORLD - 80.0) } }

fn spawn_unit(rng: &mut Rng, fac: u8) -> Unit {
    let (cx, cz) = castle(fac);
    Unit {
        p: Point3::new((cx + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0), 0.0, (cz + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0)),
        vel: (0.0, 0.0), fac, hp: 100.0, cd: 0.0, respawn: f64::INFINITY, face: 0.0,
    }
}

fn spawn_army(rng: &mut Rng) -> Vec<Unit> {
    let mut u = Vec::with_capacity(PER_FACTION * 2);
    for _ in 0..PER_FACTION { u.push(spawn_unit(rng, 0)); }
    for _ in 0..PER_FACTION { u.push(spawn_unit(rng, 1)); }
    u
}

const REACH: f64 = 9.0;
const SPEED: f64 = 26.0;
const SEP: f64 = 7.0;

/// One sim step: rebuild the index, each unit targets its nearest enemy (k-NN)
/// and steers toward it with separation; melee damage resolved serially.
fn step(units: &mut [Unit], index: &mut Tree3<IUnit>, rng: &mut Rng, dt: f64, now: f64) {
    index.clear();
    for (i, u) in units.iter().enumerate() {
        if u.hp > 0.0 { index.insert(IUnit { id: i as u32, fac: u.fac, p: u.p }); }
    }
    let mut dmg = vec![0.0f64; units.len()];
    // decide (read-only on the index): velocity + attack into per-unit scratch
    let mut hits: Vec<(usize, f64)> = Vec::new();
    for i in 0..units.len() {
        if units[i].hp <= 0.0 { units[i].vel = (0.0, 0.0); continue; }
        let (p, fac) = (units[i].p, units[i].fac);
        let mut target: Option<(Point3, u32, f64)> = None;
        let (mut sx, mut sz) = (0.0, 0.0);
        for (d, it) in index.knn(p, 14) {
            if it.id == i as u32 { continue; }
            if d < SEP { let dd = d.max(1e-3); sx += (p.x - it.p.x) / dd; sz += (p.z - it.p.z) / dd; }
            if it.fac != fac && target.is_none() { target = Some((it.p, it.id, d)); }
        }
        let (tx, tz, td) = match target { Some((tp, _, d)) => (tp.x, tp.z, d), None => { let (cx, cz) = castle(1 - fac); (cx, cz, f64::INFINITY) } };
        let (dx, dz) = (tx - p.x, tz - p.z);
        let l = (dx * dx + dz * dz).sqrt().max(1e-6);
        let appr = if td < REACH * 0.8 { 0.0 } else { SPEED };
        let mut vx = dx / l * appr + sx * SPEED * 0.7;
        let mut vz = dz / l * appr + sz * SPEED * 0.7;
        let vl = (vx * vx + vz * vz).sqrt();
        if vl > SPEED * 1.5 { let s = SPEED * 1.5 / vl; vx *= s; vz *= s; }
        units[i].vel = (vx, vz);
        if units[i].cd <= 0.0 && td <= REACH {
            if let Some((_, tid, _)) = target { hits.push((tid as usize, 14.0)); units[i].cd = 0.8; }
        }
    }
    for (tid, d) in hits { if let Some(s) = dmg.get_mut(tid) { *s += d; } }
    // apply
    for (u, d) in units.iter_mut().zip(dmg) {
        if u.hp <= 0.0 {
            if u.respawn.is_finite() && now >= u.respawn { let (cx, cz) = castle(u.fac); u.p = Point3::new((cx + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0), 0.0, (cz + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0)); u.hp = 100.0; u.respawn = f64::INFINITY; }
            continue;
        }
        u.cd = (u.cd - dt).max(0.0);
        let nx = (u.p.x + u.vel.0 * dt).clamp(2.0, WORLD - 2.0);
        let nz = (u.p.z + u.vel.1 * dt).clamp(2.0, WORLD - 2.0);
        if u.vel.0 * u.vel.0 + u.vel.1 * u.vel.1 > 1.0 { u.face = (u.vel.0 as f32).atan2(u.vel.1 as f32); }
        u.p = Point3::new(nx, 0.0, nz);
        if d > 0.0 { u.hp -= d; if u.hp <= 0.0 { u.respawn = now + 4.0; } }
    }
}

// ============================================================ terrain

fn hash2(x: i32, z: i32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x1659_5e3d).wrapping_add((z as u32).wrapping_mul(0x27d4_eb2f));
    h ^= h >> 15; h = h.wrapping_mul(0x85eb_ca6b); h ^= h >> 13;
    (h & 0x00ff_ffff) as f64 / 0x00ff_ffff as f64
}
fn vnoise(x: f64, z: f64) -> f64 {
    let (xi, zi) = (x.floor() as i32, z.floor() as i32);
    let (fx, fz) = (x - xi as f64, z - zi as f64);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let a = hash2(xi, zi); let b = hash2(xi + 1, zi);
    let c = hash2(xi, zi + 1); let d = hash2(xi + 1, zi + 1);
    let ab = a + (b - a) * sx; let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}
/// Hills + a central volcano cone (matches the macroquad siege's terrain).
fn terrain_height(x: f64, z: f64) -> f64 {
    let s = 1.0 / 150.0;
    let mut h = vnoise(x * s, z * s) * 42.0 + vnoise(x * s * 2.7, z * s * 2.7) * 15.0;
    let (cx, cz) = (WORLD * 0.5, WORLD * 0.5);
    let d = ((x - cx) * (x - cx) + (z - cz) * (z - cz)).sqrt();
    if d < 160.0 { h += (160.0 - d) * 0.6; if d < 26.0 { h -= (26.0 - d) * 1.2; } }
    h
}
fn terrain_color(h: f64) -> [f32; 3] {
    if h < 6.0 { [0.12, 0.28, 0.45] } else if h < 10.0 { [0.62, 0.56, 0.34] }
    else if h < 55.0 { [0.20, 0.42, 0.18] } else if h < 90.0 { [0.38, 0.34, 0.30] }
    else { [0.82, 0.84, 0.88] }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TVertex { pos: [f32; 3], normal: [f32; 3], color: [f32; 4] }

/// Build the terrain as a single triangle mesh (u32 indices — wgpu has no small
/// drawcall cap, so one mesh is fine). Normals from finite differences.
fn build_terrain() -> (Vec<TVertex>, Vec<u32>) {
    const RES: usize = 180;
    let step = WORLD / RES as f64;
    let (mut v, mut idx) = (Vec::with_capacity((RES + 1) * (RES + 1)), Vec::new());
    for iz in 0..=RES {
        for ix in 0..=RES {
            let (x, z) = (ix as f64 * step, iz as f64 * step);
            let h = terrain_height(x, z);
            let hx = terrain_height(x + step, z) - terrain_height(x - step, z);
            let hz = terrain_height(x, z + step) - terrain_height(x, z - step);
            let n = Vec3::new((-hx / (2.0 * step)) as f32, 1.0, (-hz / (2.0 * step)) as f32).normalize();
            v.push(TVertex { pos: [x as f32, h as f32, z as f32], normal: n.to_array(), color: { let c = terrain_color(h); [c[0], c[1], c[2], 1.0] } });
        }
    }
    let w = (RES + 1) as u32;
    for iz in 0..RES as u32 {
        for ix in 0..RES as u32 {
            let a = iz * w + ix;
            idx.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
        }
    }
    (v, idx)
}

// ============================================================ gpu types

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform { vp: [[f32; 4]; 4], light: [f32; 4] }

/// Per-instance data for a GPU-skinned unit: model transform, tint, and the base
/// offset into the bone-matrix buffer (`frame * num_joints`) for its current
/// animation frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinInstance { model: [[f32; 4]; 4], color: [f32; 4], frame_base: u32, _pad: [u32; 3] }

fn faction_color(fac: u8, alive_hp: f64) -> [f32; 4] {
    let shade = (0.4 + 0.6 * (alive_hp / 100.0).clamp(0.0, 1.0)) as f32;
    if fac == 0 { [0.85 * shade, 0.22 * shade, 0.18 * shade, 1.0] } else { [0.20 * shade, 0.38 * shade, 0.9 * shade, 1.0] }
}

// ============================================================ renderer

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // GPU-skinned units
    skin_pipeline: wgpu::RenderPipeline,
    skin_vbuf: wgpu::Buffer,
    skin_ibuf: wgpu::Buffer,
    skin_nidx: u32,
    skin_inst_buf: wgpu::Buffer,
    skin_bg: wgpu::BindGroup,
    num_joints: u32,
    n_frames: u32,
    terrain_pipeline: wgpu::RenderPipeline,
    terrain_vbuf: wgpu::Buffer,
    terrain_ibuf: wgpu::Buffer,
    terrain_nidx: u32,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    depth: wgpu::TextureView,
    // scene
    units: Vec<Unit>,
    index: Tree3<IUnit>,
    rng: Rng,
    now: f64,
    last: Instant,
    yaw: f32,
    pitch: f32,
    dist: f32,
    dragging: bool,
    last_mouse: (f64, f64),
    skin_instances: Vec<SkinInstance>,
}

impl State {
    async fn new(window: Arc<winit::window::Window>) -> State {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).expect("surface");
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.expect("adapter");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.expect("device");
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: wgpu::PresentMode::AutoVsync, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
        surface.configure(&device, &config);

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<CameraUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cam-l"), entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cam-bg"), layout: &cam_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }] });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pl"), bind_group_layouts: &[&cam_layout], push_constant_ranges: &[] });

        // GPU-skinned unit model: rest mesh + a storage buffer of per-frame bone
        // matrices. The vertex shader skins on the GPU per instance.
        let skinned = vectorial_hash_demos::model::load_glb_skinned(include_bytes!("../../assets/siege/models/skeleton_a.glb"), 12, &["walk", "run"]).expect("skinned model");
        let (num_joints, n_frames) = (skinned.num_joints as u32, skinned.n_frames as u32);
        let skin_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-v"), contents: bytemuck::cast_slice(&skinned.vertices), usage: wgpu::BufferUsages::VERTEX });
        let skin_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-i"), contents: bytemuck::cast_slice(&skinned.indices), usage: wgpu::BufferUsages::INDEX });
        let skin_nidx = skinned.indices.len() as u32;
        let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bones"), contents: bytemuck::cast_slice(&skinned.joint_frames), usage: wgpu::BufferUsages::STORAGE });
        let skin_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("skin-inst"), size: (PER_FACTION * 2 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin-l"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let skin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("skin-bg"), layout: &skin_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: bone_buf.as_entire_binding() }] });
        let skin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("skin-shader"), source: wgpu::ShaderSource::Wgsl(SKIN_SHADER.into()) });
        let skin_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("skin-pl"), bind_group_layouts: &[&skin_layout], push_constant_ranges: &[] });
        let skin_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skin-pipe"),
            layout: Some(&skin_pipe_layout),
            vertex: wgpu::VertexState {
                module: &skin_shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<vectorial_hash_demos::model::SkinVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32x4, 3 => Float32x4] },
                    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<SkinInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Uint32] },
                ],
            },
            fragment: Some(wgpu::FragmentState { module: &skin_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });

        // Terrain: a single colour+normal mesh, its own pipeline (no instancing).
        let (tv, ti) = build_terrain();
        let terrain_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ter-v"), contents: bytemuck::cast_slice(&tv), usage: wgpu::BufferUsages::VERTEX });
        let terrain_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ter-i"), contents: bytemuck::cast_slice(&ti), usage: wgpu::BufferUsages::INDEX });
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ter-shader"), source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()) });
        let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ter-pipe"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState { module: &terrain_shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4] }] },
            fragment: Some(wgpu::FragmentState { module: &terrain_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });

        let depth = make_depth(&device, &config);
        let mut rng = Rng::new(0x51E6E);
        let units = spawn_army(&mut rng);
        let index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, 100.0, WORLD), 8);

        State {
            surface, device, queue, config,
            skin_pipeline, skin_vbuf, skin_ibuf, skin_nidx, skin_inst_buf, skin_bg, num_joints, n_frames,
            terrain_pipeline, terrain_vbuf, terrain_ibuf, terrain_nidx: ti.len() as u32,
            cam_buf, cam_bg, depth, units, index, rng, now: 0.0, last: Instant::now(),
            yaw: 0.9, pitch: 0.7, dist: 760.0, dragging: false, last_mouse: (0.0, 0.0),
            skin_instances: Vec::with_capacity(PER_FACTION * 2),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w; self.config.height = h;
            self.surface.configure(&self.device, &self.config);
            self.depth = make_depth(&self.device, &self.config);
        }
    }

    fn camera(&self) -> CameraUniform {
        let target = Vec3::new((WORLD * 0.5) as f32, 20.0, (WORLD * 0.5) as f32);
        let eye = target + Vec3::new(self.dist * self.pitch.cos() * self.yaw.cos(), self.dist * self.pitch.sin(), self.dist * self.pitch.cos() * self.yaw.sin());
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 1.0, 4000.0);
        CameraUniform { vp: (proj * view).to_cols_array_2d(), light: [-0.45, 0.84, -0.30, 0.0] }
    }

    fn update_and_render(&mut self) {
        let dt = self.last.elapsed().as_secs_f64().min(0.05);
        self.last = Instant::now();
        self.now += dt;
        step(&mut self.units, &mut self.index, &mut self.rng, dt, self.now);

        // Units → GPU-skinned instances: a model matrix (place · face · scale on
        // the terrain) + tint + the bone-frame base for the unit's current frame.
        self.skin_instances.clear();
        let nf = self.n_frames.max(1);
        let scale = glam::Vec3::splat(9.0); // model height
        for (i, u) in self.units.iter().enumerate() {
            if u.hp <= 0.0 { continue; }
            let y = terrain_height(u.p.x, u.p.z) as f32;
            let model = glam::Mat4::from_translation(glam::Vec3::new(u.p.x as f32, y, u.p.z as f32)) * glam::Mat4::from_rotation_y(u.face) * glam::Mat4::from_scale(scale);
            // phase-grouped frame (same trick as the macroquad version)
            let group = (i as u32 % 5) as f32 / 5.0;
            let frame = (((self.now as f32 * 1.6 + group) * nf as f32) as u32) % nf;
            self.skin_instances.push(SkinInstance { model: model.to_cols_array_2d(), color: faction_color(u.fac, u.hp), frame_base: frame * self.num_joints, _pad: [0; 3] });
        }
        self.queue.write_buffer(&self.skin_inst_buf, 0, bytemuck::cast_slice(&self.skin_instances));
        let cam = self.camera();
        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::cast_slice(&[cam]));

        let frame = match self.surface.get_current_texture() { Ok(f) => f, Err(_) => return };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.55, g: 0.68, b: 0.85, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            // terrain (single mesh)
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_bind_group(0, &self.cam_bg, &[]);
            pass.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
            pass.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.terrain_nidx, 0, 0..1);
            // units (GPU-skinned models, instanced)
            pass.set_pipeline(&self.skin_pipeline);
            pass.set_bind_group(0, &self.skin_bg, &[]);
            pass.set_vertex_buffer(0, self.skin_vbuf.slice(..));
            pass.set_vertex_buffer(1, self.skin_inst_buf.slice(..));
            pass.set_index_buffer(self.skin_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.skin_nidx, 0, 0..self.skin_instances.len() as u32);
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
    }
}

fn make_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor { label: Some("depth"), size: wgpu::Extent3d { width: config.width.max(1), height: config.height.max(1), depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[] })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

// Real GPU skeletal skinning: the vertex shader looks up this instance's bone
// matrices (frame_base + jointᵢ) from a storage buffer and blends them by weight.
// Per-instance animation, thousands of units, zero CPU vertex skinning — the
// thing macroquad's WebGL1 stack cannot do.
const SKIN_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> bones: array<mat4x4<f32>>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) joints: vec4<u32>, @location(3) weights: vec4<f32>,
      @location(4) m0: vec4<f32>, @location(5) m1: vec4<f32>, @location(6) m2: vec4<f32>, @location(7) m3: vec4<f32>,
      @location(8) icolor: vec4<f32>, @location(9) frame_base: u32) -> VOut {
    let model = mat4x4<f32>(m0, m1, m2, m3);
    var sp = vec3<f32>(0.0);
    var sn = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let w = weights[i];
        if (w > 0.0) {
            let bm = bones[frame_base + joints[i]];
            sp = sp + w * (bm * vec4<f32>(p, 1.0)).xyz;
            sn = sn + w * (bm * vec4<f32>(n, 0.0)).xyz;
        }
    }
    let world = model * vec4<f32>(sp, 1.0);
    var o: VOut;
    o.clip = cam.vp * world;
    let nn = normalize((model * vec4<f32>(sn, 0.0)).xyz);
    let diff = max(dot(nn, normalize(cam.light.xyz)), 0.0);
    let sh = 0.40 + 0.60 * diff;
    o.color = vec4<f32>(icolor.rgb * sh, 1.0);
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;

const TERRAIN_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) col: vec4<f32>) -> VOut {
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(p, 1.0);
    let diff = max(dot(normalize(n), normalize(cam.light.xyz)), 0.0);
    let sh = 0.40 + 0.60 * diff;
    o.color = vec4<f32>(col.rgb * sh, 1.0);
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash — siege (wgpu)").with_inner_size(winit::dpi::LogicalSize::new(1600, 1000)).build(&event_loop).expect("window"));
    let mut st = State::new(window.clone()).await;
    let max_frames: Option<u64> = std::env::var("SIEGE_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut frame: u64 = 0;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(s) => st.resize(s.width, s.height),
                    WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => st.dragging = state == ElementState::Pressed,
                    WindowEvent::CursorMoved { position, .. } => {
                        if st.dragging {
                            st.yaw += (position.x - st.last_mouse.0) as f32 * 0.01;
                            st.pitch = (st.pitch + (position.y - st.last_mouse.1) as f32 * 0.01).clamp(0.05, 1.5);
                        }
                        st.last_mouse = (position.x, position.y);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let d = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02 };
                        st.dist = (st.dist - d * 30.0).clamp(150.0, 1800.0);
                    }
                    WindowEvent::RedrawRequested => {
                        st.update_and_render();
                        frame += 1;
                        if let Some(m) = max_frames { if frame >= m { elwt.exit(); } }
                    }
                    _ => {}
                },
                Event::AboutToWait => window.request_redraw(),
                _ => {}
            }
        })
        .expect("run");
}
