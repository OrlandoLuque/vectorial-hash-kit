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
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

use vectorial_hash::{Aabb, Tree3};
// The whole battle simulation is shared with the macroquad `siege` binary so the
// two renderers stay in lockstep — see `siege_sim`. This file is wgpu render-only.
use rayon::prelude::*;
use vectorial_hash_demos::siege_sim::{
    apply, decide, default_body_radius, faction_tint, set_map_seed, spawn_army, terrain_height,
    terrain_surface, volcano_step, Faction, Fx, IUnit, Projectile, Puff, Rng, Unit, Volcano,
    PER_FACTION, SKY, WORLD,
};

// ============================================================ terrain

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TVertex { pos: [f32; 3], normal: [f32; 3], color: [f32; 4] }

/// Build the terrain as a single triangle mesh from the shared heightfield (u32
/// indices — wgpu has no small drawcall cap, so one mesh is fine). Normals from
/// finite differences; lava cells are flagged emissive (full-bright) via alpha=0.
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
            let (c, emissive) = terrain_surface(x, z, h); // shared ramp + lava flag
            v.push(TVertex { pos: [x as f32, h as f32, z as f32], normal: n.to_array(), color: [c[0], c[1], c[2], if emissive { 0.0 } else { 1.0 }] });
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

/// Per-unit instance tint: the shared faction colour, dimmed as the unit loses HP.
fn faction_color(f: Faction, hp_frac: f64) -> [f32; 4] {
    let shade = (0.45 + 0.55 * hp_frac.clamp(0.0, 1.0)) as f32;
    let t = faction_tint(f);
    [t[0] * shade, t[1] * shade, t[2] * shade, 1.0]
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
    // scene — the shared simulation (same battle as the macroquad binary)
    units: Vec<Unit>,
    index: Tree3<IUnit>,
    smoke: Vec<Puff>,
    effects: Vec<Fx>,
    projectiles: Vec<Projectile>,
    volcano: Volcano,
    body_radius: [[f64; 8]; 2],
    paused: bool,
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
        // Per-run seed (reproducible via $SIEGE_SEED) drives the shared map + army.
        let seed = std::env::var("SIEGE_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0x51E6E);
        set_map_seed((seed % 100_000) as f64 * 0.01);
        let mut rng = Rng::new(seed | 1);
        let units = spawn_army(&mut rng, PER_FACTION);
        let index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);

        State {
            surface, device, queue, config,
            skin_pipeline, skin_vbuf, skin_ibuf, skin_nidx, skin_inst_buf, skin_bg, num_joints, n_frames,
            terrain_pipeline, terrain_vbuf, terrain_ibuf, terrain_nidx: ti.len() as u32,
            cam_buf, cam_bg, depth, units, index,
            smoke: Vec::new(), effects: Vec::new(), projectiles: Vec::new(),
            volcano: Volcano::new(), body_radius: default_body_radius(), paused: false,
            rng, now: 0.0, last: Instant::now(),
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
        if !self.paused {
            self.now += dt;
            // Rebuild the unit index from live positions each frame.
            self.index.clear();
            for (i, u) in self.units.iter().enumerate() {
                if u.alive() { self.index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32 }); }
            }
            // Smoke index (LoS blockers) for the archer / ballista raycasts.
            let mut smoke_index = Tree3::<Puff>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);
            for s in &self.smoke { smoke_index.insert(*s); }
            // Decide (parallel, read-only on the indices) → apply (serial) → volcano.
            let (idx, smk, br) = (&self.index, &smoke_index, &self.body_radius);
            self.units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, br));
            apply(&mut self.units, &mut self.smoke, &mut self.effects, &mut self.projectiles, &mut self.rng, dt, self.now);
            volcano_step(&mut self.volcano, &mut self.smoke, &mut self.effects, &mut self.projectiles, &mut self.rng, dt, self.now);
        }

        // Units → GPU-skinned instances: a model matrix (place · face · scale on
        // the terrain) + tint + the bone-frame base for the unit's current frame.
        // (Single shared model for now — per-kind/faction models are the next step.)
        self.skin_instances.clear();
        let nf = self.n_frames.max(1);
        let scale = glam::Vec3::splat(9.0); // model height
        for (i, u) in self.units.iter().enumerate() {
            if !u.alive() { continue; }
            let y = (terrain_height(u.p.x, u.p.z) + u.kind.altitude()) as f32; // feet on terrain (dragon flies)
            let model = glam::Mat4::from_translation(glam::Vec3::new(u.p.x as f32, y, u.p.z as f32)) * glam::Mat4::from_rotation_y(u.face) * glam::Mat4::from_scale(scale);
            // phase-grouped frame (same trick as the macroquad version)
            let group = (i as u32 % 5) as f32 / 5.0;
            let frame = (((self.now as f32 * 1.6 + group) * nf as f32) as u32) % nf;
            self.skin_instances.push(SkinInstance { model: model.to_cols_array_2d(), color: faction_color(u.faction, u.hp / u.kind.max_hp()), frame_base: frame * self.num_joints, _pad: [0; 3] });
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
    // alpha < 0.5 flags an emissive (lava) cell → draw it full-bright.
    let sh = select(0.40 + 0.60 * diff, 1.0, col.a < 0.5);
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
                    WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(KeyCode::KeyP), state: ElementState::Pressed, .. }, .. } => st.paused = !st.paused,
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
