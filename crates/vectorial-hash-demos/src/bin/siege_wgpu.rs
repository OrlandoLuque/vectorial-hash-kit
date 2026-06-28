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
    apply, decide, default_body_radius, faction_tint, model_for, set_map_seed, spawn_army,
    terrain_height, terrain_surface, volcano_step, Faction, Fx, FxKind, IUnit, Kind, ProjKind,
    Projectile, Puff, Rng, Unit, Volcano, ANIM_FRAMES, MOVE_PREFS, PER_FACTION, SKY, WORLD,
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

/// One coloured line-segment endpoint — for the combat effects (arrow / bolt /
/// lightning / ring / spark) and the projectile markers, drawn as a LineList.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LVertex { pos: [f32; 3], color: [f32; 4] }

/// A GPU-resident unit model: rest mesh + per-frame bone matrices + its own bind
/// group (camera + this model's bone storage buffer). One per distinct `.glb`.
struct GpuModel {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,
    bind: wgpu::BindGroup,
    num_joints: u32,
    n_frames: u32,
}

/// Load a unit `.glb` to the GPU for skinning (static fallback for props like the
/// cannon). The bind group keeps the bone buffer alive, so it can drop here.
fn build_gpu_model(device: &wgpu::Device, cam_buf: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, bytes: &[u8]) -> GpuModel {
    let m = vectorial_hash_demos::model::load_unit_model(bytes, ANIM_FRAMES, MOVE_PREFS);
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-v"), contents: bytemuck::cast_slice(&m.vertices), usage: wgpu::BufferUsages::VERTEX });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-i"), contents: bytemuck::cast_slice(&m.indices), usage: wgpu::BufferUsages::INDEX });
    let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bones"), contents: bytemuck::cast_slice(&m.joint_frames), usage: wgpu::BufferUsages::STORAGE });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("skin-bg"), layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bone_buf.as_entire_binding() },
    ] });
    GpuModel { vbuf, ibuf, nidx: m.indices.len() as u32, bind, num_joints: m.num_joints as u32, n_frames: m.n_frames as u32 }
}

// ============================================================ renderer

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // GPU-skinned units: one shared pipeline, one model per distinct (faction,kind)
    // glb, a shared instance buffer (units bucketed by model each frame).
    skin_pipeline: wgpu::RenderPipeline,
    models: Vec<GpuModel>,
    model_idx: [[usize; 8]; 2], // [faction][kind] → models index
    inst_buf: wgpu::Buffer,
    // The two faction keeps: a static castle model, two instances.
    castle_model: GpuModel,
    castle_inst_buf: wgpu::Buffer,
    // Combat effects + projectile markers, drawn as coloured line segments.
    line_pipeline: wgpu::RenderPipeline,
    line_buf: wgpu::Buffer,
    line_cap: usize,
    line_verts: Vec<LVertex>,
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
    red: usize,  // live Red units (for the window-title HUD)
    blue: usize, // live Blue units
    fps: f32,    // smoothed frames/second
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

        // GPU skinning: one shared bind-group layout + pipeline, then one model per
        // distinct (faction,kind) glb. The instance buffer is shared — units are
        // bucketed by model each frame.
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("skin-inst"), size: (PER_FACTION * 2 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin-l"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
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
                    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<vectorial_hash_demos::model::SkinVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32x4, 3 => Float32x4, 10 => Float32x4] },
                    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<SkinInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Uint32] },
                ],
            },
            fragment: Some(wgpu::FragmentState { module: &skin_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });

        // One GpuModel per distinct (faction,kind) glb — the dragon and the cannon
        // are shared across factions, deduped by the byte-slice pointer.
        let mut models: Vec<GpuModel> = Vec::new();
        let mut model_idx = [[0usize; 8]; 2];
        let mut seen: Vec<(*const u8, usize)> = Vec::new();
        for f in Faction::ALL {
            for k in Kind::ALL {
                let bytes = model_for(f, k);
                let ptr = bytes.as_ptr();
                let idx = match seen.iter().find(|(p, _)| *p == ptr) {
                    Some(&(_, i)) => i,
                    None => { let i = models.len(); models.push(build_gpu_model(&device, &cam_buf, &skin_layout, bytes)); seen.push((ptr, i)); i }
                };
                model_idx[f.index()][k.index()] = idx;
            }
        }

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

        // Castle model (static) for the two keeps — two fixed instances, facing
        // the map centre, tinted by faction. Reuses the skin pipeline (1 identity
        // joint via the static fallback).
        let castle_model = build_gpu_model(&device, &cam_buf, &skin_layout, include_bytes!("../../assets/siege/models/castle.glb"));
        let castle_inst: Vec<SkinInstance> = Faction::ALL.iter().map(|&f| {
            let (cx, cz) = f.castle();
            let yaw = ((WORLD * 0.5 - cx) as f32).atan2((WORLD * 0.5 - cz) as f32);
            let m = glam::Mat4::from_translation(glam::Vec3::new(cx as f32, terrain_height(cx, cz) as f32, cz as f32)) * glam::Mat4::from_rotation_y(yaw) * glam::Mat4::from_scale(glam::Vec3::splat(62.0));
            SkinInstance { model: m.to_cols_array_2d(), color: faction_tint(f), frame_base: 0, _pad: [0; 3] }
        }).collect();
        let castle_inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("castle-inst"), contents: bytemuck::cast_slice(&castle_inst), usage: wgpu::BufferUsages::VERTEX });

        // Line pipeline for combat effects + projectile markers (LineList, unlit,
        // alpha-blended, depth-tested but not depth-writing).
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("line-shader"), source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()) });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line-pipe"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState { module: &line_shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<LVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4] }] },
            fragment: Some(wgpu::FragmentState { module: &line_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });
        let line_cap = 8192usize;
        let line_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("line-v"), size: (line_cap * std::mem::size_of::<LVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let depth = make_depth(&device, &config);
        // Per-run seed (reproducible via $SIEGE_SEED) drives the shared map + army.
        let seed = std::env::var("SIEGE_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0x51E6E);
        set_map_seed((seed % 100_000) as f64 * 0.01);
        let mut rng = Rng::new(seed | 1);
        let units = spawn_army(&mut rng, PER_FACTION);
        let index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);

        State {
            surface, device, queue, config,
            skin_pipeline, models, model_idx, inst_buf,
            castle_model, castle_inst_buf,
            line_pipeline, line_buf, line_cap, line_verts: Vec::new(),
            terrain_pipeline, terrain_vbuf, terrain_ibuf, terrain_nidx: ti.len() as u32,
            cam_buf, cam_bg, depth, units, index,
            smoke: Vec::new(), effects: Vec::new(), projectiles: Vec::new(),
            volcano: Volcano::new(), body_radius: default_body_radius(), paused: false,
            red: 0, blue: 0, fps: 0.0,
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
                if u.alive() { self.index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face }); }
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

        // HUD stats (shown in the window title — wgpu has no built-in text).
        let inst = (1.0 / dt.max(1e-3)) as f32;
        self.fps = if self.fps == 0.0 { inst } else { self.fps * 0.92 + inst * 0.08 };
        self.red = self.units.iter().filter(|u| u.alive() && u.faction == Faction::Red).count();
        self.blue = self.units.iter().filter(|u| u.alive() && u.faction == Faction::Blue).count();

        // Units → GPU-skinned instances, bucketed by model so each distinct glb
        // draws in one call. Per-model frame_base uses that model's own joint +
        // frame count; the instance colour is the faction tint (the shader mixes
        // the model's own colours toward it). Model height varies by kind.
        let mut buckets: Vec<Vec<SkinInstance>> = (0..self.models.len()).map(|_| Vec::new()).collect();
        for (i, u) in self.units.iter().enumerate() {
            if !u.alive() { continue; }
            let mi = self.model_idx[u.faction.index()][u.kind.index()];
            let m = &self.models[mi];
            let y = (terrain_height(u.p.x, u.p.z) + u.kind.altitude()) as f32; // feet on terrain (dragon flies)
            let model = glam::Mat4::from_translation(glam::Vec3::new(u.p.x as f32, y, u.p.z as f32)) * glam::Mat4::from_rotation_y(u.face) * glam::Mat4::from_scale(glam::Vec3::splat(u.kind.model_height()));
            let nf = m.n_frames.max(1);
            let group = (i as u32 % 5) as f32 / 5.0; // phase-grouped frame (as macroquad)
            let frame = (((self.now as f32 * 1.6 + group) * nf as f32) as u32) % nf;
            buckets[mi].push(SkinInstance { model: model.to_cols_array_2d(), color: faction_tint(u.faction), frame_base: frame * m.num_joints, _pad: [0; 3] });
        }
        // Flatten into the shared instance buffer, remembering each model's range.
        self.skin_instances.clear();
        let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let start = self.skin_instances.len() as u32;
            self.skin_instances.extend_from_slice(b);
            ranges.push((start, self.skin_instances.len() as u32));
        }
        self.queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&self.skin_instances));
        let cam = self.camera();
        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::cast_slice(&[cam]));

        // Combat effects + projectile markers → coloured line segments (built into
        // a detached Vec so the loops can read self.effects/self.projectiles).
        let mut lv = std::mem::take(&mut self.line_verts);
        lv.clear();
        let push = |v: &mut Vec<LVertex>, a: glam::Vec3, b: glam::Vec3, c: [f32; 4]| {
            v.push(LVertex { pos: a.to_array(), color: c });
            v.push(LVertex { pos: b.to_array(), color: c });
        };
        for f in &self.effects {
            let age = ((self.now - f.born) / Fx::life(f.kind)).clamp(0.0, 1.0) as f32;
            let (a, b) = (glam::Vec3::from_array(f.a), glam::Vec3::from_array(f.b));
            match f.kind {
                FxKind::Arrow => push(&mut lv, a, b, [0.96, 0.90, 0.45, 1.0]),
                FxKind::Bolt => push(&mut lv, a, b, [1.0, 0.58, 0.16, 1.0]),
                FxKind::Lightning => push(&mut lv, a, b, [0.62, 0.86, 1.0, 1.0]),
                FxKind::Spark => push(&mut lv, a, a + glam::Vec3::Y * 7.0, [0.40, 1.0, 0.55, 1.0]),
                FxKind::Ring => {
                    let (r, col, n) = (8.0 + 26.0 * age, [1.0, 0.45, 0.12, 1.0 - age], 20);
                    let mut prev = a + glam::Vec3::new(r, 1.5, 0.0);
                    for i in 1..=n {
                        let t = i as f32 / n as f32 * std::f32::consts::TAU;
                        let p = a + glam::Vec3::new(r * t.cos(), 1.5, r * t.sin());
                        push(&mut lv, prev, p, col);
                        prev = p;
                    }
                }
            }
        }
        for pr in &self.projectiles {
            let c = glam::Vec3::new(pr.p.x as f32, pr.p.y as f32, pr.p.z as f32);
            let col = match pr.kind { ProjKind::Cannon => [0.08, 0.08, 0.08, 1.0], ProjKind::LavaRock => [1.0, 0.45, 0.10, 1.0] };
            let r = 2.2;
            push(&mut lv, c - glam::Vec3::X * r, c + glam::Vec3::X * r, col);
            push(&mut lv, c - glam::Vec3::Y * r, c + glam::Vec3::Y * r, col);
            push(&mut lv, c - glam::Vec3::Z * r, c + glam::Vec3::Z * r, col);
        }
        self.line_verts = lv;
        // Grow the line buffer if this frame overflows it.
        if self.line_verts.len() > self.line_cap {
            self.line_cap = (self.line_verts.len() * 2).next_power_of_two();
            self.line_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("line-v"), size: (self.line_cap * std::mem::size_of::<LVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        }
        if !self.line_verts.is_empty() { self.queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&self.line_verts)); }
        let line_n = self.line_verts.len() as u32;

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
            // units — one GPU-skinned draw per distinct model, instances bucketed.
            pass.set_pipeline(&self.skin_pipeline);
            pass.set_vertex_buffer(1, self.inst_buf.slice(..));
            for (mi, m) in self.models.iter().enumerate() {
                let (s, e) = ranges[mi];
                if s == e { continue; }
                pass.set_bind_group(0, &m.bind, &[]);
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.nidx, 0, s..e);
            }
            // castles — the static model, two instances (one per keep).
            let cm = &self.castle_model;
            pass.set_bind_group(0, &cm.bind, &[]);
            pass.set_vertex_buffer(0, cm.vbuf.slice(..));
            pass.set_vertex_buffer(1, self.castle_inst_buf.slice(..));
            pass.set_index_buffer(cm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cm.nidx, 0, 0..2);
            // combat effects + projectile markers (alpha-blended lines, drawn last).
            if line_n > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.cam_bg, &[]);
                pass.set_vertex_buffer(0, self.line_buf.slice(..));
                pass.draw(0..line_n, 0..1);
            }
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
      @location(10) vcolor: vec4<f32>,
      @location(4) m0: vec4<f32>, @location(5) m1: vec4<f32>, @location(6) m2: vec4<f32>, @location(7) m3: vec4<f32>,
      @location(8) tint: vec4<f32>, @location(9) frame_base: u32) -> VOut {
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
    // The model's own colour, nudged toward the faction tint by the tint's alpha
    // (so even dark models read clearly as Red / Blue) — matches the macroquad mix.
    let base = mix(vcolor.rgb, tint.rgb, tint.a);
    o.color = vec4<f32>(base * sh, 1.0);
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

// Unlit coloured lines for the combat effects + projectile markers.
const LINE_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(p, 1.0);
    o.color = color;
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
                        if frame % 15 == 0 {
                            window.set_title(&format!("vectorial-hash — siege (wgpu) · Red {} | Blue {} · {:.0} fps{}", st.red, st.blue, st.fps, if st.paused { " · PAUSED" } else { "" }));
                        }
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
