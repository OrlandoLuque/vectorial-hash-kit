//! `formations_wgpu` — the Total War-style **automatic army battle** rendered
//! with wgpu (GPU skeletal skinning, the siege/horde machinery). Two armies
//! deploy in regiment lines and fight to a rout: melee pairing, flank/rear
//! morale, cavalry charges, archer volleys with honest friendly fire, chain
//! routs — every mechanic a library query (see `formations_sim` + the design
//! doc `docs/FORMATIONS_DESIGN.md`). This file is render-only.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin formations_wgpu --release --features parallel`
//!
//! drag: orbit · scroll: zoom · `P`: pause · `K`: frustum cull · `F`:
//! free-fly (WASD/QE) · `[` `]` + slider: army size · thread slider (native).

// Native always; on wasm only with the `web-wgpu` feature (WebGPU via wasm-bindgen).
#![cfg(any(not(target_arch = "wasm32"), feature = "web-wgpu"))]
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

use vectorial_hash_demos::formations_sim::{
    default_pop, ground_h, Faction, Formations, RKind, RState, WORLD,
};
// Generic glb-baking knobs shared with the siege/horde loaders.
use vectorial_hash_demos::siege_sim::{ANIM_FRAMES, MOVE_PREFS};

const MAX_POP: usize = 4_000; // per-side ceiling for the slider (instance-buffer cap)
const CORPSE_DRAW: usize = 16_000; // most-recent fallen drawn (flattened dark)

/// Fetch a file over HTTP (web build) — streams the `.glb` models at startup.
#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Vec<u8> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    let win = web_sys::window().expect("no window");
    let resp: web_sys::Response = JsFuture::from(win.fetch_with_str(url)).await
        .unwrap_or_else(|e| panic!("fetch {url}: {e:?}")).dyn_into().expect("not a Response");
    let buf = JsFuture::from(resp.array_buffer().expect("array_buffer")).await.expect("await array_buffer");
    js_sys::Uint8Array::new(&buf).to_vec()
}

// ============================================================ casting
// Pirates (Red) vs undead (Blue), reusing the Quaternius CC0 glbs the siege
// demo already ships. Two baked clips per casting entry: the MOVE clip and the
// ATTACK clip (soldiers in an Engage(d) regiment swing; everyone else walks).

const UNIT_FILES: [&str; 8] = [
    "pirate_captain.glb", // Red Sword + General
    "henry.glb",          // Red Spear
    "anne.glb",           // Red Archer
    "sharky.glb",         // Red Cavalry rider
    "skeleton_sword.glb", // Blue Sword
    "skeleton_a.glb",     // Blue Spear + Blue General
    "witch.glb",          // Blue Archer
    "zombie.glb",         // Blue Cavalry rider
];
fn model_slot(f: Faction, k: RKind) -> usize {
    let base = if f == Faction::Red { 0 } else { 4 };
    base + match k { RKind::Sword | RKind::General => 0, RKind::Spear => 1, RKind::Archer => 2, RKind::Cavalry => 3 }
}
fn faction_tint(f: Faction) -> [f32; 4] {
    match f { Faction::Red => [0.85, 0.30, 0.22, 0.30], Faction::Blue => [0.30, 0.42, 0.90, 0.30] }
}

// ============================================================ terrain

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TVertex { pos: [f32; 3], normal: [f32; 3], color: [f32; 4] }

/// One smooth battlefield mesh: near-flat grass with mowed stripes (fields), a
/// worn dirt band across the middle where the lines will meet.
fn build_terrain(seed: f64) -> (Vec<TVertex>, Vec<u32>) {
    const RES: usize = 200;
    let step = WORLD / RES as f64;
    let cz = WORLD / 2.0;
    let (mut v, mut idx) = (Vec::with_capacity((RES + 1) * (RES + 1)), Vec::new());
    let mix3 = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    for iz in 0..=RES {
        for ix in 0..=RES {
            let (x, z) = (ix as f64 * step, iz as f64 * step);
            let h = ground_h(x, z, seed);
            let hx = ground_h(x + step, z, seed) - ground_h(x - step, z, seed);
            let hz = ground_h(x, z + step, seed) - ground_h(x, z - step, seed);
            let n = Vec3::new((-hx / (2.0 * step)) as f32, 1.0, (-hz / (2.0 * step)) as f32).normalize();
            // mowed-field stripes + the worn centre band
            let stripe = 0.5 + 0.5 * ((x * 0.045).sin() * 0.5 + 0.5);
            let mut c = mix3([0.30, 0.45, 0.23], [0.34, 0.50, 0.26], stripe as f32);
            let worn = (1.0 - ((z - cz).abs() / 120.0).min(1.0)) as f32;
            c = mix3(c, [0.48, 0.42, 0.30], worn * 0.55);
            v.push(TVertex { pos: [x as f32, h as f32, z as f32], normal: n.to_array(), color: [c[0], c[1], c[2], 1.0] });
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinInstance { model: [[f32; 4]; 4], color: [f32; 4], frame_base: u32, _pad: [u32; 3] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LVertex { pos: [f32; 3], color: [f32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }

const UI_BAR: (f32, f32, f32, f32) = (18.0, 18.0, 280.0, 16.0);      // red|blue strength
const UI_SLIDER: (f32, f32, f32, f32) = (18.0, 46.0, 280.0, 12.0);   // per-side size
const UI_THREADS: (f32, f32, f32, f32) = (18.0, 70.0, 280.0, 12.0);  // rayon pool
const UI_PAUSE: (f32, f32, f32, f32) = (18.0, 92.0, 96.0, 30.0);     // pause button

#[allow(clippy::too_many_arguments)]
fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let (x0, x1) = (px / sw * 2.0 - 1.0, (px + w) / sw * 2.0 - 1.0);
    let (y0, y1) = (1.0 - py / sh * 2.0, 1.0 - (py + h) / sh * 2.0);
    let q = |x, y| UiVertex { pos: [x, y], color };
    v.extend_from_slice(&[q(x0, y0), q(x1, y0), q(x1, y1), q(x0, y0), q(x1, y1), q(x0, y1)]);
}

fn frustum_planes(vp: glam::Mat4) -> [glam::Vec4; 6] {
    let (r0, r1, r2, r3) = (vp.row(0), vp.row(1), vp.row(2), vp.row(3));
    let mut p = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2];
    for pl in &mut p { let n = pl.truncate().length().max(1e-6); *pl /= n; }
    p
}
fn sphere_in_frustum(planes: &[glam::Vec4; 6], c: glam::Vec3, r: f32) -> bool {
    planes.iter().all(|pl| pl.x * c.x + pl.y * c.y + pl.z * c.z + pl.w >= -r)
}

/// A GPU-resident model: rest mesh + per-frame bone matrices + its bind group.
struct GpuModel {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,
    bind: wgpu::BindGroup,
    num_joints: u32,
    n_frames: u32,
}

fn upload_skinned(device: &wgpu::Device, cam_buf: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, m: &vectorial_hash_demos::model::SkinnedModel) -> GpuModel {
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-v"), contents: bytemuck::cast_slice(&m.vertices), usage: wgpu::BufferUsages::VERTEX });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin-i"), contents: bytemuck::cast_slice(&m.indices), usage: wgpu::BufferUsages::INDEX });
    let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bones"), contents: bytemuck::cast_slice(&m.joint_frames), usage: wgpu::BufferUsages::STORAGE });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("skin-bg"), layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bone_buf.as_entire_binding() },
    ] });
    GpuModel { vbuf, ibuf, nidx: m.indices.len() as u32, bind, num_joints: m.num_joints as u32, n_frames: m.n_frames as u32 }
}

/// Load a `.glb` baked with a clip picked by `prefs` (MOVE_PREFS = walk/run,
/// or the attack list) — one GpuModel per (file, clip).
fn build_gpu_model(device: &wgpu::Device, cam_buf: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, bytes: &[u8], prefs: &[&str]) -> GpuModel {
    let m = vectorial_hash_demos::model::load_unit_model(bytes, ANIM_FRAMES, prefs);
    upload_skinned(device, cam_buf, layout, &m)
}

/// A unit box (base at y=0, unit height, white) for the regiment banners —
/// same trick as horde's structures: 1 identity joint through the skin pipeline.
fn unit_box() -> vectorial_hash_demos::model::SkinnedModel {
    use vectorial_hash_demos::model::{SkinVertex, SkinnedModel};
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[-0.5, 1.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5]]),
        ([0.0, -1.0, 0.0], [[-0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 0.0, -0.5]]),
        ([1.0, 0.0, 0.0], [[0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 1.0, -0.5]]),
        ([-1.0, 0.0, 0.0], [[-0.5, 0.0, 0.5], [-0.5, 0.0, -0.5], [-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5]]),
        ([0.0, 0.0, 1.0], [[0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [-0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]),
        ([0.0, 0.0, -1.0], [[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [-0.5, 1.0, -0.5]]),
    ];
    for (n, quad) in faces {
        let base = verts.len() as u32;
        for p in quad { verts.push(SkinVertex { pos: p, normal: n, joints: [0; 4], weights: [1.0, 0.0, 0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] }); }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    SkinnedModel { vertices: verts, indices: idx, joint_frames: vec![glam::Mat4::IDENTITY.to_cols_array_2d()], num_joints: 1, n_frames: 1 }
}

// ============================================================ renderer

struct State {
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    skin_pipeline: wgpu::RenderPipeline,
    move_models: Vec<GpuModel>,   // UNIT_FILES order, MOVE clip
    attack_models: Vec<GpuModel>, // UNIT_FILES order, ATTACK clip
    horse_move: GpuModel,
    inst_buf: wgpu::Buffer,
    box_model: GpuModel, // banners (pole + flag)
    box_inst_buf: wgpu::Buffer,
    line_pipeline: wgpu::RenderPipeline,
    line_buf: wgpu::Buffer,
    line_cap: usize,
    ui_pipeline: wgpu::RenderPipeline,
    ui_buf: wgpu::Buffer,
    ui_drag: u8,
    #[cfg(not(target_arch = "wasm32"))]
    pool: rayon::ThreadPool,
    #[cfg(not(target_arch = "wasm32"))]
    n_threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    max_threads: usize,
    terrain_pipeline: wgpu::RenderPipeline,
    terrain_vbuf: wgpu::Buffer,
    terrain_ibuf: wgpu::Buffer,
    terrain_nidx: u32,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    depth: wgpu::TextureView,
    sim: Formations,
    seed: u64,
    pop: usize,
    paused: bool,
    frustum_cull: bool,
    fps: f32,
    last: Instant,
    yaw: f32,
    pitch: f32,
    dist: f32,
    dragging: bool,
    last_mouse: (f64, f64),
    free_cam: bool,
    cam_pos: Vec3,
    mv: [bool; 6],
    skin_instances: Vec<SkinInstance>,
}

fn make_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: config.width.max(1), height: config.height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

impl State {
    async fn new(window: Arc<winit::window::Window>) -> State {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = Some(instance.create_surface(window).expect("surface"));
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: surface.as_ref(), force_fallback_adapter: false }).await.expect("adapter");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.expect("device");
        let (format, alpha) = { let s = surface.as_ref().unwrap(); let caps = s.get_capabilities(&adapter); (caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]), caps.alpha_modes[0]) };
        let present = if std::env::var("FORM_NOVSYNC").is_ok() { wgpu::PresentMode::AutoNoVsync } else { wgpu::PresentMode::AutoVsync };
        let config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: present, desired_maximum_frame_latency: 2, alpha_mode: alpha, view_formats: vec![] };
        if let Some(s) = &surface { s.configure(&device, &config); }

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<CameraUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cam-l"), entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cam-bg"), layout: &cam_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }] });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pl"), bind_group_layouts: &[&cam_layout], push_constant_ranges: &[] });

        // The one skinning pipeline (same as siege/horde).
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("skin-inst"), size: ((MAX_POP * 2 + CORPSE_DRAW + 512) * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin-l"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
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

        // Models: native embeds; web fetches from models/ (the Pages site
        // already hosts these exact files for siege/horde).
        #[cfg(target_arch = "wasm32")]
        let table: std::collections::HashMap<&'static str, Vec<u8>> = {
            let mut m = std::collections::HashMap::new();
            for name in UNIT_FILES.iter().copied().chain(["horse.glb"]) {
                m.insert(name, fetch_bytes(&format!("models/{name}")).await);
            }
            m
        };
        #[cfg(not(target_arch = "wasm32"))]
        let bytes_of = |name: &str| -> Vec<u8> {
            match name {
                "pirate_captain.glb" => include_bytes!("../../assets/siege/models/pirate_captain.glb").to_vec(),
                "henry.glb" => include_bytes!("../../assets/siege/models/henry.glb").to_vec(),
                "anne.glb" => include_bytes!("../../assets/siege/models/anne.glb").to_vec(),
                "sharky.glb" => include_bytes!("../../assets/siege/models/sharky.glb").to_vec(),
                "skeleton_sword.glb" => include_bytes!("../../assets/siege/models/skeleton_sword.glb").to_vec(),
                "skeleton_a.glb" => include_bytes!("../../assets/siege/models/skeleton_a.glb").to_vec(),
                "witch.glb" => include_bytes!("../../assets/siege/models/witch.glb").to_vec(),
                "zombie.glb" => include_bytes!("../../assets/siege/models/zombie.glb").to_vec(),
                "horse.glb" => include_bytes!("../../assets/siege/models/horse.glb").to_vec(),
                _ => unreachable!("unknown formations model {name}"),
            }
        };
        #[cfg(target_arch = "wasm32")]
        let bytes_of = |name: &str| -> Vec<u8> { table[name].clone() };

        const ATTACK_PREFS: [&str; 6] = ["attack", "punch", "swing", "shoot", "bite", "walk"];
        let move_models: Vec<GpuModel> = UNIT_FILES.iter().map(|f| build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of(f), MOVE_PREFS)).collect();
        let attack_models: Vec<GpuModel> = UNIT_FILES.iter().map(|f| build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of(f), &ATTACK_PREFS)).collect();
        let horse_move = build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of("horse.glb"), MOVE_PREFS);
        let box_model = upload_skinned(&device, &cam_buf, &skin_layout, &unit_box());
        let box_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("box-inst"), size: (512 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let seed = std::env::var("FORM_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0xF0421);
        let pop = std::env::var("FORM_POP").ok().and_then(|s| s.parse().ok()).unwrap_or(default_pop()).clamp(200, MAX_POP);
        let sim = Formations::new(seed, pop);

        let (tv, ti) = build_terrain(sim.seed);
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
        let line_cap = 16_384usize;
        let line_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("line-v"), size: (line_cap * std::mem::size_of::<LVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ui-shader"), source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
        let ui_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ui-pl"), bind_group_layouts: &[], push_constant_ranges: &[] });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui-pipe"),
            layout: Some(&ui_pipe_layout),
            vertex: wgpu::VertexState { module: &ui_shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
            fragment: Some(wgpu::FragmentState { module: &ui_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });
        let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui-v"), size: (4096 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let depth = make_depth(&device, &config);
        #[cfg(not(target_arch = "wasm32"))]
        let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        #[cfg(not(target_arch = "wasm32"))]
        let pool = rayon::ThreadPoolBuilder::new().num_threads(max_threads).build().unwrap();

        State {
            surface, device, queue, config,
            skin_pipeline, move_models, attack_models, horse_move, inst_buf,
            box_model, box_inst_buf,
            line_pipeline, line_buf, line_cap,
            ui_pipeline, ui_buf, ui_drag: 0,
            #[cfg(not(target_arch = "wasm32"))]
            pool,
            #[cfg(not(target_arch = "wasm32"))]
            n_threads: max_threads,
            #[cfg(not(target_arch = "wasm32"))]
            max_threads,
            terrain_pipeline, terrain_vbuf, terrain_ibuf, terrain_nidx: ti.len() as u32,
            cam_buf, cam_bg, depth,
            sim, seed, pop,
            paused: false, frustum_cull: true, fps: 0.0, last: Instant::now(),
            yaw: 1.57, pitch: 0.62, dist: 620.0, dragging: false, last_mouse: (0.0, 0.0),
            free_cam: false, cam_pos: Vec3::ZERO, mv: [false; 6],
            skin_instances: Vec::with_capacity(MAX_POP * 2 + CORPSE_DRAW),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w; self.config.height = h;
            if let Some(s) = &self.surface { s.configure(&self.device, &self.config); }
            self.depth = make_depth(&self.device, &self.config);
        }
    }

    /// Fresh armies at a new per-side size (slider / [ ] keys).
    fn set_population(&mut self, pop: usize) {
        let pop = pop.clamp(200, MAX_POP);
        if pop == self.pop { return; }
        self.pop = pop;
        self.sim = Formations::new(self.seed, pop);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn set_threads(&mut self, n: usize) {
        let n = n.clamp(1, self.max_threads);
        if n == self.n_threads { return; }
        self.n_threads = n;
        self.pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    }

    fn apply_slider(&mut self, which: u8, mx: f32) {
        match which {
            1 => { let (tx, _, tw, _) = UI_SLIDER; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_population(200 + (frac * (MAX_POP as f32 - 200.0)) as usize); }
            #[cfg(not(target_arch = "wasm32"))]
            2 => { let (tx, _, tw, _) = UI_THREADS; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_threads(1 + (frac * (self.max_threads as f32 - 1.0)).round() as usize); }
            _ => {}
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(-self.pitch.cos() * self.yaw.cos(), -self.pitch.sin(), -self.pitch.cos() * self.yaw.sin()).normalize_or_zero()
    }

    fn camera(&self) -> CameraUniform {
        let (eye, target) = if self.free_cam {
            (self.cam_pos, self.cam_pos + self.forward())
        } else {
            let target = Vec3::new((WORLD * 0.5) as f32, 8.0, (WORLD * 0.5) as f32);
            (target + Vec3::new(self.dist * self.pitch.cos() * self.yaw.cos(), self.dist * self.pitch.sin(), self.dist * self.pitch.cos() * self.yaw.sin()), target)
        };
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 1.0, 4200.0);
        CameraUniform { vp: (proj * view).to_cols_array_2d(), light: [-0.45, 0.84, -0.30, 0.0] }
    }

    fn update_and_render(&mut self) {
        let dt = { let d = self.last.elapsed().as_secs_f64().min(0.05); self.last = Instant::now(); d };
        if self.free_cam {
            let fwd = self.forward();
            let right = fwd.cross(Vec3::Y).normalize_or_zero();
            let sp = 260.0 * dt as f32;
            let mut mv = Vec3::ZERO;
            if self.mv[0] { mv += fwd; } if self.mv[1] { mv -= fwd; }
            if self.mv[2] { mv -= right; } if self.mv[3] { mv += right; }
            if self.mv[4] { mv -= Vec3::Y; } if self.mv[5] { mv += Vec3::Y; }
            self.cam_pos += mv.normalize_or_zero() * sp;
        }
        if !self.paused {
            #[cfg(not(target_arch = "wasm32"))]
            { let sim = &mut self.sim; self.pool.install(|| sim.step(dt)); }
            #[cfg(target_arch = "wasm32")]
            self.sim.step(dt);
        }
        let now = self.sim.now;

        let inst_fps = (1.0 / dt.max(1e-3)) as f32;
        self.fps = if self.fps == 0.0 { inst_fps } else { self.fps * 0.92 + inst_fps * 0.08 };

        // ---- soldiers → skinned instances, bucketed by (model, clip).
        let cam = self.camera();
        let planes = frustum_planes(Mat4::from_cols_array_2d(&cam.vp));
        let do_cull = self.frustum_cull;
        let nm = self.move_models.len();
        // buckets: [0..nm) move-clip, [nm..2nm) attack-clip, [2nm] horses
        let mut buckets: Vec<Vec<SkinInstance>> = (0..nm * 2 + 1).map(|_| Vec::new()).collect();
        let scale_of = |k: RKind| -> f32 { match k { RKind::General => 9.0, RKind::Cavalry => 5.2, _ => 7.0 } };
        for (i, s) in self.sim.soldiers.iter().enumerate() {
            if !s.alive() { continue; }
            let p = Vec3::new(s.p.x as f32, s.p.y as f32, s.p.z as f32);
            if do_cull && !sphere_in_frustum(&planes, p, 12.0) { continue; }
            let r = &self.sim.regiments[s.regiment as usize];
            let slot = model_slot(s.faction, r.kind);
            // Engaged regiments swing; routers run away (walk clip, faced back).
            let attacking = r.state == RState::Engage;
            let mi = if attacking { nm + slot } else { slot };
            let m = if attacking { &self.attack_models[slot] } else { &self.move_models[slot] };
            let face = if matches!(r.state, RState::Routing | RState::Shattered) { r.facing + std::f32::consts::PI as f64 } else { r.facing };
            // Model yaw: glbs face +Z at yaw 0; sim facing is the +X-based angle.
            let yaw = -(face as f32) + std::f32::consts::FRAC_PI_2;
            let mut y = p.y;
            let mut scale = scale_of(r.kind);
            if r.kind == RKind::Cavalry || r.kind == RKind::General { y += 3.4; scale *= 0.74; } // rider on the horse
            let model = Mat4::from_translation(Vec3::new(p.x, y, p.z)) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(scale));
            let nf = m.n_frames.max(1);
            let group = (i as u32 % 7) as f32 / 7.0;
            let frame = (((now as f32 * 1.6 + group) * nf as f32) as u32) % nf;
            buckets[mi].push(SkinInstance { model: model.to_cols_array_2d(), color: faction_tint(s.faction), frame_base: frame * m.num_joints, _pad: [0; 3] });
            // The mount under every cavalry/general rider.
            if r.kind == RKind::Cavalry || r.kind == RKind::General {
                let hm = Mat4::from_translation(p) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(scale_of(r.kind)));
                let hnf = self.horse_move.n_frames.max(1);
                let hframe = (((now as f32 * 1.6 + group) * hnf as f32) as u32) % hnf;
                buckets[nm * 2].push(SkinInstance { model: hm.to_cols_array_2d(), color: faction_tint(s.faction), frame_base: hframe * self.horse_move.num_joints, _pad: [0; 3] });
            }
        }
        // Corpses: flattened dark rest-pose instances of the move model.
        let c0 = self.sim.corpses.len().saturating_sub(CORPSE_DRAW);
        for (p, k, f, _) in &self.sim.corpses[c0..] {
            let pv = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            if do_cull && !sphere_in_frustum(&planes, pv, 10.0) { continue; }
            let slot = model_slot(*f, *k);
            let model = Mat4::from_translation(pv) * Mat4::from_rotation_y((p.x.to_bits() % 628) as f32 * 0.01) * Mat4::from_scale(Vec3::new(scale_of(*k), scale_of(*k) * 0.22, scale_of(*k)));
            buckets[slot].push(SkinInstance { model: model.to_cols_array_2d(), color: [0.10, 0.07, 0.07, 0.55], frame_base: 0, _pad: [0; 3] });
        }
        self.skin_instances.clear();
        let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let start = self.skin_instances.len() as u32;
            self.skin_instances.extend_from_slice(b);
            ranges.push((start, self.skin_instances.len() as u32));
        }
        let cap = MAX_POP * 2 + CORPSE_DRAW + 512;
        if self.skin_instances.len() > cap { self.skin_instances.truncate(cap); }
        self.queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&self.skin_instances));

        // ---- regiment banners: pole + flag boxes (colour = faction; flag
        // height/brightness = strength; routing/shattered = a white rag).
        let mut box_inst: Vec<SkinInstance> = Vec::new();
        for (c, facing, fac, _kind, state, strength) in self.sim.banners() {
            let (x, y, z) = (c.x as f32, c.y as f32, c.z as f32);
            let pole = Mat4::from_translation(Vec3::new(x, y, z)) * Mat4::from_scale(Vec3::new(0.5, 16.0, 0.5));
            box_inst.push(SkinInstance { model: pole.to_cols_array_2d(), color: [0.35, 0.30, 0.24, 1.0], frame_base: 0, _pad: [0; 3] });
            let t = faction_tint(fac);
            let routed = matches!(state, RState::Routing | RState::Shattered);
            let col = if routed { [0.88, 0.88, 0.84] } else { [t[0], t[1], t[2]] };
            let b = 0.45 + 0.55 * strength;
            let flag = Mat4::from_translation(Vec3::new(x, y + 12.0, z)) * Mat4::from_rotation_y(-(facing as f32)) * Mat4::from_scale(Vec3::new(7.0, 4.2, 0.5));
            box_inst.push(SkinInstance { model: flag.to_cols_array_2d(), color: [col[0] * b, col[1] * b, col[2] * b, 1.0], frame_base: 0, _pad: [0; 3] });
        }
        box_inst.truncate(512);
        let box_n = box_inst.len() as u32;
        if box_n > 0 { self.queue.write_buffer(&self.box_inst_buf, 0, bytemuck::cast_slice(&box_inst)); }

        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::cast_slice(&[cam]));

        // ---- arrows → short line segments with a trail (arcing volleys).
        let mut lv: Vec<LVertex> = Vec::new();
        for a in &self.sim.arrows {
            let p0 = a.pos(now - 0.05);
            let p1 = a.pos(now);
            let c = if a.faction == Faction::Red { [1.0, 0.75, 0.45, 0.9] } else { [0.6, 0.75, 1.0, 0.9] };
            lv.push(LVertex { pos: [p0.x as f32, p0.y as f32, p0.z as f32], color: c });
            lv.push(LVertex { pos: [p1.x as f32, p1.y as f32, p1.z as f32], color: c });
        }
        lv.truncate(self.line_cap);
        let line_n = lv.len() as u32;
        if line_n > 0 { self.queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&lv)); }

        // ---- HUD overlay (bars + sliders + pause; text lives in the title).
        let (sw, sh) = (self.config.width as f32, self.config.height as f32);
        let (red, blue) = self.sim.counts();
        let mut ui: Vec<UiVertex> = Vec::new();
        let total = (red + blue).max(1) as f32;
        let (bx, by, bw, bh) = UI_BAR;
        push_quad(&mut ui, bx - 2.0, by - 2.0, bw + 4.0, bh + 4.0, [0.0, 0.0, 0.0, 0.55], sw, sh);
        let rw = bw * red as f32 / total;
        push_quad(&mut ui, bx, by, rw, bh, [0.85, 0.30, 0.22, 0.95], sw, sh);
        push_quad(&mut ui, bx + rw, by, bw - rw, bh, [0.30, 0.42, 0.90, 0.95], sw, sh);
        let (tx, ty, tw, th) = UI_SLIDER;
        push_quad(&mut ui, tx - 2.0, ty - 2.0, tw + 4.0, th + 4.0, [0.0, 0.0, 0.0, 0.55], sw, sh);
        let frac = (self.pop as f32 - 200.0) / (MAX_POP as f32 - 200.0);
        push_quad(&mut ui, tx, ty, tw * frac, th, [0.85, 0.80, 0.55, 0.95], sw, sh);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (hx, hy, hw, hh) = UI_THREADS;
            push_quad(&mut ui, hx - 2.0, hy - 2.0, hw + 4.0, hh + 4.0, [0.0, 0.0, 0.0, 0.55], sw, sh);
            let tf = (self.n_threads as f32 - 1.0) / (self.max_threads as f32 - 1.0).max(1.0);
            push_quad(&mut ui, hx, hy, hw * tf, hh, [0.55, 0.85, 0.60, 0.95], sw, sh);
        }
        let (px, py, pw, ph) = UI_PAUSE;
        push_quad(&mut ui, px, py, pw, ph, if self.paused { [0.85, 0.65, 0.20, 0.9] } else { [0.15, 0.15, 0.18, 0.75] }, sw, sh);
        if self.paused {
            push_quad(&mut ui, px + 34.0, py + 7.0, 28.0, 16.0, [0.05, 0.05, 0.05, 0.9], sw, sh); // play block
        } else {
            push_quad(&mut ui, px + 34.0, py + 7.0, 9.0, 16.0, [0.9, 0.9, 0.9, 0.9], sw, sh);
            push_quad(&mut ui, px + 53.0, py + 7.0, 9.0, 16.0, [0.9, 0.9, 0.9, 0.9], sw, sh);
        }
        let ui_n = ui.len().min(4096) as u32;
        if ui_n > 0 { self.queue.write_buffer(&self.ui_buf, 0, bytemuck::cast_slice(&ui[..ui_n as usize])); }

        // ---- render
        let Some(surface) = &self.surface else { return; };
        let frame = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&self.device, &self.config); return; } };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.46, g: 0.55, b: 0.66, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_bind_group(0, &self.cam_bg, &[]);
            pass.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
            pass.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.terrain_nidx, 0, 0..1);
            // skinned soldiers: one draw per (model, clip) bucket
            pass.set_pipeline(&self.skin_pipeline);
            for (bi, (s, e)) in ranges.iter().enumerate() {
                if s == e { continue; }
                let m = if bi < nm { &self.move_models[bi] } else if bi < nm * 2 { &self.attack_models[bi - nm] } else { &self.horse_move };
                pass.set_bind_group(0, &m.bind, &[]);
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.inst_buf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.nidx, 0, *s..*e);
            }
            // banners
            if box_n > 0 {
                let bm = &self.box_model;
                pass.set_bind_group(0, &bm.bind, &[]);
                pass.set_vertex_buffer(0, bm.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.box_inst_buf.slice(..));
                pass.set_index_buffer(bm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..bm.nidx, 0, 0..box_n);
            }
            // arrows
            if line_n > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.cam_bg, &[]);
                pass.set_vertex_buffer(0, self.line_buf.slice(..));
                pass.draw(0..line_n, 0..1);
            }
            // 2D overlay
            if ui_n > 0 {
                pass.set_pipeline(&self.ui_pipeline);
                pass.set_vertex_buffer(0, self.ui_buf.slice(..));
                pass.draw(0..ui_n, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        frame.present();
    }
}

// ============================================================ event loop

#[cfg(not(target_arch = "wasm32"))]
fn main() { pollster::block_on(run()); }

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash — formations (wgpu)").with_inner_size(winit::dpi::LogicalSize::new(1600, 1000)).build(&event_loop).expect("window"));
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        web_sys::window().and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("wgpu-canvas").or_else(|| d.body().map(|b| b.into())))
            .map(|c| c.append_child(&canvas.into()).ok())
            .expect("attach canvas");
    }
    let mut st = State::new(window.clone()).await;
    #[cfg(not(target_arch = "wasm32"))]
    let max_frames: Option<u64> = std::env::var("FORM_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    #[cfg(not(target_arch = "wasm32"))]
    let mut frames_left = max_frames;
    #[cfg(not(target_arch = "wasm32"))]
    let mut fps_acc: (f64, u64) = (0.0, 0);
    let mut frame: u64 = 0;
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(sz) => st.resize(sz.width, sz.height),
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    if state == ElementState::Pressed {
                        let (mx, my) = (st.last_mouse.0 as f32, st.last_mouse.1 as f32);
                        let hit = |r: (f32, f32, f32, f32)| mx >= r.0 - 8.0 && mx <= r.0 + r.2 + 8.0 && my >= r.1 - 6.0 && my <= r.1 + r.3 + 6.0;
                        if hit(UI_PAUSE) { st.paused = !st.paused; }
                        else if hit(UI_SLIDER) { st.ui_drag = 1; st.apply_slider(1, mx); }
                        else if cfg!(not(target_arch = "wasm32")) && hit(UI_THREADS) { st.ui_drag = 2; st.apply_slider(2, mx); }
                        else { st.dragging = true; }
                    } else { st.dragging = false; st.ui_drag = 0; }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let (dx, dy) = (position.x - st.last_mouse.0, position.y - st.last_mouse.1);
                    st.last_mouse = (position.x, position.y);
                    if st.ui_drag != 0 { let d = st.ui_drag; st.apply_slider(d, position.x as f32); }
                    else if st.dragging {
                        st.yaw += dx as f32 * 0.005;
                        let lim = if st.free_cam { 1.45 } else { 1.35 };
                        st.pitch = (st.pitch + dy as f32 * 0.004).clamp(-lim, lim);
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta { MouseScrollDelta::LineDelta(_, y) => y * 40.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 };
                    st.dist = (st.dist - dy).clamp(60.0, 2600.0);
                }
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. }, .. } => {
                    let pressed = state == ElementState::Pressed;
                    match code {
                        KeyCode::KeyW => st.mv[0] = pressed,
                        KeyCode::KeyS => st.mv[1] = pressed,
                        KeyCode::KeyA => st.mv[2] = pressed,
                        KeyCode::KeyD => st.mv[3] = pressed,
                        KeyCode::KeyQ => st.mv[4] = pressed,
                        KeyCode::KeyE => st.mv[5] = pressed,
                        _ => {}
                    }
                    if pressed { match code {
                        KeyCode::KeyP => st.paused = !st.paused,
                        KeyCode::KeyK => st.frustum_cull = !st.frustum_cull,
                        KeyCode::KeyF => {
                            st.free_cam = !st.free_cam;
                            if st.free_cam {
                                let t = Vec3::new((WORLD * 0.5) as f32, 8.0, (WORLD * 0.5) as f32);
                                st.cam_pos = t + Vec3::new(st.dist * st.pitch.cos() * st.yaw.cos(), st.dist * st.pitch.sin(), st.dist * st.pitch.cos() * st.yaw.sin());
                            }
                        }
                        KeyCode::BracketRight => { let p = st.pop + 400; st.set_population(p); }
                        KeyCode::BracketLeft => { let p = st.pop.saturating_sub(400); st.set_population(p); }
                        _ => {}
                    }}
                },
                WindowEvent::Touch(t) => {
                    let (mx, my) = (t.location.x as f32, t.location.y as f32);
                    let hit = |r: (f32, f32, f32, f32)| mx >= r.0 - 8.0 && mx <= r.0 + r.2 + 8.0 && my >= r.1 - 6.0 && my <= r.1 + r.3 + 6.0;
                    match t.phase {
                        winit::event::TouchPhase::Started => {
                            if hit(UI_PAUSE) { st.paused = !st.paused; }
                            else if hit(UI_SLIDER) { st.ui_drag = 1; st.apply_slider(1, mx); }
                            else { st.dragging = true; }
                            st.last_mouse = (t.location.x, t.location.y);
                        }
                        winit::event::TouchPhase::Moved => {
                            let (dx, dy) = (t.location.x - st.last_mouse.0, t.location.y - st.last_mouse.1);
                            st.last_mouse = (t.location.x, t.location.y);
                            if st.ui_drag != 0 { let d = st.ui_drag; st.apply_slider(d, mx); }
                            else if st.dragging { st.yaw += dx as f32 * 0.005; st.pitch = (st.pitch + dy as f32 * 0.004).clamp(-1.35, 1.35); }
                        }
                        _ => { st.dragging = false; st.ui_drag = 0; }
                    }
                }
                WindowEvent::RedrawRequested => {
                    #[cfg(not(target_arch = "wasm32"))]
                    let t0 = Instant::now();
                    st.update_and_render();
                    frame += 1;
                    if frame % 15 == 0 {
                        let (r, b) = st.sim.counts();
                        let (sr, sb) = st.sim.standing();
                        window.set_title(&format!(
                            "vectorial-hash — formations (wgpu) · red {r} ({sr} regs) | blue {b} ({sb} regs) · kills {}:{} · run {} · {:.0} fps{}",
                            st.sim.kills[0], st.sim.kills[1], st.sim.run, st.fps, if st.paused { " · PAUSED" } else { "" }));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(m) = max_frames {
                        fps_acc.0 += t0.elapsed().as_secs_f64();
                        fps_acc.1 += 1;
                        if let Some(left) = frames_left.as_mut() {
                            *left -= 1;
                            if *left == 0 {
                                let (r, b) = st.sim.counts();
                                println!("formations_wgpu end-to-end: {:.1} fps avg over {} frames (pop {}, red {r}, blue {b})", fps_acc.1 as f64 / fps_acc.0.max(1e-9), m, st.pop);
                                elwt.exit();
                            }
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    }).expect("event loop run");
}

// ============================================================ shaders

const SKIN_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> bones: array<mat4x4<f32>>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32>, @location(1) normal: vec3<f32> };
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
    o.normal = (model * vec4<f32>(sn, 0.0)).xyz;
    o.color = vec4<f32>(mix(vcolor.rgb, tint.rgb, tint.a), 1.0);
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let diff = max(dot(normalize(in.normal), normalize(cam.light.xyz)), 0.0);
    return vec4<f32>(in.color.rgb * (0.40 + 0.60 * diff), 1.0);
}
"#;

const TERRAIN_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32>, @location(1) normal: vec3<f32> };
@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) col: vec4<f32>) -> VOut {
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(p, 1.0);
    o.color = col;
    o.normal = n;
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let diff = max(dot(normalize(in.normal), normalize(cam.light.xyz)), 0.0);
    return vec4<f32>(in.color.rgb * (0.40 + 0.60 * diff), 1.0);
}
"#;

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

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(p, 0.0, 1.0);
    o.color = color;
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
