//! `horde_wgpu` — the They Are Billions-style zombie assault (see
//! `docs/HORDE_DESIGN.md`) rendered with **wgpu** + real GPU skeletal skinning.
//! The whole battle lives in the shared, graphics-free `horde_sim`; this file is
//! render + input only (the siege lesson: sim shared, renderers thin).
//!
//! Controls: drag = orbit · scroll = zoom · `F` free-fly (WASD/QE) · `P`/button
//! pause · `K` frustum cull · `T` tower targeting (nearest ↔ highest-threat) ·
//! `[` `]` population · sliders: population + rayon threads (native).
//!
//! Run: `cargo run -p vectorial-hash-demos --bin horde_wgpu --release --features parallel`

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

use vectorial_hash_demos::horde_sim::{ground_h, DKind, Horde, SKind, ZClass, BASE_R, WORLD};
// Generic glb-baking knobs shared with the siege loader.
use vectorial_hash_demos::siege_sim::{ANIM_FRAMES, MOVE_PREFS};

const MAX_POP: usize = 100_000; // dormant-field slider ceiling (instance-buffer cap)
const CORPSE_DRAW: usize = 32_000; // most-recent corpses drawn (aftermath field)

/// The models this demo needs (all already in `assets/siege/models/`; the web
/// build fetches the same files from `models/` on the Pages site).
const UNIT_FILES: [&str; 10] = [
    "zombie.glb",         // Walker
    "skeleton_a.glb",     // Runner
    "slime.glb",          // Chubby
    "skeleton_sword.glb", // Venom
    "bat.glb",            // Harpy
    "anne.glb",           // Ranger
    "pirate_captain.glb", // Soldier
    "sharky.glb",         // Sniper
    "henry.glb",          // Crew
    "witch.glb",          // Porter
];
fn zmodel(c: ZClass) -> usize { c.index() } // 0..5 in UNIT_FILES order
fn dmodel(k: DKind) -> usize { match k { DKind::Ranger => 5, DKind::Soldier => 6, DKind::Sniper => 7, DKind::Crew => 8, DKind::Porter => 9 } }
fn zscale(c: ZClass) -> f32 { match c { ZClass::Chubby => 9.0, ZClass::Harpy => 5.0, _ => 7.0 } }
fn ztint(c: ZClass, dormant: bool) -> [f32; 4] {
    // Greenish undead cast; dormant reads darker (the sleeping carpet).
    let a = if dormant { 0.42 } else { 0.22 };
    match c {
        ZClass::Walker => [0.35, 0.52, 0.30, a],
        ZClass::Runner => [0.55, 0.58, 0.35, a],
        ZClass::Chubby => [0.30, 0.55, 0.25, a],
        ZClass::Venom => [0.62, 0.42, 0.62, a],
        ZClass::Harpy => [0.60, 0.30, 0.30, a],
    }
}
// Strong colour-coding (the alpha is the mix weight toward the tint): green =
// ranger, red = soldier, blue = sniper, yellow = crew, white = porter (who
// also carries a visible brown bundle while hauling).
fn dtint(k: DKind) -> [f32; 4] {
    match k {
        DKind::Ranger => [0.20, 0.90, 0.40, 0.55],
        DKind::Soldier => [0.95, 0.30, 0.20, 0.55],
        DKind::Sniper => [0.25, 0.45, 1.00, 0.55],
        DKind::Crew => [1.00, 0.85, 0.20, 0.60],
        DKind::Porter => [0.95, 0.95, 0.92, 0.60],
    }
}

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

// ============================================================ terrain

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TVertex { pos: [f32; 3], normal: [f32; 3], color: [f32; 4] }

/// One smooth mesh from the horde heightfield: grass ramp, dust ring around the
/// base (trampled ground), darker rim toward the map edge (where the dead wait).
fn build_terrain(seed: f64) -> (Vec<TVertex>, Vec<u32>) {
    const RES: usize = 200;
    let step = WORLD / RES as f64;
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    let (mut v, mut idx) = (Vec::with_capacity((RES + 1) * (RES + 1)), Vec::new());
    for iz in 0..=RES {
        for ix in 0..=RES {
            let (x, z) = (ix as f64 * step, iz as f64 * step);
            let h = ground_h(x, z, seed);
            let hx = ground_h(x + step, z, seed) - ground_h(x - step, z, seed);
            let hz = ground_h(x, z + step, seed) - ground_h(x, z - step, seed);
            let n = Vec3::new((-hx / (2.0 * step)) as f32, 1.0, (-hz / (2.0 * step)) as f32).normalize();
            let dc = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt();
            // colour: trampled dust inside the walls → grass → sickly rim
            let t_dust = (1.0 - (dc / (BASE_R * 1.05)).min(1.0)) as f32;
            let t_rim = (((dc - 380.0) / 200.0).clamp(0.0, 1.0)) as f32;
            let grass = [0.30 + 0.06 * (h as f32 * 0.15).sin(), 0.44, 0.24];
            let dust = [0.52, 0.44, 0.32];
            let rim = [0.30, 0.32, 0.24];
            let mix3 = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
            let c = mix3(mix3(grass, dust, t_dust), rim, t_rim);
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
struct CameraUniform { vp: [[f32; 4]; 4], light: [f32; 4], eye_time: [f32; 4] } // eye.xyz + sim time (billboards)

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinInstance { model: [[f32; 4]; 4], color: [f32; 4], frame_base: u32, _pad: [u32; 3] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LVertex { pos: [f32; 3], color: [f32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }

const UI_BAR: (f32, f32, f32, f32) = (18.0, 18.0, 280.0, 16.0);      // dormant|active
const UI_SLIDER: (f32, f32, f32, f32) = (18.0, 46.0, 280.0, 12.0);   // population
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

/// The siege 3x5 bitmap font + the letters this HUD adds (N, V, G).
fn glyph(c: char) -> [&'static str; 5] {
    match c {
        '0' => ["111", "101", "101", "101", "111"],
        '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"],
        '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"],
        '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"],
        '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"],
        '9' => ["111", "101", "111", "001", "111"],
        'A' => ["111", "101", "111", "101", "101"],
        'B' => ["110", "101", "110", "101", "110"],
        'C' => ["111", "100", "100", "100", "111"],
        'D' => ["110", "101", "101", "101", "110"],
        'E' => ["111", "100", "110", "100", "111"],
        'F' => ["111", "100", "110", "100", "100"],
        'G' => ["111", "100", "101", "101", "111"],
        'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"],
        'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"],
        'M' => ["101", "111", "111", "101", "101"],
        'N' => ["101", "111", "111", "111", "101"],
        'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"],
        'R' => ["111", "101", "110", "101", "101"],
        'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"],
        'U' => ["101", "101", "101", "101", "111"],
        'V' => ["101", "101", "101", "101", "010"],
        'W' => ["101", "101", "101", "111", "101"],
        'Y' => ["101", "101", "010", "010", "010"],
        ':' => ["000", "010", "000", "010", "000"],
        '-' => ["000", "000", "111", "000", "000"],
        '.' => ["000", "000", "000", "000", "010"],
        _ => ["000", "000", "000", "000", "000"],
    }
}

fn tri_label(t: u64) -> String {
    if t >= 1_000_000 { format!("TRI {}.{}M", t / 1_000_000, (t % 1_000_000) / 100_000) }
    else if t >= 1_000 { format!("TRI {}K", t / 1_000) }
    else { format!("TRI {t}") }
}

#[allow(clippy::too_many_arguments)]
fn push_text(v: &mut Vec<UiVertex>, x: f32, y: f32, px: f32, color: [f32; 4], text: &str, sw: f32, sh: f32) {
    let mut cx = x;
    for c in text.chars() {
        let g = glyph(c.to_ascii_uppercase());
        for (row, bits) in g.iter().enumerate() {
            for (col, ch) in bits.char_indices() {
                if ch == '1' { push_quad(v, cx + col as f32 * px, y + row as f32 * px, px, px, color, sw, sh); }
            }
        }
        cx += 4.0 * px;
    }
}

struct GpuModel {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,
    bind: wgpu::BindGroup,
    num_joints: u32,
    n_frames: u32,
}

fn build_gpu_model(device: &wgpu::Device, cam_buf: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, bytes: &[u8]) -> GpuModel {
    build_gpu_model_prefs(device, cam_buf, layout, bytes, MOVE_PREFS)
}

/// Like [`build_gpu_model`] but with an explicit clip preference — the impostor
/// atlas photographs the Idle clip (dormant carpet) and the Death pose (corpses).
fn build_gpu_model_prefs(device: &wgpu::Device, cam_buf: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, bytes: &[u8], prefs: &[&str]) -> GpuModel {
    let m = vectorial_hash_demos::model::load_unit_model(bytes, ANIM_FRAMES, prefs);
    upload_skinned(device, cam_buf, layout, &m)
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

/// A unit box (XZ centred, base at y=0, height 1) as a static `SkinnedModel` —
/// the walls / gates / towers / houses are instanced boxes through the same
/// skin pipeline (1 identity joint), so no extra pipeline is needed.
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

/// Per-model orientation/size corrections (the slime faces +X — same tweak as
/// siege's `model_tweak`, and it reads big). Baked into the impostor photos at
/// capture time, applied live on the skinned path.
fn ztweak(c: ZClass) -> (f32, f32) {
    match c { ZClass::Chubby => (-std::f32::consts::FRAC_PI_2, 0.80), _ => (0.0, 1.0) }
}

/// LOD switch distance (camera→unit, wu): nearer = full GPU-skinned model,
/// farther = an **impostor billboard** (a photo of the model). At the default
/// orbit the whole field is impostors (units are a few pixels anyway); zoom in
/// and the front line turns into real models.
const LOD_DIST: f32 = 620.0;

// Impostor atlas: per class (texture-array layer), 8 yaw views × 16 rows of
// 64 px cells — rows 0..8 = walk cycle, 8..12 = idle sway (the dormant carpet
// breathes), 12 = the death pose (corpses). Captured at startup by photographing
// each GPU-skinned model with an orbit-pitch ortho camera.
const IMP_VIEWS: u32 = 8;
/// Elevation bands (camera pitch tiers): low / orbit / high — the shader picks
/// the band from the live camera→unit pitch, so top-down views photograph too.
const IMP_ELEVS: [f32; 3] = [0.17, 0.62, 1.10];
const IMP_ROWS: u32 = 16;
const IMP_CELL: u32 = 128;
// Fixed capture framing: the models are height-≈1 by the loader convention
// (siege's model_height contract) — the raw REST vertices are bind-pose
// coordinates and do NOT predict the skinned pose bounds, so mesh-derived
// framing shrinks the photo. One conservative box for every clip keeps all
// rows on the same scale.
const IMP_HALF: f32 = 0.80;
const IMP_CY: f32 = 0.50;
const IMP_WALK_FRAMES: u32 = 8;
const IMP_IDLE_FRAMES: u32 = 4;
const IMP_DEATH_ROW: u32 = 12;

/// One impostor billboard: camera-facing quad, view/frame picked in-shader.
/// `mode`: 0 = walking (animated), 1 = idle (slow sway), 2 = death (still).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardInst { pos: [f32; 3], size: f32, heading: f32, phase: f32, mode: u32, layer: u32, tint: [f32; 4] }

// ============================================================ renderer

struct State {
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    skin_pipeline: wgpu::RenderPipeline,
    models: Vec<GpuModel>, // UNIT_FILES order
    inst_buf: wgpu::Buffer,
    box_model: GpuModel,
    box_inst_buf: wgpu::Buffer,
    castle_model: GpuModel,
    cannon_model: GpuModel,
    cannon_inst_buf: wgpu::Buffer,
    // Impostor billboards (the "photos"): pipeline + atlas + per-class geometry,
    // and the static dormant carpet / append-only corpse buffers.
    bb_pipeline: wgpu::RenderPipeline,
    bb_bind: wgpu::BindGroup,
    /// Per zombie class: (billboard world size, world y-centre offset).
    bb_geom: [(f32, f32); 5],
    dormant_buf: wgpu::Buffer,
    dormant_n: u32,
    dormant_key: (u32, u64), // (run, dormant_epoch) that built the buffer
    proxy_buf: wgpu::Buffer, // far-LOD active zombies, rebuilt per frame
    corpse_buf: wgpu::Buffer,
    corpse_n: u32,
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
    // scene
    sim: Horde,
    seed: u64,
    pop: usize,
    paused: bool,
    frustum_cull: bool,
    /// HORDE_NOLOD=1 → everything fully skinned (the pre-LOD path, for A/B).
    lod: bool,
    fps: f32,
    last: Instant,
    yaw: f32,
    pitch: f32,
    dist: f32,
    dragging: bool,
    last_mouse: (f64, f64),
    free_cam: bool,
    cam_pos: glam::Vec3,
    mv: [bool; 6],
    skin_instances: Vec<SkinInstance>,
}

impl State {
    async fn new(window: Option<Arc<winit::window::Window>>, size_hint: (u32, u32)) -> State {
        let size = match &window { Some(w) => w.inner_size(), None => winit::dpi::PhysicalSize::new(size_hint.0, size_hint.1) };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = window.map(|w| instance.create_surface(w).expect("surface"));
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: surface.as_ref(), force_fallback_adapter: false }).await.expect("adapter");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.expect("device");
        let (format, alpha) = match &surface {
            Some(s) => { let caps = s.get_capabilities(&adapter); (caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]), caps.alpha_modes[0]) }
            None => (wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::CompositeAlphaMode::Opaque),
        };
        // HORDE_NOVSYNC=1 → uncapped presentation (for FPS measurements).
        let present = if std::env::var("HORDE_NOVSYNC").is_ok() { wgpu::PresentMode::AutoNoVsync } else { wgpu::PresentMode::AutoVsync };
        let config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: present, desired_maximum_frame_latency: 2, alpha_mode: alpha, view_formats: vec![] };
        if let Some(s) = &surface { s.configure(&device, &config); }

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<CameraUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cam-l"), entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cam-bg"), layout: &cam_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }] });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pl"), bind_group_layouts: &[&cam_layout], push_constant_ranges: &[] });

        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("skin-inst"), size: ((MAX_POP + CORPSE_DRAW + 4096) * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
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

        // Models: native embeds; web fetches from models/ (same files Pages
        // already hosts for siege).
        #[cfg(target_arch = "wasm32")]
        let table: std::collections::HashMap<&'static str, Vec<u8>> = {
            let mut m = std::collections::HashMap::new();
            for name in UNIT_FILES.iter().copied().chain(["castle.glb", "cannon.glb"]) {
                m.insert(name, fetch_bytes(&format!("models/{name}")).await);
            }
            m
        };
        #[cfg(not(target_arch = "wasm32"))]
        let bytes_of = |name: &str| -> Vec<u8> {
            match name {
                "zombie.glb" => include_bytes!("../../assets/siege/models/zombie.glb").to_vec(),
                "skeleton_a.glb" => include_bytes!("../../assets/siege/models/skeleton_a.glb").to_vec(),
                "slime.glb" => include_bytes!("../../assets/siege/models/slime.glb").to_vec(),
                "skeleton_sword.glb" => include_bytes!("../../assets/siege/models/skeleton_sword.glb").to_vec(),
                "bat.glb" => include_bytes!("../../assets/siege/models/bat.glb").to_vec(),
                "anne.glb" => include_bytes!("../../assets/siege/models/anne.glb").to_vec(),
                "pirate_captain.glb" => include_bytes!("../../assets/siege/models/pirate_captain.glb").to_vec(),
                "sharky.glb" => include_bytes!("../../assets/siege/models/sharky.glb").to_vec(),
                "henry.glb" => include_bytes!("../../assets/siege/models/henry.glb").to_vec(),
                "witch.glb" => include_bytes!("../../assets/siege/models/witch.glb").to_vec(),
                "castle.glb" => include_bytes!("../../assets/siege/models/castle.glb").to_vec(),
                "cannon.glb" => include_bytes!("../../assets/siege/models/cannon.glb").to_vec(),
                _ => unreachable!("unknown horde model {name}"),
            }
        };
        #[cfg(target_arch = "wasm32")]
        let bytes_of = |name: &str| -> Vec<u8> { table[name].clone() };

        let models: Vec<GpuModel> = UNIT_FILES.iter().map(|f| build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of(f))).collect();
        let castle_model = build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of("castle.glb"));
        let cannon_model = build_gpu_model(&device, &cam_buf, &skin_layout, &bytes_of("cannon.glb"));
        let box_model = upload_skinned(&device, &cam_buf, &skin_layout, &unit_box());
        let box_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("box-inst"), size: (1024 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        // Billboard instance buffers (48 B each — half a SkinInstance).
        let dormant_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dormant-inst"), size: (MAX_POP * std::mem::size_of::<BillboardInst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let proxy_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("proxy-inst"), size: ((MAX_POP + 8192) * std::mem::size_of::<BillboardInst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let corpse_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("corpse-inst"), size: (46_000 * std::mem::size_of::<BillboardInst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cannon_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cannon-inst"), size: (64 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        // ---- Impostor atlas: photograph every zombie model (walk / idle /
        // death) from 8 yaw angles into a texture array, then billboards
        // replace the far models. All through the existing skin pipeline.
        let idle_models: Vec<GpuModel> = UNIT_FILES[..5].iter().map(|f| build_gpu_model_prefs(&device, &cam_buf, &skin_layout, &bytes_of(f), &["idle", "fly", "walk"])).collect();
        let death_models: Vec<GpuModel> = UNIT_FILES[..5].iter().map(|f| build_gpu_model_prefs(&device, &cam_buf, &skin_layout, &bytes_of(f), &["death", "hit", "idle"])).collect();
        let atlas_w = IMP_VIEWS * IMP_ELEVS.len() as u32 * IMP_CELL; // 8 yaws × 3 elevations
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("impostor-atlas"),
            size: wgpu::Extent3d { width: atlas_w, height: IMP_ROWS * IMP_CELL, depth_or_array_layers: 5 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let atlas_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("impostor-depth"),
            size: wgpu::Extent3d { width: atlas_w, height: IMP_ROWS * IMP_CELL, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());
        let mut bb_geom = [(0.0f32, 0.0f32); 5];
        for (ci, walk) in models[..5].iter().enumerate() {
            let class = [ZClass::Walker, ZClass::Runner, ZClass::Chubby, ZClass::Venom, ZClass::Harpy][ci];
            let (tw_yaw, tw_scale) = ztweak(class);
            let world_scale = zscale(class) * tw_scale;
            bb_geom[ci] = (IMP_HALF * 2.0 * world_scale, IMP_CY * world_scale);
            let layer_view = atlas.create_view(&wgpu::TextureViewDescriptor { base_array_layer: ci as u32, array_layer_count: Some(1), dimension: Some(wgpu::TextureViewDimension::D2), ..Default::default() });
            // (row, model, that row's frame) — walk cycle, idle sway, death pose.
            let mut shots: Vec<(u32, &GpuModel, u32)> = Vec::new();
            for r in 0..IMP_WALK_FRAMES { shots.push((r, walk, r * walk.n_frames.max(1) / IMP_WALK_FRAMES)); }
            let idm = &idle_models[ci];
            for k in 0..IMP_IDLE_FRAMES { shots.push((IMP_WALK_FRAMES + k, idm, k * idm.n_frames.max(1) / IMP_IDLE_FRAMES)); }
            let dm = &death_models[ci];
            shots.push((IMP_DEATH_ROW, dm, dm.n_frames.max(1) - 1));
            let mut first = true;
            for (row, m, frame) in shots {
                for (e, elev) in IMP_ELEVS.iter().enumerate() {
                    for v in 0..IMP_VIEWS {
                        // Ortho camera: azimuth v/8·τ, this band's pitch, fixed framing.
                        let az = v as f32 / IMP_VIEWS as f32 * std::f32::consts::TAU;
                        let (h, cy) = (IMP_HALF, IMP_CY);
                        let dir = Vec3::new(az.cos() * elev.cos(), elev.sin(), az.sin() * elev.cos());
                        let view = Mat4::look_at_rh(Vec3::new(0.0, cy, 0.0) + dir * (h * 6.0), Vec3::new(0.0, cy, 0.0), Vec3::Y);
                        let proj = Mat4::orthographic_rh(-h, h, -h, h, 0.1, h * 14.0);
                        let cam = CameraUniform { vp: (proj * view).to_cols_array_2d(), light: [-0.45, 0.84, -0.30, 0.0], eye_time: [0.0; 4] };
                        queue.write_buffer(&cam_buf, 0, bytemuck::cast_slice(&[cam]));
                        let inst = SkinInstance { model: Mat4::from_rotation_y(tw_yaw).to_cols_array_2d(), color: [0.0, 0.0, 0.0, 0.0], frame_base: frame * m.num_joints, _pad: [0; 3] };
                        queue.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&[inst]));
                        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        {
                            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("impostor-shot"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &layer_view, resolve_target: None, ops: wgpu::Operations { load: if first { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store } })],
                                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &atlas_depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                                timestamp_writes: None, occlusion_query_set: None,
                            });
                            pass.set_viewport(((e as u32 * IMP_VIEWS + v) * IMP_CELL) as f32, (row * IMP_CELL) as f32, IMP_CELL as f32, IMP_CELL as f32, 0.0, 1.0);
                            pass.set_pipeline(&skin_pipeline);
                            pass.set_bind_group(0, &m.bind, &[]);
                            pass.set_vertex_buffer(0, m.vbuf.slice(..));
                            pass.set_vertex_buffer(1, inst_buf.slice(..));
                            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..m.nidx, 0, 0..1);
                        }
                        queue.submit(Some(enc.finish()));
                        first = false;
                    }
                }
            }
        }
        // Billboard pipeline: camera + the atlas array + a sampler; the quad is
        // generated from the vertex index (no vertex buffer), faced in-shader.
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("impostor-sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge, ..Default::default() });
        let bb_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bb-l"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bb_bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bb-bg"), layout: &bb_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
        ] });
        let bb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("bb-shader"), source: wgpu::ShaderSource::Wgsl(BB_SHADER.into()) });
        let bb_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bb-pl"), bind_group_layouts: &[&bb_layout], push_constant_ranges: &[] });
        let bb_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bb-pipe"),
            layout: Some(&bb_pipe_layout),
            vertex: wgpu::VertexState {
                module: &bb_shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<BillboardInst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32, 3 => Float32, 4 => Uint32, 5 => Uint32, 6 => Float32x4] }],
            },
            fragment: Some(wgpu::FragmentState { module: &bb_shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });

        let seed = std::env::var("HORDE_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0x40BDE);
        let pop = std::env::var("HORDE_POP").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000).clamp(2_000, MAX_POP);
        let sim = Horde::new(seed, pop);

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
        let line_cap = 8192usize;
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
        let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui-v"), size: (16384 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let depth = make_depth(&device, &config);
        #[cfg(not(target_arch = "wasm32"))]
        let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        #[cfg(not(target_arch = "wasm32"))]
        let pool = rayon::ThreadPoolBuilder::new().num_threads(max_threads).build().unwrap();

        State {
            surface, device, queue, config,
            skin_pipeline, models, inst_buf,
            box_model, box_inst_buf, castle_model, cannon_model, cannon_inst_buf,
            bb_pipeline, bb_bind, bb_geom, dormant_buf, dormant_n: 0, dormant_key: (0, 0),
            proxy_buf, corpse_buf, corpse_n: 0,
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
            paused: false, frustum_cull: true, lod: std::env::var("HORDE_NOLOD").is_err(), fps: 0.0, last: Instant::now(),
            yaw: 0.9, pitch: 0.7, dist: 820.0, dragging: false, last_mouse: (0.0, 0.0),
            free_cam: false, cam_pos: glam::Vec3::ZERO, mv: [false; 6],
            skin_instances: Vec::with_capacity(64 * 1024),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w; self.config.height = h;
            if let Some(s) = &self.surface { s.configure(&self.device, &self.config); }
            self.depth = make_depth(&self.device, &self.config);
        }
    }

    /// Rebuild the whole run at a new dormant-field size (slider / [ ] keys).
    fn set_population(&mut self, pop: usize) {
        let pop = pop.clamp(2_000, MAX_POP);
        if pop == self.pop { return; }
        self.pop = pop;
        self.sim = Horde::new(self.seed, pop);
        // The fresh sim restarts at (run 1, epoch 1) — the same key the old one
        // started with, so the static dormant carpet would never re-upload.
        self.dormant_key = (u32::MAX, u64::MAX);
        self.corpse_n = 0;
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
            1 => { let (tx, _, tw, _) = UI_SLIDER; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_population(2_000 + (frac * (MAX_POP as f32 - 2_000.0)) as usize); }
            #[cfg(not(target_arch = "wasm32"))]
            2 => { let (tx, _, tw, _) = UI_THREADS; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_threads(1 + (frac * (self.max_threads as f32 - 1.0)).round() as usize); }
            _ => {}
        }
    }

    fn forward(&self) -> Vec3 {
        -Vec3::new(self.pitch.cos() * self.yaw.cos(), self.pitch.sin(), self.pitch.cos() * self.yaw.sin())
    }

    /// The camera eye in world space (orbit or free-fly) — LOD distances key off it.
    fn eye(&self) -> Vec3 {
        if self.free_cam { return self.cam_pos; }
        let target = Vec3::new((WORLD * 0.5) as f32, 12.0, (WORLD * 0.5) as f32);
        target + Vec3::new(self.dist * self.pitch.cos() * self.yaw.cos(), self.dist * self.pitch.sin(), self.dist * self.pitch.cos() * self.yaw.sin())
    }

    fn camera(&self) -> CameraUniform {
        let (eye, target) = if self.free_cam {
            (self.cam_pos, self.cam_pos + self.forward())
        } else {
            let target = Vec3::new((WORLD * 0.5) as f32, 12.0, (WORLD * 0.5) as f32);
            (target + Vec3::new(self.dist * self.pitch.cos() * self.yaw.cos(), self.dist * self.pitch.sin(), self.dist * self.pitch.cos() * self.yaw.sin()), target)
        };
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 1.0, 5200.0);
        CameraUniform { vp: (proj * view).to_cols_array_2d(), light: [-0.45, 0.84, -0.30, 0.0], eye_time: [eye.x, eye.y, eye.z, self.sim.now as f32] }
    }

    fn update_and_render(&mut self) {
        let dt = { let d = self.last.elapsed().as_secs_f64().min(0.05); self.last = Instant::now(); d };
        if self.free_cam {
            let fwd = self.forward();
            let right = fwd.cross(Vec3::Y).normalize_or_zero();
            let sp = 500.0 * dt as f32;
            let mut d = Vec3::ZERO;
            if self.mv[0] { d += fwd; } if self.mv[1] { d -= fwd; }
            if self.mv[3] { d += right; } if self.mv[2] { d -= right; }
            if self.mv[5] { d += Vec3::Y; } if self.mv[4] { d -= Vec3::Y; }
            if d.length_squared() > 0.0 { self.cam_pos += d.normalize() * sp; }
        }
        if !self.paused {
            #[cfg(not(target_arch = "wasm32"))]
            { let sim = &mut self.sim; self.pool.install(|| sim.step(dt)); }
            #[cfg(target_arch = "wasm32")]
            self.sim.step(dt);
        }
        let inst_fps = (1.0 / dt.max(1e-3)) as f32;
        self.fps = if self.fps == 0.0 { inst_fps } else { self.fps * 0.92 + inst_fps * 0.08 };

        let cam = self.camera();
        let planes = frustum_planes(glam::Mat4::from_cols_array_2d(&cam.vp));
        let do_cull = self.frustum_cull;
        let now = self.sim.now;

        let eye = self.eye();

        // The sleeping carpet: a STATIC instance buffer of slump proxies,
        // rebuilt only when the dormant set changes (the sim's dormant_epoch) —
        // between wakes/sleeps/deaths the 100k-sleeper upload cost is ZERO.
        let key = (self.sim.run, self.sim.dormant_epoch);
        if !self.lod { self.dormant_n = 0; self.dormant_key = (0, 0); }
        if self.lod && key != self.dormant_key {
            let mut di: Vec<BillboardInst> = Vec::with_capacity(self.sim.units.len());
            for (i, z) in self.sim.units.iter().enumerate() {
                if !z.alive() || !z.dormant() { continue; }
                let (size, cy) = self.bb_geom[zmodel(z.class)];
                di.push(BillboardInst {
                    pos: [z.p.x as f32, z.p.y as f32 + cy, z.p.z as f32],
                    size, heading: (i as f32) * 2.399963, phase: (i % 97) as f32 * 0.0103,
                    mode: 1, layer: zmodel(z.class) as u32,
                    tint: [0.0, 0.0, 0.0, 0.30], // sleepers read darker
                });
            }
            di.truncate(MAX_POP);
            self.queue.write_buffer(&self.dormant_buf, 0, bytemuck::cast_slice(&di));
            self.dormant_n = di.len() as u32;
            self.dormant_key = key;
        }

        // Corpses: append-only — upload only the new bodies since last frame
        // (full rebuild after a drain/reset shrinks the vec). Death-pose photo.
        {
            let corpses = &self.sim.corpses;
            let n = corpses.len().min(46_000);
            let from = if (self.corpse_n as usize) <= n { self.corpse_n as usize } else { 0 };
            if n > from || from == 0 && n < self.corpse_n as usize {
                let mut ci: Vec<BillboardInst> = Vec::with_capacity(n - from.min(n));
                for (p, class, _) in &corpses[from..n] {
                    let (size, cy) = self.bb_geom[zmodel(*class)];
                    ci.push(BillboardInst {
                        pos: [p.x as f32, p.y as f32 + cy * 0.4, p.z as f32],
                        size: size * 0.9, heading: (p.x.to_bits() % 628) as f32 * 0.01, phase: 0.0,
                        mode: 2, layer: zmodel(*class) as u32,
                        tint: [0.05, 0.03, 0.03, 0.45], // cold bodies
                    });
                }
                if !ci.is_empty() { self.queue.write_buffer(&self.corpse_buf, (from * std::mem::size_of::<BillboardInst>()) as u64, bytemuck::cast_slice(&ci)); }
                self.corpse_n = n as u32;
            }
        }

        // ACTIVE zombies: full skinned model when near (LOD_DIST), the standing
        // proxy when far; defenders always skinned (there are ~50 of them).
        let mut buckets: Vec<Vec<SkinInstance>> = (0..self.models.len()).map(|_| Vec::new()).collect();
        let mut proxies: Vec<BillboardInst> = Vec::new();
        for (i, z) in self.sim.units.iter().enumerate() {
            if !z.alive() { continue; }
            let dormant = z.dormant();
            if self.lod && dormant { continue; } // in the static carpet buffer
            let (x, y, zz) = (z.p.x as f32, z.p.y as f32, z.p.z as f32);
            let scale = zscale(z.class);
            if do_cull && !sphere_in_frustum(&planes, glam::Vec3::new(x, y, zz), scale * 1.6) { continue; }
            let yaw = (z.vel.1 as f32).atan2(z.vel.0 as f32);
            let d2 = (Vec3::new(x, y, zz) - eye).length_squared();
            if self.lod && d2 > LOD_DIST * LOD_DIST {
                // Far: a walking impostor (photo billboard, animated in-shader).
                let (size, cy) = self.bb_geom[zmodel(z.class)];
                proxies.push(BillboardInst {
                    pos: [x, y + cy, zz], size, heading: -yaw,
                    phase: (i % 89) as f32 * 0.0112,
                    mode: 0, layer: zmodel(z.class) as u32, tint: [0.0, 0.0, 0.0, 0.0],
                });
                continue;
            }
            let (tw_yaw, tw_scale) = ztweak(z.class);
            let mi = zmodel(z.class);
            let m = &self.models[mi];
            let nf = m.n_frames.max(1);
            let phase = (i as u32 % 8) as f32 / 8.0; // per-instance anim jitter
            let frame = if dormant { ((phase * nf as f32) as u32) % nf } // NOLOD A/B path
                else { (((now as f32 * 1.6 + phase) * nf as f32) as u32) % nf };
            let model = Mat4::from_translation(Vec3::new(x, y, zz)) * Mat4::from_rotation_y(-yaw + std::f32::consts::FRAC_PI_2 + tw_yaw) * Mat4::from_scale(Vec3::splat(scale * tw_scale));
            buckets[mi].push(SkinInstance { model: model.to_cols_array_2d(), color: ztint(z.class, dormant), frame_base: frame * m.num_joints, _pad: [0; 3] });
        }
        proxies.truncate(MAX_POP + 8192);
        if !proxies.is_empty() { self.queue.write_buffer(&self.proxy_buf, 0, bytemuck::cast_slice(&proxies)); }
        let proxy_n = proxies.len() as u32;
        for d in &self.sim.defenders {
            if !d.alive() { continue; }
            let (x, y, zz) = (d.p.x as f32, d.p.y as f32, d.p.z as f32);
            if do_cull && !sphere_in_frustum(&planes, glam::Vec3::new(x, y, zz), 11.0) { continue; }
            let mi = dmodel(d.kind);
            let m = &self.models[mi];
            let nf = m.n_frames.max(1);
            let frame = (((now as f32 * 1.8) * nf as f32) as u32) % nf;
            let model = Mat4::from_translation(Vec3::new(x, y, zz)) * Mat4::from_scale(Vec3::splat(7.0));
            buckets[mi].push(SkinInstance { model: model.to_cols_array_2d(), color: dtint(d.kind), frame_base: frame * m.num_joints, _pad: [0; 3] });
        }
        self.skin_instances.clear();
        let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let start = self.skin_instances.len() as u32;
            self.skin_instances.extend_from_slice(b);
            ranges.push((start, self.skin_instances.len() as u32));
        }
        let cap = MAX_POP + CORPSE_DRAW + 4096;
        if self.skin_instances.len() > cap { self.skin_instances.truncate(cap); }
        self.queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&self.skin_instances));

        // Structures → box instances (walls tinted by HP; rubble = flat slab).
        let mut box_inst: Vec<SkinInstance> = Vec::new();
        let mut cannon_inst: Vec<SkinInstance> = Vec::new();
        let (ccx, ccz) = (WORLD as f32 * 0.5, WORLD as f32 * 0.5);
        for s in &self.sim.structures {
            let (x, y, zz) = (s.p.x as f32, s.p.y as f32, s.p.z as f32);
            let frac = (s.hp / s.kind.max_hp()).clamp(0.0, 1.0) as f32;
            let ang = (zz - ccz).atan2(x - ccx);
            let (sx, sy, sz, mut col) = match s.kind {
                SKind::Wall => (8.4, 13.0, 4.5, [0.58, 0.58, 0.62]),
                SKind::Gate => (10.0, 10.0, 6.0, [0.48, 0.36, 0.24]),
                SKind::Tower => (11.0, 22.0, 11.0, [0.52, 0.52, 0.58]),
                SKind::House => (12.0, 9.0, 12.0, [0.72, 0.60, 0.44]),
                SKind::Storehouse => (15.0, 10.0, 15.0, [0.55, 0.42, 0.30]),
                SKind::CommandCenter => (0.0, 0.0, 0.0, [0.0, 0.0, 0.0]), // castle model below
            };
            if s.kind == SKind::CommandCenter { continue; }
            let destroyed = s.hp <= 0.0;
            // Damage must READ from orbit: lerp hard toward glowing red-brown
            // as HP drops (full = clean stone, half = clearly wounded, near
            // death = alarm red); rubble is a charred low slab.
            let dmg = 1.0 - frac;
            let wound = [0.85, 0.16, 0.10];
            col = [col[0] + (wound[0] - col[0]) * dmg, col[1] + (wound[1] - col[1]) * dmg, col[2] + (wound[2] - col[2]) * dmg];
            if destroyed { col = [0.16, 0.13, 0.11]; }
            let h = if destroyed { sy * 0.14 } else { sy * (0.35 + 0.65 * frac) };
            let m = Mat4::from_translation(Vec3::new(x, y, zz)) * Mat4::from_rotation_y(-ang) * Mat4::from_scale(Vec3::new(sx, h, sz));
            box_inst.push(SkinInstance { model: m.to_cols_array_2d(), color: [col[0], col[1], col[2], 0.0], frame_base: 0, _pad: [0; 3] });
            if s.kind == SKind::Tower && !destroyed {
                let cm = Mat4::from_translation(Vec3::new(x, y + h, zz)) * Mat4::from_rotation_y(-ang + std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(7.0));
                cannon_inst.push(SkinInstance { model: cm.to_cols_array_2d(), color: [0.2, 0.2, 0.22, 0.25], frame_base: 0, _pad: [0; 3] });
            }
        }
        // Loaded porters carry a visible brown bundle — the supply line reads
        // at a glance (and you can tell who the porters are).
        for d in &self.sim.defenders {
            if !d.alive() { continue; }
            if let vectorial_hash_demos::horde_sim::DState::Hauling { loaded: true, .. } = d.state {
                let m = Mat4::from_translation(Vec3::new(d.p.x as f32, d.p.y as f32 + 7.6, d.p.z as f32)) * Mat4::from_scale(Vec3::new(3.2, 2.4, 3.2));
                box_inst.push(SkinInstance { model: m.to_cols_array_2d(), color: [0.52, 0.36, 0.18, 0.0], frame_base: 0, _pad: [0; 3] });
            }
        }
        box_inst.truncate(1024);
        cannon_inst.truncate(64);
        self.queue.write_buffer(&self.box_inst_buf, 0, bytemuck::cast_slice(&box_inst));
        if !cannon_inst.is_empty() { self.queue.write_buffer(&self.cannon_inst_buf, 0, bytemuck::cast_slice(&cannon_inst)); }
        let (box_n, cannon_n) = (box_inst.len() as u32, cannon_inst.len() as u32);

        // The Command Center: the castle model, alive or fallen (sunken + dark).
        let cc = self.sim.structures.iter().find(|s| s.kind == SKind::CommandCenter).unwrap();
        let cc_dead = cc.hp <= 0.0;
        let cc_m = Mat4::from_translation(Vec3::new(cc.p.x as f32, cc.p.y as f32 - if cc_dead { 10.0 } else { 0.0 }, cc.p.z as f32)) * Mat4::from_scale(Vec3::splat(40.0));
        let cc_inst = [SkinInstance { model: cc_m.to_cols_array_2d(), color: if cc_dead { [0.15, 0.12, 0.12, 0.6] } else { [0.75, 0.72, 0.66, 0.15] }, frame_base: 0, _pad: [0; 3] }];
        // castle instance rides at the very end of the big instance buffer
        let cc_off = (self.skin_instances.len().min(cap)) as u64 * std::mem::size_of::<SkinInstance>() as u64;
        self.queue.write_buffer(&self.inst_buf, cc_off, bytemuck::cast_slice(&cc_inst));
        let cc_range = (self.skin_instances.len().min(cap)) as u32;

        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::cast_slice(&[cam]));

        // Tracers (tower + defender shots) → fading lines.
        let mut lv: Vec<LVertex> = Vec::new();
        for (a, b, t) in &self.sim.tracers {
            let age = ((now - t) / 0.12).clamp(0.0, 1.0) as f32;
            let c = [1.0, 0.9, 0.5, 1.0 - age];
            lv.push(LVertex { pos: [a.x as f32, a.y as f32, a.z as f32], color: c });
            lv.push(LVertex { pos: [b.x as f32, b.y as f32 + 2.0, b.z as f32], color: c });
        }
        if lv.len() > self.line_cap {
            self.line_cap = (lv.len() * 2).next_power_of_two();
            self.line_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("line-v"), size: (self.line_cap * std::mem::size_of::<LVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        }
        if !lv.is_empty() { self.queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&lv)); }
        let line_n = lv.len() as u32;

        // ---- 2D overlay: dormant|active bar, sliders, pause, numeric HUD.
        let (sw, sh) = (self.config.width as f32, self.config.height as f32);
        let (dormant, active) = self.sim.counts();
        let mut ui: Vec<UiVertex> = Vec::new();
        let (bx, by, bw, bh) = UI_BAR;
        let total = (dormant + active).max(1) as f32;
        let dw = bw * dormant as f32 / total;
        push_quad(&mut ui, bx - 2.0, by - 2.0, bw + 4.0, bh + 4.0, [0.0, 0.0, 0.0, 0.45], sw, sh);
        push_quad(&mut ui, bx, by, dw, bh, [0.25, 0.45, 0.28, 0.95], sw, sh); // sleeping
        push_quad(&mut ui, bx + dw, by, bw - dw, bh, [0.85, 0.28, 0.20, 0.95], sw, sh); // awake
        let (tx, ty, tw, th) = UI_SLIDER;
        push_quad(&mut ui, tx, ty, tw, th, [0.22, 0.22, 0.28, 0.85], sw, sh);
        let frac = ((self.pop as f32 - 2_000.0) / (MAX_POP as f32 - 2_000.0)).clamp(0.0, 1.0);
        push_quad(&mut ui, tx + frac * tw - 5.0, ty - 3.0, 10.0, th + 6.0, [0.95, 0.95, 0.98, 1.0], sw, sh);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (hx, hy, hw, hh) = UI_THREADS;
            push_quad(&mut ui, hx, hy, hw, hh, [0.22, 0.22, 0.28, 0.85], sw, sh);
            let tfrac = if self.max_threads > 1 { (self.n_threads as f32 - 1.0) / (self.max_threads as f32 - 1.0) } else { 1.0 };
            push_quad(&mut ui, hx + tfrac * hw - 5.0, hy - 3.0, 10.0, hh + 6.0, [0.40, 0.95, 0.55, 1.0], sw, sh);
        }
        {
            let (px, py, pw, ph) = UI_PAUSE;
            let running = !self.paused;
            push_quad(&mut ui, px, py, pw, ph, if running { [0.18, 0.34, 0.22, 0.9] } else { [0.42, 0.22, 0.16, 0.95] }, sw, sh);
            push_text(&mut ui, px + 8.0, py + 9.0, 3.0, [0.92, 0.94, 0.98, 1.0], if running { "PAUSE" } else { "PLAY" }, sw, sh);
        }
        let mut tris: u64 = (self.terrain_nidx / 3) as u64;
        for (mi, m) in self.models.iter().enumerate() {
            let (s, e) = ranges[mi];
            tris += (m.nidx as u64 / 3) * (e - s) as u64;
        }
        tris += (self.box_model.nidx as u64 / 3) * box_n as u64 + (self.cannon_model.nidx as u64 / 3) * cannon_n as u64 + self.castle_model.nidx as u64 / 3;
        tris += 2 * (self.dormant_n + self.corpse_n + proxy_n) as u64; // impostor quads
        let white = [0.92, 0.94, 0.98, 1.0];
        let hx = sw - 170.0;
        let (wave_k, announced, wdir, eta) = self.sim.wave_info();
        push_text(&mut ui, hx, 12.0, 3.0, white, &format!("FPS {:.0}", self.fps), sw, sh);
        push_text(&mut ui, hx, 30.0, 3.0, [0.55, 0.95, 0.60, 1.0], &format!("SLP {dormant}"), sw, sh);
        push_text(&mut ui, hx, 48.0, 3.0, [1.0, 0.55, 0.45, 1.0], &format!("ACT {active}"), sw, sh);
        push_text(&mut ui, hx, 66.0, 3.0, white, &format!("KIL {}", self.sim.kills), sw, sh);
        push_text(&mut ui, hx, 84.0, 3.0, white, &format!("RUN {}", self.sim.run), sw, sh);
        push_text(&mut ui, hx, 102.0, 3.0, white, &tri_label(tris), sw, sh);
        #[cfg(not(target_arch = "wasm32"))]
        push_text(&mut ui, hx, 120.0, 3.0, white, &format!("THR {}", self.n_threads), sw, sh);
        let mode = if self.sim.tower_threat_mode { "T: THREAT" } else { "T: NEAR" };
        push_text(&mut ui, hx, 138.0, 3.0, [0.9, 0.85, 0.5, 1.0], mode, sw, sh);
        // Ring integrity: average HP of the wall line (walls+gates+towers).
        let (mut got_hp, mut max_hp) = (0.0f64, 0.0f64);
        for s in &self.sim.structures {
            if matches!(s.kind, SKind::Wall | SKind::Gate | SKind::Tower) { got_hp += s.hp.max(0.0); max_hp += s.kind.max_hp(); }
        }
        let wal = (got_hp / max_hp.max(1.0) * 100.0) as u32;
        let walc = if wal > 70 { [0.55, 1.0, 0.60, 1.0] } else if wal > 35 { [1.0, 0.85, 0.4, 1.0] } else { [1.0, 0.45, 0.35, 1.0] };
        push_text(&mut ui, hx, 156.0, 3.0, walc, &format!("WAL {wal}"), sw, sh);
        // Wave banner — centre-top, with compass direction + countdown.
        if announced {
            let (dx, dz) = (wdir.cos(), wdir.sin());
            let comp = if dz.abs() > dx.abs() { if dz > 0.0 { "S" } else { "N" } } else if dx > 0.0 { "E" } else { "W" };
            let msg = format!("WAVE {wave_k} FROM {comp} T-{:.0}", eta);
            let w = msg.len() as f32 * 4.0 * 4.0;
            push_quad(&mut ui, sw * 0.5 - w * 0.5 - 10.0, 14.0, w + 20.0, 30.0, [0.45, 0.10, 0.08, 0.85], sw, sh);
            push_text(&mut ui, sw * 0.5 - w * 0.5, 22.0, 4.0, [1.0, 0.85, 0.6, 1.0], &msg, sw, sh);
        }
        if let Some((_, victory)) = self.sim.game_over {
            let msg = if victory { "THE COLONY SURVIVED" } else { "THE COLONY HAS FALLEN" };
            let w = msg.len() as f32 * 4.0 * 5.0;
            push_quad(&mut ui, sw * 0.5 - w * 0.5 - 12.0, sh * 0.42, w + 24.0, 44.0, [0.0, 0.0, 0.0, 0.72], sw, sh);
            push_text(&mut ui, sw * 0.5 - w * 0.5, sh * 0.42 + 12.0, 5.0, if victory { [0.6, 1.0, 0.65, 1.0] } else { [1.0, 0.45, 0.35, 1.0] }, msg, sw, sh);
        }
        ui.truncate(16384);
        self.queue.write_buffer(&self.ui_buf, 0, bytemuck::cast_slice(&ui));
        let ui_n = ui.len() as u32;

        // ---- render
        let frame = match &self.surface { Some(s) => match s.get_current_texture() { Ok(f) => f, Err(_) => return }, None => return };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.42, g: 0.50, b: 0.62, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_bind_group(0, &self.cam_bg, &[]);
            pass.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
            pass.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.terrain_nidx, 0, 0..1);
            // Impostor billboards: the sleeping carpet (static buffer, zero
            // per-frame CPU), the aftermath corpses (append-only), and the far
            // actives — all photos from the atlas, faced + animated in-shader.
            pass.set_pipeline(&self.bb_pipeline);
            pass.set_bind_group(0, &self.bb_bind, &[]);
            if self.dormant_n > 0 {
                pass.set_vertex_buffer(0, self.dormant_buf.slice(..));
                pass.draw(0..6, 0..self.dormant_n);
            }
            if self.corpse_n > 0 {
                pass.set_vertex_buffer(0, self.corpse_buf.slice(..));
                pass.draw(0..6, 0..self.corpse_n);
            }
            if proxy_n > 0 {
                pass.set_vertex_buffer(0, self.proxy_buf.slice(..));
                pass.draw(0..6, 0..proxy_n);
            }
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
            // the Command Center castle (1 instance parked at the buffer tail)
            let cm = &self.castle_model;
            pass.set_bind_group(0, &cm.bind, &[]);
            pass.set_vertex_buffer(0, cm.vbuf.slice(..));
            pass.set_index_buffer(cm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cm.nidx, 0, cc_range..cc_range + 1);
            // structures (instanced boxes)
            if box_n > 0 {
                let bm = &self.box_model;
                pass.set_bind_group(0, &bm.bind, &[]);
                pass.set_vertex_buffer(0, bm.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.box_inst_buf.slice(..));
                pass.set_index_buffer(bm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..bm.nidx, 0, 0..box_n);
            }
            // tower cannons
            if cannon_n > 0 {
                let km = &self.cannon_model;
                pass.set_bind_group(0, &km.bind, &[]);
                pass.set_vertex_buffer(0, km.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.cannon_inst_buf.slice(..));
                pass.set_index_buffer(km.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..km.nidx, 0, 0..cannon_n);
            }
            if line_n > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.cam_bg, &[]);
                pass.set_vertex_buffer(0, self.line_buf.slice(..));
                pass.draw(0..line_n, 0..1);
            }
            if ui_n > 0 {
                pass.set_pipeline(&self.ui_pipeline);
                pass.set_vertex_buffer(0, self.ui_buf.slice(..));
                pass.draw(0..ui_n, 0..1);
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

// Impostor billboards: a camera-facing quad per instance (no vertex buffer —
// corners from the vertex index); the shader picks the atlas VIEW column from
// the camera→instance azimuth vs the instance heading, and the FRAME row from
// the mode (0 walk cycle · 1 idle sway · 2 death pose) + time. Alpha-cutout.
const BB_SHADER: &str = r#"
struct Camera { vp: mat4x4<f32>, light: vec4<f32>, eye_time: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var atlas: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) tint: vec4<f32>, @location(2) layer: u32 };
const TAU: f32 = 6.2831853;
@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @location(0) pos: vec3<f32>, @location(1) size: f32, @location(2) heading: f32,
      @location(3) phase: f32, @location(4) mode: u32, @location(5) layer: u32,
      @location(6) tint: vec4<f32>) -> VOut {
    // corner of a unit quad (two triangles, centre-anchored)
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
        vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
    let c = corners[vi];
    // face the camera fully (view-plane billboard — stays readable from above)
    let to_cam = cam.eye_time.xyz - pos;
    let dist = max(length(to_cam), 0.001);
    let dirc = to_cam / dist;
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), dirc));
    let up = normalize(cross(dirc, right));
    let world = pos + right * (c.x * size) + up * (c.y * size);
    // atlas view column: yaw sector + elevation band of this sight line
    let a_cam = atan2(to_cam.z, to_cam.x);
    var rel = a_cam - heading;
    rel = rel - TAU * floor(rel / TAU);
    let view = u32(floor(rel / TAU * 8.0 + 0.5)) % 8u;
    let pitch = atan2(to_cam.y, max(length(vec2(to_cam.x, to_cam.z)), 0.001));
    var band = 0u; // captured at 0.17 / 0.62 / 1.10 rad
    if (pitch > 0.86) { band = 2u; } else if (pitch > 0.40) { band = 1u; }
    // frame row by mode
    let t = cam.eye_time.w;
    var row = 12u; // death pose
    if (mode == 0u) { row = u32(floor((t * 1.6 + phase * 8.0) * 8.0)) % 8u; }
    else if (mode == 1u) { row = 8u + u32(floor((t * 0.7 + phase * 4.0) * 4.0)) % 4u; }
    let u0 = (f32(band * 8u + view) + (c.x + 0.5)) / 24.0;
    let v0 = (f32(row) + (0.5 - c.y)) / 16.0;
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(world, 1.0);
    o.uv = vec2(u0, v0);
    o.tint = tint;
    o.layer = layer;
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = textureSample(atlas, samp, in.uv, in.layer);
    if (px.a < 0.5) { discard; }
    let rgb = mix(px.rgb, in.tint.rgb, in.tint.a);
    return vec4<f32>(rgb, 1.0);
}
"#;

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
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash — horde (wgpu)").with_inner_size(winit::dpi::LogicalSize::new(1600, 1000)).build(&event_loop).expect("window"));
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        let _ = canvas.set_attribute("style", "width:100vw;height:100vh;display:block");
        web_sys::window().and_then(|w| w.document()).and_then(|d| d.body()).expect("body").append_child(&canvas).expect("append canvas");
    }
    let mut st = State::new(Some(window.clone()), (1600, 1000)).await;
    let mut frame: u64 = 0;
    // HORDE_MAX_FRAMES=N → run N frames, print the measured average FPS
    // (frames past a 60-frame warmup over wall time), then exit — the
    // end-to-end number: sim + instance build + render + present.
    #[cfg(not(target_arch = "wasm32"))]
    let max_frames: Option<u64> = std::env::var("HORDE_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    #[cfg(not(target_arch = "wasm32"))]
    let mut t0: Option<Instant> = None;

    let handler = move |event, elwt: &winit::event_loop::EventLoopWindowTarget<()>| {
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => st.resize(s.width, s.height),
                WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                    if state == ElementState::Pressed {
                        let (mx, my) = (st.last_mouse.0 as f32, st.last_mouse.1 as f32);
                        let hit = |r: (f32, f32, f32, f32)| mx >= r.0 - 8.0 && mx <= r.0 + r.2 + 8.0 && my >= r.1 - 6.0 && my <= r.1 + r.3 + 6.0;
                        if hit(UI_PAUSE) { st.paused = !st.paused; }
                        else if hit(UI_SLIDER) { st.ui_drag = 1; st.apply_slider(1, mx); }
                        else if cfg!(not(target_arch = "wasm32")) && hit(UI_THREADS) { st.ui_drag = 2; st.apply_slider(2, mx); }
                        else { st.dragging = true; }
                    } else { st.dragging = false; st.ui_drag = 0; }
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
                        KeyCode::KeyT => st.sim.tower_threat_mode = !st.sim.tower_threat_mode,
                        KeyCode::KeyF => {
                            st.free_cam = !st.free_cam;
                            if st.free_cam {
                                let t = Vec3::new((WORLD * 0.5) as f32, 12.0, (WORLD * 0.5) as f32);
                                st.cam_pos = t + Vec3::new(st.dist * st.pitch.cos() * st.yaw.cos(), st.dist * st.pitch.sin(), st.dist * st.pitch.cos() * st.yaw.sin());
                            }
                        }
                        KeyCode::BracketRight => { let p = st.pop + 4_000; st.set_population(p); }
                        KeyCode::BracketLeft => { let p = st.pop.saturating_sub(4_000); st.set_population(p); }
                        _ => {}
                    }}
                },
                WindowEvent::CursorMoved { position, .. } => {
                    if st.ui_drag != 0 {
                        let w = st.ui_drag;
                        st.apply_slider(w, position.x as f32);
                    } else if st.dragging {
                        st.yaw += (position.x - st.last_mouse.0) as f32 * 0.01;
                        let (lo, hi) = if st.free_cam { (-1.45, 1.45) } else { (0.05, 1.5) };
                        st.pitch = (st.pitch + (position.y - st.last_mouse.1) as f32 * 0.01).clamp(lo, hi);
                    }
                    st.last_mouse = (position.x, position.y);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let d = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02 };
                    st.dist = (st.dist - d * 30.0).clamp(120.0, 2200.0);
                }
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
                            if st.ui_drag != 0 { let w = st.ui_drag; st.apply_slider(w, mx); }
                            else if st.dragging {
                                st.yaw += (t.location.x - st.last_mouse.0) as f32 * 0.01;
                                let (lo, hi) = if st.free_cam { (-1.45, 1.45) } else { (0.05, 1.5) };
                                st.pitch = (st.pitch + (t.location.y - st.last_mouse.1) as f32 * 0.01).clamp(lo, hi);
                            }
                            st.last_mouse = (t.location.x, t.location.y);
                        }
                        _ => { st.dragging = false; st.ui_drag = 0; }
                    }
                }
                WindowEvent::RedrawRequested => {
                    st.update_and_render();
                    frame += 1;
                    if frame % 15 == 0 {
                        let (d, a) = st.sim.counts();
                        window.set_title(&format!("vectorial-hash — horde (wgpu) · sleep {d} | awake {a} | kills {} · run {} · {:.0} fps{}", st.sim.kills, st.sim.run, st.fps, if st.paused { " · PAUSED" } else { "" }));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(m) = max_frames {
                        if frame == 60 { t0 = Some(Instant::now()); }
                        if frame >= m {
                            if let Some(t) = t0 {
                                let (d, a) = st.sim.counts();
                                println!("horde_wgpu end-to-end: {:.1} fps avg over {} frames (pop {}, sleep {d}, awake {a}, kills {})",
                                    (frame - 60) as f64 / t.elapsed().as_secs_f64(), frame - 60, st.pop, st.sim.kills);
                            }
                            elwt.exit();
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(handler).expect("run");
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn(handler);
    }
}
