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

// Native always; on wasm only with the `web-wgpu` feature (WebGPU via wasm-bindgen).
#![cfg(any(not(target_arch = "wasm32"), feature = "web-wgpu"))]
// On wasm the entry point is `start` (#[wasm_bindgen(start)]), not `main`.
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant; // performance.now()-backed Instant for the wasm-bindgen build

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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use vectorial_hash_demos::siege_sim::model_for;
#[cfg(target_arch = "wasm32")]
use vectorial_hash_demos::siege_sim::SIEGE_MODEL_FILES;
use vectorial_hash_demos::siege_sim::{
    apply, decide, default_body_radius, faction_tint, forest_trees, ground_height, model_file, set_map_seed,
    spawn_army, terrain_height, terrain_surface, volcano_step, Craters, Faction, Fx, FxKind, IUnit,
    Kind, ProjKind, Projectile, Puff, Rng, SepTables, Unit, Volcano, ANIM_FRAMES, MOVE_PREFS,
    PER_FACTION, SKY, WORLD,
};

const MAX_POP: usize = 10000; // max army/side the slider + [ ] keys allow (instance-buffer cap)

/// Fetch a file over HTTP (web build) — used to stream the `.glb` models at
/// startup instead of baking them into the wasm (keeps siege_wgpu.wasm small).
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

/// Build the terrain as a single triangle mesh from the shared heightfield (u32
/// indices — wgpu has no small drawcall cap, so one mesh is fine). Normals from
/// finite differences; lava cells are flagged emissive (full-bright) via alpha=0.
fn build_terrain(craters: &Craters) -> (Vec<TVertex>, Vec<u32>) {
    const RES: usize = 260; // fine grid so crater bowls read smooth
    let step = WORLD / RES as f64;
    let (mut v, mut idx) = (Vec::with_capacity((RES + 1) * (RES + 1)), Vec::new());
    for iz in 0..=RES {
        for ix in 0..=RES {
            let (x, z) = (ix as f64 * step, iz as f64 * step);
            let h = ground_height(x, z, craters);
            let hx = ground_height(x + step, z, craters) - ground_height(x - step, z, craters);
            let hz = ground_height(x, z + step, craters) - ground_height(x, z - step, craters);
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

/// Voxel (blocky) terrain — the wgpu twin of the macroquad `build_voxel_chunks`:
/// each ~10-unit cell is a flat-topped prism at its true height with vertical
/// cliff walls to lower neighbours, per-corner baked AO (darker where taller
/// neighbours crowd a corner) folded into the vertex colour. wgpu has no drawcall
/// cap, so it's one mesh with u32 indices. Normals (up for tops, sideways for
/// walls) feed the shader's lambert; emissive lava cells get alpha=0.
fn build_voxel_terrain(craters: &Craters) -> (Vec<TVertex>, Vec<u32>) {
    const VRES: usize = 80;
    let step = WORLD / VRES as f64;
    let heights: Vec<f64> = (0..VRES * VRES).map(|k| ground_height(((k % VRES) as f64 + 0.5) * step, ((k / VRES) as f64 + 0.5) * step, craters)).collect();
    let hc = |i: i32, j: i32| -> f64 { if i < 0 || j < 0 || i >= VRES as i32 || j >= VRES as i32 { -1000.0 } else { heights[j as usize * VRES + i as usize] } };
    let (mut v, mut idx): (Vec<TVertex>, Vec<u32>) = (Vec::new(), Vec::new());
    for j in 0..VRES {
        for i in 0..VRES {
            let (ii, jj) = (i as i32, j as i32);
            let h = hc(ii, jj);
            let (x0, z0) = (i as f64 * step, j as f64 * step);
            let (x1, z1) = (x0 + step, z0 + step);
            let (base, emissive) = terrain_surface(x0 + step * 0.5, z0 + step * 0.5, h);
            let a = if emissive { 0.0 } else { 1.0 };
            // Per-corner AO from taller neighbours (matches the macroquad mesher).
            let ao = |dx: i32, dz: i32| -> f32 {
                if emissive { return 1.0; }
                let up = h + step * 0.5;
                let (s1, s2) = (hc(ii + dx, jj) > up, hc(ii, jj + dz) > up);
                let sc = hc(ii + dx, jj + dz) > up;
                1.0 - 0.16 * (s1 as i32 + s2 as i32 + (sc && (s1 || s2)) as i32) as f32
            };
            let corners = [(x0, z0, ao(-1, -1)), (x1, z0, ao(1, -1)), (x1, z1, ao(1, 1)), (x0, z1, ao(-1, 1))];
            let bi = v.len() as u32;
            for (vx, vz, k) in corners {
                v.push(TVertex { pos: [vx as f32, h as f32, vz as f32], normal: [0.0, 1.0, 0.0], color: [base[0] * k, base[1] * k, base[2] * k, a] });
            }
            // Quad-flip so AO interpolates along the brighter diagonal.
            if corners[0].2 + corners[2].2 >= corners[1].2 + corners[3].2 {
                idx.extend_from_slice(&[bi, bi + 1, bi + 2, bi, bi + 2, bi + 3]);
            } else {
                idx.extend_from_slice(&[bi + 1, bi + 2, bi + 3, bi + 1, bi + 3, bi]);
            }
            // Cliff walls down to any lower neighbour.
            for (dx, dz, e0, e1, nrm) in [
                (-1i32, 0i32, (x0, z1), (x0, z0), [-1.0f32, 0.0, 0.0]),
                (1, 0, (x1, z0), (x1, z1), [1.0, 0.0, 0.0]),
                (0, -1, (x0, z0), (x1, z0), [0.0, 0.0, -1.0]),
                (0, 1, (x1, z1), (x0, z1), [0.0, 0.0, 1.0]),
            ] {
                let hn = hc(ii + dx, jj + dz);
                if hn < h - 0.01 {
                    let bottom = hn.max(-25.0) as f32;
                    let si = v.len() as u32;
                    let col = [base[0] * 0.78, base[1] * 0.78, base[2] * 0.78, a];
                    v.push(TVertex { pos: [e0.0 as f32, h as f32, e0.1 as f32], normal: nrm, color: col });
                    v.push(TVertex { pos: [e1.0 as f32, h as f32, e1.1 as f32], normal: nrm, color: col });
                    v.push(TVertex { pos: [e1.0 as f32, bottom, e1.1 as f32], normal: nrm, color: col });
                    v.push(TVertex { pos: [e0.0 as f32, bottom, e0.1 as f32], normal: nrm, color: col });
                    idx.extend_from_slice(&[si, si + 1, si + 2, si, si + 2, si + 3]);
                }
            }
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

/// One screen-space (NDC) UI vertex — the 2D overlay (strength bar + sliders),
/// since wgpu has no built-in text. Filled triangles, no depth, alpha-blended.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }

// Overlay layout in **pixels** (top-left origin): the strength bar + the
// population slider track. Shared by the renderer and the mouse hit-test.
const UI_BAR: (f32, f32, f32, f32) = (18.0, 18.0, 280.0, 16.0);
const UI_SLIDER: (f32, f32, f32, f32) = (18.0, 46.0, 280.0, 12.0); // population
const UI_THREADS: (f32, f32, f32, f32) = (18.0, 70.0, 280.0, 12.0); // rayon pool size

/// Push a pixel-space rectangle (top-left origin) as two triangles in NDC.
#[allow(clippy::too_many_arguments)] // a rect + colour + screen dims
fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let (x0, x1) = (px / sw * 2.0 - 1.0, (px + w) / sw * 2.0 - 1.0);
    let (y0, y1) = (1.0 - py / sh * 2.0, 1.0 - (py + h) / sh * 2.0);
    let q = |x, y| UiVertex { pos: [x, y], color };
    v.extend_from_slice(&[q(x0, y0), q(x1, y0), q(x1, y1), q(x0, y0), q(x1, y1), q(x0, y1)]);
}

/// A 3x5 bitmap font (wgpu has no text) — glyphs as ASCII art so they're
/// verifiable in-source. Covers the digits + the few letters the HUD labels use.
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
        'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"],
        'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"],
        'M' => ["101", "111", "111", "101", "101"],
        'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"],
        'R' => ["111", "101", "110", "101", "101"],
        'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"],
        'U' => ["101", "101", "101", "101", "111"],
        'W' => ["101", "101", "101", "111", "101"],
        ':' => ["000", "010", "000", "010", "000"],
        '.' => ["000", "000", "000", "000", "010"],
        _ => ["000", "000", "000", "000", "000"], // space / unknown
    }
}

/// "TRI 1.2M" / "TRI 345K" / "TRI 900" — the frame's triangle count, short.
fn tri_label(t: u64) -> String {
    if t >= 1_000_000 { format!("TRI {}.{}M", t / 1_000_000, (t % 1_000_000) / 100_000) }
    else if t >= 1_000 { format!("TRI {}K", t / 1_000) }
    else { format!("TRI {t}") }
}

/// Draw a string as `px`-sized pixels of the 3x5 font, left-aligned at (x, y).
#[allow(clippy::too_many_arguments)]
fn push_text(v: &mut Vec<UiVertex>, x: f32, y: f32, px: f32, color: [f32; 4], text: &str, sw: f32, sh: f32) {
    let mut cx = x;
    for c in text.chars() {
        let g = glyph(c.to_ascii_uppercase());
        for (row, bits) in g.iter().enumerate() {
            for (col, ch) in bits.char_indices() {
                if ch == '1' {
                    push_quad(v, cx + col as f32 * px, y + row as f32 * px, px, px, color, sw, sh);
                }
            }
        }
        cx += 4.0 * px; // 3 wide + 1 gap
    }
}

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
    upload_skinned(device, cam_buf, layout, &m)
}

/// Upload an already-built `SkinnedModel` (file or procedural) to the GPU.
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

/// A small unit UV sphere as a static `SkinnedModel` (1 identity joint) for the
/// projectile markers — rendered through the same skin pipeline as the units, so
/// cannonballs / lava bombs read as real little balls (not line crosses).
fn proj_sphere() -> vectorial_hash_demos::model::SkinnedModel {
    use vectorial_hash_demos::model::{SkinVertex, SkinnedModel};
    let (rings, sectors) = (8usize, 12usize);
    let mut verts = Vec::new();
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for s in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * s as f32 / sectors as f32;
            let (st, ct) = theta.sin_cos();
            let n = [sp * ct, cp, sp * st]; // unit sphere: pos == normal
            verts.push(SkinVertex { pos: n, normal: n, joints: [0; 4], weights: [1.0, 0.0, 0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] });
        }
    }
    let mut idx = Vec::new();
    let w = (sectors + 1) as u32;
    for r in 0..rings as u32 {
        for s in 0..sectors as u32 {
            let a = r * w + s;
            idx.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
        }
    }
    SkinnedModel { vertices: verts, indices: idx, joint_frames: vec![glam::Mat4::IDENTITY.to_cols_array_2d()], num_joints: 1, n_frames: 1 }
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
    // Cavalry mounts: a horse under every knight (parity with the macroquad demo).
    horse_model: GpuModel,
    horse_inst_buf: wgpu::Buffer,
    // Forest canopies: green sphere blobs (static per map) reusing the sphere mesh.
    tree_inst_buf: wgpu::Buffer,
    tree_n: u32,
    // Projectile markers: a small sphere drawn instanced per live projectile.
    proj_model: GpuModel,
    proj_inst_buf: wgpu::Buffer,
    // Combat effects + projectile markers, drawn as coloured line segments.
    line_pipeline: wgpu::RenderPipeline,
    line_buf: wgpu::Buffer,
    line_cap: usize,
    line_verts: Vec<LVertex>,
    // 2D overlay (strength bar + sliders) — wgpu has no text.
    ui_pipeline: wgpu::RenderPipeline,
    ui_buf: wgpu::Buffer,
    ui_drag: u8, // which slider is being dragged: 0 none, 1 population, 2 threads
    // Live rayon pool for the parallel `decide` pass (the thread slider). Native
    // only — wasm has no threads, so the web build runs `decide` serially.
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
    // scene — the shared simulation (same battle as the macroquad binary)
    units: Vec<Unit>,
    index: Tree3<IUnit>,
    smoke: Vec<Puff>,
    effects: Vec<Fx>,
    projectiles: Vec<Projectile>,
    volcano: Volcano,
    sep: SepTables, // precomputed boids separation-force table
    paused: bool,
    red: usize,  // live Red units (for the window-title HUD)
    blue: usize, // live Blue units
    fps: f32,    // smoothed frames/second
    smooth: bool, // terrain mode: false = voxel (default), true = smooth heightfield
    pop: usize,  // army size per side (live, via the [ ] keys)
    craters: Craters, // shared alterable-terrain state (units sink, mesh deforms)
    rebuild_t: f64,   // throttle for the crater remesh
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
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cam-l"), entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cam-bg"), layout: &cam_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }] });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pl"), bind_group_layouts: &[&cam_layout], push_constant_ranges: &[] });

        // GPU skinning: one shared bind-group layout + pipeline, then one model per
        // distinct (faction,kind) glb. The instance buffer is shared — units are
        // bucketed by model each frame.
        // Sized for the max army the slider / [ ] keys allow (10000/side, i.e.
        // 20000 instances ~= 1.9 MB), so growing the population never resizes.
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("skin-inst"), size: (MAX_POP * 2 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
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

        // On the web, stream the models from `models/<name>.glb` at startup so
        // they aren't baked into the wasm (keeps it ~1.5 MB instead of ~9). Native
        // embeds them. `unit_bytes` / `prop_bytes` hide the split.
        #[cfg(target_arch = "wasm32")]
        let model_table: std::collections::HashMap<&'static str, Vec<u8>> = {
            let mut m = std::collections::HashMap::new();
            for &name in SIEGE_MODEL_FILES {
                m.insert(name, fetch_bytes(&format!("models/{name}")).await);
            }
            m
        };
        #[cfg(not(target_arch = "wasm32"))]
        let unit_bytes = |f: Faction, k: Kind| -> Vec<u8> { model_for(f, k).to_vec() };
        #[cfg(target_arch = "wasm32")]
        let unit_bytes = |f: Faction, k: Kind| -> Vec<u8> { model_table[model_file(f, k)].clone() };
        #[cfg(not(target_arch = "wasm32"))]
        let prop_bytes = |name: &str| -> Vec<u8> {
            match name {
                "castle.glb" => include_bytes!("../../assets/siege/models/castle.glb").to_vec(),
                "horse.glb" => include_bytes!("../../assets/siege/models/horse.glb").to_vec(),
                _ => unreachable!("unknown siege prop {name}"),
            }
        };
        #[cfg(target_arch = "wasm32")]
        let prop_bytes = |name: &str| -> Vec<u8> { model_table[name].clone() };

        // One GpuModel per distinct (faction,kind) glb — the dragon and the cannon
        // are shared across factions, deduped by file name.
        let mut models: Vec<GpuModel> = Vec::new();
        let mut model_idx = [[0usize; 8]; 2];
        let mut seen: Vec<(&'static str, usize)> = Vec::new();
        for f in Faction::ALL {
            for k in Kind::ALL {
                let name = model_file(f, k);
                let idx = match seen.iter().find(|(n, _)| *n == name) {
                    Some(&(_, i)) => i,
                    None => { let i = models.len(); models.push(build_gpu_model(&device, &cam_buf, &skin_layout, &unit_bytes(f, k))); seen.push((name, i)); i }
                };
                model_idx[f.index()][k.index()] = idx;
            }
        }

        // Terrain: voxel (blocky) by default; $SIEGE_SMOOTH=1 or the `V` key
        // switches to the smooth heightfield. A single mesh + its own pipeline.
        let smooth = std::env::var("SIEGE_SMOOTH").is_ok();
        let craters = Craters::new();
        let (tv, ti) = if smooth { build_terrain(&craters) } else { build_voxel_terrain(&craters) };
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
        let castle_model = build_gpu_model(&device, &cam_buf, &skin_layout, &prop_bytes("castle.glb"));
        let castle_inst: Vec<SkinInstance> = Faction::ALL.iter().map(|&f| {
            let (cx, cz) = f.castle();
            let yaw = ((WORLD * 0.5 - cx) as f32).atan2((WORLD * 0.5 - cz) as f32);
            let m = glam::Mat4::from_translation(glam::Vec3::new(cx as f32, terrain_height(cx, cz) as f32, cz as f32)) * glam::Mat4::from_rotation_y(yaw) * glam::Mat4::from_scale(glam::Vec3::splat(62.0));
            SkinInstance { model: m.to_cols_array_2d(), color: faction_tint(f), frame_base: 0, _pad: [0; 3] }
        }).collect();
        let castle_inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("castle-inst"), contents: bytemuck::cast_slice(&castle_inst), usage: wgpu::BufferUsages::VERTEX });

        // Cavalry mount: one GPU-skinned horse per live knight. Instances rebuilt
        // each frame; the buffer is sized for the worst case (every unit a knight).
        let horse_model = build_gpu_model(&device, &cam_buf, &skin_layout, &prop_bytes("horse.glb"));
        let horse_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("horse-inst"), size: (MAX_POP * 2 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        // Forest canopies: two green sphere blobs per tree (movement slowdown +
        // ranged cover live in the sim). Static per map — drawn with the sphere mesh.
        let mut tree_inst: Vec<SkinInstance> = Vec::new();
        for (fx, fz) in forest_trees() {
            let ty = terrain_height(fx, fz) as f32;
            let blob = |y: f32, r: f32, c: [f32; 4]| {
                let m = glam::Mat4::from_translation(glam::Vec3::new(fx as f32, ty + y, fz as f32)) * glam::Mat4::from_scale(glam::Vec3::splat(r));
                SkinInstance { model: m.to_cols_array_2d(), color: c, frame_base: 0, _pad: [0; 3] }
            };
            tree_inst.push(blob(9.0, 7.5, [0.11, 0.32, 0.13, 1.0]));
            tree_inst.push(blob(15.0, 5.2, [0.16, 0.42, 0.18, 1.0]));
        }
        let tree_n = tree_inst.len() as u32;
        if tree_inst.is_empty() { tree_inst.push(SkinInstance { model: [[0.0; 4]; 4], color: [0.0; 4], frame_base: 0, _pad: [0; 3] }); }
        let tree_inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tree-inst"), contents: bytemuck::cast_slice(&tree_inst), usage: wgpu::BufferUsages::VERTEX });

        // Projectile sphere + its instance buffer (one instance per live shot).
        let proj_model = upload_skinned(&device, &cam_buf, &skin_layout, &proj_sphere());
        let proj_inst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("proj-inst"), size: (512 * std::mem::size_of::<SkinInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

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

        // 2D overlay pipeline (NDC, no camera bind group, drawn over everything).
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
        // Sized for the bars/sliders + the 3x5-font numeric HUD (each glyph pixel
        // is a quad = 6 verts, so a few labelled lines run to a few thousand).
        let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui-v"), size: (16384 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let depth = make_depth(&device, &config);
        // Per-run seed (reproducible via $SIEGE_SEED) drives the shared map + army.
        let seed = std::env::var("SIEGE_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0x51E6E);
        set_map_seed((seed % 100_000) as f64 * 0.01);
        let mut rng = Rng::new(seed | 1);
        let pop = PER_FACTION;
        let units = spawn_army(&mut rng, pop);
        let index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);
        // Live rayon pool for the parallel decide pass (the thread slider). Native
        // only — wasm decides serially.
        #[cfg(not(target_arch = "wasm32"))]
        let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        #[cfg(not(target_arch = "wasm32"))]
        let pool = rayon::ThreadPoolBuilder::new().num_threads(max_threads).build().unwrap();

        State {
            surface, device, queue, config,
            skin_pipeline, models, model_idx, inst_buf,
            castle_model, castle_inst_buf,
            horse_model, horse_inst_buf,
            tree_inst_buf, tree_n,
            proj_model, proj_inst_buf,
            line_pipeline, line_buf, line_cap, line_verts: Vec::new(),
            ui_pipeline, ui_buf, ui_drag: 0,
            #[cfg(not(target_arch = "wasm32"))]
            pool,
            #[cfg(not(target_arch = "wasm32"))]
            n_threads: max_threads,
            #[cfg(not(target_arch = "wasm32"))]
            max_threads,
            terrain_pipeline, terrain_vbuf, terrain_ibuf, terrain_nidx: ti.len() as u32,
            cam_buf, cam_bg, depth, units, index,
            smoke: Vec::new(), effects: Vec::new(), projectiles: Vec::new(),
            volcano: Volcano::new(), sep: SepTables::new(&default_body_radius()), paused: false,
            red: 0, blue: 0, fps: 0.0, smooth, pop, craters, rebuild_t: 0.0,
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

    /// Rebuild the terrain mesh for the current `smooth` mode + craters (V key,
    /// or after a crater lands).
    fn rebuild_terrain(&mut self) {
        let (tv, ti) = if self.smooth { build_terrain(&self.craters) } else { build_voxel_terrain(&self.craters) };
        self.terrain_vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ter-v"), contents: bytemuck::cast_slice(&tv), usage: wgpu::BufferUsages::VERTEX });
        self.terrain_ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ter-i"), contents: bytemuck::cast_slice(&ti), usage: wgpu::BufferUsages::INDEX });
        self.terrain_nidx = ti.len() as u32;
    }

    /// Respawn both armies at a new per-side size (the [ ] keys / slider).
    fn set_population(&mut self, pop: usize) {
        let pop = pop.clamp(20, MAX_POP);
        if pop == self.pop { return; } // no churn while the slider sits still
        self.pop = pop;
        self.units = spawn_army(&mut self.rng, self.pop);
    }

    /// Resize the rayon pool that runs the parallel decide pass (thread slider).
    #[cfg(not(target_arch = "wasm32"))]
    fn set_threads(&mut self, n: usize) {
        let n = n.clamp(1, self.max_threads);
        if n == self.n_threads { return; }
        self.n_threads = n;
        self.pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    }

    /// Map a mouse-x to slider `which` (1 = population, 2 = threads) and apply it.
    fn apply_slider(&mut self, which: u8, mx: f32) {
        match which {
            1 => { let (tx, _, tw, _) = UI_SLIDER; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_population(20 + (frac * (MAX_POP as f32 - 20.0)) as usize); }
            #[cfg(not(target_arch = "wasm32"))]
            2 => { let (tx, _, tw, _) = UI_THREADS; let frac = ((mx - tx) / tw).clamp(0.0, 1.0); self.set_threads(1 + (frac * (self.max_threads as f32 - 1.0)).round() as usize); }
            _ => {}
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
            // Decide (read-only on the indices) → apply (serial) → volcano. Native:
            // fan out over the sized rayon pool (the thread slider). wasm: serial.
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (pool, units) = (&self.pool, &mut self.units);
                let (idx, smk, br) = (&self.index, &smoke_index, &self.sep);
                pool.install(|| units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, br)));
            }
            #[cfg(target_arch = "wasm32")]
            for i in 0..self.units.len() { decide(&mut self.units[i], i as u32, &self.index, &smoke_index, &self.sep); }
            let impacts = apply(&mut self.units, &mut self.smoke, &mut self.effects, &mut self.projectiles, &self.craters, &mut self.rng, dt, self.now);
            volcano_step(&mut self.volcano, &mut self.smoke, &mut self.effects, &mut self.projectiles, &mut self.rng, dt, self.now);
            // Alterable terrain: carve craters at the impacts (shared with the sim,
            // so units sink in too), remesh throttled — voxel *and* smooth deform.
            for (ip, r) in impacts { self.craters.carve(ip.x, ip.z, r * 0.85); }
            self.rebuild_t -= dt;
            if self.craters.dirty && self.rebuild_t <= 0.0 {
                self.craters.dirty = false;
                self.rebuild_t = 0.4;
                self.rebuild_terrain();
            }
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
            let mut y = (u.p.y - u.kind.radius() as f64) as f32; // feet: drop the sim's (crater-aware) centre to the ground — no terrain recompute
            // Per-model orientation/size correction (e.g. the undead mage slime
            // faces +X and reads big) — same tweak the macroquad demo applies.
            let (yaw_off, mut scale_mul) = u.kind.model_tweak(u.faction);
            // Cavalry rider: lift onto the horse and shrink to rider size (the
            // horse is drawn separately) — mirrors the macroquad placement.
            if u.kind == Kind::Knight { y += (u.kind.model_height() * 0.5) as f32; scale_mul *= 0.72; }
            let model = glam::Mat4::from_translation(glam::Vec3::new(u.p.x as f32, y, u.p.z as f32)) * glam::Mat4::from_rotation_y(u.face + yaw_off) * glam::Mat4::from_scale(glam::Vec3::splat(u.kind.model_height() * scale_mul));
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
        // Cavalry mounts: a horse under each live knight, animated with the horse's
        // own clip (parity with the macroquad demo, which the wgpu build lacked).
        let mut horse_inst: Vec<SkinInstance> = Vec::new();
        let hnf = self.horse_model.n_frames.max(1);
        let hnj = self.horse_model.num_joints;
        for (i, u) in self.units.iter().enumerate() {
            if !u.alive() || u.kind != Kind::Knight { continue; }
            let gy = ground_height(u.p.x, u.p.z, &self.craters) as f32;
            let model = glam::Mat4::from_translation(glam::Vec3::new(u.p.x as f32, gy, u.p.z as f32)) * glam::Mat4::from_rotation_y(u.face) * glam::Mat4::from_scale(glam::Vec3::splat(Kind::Knight.model_height()));
            let group = (i as u32 % 5) as f32 / 5.0;
            let frame = (((self.now as f32 * 1.6 + group) * hnf as f32) as u32) % hnf;
            horse_inst.push(SkinInstance { model: model.to_cols_array_2d(), color: faction_tint(u.faction), frame_base: frame * hnj, _pad: [0; 3] });
        }
        let horse_n = (horse_inst.len().min(MAX_POP * 2)) as u32;
        if horse_n > 0 { self.queue.write_buffer(&self.horse_inst_buf, 0, bytemuck::cast_slice(&horse_inst[..horse_n as usize])); }
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
        self.line_verts = lv;
        // Projectiles → small sphere instances (cannonball / lava bomb), drawn
        // through the skin pipeline like the units — no more line crosses.
        let mut proj_inst: Vec<SkinInstance> = Vec::with_capacity(self.projectiles.len().min(512));
        for pr in self.projectiles.iter().take(512) {
            let r = match pr.kind { ProjKind::Cannon => 3.4, ProjKind::LavaRock => 4.6 };
            let col = match pr.kind { ProjKind::Cannon => [0.10, 0.10, 0.12, 1.0], ProjKind::LavaRock => [1.0, 0.45, 0.10, 1.0] };
            let m = glam::Mat4::from_translation(glam::Vec3::new(pr.p.x as f32, pr.p.y as f32, pr.p.z as f32)) * glam::Mat4::from_scale(glam::Vec3::splat(r));
            proj_inst.push(SkinInstance { model: m.to_cols_array_2d(), color: col, frame_base: 0, _pad: [0; 3] });
        }
        if !proj_inst.is_empty() { self.queue.write_buffer(&self.proj_inst_buf, 0, bytemuck::cast_slice(&proj_inst)); }
        let proj_n = proj_inst.len() as u32;
        // Grow the line buffer if this frame overflows it.
        if self.line_verts.len() > self.line_cap {
            self.line_cap = (self.line_verts.len() * 2).next_power_of_two();
            self.line_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("line-v"), size: (self.line_cap * std::mem::size_of::<LVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        }
        if !self.line_verts.is_empty() { self.queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&self.line_verts)); }
        let line_n = self.line_verts.len() as u32;

        // 2D overlay: a Red|Blue strength bar (live counts) + the population slider.
        let (sw, sh) = (self.config.width as f32, self.config.height as f32);
        let mut ui: Vec<UiVertex> = Vec::new();
        let (bx, by, bw, bh) = UI_BAR;
        let total = (self.red + self.blue).max(1) as f32;
        let rw = bw * self.red as f32 / total;
        push_quad(&mut ui, bx - 2.0, by - 2.0, bw + 4.0, bh + 4.0, [0.0, 0.0, 0.0, 0.45], sw, sh); // backing
        push_quad(&mut ui, bx, by, rw, bh, [0.90, 0.25, 0.20, 0.95], sw, sh); // Red strength
        push_quad(&mut ui, bx + rw, by, bw - rw, bh, [0.30, 0.45, 1.0, 0.95], sw, sh); // Blue strength
        let (tx, ty, tw, th) = UI_SLIDER; // population (white handle)
        push_quad(&mut ui, tx, ty, tw, th, [0.22, 0.22, 0.28, 0.85], sw, sh);
        let frac = ((self.pop as f32 - 20.0) / (MAX_POP as f32 - 20.0)).clamp(0.0, 1.0);
        push_quad(&mut ui, tx + frac * tw - 5.0, ty - 3.0, 10.0, th + 6.0, [0.95, 0.95, 0.98, 1.0], sw, sh);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (hx, hy, hw, hh) = UI_THREADS; // rayon threads (green handle) — native only
            push_quad(&mut ui, hx, hy, hw, hh, [0.22, 0.22, 0.28, 0.85], sw, sh);
            let tfrac = if self.max_threads > 1 { (self.n_threads as f32 - 1.0) / (self.max_threads as f32 - 1.0) } else { 1.0 };
            push_quad(&mut ui, hx + tfrac * hw - 5.0, hy - 3.0, 10.0, hh + 6.0, [0.40, 0.95, 0.55, 1.0], sw, sh);
        }
        // Triangles drawn this frame (terrain + every instanced model).
        let mut tris: u64 = (self.terrain_nidx / 3) as u64;
        for (mi, m) in self.models.iter().enumerate() {
            let (s, e) = ranges[mi];
            tris += (m.nidx as u64 / 3) * (e - s) as u64;
        }
        tris += (self.castle_model.nidx as u64 / 3) * 2;
        tris += (self.horse_model.nidx as u64 / 3) * horse_n as u64;
        tris += (self.proj_model.nidx as u64 / 3) * (proj_n + self.tree_n) as u64;
        // Numeric HUD — a labelled column top-right (wgpu has no text; 3x5 font).
        let white = [0.92, 0.94, 0.98, 1.0];
        let redc = [1.0, 0.50, 0.44, 1.0];
        let bluec = [0.55, 0.68, 1.0, 1.0];
        let hx = sw - 150.0;
        push_text(&mut ui, hx, 12.0, 3.0, white, &format!("FPS {:.0}", self.fps), sw, sh);
        push_text(&mut ui, hx, 30.0, 3.0, redc, &format!("RED {}", self.red), sw, sh);
        push_text(&mut ui, hx, 48.0, 3.0, bluec, &format!("BLU {}", self.blue), sw, sh);
        push_text(&mut ui, hx, 66.0, 3.0, white, &format!("POP {}", self.pop), sw, sh);
        push_text(&mut ui, hx, 84.0, 3.0, white, &tri_label(tris), sw, sh);
        // Collision (separation) state — green = on, red = off (toggle with C).
        let col_on = vectorial_hash_demos::siege_sim::separation_on();
        let colc = if col_on { [0.55, 1.0, 0.60, 1.0] } else { [1.0, 0.55, 0.40, 1.0] };
        push_text(&mut ui, hx, 102.0, 3.0, colc, "COL", sw, sh);
        ui.truncate(16384); // never exceed the ui buffer (guards write_buffer)
        self.queue.write_buffer(&self.ui_buf, 0, bytemuck::cast_slice(&ui));
        let ui_n = ui.len() as u32;

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
            // cavalry horses — one instance per live knight.
            if horse_n > 0 {
                let hm = &self.horse_model;
                pass.set_bind_group(0, &hm.bind, &[]);
                pass.set_vertex_buffer(0, hm.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.horse_inst_buf.slice(..));
                pass.set_index_buffer(hm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..hm.nidx, 0, 0..horse_n);
            }
            // castles — the static model, two instances (one per keep).
            let cm = &self.castle_model;
            pass.set_bind_group(0, &cm.bind, &[]);
            pass.set_vertex_buffer(0, cm.vbuf.slice(..));
            pass.set_vertex_buffer(1, self.castle_inst_buf.slice(..));
            pass.set_index_buffer(cm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cm.nidx, 0, 0..2);
            // forest canopies — green sphere blobs (reuse the projectile sphere mesh).
            if self.tree_n > 0 {
                let pm = &self.proj_model;
                pass.set_bind_group(0, &pm.bind, &[]);
                pass.set_vertex_buffer(0, pm.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.tree_inst_buf.slice(..));
                pass.set_index_buffer(pm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..pm.nidx, 0, 0..self.tree_n);
            }
            // projectiles — the sphere model, one instance per live shot.
            if proj_n > 0 {
                let pm = &self.proj_model;
                pass.set_bind_group(0, &pm.bind, &[]);
                pass.set_vertex_buffer(0, pm.vbuf.slice(..));
                pass.set_vertex_buffer(1, self.proj_inst_buf.slice(..));
                pass.set_index_buffer(pm.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..pm.nidx, 0, 0..proj_n);
            }
            // combat effects (alpha-blended lines).
            if line_n > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.cam_bg, &[]);
                pass.set_vertex_buffer(0, self.line_buf.slice(..));
                pass.draw(0..line_n, 0..1);
            }
            // 2D overlay (strength bar + slider), drawn over everything.
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

// Real GPU skeletal skinning: the vertex shader looks up this instance's bone
// matrices (frame_base + jointᵢ) from a storage buffer and blends them by weight.
// Per-instance animation, thousands of units, zero CPU vertex skinning — the
// thing macroquad's WebGL1 stack cannot do.
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
    // The model's own colour, nudged toward the faction tint by the tint's alpha
    // (so even dark models read clearly as Red / Blue) — matches the macroquad mix.
    o.color = vec4<f32>(mix(vcolor.rgb, tint.rgb, tint.a), 1.0);
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Per-pixel lambert (interpolated normal) — smooth on big models (slime, dragon).
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
    o.color = col;  // alpha < 0.5 flags an emissive (lava) cell
    o.normal = n;
    return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Per-pixel lighting (the interpolated normal) — smooth on steep slopes,
    // no Gouraud facets on the volcano cone.
    let diff = max(dot(normalize(in.normal), normalize(cam.light.xyz)), 0.0);
    let sh = select(0.40 + 0.60 * diff, 1.0, in.color.a < 0.5);
    return vec4<f32>(in.color.rgb * sh, 1.0);
}
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

// 2D screen-space (NDC) overlay — the strength bar + sliders. No camera.
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

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    pollster::block_on(run());
}

// Web entry (wasm-bindgen): set the panic hook + drive the async `run` on the
// browser's event loop. `main` is native-only; on wasm the runtime calls `start`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash — siege (wgpu)").with_inner_size(winit::dpi::LogicalSize::new(1600, 1000)).build(&event_loop).expect("window"));
    // On the web, mount the winit canvas into the page (full-window).
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        let _ = canvas.set_attribute("style", "width:100vw;height:100vh;display:block");
        web_sys::window().and_then(|w| w.document()).and_then(|d| d.body()).expect("body").append_child(&canvas).expect("append canvas");
    }
    let mut st = State::new(window.clone()).await;
    let max_frames: Option<u64> = std::env::var("SIEGE_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut frame: u64 = 0;

    let handler = move |event, elwt: &winit::event_loop::EventLoopWindowTarget<()>| {
            elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(s) => st.resize(s.width, s.height),
                    WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                        if state == ElementState::Pressed {
                            // Over a slider → drag it (population / threads), not the camera.
                            let (mx, my) = (st.last_mouse.0 as f32, st.last_mouse.1 as f32);
                            let hit = |r: (f32, f32, f32, f32)| mx >= r.0 - 8.0 && mx <= r.0 + r.2 + 8.0 && my >= r.1 - 6.0 && my <= r.1 + r.3 + 6.0;
                            if hit(UI_SLIDER) { st.ui_drag = 1; st.apply_slider(1, mx); }
                            else if cfg!(not(target_arch = "wasm32")) && hit(UI_THREADS) { st.ui_drag = 2; st.apply_slider(2, mx); }
                            else { st.dragging = true; }
                        } else { st.dragging = false; st.ui_drag = 0; }
                    }
                    WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, .. }, .. } => match code {
                        KeyCode::KeyP => st.paused = !st.paused,
                        KeyCode::KeyV => { st.smooth = !st.smooth; st.rebuild_terrain(); } // voxel ↔ smooth
                        KeyCode::KeyC => vectorial_hash_demos::siege_sim::set_separation(!vectorial_hash_demos::siege_sim::separation_on()), // collisions on/off
                        KeyCode::BracketRight => { let p = st.pop + 100; st.set_population(p); }
                        KeyCode::BracketLeft => { let p = st.pop.saturating_sub(100); st.set_population(p); }
                        _ => {}
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        if st.ui_drag != 0 {
                            let w = st.ui_drag;
                            st.apply_slider(w, position.x as f32);
                        } else if st.dragging {
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
        };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(handler).expect("run");
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn(handler); // returns immediately; the browser drives the loop
    }
}
