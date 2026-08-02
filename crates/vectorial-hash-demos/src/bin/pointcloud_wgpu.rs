//! pointcloud_wgpu — a viewer for a large, **static, strongly skewed** point cloud:
//! exactly the case the median-split [`KdTree3`] was built for, made visible.
//!
//! The cloud is a procedurally "scanned" scene (ground surface + building shells +
//! tree canopies), so its density is wildly uneven: dense on surfaces, empty in the
//! air between them. That skew is the whole point — a uniform grid has to pick one
//! cell size for both.
//!
//! What it measures, live, on the real cloud (`M` cycles the structure):
//!   · **build** — one from-scratch construction,
//!   · **k-NN pass** — the k nearest for EVERY point (that's what colours the cloud by
//!     local density), i.e. N k-NN queries back to back,
//!   · **radius query** — the sphere you drag around with the mouse.
//! Switching structures re-runs all three and cross-checks the k-NN distances against
//! the previous structure, so a fast number is also a correct one.
//!
//! Controls: drag to orbit · wheel to zoom · move the mouse to sweep the probe sphere ·
//! `M` structure · `K` k (8/16/32) · `[` `]` cloud size · `C` colour mode.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin pointcloud_wgpu --release
//! ```
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
use vectorial_hash::{Aabb, KdTree3, MortonGrid3, Octree3, Point3, Positioned3, Sphere3};

const WORLD: f32 = 1000.0;
const MAX_N: usize = 400_000;
const PROBE_R: f32 = 55.0;
// The world box every structure is built over. Points MUST stay inside it: Aabb is
// half-open, so a point at exactly WORLD (or below the floor) is outside — which makes
// Octree3::bulk_load panic ("octants tile the parent") and MortonGrid3 silently reject
// the insert. Generated points are clamped into it (caught by the smoke test's bounds).
const Y_LO: f32 = -20.0;
const Y_HI: f32 = 380.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Inst { x: f32, y: f32, z: f32, heat: f32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }

fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let x0 = px / sw * 2.0 - 1.0; let x1 = (px + w) / sw * 2.0 - 1.0;
    let y0 = 1.0 - py / sh * 2.0; let y1 = 1.0 - (py + h) / sh * 2.0;
    for p in [[x0, y0], [x1, y0], [x0, y1], [x0, y1], [x1, y0], [x1, y1]] { v.push(UiVertex { pos: p, color }); }
}
fn glyph(c: char) -> [&'static str; 5] {
    match c {
        '0' => ["111", "101", "101", "101", "111"], '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"], '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"], '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"], '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"], '9' => ["111", "101", "111", "001", "111"],
        'A' => ["111", "101", "111", "101", "101"], 'B' => ["110", "101", "110", "101", "110"],
        'C' => ["111", "100", "100", "100", "111"], 'D' => ["110", "101", "101", "101", "110"],
        'E' => ["111", "100", "110", "100", "111"], 'F' => ["111", "100", "110", "100", "100"],
        'G' => ["111", "100", "101", "101", "111"], 'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"], 'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"], 'M' => ["101", "111", "111", "101", "101"],
        'N' => ["101", "111", "111", "111", "101"], 'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"], 'Q' => ["111", "101", "101", "111", "011"],
        'R' => ["111", "101", "111", "110", "101"], 'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"], 'U' => ["101", "101", "101", "101", "111"],
        'V' => ["101", "101", "101", "101", "010"], 'W' => ["101", "101", "111", "111", "101"],
        'X' => ["101", "101", "010", "101", "101"], 'Y' => ["101", "101", "010", "010", "010"],
        'Z' => ["111", "001", "010", "100", "111"], '.' => ["000", "000", "000", "000", "010"],
        '-' => ["000", "000", "111", "000", "000"], ':' => ["000", "010", "000", "010", "000"],
        '/' => ["001", "001", "010", "100", "100"], '%' => ["101", "001", "010", "100", "101"],
        _ => ["000", "000", "000", "000", "000"],
    }
}
fn push_text(v: &mut Vec<UiVertex>, x: f32, y: f32, px: f32, color: [f32; 4], text: &str, sw: f32, sh: f32) {
    let mut cx = x;
    for c in text.chars() {
        for (row, bits) in glyph(c.to_ascii_uppercase()).iter().enumerate() {
            for (col, ch) in bits.char_indices() {
                if ch == '1' { push_quad(v, cx + col as f32 * px, y + row as f32 * px, px, px, color, sw, sh); }
            }
        }
        cx += 4.0 * px;
    }
}

// ---- the cloud ---------------------------------------------------------------
#[derive(Clone, Copy)]
struct CP { id: u32, p: Point3 }
impl Positioned3 for CP { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u32 << 24) as f32 }
    fn r(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.f() }
}

/// A "scanned" scene: a rolling ground surface, a few building shells and some tree
/// canopies. Points sit ON surfaces, so the cloud is dense in thin sheets and empty
/// in between — the density skew a uniform grid can't size a cell for.
fn scan_scene(n: usize) -> Vec<Point3> {
    let mut r = Rng(0xC10D_5EED);
    let mut pts: Vec<Point3> = Vec::with_capacity(n);
    let keep = |x: f32, y: f32, z: f32| Point3::new(
        x.clamp(0.5, WORLD - 0.5) as f64,
        y.clamp(Y_LO + 0.5, Y_HI - 0.5) as f64,
        z.clamp(0.5, WORLD - 0.5) as f64);
    let ground_h = |x: f32, z: f32| {
        30.0 + 45.0 * ((x * 0.006).sin() * (z * 0.005).cos())
            + 18.0 * ((x * 0.017 + 1.3).sin() * (z * 0.013).sin())
    };
    // buildings: axis-aligned shells scattered on the ground
    let boxes: Vec<(f32, f32, f32, f32, f32)> = (0..7)
        .map(|_| {
            let (cx, cz) = (r.r(120.0, WORLD - 120.0), r.r(120.0, WORLD - 120.0));
            (cx, cz, r.r(45.0, 120.0), r.r(45.0, 120.0), r.r(60.0, 190.0))
        })
        .collect();
    let trees: Vec<(f32, f32, f32)> = (0..26)
        .map(|_| { let (x, z) = (r.r(40.0, WORLD - 40.0), r.r(40.0, WORLD - 40.0)); (x, z, r.r(18.0, 34.0)) })
        .collect();
    while pts.len() < n {
        let pick = r.f();
        if pick < 0.55 {
            // ground sheet
            let (x, z) = (r.r(0.0, WORLD), r.r(0.0, WORLD));
            pts.push(keep(x, ground_h(x, z) + r.r(-1.2, 1.2), z));
        } else if pick < 0.85 {
            // building shell: pick a face, then a point on it
            let b = boxes[(r.f() * boxes.len() as f32) as usize % boxes.len()];
            let (cx, cz, hw, hd, hh) = b;
            let base = ground_h(cx, cz);
            let (u, v) = (r.r(-1.0, 1.0), r.f());
            let (x, y, z) = match (r.f() * 5.0) as u32 {
                0 => (cx - hw, base + v * hh, cz + u * hd),
                1 => (cx + hw, base + v * hh, cz + u * hd),
                2 => (cx + u * hw, base + v * hh, cz - hd),
                3 => (cx + u * hw, base + v * hh, cz + hd),
                _ => (cx + u * hw, base + hh, cz + r.r(-1.0, 1.0) * hd), // roof
            };
            pts.push(keep(x + r.r(-0.8, 0.8), y + r.r(-0.8, 0.8), z + r.r(-0.8, 0.8)));
        } else {
            // tree canopy: a fuzzy shell of a sphere on a trunk
            let t = trees[(r.f() * trees.len() as f32) as usize % trees.len()];
            let (x0, z0, rad) = t;
            let base = ground_h(x0, z0);
            if r.f() < 0.25 {
                pts.push(keep(x0 + r.r(-1.5, 1.5), base + r.f() * rad * 1.4, z0 + r.r(-1.5, 1.5)));
            } else {
                let (a, b2) = (r.r(0.0, std::f32::consts::TAU), r.r(-1.0, 1.0));
                let s = (1.0 - b2 * b2).max(0.0).sqrt();
                let rr = rad * r.r(0.75, 1.0);
                pts.push(keep(x0 + rr * s * a.cos(), base + rad * 1.5 + rr * b2, z0 + rr * s * a.sin()));
            }
        }
    }
    pts
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Structure { Kd, Oct, Mor }
impl Structure {
    fn next(self) -> Self { match self { Structure::Kd => Structure::Oct, Structure::Oct => Structure::Mor, Structure::Mor => Structure::Kd } }
    fn label(self) -> &'static str { match self { Structure::Kd => "KDTREE3 MEDIAN", Structure::Oct => "OCTREE3 MIDPOINT", Structure::Mor => "MORTONGRID3 FLAT" } }
}

enum Index { Kd(KdTree3<CP>), Oct(Octree3<CP>), Mor(MortonGrid3<CP>) }
impl Index {
    fn build(kind: Structure, pts: &[Point3]) -> (Index, f32) {
        let world = Aabb::new(0.0, Y_LO as f64, 0.0, WORLD as f64, (Y_HI - Y_LO) as f64, WORLD as f64);
        let items: Vec<CP> = pts.iter().enumerate().map(|(i, p)| CP { id: i as u32, p: *p }).collect();
        let t = Instant::now();
        let idx = match kind {
            Structure::Kd => Index::Kd(KdTree3::from_items(24, items)),
            Structure::Oct => Index::Oct(Octree3::bulk_load(world, 24, items)),
            Structure::Mor => {
                // cell ~ the k-NN neighbourhood scale; the flat grid must pick ONE
                let lv = MortonGrid3::<CP>::levels_for_cell_size(world, 16.0);
                let mut g = MortonGrid3::new(world, lv);
                for it in pts.iter().enumerate().map(|(i, p)| CP { id: i as u32, p: *p }) { g.insert(it); }
                Index::Mor(g)
            }
        };
        (idx, t.elapsed().as_secs_f32() * 1e3)
    }
    fn knn_d(&self, q: Point3, k: usize) -> f64 {
        // mean distance to the k nearest — the local-density estimate that colours the cloud
        let v = match self { Index::Kd(t) => t.knn(q, k), Index::Oct(t) => t.knn(q, k), Index::Mor(g) => g.knn(q, k) };
        if v.is_empty() { return 0.0; }
        v.iter().map(|(d, _)| *d).sum::<f64>() / v.len() as f64
    }
    fn cull_ids(&self, s: &Sphere3, out: &mut Vec<u32>) {
        match self {
            Index::Kd(t) => out.extend(t.cull(s).iter().map(|c| c.id)),
            Index::Oct(t) => out.extend(t.cull(s).iter().map(|c| c.id)),
            Index::Mor(g) => out.extend(g.cull(s).iter().map(|c| c.id)),
        }
    }
}

/// The k-NN pass that colours the cloud: N queries, timed. Returns (ms, per-point mean
/// neighbour distance) so the caller can both time it and paint with it.
fn knn_pass(index: &Index, pts: &[Point3], k: usize, out: &mut Vec<f32>) -> f32 {
    out.clear();
    out.reserve(pts.len());
    let t = Instant::now();
    for p in pts { out.push(index.knn_d(*p, k) as f32); }
    t.elapsed().as_secs_f32() * 1e3
}

/// Robust colour range for the density read-out: the 5th/95th percentiles of a sample.
/// A fixed scale washes the cloud out — the actual spread of mean-k-NN distance depends
/// on point count and k, so calibrate to what's really there (the first screenshot came
/// out uniformly cream because everything mapped to the top of a guessed 0..30 range).
fn density_range(dens: &[f32]) -> (f32, f32) {
    if dens.is_empty() { return (0.0, 1.0); }
    let mut sample: Vec<f32> = dens.iter().step_by((dens.len() / 4096).max(1)).copied().collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f32| sample[((sample.len() - 1) as f32 * q) as usize];
    let (lo, hi) = (at(0.05), at(0.95));
    (lo, if hi > lo + 1e-6 { hi } else { lo + 1.0 })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // `$CLOUD_HEADLESS=1` builds the cloud, builds the index and runs the k-NN pass with no
    // window and no GPU. This demo's numbers are a build time and a one-k-NN-per-point pass —
    // neither needs a renderer to exist, and requiring one meant the comparison could not be
    // swept in CI or on a display-less machine.
    if std::env::var("CLOUD_HEADLESS").ok().as_deref() == Some("1") {
        headless();
        return;
    }
    pollster::block_on(run());
}

/// Build + k-NN with no renderer, reporting what the HUD reports.
#[cfg(not(target_arch = "wasm32"))]
fn headless() {
    let n = std::env::var("CLOUD_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120_000usize).min(MAX_N);
    let k = std::env::var("CLOUD_K").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let pts = scan_scene(n);
    let mut dens: Vec<f32> = Vec::new();

    println!("pointcloud headless | {n} points | k={k}\n");
    println!("{:>10} {:>12} {:>12} {:>14}", "structure", "build ms", "knn ms", "us/point");
    // Every structure answers the same question on the same cloud, so their k-NN distances must
    // agree. A build-once structure that answers *differently* is not a faster option, it is a
    // wrong one — and a headless run has no HUD for anyone to notice that on.
    let mut reference: Option<Vec<f32>> = None;
    for kind in [Structure::Kd, Structure::Oct, Structure::Mor] {
        let (index, build_ms) = Index::build(kind, &pts);
        let knn_ms = knn_pass(&index, &pts, k, &mut dens);
        println!("{:>10} {:>12.1} {:>12.1} {:>14.3}", kind.label(), build_ms, knn_ms, knn_ms * 1000.0 / n as f32);
        // A single token: `kind.label()` is prose ("KDTREE3 MEDIAN") and `#M` keys are parsed
        // on whitespace, so using it here produced two fields where bench-runner expects one.
        let tag = match kind { Structure::Kd => "kdtree3", Structure::Oct => "octree3", Structure::Mor => "morton3" };
        println!("#M cloud_{tag}.build_ms {build_ms:.2} ms");
        println!("#M cloud_{tag}.knn_ms {knn_ms:.2} ms");
        match &reference {
            None => reference = Some(dens.clone()),
            Some(r) => {
                let bad = r.iter().zip(&dens).filter(|(a, b)| (**a - **b).abs() > 1e-4 * a.abs().max(1.0)).count();
                assert!(bad == 0, "{} k-NN distances differ from the first structure's — see docs/POINTCLOUD.md", bad);
            }
        }
    }
    println!("\nall three structures returned identical k-NN distances on {n} points");
}

#[cfg(target_arch = "wasm32")]
fn headless() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); wasm_bindgen_futures::spawn_local(run()); }

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash point cloud").with_inner_size(winit::dpi::LogicalSize::new(1300, 900)).build(&event_loop).unwrap());
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        let _ = canvas.set_attribute("style", "width:100vw;height:100vh;display:block;touch-action:none");
        web_sys::window().and_then(|w| w.document()).and_then(|d| d.body()).expect("body").append_child(&canvas.into()).expect("append canvas");
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.unwrap();
    let size = window.inner_size();
    let dpr = window.scale_factor() as f32;
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: wgpu::PresentMode::AutoNoVsync, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
    surface.configure(&device, &config);

    // ---- cloud + index
    let mut n = std::env::var("CLOUD_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120_000usize).min(MAX_N);
    let mut kind = match std::env::var("CLOUD_INDEX").ok().as_deref() {
        Some("octree") | Some("oct") => Structure::Oct,
        Some("morton") | Some("mor") => Structure::Mor,
        _ => Structure::Kd,
    };
    let mut k = 16usize;
    let mut pts = scan_scene(n);
    let (mut index, mut build_ms) = Index::build(kind, &pts);
    let mut dens: Vec<f32> = Vec::new();
    let mut knn_ms = knn_pass(&index, &pts, k, &mut dens);
    let mut drange = density_range(&dens);
    // cross-structure agreement: the k-NN distances must not depend on the structure
    let mut agree = String::from("-");

    // ---- gpu
    let inst_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("inst"), size: (MAX_N * std::mem::size_of::<Inst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("points"), layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Inst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(), multiview: None,
    });
    let make_depth = |device: &wgpu::Device, w: u32, h: u32| device.create_texture(&wgpu::TextureDescriptor { label: Some("depth"), size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[] }).create_view(&Default::default());
    let mut depth_view = make_depth(&device, config.width, config.height);
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(), multiview: None,
    });
    let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui"), size: (60_000 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let smoke: Option<u64> = std::env::var("CLOUD_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let (mut yaw, mut pitch, mut dist) = (0.9f32, 0.55f32, 1500.0f32);
    let (mut drag, mut last_mouse) = (false, (0.0f64, 0.0f64));
    let (mut probe, mut probe_us) = (Vec3::new(WORLD * 0.5, 60.0, WORLD * 0.5), 0.0f32);
    let mut probe_n = 0usize;
    let mut sel: Vec<u32> = Vec::new();
    let (mut fps, mut frame) = (0.0f32, 0u64);
    let mut last = Instant::now();
    let mut inst: Vec<Inst> = Vec::with_capacity(MAX_N);
    let mut colour_by_density = true;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); depth_view = make_depth(&device, config.width, config.height); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => drag = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - last_mouse.0, position.y - last_mouse.1);
                last_mouse = (position.x, position.y);
                if drag { yaw += dx as f32 * 0.005; pitch = (pitch + dy as f32 * 0.004).clamp(-1.45, 1.45); }
                else {
                    // sweep the probe over the ground plane under the cursor
                    let fx = (position.x as f32 / config.width as f32).clamp(0.0, 1.0);
                    let fz = (position.y as f32 / config.height as f32).clamp(0.0, 1.0);
                    probe = Vec3::new(fx * WORLD, 90.0, fz * WORLD);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => { let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 90.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 }; dist = (dist - d).clamp(260.0, 3400.0); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                KeyCode::KeyM => {
                    // rebuild + re-run the k-NN pass, and CHECK the new structure agrees
                    let before: Vec<f32> = dens.iter().step_by((n / 512).max(1)).copied().collect();
                    kind = kind.next();
                    let (i2, b) = Index::build(kind, &pts);
                    index = i2; build_ms = b;
                    knn_ms = knn_pass(&index, &pts, k, &mut dens);
                    drange = density_range(&dens);
                    let after: Vec<f32> = dens.iter().step_by((n / 512).max(1)).copied().collect();
                    let bad = before.iter().zip(after.iter()).filter(|(a, b)| (*a - *b).abs() > 1e-3 * (1.0 + a.abs())).count();
                    agree = if bad == 0 { "EXACT".into() } else { format!("{bad} DIFFER") };
                }
                KeyCode::KeyK => {
                    k = match k { 8 => 16, 16 => 32, _ => 8 };
                    knn_ms = knn_pass(&index, &pts, k, &mut dens);
                    drange = density_range(&dens);
                }
                KeyCode::KeyC => colour_by_density = !colour_by_density,
                KeyCode::BracketRight | KeyCode::BracketLeft => {
                    n = if c == KeyCode::BracketRight { (n + 40_000).min(MAX_N) } else { n.saturating_sub(40_000).max(20_000) };
                    pts = scan_scene(n);
                    let (i2, b) = Index::build(kind, &pts);
                    index = i2; build_ms = b;
                    knn_ms = knn_pass(&index, &pts, k, &mut dens);
                    drange = density_range(&dens);
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let fdt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                fps = if fps == 0.0 { 1.0 / fdt } else { fps * 0.9 + 0.1 / fdt };
                frame += 1;

                // ---- the interactive query: a probe sphere swept with the mouse
                sel.clear();
                let t = Instant::now();
                index.cull_ids(&Sphere3::new(probe.x as f64, probe.y as f64, probe.z as f64, PROBE_R as f64), &mut sel);
                probe_us = probe_us * 0.8 + 0.2 * t.elapsed().as_secs_f32() * 1e6;
                probe_n = sel.len();

                let target = Vec3::new(WORLD * 0.5, 70.0, WORLD * 0.5);
                let eye = target + Vec3::new(dist * pitch.cos() * yaw.cos(), dist * pitch.sin(), dist * pitch.cos() * yaw.sin());
                let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                let proj = Mat4::perspective_rh(50f32.to_radians(), config.width as f32 / config.height as f32, 1.0, 9000.0);
                queue.write_buffer(&cam_b, 0, bytemuck::bytes_of(&Cam { vp: (proj * view).to_cols_array_2d() }));

                // ---- instances: colour by local density (or height), selection highlighted
                inst.clear();
                let mut hot = vec![false; pts.len()];
                for &i in &sel { hot[i as usize] = true; }
                let (dlo, dhi) = drange;
                for (i, p) in pts.iter().enumerate() {
                    let heat = if hot[i] { -1.0 }
                        else if colour_by_density { 1.0 - ((dens[i] - dlo) / (dhi - dlo)).clamp(0.0, 1.0) }
                        else { ((p.y as f32) / 260.0).clamp(0.0, 1.0) };
                    inst.push(Inst { x: p.x as f32, y: p.y as f32, z: p.z as f32, heat });
                }
                queue.write_buffer(&inst_b, 0, bytemuck::cast_slice(&inst));

                // ---- HUD
                let tp = 3.0 * dpr.clamp(1.0, 3.0);
                let mut ui: Vec<UiVertex> = Vec::new();
                let pad = 6.0 * tp;
                let row = 8.0 * tp;
                push_quad(&mut ui, pad, pad, 128.0 * tp, 7.0 * row, [0.03, 0.05, 0.10, 0.66], config.width as f32, config.height as f32);
                let (sw, sh) = (config.width as f32, config.height as f32);
                let white = [0.92, 0.95, 1.0, 1.0];
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp, tp, white, kind.label(), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp + row, tp * 0.8, [0.75, 0.82, 0.95, 0.95], &format!("{} PTS - {:.0} FPS", n, fps), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp + 2.0 * row, tp * 0.8, [0.55, 0.85, 1.0, 0.95], &format!("BUILD {build_ms:.0}MS"), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp + 3.0 * row, tp * 0.8, [1.0, 0.78, 0.35, 0.95], &format!("KNN K{k} ALL {knn_ms:.0}MS"), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp + 4.0 * row, tp * 0.8, [0.55, 0.95, 0.6, 0.95], &format!("PROBE {probe_n} IN {probe_us:.0}US"), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp + 5.0 * row, tp * 0.8, [0.8, 0.8, 0.9, 0.9], &format!("AGREE {agree} - M K C [ ]"), sw, sh);
                queue.write_buffer(&ui_buf, 0, bytemuck::cast_slice(&ui));
                let ui_count = ui.len() as u32;

                let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view_tex = frame_tex.texture.create_view(&Default::default());
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.025, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                        timestamp_writes: None, occlusion_query_set: None,
                    });
                    rp.set_pipeline(&render_pipe);
                    rp.set_bind_group(0, &cam_bg, &[]);
                    rp.set_vertex_buffer(0, inst_b.slice(..));
                    rp.draw(0..6, 0..inst.len() as u32);
                    if ui_count > 0 { rp.set_pipeline(&ui_pipeline); rp.set_vertex_buffer(0, ui_buf.slice(..)); rp.draw(0..ui_count, 0..1); }
                }
                queue.submit(Some(enc.finish()));
                frame_tex.present();

                if let Some(max) = smoke {
                    if frame >= max {
                        // Self-check: the probe cull must equal brute force over the
                        // same sphere, and the cloud must be finite and inside the world
                        // box (an out-of-box point makes Octree3::bulk_load panic and
                        // poisons a tight-box tree's root).
                        let brute = pts.iter().filter(|q| {
                            let (dx, dy, dz) = (q.x - probe.x as f64, q.y - probe.y as f64, q.z - probe.z as f64);
                            dx * dx + dy * dy + dz * dz <= (PROBE_R as f64) * (PROBE_R as f64)
                        }).count();
                        let nonfinite = pts.iter().filter(|q| !(q.x.is_finite() && q.y.is_finite() && q.z.is_finite())).count();
                        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                        for q in &pts {
                            for (a, v) in [q.x, q.y, q.z].into_iter().enumerate() { lo[a] = lo[a].min(v); hi[a] = hi[a].max(v); }
                        }
                        println!("pointcloud_wgpu: {} pts, {} - build {:.0} ms - knn(k={}) over ALL points {:.0} ms ({:.2} us/query) - probe cull {} hits in {:.0} us - {:.0} fps",
                            n, kind.label(), build_ms, k, knn_ms, knn_ms * 1000.0 / n as f32, probe_n, probe_us, fps);
                            let tag = kind.label().split_whitespace().next().unwrap_or("?").to_lowercase();
                            println!("#M {tag}.build {build_ms:.3} ms");
                            println!("#M {tag}.knn_all {knn_ms:.3} ms");
                            println!("#M {tag}.knn_per_query {:.4} us", knn_ms * 1000.0 / n as f32);
                            println!("#M {tag}.fps {fps:.1} fps");
                        println!("  check: probe {} vs brute {} ({}) - nonfinite {} - bounds x[{:.1},{:.1}] y[{:.1},{:.1}] z[{:.1},{:.1}]",
                            probe_n, brute, if probe_n == brute { "MATCH" } else { "MISMATCH" }, nonfinite,
                            lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
                        elwt.exit();
                    }
                }
                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    });
}

const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) heat: f32 };
@vertex
fn vs(@location(0) inst: vec4<f32>, @builtin(vertex_index) vi: u32) -> VO {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let c = corners[vi];
    var clip = cam.vp * vec4<f32>(inst.xyz, 1.0);
    // screen-space point sprite: constant pixel size regardless of distance
    let sz = select(2.6, 4.2, inst.w < -0.5);
    clip = vec4<f32>(clip.xy + c * sz * clip.w / 900.0, clip.z, clip.w);
    var o: VO; o.clip = clip; o.uv = c; o.heat = inst.w; return o;
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    if (dot(v.uv, v.uv) > 1.0) { discard; }
    if (v.heat < -0.5) { return vec4<f32>(1.0, 0.42, 0.15, 1.0); }   // probe selection
    // sparse (cold, blue) -> dense (warm, white): the local-density read-out
    let t = clamp(v.heat, 0.0, 1.0);
    let cold = vec3<f32>(0.18, 0.42, 0.75);
    let warm = vec3<f32>(1.0, 0.92, 0.72);
    return vec4<f32>(mix(cold, warm, t * t), 1.0);
}
"#;

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
