//! stealth_wgpu — guards with **real vision**: a view cone that is an actual frustum
//! cull, and line of sight that is an actual segment-vs-solid test. Sneak to the exit
//! without being seen.
//!
//! Every current demo culls with spheres. This one centres the two query verbs none of
//! them showcase, and the detection loop is *only* kit calls:
//!   1. **Frustum cull** — the guard's view cone is a `Polyhedron3::from_corners`
//!      (6 inward half-spaces); culling the agent index with it answers "who is inside
//!      my cone" in one query, no per-agent angle maths.
//!   2. **Capsule cull** — for each candidate, a `Segment3` from the guard's eye to the
//!      target collects the crates *near* that sight line (a cheap broadphase).
//!   3. **`Polyhedron3::segment_hit`** — the exact segment↔solid test on those few
//!      crates: `Some(t)` with `t < 1` means the crate blocks the line.
//!
//! That is the same broadphase-then-exact shape the GPU visibility bench measures, on
//! the CPU, driving a game.
//!
//! It also answers "does the index even pay here?" honestly: every frame the same cones
//! are ALSO resolved by a linear scan of every agent against the same 6 half-spaces, the
//! two answers are compared (they must agree exactly), and both costs go on the HUD.
//! `[` `]` change the crowd size so you can walk right over the crossover.
//!
//! Controls: `W A S D` move (camera-relative) · drag to orbit · wheel to zoom ·
//! `[` `]` crowd size · `L` draw sight lines · `R` restart · `P` pause.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin stealth_wgpu --release
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
use vectorial_hash::{Aabb, Point3, Polyhedron3, Positioned3, Segment3, Shape3, Tree3};

const WORLD: f32 = 900.0;
const EYE_H: f32 = 9.0;        // guard eye height
const FOV: f32 = 1.15;         // view cone half-angle is FOV/2 (radians)
const SIGHT: f32 = 300.0;      // cone length
const CRATES: usize = 90;
const GUARDS: usize = 9;
const CIVS: usize = 40;
const SPOT_RATE: f32 = 0.55;   // alert gained per second while exposed
const CALM_RATE: f32 = 0.30;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cube { cx: f32, cy: f32, cz: f32, _p: f32, hx: f32, hy: f32, hz: f32, _p2: f32, col: [f32; 4] }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LineV { pos: [f32; 3], _p: f32, col: [f32; 4] }
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
        '-' => ["000", "000", "111", "000", "000"], '!' => ["010", "010", "010", "000", "010"],
        '%' => ["101", "001", "010", "100", "101"], ':' => ["000", "010", "000", "010", "000"],
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

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u32 << 24) as f32 }
    fn r(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.f() }
}

// ---- world -------------------------------------------------------------------
/// A crate: an axis-aligned solid. Indexed by its centre; the exact occlusion test
/// uses the `Polyhedron3` built from its 8 corners (which recovers its 6 faces).
struct Crate { c: Vec3, h: Vec3, solid: Polyhedron3 }
fn make_crate(c: Vec3, h: Vec3) -> Crate {
    let k = |sx: f32, sy: f32, sz: f32| Point3::new((c.x + sx * h.x) as f64, (c.y + sy * h.y) as f64, (c.z + sz * h.z) as f64);
    // near face then far face, each (bottom-left, bottom-right, top-right, top-left)
    let corners = [
        k(-1.0, -1.0, -1.0), k(1.0, -1.0, -1.0), k(1.0, 1.0, -1.0), k(-1.0, 1.0, -1.0),
        k(-1.0, -1.0, 1.0), k(1.0, -1.0, 1.0), k(1.0, 1.0, 1.0), k(-1.0, 1.0, 1.0),
    ];
    Crate { c, h, solid: Polyhedron3::from_corners(corners) }
}

/// An indexed agent (the player, a guard or a civilian) — what a view cone culls.
#[derive(Clone, Copy)]
struct Agent { id: u32, p: Point3 }
impl Positioned3 for Agent { fn position(&self) -> Point3 { self.p } }
/// A crate's centre in the occluder index.
#[derive(Clone, Copy)]
struct OccRef { id: u32, p: Point3 }
impl Positioned3 for OccRef { fn position(&self) -> Point3 { self.p } }

struct Guard { p: Vec3, face: f32, wps: Vec<Vec3>, wp: usize, alert: f32 }

struct World {
    crates: Vec<Crate>,
    occ: Tree3<OccRef>,
    occ_r: f32,          // capsule radius that covers any crate from its centre
    guards: Vec<Guard>,
    civs: Vec<(Vec3, Vec3)>, // position, velocity
    player: Vec3,
    goal: Vec3,
    caught: f32,         // 0..1 detection meter
    escaped: bool,
    seen_now: usize,
    // per-frame query costs (µs)
    cone_us: f32, los_us: f32, brute_us: f32,
    agree: bool, mismatch: (usize, usize), bad_frames: u32,
    // Running sums, because a single frame's reading is not a measurement: if the last
    // frame happens not to step, "the number" is 0. (It did, once, in a batch run.)
    acc_cone: f64, acc_brute: f64, acc_los: f64, acc_seen: f64, acc_frames: u32,
}

impl World {
    fn new(n_civs: usize) -> World {
        let mut r = Rng(0x0057_EA17u64);
        let mut crates = Vec::with_capacity(CRATES);
        for _ in 0..CRATES {
            let (x, z) = (r.r(60.0, WORLD - 60.0), r.r(60.0, WORLD - 60.0));
            let h = Vec3::new(r.r(12.0, 34.0), r.r(10.0, 30.0), r.r(12.0, 34.0));
            crates.push(make_crate(Vec3::new(x, h.y, z), h));
        }
        let occ_r = crates.iter().fold(0.0f32, |m, c| m.max(c.h.length()));
        let world = Aabb::new(-10.0, -10.0, -10.0, (WORLD + 20.0) as f64, 220.0, (WORLD + 20.0) as f64);
        let occ = Tree3::bulk_load(world, 8, crates.iter().enumerate()
            .map(|(i, c)| OccRef { id: i as u32, p: Point3::new(c.c.x as f64, c.c.y as f64, c.c.z as f64) }).collect());
        let guards = (0..GUARDS).map(|_| {
            let wps: Vec<Vec3> = (0..3).map(|_| Vec3::new(r.r(70.0, WORLD - 70.0), EYE_H, r.r(70.0, WORLD - 70.0))).collect();
            Guard { p: wps[0], face: 0.0, wps, wp: 1, alert: 0.0 }
        }).collect();
        let civs = (0..n_civs).map(|_| (
            Vec3::new(r.r(40.0, WORLD - 40.0), 7.0, r.r(40.0, WORLD - 40.0)),
            Vec3::new(r.r(-26.0, 26.0), 0.0, r.r(-26.0, 26.0)),
        )).collect();
        World { crates, occ, occ_r, guards, civs, player: Vec3::new(40.0, 7.0, 40.0), goal: Vec3::new(WORLD - 60.0, 7.0, WORLD - 60.0), caught: 0.0, escaped: false, seen_now: 0, cone_us: 0.0, los_us: 0.0, brute_us: 0.0, agree: true, mismatch: (0, 0), bad_frames: 0,
            acc_cone: 0.0, acc_brute: 0.0, acc_los: 0.0, acc_seen: 0.0, acc_frames: 0 }
    }

    /// The guard's view cone as a frustum: a near quad just in front of the eye and a
    /// far quad at `SIGHT`, handed to `Polyhedron3::from_corners`.
    fn cone(&self, g: &Guard) -> Polyhedron3 {
        let (s, c) = (g.face.sin(), g.face.cos());
        let fwd = Vec3::new(c, 0.0, s);
        let right = Vec3::new(-s, 0.0, c);
        let quad = |dist: f32, half: f32, vh: f32| {
            let ctr = g.p + fwd * dist;
            [
                ctr - right * half - Vec3::Y * vh, ctr + right * half - Vec3::Y * vh,
                ctr + right * half + Vec3::Y * vh, ctr - right * half + Vec3::Y * vh,
            ]
        };
        let n = quad(6.0, 6.0 * (FOV * 0.5).tan() + 3.0, 7.0);
        let f = quad(SIGHT, SIGHT * (FOV * 0.5).tan(), 34.0);
        let pt = |v: Vec3| Point3::new(v.x as f64, v.y as f64, v.z as f64);
        Polyhedron3::from_corners([pt(n[0]), pt(n[1]), pt(n[2]), pt(n[3]), pt(f[0]), pt(f[1]), pt(f[2]), pt(f[3])])
    }

    /// The same 8 corners the cull uses, so what you see IS the query volume.
    fn cone_corners(&self, g: &Guard) -> [Vec3; 8] {
        let (s, c) = (g.face.sin(), g.face.cos());
        let fwd = Vec3::new(c, 0.0, s);
        let right = Vec3::new(-s, 0.0, c);
        let quad = |dist: f32, half: f32, vh: f32| {
            let ctr = g.p + fwd * dist;
            [ctr - right * half - Vec3::Y * vh, ctr + right * half - Vec3::Y * vh,
             ctr + right * half + Vec3::Y * vh, ctr - right * half + Vec3::Y * vh]
        };
        let n = quad(6.0, 6.0 * (FOV * 0.5).tan() + 3.0, 7.0);
        let f = quad(SIGHT, SIGHT * (FOV * 0.5).tan(), 34.0);
        [n[0], n[1], n[2], n[3], f[0], f[1], f[2], f[3]]
    }

    /// Exact line of sight: broadphase the crates NEAR the sight line with a capsule
    /// cull, then run the exact segment↔solid test on those few.
    fn clear_los(&self, a: Vec3, b: Vec3) -> bool {
        let (pa, pb) = (Point3::new(a.x as f64, a.y as f64, a.z as f64), Point3::new(b.x as f64, b.y as f64, b.z as f64));
        for occ in self.occ.cull(&Segment3::new(pa, pb, self.occ_r as f64)) {
            if let Some(t) = self.crates[occ.id as usize].solid.segment_hit(pa, pb) {
                if t < 1.0 { return false; }
            }
        }
        true
    }

    fn step(&mut self, dt: f32, agents: &Tree3<Agent>, all: &[Agent], draw_lines: bool, lines: &mut Vec<LineV>) {
        // guards patrol their loop
        for g in self.guards.iter_mut() {
            let tgt = g.wps[g.wp];
            let d = tgt - g.p;
            let l = d.length();
            if l < 12.0 { g.wp = (g.wp + 1) % g.wps.len(); }
            else {
                let step = d / l * 52.0 * dt;
                g.p += step;
                g.face = d.z.atan2(d.x);
            }
        }
        // civilians wander (they give the cone cull something to reject)
        // Reflect only when actually heading OUT (flipping on the test alone lets a
        // wanderer chatter its way through the wall) and clamp: an agent outside the
        // index's world box is dropped by `bulk_load`, and then the index and a linear
        // scan are answering questions about different sets. See docs/STEALTH.md.
        for (p, v) in self.civs.iter_mut() {
            *p += *v * dt;
            if (p.x < 20.0 && v.x < 0.0) || (p.x > WORLD - 20.0 && v.x > 0.0) { v.x = -v.x; }
            if (p.z < 20.0 && v.z < 0.0) || (p.z > WORLD - 20.0 && v.z > 0.0) { v.z = -v.z; }
            p.x = p.x.clamp(20.0, WORLD - 20.0);
            p.z = p.z.clamp(20.0, WORLD - 20.0);
        }

        // ---- detection: one frustum cull per guard, then exact LoS on the hits
        let mut exposed = false;
        self.seen_now = 0;
        let cones: Vec<Polyhedron3> = self.guards.iter().map(|g| self.cone(g)).collect();
        let t0 = Instant::now();
        let mut candidates: Vec<(usize, Vec<u32>)> = Vec::with_capacity(self.guards.len());
        for (gi, cone) in cones.iter().enumerate() {
            let ids: Vec<u32> = agents.cull(cone).iter().map(|a| a.id).collect();
            candidates.push((gi, ids));
        }
        self.cone_us = (Instant::now() - t0).as_secs_f32() * 1e6;

        // The same question with no index at all: every agent tested against the same 6
        // half-spaces. Same answer by construction - the only difference is the cost, and
        // with a small crowd the linear scan legitimately wins (see the HUD).
        let t_b = Instant::now();
        let mut brute_total = 0usize;
        for cone in &cones {
            for a in all.iter() { if cone.contains_point(a.p) { brute_total += 1; } }
        }
        self.brute_us = (Instant::now() - t_b).as_secs_f32() * 1e6;
        let idx_total = candidates.iter().map(|(_, v)| v.len()).sum::<usize>();
        self.agree = brute_total == idx_total;
        if !self.agree { self.mismatch = (idx_total, brute_total); self.bad_frames += 1; }

        // ---- exact line of sight on the candidates: capsule broadphase, then segment↔solid
        let t1 = Instant::now();
        for (gi, ids) in &candidates {
            let eye = self.guards[*gi].p + Vec3::Y * 4.0;
            for &id in ids {
                let target = if id == 0 { self.player } else { self.civs[(id - 1) as usize].0 };
                let clear = self.clear_los(eye, target + Vec3::Y * 3.0);
                if clear { self.seen_now += 1; }
                if clear && id == 0 { exposed = true; }
                if draw_lines {
                    let col = if clear { [0.25, 1.0, 0.45, 0.55] } else { [1.0, 0.30, 0.25, 0.35] };
                    lines.push(LineV { pos: [eye.x, eye.y, eye.z], _p: 0.0, col });
                    lines.push(LineV { pos: [target.x, target.y + 3.0, target.z], _p: 0.0, col });
                }
            }
        }
        self.los_us = (Instant::now() - t1).as_secs_f32() * 1e6;
        self.acc_cone += self.cone_us as f64;
        self.acc_brute += self.brute_us as f64;
        self.acc_los += self.los_us as f64;
        self.acc_seen += self.seen_now as f64;
        self.acc_frames += 1;

        // alert meter + win/lose
        self.caught = (self.caught + if exposed { SPOT_RATE * dt } else { -CALM_RATE * dt }).clamp(0.0, 1.0);
        for g in self.guards.iter_mut() { g.alert = self.caught; }
        if (self.player - self.goal).length() < 30.0 { self.escaped = true; }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() { pollster::block_on(run()); }

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); wasm_bindgen_futures::spawn_local(run()); }

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash stealth").with_inner_size(winit::dpi::LogicalSize::new(1300, 900)).build(&event_loop).unwrap());
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

    let mut n_civs: usize = std::env::var("STEALTH_CIVS").ok().and_then(|s| s.parse().ok()).unwrap_or(CIVS);
    let mut w = World::new(n_civs);
    let agents_world = Aabb::new(-10.0, -10.0, -10.0, (WORLD + 20.0) as f64, 220.0, (WORLD + 20.0) as f64);

    // ---- pipelines: instanced cubes, world-space lines, screen-space UI
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let cube_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(CUBE_SHADER.into()) });
    let depth_fmt = wgpu::TextureFormat::Depth32Float;
    let cube_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cubes"), layout: Some(&pl),
        vertex: wgpu::VertexState { module: &cube_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Cube>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &cube_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: depth_fmt, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(), multiview: None,
    });
    let line_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()) });
    let line_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lines"), layout: Some(&pl),
        vertex: wgpu::VertexState { module: &line_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<LineV>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &line_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: depth_fmt, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(), multiview: None,
    });
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: depth_fmt, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(), multiview: None,
    });
    let make_depth = |device: &wgpu::Device, w: u32, h: u32| device.create_texture(&wgpu::TextureDescriptor { label: Some("depth"), size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: depth_fmt, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[] }).create_view(&Default::default());
    let mut depth_view = make_depth(&device, config.width, config.height);
    let cube_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cubes"), size: (4096 * std::mem::size_of::<Cube>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let line_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("lines"), size: (40_000 * std::mem::size_of::<LineV>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let ui_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui"), size: (60_000 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let smoke: Option<u64> = std::env::var("STEALTH_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let (mut yaw, mut pitch, mut dist) = (0.9f32, 0.75f32, 1250.0f32);
    let (mut drag, mut last_mouse) = (false, (0.0f64, 0.0f64));
    let mut mv = [false; 4];
    let (mut paused, mut draw_lines, mut fps, mut frame) = (false, true, 0.0f32, 0u64);
    let mut last = Instant::now();
    let (mut cubes, mut lines) = (Vec::<Cube>::new(), Vec::<LineV>::new());

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); depth_view = make_depth(&device, config.width, config.height); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => drag = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - last_mouse.0, position.y - last_mouse.1);
                last_mouse = (position.x, position.y);
                if drag { yaw += dx as f32 * 0.005; pitch = (pitch + dy as f32 * 0.004).clamp(0.15, 1.45); }
            }
            WindowEvent::MouseWheel { delta, .. } => { let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 90.0, MouseScrollDelta::PixelDelta(p) => p.y as f32 }; dist = (dist - d).clamp(300.0, 2600.0); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state, .. }, .. } => {
                let down = state == ElementState::Pressed;
                match c {
                    KeyCode::KeyW => mv[0] = down,
                    KeyCode::KeyS => mv[1] = down,
                    KeyCode::KeyA => mv[2] = down,
                    KeyCode::KeyD => mv[3] = down,
                    KeyCode::KeyL if down => draw_lines = !draw_lines,
                    KeyCode::KeyP if down => paused = !paused,
                    KeyCode::KeyR if down => w = World::new(n_civs),
                    KeyCode::BracketLeft if down => { n_civs = (n_civs / 2).max(10); w = World::new(n_civs); }
                    KeyCode::BracketRight if down => { n_civs = (n_civs * 2).min(40_000); w = World::new(n_civs); }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };
                frame += 1;

                // camera-relative movement, so W is always "away from the camera"
                let (sy, cy2) = (yaw.sin(), yaw.cos());
                let fwd = Vec3::new(-cy2, 0.0, -sy);
                let right = Vec3::new(sy, 0.0, -cy2);
                let mut d = Vec3::ZERO;
                if mv[0] { d += fwd; }
                if mv[1] { d -= fwd; }
                if mv[3] { d += right; }
                if mv[2] { d -= right; }
                if d.length_squared() > 0.0 && !paused && !w.escaped && w.caught < 1.0 {
                    w.player += d.normalize() * 150.0 * dt;
                    w.player.x = w.player.x.clamp(12.0, WORLD - 12.0);
                    w.player.z = w.player.z.clamp(12.0, WORLD - 12.0);
                }

                lines.clear();
                for g in &w.guards {
                    let k = w.cone_corners(g);
                    let a = g.alert;
                    let col = [0.35 + 0.6 * a, 0.85 - 0.5 * a, 0.95 - 0.6 * a, 0.30 + 0.25 * a];
                    let mut seg = |i: usize, j: usize| {
                        lines.push(LineV { pos: [k[i].x, k[i].y, k[i].z], _p: 0.0, col });
                        lines.push(LineV { pos: [k[j].x, k[j].y, k[j].z], _p: 0.0, col });
                    };
                    for e in 0..4 { seg(e, 4 + e); }                       // near corner -> far corner
                    for e in 0..4 { seg(4 + e, 4 + (e + 1) % 4); }         // the far face
                }
                if !paused {
                    // The agent index is rebuilt each frame (a few dozen movers) and is
                    // what every view cone culls.
                    let mut items = Vec::with_capacity(1 + w.civs.len());
                    items.push(Agent { id: 0, p: Point3::new(w.player.x as f64, w.player.y as f64, w.player.z as f64) });
                    for (i, (p, _)) in w.civs.iter().enumerate() { items.push(Agent { id: 1 + i as u32, p: Point3::new(p.x as f64, p.y as f64, p.z as f64) }); }
                    let agents = Tree3::bulk_load(agents_world, 8, items.clone());
                    w.step(dt, &agents, &items, draw_lines, &mut lines);
                }

                // ---- scene
                cubes.clear();
                cubes.push(Cube { cx: WORLD * 0.5, cy: -6.0, cz: WORLD * 0.5, _p: 0.0, hx: WORLD * 0.5 + 20.0, hy: 6.0, hz: WORLD * 0.5 + 20.0, _p2: 0.0, col: [0.10, 0.12, 0.16, 1.0] });
                for c in &w.crates { cubes.push(Cube { cx: c.c.x, cy: c.c.y, cz: c.c.z, _p: 0.0, hx: c.h.x, hy: c.h.y, hz: c.h.z, _p2: 0.0, col: [0.32, 0.30, 0.27, 1.0] }); }
                for (p, _) in &w.civs { cubes.push(Cube { cx: p.x, cy: p.y, cz: p.z, _p: 0.0, hx: 5.0, hy: 7.0, hz: 5.0, _p2: 0.0, col: [0.45, 0.50, 0.58, 1.0] }); }
                for g in &w.guards {
                    let a = g.alert;
                    cubes.push(Cube { cx: g.p.x, cy: g.p.y, cz: g.p.z, _p: 0.0, hx: 6.0, hy: 9.0, hz: 6.0, _p2: 0.0, col: [0.85 * (0.4 + a), 0.35, 0.30, 1.0] });
                    // a stub showing the facing
                    let (s2, c2) = (g.face.sin(), g.face.cos());
                    cubes.push(Cube { cx: g.p.x + c2 * 9.0, cy: g.p.y + 3.0, cz: g.p.z + s2 * 9.0, _p: 0.0, hx: 3.0, hy: 2.0, hz: 3.0, _p2: 0.0, col: [1.0, 0.75, 0.35, 1.0] });
                }
                cubes.push(Cube { cx: w.goal.x, cy: 1.0, cz: w.goal.z, _p: 0.0, hx: 26.0, hy: 1.0, hz: 26.0, _p2: 0.0, col: [0.25, 0.85, 0.45, 0.75] });
                cubes.push(Cube { cx: w.player.x, cy: w.player.y, cz: w.player.z, _p: 0.0, hx: 5.5, hy: 8.0, hz: 5.5, _p2: 0.0, col: [0.35, 0.80, 1.0, 1.0] });
                cubes.truncate(4096);
                queue.write_buffer(&cube_b, 0, bytemuck::cast_slice(&cubes));
                lines.truncate(40_000);
                if !lines.is_empty() { queue.write_buffer(&line_b, 0, bytemuck::cast_slice(&lines)); }

                let target = Vec3::new(WORLD * 0.5, 0.0, WORLD * 0.5);
                let eye = target + Vec3::new(dist * pitch.cos() * yaw.cos(), dist * pitch.sin(), dist * pitch.cos() * yaw.sin());
                let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                let proj = Mat4::perspective_rh(48f32.to_radians(), config.width as f32 / config.height as f32, 1.0, 8000.0);
                queue.write_buffer(&cam_b, 0, bytemuck::bytes_of(&Cam { vp: (proj * view).to_cols_array_2d() }));

                // ---- HUD
                let (sw, sh) = (config.width as f32, config.height as f32);
                let tp = 3.0 * dpr.clamp(1.0, 3.0);
                let mut ui: Vec<UiVertex> = Vec::new();
                let pad = 6.0 * tp;
                push_quad(&mut ui, pad, pad, 150.0 * tp, 41.0 * tp, [0.03, 0.05, 0.10, 0.66], sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp, tp, [0.92, 0.95, 1.0, 1.0], &format!("STEALTH - {:.0} FPS", fps), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 9.0 * tp, tp * 0.8, [0.75, 0.82, 0.95, 0.95], &format!("{} GUARDS - {} CROWD - {} SEEN", w.guards.len(), w.civs.len(), w.seen_now), sw, sh);
                let (fast, col) = if w.cone_us <= w.brute_us { ("INDEX", [0.45, 1.0, 0.6, 1.0]) } else { ("SCAN", [1.0, 0.75, 0.35, 1.0]) };
                push_text(&mut ui, pad + 3.0 * tp, pad + 16.0 * tp, tp * 0.8, col, &format!("CONES: CULL {:.0}US VS SCAN {:.0}US - {} WINS", w.cone_us, w.brute_us, fast), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 23.0 * tp, tp * 0.8, [0.55, 0.85, 1.0, 0.95], &format!("EXACT LOS {:.0}US - AGREE {}", w.los_us, if w.agree { "YES" } else { "NO" }), sw, sh);
                push_text(&mut ui, pad + 3.0 * tp, pad + 30.0 * tp, tp * 0.8, [0.8, 0.8, 0.9, 0.9], "WASD MOVE - BRACKETS CROWD - L LINES", sw, sh);
                // detection meter
                let (bx, by, bw2, bh) = (pad + 3.0 * tp, pad + 36.0 * tp, 120.0 * tp, 3.5 * tp);
                push_quad(&mut ui, bx, by, bw2, bh, [0.16, 0.16, 0.2, 0.9], sw, sh);
                push_quad(&mut ui, bx, by, bw2 * w.caught, bh, [1.0, 0.35 * (1.0 - w.caught), 0.25, 0.95], sw, sh);
                if w.escaped { push_text(&mut ui, sw * 0.5 - 26.0 * tp, sh * 0.42, tp * 2.2, [0.4, 1.0, 0.6, 1.0], "ESCAPED", sw, sh); }
                else if w.caught >= 1.0 { push_text(&mut ui, sw * 0.5 - 26.0 * tp, sh * 0.42, tp * 2.2, [1.0, 0.35, 0.3, 1.0], "SPOTTED", sw, sh); }
                queue.write_buffer(&ui_b, 0, bytemuck::cast_slice(&ui));
                let ui_count = ui.len() as u32;

                let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view_tex = frame_tex.texture.create_view(&Default::default());
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.03, b: 0.05, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                        timestamp_writes: None, occlusion_query_set: None,
                    });
                    rp.set_pipeline(&cube_pipe);
                    rp.set_bind_group(0, &cam_bg, &[]);
                    rp.set_vertex_buffer(0, cube_b.slice(..));
                    rp.draw(0..36, 0..cubes.len() as u32);
                    if !lines.is_empty() {
                        rp.set_pipeline(&line_pipe);
                        rp.set_vertex_buffer(0, line_b.slice(..));
                        rp.draw(0..lines.len() as u32, 0..1);
                    }
                    if ui_count > 0 { rp.set_pipeline(&ui_pipe); rp.set_vertex_buffer(0, ui_b.slice(..)); rp.draw(0..ui_count, 0..1); }
                }
                queue.submit(Some(enc.finish()));
                frame_tex.present();

                if let Some(max) = smoke {
                    if frame >= max {
                        // MEANS over every stepped frame, not the last frame's values.
                        let n = w.acc_frames.max(1) as f64;
                        let (cone, brute, los, seen) = (w.acc_cone / n, w.acc_brute / n, w.acc_los / n, w.acc_seen / n);
                        println!("stealth_wgpu: {} guards / {} crowd / {} crates over {} stepped frames - cones: index cull {:.1} us vs linear scan {:.1} us (agree every frame: {}) - exact LoS {:.1} us - seen {:.0} - {:.0} fps  [means]",
                            w.guards.len(), w.civs.len(), w.crates.len(), w.acc_frames, cone, brute, los, w.bad_frames == 0, seen, fps);
                        if w.bad_frames > 0 { println!("  MISMATCH on {} frames - last index {} vs scan {}", w.bad_frames, w.mismatch.0, w.mismatch.1); }
                        // machine-readable for `bench-runner`; keyed by crowd size so a
                        // sweep produces the index-vs-scan crossover table directly.
                        let tag = w.civs.len();
                        println!("#M crowd{tag}.index_cull {cone:.2} us");
                        println!("#M crowd{tag}.linear_scan {brute:.2} us");
                        println!("#M crowd{tag}.scan_over_index {:.3} x", brute / cone.max(1e-9));
                        println!("#M crowd{tag}.exact_los {los:.2} us");
                        println!("#M crowd{tag}.stepped_frames {} n", w.acc_frames);
                        println!("#M crowd{tag}.agree {} bool", if w.bad_frames == 0 { 1 } else { 0 });
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

const CUBE_SHADER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) col: vec4<f32>, @location(1) shade: f32 };
// unit cube: 6 faces x 2 triangles, corners in -1..1 with a per-face shade
@vertex
fn vs(@location(0) c: vec4<f32>, @location(1) h: vec4<f32>, @location(2) col: vec4<f32>, @builtin(vertex_index) vi: u32) -> VO {
    var quad = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let face = vi / 6u;
    let q = quad[vi % 6u];
    var p = vec3<f32>(0.0);
    var shade = 1.0;
    if (face == 0u)      { p = vec3<f32>(q.x, q.y,  1.0); shade = 0.82; }
    else if (face == 1u) { p = vec3<f32>(q.x, q.y, -1.0); shade = 0.62; }
    else if (face == 2u) { p = vec3<f32>( 1.0, q.y, q.x); shade = 0.74; }
    else if (face == 3u) { p = vec3<f32>(-1.0, q.y, q.x); shade = 0.54; }
    else if (face == 4u) { p = vec3<f32>(q.x,  1.0, q.y); shade = 1.0; }
    else                 { p = vec3<f32>(q.x, -1.0, q.y); shade = 0.35; }
    let world = c.xyz + p * h.xyz;
    var o: VO;
    o.clip = cam.vp * vec4<f32>(world, 1.0);
    o.col = col;
    o.shade = shade;
    return o;
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> { return vec4<f32>(v.col.rgb * v.shade, v.col.a); }
"#;

const LINE_SHADER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) col: vec4<f32> };
@vertex fn vs(@location(0) p: vec4<f32>, @location(1) col: vec4<f32>) -> VO {
    var o: VO; o.clip = cam.vp * vec4<f32>(p.xyz, 1.0); o.col = col; return o;
}
@fragment fn fs(v: VO) -> @location(0) vec4<f32> { return v.col; }
"#;

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
