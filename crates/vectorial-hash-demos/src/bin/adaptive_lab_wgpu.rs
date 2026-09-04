//! `adaptive_lab_wgpu` — **change the workload with your hands and watch the index change its
//! mind.**
//!
//! `AdaptiveIndex` exists for a workload whose character changes, and until now nothing in this
//! repo let you see it happen. `fluid_wgpu` uses it, but an SPH tank looks the same in frame
//! 10 000 as in frame 100. The horde's workload *does* change, and its adaptive arm still only
//! ever walks `Brute → KeepTree`. The one varying-load workload was `examples/adaptive_vs_pinned`
//! — headless, synthetic, and it reports a **total**, which cannot tell a policy that flapped
//! from one that lagged from one that simply chose wrong. On a screen those look nothing alike.
//!
//! Four sliders, one per boundary the policy reasons about:
//!
//! | slider | crosses | question it forces |
//! | --- | --- | --- |
//! | `[` `]` population | `brute_max` | is an index worth having at all? |
//! | `,` `.` queries/item | `rebuild_query_ratio` | grid, or keep-tree? |
//! | `-` `=` radius | `grid_min_hits` | do the queries FIND anything? |
//! | `;` `'` churn | — | how much does maintenance cost? |
//! | `F` freeze | `static_ticks` | has everything stopped moving? |
//!
//! The **timeline strip** along the bottom is the point of the whole demo: one column per recent
//! step, coloured by the backend that was live. Flapping is a barcode, lag is a band that arrives
//! late, and a wrong choice is a solid colour while the bake-off says otherwise.
//!
//! `C` races the live policy against every backend **pinned** (`migrate_to` + `freeze`), so the
//! HUD can say not just what was chosen but whether it was right.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin adaptive_lab_wgpu --release
//! LAB_HEADLESS=1 cargo run -p vectorial-hash-demos --bin adaptive_lab_wgpu --release
//! ```
//!
//! Controls: `[` `]` population · `,` `.` query load · `-` `=` radius · `;` `'` churn ·
//! `F` freeze · `P` pause · `C` bake-off · `R` reset · drag a slider with the mouse.
#![cfg(any(not(target_arch = "wasm32"), feature = "web-wgpu"))]
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use winit::{event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};
use vectorial_hash::Backend;
use vectorial_hash_demos::adaptive_lab::{Knobs, Lab, HISTORY, H, MAX_N, W};
use vectorial_hash_demos::ui2d::{push_frame, push_quad, push_text, text_width, UiVertex};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }

/// One agent: position, radius, and a 0/1 flag for "a query returned me last step".
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Inst { x: f32, y: f32, r: f32, hit: f32 }

/// The colour a backend is drawn in, everywhere — strip, HUD label and bake-off row.
///
/// One table, because a legend that disagrees with the strip is worse than no legend.
fn backend_color(b: Backend) -> [f32; 4] {
    match b {
        Backend::Brute => [0.62, 0.64, 0.70, 1.0],    // grey: no structure at all
        Backend::KeepTree => [0.35, 0.72, 1.00, 1.0], // blue
        Backend::Grid => [1.00, 0.68, 0.25, 1.0],     // amber
        Backend::Static => [0.45, 0.92, 0.55, 1.0],   // green: build once, then still
    }
}
fn backend_name(b: Backend) -> &'static str {
    match b { Backend::Brute => "BRUTE SCAN", Backend::KeepTree => "KEEP-TREE", Backend::Grid => "GRID", Backend::Static => "BUILD-ONCE" }
}

/// The knobs a slider can drive, and how each maps to and from a 0..1 track position.
#[derive(Clone, Copy, PartialEq)]
enum Knob { Population, Queries, Radius, Churn }

impl Knob {
    const ALL: [Knob; 4] = [Knob::Population, Knob::Queries, Knob::Radius, Knob::Churn];
    fn label(self) -> &'static str {
        match self { Knob::Population => "POPULATION", Knob::Queries => "QUERIES/ITEM", Knob::Radius => "RADIUS", Knob::Churn => "CHURN" }
    }
    /// Population is logarithmic: the interesting boundary (`brute_max`, 64 by default) sits far
    /// down a linear 8..20 000 track, where one pixel would be worth 80 agents and the whole
    /// brute-scan regime would be unreachable with a mouse.
    fn get(self, k: &Knobs) -> f32 {
        match self {
            Knob::Population => ((k.population.max(8) as f32).ln() - 8f32.ln()) / ((MAX_N as f32).ln() - 8f32.ln()),
            Knob::Queries => (k.queries_per_item as f32 / 2.0).clamp(0.0, 1.0),
            Knob::Radius => ((k.radius as f32 - 2.0) / 118.0).clamp(0.0, 1.0),
            Knob::Churn => k.churn as f32,
        }
    }
    fn set(self, k: &mut Knobs, t: f32) {
        let t = t.clamp(0.0, 1.0);
        match self {
            Knob::Population => k.population = ((8f32.ln() + t * ((MAX_N as f32).ln() - 8f32.ln())).exp()).round() as usize,
            Knob::Queries => k.queries_per_item = (t * 2.0) as f64,
            Knob::Radius => k.radius = (2.0 + t * 118.0) as f64,
            Knob::Churn => k.churn = t as f64,
        }
    }
    fn nudge(self, k: &mut Knobs, up: bool) {
        let step = if up { 0.06 } else { -0.06 };
        let t = self.get(k) + step;
        self.set(k, t);
    }
    fn text(self, k: &Knobs) -> String {
        match self {
            Knob::Population => format!("{}", k.population),
            Knob::Queries => format!("{:.2}", k.queries_per_item),
            Knob::Radius => format!("{:.0}", k.radius),
            Knob::Churn => format!("{:.0}%", k.churn * 100.0),
        }
    }
}

// ---- HUD layout, in unscaled pixels; everything is multiplied by `tp` -------
const PANEL_W: f32 = 128.0;
const SLIDER_W: f32 = 110.0;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // A scripted run with no window and no GPU: the four acts, printed as a text timeline.
    // This is how the demo is checked on a machine with no display — and it drives the same
    // `Lab::step` the renderer does, so a green run says something about what is on screen.
    if std::env::var("LAB_HEADLESS").is_ok() { headless(); return; }
    pollster::block_on(run());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); wasm_bindgen_futures::spawn_local(run()); }

/// The five acts, run headless, reported as the strip would draw them.
#[cfg(not(target_arch = "wasm32"))]
fn headless() {
    let mut lab = Lab::new(7);
    // Each act is (label, knobs, steps). They are the same regimes `adaptive_vs_pinned` uses,
    // because the point is to be able to compare the two.
    let acts: [(&str, Knobs, usize); 5] = [
        ("quiet, tiny", Knobs { population: 40, queries_per_item: 0.4, radius: 26.0, churn: 0.5, frozen: false, paused: false }, 60),
        ("grown, churning, few queries", Knobs { population: 4000, queries_per_item: 0.05, radius: 26.0, churn: 1.0, frozen: false, paused: false }, 90),
        ("query storm, wide", Knobs { population: 4000, queries_per_item: 1.0, radius: 70.0, churn: 0.3, frozen: false, paused: false }, 90),
        ("query storm, NARROW", Knobs { population: 4000, queries_per_item: 1.0, radius: 4.0, churn: 0.3, frozen: false, paused: false }, 90),
        // 240 steps, not 90, and the difference is a lesson rather than a tuning: at 90 this act
        // ended on GRID because the default `cooldown` is 120 ticks, so the policy WANTED the
        // build-once backend and was not allowed to move yet. That is the lag this demo exists to
        // make visible — on screen it is a band of amber that turns green late, and in the
        // near-miss counter it is a number climbing while nothing changes.
        ("frozen", Knobs { population: 4000, queries_per_item: 0.4, radius: 40.0, churn: 0.0, frozen: true, paused: false }, 240),
    ];
    println!("adaptive_lab — four acts, and what the policy did in each\n");
    for (label, knobs, steps) in acts {
        lab.knobs = knobs;
        let before = lab.ix.switch_count();
        for _ in 0..steps { lab.step(1.0 / 60.0); }
        let (n, q, m) = lab.ix.observed();
        println!("{label:>30} -> {:<11} ({} migrations here) | n {n}, q/item {q:.3}, mv/item {m:.3}",
                 backend_name(lab.ix.backend()), lab.ix.switch_count() - before);
        println!("{:>30}    mean hits/query {:.1}, policy predicted {:.1} | maintain {:.0} us, query {:.0} us",
                 "", lab.mean_hits(), lab.stats.predicted_hits, lab.stats.maintain_us, lab.stats.query_us);
    }
    // The strip, as text: one character per step, so the shape is visible without a window.
    println!("\ntimeline (oldest first, one char per step — the strip the demo draws):");
    let line: String = lab.history.iter().map(|b| match b {
        Backend::Brute => '.', Backend::KeepTree => 'T', Backend::Grid => 'G', Backend::Static => 'S',
    }).collect();
    for chunk in line.as_bytes().chunks(100) { println!("  {}", std::str::from_utf8(chunk).unwrap()); }
    println!("  . brute   T keep-tree   G grid   S build-once");

    let st = lab.ix.stats();
    println!("\n{} migrations over {} steps, {} near-misses", lab.ix.switch_count(), lab.steps, st.near_misses);
    // #161: how long the policy spends KNOWING BETTER. A count of steps, so it means the same on
    // every machine -- which is why it is measured here rather than in milliseconds on a laptop.
    println!("lag: {} of {} steps ({:.1}%) held on a backend the policy had already rejected",
             lab.lag.wanting, lab.steps, lab.lag.wanting_fraction(lab.steps) * 100.0);
    println!("     {:.0} steps of wanting before a typical migration, worst {}",
             lab.lag.mean_lag(), lab.lag.max_lag());
    if let Some((a, b, n)) = st.hottest_pair() { println!("hottest pair: {a:?} -> {b:?} x{n}"); }
    println!("\nbake-off on the final state (us/step, min of two counterbalanced passes):");
    let rows = lab.bakeoff(40);
    for (label, us) in &rows { println!("  {label:>10} {us:>10.1}"); }
    // Say what it means, and say it carefully. The arms here are a few percent apart on a
    // machine whose noise is episodic (MEASURING.md § 8e), so the ORDER is worth more than the
    // ratio and a 5 % gap is not a finding.
    if let Some((bl, bus)) = rows.iter().skip(1).min_by(|a, b| a.1.total_cmp(&b.1)).copied() {
        let r = bus / rows[0].1.max(1e-9);
        println!("  -> best fixed is {bl} at {bus:.0} us; the policy runs at {:.0}, i.e. {r:.2}x it.", rows[0].1);
        if r < 0.95 {
            println!("     It is BEHIND the choice you would have made knowing the ending — which is");
            println!("     the honest case for this layer: insurance, not optimisation.");
        } else if r > 1.05 {
            println!("     It BEAT every fixed choice, so no single structure suited the whole run.");
        } else {
            println!("     That is parity: it found the right answer by itself, which is the goal.");
        }
    }
    println!("\nA total cannot tell flapping from lag from a plain wrong choice. The strip can,");
    println!("and the bake-off says whether the choice was right. That is the whole demo.");
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("vectorial-hash — adaptive index lab")
        .with_inner_size(winit::dpi::LogicalSize::new(1400, 900)).build(&event_loop).unwrap());
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

    let mut lab = Lab::new(0xA11CE);
    // Start knobs from the environment, so a scripted shot or a bug report can open on the
    // exact state that matters instead of on the defaults plus a paragraph of instructions.
    let envf = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
    if let Some(v) = envf("LAB_N") { lab.knobs.population = v as usize; }
    if let Some(v) = envf("LAB_Q") { lab.knobs.queries_per_item = v; }
    if let Some(v) = envf("LAB_R") { lab.knobs.radius = v; }
    if let Some(v) = envf("LAB_CHURN") { lab.knobs.churn = v; }
    // `A`, or $LAB_AUTO=1: walk the five acts on a timer. The strip only tells its story once
    // the workload has actually changed character, and asking a viewer to drag four sliders in
    // the right order first is asking them to already know the answer.
    let mut auto = std::env::var("LAB_AUTO").is_ok();
    let mut auto_step = 0usize;

    let inst_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("inst"), size: (MAX_N * std::mem::size_of::<Inst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("agents"), layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Inst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });
    let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui"), size: (400_000 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    #[cfg(not(target_arch = "wasm32"))]
    let mut shot = vectorial_hash_demos::shot::Shot::from_env("LAB");
    let smoke: Option<u64> = std::env::var("LAB_MAX_FRAMES").ok().and_then(|s| s.parse().ok());

    let (mut frame, mut fps) = (0u64, 0.0f32);
    let mut last = Instant::now();
    let mut bake: Option<Vec<(&'static str, f64)>> = None;
    let mut dragging: Option<Knob> = None;
    let mut cursor = (0.0f32, 0.0f32);
    let mut inst: Vec<Inst> = Vec::with_capacity(MAX_N);
    // Smoothed, because raw per-frame microseconds on a 4000-agent scene are unreadable jitter.
    let (mut maint_us, mut query_us) = (0.0f64, 0.0f64);

    let _ = event_loop.run(move |event, elwt| {
        let tp = 3.0 * dpr.clamp(1.0, 3.0);
        // Where slider `i`'s track sits, in physical pixels. One function, used both to draw and
        // to hit-test — two copies of this is how a slider ends up looking right and grabbing
        // somewhere else.
        let track = |i: usize| -> (f32, f32, f32) {
            let x = 8.0 * tp;
            let y = (44.0 + i as f32 * 22.0) * tp;
            (x, y, SLIDER_W * tp)
        };
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x as f32, position.y as f32);
                    if let Some(k) = dragging {
                        let i = Knob::ALL.iter().position(|&x| x == k).unwrap();
                        let (x, _, w) = track(i);
                        k.set(&mut lab.knobs, (cursor.0 - x) / w);
                    }
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    if state == ElementState::Pressed {
                        dragging = None;
                        for (i, k) in Knob::ALL.iter().enumerate() {
                            let (x, y, w) = track(i);
                            if cursor.0 >= x - 4.0 * tp && cursor.0 <= x + w + 4.0 * tp && cursor.1 >= y - 6.0 * tp && cursor.1 <= y + 10.0 * tp {
                                dragging = Some(*k);
                                k.set(&mut lab.knobs, (cursor.0 - x) / w);
                            }
                        }
                    } else { dragging = None; }
                }
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                    KeyCode::BracketRight => Knob::Population.nudge(&mut lab.knobs, true),
                    KeyCode::BracketLeft => Knob::Population.nudge(&mut lab.knobs, false),
                    KeyCode::Period => Knob::Queries.nudge(&mut lab.knobs, true),
                    KeyCode::Comma => Knob::Queries.nudge(&mut lab.knobs, false),
                    KeyCode::Equal => Knob::Radius.nudge(&mut lab.knobs, true),
                    KeyCode::Minus => Knob::Radius.nudge(&mut lab.knobs, false),
                    KeyCode::Quote => Knob::Churn.nudge(&mut lab.knobs, true),
                    KeyCode::Semicolon => Knob::Churn.nudge(&mut lab.knobs, false),
                    KeyCode::KeyA => { auto = !auto; auto_step = 0; }
                    KeyCode::KeyF => lab.knobs.frozen = !lab.knobs.frozen,
                    KeyCode::KeyP => lab.knobs.paused = !lab.knobs.paused,
                    KeyCode::KeyR => { let k = lab.knobs; lab = Lab::new(0xA11CE); lab.knobs = k; bake = None; }
                    // The honest half of the demo: the HUD says what the policy CHOSE, this says
                    // whether that was right. It costs a visible stutter, which is the correct
                    // price for an answer nobody can get by watching.
                    KeyCode::KeyC => {
                        let rows = lab.bakeoff(40);
                        println!("\nbake-off | {} agents | q/item {:.2} | radius {:.0} | churn {:.0}% | live: {}",
                                 lab.agents.len(), lab.knobs.queries_per_item, lab.knobs.radius,
                                 lab.knobs.churn * 100.0, backend_name(lab.ix.backend()));
                        for (l, us) in &rows { println!("  {l:>10} {us:>10.1} us/step"); }
                        let fixed = rows.iter().skip(1).min_by(|a, b| a.1.total_cmp(&b.1)).copied();
                        if let Some((fl, fus)) = fixed {
                            println!("  -> best fixed {fl} at {fus:.1}; adaptive {:.2}x it", fus / rows[0].1.max(1e-9));
                        }
                        bake = Some(rows);
                    }
                    _ => {}
                },
                WindowEvent::RedrawRequested => {
                    let fdt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                    fps = if fps == 0.0 { 1.0 / fdt } else { fps * 0.9 + 0.1 / fdt };
                    frame += 1;

                    if auto {
                        // One act per 110 frames — comfortably past the 120-tick cooldown only
                        // in the last act, on purpose: the earlier ones show the policy arriving
                        // late, which is the behaviour a total would hide.
                        let acts: [Knobs; 5] = [
                            Knobs { population: 40, queries_per_item: 0.4, radius: 26.0, churn: 0.5, frozen: false, paused: false },
                            Knobs { population: 4000, queries_per_item: 0.05, radius: 26.0, churn: 1.0, frozen: false, paused: false },
                            Knobs { population: 4000, queries_per_item: 1.0, radius: 70.0, churn: 0.3, frozen: false, paused: false },
                            Knobs { population: 4000, queries_per_item: 1.0, radius: 4.0, churn: 0.3, frozen: false, paused: false },
                            Knobs { population: 4000, queries_per_item: 0.4, radius: 40.0, churn: 0.0, frozen: true, paused: false },
                        ];
                        let i = (auto_step / 110).min(acts.len() - 1);
                        let paused = lab.knobs.paused;
                        lab.knobs = acts[i];
                        lab.knobs.paused = paused;
                        auto_step += 1;
                    }
                    lab.step(1.0 / 60.0);
                    maint_us = maint_us * 0.85 + lab.stats.maintain_us * 0.15;
                    query_us = query_us * 0.85 + lab.stats.query_us * 0.15;

                    let (cw, ch) = (config.width as f32, config.height as f32);
                    // Fit the world, preserving aspect, and leave room at the bottom for the strip.
                    let aspect = cw / ch.max(1.0);
                    let (vw, vh) = if aspect > (W / H) as f32 { ((H as f32) * aspect, H as f32) } else { (W as f32, (W as f32) / aspect) };
                    let proj = Mat4::orthographic_rh(
                        (W as f32) * 0.5 - vw * 0.5, (W as f32) * 0.5 + vw * 0.5,
                        (H as f32) * 0.5 - vh * 0.5, (H as f32) * 0.5 + vh * 0.5, -1.0, 1.0);
                    queue.write_buffer(&cam_b, 0, bytemuck::bytes_of(&Cam { vp: proj.to_cols_array_2d() }));

                    // Radius scales down as the crowd grows, or 20 000 agents is a solid block.
                    let r = (6.0 - (lab.agents.len() as f32).log10()).max(1.6);
                    inst.clear();
                    for a in &lab.agents {
                        inst.push(Inst { x: a.p.x as f32, y: a.p.y as f32, r, hit: if a.hit { 1.0 } else { 0.0 } });
                    }
                    queue.write_buffer(&inst_b, 0, bytemuck::cast_slice(&inst));

                    let mut ui: Vec<UiVertex> = Vec::new();
                    let live = lab.ix.backend();
                    let lc = backend_color(live);
                    let pad = 6.0 * tp;

                    // ---- left panel: the knobs, and what the policy is doing about them
                    let panel_h = (44.0 + 4.0 * 22.0 + 76.0) * tp;
                    push_quad(&mut ui, pad, pad, PANEL_W * tp, panel_h, [0.03, 0.05, 0.10, 0.94], cw, ch);
                    push_text(&mut ui, pad + 4.0 * tp, pad + 3.0 * tp, tp * 1.1, lc, backend_name(live), cw, ch);
                    push_text(&mut ui, pad + 4.0 * tp, pad + 12.0 * tp, tp * 0.8, [0.70, 0.78, 0.92, 0.95],
                              &format!("{} MIG  {} NEAR  {:.0}FPS", lab.ix.switch_count(), lab.ix.stats().near_misses, fps), cw, ch);
                    push_text(&mut ui, pad + 4.0 * tp, pad + 20.0 * tp, tp * 0.8, [0.70, 0.78, 0.92, 0.95],
                              &format!("{}{}", if lab.knobs.frozen { "FROZEN " } else { "" }, if lab.knobs.paused { "PAUSED" } else { "" }), cw, ch);

                    for (i, k) in Knob::ALL.iter().enumerate() {
                        let (x, y, w) = track(i);
                        push_text(&mut ui, x, y - 7.0 * tp, tp * 0.8, [0.80, 0.86, 0.98, 0.95],
                                  &format!("{} {}", k.label(), k.text(&lab.knobs)), cw, ch);
                        push_quad(&mut ui, x, y, w, 4.0 * tp, [0.14, 0.16, 0.24, 0.95], cw, ch);
                        push_quad(&mut ui, x, y, w * k.get(&lab.knobs), 4.0 * tp, [0.45, 0.60, 0.85, 0.95], cw, ch);
                        let hx = x + w * k.get(&lab.knobs) - 2.0 * tp;
                        push_quad(&mut ui, hx, y - 3.0 * tp, 4.0 * tp, 10.0 * tp, [0.90, 0.94, 1.0, 0.98], cw, ch);
                    }

                    // ---- what the policy READS, next to what is actually true. `expected_hits`
                    // being wrong is how a rule with a correct threshold still decides wrongly,
                    // so the two numbers belong side by side rather than in separate docs.
                    let (n, q, m) = lab.ix.observed();
                    let sy = (44.0 + 4.0 * 22.0 + 6.0) * tp;
                    let rows = [
                        format!("N {n}  Q/I {q:.2}  MV/I {m:.2}"),
                        format!("HITS {:.1} PREDICT {:.1}", lab.mean_hits(), lab.stats.predicted_hits),
                        format!("MAINTAIN {:.0}US", maint_us),
                        format!("QUERY {:.0}US", query_us),
                    ];
                    for (i, t) in rows.iter().enumerate() {
                        push_text(&mut ui, pad + 4.0 * tp, sy + i as f32 * 9.0 * tp, tp * 0.8, [0.78, 0.84, 0.95, 0.95], t, cw, ch);
                    }
                    push_text(&mut ui, pad + 4.0 * tp, sy + 4.0 * 9.0 * tp, tp * 0.8, [0.55, 0.60, 0.72, 0.9],
                              &format!("F FREEZE P PAUSE C RACE R RESET A AUTO{}", if auto { " *" } else { "" }), cw, ch);

                    // ---- the timeline strip: one column per step, coloured by backend
                    let strip_h = 16.0 * tp;
                    let strip_y = ch - strip_h - pad;
                    push_quad(&mut ui, pad, strip_y, cw - 2.0 * pad, strip_h, [0.05, 0.06, 0.10, 0.85], cw, ch);
                    let colw = (cw - 2.0 * pad) / HISTORY as f32;
                    for (i, b) in lab.history.iter().enumerate() {
                        push_quad(&mut ui, pad + i as f32 * colw, strip_y, colw.max(1.0), strip_h, backend_color(*b), cw, ch);
                    }
                    push_frame(&mut ui, pad, strip_y, cw - 2.0 * pad, strip_h, tp * 0.5, [0.25, 0.30, 0.40, 0.9], cw, ch);
                    // Legend, right-aligned above the strip, in the same colours as the strip.
                    // On its own plate: the first screenshot had it disappearing into a field of
                    // 4 000 dots, and a legend you cannot read is a legend that is not there.
                    let leg_w: f32 = [Backend::Static, Backend::Grid, Backend::KeepTree, Backend::Brute]
                        .iter().map(|b| text_width(backend_name(*b), tp * 0.8) + 10.0 * tp).sum();
                    push_quad(&mut ui, cw - pad - leg_w - 3.0 * tp, strip_y - 11.0 * tp,
                              leg_w + 6.0 * tp, 11.0 * tp, [0.04, 0.05, 0.09, 0.88], cw, ch);
                    let mut lx = cw - pad;
                    for b in [Backend::Static, Backend::Grid, Backend::KeepTree, Backend::Brute] {
                        let t = backend_name(b);
                        lx -= text_width(t, tp * 0.8) + 10.0 * tp;
                        push_text(&mut ui, lx, strip_y - 9.0 * tp, tp * 0.8, backend_color(b), t, cw, ch);
                    }

                    // ---- the bake-off verdict, if `C` was ever pressed
                    if let Some(rows) = &bake {
                        let by = strip_y - 9.0 * tp - (rows.len() as f32 + 1.0) * 9.0 * tp;
                        let best = rows.iter().skip(1).min_by(|a, b| a.1.total_cmp(&b.1)).copied();
                        push_text(&mut ui, pad, by, tp * 0.8, [1.0, 0.95, 0.55, 0.95], "BAKE-OFF US/STEP", cw, ch);
                        for (i, (l, us)) in rows.iter().enumerate() {
                            let win = best.map(|(bl, _)| bl == *l).unwrap_or(false);
                            let col = if i == 0 { [1.0, 0.95, 0.55, 0.95] } else if win { [0.55, 0.95, 0.6, 0.95] } else { [0.70, 0.75, 0.85, 0.9] };
                            push_text(&mut ui, pad, by + (i as f32 + 1.0) * 9.0 * tp, tp * 0.8, col, &format!("{l} {us:.0}"), cw, ch);
                        }
                    }

                    queue.write_buffer(&ui_buf, 0, bytemuck::cast_slice(&ui));
                    let ui_count = ui.len() as u32;

                    #[cfg(not(target_arch = "wasm32"))]
                    let target = vectorial_hash_demos::shot::Target::begin(&device, &config, Some(&surface), &mut shot);
                    #[cfg(target_arch = "wasm32")]
                    let frame_tex = surface.get_current_texture().unwrap();
                    #[cfg(target_arch = "wasm32")]
                    let view_tex = frame_tex.texture.create_view(&Default::default());

                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        let view_tex = target.view().expect("a render target");
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("main"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.025, b: 0.05, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                        });
                        rp.set_pipeline(&render_pipe);
                        rp.set_bind_group(0, &cam_bg, &[]);
                        rp.set_vertex_buffer(0, inst_b.slice(..));
                        rp.draw(0..6, 0..inst.len() as u32);
                        if ui_count > 0 { rp.set_pipeline(&ui_pipeline); rp.set_vertex_buffer(0, ui_buf.slice(..)); rp.draw(0..ui_count, 0..1); }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    { target.finish(&device, &queue, enc); }
                    #[cfg(target_arch = "wasm32")]
                    { queue.submit(Some(enc.finish())); frame_tex.present(); }

                    if let Some(max) = smoke {
                        if frame >= max {
                            // No eyes on this window, so the smoke run reports the health of the
                            // thing the demo is about: did the policy actually move, and does the
                            // index still agree with brute force?
                            let c = vectorial_hash::Point::new(W * 0.5, H * 0.5);
                            let (want, got) = (lab.brute(c, 60.0), lab.indexed(c, 60.0));
                            let agree = want == got;
                            println!("adaptive_lab end-to-end: {:.0} fps, {} agents, holding {} after {} migrations ({} near-misses)",
                                     fps, lab.agents.len(), backend_name(lab.ix.backend()), lab.ix.switch_count(), lab.ix.stats().near_misses);
                            println!("  index agrees with brute force: {agree} ({} vs {} hits)", got.len(), want.len());
                            if !agree { println!("  !! the demo is drawing answers the index got WRONG"); }
                            elwt.exit();
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    });
}

const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) hit: f32 };
@vertex
fn vs(@location(0) inst: vec4<f32>, @builtin(vertex_index) vi: u32) -> VO {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let c = corners[vi];
    var o: VO;
    o.clip = cam.vp * vec4<f32>(inst.xy + c * inst.z, 0.0, 1.0);
    o.uv = c;
    o.hit = inst.w;
    return o;
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    let d = dot(v.uv, v.uv);
    if (d > 1.0) { discard; }
    // Returned by a query this step = hot. Everything else is a dim dot, so the query load is
    // legible as brightness: turn the slider up and the field lights.
    let cold = vec3<f32>(0.22, 0.28, 0.42);
    let hot = vec3<f32>(1.0, 0.80, 0.35);
    return vec4<f32>(mix(cold, hot, v.hit), smoothstep(1.0, 0.3, d));
}
"#;

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
