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
use vectorial_hash_demos::adaptive_lab::{act_at, Knobs, Lab, ACTS, HISTORY, H, MAX_N, W};
use vectorial_hash_demos::ui2d::{push_frame, push_quad, push_text, text_width, UiVertex};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }

/// One agent: position, radius, and how hard the queries hit it last step.
///
/// `heat` is a normalised COUNT, not a flag. As a flag the field went uniformly yellow the moment
/// the query load reached one cull per item — every agent found by something, nothing to see.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Inst { x: f32, y: f32, r: f32, heat: f32 }

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
    // $LAB_TRACE=<path> writes one CSV row per step. The strip shows a shape; this can be
    // plotted, diffed and attached to a bug report -- and it is what makes the lag arithmetic
    // rather than eyeballing. Buffered and written once, never per step (MEASURING 8g).
    let trace_path = std::env::var("LAB_TRACE").ok();
    if trace_path.is_some() { lab.enable_trace(); }
    // The five acts live in `adaptive_lab::ACTS`, shared with the renderer's autopilot.
    println!("adaptive_lab — five acts, and what the policy did in each
");
    // #167: price the lag PER ACT. `C` answers "what would each backend cost now"; the lag is
    // spread over five regimes, and 20 steps held during a query storm is not the same money as
    // 90 steps held while frozen. The trace gives held-vs-wanted per step EXACTLY; the bake-off
    // gives what each backend costs in that regime, timed. An exact count times a timed gap.
    lab.enable_trace();
    let mut act_rows: Vec<(&str, usize, f64)> = Vec::new();
    let mut act_choice: Vec<(&str, usize, f64)> = Vec::new();
    let mut seen_rows = 0usize;
    for (label, knobs, steps) in ACTS {
        lab.knobs = knobs;
        let before = lab.ix.switch_count();
        for _ in 0..steps { lab.step(1.0 / 60.0); }
        let (n, q, m) = lab.ix.observed();
        println!("{label:>30} -> {:<11} ({} migrations here) | n {n}, q/item {q:.3}, mv/item {m:.3}",
                 backend_name(lab.ix.backend()), lab.ix.switch_count() - before);
        println!("{:>30}    mean hits/query {:.1}, policy predicted {:.1} | maintain {:.0} us, query {:.0} us",
                 "", lab.mean_hits(), lab.stats.predicted_hits, lab.stats.maintain_us, lab.stats.query_us);

        // Everything this act contributed to the trace, and what each backend costs HERE.
        let act_start = seen_rows;
        seen_rows = lab.trace.as_ref().expect("tracing on").len();
        let costs = lab.backend_costs(30);
        let cost_of = |b: Backend| costs.iter().find(|(k, _)| *k == b).map(|(_, c)| *c).unwrap_or(0.0);
        // A per-act ORACLE: the cheapest backend for this regime. No pinned arm can be this,
        // because a pin holds one backend for the whole run — so it is a tighter bound than the
        // bake-off's "best fixed choice", and the right thing to charge a switching policy against.
        let best_here = costs.iter().map(|(_, c)| *c).fold(f64::INFINITY, f64::min);

        let (mut lag_steps, mut lag_price) = (0usize, 0.0f64);
        let (mut obeyed, mut wrong_price) = (0usize, 0.0f64);
        for r in &lab.trace.as_ref().expect("tracing on")[act_start..seen_rows] {
            if r.held != r.wanted {
                // "Right but slow": it had decided, and hysteresis held it. Signed, because the
                // thing it wanted is sometimes the more expensive one.
                lag_steps += 1;
                lag_price += cost_of(r.held) - cost_of(r.wanted);
            } else {
                // "Obeyed and still wrong": it got exactly what it asked for, and what it asked
                // for was not the cheapest available. Only this half is a thresholds problem.
                obeyed += 1;
                wrong_price += cost_of(r.held) - best_here;
            }
        }
        if lag_steps > 0 {
            println!("{:>30}    lag {lag_steps} steps, priced {lag_price:+.0} us ({:+.0} us/step)",
                     "", lag_price / lag_steps as f64);
        }
        if obeyed > 0 && wrong_price.abs() > 1.0 {
            println!("{:>30}    obeyed {obeyed} steps, {wrong_price:+.0} us above the act's cheapest",
                     "");
        }
        act_rows.push((label, lag_steps, lag_price));
        act_choice.push((label, obeyed, wrong_price));
    }
    // The strip, as text: one character per step, so the shape is visible without a window.
    // #167: what the hysteresis actually cost, act by act. Positive means the lag held us on a
    // more expensive backend; NEGATIVE means the policy was wrong about where it wanted to go and
    // being slow to obey saved money. Both happen, and an arithmetic that only ever added would
    // have reported the first and hidden the second.
    let lag_total: f64 = act_rows.iter().map(|(_, _, p)| p).sum();
    let lag_steps: usize = act_rows.iter().map(|(_, s, _)| s).sum();
    println!();
    println!("what the lag COST, per act (#167 - exact step counts, timed per-step gaps):");
    println!("{:>30} {:>10} {:>14} {:>14}", "act", "lag steps", "total us", "us/step");
    for (label, s, p) in &act_rows {
        if *s == 0 { println!("{label:>30} {:>10} {:>14} {:>14}", 0, "-", "-"); continue; }
        println!("{label:>30} {s:>10} {p:>+14.0} {:>+14.1}", p / *s as f64);
    }
    println!("{:>30} {lag_steps:>10} {lag_total:>+14.0} {:>+14.1}", "ALL",
             lag_total / lag_steps.max(1) as f64);
    // The share is the number that actually decides #149: a fix for the lag can recover at most
    // the lag. If that is a few percent while the gap to the best pinned choice is tens, the
    // attribution was wrong and a faster obeyer is not where the loss lives.
    let run_us: f64 = lab.trace.as_ref().map(|rows|
        rows.iter().map(|r| r.maintain_us + r.query_us).sum()).unwrap_or(0.0);
    if run_us > 0.0 {
        println!("That is {:.1}% of the {:.0} ms this run spent maintaining and querying.",
                 lag_total / run_us * 100.0, run_us / 1000.0);
        println!("#149 says the policy's shortfall is detector lag plus the migration's own");
        println!("rebuild. A fix for the lag can recover AT MOST the lag, and the bake-off below");
        println!("puts the policy 6-56% behind the best fixed choice across runs. A few percent");
        println!("cannot explain tens, so the attribution is mostly wrong and a faster obeyer is");
        println!("not where the loss lives. Quote this as 1-3%: three runs read 0.9, 1.8 and 3.3.");
    }

    // ...and where the rest of it lives. Two disjoint buckets over the same steps: the policy
    // was held against its will, or it was obeyed and still not cheapest. Charged against a
    // per-act ORACLE that switches perfectly, so this is the whole shortfall of a switching
    // policy, not the shortfall against one pinned arm.
    let wrong_total: f64 = act_choice.iter().map(|(_, _, p)| p).sum();
    let obeyed_total: usize = act_choice.iter().map(|(_, s, _)| s).sum();
    println!();
    println!("where the shortfall lives, against a per-act oracle:");
    println!("  right but slow (lag)      {lag_steps:>5} steps   {lag_total:>+10.0} us");
    println!("  obeyed and still wrong    {obeyed_total:>5} steps   {wrong_total:>+10.0} us");
    let both = lag_total + wrong_total;
    if both.abs() > 1.0 {
        println!("  -> the choice accounts for {:.0}% of the two, the lag for {:.0}%.",
                 wrong_total / both * 100.0, lag_total / both * 100.0);
        println!("     Only the FIRST is a latency problem (#149); the second is a thresholds");
        println!("     problem (#158). Build for whichever is larger, and it is not close.");
    }
    if lag_total > 0.0 {
        println!("Hysteresis cost {lag_total:.0} us over {lag_steps} held steps. Against a run whose");
        println!("own arms move several percent between passes, read that as an order of magnitude:");
        println!("it is the first number #149 has ever had, not a precise one.");
    } else {
        println!("Hysteresis SAVED {:.0} us on balance: over those {lag_steps} steps the backend it", -lag_total);
        println!("wanted was on average the MORE expensive one, so being slow to obey was right.");
        println!("Worth knowing before building a faster obeyer (#149).");
    }

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
    for (label, _, us) in &rows { println!("  {label:>10} {us:>10.1}"); }
    // Say what it means, and say it carefully. The arms here are a few percent apart on a
    // machine whose noise is episodic (MEASURING.md § 8e), so the ORDER is worth more than the
    // ratio and a 5 % gap is not a finding.
    if let Some((bl, _, bus)) = rows.iter().skip(1).min_by(|a, b| a.2.total_cmp(&b.2)).copied() {
        let r = bus / rows[0].2.max(1e-9);
        println!("  -> best fixed is {bl} at {bus:.0} us; the policy runs at {:.0}, i.e. {r:.2}x it.", rows[0].2);
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

    if let Some(path) = trace_path {
        let csv = lab.trace_csv();
        let rows = csv.lines().count().saturating_sub(1);
        match std::fs::write(&path, &csv) {
            Ok(()) => println!("
trace -> {path} ({rows} steps)"),
            Err(e) => eprintln!("
could not write {path}: {e}"),
        }
    }
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
                    KeyCode::KeyA => {
                        auto = !auto;
                        auto_step = 0;
                        // Handing control back inside the frozen act would leave the scene still
                        // and the key looking broken. Thaw on the way out; `F` freezes again.
                        if !auto { lab.knobs.frozen = false; }
                    }
                    KeyCode::KeyF => lab.knobs.frozen = !lab.knobs.frozen,
                    KeyCode::KeyP => lab.knobs.paused = !lab.knobs.paused,
                    KeyCode::KeyR => { let k = lab.knobs; lab = Lab::new(0xA11CE); lab.knobs = k; bake = None; }
                    // The honest half of the demo: the HUD says what the policy CHOSE, this says
                    // whether that was right. It costs a visible stutter, which is the correct
                    // price for an answer nobody can get by watching.
                    KeyCode::KeyC => {
                        let rows: Vec<(&'static str, f64)> =
                            lab.bakeoff(40).into_iter().map(|(l, _, us)| (l, us)).collect();
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

                    if auto && !lab.knobs.paused {
                        // One act per 110 frames — comfortably past the 120-tick cooldown only
                        // in the last act, on purpose: the earlier ones show the policy arriving
                        // late, which is the behaviour a total would hide.
                        // `adaptive_lab::ACTS` — the same table the headless run walks, so the
                        // strip on screen and the text timeline are provably one script.
                        let i = act_at(auto_step);
                        let paused = lab.knobs.paused;
                        lab.knobs = ACTS[i].1;
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
                    // Against the MEAN, not the peak. In the wide-radius act every agent is found
                    // by tens of queries, so peak and mean are within a few percent and dividing
                    // by the peak pins the whole field near 1.0 — reported as "some light yellow,
                    // some slightly darker, you can hardly tell them apart". Mid-ramp is now the
                    // typical agent, so what you see is variation ABOUT the typical, which is the
                    // only thing there is to see once everything is being found.
                    let total: u64 = lab.agents.iter().map(|a| a.hits as u64).sum();
                    let mean = (total as f32 / lab.agents.len().max(1) as f32).max(0.001);
                    inst.clear();
                    for a in &lab.agents {
                        // Normalise against the busiest agent this step, so the picture keeps its
                        // contrast at any query load instead of saturating. `+1` keeps a quiet
                        // frame from dividing by zero and from exploding one stray hit into white.
                        inst.push(Inst { x: a.p.x as f32, y: a.p.y as f32, r,
                                         heat: (a.hits as f32 / (2.0 * mean)).min(1.0) });
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
                        // `n` is what the index holds; the slider is the target. They differ while
                        // the population walks toward it, and showing only one made that look wrong.
                        format!("N {n}{}  Q/I {q:.2}  MV/I {m:.2}",
                                if n != lab.knobs.population { format!("/{}", lab.knobs.population) } else { String::new() }),
                        format!("HITS {:.1} PREDICT {:.1}", lab.mean_hits(), lab.stats.predicted_hits),
                        format!("MAINTAIN {:.0}US", maint_us),
                        format!("QUERY {:.0}US", query_us),
                    ];
                    for (i, t) in rows.iter().enumerate() {
                        push_text(&mut ui, pad + 4.0 * tp, sy + i as f32 * 9.0 * tp, tp * 0.8, [0.78, 0.84, 0.95, 0.95], t, cw, ch);
                    }
                    push_text(&mut ui, pad + 4.0 * tp, sy + 4.0 * 9.0 * tp, tp * 0.8, [0.55, 0.60, 0.72, 0.9],
                              &format!("F FREEZE P PAUSE C RACE R RESET A AUTO{}", if auto { " ON" } else { "" }), cw, ch);

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
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) heat: f32 };
@vertex
fn vs(@location(0) inst: vec4<f32>, @builtin(vertex_index) vi: u32) -> VO {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let c = corners[vi];
    var o: VO;
    o.clip = cam.vp * vec4<f32>(inst.xy + c * inst.z, 0.0, 1.0);
    o.uv = c;
    o.heat = inst.w;
    return o;
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    let d = dot(v.uv, v.uv);
    if (d > 1.0) { discard; }
    // How many queries found this agent, normalised: a RAMP, not a switch. A boolean saturated
    // the whole field as soon as the load reached one cull per item, which is exactly where the
    // interesting behaviour lives. Three stops so the busy middle stays distinguishable.
    let cold = vec3<f32>(0.20, 0.26, 0.40);
    let warm = vec3<f32>(0.95, 0.62, 0.22);
    let hot = vec3<f32>(1.0, 0.97, 0.80);
    let h = clamp(v.heat, 0.0, 1.0);
    let col = select(mix(cold, warm, h / 0.5), mix(warm, hot, (h - 0.5) / 0.5), h > 0.5);
    return vec4<f32>(col, smoothstep(1.0, 0.3, d));
}
"#;

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
