//! Visual critters demo — renders the shared simulation core
//! (`vectorial_hash_demos::sim`) with macroquad. The same simulation runs
//! headless via the `critters_headless` binary for statistics.
//!
//! Three structure modes (panel button or `M`): binary-split tree, the
//! reference quadtree, or both at once — in dual mode every operation runs
//! on both structures, cull agreement is checked live, and the graphs plot
//! one polyline per structure.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters --release`
//!
//! Controls:
//! - panel button or `M`: cycle binary tree / quadtree / both
//! - `1`/`2`/`3` select the spawn brush (drifter / hunter / pulsar)
//! - left click (hold to paint) spawns at the cursor, right click removes
//! - `+`/`-` spawn / remove five critters at random
//! - `R` cycles region rendering (fill+lines / lines / off)
//! - `D` toggles agent body radius (Minkowski dilation); the panel
//!   "agent r" slider tunes it — attacks/vision then hit critters whose
//!   centre is within the radius of the figure, and each critter draws its
//!   body ring
//! - `[` / `]` halve / double simulation speed, `Space` pauses, `Esc` quits
//! - the "tuning (live)" panel: split/merge thresholds (rebuilds the
//!   structures), per-kind population targets (up to 10000 each), respawn
//!   delay, speed, fire rate
//!
//! The whole simulation runs on ONE thread (only the startup bank
//! generation parallelizes); the graphs show where that thread's time goes.
//!
//! Env: CRITTERS_MAX_FRAMES=N exits after N frames (smoke testing);
//! CRITTERS_WORLD=N starts at world size N (the panel's "world" button steps it).

use std::collections::HashMap;

use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};

use vectorial_hash::{CellState, Point as VPoint, TemplateGrid};
use vectorial_hash_demos::sim::{
    build_arsenal, Arsenal, Kind, Mode, Sim, SimEvent, SimParams, ITEM_LIMIT, MAX_CRITTERS,
    RESPAWN_DELAY, WORLD_STEPS,
};
use vectorial_hash_templates::polygon::Polygon;

/// The on-screen map square, in pixels (fixed; the window is built for it).
/// Historically the world was 1024 drawn at 0.75× = 768 px; now the world size
/// is a live stepper, so the per-frame scale is `MAP_PX / world_size` (see the
/// loop) and the world always fills this same square regardless of its size.
const MAP_PX: f32 = 768.0;
const PANEL_W: f32 = 320.0;
const GRAPH_W: f32 = 310.0;
const EFFECT_TTL: f64 = 0.45;

const COL_BIN: Color = ORANGE;
const COL_QUAD: Color = SKYBLUE;
const COL_INT: Color = VIOLET;

fn kind_color(kind: Kind) -> Color {
    match kind {
        Kind::Drifter => SKYBLUE,
        Kind::Hunter => RED,
        Kind::Pulsar => GOLD,
    }
}

struct Effect {
    grid: Option<TemplateGrid>,
    outline: Vec<(f32, f32)>,
    origin: VPoint,
    color: Color,
    until: f64,
}

struct Ring {
    pos: VPoint,
    color: Color,
    start: f64,
    until: f64,
}

fn sample_polygon(poly: &Polygon) -> Vec<(f32, f32)> {
    let n = poly.vertices.len();
    let mut pts = Vec::new();
    for i in 0..n {
        let v = &poly.vertices[i];
        let w = &poly.vertices[(i + 1) % n];
        pts.push((v.x as f32, v.y as f32));
        if v.seg.d != 0 {
            let (xc, yc) = (v.seg.xc, v.seg.yc);
            let r = ((v.x - xc).powi(2) + (v.y - yc).powi(2)).sqrt();
            let a0 = (v.y - yc).atan2(v.x - xc);
            let a1 = (w.y - yc).atan2(w.x - xc);
            let mut sweep = a1 - a0;
            if v.seg.d == 1 && sweep <= 0.0 {
                sweep += std::f64::consts::TAU;
            }
            if v.seg.d == -1 && sweep >= 0.0 {
                sweep -= std::f64::consts::TAU;
            }
            let steps = (sweep.abs() / 0.2).ceil().max(2.0) as usize;
            for s in 1..steps {
                let a = a0 + sweep * s as f64 / steps as f64;
                pts.push(((xc + r * a.cos()) as f32, (yc + r * a.sin()) as f32));
            }
        }
    }
    pts
}

/// Translate simulation events into visual effects.
fn consume_events(
    sim: &mut Sim,
    arsenal: &Arsenal,
    now: f64,
    effects: &mut Vec<Effect>,
    rings: &mut Vec<Ring>,
) {
    for ev in sim.events.drain(..) {
        match ev {
            SimEvent::Attack { kind, origin, shape_id, angle_deg, origin_int } => {
                let figure = &arsenal.figures[&shape_id];
                let grid = arsenal.bank.template_for(figure, 16, 16, angle_deg, origin_int);
                let outline = arsenal
                    .base_polys
                    .get(&(shape_id, angle_deg as i64))
                    .map(|base| {
                        let mut poly = base.clone();
                        poly.move_by(origin_int.0 as f64, origin_int.1 as f64);
                        sample_polygon(&poly)
                    })
                    .unwrap_or_default();
                effects.push(Effect {
                    grid,
                    outline,
                    origin,
                    color: kind_color(kind),
                    until: now + EFFECT_TTL,
                });
            }
            SimEvent::Kill { pos, kind } => rings.push(Ring {
                pos,
                color: kind_color(kind),
                start: now,
                until: now + 0.6,
            }),
            SimEvent::Spawn { pos, kind, respawn } => rings.push(Ring {
                pos,
                color: if respawn { WHITE } else { kind_color(kind) },
                start: now,
                until: now + if respawn { 0.5 } else { 0.4 },
            }),
            SimEvent::Removed { pos, kind } => rings.push(Ring {
                pos,
                color: kind_color(kind),
                start: now,
                until: now + 0.4,
            }),
        }
    }
}

/// Stable distinct colour per node id (golden-ratio hue walk).
fn region_color(id: u32) -> Color {
    let hue = (id as f64 * 0.618_033_988_75).fract() as f32;
    let h = hue * 6.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    Color::new(0.25 + 0.75 * r, 0.25 + 0.75 * g, 0.25 + 0.75 * b, 1.0)
}

/// A region colour keyed on the cell's **position + size**, not its node id.
/// Node ids churn as leaves split/merge (the free-list reuses them), so colours
/// keyed on the id flicker every frame on any cell that's restructuring. A cell
/// at a given place keeps the same colour here, however the ids shuffle; a
/// split's children differ from the parent because the size is in the key.
fn region_color_at(x: f64, y: f64, w: f64) -> Color {
    let key = (x.round() as i64 as u32).wrapping_mul(73_856_093)
        ^ (y.round() as i64 as u32).wrapping_mul(19_349_663)
        ^ (w.round() as i64 as u32).wrapping_mul(83_492_791);
    region_color(key)
}

struct Series {
    data: Vec<f32>,
    cap: usize,
}

impl Series {
    fn new(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), cap }
    }
    fn push(&mut self, v: f32) {
        if self.data.len() == self.cap {
            self.data.remove(0);
        }
        self.data.push(v);
    }
    fn last(&self) -> f32 {
        self.data.last().copied().unwrap_or(0.0)
    }
}

fn draw_graph(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: &str,
    unit: &str,
    series: &[(&Series, Color, &str)],
) {
    draw_rectangle(x, y, w, h, Color::new(0.02, 0.02, 0.045, 1.0));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    let max = series
        .iter()
        .flat_map(|(s, _, _)| s.data.iter().cloned())
        .fold(0.0_f32, f32::max)
        .max(1e-6)
        * 1.1;
    for (s, color, _) in series {
        if s.data.len() < 2 {
            continue;
        }
        let plot_h = h - 26.0;
        let step = w / (s.cap.max(2) - 1) as f32;
        for i in 1..s.data.len() {
            let x1 = x + (i - 1) as f32 * step;
            let x2 = x + i as f32 * step;
            let y1 = y + h - 4.0 - (s.data[i - 1] / max) * plot_h;
            let y2 = y + h - 4.0 - (s.data[i] / max) * plot_h;
            draw_line(x1, y1, x2, y2, 1.5, *color);
        }
    }
    let mut header = format!("{label}:");
    for (s, _, name) in series {
        header.push_str(&format!(" {name} {:.2}", s.last()));
    }
    header.push_str(unit);
    draw_text(&header, x + 6.0, y + 16.0, 15.0, WHITE);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash critters".to_owned(),
        window_width: MAP_PX as i32 + PANEL_W as i32 + GRAPH_W as i32,
        window_height: MAP_PX as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    clear_background(Color::new(0.07, 0.07, 0.10, 1.0));
    draw_text("generating template bank...", 40.0, 60.0, 28.0, WHITE);
    next_frame().await;

    let arsenal = build_arsenal();
    let mem = arsenal.bank.memory_usage();
    println!(
        "template bank: {} combos -> {} unique grids in {:.2}s",
        arsenal.bank.entry_count(),
        arsenal.bank.unique_count(),
        arsenal.gen_seconds,
    );
    println!(
        "bank memory: {:.2} MB total (grids {:.2} MB, index {:.2} MB, dedup keys {:.2} MB)",
        mem.total() as f64 / 1e6,
        mem.grids_bytes as f64 / 1e6,
        mem.index_bytes as f64 / 1e6,
        mem.dedup_keys_bytes as f64 / 1e6,
    );

    let init_mode = std::env::var("CRITTERS_MODE").ok().and_then(|s| Mode::parse(&s)).unwrap_or(Mode::Binary);
    let mut sim = Sim::new(init_mode, ITEM_LIMIT, ITEM_LIMIT, 42);
    // Optional initial world size (the panel's "world" button steps it live).
    if let Some(w) = std::env::var("CRITTERS_WORLD").ok().and_then(|v| v.parse::<f64>().ok()) {
        sim.set_world_size(w, ITEM_LIMIT, ITEM_LIMIT);
    }

    // Live-tunable settings.
    let mut split_f: f32 = ITEM_LIMIT as f32;
    let mut merge_f: f32 = ITEM_LIMIT as f32;
    let mut drifters_f: f32 = 16.0;
    let mut hunters_f: f32 = 12.0;
    let mut pulsars_f: f32 = 14.0;
    let mut respawn_f: f32 = RESPAWN_DELAY as f32;
    let mut speed_f: f32 = 1.0;
    let mut fire_f: f32 = 1.0;
    // Live thread count for the combat wave's parallel attack culls (native only;
    // wasm has no threads and culls the wave serially). Defaults to all cores.
    #[cfg(not(target_arch = "wasm32"))]
    let max_threads_f = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f32;
    #[cfg(not(target_arch = "wasm32"))]
    let mut threads_f: f32 = max_threads_f;
    // Agent body radius (Minkowski dilation): 0 = point agents. Toggled with
    // `D` and tuned with the panel slider.
    let mut agent_radius_f: f32 = 0.0;

    let mut effects: Vec<Effect> = Vec::new();
    let mut rings: Vec<Ring> = Vec::new();
    let mut last_paint: f64 = 0.0;
    let mut paused = false;
    let mut ui_hidden = false; // visual-only mode (hide tuning window + panel + graphs)
    let mut brush = Kind::Hunter;
    let mut region_mode: u8 = 0;

    let max_frames: u64 = std::env::var("CRITTERS_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut frame: u64 = 0;

    let cap = 240;
    let mut g_frame = Series::new(cap);
    let (mut g_atk_t, mut g_atk_q, mut g_atk_it) = (Series::new(cap), Series::new(cap), Series::new(cap));
    let (mut g_vis_t, mut g_vis_q, mut g_vis_it) = (Series::new(cap), Series::new(cap), Series::new(cap));
    let (mut g_mv_t, mut g_mv_q, mut g_mv_it) = (Series::new(cap), Series::new(cap), Series::new(cap));
    let (mut g_rm_t, mut g_rm_q, mut g_rm_it) = (Series::new(cap), Series::new(cap), Series::new(cap));

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        let mut want_mode_switch = is_key_pressed(KeyCode::M);
        if is_key_pressed(KeyCode::Key1) {
            brush = Kind::Drifter;
        }
        if is_key_pressed(KeyCode::Key2) {
            brush = Kind::Hunter;
        }
        if is_key_pressed(KeyCode::Key3) {
            brush = Kind::Pulsar;
        }
        if is_key_pressed(KeyCode::R) {
            region_mode = (region_mode + 1) % 3;
        }
        if is_key_pressed(KeyCode::D) {
            // Toggle Minkowski dilation: agents gain / lose a body radius.
            agent_radius_f = if agent_radius_f > 0.0 { 0.0 } else { 16.0 };
        }
        if is_key_pressed(KeyCode::U) {
            // Visual-only mode: hide the tuning window, side panel and graphs,
            // leaving just the map. Keys still work. ([U] or the web button.)
            ui_hidden = !ui_hidden;
        }
        if is_key_pressed(KeyCode::LeftBracket) {
            speed_f = (speed_f * 0.5).max(0.25);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            speed_f = (speed_f * 2.0).min(4.0);
        }

        let now = get_time();
        let dt = (get_frame_time() as f64).min(0.05) * speed_f as f64;

        // Live render scale: the world (any stepper size) always fills the same
        // fixed on-screen map square, so the scale follows the world size. `S`
        // shadows the default const for the rest of the loop; the world stepper
        // below updates it in place so the resize shows the same frame.
        let world_size = sim.world_size();
        #[allow(non_snake_case)]
        let mut S = MAP_PX / world_size as f32;

        // --- interactive add / remove ---
        let (mx, my) = mouse_position();
        let (wx, wy) = ((mx / S) as f64, (my / S) as f64);
        let mouse_in_map = wx >= 0.0 && wx < world_size && wy >= 0.0 && wy < world_size && mx < MAP_PX;
        if is_mouse_button_down(MouseButton::Left)
            && mouse_in_map
            && sim.sims.item_count() < MAX_CRITTERS
            && now - last_paint > 0.12
        {
            last_paint = now;
            sim.spawn_at(brush, VPoint::new(wx, wy), false);
            match brush {
                Kind::Drifter => drifters_f += 1.0,
                Kind::Hunter => hunters_f += 1.0,
                Kind::Pulsar => pulsars_f += 1.0,
            }
        }
        if is_mouse_button_pressed(MouseButton::Right) && mouse_in_map {
            let snap = sim.sims.snapshot();
            let best = snap
                .iter()
                .map(|&(id, kind, pos, _)| {
                    let d = (pos.x - wx).powi(2) + (pos.y - wy).powi(2);
                    (d, id, pos, kind)
                })
                .filter(|(d, ..)| *d < 30.0 * 30.0)
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if let Some((_, id, pos, kind)) = best {
                sim.remove_by_id(pos, id, kind);
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }
        if is_key_pressed(KeyCode::Equal) && sim.sims.item_count() < MAX_CRITTERS {
            for _ in 0..5 {
                let pos = sim.random_pos();
                sim.spawn_at(brush, pos, false);
                match brush {
                    Kind::Drifter => drifters_f += 1.0,
                    Kind::Hunter => hunters_f += 1.0,
                    Kind::Pulsar => pulsars_f += 1.0,
                }
            }
        }
        if is_key_pressed(KeyCode::Minus) {
            let mut all = sim.sims.snapshot();
            for _ in 0..5 {
                if all.is_empty() {
                    break;
                }
                let i = sim.rng.below(all.len());
                let (id, kind, pos, _) = all.swap_remove(i);
                sim.remove_by_id(pos, id, kind);
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }

        // --- tuning panel ---
        let mut want_world_step = false;
        if !ui_hidden {
        widgets::Window::new(
            hash!(),
            vec2(MAP_PX + 10.0, 446.0),
            vec2(PANEL_W - 20.0, 308.0),
        )
        .label("tuning (live)")
        .titlebar(true)
        .movable(false)
        .ui(&mut *root_ui(), |ui| {
            if ui.button(None, format!("structure: {}  [click or M]", sim.sims.mode().name())) {
                want_mode_switch = true;
            }
            if ui.button(None, format!("world: {}^2  [click]", world_size as i32)) {
                want_world_step = true;
            }
            ui.separator();
            ui.slider(hash!(), "split >", 1f32..12f32, &mut split_f);
            ui.slider(hash!(), "merge <=", 1f32..12f32, &mut merge_f);
            ui.separator();
            ui.slider(hash!(), "drifters", 0f32..10000f32, &mut drifters_f);
            ui.slider(hash!(), "hunters", 0f32..10000f32, &mut hunters_f);
            ui.slider(hash!(), "pulsars", 0f32..10000f32, &mut pulsars_f);
            ui.separator();
            ui.slider(hash!(), "respawn s", 0.5f32..10f32, &mut respawn_f);
            ui.slider(hash!(), "speed x", 0.25f32..4f32, &mut speed_f);
            ui.slider(hash!(), "fire x", 0.25f32..3f32, &mut fire_f);
            // Combat attack culls fan out over a rayon pool; size it live (native
            // only). Watch the "attack cull" graph / fps as you drag — the same
            // crossover PARALLEL.md measures, now in the 2D demo.
            #[cfg(not(target_arch = "wasm32"))]
            ui.slider(hash!(), "threads", 1f32..max_threads_f, &mut threads_f);
            ui.separator();
            ui.slider(hash!(), "agent r [D]", 0f32..40f32, &mut agent_radius_f);
        });
        }

        split_f = split_f.round().clamp(1.0, 12.0);
        merge_f = merge_f.round().clamp(1.0, split_f);
        drifters_f = drifters_f.round();
        hunters_f = hunters_f.round();
        pulsars_f = pulsars_f.round();
        let want_split = split_f as usize;
        let want_merge = (merge_f as usize).min(want_split);

        if want_mode_switch {
            let next = sim.sims.mode().next();
            sim.set_mode(next, want_split, want_merge);
        }
        if want_world_step {
            // Cycle to the next stepped world size; re-bounds critters and
            // rebuilds the index. Update the live scale so the new size draws
            // this same frame (no one-frame glitch at the old scale).
            let cur = sim.world_size();
            let idx = WORLD_STEPS.iter().position(|&w| (w - cur).abs() < 0.5).unwrap_or(2);
            let next = WORLD_STEPS[(idx + 1) % WORLD_STEPS.len()];
            sim.set_world_size(next, want_split, want_merge);
            S = MAP_PX / next as f32;
        }
        let current_limits = sim
            .sims
            .tree
            .as_ref()
            .map(|t| (t.item_limit, t.merge_limit))
            .or_else(|| sim.sims.quad.as_ref().map(|q| (q.item_limit, q.merge_limit)));
        if current_limits != Some((want_split, want_merge)) {
            sim.set_limits(want_split, want_merge);
        }

        // --- simulation step ---
        sim.sims.begin_frame();
        if !paused {
            let params = SimParams {
                targets: [drifters_f as usize, hunters_f as usize, pulsars_f as usize],
                respawn_delay: respawn_f as f64,
                fire_rate: fire_f as f64,
                no_attack: false,
                agent_radius: agent_radius_f as f64,
            };
            #[cfg(not(target_arch = "wasm32"))]
            sim.set_threads(threads_f.round() as usize);
            sim.step(dt, &arsenal, &params);
            consume_events(&mut sim, &arsenal, now, &mut effects, &mut rings);
        }
        effects.retain(|e| e.until > now);
        rings.retain(|r| r.until > now);

        // Graph samples.
        g_frame.push(get_frame_time() * 1000.0);
        let avg = |total: f64, n: u32| if n > 0 { (total / n as f64) as f32 } else { 0.0 };
        g_atk_t.push(avg(sim.sims.t.atk, sim.sims.t.atk_n));
        g_atk_q.push(avg(sim.sims.q.atk, sim.sims.q.atk_n));
        g_atk_it.push(avg(sim.sims.it.atk, sim.sims.it.atk_n));
        g_vis_t.push(avg(sim.sims.t.vis, sim.sims.t.vis_n));
        g_vis_q.push(avg(sim.sims.q.vis, sim.sims.q.vis_n));
        g_vis_it.push(avg(sim.sims.it.vis, sim.sims.it.vis_n));
        g_mv_t.push((sim.sims.t.mv / 1000.0) as f32);
        g_mv_q.push((sim.sims.q.mv / 1000.0) as f32);
        g_mv_it.push((sim.sims.it.mv / 1000.0) as f32);
        g_rm_t.push(sim.sims.t.rm as f32);
        g_rm_q.push(sim.sims.q.rm as f32);
        g_rm_it.push(sim.sims.it.rm as f32);

        // ---------- draw ----------
        clear_background(Color::new(0.07, 0.07, 0.10, 1.0));

        let mode = sim.sims.mode();
        let mut tree_leaves = 0usize;
        let mut quad_leaves = 0usize;
        let mut items = 0usize;

        if let Some(t) = &sim.sims.tree {
            t.visit_leaves(|_id, leaf| {
                tree_leaves += 1;
                items += leaf.items.len();
                let b = leaf.bbox;
                let c = region_color_at(b.x, b.y, b.width);
                if region_mode == 0 {
                    draw_rectangle(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        Color::new(c.r, c.g, c.b, 0.28),
                    );
                }
                if region_mode <= 1 {
                    draw_rectangle_lines(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        1.0,
                        Color::new(1.0, 1.0, 1.0, 0.18),
                    );
                }
            });
        }
        if let Some(q) = &sim.sims.quad {
            let quad_only = sim.sims.tree.is_none();
            q.visit_leaves(|_id, leaf| {
                quad_leaves += 1;
                if quad_only {
                    items += leaf.items.len();
                }
                let b = leaf.bbox;
                if quad_only {
                    let c = region_color_at(b.x, b.y, b.width);
                    if region_mode == 0 {
                        draw_rectangle(
                            b.x as f32 * S,
                            b.y as f32 * S,
                            b.width as f32 * S,
                            b.height as f32 * S,
                            Color::new(c.r, c.g, c.b, 0.28),
                        );
                    }
                    if region_mode <= 1 {
                        draw_rectangle_lines(
                            b.x as f32 * S,
                            b.y as f32 * S,
                            b.width as f32 * S,
                            b.height as f32 * S,
                            1.0,
                            Color::new(1.0, 1.0, 1.0, 0.18),
                        );
                    }
                } else if region_mode <= 1 {
                    draw_rectangle_lines(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        1.5,
                        Color::new(COL_QUAD.r, COL_QUAD.g, COL_QUAD.b, 0.55),
                    );
                }
            });
        }
        if let Some(t) = &sim.sims.itree {
            // IntegerTree (IBinary mode): same region rendering as the binary
            // tree, but the leaf bbox is integer (`IRect` x/y/w/h).
            t.visit_leaves(|_id, leaf| {
                tree_leaves += 1;
                items += leaf.items.len();
                let b = leaf.bbox;
                let c = region_color_at(b.x as f64, b.y as f64, b.w as f64);
                if region_mode == 0 {
                    draw_rectangle(
                        b.x as f32 * S, b.y as f32 * S, b.w as f32 * S, b.h as f32 * S,
                        Color::new(c.r, c.g, c.b, 0.28),
                    );
                }
                if region_mode <= 1 {
                    draw_rectangle_lines(
                        b.x as f32 * S, b.y as f32 * S, b.w as f32 * S, b.h as f32 * S,
                        1.0, Color::new(COL_INT.r, COL_INT.g, COL_INT.b, 0.45),
                    );
                }
            });
        }

        for e in &effects {
            let fade = ((e.until - now) / EFFECT_TTL) as f32;
            if let Some(grid) = &e.grid {
                for row in 0..grid.rows {
                    for col in 0..grid.cols {
                        let alpha = match grid.cell(col, row) {
                            CellState::In => 0.5 * fade,
                            CellState::Maybe => 0.22 * fade,
                            CellState::Out => continue,
                        };
                        draw_rectangle(
                            (grid.origin_x + col as f64 * grid.cell_w) as f32 * S,
                            (grid.origin_y + row as f64 * grid.cell_h) as f32 * S,
                            grid.cell_w as f32 * S,
                            grid.cell_h as f32 * S,
                            Color::new(e.color.r, e.color.g, e.color.b, alpha),
                        );
                    }
                }
            }
            let oc = Color::new(e.color.r, e.color.g, e.color.b, (0.9 * fade).min(1.0));
            for i in 0..e.outline.len() {
                let (x1, y1) = e.outline[i];
                let (x2, y2) = e.outline[(i + 1) % e.outline.len()];
                draw_line(x1 * S, y1 * S, x2 * S, y2 * S, 1.5, oc);
            }
            draw_circle(e.origin.x as f32 * S, e.origin.y as f32 * S, 3.0, e.color);
        }

        for &(from, to) in &sim.sightlines {
            draw_line(
                from.x as f32 * S,
                from.y as f32 * S,
                to.x as f32 * S,
                to.y as f32 * S,
                1.0,
                Color::new(1.0, 0.3, 0.3, 0.18),
            );
        }
        let draw_snap = sim.sims.snapshot();
        for &(_, kind, pos, heading) in &draw_snap {
            let (x, y) = (pos.x as f32 * S, pos.y as f32 * S);
            // Body radius ring (Minkowski dilation active when agent_radius > 0).
            if agent_radius_f > 0.0 {
                draw_circle_lines(x, y, agent_radius_f * S, 1.0, Color::new(1.0, 1.0, 1.0, 0.18));
            }
            draw_circle(x, y, 5.0, kind_color(kind));
            draw_line(
                x,
                y,
                x + (heading.cos() * 9.0) as f32 * S,
                y + (heading.sin() * 9.0) as f32 * S,
                1.5,
                WHITE,
            );
        }
        for r in &rings {
            let t = ((now - r.start) / (r.until - r.start).max(1e-6)) as f32;
            draw_circle_lines(
                r.pos.x as f32 * S,
                r.pos.y as f32 * S,
                (4.0 + 16.0 * t) * S,
                2.0,
                Color::new(r.color.r, r.color.g, r.color.b, (1.0 - t).max(0.0)),
            );
        }

        // ---------- side panel (hidden in visual-only mode) ----------
        if !ui_hidden {
        let map_px = MAP_PX;
        let map_py = MAP_PX;
        draw_rectangle_lines(0.0, 0.0, map_px, map_py, 2.0, GRAY);
        let px = map_px + 14.0;
        draw_rectangle(map_px, 0.0, PANEL_W, map_py, Color::new(0.04, 0.04, 0.06, 1.0));
        let mut ty = 28.0;
        let line = |text: &str, size: f32, color: Color, ty: &mut f32| {
            draw_text(text, px, *ty, size, color);
            *ty += size * 1.15;
        };
        line("vectorial-hash critters", 26.0, WHITE, &mut ty);
        ty += 6.0;
        line(
            &format!("[M] structure: {}", mode.name()),
            19.0,
            match mode {
                Mode::Binary => COL_BIN,
                Mode::Quad => COL_QUAD,
                Mode::Both => YELLOW,
                Mode::IBinary => COL_INT,
            },
            &mut ty,
        );
        if mode == Mode::Both {
            line(
                &format!(
                    "  culls agree: {}",
                    if sim.sims.mismatches == 0 {
                        "yes".to_string()
                    } else {
                        format!("{} mismatches!", sim.sims.mismatches)
                    },
                ),
                16.0,
                if sim.sims.mismatches == 0 { GREEN } else { RED },
                &mut ty,
            );
        }
        ty += 6.0;
        line(
            &format!(
                "bank: {} combos -> {} unique",
                arsenal.bank.entry_count(),
                arsenal.bank.unique_count(),
            ),
            17.0,
            LIGHTGRAY,
            &mut ty,
        );
        ty += 8.0;
        let mut counts: HashMap<Kind, usize> = HashMap::new();
        for &(_, kind, _, _) in &draw_snap {
            *counts.entry(kind).or_default() += 1;
        }
        for kind in Kind::ALL {
            draw_circle(px + 6.0, ty - 5.0, 5.0, kind_color(kind));
            line(
                &format!("    {} x{}", kind.name(), counts.get(&kind).copied().unwrap_or(0)),
                16.0,
                WHITE,
                &mut ty,
            );
        }
        ty += 8.0;
        line(&format!("alive: {}   kills: {}", items, sim.kills), 18.0, WHITE, &mut ty);
        line(
            &format!(
                "  by: drf {} / hun {} / pul {}",
                sim.kills_by.get(&Kind::Drifter).copied().unwrap_or(0),
                sim.kills_by.get(&Kind::Hunter).copied().unwrap_or(0),
                sim.kills_by.get(&Kind::Pulsar).copied().unwrap_or(0),
            ),
            15.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!("respawning: {}", sim.respawn_queue_len()),
            17.0,
            WHITE,
            &mut ty,
        );
        ty += 8.0;
        if let Some(t) = &sim.sims.tree {
            line(
                &format!("binary: {} leaves, {} nodes", tree_leaves, t.node_count()),
                16.0,
                COL_BIN,
                &mut ty,
            );
        }
        if let Some(q) = &sim.sims.quad {
            line(
                &format!("quad:   {} leaves, {} nodes", quad_leaves, q.node_count()),
                16.0,
                COL_QUAD,
                &mut ty,
            );
        }
        line(
            &format!("limits: split >{want_split}, merge <={want_merge}"),
            16.0,
            LIGHTGRAY,
            &mut ty,
        );
        ty += 8.0;
        draw_circle(px + 6.0, ty - 5.0, 5.0, kind_color(brush));
        line(&format!("    brush: {} (1/2/3)", brush.name()), 17.0, WHITE, &mut ty);
        line(
            &format!("speed {:.2}x   regions: {}", speed_f, match region_mode {
                0 => "fill+lines",
                1 => "lines",
                _ => "off",
            }),
            16.0,
            WHITE,
            &mut ty,
        );
        line(&format!("fps: {}", get_fps()), 17.0, GREEN, &mut ty);
        if paused {
            line("PAUSED", 22.0, ORANGE, &mut ty);
        }

        // ---------- graphs column ----------
        let gx = map_px + PANEL_W;
        draw_rectangle(gx, 0.0, GRAPH_W, map_py, Color::new(0.05, 0.05, 0.08, 1.0));
        let gpad = 10.0;
        let gw = GRAPH_W - 2.0 * gpad;
        let gh = 118.0;
        let mut gy = 14.0;
        draw_text("performance (live, 1 thread)", gx + gpad, gy + 6.0, 19.0, WHITE);
        gy += 16.0;
        draw_graph(gx + gpad, gy, gw, gh, "frame", " ms", &[(&g_frame, GREEN, "")]);
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "attack cull", " us",
            &[(&g_atk_t, COL_BIN, "bin"), (&g_atk_q, COL_QUAD, "quad"), (&g_atk_it, COL_INT, "int")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "vision cull", " us",
            &[(&g_vis_t, COL_BIN, "bin"), (&g_vis_q, COL_QUAD, "quad"), (&g_vis_it, COL_INT, "int")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "move+update", " ms",
            &[(&g_mv_t, COL_BIN, "bin"), (&g_mv_q, COL_QUAD, "quad"), (&g_mv_it, COL_INT, "int")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "insert+remove", " us",
            &[(&g_rm_t, COL_BIN, "bin"), (&g_rm_q, COL_QUAD, "quad"), (&g_rm_it, COL_INT, "int")],
        );

        let help = [
            "[M] structure   [1/2/3] brush",
            "[LMB] paint   [RMB] remove   [+/-] x5",
            "[R] regions   [ / ] speed   [Space] pause",
        ];
        for (i, h) in help.iter().enumerate() {
            draw_text(
                h,
                gx + gpad,
                map_py - 12.0 - 18.0 * (help.len() - 1 - i) as f32,
                15.0,
                GRAY,
            );
        }
        }

        frame += 1;
        if max_frames > 0 && frame >= max_frames {
            break;
        }
        if let Some(p) = std::env::var_os("SHOT") { if frame >= 120 { let _ = get_screen_data().export_png(&p.to_string_lossy()); std::process::exit(0); } }
        next_frame().await
    }
}
