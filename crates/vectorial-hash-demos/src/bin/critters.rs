//! Visual critters demo — precise template application.
//!
//! A 2D map indexed by a `vectorial_hash::Tree`. Critters of three kinds move
//! with distinct behaviours and attack using **precomputed template areas**
//! served from a hierarchical [`TemplateBank`] (shape → dims → cell width →
//! cell height → angle → offset x → offset y), generated at startup.
//!
//! The attack figure is **never moved to fit a grid**: it is applied at its
//! real (integer) origin, and for every tree-cell size touched the bank
//! serves the template whose generation offset matches the origin's
//! displacement within the global virtual grid of that size — so template
//! cells align 1:1 with the map's cells. Leaf items are resolved with a 1×1
//! raster (only boundary pixels need exact geometry). Dead critters respawn;
//! each live tree region is drawn in its own colour, so splits and merge-ups
//! are visible in real time.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters --release`
//!
//! Controls:
//! - `1`/`2`/`3` select the spawn brush (drifter / hunter / pulsar)
//! - left click (hold to paint) spawns at the cursor, right click removes
//! - `+`/`-` spawn / remove five critters at random
//! - `R` cycles region rendering (fill+lines / lines / off)
//! - `[` / `]` halve / double simulation speed, `Space` pauses, `Esc` quits
//! - the "tuning (live)" panel adjusts split/merge thresholds (tree rebuilt
//!   on change), per-kind population targets (up to 400 each), respawn
//!   delay, speed and fire rate while the simulation runs
//!
//! Hunters acquire prey with a real `Tree::cull` over a vision circle. The
//! right-hand column plots live performance: frame time, average attack-cull
//! and vision-cull times, and the movement/update cost per frame.
//!
//! Env: CRITTERS_MAX_FRAMES=N exits after N frames (smoke testing).

use std::collections::{HashMap, HashSet};

use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};

use vectorial_hash::{
    CellState, PlacedTemplate, Point as VPoint, Positioned, Rect as VRect, Shape, TemplateGrid,
    Tree,
};
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{create_circle, create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::angle_to_radians;

/// World size (map units). Power of two so binary splits produce integer
/// cell sizes aligned with the global virtual grids.
const MAP_W: f64 = 1024.0;
const MAP_H: f64 = 1024.0;
/// Render scale: world 1024 drawn at 768 px.
const S: f32 = 0.75;
const PANEL_W: f32 = 320.0;
/// Right-hand column with live performance graphs.
const GRAPH_W: f32 = 300.0;
const MARGIN: f64 = 4.0;
/// Hunters acquire prey with a `Tree::cull` of this vision radius.
const VISION_RADIUS: f64 = 280.0;
/// Hard cap on the total population.
const MAX_CRITTERS: usize = 1500;

const ANGLE_STEP_DEG: f64 = 15.0;
const DROP_ID: u32 = 0;
const CIRCLE_ID: u32 = 1;
const DROP_SCALE: f64 = 110.0;
const CIRCLE_RADIUS: f64 = 48.0;

/// Cell sizes (w, h) with full template sets. Tree cells of other sizes fall
/// back to bbox recursion (internal nodes) or the 1×1 raster (leaf items).
const TEMPLATE_SIZES: [(u32, u32); 7] =
    [(8, 8), (16, 16), (32, 32), (8, 16), (16, 8), (16, 32), (32, 16)];

const ITEM_LIMIT: usize = 3;
const RESPAWN_DELAY: f64 = 2.5;
const EFFECT_TTL: f64 = 0.45;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Drifter,
    Hunter,
    Pulsar,
}

impl Kind {
    fn color(self) -> Color {
        match self {
            Kind::Drifter => SKYBLUE,
            Kind::Hunter => RED,
            Kind::Pulsar => GOLD,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::Drifter => "drifter",
            Kind::Hunter => "hunter",
            Kind::Pulsar => "pulsar",
        }
    }
    fn speed(self) -> f64 {
        match self {
            Kind::Drifter => 75.0,
            Kind::Hunter => 105.0,
            Kind::Pulsar => 88.0,
        }
    }
    fn cooldown(self) -> (f32, f32) {
        match self {
            Kind::Drifter => (2.5, 4.5),
            Kind::Hunter => (1.8, 3.5),
            Kind::Pulsar => (3.0, 5.0),
        }
    }
}

#[derive(Clone)]
struct Critter {
    id: u32,
    kind: Kind,
    pos: VPoint,
    heading: f64,
}

impl Positioned for Critter {
    fn position(&self) -> VPoint {
        self.pos
    }
}

/// The startup-generated template bank plus per-figure metadata.
struct Arsenal {
    bank: TemplateBank,
    figures: HashMap<u32, FigureKey>,
    base_polys: HashMap<(u32, i64), Polygon>,
    gen_seconds: f64,
}

fn build_arsenal() -> Arsenal {
    let start = std::time::Instant::now();
    let mut bank = TemplateBank::new();
    let mut figures = HashMap::new();
    let mut base_polys = HashMap::new();

    let drop_angles: Vec<f64> = (0..(360.0 / ANGLE_STEP_DEG) as i64)
        .map(|i| i as f64 * ANGLE_STEP_DEG)
        .collect();

    // (shape_id, dims, scaled base polygon, angles)
    let shapes: [(u32, Vec<f64>, Polygon, Vec<f64>); 2] = [
        (
            DROP_ID,
            vec![0.2 * DROP_SCALE, 0.8 * DROP_SCALE],
            scaled_copy(&create_drop(0.2, 0.8), DROP_SCALE, DROP_SCALE),
            drop_angles,
        ),
        (
            CIRCLE_ID,
            vec![CIRCLE_RADIUS],
            scaled_copy(&create_circle(1.0), CIRCLE_RADIUS, CIRCLE_RADIUS),
            vec![0.0],
        ),
    ];

    for (shape_id, dims, base, angles) in shapes {
        let figure = FigureKey::new(shape_id, &dims);
        for &(cw, ch) in &TEMPLATE_SIZES {
            bank.generate_size(&figure, &base, &angles, cw, ch);
        }
        // 1×1 point rasters: one per angle (integer origins → offset (0,0)).
        bank.generate_size(&figure, &base, &angles, 1, 1);
        for &angle in &angles {
            base_polys.insert((shape_id, angle as i64), rotated_copy(&base, angle_to_radians(angle)));
        }
        figures.insert(shape_id, figure);
    }

    Arsenal {
        bank,
        figures,
        base_polys,
        gen_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Attack area applied at its real integer origin. Internal tree nodes are
/// classified via per-size templates from the bank; leaf items use the 1×1
/// raster with exact geometry only on boundary pixels.
struct AttackShape<'a> {
    bank: &'a TemplateBank,
    figure: FigureKey,
    angle_deg: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: VRect,
    raster: Option<PlacedTemplate>,
}

impl Shape for AttackShape<'_> {
    fn bounding_box(&self) -> VRect {
        self.bbox
    }
    fn contains_point(&self, p: VPoint) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None;
        }
        self.bank.placed_for(
            &self.figure,
            cell_w as u32,
            cell_h as u32,
            self.angle_deg,
            self.origin,
        )
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        self.raster.as_ref()
    }
}

/// Build an attack at `pos` (rounded to the integer lattice the templates
/// were generated on). For the drop, `aim` is the firing direction (the
/// drop's body extends along +y at angle 0); the circle blast passes `None`.
fn make_attack<'a>(
    arsenal: &'a Arsenal,
    shape_id: u32,
    pos: VPoint,
    aim: Option<(f64, f64)>,
) -> Option<AttackShape<'a>> {
    let angle_deg = match aim {
        Some((dx, dy)) => {
            let a = (-dx).atan2(dy).to_degrees();
            ((a / ANGLE_STEP_DEG).round() * ANGLE_STEP_DEG).rem_euclid(360.0)
        }
        None => 0.0,
    };
    let origin = (pos.x.round() as i64, pos.y.round() as i64);

    let figure = arsenal.figures.get(&shape_id)?.clone();
    let mut poly = arsenal.base_polys.get(&(shape_id, angle_deg as i64))?.clone();
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = VRect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );
    let raster = arsenal.bank.placed_raster(&figure, angle_deg, origin);

    Some(AttackShape {
        bank: &arsenal.bank,
        figure,
        angle_deg,
        origin,
        poly,
        bbox,
        raster,
    })
}

struct Effect {
    /// Template at a representative cell size (for visualisation).
    grid: Option<TemplateGrid>,
    outline: Vec<(f32, f32)>,
    origin: VPoint,
    color: Color,
    until: f64,
}

/// Expanding ring marking a spawn, a removal, or a death.
struct Ring {
    pos: VPoint,
    color: Color,
    start: f64,
    until: f64,
}

/// Flatten a polygon (lines + arcs) into a point list for rendering.
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

fn spawn_at(tree: &mut Tree<Critter>, next_id: &mut u32, kind: Kind, pos: VPoint) -> VPoint {
    let id = *next_id;
    *next_id += 1;
    let heading = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
    tree.insert(Critter { id, kind, pos, heading });
    pos
}

fn spawn(tree: &mut Tree<Critter>, next_id: &mut u32, kind: Kind) -> VPoint {
    let pos = VPoint::new(
        rand::gen_range(20.0_f32, (MAP_W - 20.0) as f32) as f64,
        rand::gen_range(20.0_f32, (MAP_H - 20.0) as f32) as f64,
    );
    spawn_at(tree, next_id, kind, pos)
}

/// One movement step. Returns (new position, new heading). Hunters receive
/// their prey position (found beforehand with a vision-circle cull).
fn step_critter(
    kind: Kind,
    pos: VPoint,
    heading: f64,
    dt: f64,
    target: Option<VPoint>,
) -> (VPoint, f64) {
    let mut h = heading;
    match kind {
        // Smooth random walk.
        Kind::Drifter => h += rand::gen_range(-2.5_f32, 2.5) as f64 * dt,
        // Constant turn rate: circles around the map.
        Kind::Pulsar => h += 1.6 * dt,
        // Steer toward the prey, or wander when nothing is in sight.
        Kind::Hunter => match target {
            Some(tpos) => {
                let desired = (tpos.y - pos.y).atan2(tpos.x - pos.x);
                let diff = (desired - h + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                h += diff.clamp(-3.0 * dt, 3.0 * dt);
            }
            None => h += rand::gen_range(-2.0_f32, 2.0) as f64 * dt,
        },
    }

    let speed = kind.speed();
    let mut nx = pos.x + h.cos() * speed * dt;
    let mut ny = pos.y + h.sin() * speed * dt;
    if nx < MARGIN || nx > MAP_W - MARGIN {
        h = std::f64::consts::PI - h;
        nx = nx.clamp(MARGIN, MAP_W - MARGIN);
    }
    if ny < MARGIN || ny > MAP_H - MARGIN {
        h = -h;
        ny = ny.clamp(MARGIN, MAP_H - MARGIN);
    }
    (VPoint::new(nx, ny), h.rem_euclid(std::f64::consts::TAU))
}

/// Stable distinct colour per NodeId (golden-ratio hue walk).
fn region_color(id: u32) -> Color {
    let hue = (id as f64 * 0.618_033_988_75).fract() as f32;
    hue_to_rgb(hue)
}

fn hue_to_rgb(hue: f32) -> Color {
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

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash critters".to_owned(),
        window_width: (MAP_W as f32 * S) as i32 + PANEL_W as i32 + GRAPH_W as i32,
        window_height: (MAP_H as f32 * S) as i32,
        ..Default::default()
    }
}

/// Hunter vision: a plain circle shape culled against the tree.
struct VisionCircle {
    center: VPoint,
    r: f64,
}

impl Shape for VisionCircle {
    fn bounding_box(&self) -> VRect {
        VRect::new(
            self.center.x - self.r,
            self.center.y - self.r,
            self.r * 2.0,
            self.r * 2.0,
        )
    }
    fn contains_point(&self, p: VPoint) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        dx * dx + dy * dy <= self.r * self.r
    }
}

/// Nearest non-hunter within vision range, found with a real tree cull.
/// Returns the prey position and the cull's duration in microseconds.
fn nearest_prey(tree: &Tree<Critter>, pos: VPoint, self_id: u32) -> (Option<VPoint>, f64) {
    let t = std::time::Instant::now();
    let vision = VisionCircle { center: pos, r: VISION_RADIUS };
    let prey = tree
        .cull(&vision)
        .into_iter()
        .filter(|c| c.id != self_id && c.kind != Kind::Hunter)
        .min_by(|a, b| {
            let da = (a.pos.x - pos.x).powi(2) + (a.pos.y - pos.y).powi(2);
            let db = (b.pos.x - pos.x).powi(2) + (b.pos.y - pos.y).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|c| c.pos);
    (prey, t.elapsed().as_secs_f64() * 1e6)
}

/// Fixed-capacity rolling series for the live graphs.
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

/// Autoscaled polyline graph with label and current value.
fn draw_graph(x: f32, y: f32, w: f32, h: f32, label: &str, unit: &str, s: &Series, color: Color) {
    draw_rectangle(x, y, w, h, Color::new(0.02, 0.02, 0.045, 1.0));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    let max = s.data.iter().cloned().fold(0.0_f32, f32::max).max(1e-6) * 1.1;
    if s.data.len() >= 2 {
        let plot_h = h - 26.0;
        let step = w / (s.cap.max(2) - 1) as f32;
        for i in 1..s.data.len() {
            let x1 = x + (i - 1) as f32 * step;
            let x2 = x + i as f32 * step;
            let y1 = y + h - 4.0 - (s.data[i - 1] / max) * plot_h;
            let y2 = y + h - 4.0 - (s.data[i] / max) * plot_h;
            draw_line(x1, y1, x2, y2, 1.5, color);
        }
    }
    draw_text(
        &format!("{label}: {:.2}{unit}  (max {:.2})", s.last(), max / 1.1),
        x + 6.0,
        y + 16.0,
        16.0,
        WHITE,
    );
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(42);

    // Loading frame while the bank generates.
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

    let world = VRect::new(0.0, 0.0, MAP_W, MAP_H);
    let mut tree: Tree<Critter> = Tree::with_limits(world, ITEM_LIMIT, ITEM_LIMIT);
    let mut next_id = 0u32;
    let mut cooldowns: HashMap<u32, f64> = HashMap::new();

    // Live-tunable settings (sliders in the side panel).
    let mut split_f: f32 = ITEM_LIMIT as f32;
    let mut merge_f: f32 = ITEM_LIMIT as f32;
    let mut drifters_f: f32 = 16.0;
    let mut hunters_f: f32 = 12.0;
    let mut pulsars_f: f32 = 14.0;
    let mut respawn_f: f32 = RESPAWN_DELAY as f32;
    let mut speed_f: f32 = 1.0;
    let mut fire_f: f32 = 1.0;

    for _ in 0..16 {
        spawn(&mut tree, &mut next_id, Kind::Drifter);
    }
    for _ in 0..12 {
        spawn(&mut tree, &mut next_id, Kind::Hunter);
    }
    for _ in 0..14 {
        spawn(&mut tree, &mut next_id, Kind::Pulsar);
    }

    let mut respawns: Vec<(f64, Kind)> = Vec::new();
    let mut effects: Vec<Effect> = Vec::new();
    let mut rings: Vec<Ring> = Vec::new();
    let mut kills: u64 = 0;
    let mut kills_by: HashMap<Kind, u64> = HashMap::new();
    let mut last_paint: f64 = 0.0;
    let mut paused = false;
    let mut brush = Kind::Hunter;
    let mut region_mode: u8 = 0; // 0 = fill + lines, 1 = lines, 2 = off

    let max_frames: u64 = std::env::var("CRITTERS_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut frame: u64 = 0;

    // Live performance series (rolling windows for the graphs column).
    let mut g_frame = Series::new(240);
    let mut g_atk_cull = Series::new(240);
    let mut g_vision_cull = Series::new(240);
    let mut g_move = Series::new(240);
    let mut sightlines: Vec<(VPoint, VPoint)> = Vec::new();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
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
        if is_key_pressed(KeyCode::LeftBracket) {
            speed_f = (speed_f * 0.5).max(0.25);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            speed_f = (speed_f * 2.0).min(4.0);
        }

        let now = get_time();
        let dt = (get_frame_time() as f64).min(0.05) * speed_f as f64;

        // --- interactive add / remove (mouse works in screen px; world = /S) ---
        let (mx, my) = mouse_position();
        let (wx, wy) = ((mx / S) as f64, (my / S) as f64);
        let mouse_in_map = wx >= 0.0 && wx < MAP_W && wy >= 0.0 && wy < MAP_H && mx < MAP_W as f32 * S;
        if is_mouse_button_down(MouseButton::Left)
            && mouse_in_map
            && tree.item_count() < MAX_CRITTERS
            && now - last_paint > 0.12
        {
            last_paint = now;
            let pos = spawn_at(&mut tree, &mut next_id, brush, VPoint::new(wx, wy));
            rings.push(Ring { pos, color: brush.color(), start: now, until: now + 0.4 });
            match brush {
                Kind::Drifter => drifters_f += 1.0,
                Kind::Hunter => hunters_f += 1.0,
                Kind::Pulsar => pulsars_f += 1.0,
            }
        }
        if is_mouse_button_pressed(MouseButton::Right) && mouse_in_map {
            // Remove the critter closest to the cursor (within 30 world px).
            let mut best: Option<(f64, u32, VPoint, Kind)> = None;
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    let d = (c.pos.x - wx).powi(2) + (c.pos.y - wy).powi(2);
                    if d < 30.0 * 30.0 && best.is_none_or(|(bd, ..)| d < bd) {
                        best = Some((d, c.id, c.pos, c.kind));
                    }
                }
            });
            if let Some((_, id, pos, kind)) = best {
                tree.remove(pos, |c| c.id == id);
                cooldowns.remove(&id);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }
        if is_key_pressed(KeyCode::Equal) && tree.item_count() < MAX_CRITTERS {
            for _ in 0..5 {
                let pos = spawn(&mut tree, &mut next_id, brush);
                rings.push(Ring { pos, color: brush.color(), start: now, until: now + 0.4 });
                match brush {
                    Kind::Drifter => drifters_f += 1.0,
                    Kind::Hunter => hunters_f += 1.0,
                    Kind::Pulsar => pulsars_f += 1.0,
                }
            }
        }
        if is_key_pressed(KeyCode::Minus) {
            let mut all: Vec<(u32, VPoint, Kind)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    all.push((c.id, c.pos, c.kind));
                }
            });
            for _ in 0..5 {
                if all.is_empty() {
                    break;
                }
                let i = rand::gen_range(0, all.len());
                let (id, pos, kind) = all.swap_remove(i);
                tree.remove(pos, |c| c.id == id);
                cooldowns.remove(&id);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }

        if !paused {
            // Snapshot all live critters (id, kind, pos, heading).
            let mut snap: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    snap.push((c.id, c.kind, c.pos, c.heading));
                }
            });

            // Movement: each update relocates the item across leaves when it
            // walks out of its cell (split/merge happens live here). Hunters
            // acquire prey with a vision-circle cull against the tree.
            let move_start = std::time::Instant::now();
            let mut vision_us_total = 0.0;
            let mut vision_culls = 0u32;
            sightlines.clear();
            for &(id, kind, pos, heading) in &snap {
                let target = if kind == Kind::Hunter {
                    let (prey, us) = nearest_prey(&tree, pos, id);
                    vision_us_total += us;
                    vision_culls += 1;
                    if let Some(t) = prey {
                        sightlines.push((pos, t));
                    }
                    prey
                } else {
                    None
                };
                let (np, nh) = step_critter(kind, pos, heading, dt, target);
                tree.update(pos, |c| c.id == id, |c| {
                    c.pos = np;
                    c.heading = nh;
                });
            }
            g_move.push(move_start.elapsed().as_secs_f64() as f32 * 1000.0);
            g_vision_cull.push(if vision_culls > 0 {
                (vision_us_total / vision_culls as f64) as f32
            } else {
                0.0
            });

            // Post-move snapshot for firing.
            let mut snap2: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    snap2.push((c.id, c.kind, c.pos, c.heading));
                }
            });

            let mut killed: Vec<(u32, VPoint, Kind)> = Vec::new();
            let mut killed_ids: HashSet<u32> = HashSet::new();
            let mut atk_us_total = 0.0;
            let mut atk_culls = 0u32;

            for &(id, kind, pos, _) in &snap2 {
                if killed_ids.contains(&id) {
                    continue; // died earlier this frame
                }
                let (cd_min, cd_max) = kind.cooldown();
                let cd = cooldowns
                    .entry(id)
                    .or_insert_with(|| rand::gen_range(cd_min, cd_max) as f64);
                *cd -= dt;
                if *cd > 0.0 {
                    continue;
                }
                *cd = rand::gen_range(cd_min, cd_max) as f64 / fire_f as f64;

                let attack = match kind {
                    // Aimed: drop pointed at the prey found by the vision cull.
                    Kind::Hunter => nearest_prey(&tree, pos, id).0.and_then(|tpos| {
                        make_attack(
                            &arsenal,
                            DROP_ID,
                            pos,
                            Some((tpos.x - pos.x, tpos.y - pos.y)),
                        )
                    }),
                    // Random direction drop.
                    Kind::Drifter => {
                        let a = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
                        make_attack(&arsenal, DROP_ID, pos, Some((a.cos(), a.sin())))
                    }
                    // Circular blast centred on itself.
                    Kind::Pulsar => make_attack(&arsenal, CIRCLE_ID, pos, None),
                };

                if let Some(atk) = attack {
                    let t = std::time::Instant::now();
                    let victims = tree.cull(&atk);
                    atk_us_total += t.elapsed().as_secs_f64() * 1e6;
                    atk_culls += 1;
                    for victim in victims {
                        if victim.id != id && !killed_ids.contains(&victim.id) {
                            killed_ids.insert(victim.id);
                            killed.push((victim.id, victim.pos, kind));
                        }
                    }
                    effects.push(Effect {
                        grid: arsenal.bank.template_for(
                            &atk.figure,
                            16,
                            16,
                            atk.angle_deg,
                            atk.origin,
                        ),
                        outline: sample_polygon(&atk.poly),
                        origin: pos,
                        color: kind.color(),
                        until: now + EFFECT_TTL,
                    });
                }
            }

            for (vid, vpos, attacker) in killed {
                if let Some(c) = tree.remove(vpos, |c| c.id == vid) {
                    cooldowns.remove(&vid);
                    respawns.push((now + respawn_f as f64, c.kind));
                    kills += 1;
                    *kills_by.entry(attacker).or_default() += 1;
                    rings.push(Ring {
                        pos: vpos,
                        color: c.kind.color(),
                        start: now,
                        until: now + 0.6,
                    });
                }
            }

            let mut i = 0;
            while i < respawns.len() {
                if respawns[i].0 <= now {
                    let (_, kind) = respawns.swap_remove(i);
                    let pos = spawn(&mut tree, &mut next_id, kind);
                    rings.push(Ring { pos, color: WHITE, start: now, until: now + 0.5 });
                } else {
                    i += 1;
                }
            }

            g_atk_cull.push(if atk_culls > 0 {
                (atk_us_total / atk_culls as f64) as f32
            } else {
                0.0
            });

            effects.retain(|e| e.until > now);
        }
        rings.retain(|r| r.until > now);
        g_frame.push(get_frame_time() * 1000.0);

        // ---------- live tuning panel ----------
        widgets::Window::new(
            hash!(),
            vec2(MAP_W as f32 * S + 10.0, 446.0),
            vec2(PANEL_W - 20.0, 280.0),
        )
        .label("tuning (live)")
        .titlebar(true)
        .movable(false)
        .ui(&mut *root_ui(), |ui| {
            ui.slider(hash!(), "split >", 1f32..12f32, &mut split_f);
            ui.slider(hash!(), "merge <=", 1f32..12f32, &mut merge_f);
            ui.separator();
            ui.slider(hash!(), "drifters", 0f32..400f32, &mut drifters_f);
            ui.slider(hash!(), "hunters", 0f32..400f32, &mut hunters_f);
            ui.slider(hash!(), "pulsars", 0f32..400f32, &mut pulsars_f);
            ui.separator();
            ui.slider(hash!(), "respawn s", 0.5f32..10f32, &mut respawn_f);
            ui.slider(hash!(), "speed x", 0.25f32..4f32, &mut speed_f);
            ui.slider(hash!(), "fire x", 0.25f32..3f32, &mut fire_f);
        });

        // Integer-valued sliders: snap to whole numbers right after the UI
        // so neither the displayed value nor the applied one carries decimals.
        split_f = split_f.round().clamp(1.0, 12.0);
        merge_f = merge_f.round().clamp(1.0, split_f);
        drifters_f = drifters_f.round();
        hunters_f = hunters_f.round();
        pulsars_f = pulsars_f.round();
        let want_split = split_f.round() as usize;
        let want_merge = (merge_f.round() as usize).min(want_split);
        if want_split != tree.item_limit || want_merge != tree.merge_limit {
            let mut items: Vec<Critter> = Vec::new();
            tree.visit_leaves(|_, leaf| items.extend(leaf.items.iter().cloned()));
            let mut rebuilt = Tree::with_limits(world, want_split, want_merge);
            for c in items {
                rebuilt.insert(c);
            }
            tree = rebuilt;
        }

        // Steer populations toward the slider targets.
        let mut alive: HashMap<Kind, Vec<(u32, VPoint)>> = HashMap::new();
        tree.visit_leaves(|_, leaf| {
            for c in &leaf.items {
                alive.entry(c.kind).or_default().push((c.id, c.pos));
            }
        });
        for (kind, target_f) in [
            (Kind::Drifter, drifters_f),
            (Kind::Hunter, hunters_f),
            (Kind::Pulsar, pulsars_f),
        ] {
            let target = target_f.round() as i64;
            let queued = respawns.iter().filter(|(_, k)| *k == kind).count() as i64;
            let live = alive.get(&kind).map_or(0, |v| v.len()) as i64;
            let mut diff = target - live - queued;
            while diff > 0 {
                let pos = spawn(&mut tree, &mut next_id, kind);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                diff -= 1;
            }
            if diff < 0 {
                let mut to_remove = -diff;
                let mut i = 0;
                while i < respawns.len() && to_remove > 0 {
                    if respawns[i].1 == kind {
                        respawns.swap_remove(i);
                        to_remove -= 1;
                    } else {
                        i += 1;
                    }
                }
                if let Some(list) = alive.get_mut(&kind) {
                    while to_remove > 0 && !list.is_empty() {
                        let idx = rand::gen_range(0, list.len());
                        let (id, pos) = list.swap_remove(idx);
                        if tree.remove(pos, |c| c.id == id).is_some() {
                            cooldowns.remove(&id);
                            rings.push(Ring {
                                pos,
                                color: kind.color(),
                                start: now,
                                until: now + 0.4,
                            });
                        }
                        to_remove -= 1;
                    }
                }
            }
        }

        // ---------- draw (world coords × S) ----------
        clear_background(Color::new(0.07, 0.07, 0.10, 1.0));

        // Tree regions: one colour per live leaf.
        let mut leaves = 0usize;
        let mut items = 0usize;
        tree.visit_leaves(|id, leaf| {
            leaves += 1;
            items += leaf.items.len();
            let b = leaf.bbox;
            let c = region_color(id.0);
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

        // Attack effects: template cells at a representative size, the real
        // polygon outline on top, and the origin dot.
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

        // Critters (single snapshot reused for sightlines + counts).
        let mut draw_snap: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
        tree.visit_leaves(|_, leaf| {
            for c in &leaf.items {
                draw_snap.push((c.id, c.kind, c.pos, c.heading));
            }
        });

        // Hunter sightlines to the prey found by their vision culls.
        for &(from, to) in &sightlines {
            draw_line(
                from.x as f32 * S,
                from.y as f32 * S,
                to.x as f32 * S,
                to.y as f32 * S,
                1.0,
                Color::new(1.0, 0.3, 0.3, 0.18),
            );
        }

        for &(_, kind, pos, heading) in &draw_snap {
            let (x, y) = (pos.x as f32 * S, pos.y as f32 * S);
            draw_circle(x, y, 5.0, kind.color());
            draw_line(
                x,
                y,
                x + (heading.cos() * 9.0) as f32 * S,
                y + (heading.sin() * 9.0) as f32 * S,
                1.5,
                WHITE,
            );
        }

        // Spawn / death rings.
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

        // Map border + side panel.
        let map_px = MAP_W as f32 * S;
        let map_py = MAP_H as f32 * S;
        draw_rectangle_lines(0.0, 0.0, map_px, map_py, 2.0, GRAY);
        let px = map_px + 14.0;
        draw_rectangle(map_px, 0.0, PANEL_W, map_py, Color::new(0.04, 0.04, 0.06, 1.0));
        let mut ty = 28.0;
        let line = |text: &str, size: f32, color: Color, ty: &mut f32| {
            draw_text(text, px, *ty, size, color);
            *ty += size * 1.15;
        };
        line("vectorial-hash critters", 26.0, WHITE, &mut ty);
        ty += 8.0;
        line(
            &format!("bank: {} combos", arsenal.bank.entry_count()),
            18.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!(
                "  -> {} unique grids ({:.1}s gen)",
                arsenal.bank.unique_count(),
                arsenal.gen_seconds,
            ),
            18.0,
            LIGHTGRAY,
            &mut ty,
        );
        line("  sizes 8..32 + 1x1 raster", 18.0, LIGHTGRAY, &mut ty);
        ty += 10.0;
        let mut counts: HashMap<Kind, usize> = HashMap::new();
        for &(_, kind, _, _) in &draw_snap {
            *counts.entry(kind).or_default() += 1;
        }
        for kind in [Kind::Drifter, Kind::Hunter, Kind::Pulsar] {
            draw_circle(px + 6.0, ty - 5.0, 5.0, kind.color());
            let what = match kind {
                Kind::Drifter => "random walk, random drop shot",
                Kind::Hunter => "chases prey, aimed drop shot",
                Kind::Pulsar => "circles, radial blast",
            };
            line(
                &format!("    {} x{}: {}", kind.name(), counts.get(&kind).copied().unwrap_or(0), what),
                16.0,
                WHITE,
                &mut ty,
            );
        }
        ty += 10.0;
        line(&format!("alive: {}   kills: {}", items, kills), 18.0, WHITE, &mut ty);
        line(
            &format!(
                "  by: drifter {} / hunter {} / pulsar {}",
                kills_by.get(&Kind::Drifter).copied().unwrap_or(0),
                kills_by.get(&Kind::Hunter).copied().unwrap_or(0),
                kills_by.get(&Kind::Pulsar).copied().unwrap_or(0),
            ),
            16.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!("respawning: {}", respawns.len()),
            18.0,
            WHITE,
            &mut ty,
        );
        ty += 10.0;
        line(
            &format!(
                "tree leaves: {}   (split >{}, merge <={})",
                leaves, tree.item_limit, tree.merge_limit,
            ),
            18.0,
            SKYBLUE,
            &mut ty,
        );
        line(
            &format!("arena nodes: {} (incl. orphans)", tree.node_count()),
            18.0,
            SKYBLUE,
            &mut ty,
        );
        ty += 10.0;
        draw_circle(px + 6.0, ty - 5.0, 5.0, brush.color());
        line(
            &format!("    brush: {}  (keys 1/2/3)", brush.name()),
            18.0,
            WHITE,
            &mut ty,
        );
        line(
            &format!("speed: {:.2}x   regions: {}", speed_f, match region_mode {
                0 => "fill+lines",
                1 => "lines",
                _ => "off",
            }),
            18.0,
            WHITE,
            &mut ty,
        );
        ty += 10.0;
        line(&format!("fps: {}", get_fps()), 18.0, GREEN, &mut ty);
        if paused {
            line("PAUSED", 22.0, ORANGE, &mut ty);
        }
        // ---------- graphs column ----------
        let gx = map_px + PANEL_W;
        draw_rectangle(gx, 0.0, GRAPH_W, map_py, Color::new(0.05, 0.05, 0.08, 1.0));
        let gpad = 10.0;
        let gw = GRAPH_W - 2.0 * gpad;
        let gh = 140.0;
        let mut gy = 14.0;
        draw_text("performance (live)", gx + gpad, gy + 6.0, 20.0, WHITE);
        gy += 18.0;
        draw_graph(gx + gpad, gy, gw, gh, "frame", " ms", &g_frame, GREEN);
        gy += gh + 12.0;
        draw_graph(gx + gpad, gy, gw, gh, "attack cull", " us", &g_atk_cull, ORANGE);
        gy += gh + 12.0;
        draw_graph(gx + gpad, gy, gw, gh, "vision cull", " us", &g_vision_cull, SKYBLUE);
        gy += gh + 12.0;
        draw_graph(gx + gpad, gy, gw, gh, "move+update", " ms", &g_move, PINK);

        let help = [
            "[1/2/3] brush   [LMB] spawn (hold = paint)",
            "[RMB] remove   [+/-] spawn/remove 5",
            "[R] regions   [ / ] speed",
            "[Space] pause   [Esc] quit",
        ];
        for (i, h) in help.iter().enumerate() {
            draw_text(
                h,
                gx + gpad,
                map_py - 14.0 - 20.0 * (help.len() - 1 - i) as f32,
                16.0,
                GRAY,
            );
        }

        frame += 1;
        if max_frames > 0 && frame >= max_frames {
            break;
        }
        next_frame().await
    }
}
